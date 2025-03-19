from flask import Flask, request, jsonify, Response
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from typing import Dict, Any
import uuid
import time
import json

app = Flask(__name__)

# 全局缓存存储
kv_cache_store: Dict[str, Dict[str, Any]] = {}

def create_cache(cache_name: str, cache_context: str) -> str:
    """创建新的缓存"""
    cache_id = str(uuid.uuid4())
    kv_cache_store[cache_id] = {
        'cache_name': cache_name,
        'cache_context': cache_context,
        'input_kv_cache': None,  # 输入缓存
        'output_kv_cache': None,  # 输出缓存
        'created_at': time.time(),
        'last_used': time.time()
    }
    return cache_id

def get_cache(cache_id: str) -> Dict[str, Any]:
    """获取缓存"""
    if cache_id in kv_cache_store:
        kv_cache_store[cache_id]['last_used'] = time.time()
        return kv_cache_store[cache_id]
    return None

def delete_cache(cache_id: str) -> bool:
    """删除缓存"""
    if cache_id in kv_cache_store:
        del kv_cache_store[cache_id]
        return True
    return False

# 加载模型和tokenizer
model_path = "huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 先加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto")

# 然后加载tokenizer，确保配置与模型兼容
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    pad_token=' ',
    padding_side='left',
    truncation_side='left',
    model_max_length=model.config.max_position_embeddings
)

# 设置模型的pad_token_id和eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

@app.route('/v1/completions', methods=['POST'])
def generate_completion():
    """生成文本补全，支持流式输出"""
    stream = request.json.get('stream', False)
    
    if stream:
        return stream_completion()
    else:
        return generate_completion_sync()

def stream_completion():
    """流式生成文本补全"""
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 1.0)
    input_cache_ids = data.get('input_cache_ids', [])
    output_cache_id = data.get('output_cache_id', None)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 获取所有输入缓存
    input_caches = []
    for cache_id in input_cache_ids:
        cache = get_cache(cache_id)
        if cache:
            input_caches.append(cache)
    
    # 合并所有输入缓存的input_kv_cache
    past_key_values = None
    if input_caches:
        past_key_values = input_caches[0]['input_kv_cache']
        for cache in input_caches[1:]:
            if cache['input_kv_cache']:
                past_key_values = [
                    (torch.cat([pkv1[0], pkv2[0]], dim=0),
                     torch.cat([pkv1[1], pkv2[1]], dim=0))
                    for pkv1, pkv2 in zip(past_key_values, cache['input_kv_cache'])
                ]
    
    def generate():
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_tokens,
                temperature=temperature,
                do_sample=True,
                past_key_values=past_key_values,
                streamer=TextStreamer(tokenizer),
                return_dict_in_generate=True,
                output_scores=True
            )
            for output in outputs.sequences:
                completion = tokenizer.decode(output, skip_special_tokens=True)
                yield f"data: {json.dumps({'text': completion})}\n\n"
        
            # 更新输出缓存，去除输入部分
            if output_cache_id:
                output_cache = get_cache(output_cache_id)
                if output_cache:
                    # 获取输入token数量
                    input_length = inputs.input_ids.size(1)
                    # 去除输入部分的kv cache
                    output_kv_cache = []
                    for layer in outputs.past_key_values:
                        # 每个layer包含key和value
                        k, v = layer
                        # 去除前input_length个token的kv cache
                        output_kv_cache.append((
                            k[:, :, input_length:, :],  # 去除输入key
                            v[:, :, input_length:, :]   # 去除输入value
                        ))
                    output_cache['output_kv_cache'] = tuple(output_kv_cache)
                    output_cache['cache_context'] = completion
    
                yield f"data: {json.dumps({'text': completion})}\n\n"
        
        # 更新输出缓存，去除输入部分
        if output_cache_id:
            output_cache = get_cache(output_cache_id)
            if output_cache:
                # 获取输入token数量
                input_length = inputs.input_ids.size(1)
                # 去除输入部分的kv cache
                output_kv_cache = []
                for layer in outputs.past_key_values:
                    # 每个layer包含key和value
                    k, v = layer
                    # 去除前input_length个token的kv cache
                    output_kv_cache.append((
                        k[:, :, input_length:, :],  # 去除输入key
                        v[:, :, input_length:, :]   # 去除输入value
                    ))
                output_cache['output_kv_cache'] = tuple(output_kv_cache)
                output_cache['cache_context'] = completion
    
    return Response(generate(), mimetype='text/event-stream')

def generate_completion_sync():
    """同步生成文本补全"""
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 1.0)
    input_cache_ids = data.get('input_cache_ids', [])
    output_cache_id = data.get('output_cache_id', None)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 获取所有输入缓存
    past_key_values = None
    if input_cache_ids:
        input_caches = []
        for cache_id in input_cache_ids:
            cache = get_cache(cache_id)
            if cache and cache.get('input_kv_cache'):
                input_caches.append(cache['input_kv_cache'])

        # 合并缓存（仅当所有缓存结构一致时）
        if input_caches:
            try:
                # 检查所有缓存的层数和形状是否一致
                num_layers = len(input_caches[0])
                for cache in input_caches[1:]:
                    if len(cache) != num_layers:
                        raise ValueError("缓存层数不一致")
                    for i in range(num_layers):
                        if cache[i][0].shape != input_caches[0][i][0].shape:
                            raise ValueError(f"第{i}层key形状不一致")
                        if cache[i][1].shape != input_caches[0][i][1].shape:
                            raise ValueError(f"第{i}层value形状不一致")

                # 按层合并key和value
                merged_cache = []
                for layer_idx in range(num_layers):
                    keys = [cache[layer_idx][0] for cache in input_caches]
                    values = [cache[layer_idx][1] for cache in input_caches]
                    merged_key = torch.cat(keys, dim=2)  # 在序列维度合并
                    merged_value = torch.cat(values, dim=2)
                    merged_cache.append((merged_key, merged_value))
                past_key_values = tuple(merged_cache)
            except Exception as e:
                return jsonify({
                    'error': '缓存合并失败',
                    'message': str(e),
                    'status_code': 400
                }), 400

    with torch.no_grad():
        # 输入校验
        if not hasattr(inputs, 'input_ids') or inputs.input_ids.nelement() == 0:
            return jsonify({'error': '输入文本为空', 'status_code': 400}), 400

        if inputs.input_ids.size(1) > model.config.max_position_embeddings:
            return jsonify({
                'error': '输入长度超过模型限制',
                'max_length': model.config.max_position_embeddings,
                'status_code': 400
            }), 400

        try:
            # 生成配置
            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "past_key_values": past_key_values,
                "attention_mask": inputs.attention_mask,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 50
            }

            # 执行生成
            outputs = model.generate(
                inputs.input_ids,
                **generation_config
            )
        except Exception as e:
            # 记录详细错误日志
            import traceback
            error_details = {
                'input_text': prompt,
                'input_cache_ids': input_cache_ids,
                'input_length': inputs.input_ids.size(1) if hasattr(inputs, 'input_ids') else 0,
                'past_key_values_shape': str([(k.shape if k else None, v.shape if v else None) for (k, v) in past_key_values]) if past_key_values else None,
                'model_config': {
                    'max_position_embeddings': model.config.max_position_embeddings,
                    'num_hidden_layers': model.config.num_hidden_layers
                },
                'stack_trace': traceback.format_exc()
            }
            app.logger.error("Generation failed with details: %s", error_details)
            
            return jsonify({
                'error': '生成失败',
                'message': str(e),
                'details': error_details,
                'status_code': 500
            }), 500
        else:
            # 初始化attention_mask
            if inputs.attention_mask is None:
                inputs.attention_mask = torch.ones_like(inputs.input_ids)
                
            try:
                # 检查输入序列长度
                seq_length = inputs.input_ids.size(1)
                if seq_length == 0:
                    raise ValueError("Input sequence length cannot be zero")
                
                # 初始化cache_position
                cache_position = torch.arange(
                    seq_length,
                    device=inputs.input_ids.device
                )
                
                # 如果past_key_values为None，设置为None
                if past_key_values is None:
                    past_key_values = None
                else:
                    # 确保past_key_values格式正确
                    if not isinstance(past_key_values, tuple):
                        past_key_values = tuple(past_key_values)
                    
                    # 检查past_key_values的维度
                    for layer in past_key_values:
                        if len(layer) != 2:
                            raise ValueError("Invalid past_key_values format")
                
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    past_key_values=past_key_values,
                    attention_mask=inputs.attention_mask,
                    cache_position=cache_position,
                    use_cache=True
                )
            except Exception as e:
                return jsonify({
                    'error': 'Generation failed',
                    'message': str(e),
                    'details': {
                        'input_length': inputs.input_ids.size(1) if hasattr(inputs, 'input_ids') else 0,
                        'past_key_values_shape': str([(k.shape, v.shape) for (k, v) in past_key_values]) if past_key_values and isinstance(past_key_values, tuple) else None
                    },
                    'status_code': 500
                }), 500
    
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 更新输出缓存，去除输入部分
    if output_cache_id:
        output_cache = get_cache(output_cache_id)
        if output_cache:
            # 获取输入token数量
            input_length = inputs.input_ids.size(1)
            # 去除输入部分的kv cache
            output_kv_cache = []
            for layer in outputs.past_key_values:
                # 每个layer包含key和value
                k, v = layer
                # 去除前input_length个token的kv cache
                output_kv_cache.append((
                    k[:, :, input_length:, :],  # 去除输入key
                    v[:, :, input_length:, :]   # 去除输入value
                ))
            output_cache['output_kv_cache'] = tuple(output_kv_cache)
            output_cache['cache_context'] = completion
    
    return jsonify({
        'choices': [{
            'text': completion,
            'index': 0,
            'finish_reason': 'length',
            'input_cache_ids': input_cache_ids,
            'output_cache_id': output_cache_id
        }]
    })

@app.route('/v1/cache/create', methods=['POST'])
def create_cache_endpoint():
    data = request.json
    cache_name = data.get('cache_name')
    cache_context = data.get('cache_context')
    
    if not cache_name or not cache_context:
        return jsonify({'error': 'cache_name and cache_context are required'}), 400
    
    cache_id = create_cache(cache_name, cache_context)
    return jsonify({
        'cache_id': cache_id,
        'cache_name': cache_name,
        'cache_context': cache_context
    })

@app.route('/v1/cache/get', methods=['GET'])
def get_cache_endpoint():
    cache_id = request.args.get('cache_id')
    if not cache_id:
        return jsonify({'error': 'cache_id is required'}), 400
    
    cache = get_cache(cache_id)
    if not cache:
        return jsonify({'error': 'cache not found'}), 404
    
    return jsonify({
        'cache_id': cache_id,
        'cache_name': cache['cache_name'],
        'cache_context': cache['cache_context'],
        'created_at': cache['created_at'],
        'last_used': cache['last_used']
    })

@app.route('/v1/cache/init', methods=['POST'])
def init_cache_endpoint():
    """初始化缓存，创建初始kv cache"""
    data = request.json
    cache_name = data.get('cache_name')
    cache_context = data.get('cache_context')
    text = data.get('text')
    
    if not cache_name or not cache_context or not text:
        return jsonify({'error': 'cache_name, cache_context and text are required'}), 400
    
    # 创建初始缓存
    cache_id = create_cache(cache_name, cache_context)
    
    # 生成初始kv cache
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, use_cache=True)
        cache = get_cache(cache_id)
        cache['input_kv_cache'] = outputs.past_key_values
    
    return jsonify({
        'cache_id': cache_id,
        'cache_name': cache_name,
        'cache_context': cache_context
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)