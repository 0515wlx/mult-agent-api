import requests
import json
import time
from typing import Dict, List

BASE_URL = "http://localhost:8080/v1"

class CustomerServiceAgent:
    def __init__(self):
        self.cache_map: Dict[str, str] = {}  # 缓存映射：角色 -> cache_id
        
        # 初始化各角色的缓存
        # 使用简化的初始化文本避免特殊字符问题
        self._init_cache("客服主管", """
        客服主管职责：
        1. 问题分类与路由
           - 接收并分析用户问题
           - 根据问题类型分配至相应部门
           - 监控问题处理进度
        2. 服务质量监控
           - 评估客服响应时间
           - 检查问题解决质量
           - 收集用户反馈
        3. 团队协调
           - 组织跨部门协作
           - 处理复杂问题
           - 提供决策支持
        4. 数据分析
           - 统计常见问题类型
           - 分析服务效率指标
           - 生成服务报告
        5. 流程优化
           - 识别服务瓶颈
           - 提出改进建议
           - 更新服务流程
        """)
        
        self._init_cache("技术支持", """
        技术支持职责：
        1. 技术问题诊断
           - 分析系统错误日志
           - 识别硬件/软件故障
           - 确定问题根源
        2. 故障排除
           - 提供解决方案
           - 远程协助
           - 现场支持
        3. 系统维护
           - 定期检查系统状态
           - 执行系统更新
           - 优化系统性能
        4. 用户培训
           - 提供产品使用指导
           - 编写技术文档
           - 举办培训课程
        5. 技术支持流程
           - 问题记录与跟踪
           - 解决方案文档化
           - 知识库维护
        """)
        
        self._init_cache("账单客服", """
        账单客服职责：
        1. 账单查询
           - 提供账单明细
           - 解释收费项目
           - 核对账单信息
        2. 支付处理
           - 接收支付
           - 处理退款
           - 管理支付方式
        3. 争议解决
           - 调查账单争议
           - 提供解决方案
           - 处理投诉
        4. 账户管理
           - 更新账户信息
           - 管理订阅服务
           - 处理账户异常
        5. 财务合规
           - 确保财务准确性
           - 遵守财务法规
           - 维护财务记录
        """)
        
        self._init_cache("产品顾问", """
        产品顾问职责：
        1. 产品介绍
           - 讲解产品功能
           - 演示产品使用
           - 比较产品差异
        2. 需求分析
           - 了解客户需求
           - 推荐合适产品
           - 提供定制方案
        3. 技术支持
           - 解答技术问题
           - 提供使用指导
           - 解决使用问题
        4. 售后服务
           - 处理产品问题
           - 提供维护建议
           - 跟进客户反馈
        5. 产品知识
           - 掌握产品规格
           - 了解技术参数
           - 跟踪产品更新
        """)

    def _init_cache(self, role: str, context: str, max_retries: int = 3) -> str:
        """初始化角色缓存，带重试机制"""
        url = f"{BASE_URL}/cache/init"
        payload = {
            "cache_name": role,
            "cache_context": context,
            "text": context
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    cache_id = response.json()["cache_id"]
                    self.cache_map[role] = cache_id
                    return cache_id
                else:
                    print(f"Attempt {attempt + 1} failed with status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                
        raise Exception(f"Failed to init cache for {role} after {max_retries} attempts")

    def handle_user_query(self, user_query: str) -> str:
        """处理用户查询"""
        # 1. 客服主管分析问题
        supervisor_cache = self.cache_map["客服主管"]
        analysis = self._generate_completion(
            prompt=f"用户问题：{user_query}\n请分析问题类型并指定处理部门：",
            input_cache_ids=[supervisor_cache]
        )
        
        # 2. 根据分析结果路由到对应部门
        # 使用更精确的路由判断
        if any(keyword in analysis for keyword in ["技术", "安装", "错误", "兼容性"]):
            return self._handle_tech_support(user_query)
        elif any(keyword in analysis for keyword in ["账单", "支付", "发票", "退款"]):
            return self._handle_billing(user_query)
        elif any(keyword in analysis for keyword in ["产品", "型号", "规格", "功能"]):
            return self._handle_product(user_query)
        else:
            return self._handle_general_query(user_query)

    def _handle_tech_support(self, query: str) -> str:
        """处理技术支持问题"""
        tech_cache = self.cache_map["技术支持"]
        return self._generate_completion(
            prompt=f"技术支持问题：{query}",
            input_cache_ids=[tech_cache]
        )

    def _handle_billing(self, query: str) -> str:
        """处理账单问题"""
        billing_cache = self.cache_map["账单客服"]
        return self._generate_completion(
            prompt=f"账单问题：{query}",
            input_cache_ids=[billing_cache]
        )

    def _handle_product(self, query: str) -> str:
        """处理产品问题"""
        product_cache = self.cache_map["产品顾问"]
        return self._generate_completion(
            prompt=f"产品问题：{query}",
            input_cache_ids=[product_cache]
        )

    def _handle_general_query(self, query: str) -> str:
        """处理一般性问题"""
        supervisor_cache = self.cache_map["客服主管"]
        return self._generate_completion(
            prompt=f"用户问题：{query}",
            input_cache_ids=[supervisor_cache]
        )

    def _generate_completion(self, prompt: str, input_cache_ids: List[str]) -> str:
        """调用API生成补全"""
        url = f"{BASE_URL}/completions"
        payload = {
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.7,
            "input_cache_ids": input_cache_ids
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["text"]
        else:
            error_msg = f"Failed to generate completion. Status code: {response.status_code}"
            if response.text:
                try:
                    error_detail = response.json().get("error", response.text)
                    error_msg += f", Details: {error_detail}"
                except:
                    error_msg += f", Response: {response.text}"
            raise Exception(error_msg)

# 客服团队协作流程
CUSTOMER_SERVICE_SOP = """
客服团队协作流程：
1. 客服主管首先接收并分析用户问题
2. 根据问题类型，相关部门智能体可以直接参与解答：
   - 技术支持：处理产品使用和技术问题
   - 账单客服：处理账单和支付相关问题
   - 产品顾问：解答产品功能和选型问题
3. 各部门智能体可以：
   - 直接与用户交流
   - 相互协作解决跨部门问题
   - 随时向客服主管请求支持
4. 客服主管负责：
   - 确保问题得到及时处理
   - 协调多部门协作
   - 总结对话内容
   - 评估服务满意度
5. 所有智能体都应：
   - 保持专业和友好的服务态度
   - 确保信息的准确性
   - 注意保护用户隐私
"""

if __name__ == "__main__":
    agent = CustomerServiceAgent()
    
    while True:
        query = "产品出现问题，如何退款？"
        if query.lower() in ["exit", "quit"]:
            break
            
        response = agent.handle_user_query(query)
        print(f"客服：{response}")