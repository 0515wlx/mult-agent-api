import unittest
import requests
import json
import time
from customer_service_agent import CustomerServiceAgent

class TestAPIServer(unittest.TestCase):
    BASE_URL = "http://localhost:8080/v1"
    
    def test_cache_operations(self):
        # 测试缓存创建
        cache_data = {
            "cache_name": "test_cache",
            "cache_context": "test context"
        }
        response = requests.post(f"{self.BASE_URL}/cache/create", json=cache_data)
        self.assertEqual(response.status_code, 200)
        cache_id = response.json()["cache_id"]
        
        # 测试获取缓存
        response = requests.get(f"{self.BASE_URL}/cache/get?cache_id={cache_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cache_name"], "test_cache")
        
        # 测试缓存初始化
        init_data = {
            "cache_name": "test_init_cache",
            "cache_context": "init context",
            "text": "initial text"
        }
        response = requests.post(f"{self.BASE_URL}/cache/init", json=init_data)
        self.assertEqual(response.status_code, 200)
        init_cache_id = response.json()["cache_id"]
        
    def test_completion(self):
        # 测试不使用缓存的简单补全
        prompt = "测试API"
        data = {
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7,
            "input_cache_ids": []  # 不使用缓存
        }
        response = requests.post(f"{self.BASE_URL}/completions", json=data)
        
        # 如果请求失败，打印调试信息
        if response.status_code != 200:
            print(f"API Response: {response.status_code}")
            print(f"Response Body: {response.text}")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("choices", response.json())
        
        # 测试流式补全
        data["stream"] = True
        response = requests.post(f"{self.BASE_URL}/completions", json=data, stream=True)
        self.assertEqual(response.status_code, 200)
        for chunk in response.iter_content(chunk_size=None):
            self.assertIn(b"data:", chunk)
            
    def test_customer_service(self):
        # 测试客服问题处理
        agent = CustomerServiceAgent()
        
        # 测试技术支持问题
        tech_query = "我的设备无法开机"
        response = agent.handle_user_query(tech_query)
        self.assertTrue(any(keyword in response for keyword in ["技术", "设备", "开机"]))
        
        # 测试账单问题
        billing_query = "我的账单有误"
        response = agent.handle_user_query(billing_query)
        self.assertTrue(any(keyword in response for keyword in ["账单", "支付", "发票"]))
        
        # 测试产品问题
        product_query = "这个产品的规格是什么"
        response = agent.handle_user_query(product_query)
        self.assertTrue(any(keyword in response for keyword in ["产品", "规格", "功能"]))
        
        # 测试一般问题
        general_query = "你们的营业时间是什么"
        response = agent.handle_user_query(general_query)
        self.assertFalse(any(keyword in response for keyword in ["技术", "账单", "产品"]))

if __name__ == "__main__":
    unittest.main()