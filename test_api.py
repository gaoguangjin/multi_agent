#!/usr/bin/env python3
"""
多智能体系统 - API 一键测试脚本
用法: python test_api.py
"""
import requests
import json
import os

BASE_URL = "http://localhost:8000"

def print_box(title):
    print(f"\n{'🔹'*25}")
    print(f"  {title}")
    print(f"{'🔹'*25}\n")

def test_status():
    """1. 检查知识库状态"""
    print_box("📊 测试1: 知识库状态")
    try:
        resp = requests.get(f"{BASE_URL}/knowledge_base/status", timeout=5)
        result = resp.json()
        print(f"✅ 返回: {json.dumps(result, ensure_ascii=False)}")
        return result.get("total_chunks", 0)
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return -1

def test_query(question):
    """2. 测试查询接口"""
    print_box(f"🔍 测试2: 查询 '{question}'")
    try:
        resp = requests.post(
            f"{BASE_URL}/query",
            headers={"Content-Type": "application/json"},
            json={"question": question},  # ✅ 用 json= 参数，自动处理编码
            timeout=10
        )
        result = resp.json()
        print(f"✅ 返回:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_task(task_desc):
    """3. 测试任务执行接口"""
    print_box(f"🤖 测试3: 任务 '{task_desc}'")
    try:
        resp = requests.post(
            f"{BASE_URL}/run_task",
            headers={"Content-Type": "application/json"},
            json={"task": task_desc},
            timeout=30
        )
        result = resp.json()
        print(f"✅ 返回 (预览):")
        if isinstance(result, dict):
            # 只打印关键部分，避免刷屏
            preview = {
                "task_plan": result.get("task_plan"),
                "final_result_preview": str(result.get("final_result", ""))[:300] + "..."
            }
            print(json.dumps(preview, ensure_ascii=False, indent=2))
        else:
            print(str(result)[:500])
        return result
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_upload(file_path):
    """4. 测试文件上传"""
    print_box(f"📁 测试4: 上传文件 '{file_path}'")
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            resp = requests.post(f"{BASE_URL}/upload", files=files, timeout=30)
        result = resp.json()
        print(f"✅ 返回: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return result
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

if __name__ == "__main__":
    print("🚀 多智能体系统 API 测试启动...\n")
    
    # 1. 检查服务是否在线
    print_box("🔌 检查服务连接")
    try:
        requests.get(f"{BASE_URL}/docs", timeout=3)
        print(f"✅ 服务在线: {BASE_URL}")
    except:
        print(f"❌ 无法连接服务，请先运行: uvicorn main:app --reload")
        exit(1)
    
    # 2. 检查知识库状态
    chunk_count = test_status()
    
    # 3. 如果知识库为空，提示上传
    if chunk_count == 0:
        print("\n⚠️  知识库为空！你可以:")
        print("   ① 上传测试文件: python test_api.py upload test.txt")
        print("   ② 或先创建一个 test.txt 文件")
        # 创建示例测试文件
        with open("test_qingdao.txt", "w", encoding="utf-8") as f:
            f.write("青岛是山东省下辖的副省级城市，位于山东半岛东南部，濒临黄海。")
        print(f"   ✅ 已创建测试文件: test_qingdao.txt")
        test_upload("test_qingdao.txt")
        # 重新检查状态
        chunk_count = test_status()
    
    # 4. 执行查询测试
    if chunk_count > 0:
        print("\n🔄 开始查询测试...\n")
        test_query("青岛")
        test_query("青岛是哪个省的")
        test_query("用一句话说明青岛的位置")
    
    # 5. 可选：测试任务执行（取消注释即可运行）
    # test_task("青岛是哪个省的，超简约一句话回答")
    
    print_box("✅ 所有测试完成！")

    def test_run_task(task_desc):
        print(f"\n🤖 测试任务: '{task_desc}'")
        try:
            resp = requests.post(
            "http://localhost:8000/run_task",
            headers={"Content-Type": "application/json"},
            json={"task": task_desc},
            timeout=40
        )
            res = resp.json()
        # 只打印关键部分
            print(f"✅ 最终结果 ({len(res.get('final_result',''))}字):")
            print(res.get("final_result", "无返回"))
        except Exception as e:
            print(f"❌ 失败: {e}")

if __name__ == "__main__":
    # ... 保留原有测试 ...
    
    print("\n" + "="*40)
    print("🧪 开始测试 /run_task 约束遵循能力")
    print("="*40)
    test_run_task("青岛是哪个省的，超简约一句话回答")
    test_run_task("根据文档，详细总结故事背景（300 字）")
    test_run_task("把‘Hello World’翻译成日文，正式商务风格")