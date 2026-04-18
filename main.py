from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from agents import ManagerAgent
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="多智能体协同办公系统", version="1.0")
manager_agent = ManagerAgent()

# 创建上传目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# 挂载静态目录（放在所有 @app.route 之前）
app.mount("/static", StaticFiles(directory="static"), name="static")

class TaskRequest(BaseModel):
    task: str


class QueryRequest(BaseModel):
    question: str


# 原有的Agent任务接口
@app.post("/run_task")
async def run_agent_task(request: TaskRequest):
    result = manager_agent.run(request.task)
    return result


# ============== 新增：知识库相关接口 ==============

# 上传文档到知识库
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    from tools import process_and_store

    # 检查文件格式
    allowed_ext = [".txt", ".pdf", ".docx", ".doc"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return {"error": f"不支持的文件格式：{ext}，支持：{allowed_ext}"}

    # 保存文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 处理并存入向量库
    result = process_and_store(file_path)
    return {"filename": file.filename, "result": result}


# 直接查询知识库（不走Agent）
# ============== 修复版：直接查询知识库 ==============
# ============== 修复版：检索 + LLM 智能回答 ==============
# ============== 终极版：智能总结 + 严格字数控制 ==============
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    from tools import search_knowledge_base, call_llm
    import re
    
    print(f"[🔗/query] 收到查询: {request.question}")
    
    # 1. 检索相关片段
    chunks = search_knowledge_base(request.question, top_k=3)
    
    if not chunks:
        return {
            "question": request.question,
            "answer": "知识库中没有找到相关内容",
            "relevant_chunks": [],
            "chunk_count": 0
        }
    
    # 2. 解析用户意图
    question = request.question
    need_summary = any(kw in question for kw in ["总结", "概括", "讲的啥", "主要内容", "简述", "概述"])
    
    # 提取字数限制（支持"50字"、"100字以内"、"不超过80字"等）
    word_limit = None
    match = re.search(r'(?:不超过|控制在|以内|)?(\d+)\s*字', question)
    if match:
        word_limit = int(match.group(1))
    
    context = "\n\n===\n\n".join(chunks[:3])
    
    # 3. 构建 prompt（分场景）
    if need_summary:
        # 🎯 两阶段总结策略
        if word_limit and word_limit <= 100:
            # 短总结：先提取要点，再压缩
            prompt = f"""你是一个专业的文档总结专家。

【任务】
用严格 {word_limit} 字（±5 字）总结【参考资料】的核心内容。

【输出步骤】（内部思考，不要输出）：
1. 从资料中提取：主角、核心事件、关键转折、最终结果
2. 用一句话串联这些要素
3. 精炼到 {word_limit} 字，删除冗余修饰词

【参考资料】
{context}

【输出要求】
- 只输出最终总结，不要步骤、不要解释
- 严格 {word_limit} 字（±5 字），中文标点算 1 字
- 语言流畅，像人写的，不要机械罗列

【最终总结】（直接开始写）："""
        else:
            # 长总结：正常生成
            length_hint = f"约{word_limit}字" if word_limit else "2-3 句话"
            prompt = f"""你是一个专业的文档总结助手。

【任务】
用{length_hint}概括【参考资料】的核心内容。

【参考资料】
{context}

【要求】
1. 包含：主角、主要事件、关键转折
2. 语言简洁流畅
3. 只基于资料，不编造

【总结】："""
    else:
        # 普通问答
        prompt = f"""你是一个严格的知识库问答助手。

【规则】
1. 仅使用【参考资料】回答，未提及则说"资料未提及"
2. 回答简洁准确，直接给答案

【参考资料】
{context}

【问题】{question}

【回答】："""
    
    # 4. 调用 LLM 生成
    try:
        answer = call_llm(prompt)
        answer = answer.strip()
        
        # 🔧 后处理：如果字数超标且用户有严格限制，二次压缩
        if word_limit and word_limit <= 100:
            actual_len = len(answer)  # 中文字符数
            if actual_len > word_limit + 10:  # 超标 10 字以上才压缩
                compress_prompt = f"""把下面文字精炼到严格{word_limit}字（±3 字），保留核心信息：

原文：{answer}

精炼后（只输出结果）："""
                answer = call_llm(compress_prompt).strip()
        
        print(f"[🔗/query] ✅ 答案生成，长度: {len(answer)}字")
        
    except Exception as e:
        print(f"[🔗/query] ❌ LLM 错误: {e}")
        answer = f"检索成功，但生成答案时出错：{str(e)}"
    
    # 5. 返回结果
    return {
        "question": question,
        "answer": answer,
        "relevant_chunks": chunks,
        "chunk_count": len(chunks),
        "debug": {
            "word_limit": word_limit,
            "answer_length": len(answer)
        }
    }

# 根路由返回前端页面
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# 查看知识库状态
@app.get("/knowledge_base/status")
async def knowledge_base_status():
    from tools import kb_collection
    count = kb_collection.count()
    return {"total_chunks": count}


# 启动命令：uvicorn main:app --host 0.0.0.0 --reload
# UV 启动命令：uv run uvicorn main:app --host 0.0.0.0 --reload

