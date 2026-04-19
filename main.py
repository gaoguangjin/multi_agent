from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from agents import ManagerAgent
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from tools import kb_collection

app = FastAPI(title="多智能体协同办公系统", version="1.0")
manager_agent = ManagerAgent()

# 创建上传目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# 挂载静态目录
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

# ============== 知识库相关接口 ==============

# 上传文档到知识库
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    from tools import process_and_store

    allowed_ext = [".txt", ".pdf", ".docx", ".doc"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return {"error": f"不支持的文件格式：{ext}，支持：{allowed_ext}"}

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = process_and_store(file_path)
    return {"filename": file.filename, "result": result}

# 查询知识库
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    from tools import search_knowledge_base, call_llm
    import re
    
    print(f"[🔗/query] 收到查询: {request.question}")
    
    chunks = search_knowledge_base(request.question, top_k=3)
    
    if not chunks:
        return {
            "question": request.question,
            "answer": "知识库中没有找到相关内容",
            "relevant_chunks": [],
            "chunk_count": 0
        }
    
    question = request.question
    need_summary = any(kw in question for kw in ["总结", "概括", "讲的啥", "主要内容", "简述", "概述"])
    
    word_limit = None
    match = re.search(r'(?:不超过|控制在|以内|)?(\d+)\s*字', question)
    if match:
        word_limit = int(match.group(1))
    
    context = "\n\n===\n\n".join(chunks[:3])
    
    if need_summary:
        if word_limit and word_limit <= 100:
            prompt = f"""你是一个专业的文档总结专家。

【任务】
用严格 {word_limit} 字（±5 字）总结【参考资料】的核心内容。

【输出要求】
- 只输出最终总结
- 严格 {word_limit} 字
- 语言流畅

【参考资料】
{context}

【最终总结】："""
        else:
            length_hint = f"约{word_limit}字" if word_limit else "2-3 句话"
            prompt = f"""你是文档总结助手，用{length_hint}概括核心内容：

【参考资料】
{context}

【总结】："""
    else:
        prompt = f"""严格使用参考资料回答，未提及则说"资料未提及"：

【参考资料】
{context}

【问题】{question}

【回答】："""
    
    try:
        answer = call_llm(prompt).strip()
        
        if word_limit and word_limit <= 100 and len(answer) > word_limit + 10:
            compress_prompt = f"精炼到{word_limit}字：\n{answer}"
            answer = call_llm(compress_prompt).strip()
        
    except Exception as e:
        answer = f"生成答案出错：{str(e)}"
    
    return {
        "question": question,
        "answer": answer,
        "relevant_chunks": chunks,
        "chunk_count": len(chunks)
    }

# 首页
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# 知识库状态
@app.get("/knowledge_base/status")
async def knowledge_base_status():
    count = kb_collection.count()
    return {"total_chunks": count}

# ============== 清空知识库（终极修复版）=============
@app.post("/knowledge_base/clear")
async def clear_knowledge_base():
    try:
        existing_data = kb_collection.get()
        ids = existing_data.get("ids", [])
        
        if not ids:
            return {"status": "success", "msg": "✅ 知识库已经是空的"}
            
        kb_collection.delete(ids=ids)
        
        return {
            "status": "success", 
            "msg": f"✅ 成功清空 {len(ids)} 条数据"
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"清空失败: {str(e)}"
        )