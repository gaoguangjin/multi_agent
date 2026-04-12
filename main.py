from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from agents import ManagerAgent
import shutil
import os

app = FastAPI(title="多智能体协同办公系统", version="1.0")
manager_agent = ManagerAgent()

# 创建上传目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    from tools import search_knowledge_base

    chunks = search_knowledge_base(request.question, top_k=1)
    if not chunks:
        return {"answer": "知识库中没有找到相关内容", "sources": []}

    return {
        "question": request.question,
        "relevant_chunks": chunks,
        "chunk_count": len(chunks)
    }



# 查看知识库状态
@app.get("/knowledge_base/status")
async def knowledge_base_status():
    from tools import kb_collection
    count = kb_collection.count()
    return {"total_chunks": count}


# 启动命令：uvicorn main:app --host 0.0.0.0 --reload
