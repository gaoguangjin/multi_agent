# main.py
from dotenv import load_dotenv
load_dotenv()  
import os
import shutil
import re
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import asyncio

# 项目内导入
from agents import ManagerAgent
from tools import (
    kb_collection, process_and_store, search_knowledge_base,
    knowledge_qa_tool, call_llm
)

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 检查环境变量
def check_env():
    required = ["ZHIPUAI_API_KEY", "SILICON_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"❌ 缺少环境变量: {missing}")

# 创建应用
app = FastAPI(title="多智能体协同办公系统", version="1.0")

# 线程池（解决async/sync阻塞问题）
executor = ThreadPoolExecutor(max_workers=4)

# 初始化Agent
manager_agent = ManagerAgent()

# 创建目录（绝对路径）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ============== 数据模型 ==============
class TaskRequest(BaseModel):
    task: str

class QueryRequest(BaseModel):
    question: str

# ============== 健康检查 ==============
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "env_loaded": bool(os.getenv("ZHIPUAI_API_KEY")),
        "chroma_count": kb_collection.count() if kb_collection else 0
    }

# ============== 核心接口 ==============
@app.post("/run_task")
async def run_agent_task(request: TaskRequest):
    """执行Agent任务（异步包装同步调用）"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, manager_agent.run, request.task)
        return result
    except Exception as e:
        logger.error(f"Task failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"任务执行失败: {str(e)}")

# ============== 知识库接口 ==============
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档到知识库"""
    try:
        allowed_ext = [".txt", ".pdf", ".docx", ".doc"]
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_ext:
            return {"error": f"不支持的格式: {ext}", "supported": allowed_ext}

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_and_store(file_path)
        return {"filename": file.filename, "result": result}
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    """查询知识库"""
    try:
        logger.info(f"[🔗/query] 查询: {request.question}")
        chunks = search_knowledge_base(request.question, top_k=5)

        if not chunks:
            return {"question": request.question, "answer": "知识库中没有找到相关内容", "relevant_chunks": [], "chunk_count": 0}

        question = request.question
        need_summary = any(kw in question for kw in ["总结", "概括", "讲的啥", "主要内容", "简述", "概述"])
        word_limit = None
        match = re.search(r'(?:不超过|控制在|以内|)?(\d+)\s*字', question)
        if match:
            word_limit = int(match.group(1))

        context = "\n\n===\n\n".join(chunks)

        if need_summary:
            if word_limit and word_limit <= 100:
                prompt = f"""你是一个专业的文档总结专家。
【任务】用严格 {word_limit} 字（±5 字）总结【参考资料】的核心内容。
【输出要求】只输出最终总结；严格 {word_limit} 字；语言流畅
【参考资料】{context}
【最终总结】："""
            else:
                length_hint = f"约{word_limit}字" if word_limit else "2-3 句话"
                prompt = f"""你是文档总结助手，用{length_hint}概括核心内容：
【参考资料】{context}
【总结】："""
        else:
            prompt = f"""严格使用参考资料回答，未提及则说"资料未提及"：
【参考资料】{context}
【问题】{question}
【回答】："""

        answer = call_llm(prompt).strip()
        if word_limit and word_limit <= 100 and len(answer) > word_limit + 10:
            compress_prompt = f"精炼到{word_limit}字：\n{answer}"
            answer = call_llm(compress_prompt).strip()

        return {"question": question, "answer": answer, "relevant_chunks": chunks, "chunk_count": len(chunks)}
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.get("/knowledge_base/status")
async def knowledge_base_status():
    return {"total_chunks": kb_collection.count()}

@app.post("/knowledge_base/clear")
async def clear_knowledge_base():
    try:
        existing_data = kb_collection.get()
        ids = existing_data.get("ids", [])
        if not ids:
            return {"status": "success", "msg": "✅ 知识库已经是空的"}
        kb_collection.delete(ids=ids)
        return {"status": "success", "msg": f"✅ 成功清空 {len(ids)} 条数据"}
    except Exception as e:
        logger.error(f"Clear KB failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")

# ============== 首页 ==============
@app.get("/")
async def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "系统运行中，访问 /docs 查看API"}

# ============== 启动事件 ==============
@app.on_event("startup")
def on_startup():
    check_env()
    logger.info("✅ 应用启动成功")

# ============== 全局异常处理 ==============
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return {"error": "Internal server error", "detail": str(exc)}