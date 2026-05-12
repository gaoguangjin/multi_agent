# tools.py
import os
import re
import requests
import logging
import chromadb
from chromadb.config import Settings
from zhipuai import ZhipuAI
from openai import OpenAI
from langchain_community.tools import DuckDuckGoSearchRun

logger = logging.getLogger(__name__)

# ============== 初始化 ==============
# 智谱客户端
zhipu_client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

# 硅基流动客户端
silicon_client = OpenAI(
    api_key=os.getenv("SILICON_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

# ChromaDB（绝对路径）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_data"))
os.makedirs(CHROMA_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
kb_collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# ============== LLM调用 ==============
def call_llm(prompt: str, temperature: float = 0.3) -> str:
    response = zhipu_client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

def get_embedding(text: str) -> list:
    response = silicon_client.embeddings.create(
        model="BAAI/bge-m3",
        input=text,
    )
    return response.data[0].embedding

def rerank(query: str, documents: list, top_n: int = 5) -> list:
    if not documents:
        return []
    try:
        response = requests.post(
            "https://api.siliconflow.cn/v1/rerank",
            headers={
                "Authorization": f"Bearer {os.getenv('SILICON_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False
            },
            timeout=30
        )
        result = response.json()
        if "results" not in result:
            return documents[:top_n]
        return [documents[item["index"]] for item in result["results"]]
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        return documents[:top_n]

# ============== 工具函数 ==============
def translate_text(text: str, target_lang: str = "中文") -> str:
    prompt = f"把以下文本精准翻译成{target_lang}，保留专业术语和格式，只输出翻译结果：\n{text}"
    return call_llm(prompt)

def generate_email(content: str, email_type: str = "正式商务邮件") -> str:
    prompt = f"根据以下核心内容，生成一封{email_type}，格式规范，语气得体：\n{content}"
    return call_llm(prompt)

def generate_meeting_minutes(transcript: str) -> str:
    prompt = f"""
基于以下会议文本，生成结构化会议纪要：
1. 会议主题与时间  2. 参会人员  3. 核心议题
4. 关键结论与决议  5. 待办事项（负责人+截止时间）
文本：\n{transcript}
"""
    return call_llm(prompt)

def crawl_and_summarize(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding
        text = re.sub(r'<[^>]+>', '', response.text)[:5000]
        prompt = f"对以下网页内容精简总结，提取核心信息：\n{text}"
        return call_llm(prompt)
    except Exception as e:
        return f"网页爬取失败：{str(e)}"

def duckduckgo_search_logic(query: str) -> str:
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception as e:
        return f"搜索失败：{str(e)}"

# ============== RAG知识库 ==============
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if end >= len(text): break
    return chunks

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext in [".docx", ".doc"]:
        import docx
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError(f"不支持的格式：{ext}")

def process_and_store(file_path: str) -> str:
    try:
        text = extract_text_from_file(file_path)
        if not text.strip():
            return "文件内容为空"
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue
            embedding = get_embedding(chunk)
            doc_id = f"{os.path.basename(file_path)}_{i}"
            kb_collection.add(
                ids=[doc_id], documents=[chunk], embeddings=[embedding],
                metadatas=[{"source": os.path.basename(file_path), "chunk_index": i}]
            )
        return f"成功处理，切分为 {len(chunks)} 个片段"
    except Exception as e:
        logger.error(f"Process failed: {e}", exc_info=True)
        return f"处理失败：{str(e)}"

def search_knowledge_base(query: str, top_k: int = 5) -> list:
    try:
        total = kb_collection.count()
        if total == 0: return []
        
        query_embedding = get_embedding(query)
        retrieve_n = min(20, total)
        results = kb_collection.query(
            query_embeddings=[query_embedding], n_results=retrieve_n,
            include=["documents", "distances", "metadatas"]
        )
        
        if not results["documents"] or not results["documents"][0]:
            return []
        
        docs = results["documents"][0]
        valid_docs = [doc for doc in docs if len(doc.strip()) >= 30]
        if not valid_docs: return []
        
        reranked = rerank(query, valid_docs, top_n=top_k)
        
        # 总结类问题补充开头段落
        if any(kw in query for kw in ["总结", "概括", "讲的啥", "主要内容", "简述", "概述"]):
            intro = kb_collection.get(where={"chunk_index": {"$lte": 5}}, include=["documents"])
            if intro["documents"]:
                intro_docs = [d for d in intro["documents"] if d not in reranked]
                reranked = intro_docs + reranked
                reranked = reranked[:top_k]
        
        return reranked
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return []

def knowledge_qa_tool(question: str) -> str:
    if kb_collection.count() == 0:
        return "知识库为空，请先上传文档。"
    chunks = search_knowledge_base(question, top_k=5)
    if not chunks:
        return "知识库中未找到相关内容。"
    
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""你是一个严格的知识库问答机器人。
规则：
1. 只能使用【参考资料】中的信息
2. 禁止使用内置知识
3. 未提及则回答"参考资料中未提及此内容"

【参考资料】
{context}

【用户问题】{question}

【你的回答】"""
    return call_llm(prompt)