from zhipuai import ZhipuAI
from dotenv import load_dotenv
import os
import requests
import re
import chromadb
from chromadb.config import Settings
from langchain_core.tools import tool

load_dotenv()

# 智谱LLM客户端
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


# Embedding（用硅基流动）
def get_embedding(text: str) -> list:
    from openai import OpenAI
    silicon_client = OpenAI(
        api_key=os.getenv("SILICON_API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
    )
    response = silicon_client.embeddings.create(
        model="BAAI/bge-m3",
        input=text,
    )
    return response.data[0].embedding



# ============== 工具1：文本翻译 ==============

def translate_text(text: str, target_lang: str = "中文") -> str:
    prompt = f"把以下文本精准翻译成{target_lang}，保留专业术语和格式，只输出翻译结果：\n{text}"
    return call_llm(prompt)


# ============== 工具2：邮件生成 ==============

def generate_email(content: str, email_type: str = "正式商务邮件") -> str:
    prompt = f"根据以下核心内容，生成一封{email_type}，格式规范，语气得体：\n{content}"
    return call_llm(prompt)


# ============== 工具3：会议纪要 ==============

def generate_meeting_minutes(transcript: str) -> str:
    prompt = f"""
    基于以下会议文本，生成结构化会议纪要，包含以下模块：
    1. 会议主题与时间
    2. 参会人员
    3. 会议核心议题
    4. 关键结论与决议
    5. 待办事项（含负责人、截止时间）
    文本内容：\n{transcript}
    """
    return call_llm(prompt)


# ============== 工具4：网页爬取与总结 ==============

def crawl_and_summarize(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding
        text = re.sub(r'<[^>]+>', '', response.text)
        text = text[:5000]
        prompt = f"对以下网页内容进行精简总结，提取核心信息和关键观点：\n{text}"
        return call_llm(prompt)
    except Exception as e:
        return f"网页爬取失败：{str(e)}"


# ============== RAG 知识库 ==============

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
kb_collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)


# 文本切分
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break
    return chunks


# 从文件提取文本
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
        raise ValueError(f"不支持的文件格式：{ext}")


# 处理上传的文件
def process_and_store(file_path: str) -> str:
    try:
        text = extract_text_from_file(file_path)
        if not text.strip():
            return "文件内容为空"

        chunks = chunk_text(text)
        if not chunks:
            return "文本切分失败"

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            embedding = get_embedding(chunk)
            doc_id = f"{os.path.basename(file_path)}_{i}"
            kb_collection.add(
                ids=[doc_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": os.path.basename(file_path), "chunk_index": i}]
            )

        return f"成功处理文件，共切分为 {len(chunks)} 个片段并存入知识库"

    except Exception as e:
        return f"处理文件失败：{str(e)}"


# 从知识库检索
def search_knowledge_base(query: str, top_k: int = 3) -> list:
    try:
        query_embedding = get_embedding(query)
        results = kb_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, kb_collection.count())
        )
        if results["documents"] and results["documents"][0]:
            # 过滤掉太短或不相关的片段
            docs = results["documents"][0]
            filtered = [d for d in docs if len(d.strip()) > 20]
            return filtered[:top_k]
        return []
    except Exception:
        return []



# 知识库问答（普通函数，不加@tool）
def knowledge_qa_tool(question: str) -> str:
    if kb_collection.count() == 0:
        return "知识库为空，请先通过 /upload 接口上传文档。"

    chunks = search_knowledge_base(question, top_k=1)

    if not chunks:
        return "知识库中未找到与该问题相关的内容。"

    context = chunks[0]

    prompt = f"""你是一个严格的知识库问答机器人。

规则：
1. 你只能使用下面【参考资料】中的信息回答问题
2. 禁止使用你自己的知识或训练数据
3. 如果【参考资料】中没有提到问题相关内容，必须回答："参考资料中未提及此内容"
4. 回答要简短，一句话即可

【参考资料】
{context}

【用户问题】
{question}

【你的回答】"""

    return call_llm(prompt)



