from zhipuai import ZhipuAI
from dotenv import load_dotenv
import os
import requests
import re

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

# 工具1：文本翻译（支持多语言）
def translate_text(text: str, target_lang: str = "中文") -> str:
    """
    翻译文本，支持多语言
    :param text: 要翻译的文本
    :param target_lang: 目标语言，默认中文
    :return: 翻译结果
    """
    prompt = f"把以下文本精准翻译成{target_lang}，保留专业术语和格式，只输出翻译结果：\n{text}"
    response = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# 工具2：邮件内容生成
def generate_email(content: str, email_type: str = "正式商务邮件") -> str:
    """
    生成标准化邮件
    :param content: 邮件核心内容
    :param email_type: 邮件类型，如商务邮件、请假邮件、感谢信等
    :return: 完整邮件内容
    """
    prompt = f"根据以下核心内容，生成一封{email_type}，格式规范，语气得体：\n{content}"
    response = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# 工具3：会议纪要生成
def generate_meeting_minutes(transcript: str) -> str:
    """
    从会议录音文本生成结构化会议纪要
    :param transcript: 会议录音转写文本
    :return: 结构化会议纪要
    """
    prompt = f"""
    基于以下会议文本，生成结构化会议纪要，包含以下模块：
    1. 会议主题与时间
    2. 参会人员
    3. 会议核心议题
    4. 关键结论与决议
    5. 待办事项（含负责人、截止时间）
    文本内容：\n{transcript}
    """
    response = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# 工具4：网页内容爬取与总结
def crawl_and_summarize(url: str) -> str:
    """
    爬取网页内容并生成总结
    :param url: 网页链接
    :return: 网页内容总结
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding
        # 去除HTML标签，提取纯文本
        text = re.sub(r'<[^>]+>', '', response.text)
        # 截断过长文本
        text = text[:5000]
        # 生成总结
        prompt = f"对以下网页内容进行精简总结，提取核心信息和关键观点：\n{text}"
        summary_response = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}])
        return summary_response.choices[0].message.content
    except Exception as e:
        return f"网页爬取失败：{str(e)}"