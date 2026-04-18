from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tools import translate_text, generate_email, generate_meeting_minutes, crawl_and_summarize, knowledge_qa_tool
from zhipuai import ZhipuAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

# 智谱LLM（LangChain用，OpenAI兼容格式）
llm = ChatOpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    model="glm-4-flash",
    temperature=0.3,
)

# 智谱原生客户端（管理者Agent直接调用）
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))


# 工具定义
@tool
def translate_tool(text: str, target_lang: str = "中文") -> str:
    """翻译文本到目标语言，支持中英日韩等多语言。参数：text(要翻译的文本), target_lang(目标语言，默认中文)"""
    return translate_text(text, target_lang)


@tool
def email_tool(content: str, email_type: str = "正式商务邮件") -> str:
    """生成标准化邮件。参数：content(邮件核心内容), email_type(邮件类型)"""
    return generate_email(content, email_type)


@tool
def minutes_tool(transcript: str) -> str:
    """从会议文本生成结构化会议纪要。参数：transcript(会议文本内容)"""
    return generate_meeting_minutes(transcript)


@tool
def crawl_tool(url: str) -> str:
    """爬取网页内容并生成总结。参数：url(网页链接)"""
    return crawl_and_summarize(url)


@tool
def knowledge_tool(question: str) -> str:
    """基于已上传的文档知识库回答问题。参数：question(用户的问题)"""
    return knowledge_qa_tool(question)


# 1. 调研Agent
research_agent = create_react_agent(
    llm,
    tools=[crawl_tool],
    prompt="你是专业的调研专家，负责通过网页爬取、信息整理，为任务收集精准、全面的背景资料。",
)

# 2. 内容创作Agent
content_agent = create_react_agent(
    llm,
    tools=[email_tool, minutes_tool],
    prompt="你是专业的内容创作专家，擅长生成商务邮件、会议纪要、正式文案等内容。",
)

# 3. 翻译Agent
translate_agent = create_react_agent(
    llm,
    tools=[translate_tool],
    prompt="你是专业的翻译专家，精通中英日韩等多语言翻译，保证翻译精准、专业，符合目标语言的表达习惯。",
)

# 4. 知识库问答Agent
knowledge_agent = create_react_agent(
    llm,
    tools=[knowledge_tool],
    prompt="你是专业的知识库问答专家，能够根据用户上传的文档内容，准确回答相关问题，并标注信息来源。",
)


# 管理者Agent
# ============== agent.py ==============
class ManagerAgent:
    def __init__(self):
        self.agent_map = {
            "调研": research_agent,
            "内容创作": content_agent,
            "翻译": translate_agent,
            "知识库": knowledge_agent,
        }

    def _extract_constraints(self, task: str) -> str:
        """轻量规则提取用户格式/长度/语气要求"""
        constraints = []
        if any(kw in task for kw in ["简约", "简短", "一句话", "1 句话", "超简", "100 字", "50 字", "30 字"]):
            constraints.append("严格遵循字数/句式限制，回答务必精简，不要废话")
        if any(kw in task for kw in ["详细", "报告", "展开", "全面", "长篇"]):
            constraints.append("回答需详细展开，结构完整，包含必要背景")
        if "列表" in task or "表格" in task:
            constraints.append("使用 Markdown 列表或表格格式呈现")
        if any(kw in task for kw in ["幽默", "严肃", "正式", "口语"]):
            constraints.append(f"语气保持为：{[k for k in ['幽默','严肃','正式','口语'] if k in task][0]}")
        return "；".join(constraints) if constraints else "无特殊格式/字数要求"

    def _run_agent(self, agent, task_content):
        result = agent.invoke({"messages": [("user", task_content)]})
        last_msg = result["messages"][-1]
        return last_msg.content

    def run(self, user_task):
        # 0. 提取并保留用户约束
        user_constraints = self._extract_constraints(user_task)

        # 1. 拆解任务
        prompt = f"""
用户任务：{user_task}
【用户特殊要求】：{user_constraints}

请将任务拆解为 1-3 个子任务，分配给以下 Agent 之一：调研、内容创作、翻译、知识库。
⚠️ 关键：必须将【用户特殊要求】原样写入每个子任务的 "task" 字段中！

输出格式（严格 JSON，不要其他内容）：
{{
    "sub_tasks": [
        {{"agent": "知识库", "task": "查询 XX 信息【请遵守用户要求】"}},
        {{"agent": "内容创作", "task": "生成 XX 文案【请遵守用户要求】"}}
    ]
}}"""

        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
        )
        raw_content = response.choices[0].message.content.strip()
        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            task_plan = json.loads(raw_content)
        except Exception as e:
            return f"任务拆解失败：{str(e)}\n原始输出：{raw_content}"

        # 2. 顺序执行子任务
        results = []
        for sub_task in task_plan["sub_tasks"]:
            agent_name = sub_task["agent"]
            task_content = sub_task["task"]

            # 强制注入约束（防止拆解阶段遗漏）
            if user_constraints != "无特殊格式/字数要求" and user_constraints not in task_content:
                task_content += f"\n【执行要求】{user_constraints}"

            # 附加历史结果作为上下文
            if results:
                task_content += f"\n【已有参考】\n{''.join(results[-2:])}"  # 只带最近 2 个，防上下文爆炸

            if agent_name not in self.agent_map:
                results.append(f"【{agent_name}】跳过：无对应 Agent")
                continue

            agent = self.agent_map[agent_name]
            output = self._run_agent(agent, task_content)
            results.append(f"【{agent_name}】结果：\n{output}\n")

        # 3. 最终汇总（核心修复：对齐用户约束）
        final_prompt = f"""
用户原始请求：{user_task}
【必须遵守的约束】：{user_constraints}

请基于以下 Agent 执行结果，生成最终回复。
⚠️ 规则：
1. 严格遵守【必须遵守的约束】，绝不添加多余解释、客套话或章节标题
2. 如果约束要求“简约/一句话”，直接给核心答案，不要铺垫
3. 语言与用户提问保持一致，不要中英混杂

参考结果：
{''.join(results)}

最终回复（直接输出给用户的内容，不要带任何前缀）："""

        final_response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": final_prompt}],
        )

        return {
            "task": user_task,
            "task_plan": task_plan,
            "process_results": results,
            "final_result": final_response.choices[0].message.content.strip()
        }
