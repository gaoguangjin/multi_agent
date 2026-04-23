from tools import duckduckgo_search_logic
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
def web_search_tool(query: str) -> str:
    """当你需要调研公司信息、寻找特定网址、或者查询最新实时动态时，必须优先使用此工具。
    参数 query 应该是具体的搜索关键词。"""
    return duckduckgo_search_logic(query)

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
    tools=[web_search_tool, crawl_tool],
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
    prompt="""你是私有知识库问答专家。你的唯一职责是调用 knowledge_tool 获取本地文档信息。
⚠️ 严格纪律：
1. 如果工具返回的内容包含“未找到”、“知识库为空”，你必须直接回复：“知识库中没有找到相关信息”。
2. 绝对禁止使用你的内置常识或预训练知识来回答问题。
3. 如果工具报错，直接返回报错信息。不要试图去弥补或编造答案。"""
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

请将任务拆解为 1-3 个子任务，分配给以下 Agent 之一。
⚠️ 各 Agent 能力定义（必须严格按此规范分配）：
- 调研：负责联网搜索最新信息、影视评价、公众知识、公司背景等（只要是需要查网的外部公开信息，必须用此Agent）。
- 知识库：仅负责基于用户已经上传的本地私有文档回答问题。严禁将互联网公开常识或影视娱乐问题分配给知识库！
- 内容创作：负责根据前置Agent收集的信息，撰写文章、邮件、总结等。
- 翻译：负责多语言互译。

⚠️ 拆解核心原则（防止意图丢失）：
不要篡改用户的核心诉求！如果用户要求“挑选一个最好的并深入分析”，子任务的指令就必须是“挑选评分最高的一部进行深度解析”，绝对不能降级成“列出所有列表并简要介绍”。

请严格输出 JSON，格式如下：
{{
    "sub_tasks": [
        {{"agent": "调研", "task": "联网查询XX信息的详细数据或评价【请遵守用户要求】"}},
        {{"agent": "内容创作", "task": "根据调研结果，详细论述并直接回答用户问题【请遵守用户要求】"}}
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
【用户原始核心问题】：{user_task}
【必须遵守的约束】：{user_constraints}

请基于以下 Agent 执行结果，为用户生成最终回复。
⚠️ 强制规则：
1. 直击痛点：必须正面回答【用户原始核心问题】。如果用户问“哪个最好”，你必须给出一个明确的选择，绝不能只给一个模棱两可的列表。
2. 忠于事实：严格基于下面的“参考结果”进行总结，严禁自己凭空捏造数据或评价。如果参考结果中没有答案，请诚实说明无法得出结论。
3. 严格遵守【必须遵守的约束】，绝不添加多余解释、客套话或废话。

参考结果：
{''.join(results)}

最终回复（直接输出，无需前缀）："""

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
