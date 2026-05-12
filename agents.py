# agent.py
import os
import json
import logging
from zhipuai import ZhipuAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 导入工具函数
from tools import (
    duckduckgo_search_logic, translate_text, generate_email,
    generate_meeting_minutes, crawl_and_summarize, knowledge_qa_tool
)

logger = logging.getLogger(__name__)

# 智谱客户端（管理者Agent用）
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

# 智谱LLM（LangChain兼容，子Agent用）
llm = ChatOpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    model="glm-4-flash",
    temperature=0.3,
)

# ============== 工具定义 ==============
@tool
def web_search_tool(query: str) -> str:
    """联网搜索工具"""
    return duckduckgo_search_logic(query)

@tool
def translate_tool(text: str, target_lang: str = "中文") -> str:
    """翻译工具"""
    return translate_text(text, target_lang)

@tool
def email_tool(content: str, email_type: str = "正式商务邮件") -> str:
    """邮件生成工具"""
    return generate_email(content, email_type)

@tool
def minutes_tool(transcript: str) -> str:
    """会议纪要工具"""
    return generate_meeting_minutes(transcript)

@tool
def crawl_tool(url: str) -> str:
    """网页爬取工具"""
    return crawl_and_summarize(url)

@tool
def knowledge_tool(question: str) -> str:
    """知识库问答工具"""
    return knowledge_qa_tool(question)

# ============== 子Agent定义 ==============
research_agent = create_react_agent(
    llm, tools=[web_search_tool, crawl_tool],
    prompt="你是专业的调研专家，负责通过网页爬取、信息整理，为任务收集精准、全面的背景资料。",
)

content_agent = create_react_agent(
    llm, tools=[email_tool, minutes_tool],
    prompt="你是专业的内容创作专家，擅长生成商务邮件、会议纪要、正式文案等内容。",
)

translate_agent = create_react_agent(
    llm, tools=[translate_tool],
    prompt="你是专业的翻译专家，精通中英日韩等多语言翻译，保证翻译精准、专业。",
)

knowledge_agent = create_react_agent(
    llm, tools=[knowledge_tool],
    prompt="""你是私有知识库问答专家。你的唯一职责是调用 knowledge_tool 获取本地文档信息。
⚠️ 严格纪律：
1. 如果工具返回"未找到"、"知识库为空"，直接回复："知识库中没有找到相关信息"。
2. 禁止使用内置常识回答问题。
3. 工具报错直接返回错误信息，不要编造答案。"""
)

# ============== ManagerAgent ==============
class ManagerAgent:
    def __init__(self):
        self.agent_map = {
            "调研": research_agent,
            "内容创作": content_agent,
            "翻译": translate_agent,
            "知识库": knowledge_agent,
        }

    def _extract_constraints(self, task: str) -> str:
        constraints = []
        if any(kw in task for kw in ["简约", "简短", "一句话", "1 句话", "超简", "100 字", "50 字", "30 字"]):
            constraints.append("严格遵循字数/句式限制，回答务必精简")
        if any(kw in task for kw in ["详细", "报告", "展开", "全面", "长篇"]):
            constraints.append("回答需详细展开，结构完整")
        if "列表" in task or "表格" in task:
            constraints.append("使用 Markdown 列表或表格格式")
        if any(kw in task for kw in ["幽默", "严肃", "正式", "口语"]):
            tone = [k for k in ['幽默','严肃','正式','口语'] if k in task][0]
            constraints.append(f"语气保持为：{tone}")
        return "；".join(constraints) if constraints else "无特殊要求"

    def _run_agent(self, agent, task_content: str) -> str:
        try:
            result = agent.invoke({"messages": [("user", task_content)]})
            last_msg = result["messages"][-1]
            return last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=True)
            return f"【执行错误】{str(e)}"

    def run(self, user_task: str) -> dict:
        try:
            user_constraints = self._extract_constraints(user_task)

            # 任务拆解
            prompt = f"""
用户任务：{user_task}
【用户特殊要求】：{user_constraints}

请将任务拆解为 1-3 个子任务，分配给以下 Agent 之一。
⚠️ Agent能力定义：
- 调研：联网搜索最新信息、影视评价、公司背景等外部公开信息
- 知识库：仅基于用户上传的本地私有文档回答问题
- 内容创作：根据前置信息撰写文章、邮件、总结等
- 翻译：多语言互译

⚠️ 核心原则：不要篡改用户核心诉求！

请严格输出 JSON：
{{
    "sub_tasks": [
        {{"agent": "调研", "task": "联网查询XX信息【遵守用户要求】"}},
        {{"agent": "内容创作", "task": "根据调研结果回答用户问题【遵守用户要求】"}}
    ]
}}"""

            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
            )
            raw_content = response.choices[0].message.content.strip()
            if raw_content.startswith("```"):
                raw_content = raw_content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            
            task_plan = json.loads(raw_content)

            # 执行子任务
            results = []
            for sub_task in task_plan["sub_tasks"]:
                agent_name = sub_task["agent"]
                task_content = sub_task["task"]

                if user_constraints != "无特殊要求" and user_constraints not in task_content:
                    task_content += f"\n【执行要求】{user_constraints}"
                if results:
                    task_content += f"\n【已有参考】\n{''.join(results[-2:])}"

                if agent_name not in self.agent_map:
                    results.append(f"【{agent_name}】跳过：无对应 Agent")
                    continue

                agent = self.agent_map[agent_name]
                output = self._run_agent(agent, task_content)
                results.append(f"【{agent_name}】结果：\n{output}\n")

            # 最终汇总
            final_prompt = f"""
【用户原始问题】：{user_task}
【必须遵守的约束】：{user_constraints}

基于以下执行结果生成最终回复：
⚠️ 规则：
1. 直击痛点：正面回答用户问题，如果问"哪个最好"必须给出明确选择
2. 忠于事实：严格基于参考结果，禁止捏造数据
3. 遵守约束，不添加废话

参考结果：
{''.join(results)}

最终回复（直接输出）："""

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

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}", exc_info=True)
            return {"error": "任务拆解失败：JSON解析错误", "raw_output": raw_content if 'raw_content' in locals() else None}
        except Exception as e:
            logger.error(f"Manager run failed: {e}", exc_info=True)
            return {"error": f"任务执行异常: {str(e)}"}