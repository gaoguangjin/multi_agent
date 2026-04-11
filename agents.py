from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tools import translate_text, generate_email, generate_meeting_minutes, crawl_and_summarize
from zhipuai import ZhipuAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

# 智谱AI 直接用 OpenAI 兼容接口
llm = ChatOpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    model="glm-4-flash",
    temperature=0.3, # 控制输出的随机性，值越大，输出越随机
)

# 管理者用的原生客户端
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


# 4. 管理者Agent
class ManagerAgent:
    def __init__(self):
        self.agent_map = {
            "调研": research_agent,
            "内容创作": content_agent,
            "翻译": translate_agent,
        }

    def _run_agent(self, agent, task_content):
        result = agent.invoke({"messages": [("user", task_content)]})
        last_msg = result["messages"][-1]
        return last_msg.content

    def run(self, user_task):
        # 第一步：拆解任务
        prompt = f"""
        用户的任务是：{user_task}
        请你把这个任务拆解成1-3个子任务，每个子任务只能分配给以下Agent之一：调研、内容创作、翻译
        输出格式要求：严格按照JSON格式输出，示例如下：
        {{
            "sub_tasks": [
                {{
                    "agent": "调研",
                    "task": "爬取XX官网的产品信息，总结核心功能"
                }},
                {{
                    "agent": "内容创作",
                    "task": "基于调研结果，生成一封给客户的产品介绍邮件"
                }}
            ]
        }}
        只输出JSON，不要其他内容。
        """
        response = client.chat.completions.create(
            model="glm-4-flash", messages=[{"role": "user", "content": prompt}]
        )
        try:
            task_plan = json.loads(response.choices[0].message.content)
        except:
            return "任务拆解失败，请重新描述你的需求"

        # 第二步：按顺序执行子任务
        results = []
        for sub_task in task_plan["sub_tasks"]:
            agent_name = sub_task["agent"]
            task_content = sub_task["task"]
            if agent_name not in self.agent_map:
                results.append(f"【{agent_name}】任务执行失败：无对应Agent")
                continue
            if results:
                task_content += f"\n已有参考信息：\n{''.join(results)}"
            agent = self.agent_map[agent_name]
            output = self._run_agent(agent, task_content)
            results.append(f"【{agent_name}】执行结果：\n{output}\n")

        # 第三步：汇总最终结果
        final_prompt = f"""
        基于以下各Agent的执行结果，生成一份完整、清晰的最终报告，给用户呈现最终成果。
        执行结果：
        {''.join(results)}
        """
        final_response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": final_prompt}],
        )
        return {
            "task": user_task,
            "task_plan": task_plan,
            "process_results": results,
            "final_result": final_response.choices[0].message.content,
        }
