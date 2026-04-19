# 🤖 多智能体协同办公系统 (Multi-Agent Office Assistant)

基于 **LangChain + LangGraph + ZhipuAI** 构建的企业级智能办公助手，支持任务拆解、多智能体协同、知识库问答、内容创作等能力。

---

## ✨ 功能特性

| 功能模块 | 说明 |
|---------|------|
| 🔍 智能调研 | 自动爬取网页内容，提取关键信息并生成结构化摘要 |
| ✍️ 内容创作 | 一键生成商务邮件、会议纪要、正式文案等专业内容 |
| 🌐 多语言翻译 | 支持中英日韩等多语言精准翻译，保留专业术语与格式 |
| 📚 知识库问答 | 基于上传文档构建向量知识库，支持精准问答并标注来源 |
| 🧠 任务编排 | 管理者Agent自动拆解复杂任务，调度专业子Agent协同执行 |
| 🔌 API 服务 | 提供 RESTful 接口，便于集成到企业现有系统 |

---

## 🛠️ 技术栈

Python 3.10+ | FastAPI | LangChain | LangGraph | ZhipuAI | ChromaDB | Docker

text
text

---

## 📂 项目结构

multi_agent/
├── main.py # FastAPI 入口
├── agents/ # 各专业 Agent
│ ├── manager_agent.py # 任务调度器
│ ├── research_agent.py # 调研 Agent
│ ├── content_agent.py # 内容创作 Agent
│ ├── translate_agent.py # 翻译 Agent
│ └── knowledge_agent.py # 知识库问答 Agent
├── tools/ # 工具集
├── rag/ # RAG 知识库模块
├── config/ # 配置文件
├── .env.example # 环境变量模板
└── requirements.txt # 依赖清单

text
text

---

## 📦 安装与配置

### 1️⃣ 环境准备

```bash
# 克隆项目
git clone https://github.com/gaoguangjin/multi_agent.git
cd multi_agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

2️⃣ 配置环境变量

复制 .env.example 为 .env 并填写你的 API Key：


bash
bash
cp .env.example .env

.env 文件内容示例：


env
env
ZHIPUAI_API_KEY=your_zhipuai_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
EMBEDDING_MODEL=bge-m3

💡 获取 API Key：

智谱AI：
开放平台控制台
硅基流动：
个人中心 - API密钥


🚀 快速启动

bash
bash
python main.py

服务启动后访问：


📖 API 文档：
http://localhost:8000/docs
🧪 接口测试：
http://localhost:8000/redoc


📡 API 接口说明

🔹 执行智能任务（核心接口）

http
http
POST /agent/run
Content-Type: application/json

{
  "task": "帮我调研2024年AI Agent行业趋势，生成一份500字的分析报告"
}

响应示例：


json
json
{
  "status": "success",
  "result": "根据调研，2024年AI Agent行业呈现以下趋势...",
  "steps": [
    {"agent": "research_agent", "output": "..."},
    {"agent": "content_agent", "output": "..."}
  ]
}

🔹 上传文档到知识库

http
http
POST /upload
Content-Type: multipart/form-data

file: <your_document.pdf>

响应示例：


json
json
{
  "status": "success",
  "message": "文档已成功索引，共切分为 42 个文本块"
}

🔹 直接查询知识库

http
http
POST /knowledge/query
Content-Type: application/json

{
  "question": "产品的退货政策是什么？"
}

响应示例：


json
json
{
  "answer": "根据产品手册第3章，退货政策为...",
  "sources": ["product_manual.pdf (第15页)"]
}

🔹 查看知识库状态

http
http
GET /knowledge/status

响应示例：


json
json
{
  "document_count": 3,
  "chunk_count": 128
}


💡 使用示例

场景 1：自动生成竞品分析报告

bash
bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "调研Notion AI和Copilot的功能差异，生成竞品分析报告"}'

场景 2：上传产品手册后问答

bash
bash
# 上传文档
curl -X POST http://localhost:8000/upload \
  -F "file=@product_manual.pdf"

# 提问
curl -X POST http://localhost:8000/knowledge/query \
  -H "Content-Type: application/json" \
  -d '{"question": "如何重置管理员密码？"}'

场景 3：多语言商务邮件生成

bash
bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "写一封英文商务邮件，感谢客户John参加上周的产品演示，并附上会议纪要"}'


⚙️ 核心模块说明

🧠 ManagerAgent（任务调度器）

1.接收用户自然语言任务
2.调用 LLM 智能拆解为 1~3 个子任务
3.按顺序调度专业 Agent 执行，传递上下文
4.汇总各子任务结果，生成最终报告

🤖 专业 Agent 集群

Agent	职责	可用工具
research_agent	信息调研与整理	crawl_tool（网页爬取+总结）
content_agent	商务内容创作	email_tool, minutes_tool
translate_agent	多语言精准翻译	translate_tool
knowledge_agent	知识库精准问答	knowledge_tool（RAG检索+严格约束回答）

📚 RAG 知识库流程

text
text
用户上传文档
    ↓
文本切分（RecursiveCharacterTextSplitter）
    ↓
向量化（智谱 Embedding / BGE-M3）
    ↓
存入 ChromaDB
    ↓
用户提问 → 向量检索 Top-K 相关片段
    ↓
拼接 Prompt → LLM 生成回答（附来源标注）


⚠️ 注意事项

API 配额：智谱与硅基流动均有免费额度，生产环境建议监控用量
知识库限制：
单文件建议 ≤ 10MB，避免处理超时
当前为内存级 Chroma，重启后数据丢失，生产环境需配置持久化
网络要求：crawl_tool 需服务器可访问外网
安全建议：
生产环境请添加 API 认证（如 JWT）
限制 /upload 接口文件类型与大小
.env 文件勿提交至版本库


🤝 贡献指南

欢迎提交 Issue 或 Pull Request！建议遵循：


1.新功能请先开 Issue 讨论方案
2.代码提交前运行 black . 格式化
3.接口变更请同步更新 /docs 中的示例


📜 许可证

本项目采用 
MIT License
 开源，可自由使用、修改与分发。



🌟 Star 不迷路：如果本项目对你有帮助，欢迎点个 ⭐ 支持开源！


📬 问题反馈：遇到使用问题请提交 
Issue
，我们会尽快响应。
