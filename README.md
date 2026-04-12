🤖 多智能体协同办公系统 (Multi-Agent Office Assistant)
基于 LangChain + LangGraph + ZhipuAI 构建的企业级智能办公助手，支持任务拆解、多智能体协同、知识库问答、内容创作等能力。
✨ 功能特性
功能模块
说明
🔍 智能调研
自动爬取网页内容，提取关键信息并生成结构化摘要
✍️ 内容创作
一键生成商务邮件、会议纪要、正式文案等专业内容
🌐 多语言翻译
支持中英日韩等多语言精准翻译，保留专业术语与格式
📚 知识库问答
基于上传文档构建向量知识库，支持精准问答并标注来源
🧠 任务编排
管理者Agent自动拆解复杂任务，调度专业子Agent协同执行
🔌 API 服务
提供 RESTful 接口，便于集成到企业现有系统
🛠️ 技术栈
123456
📂 项目结构
12345678
📦 安装与配置
1️⃣ 环境准备
bash
123456789
2️⃣ 配置环境变量
复制 .env.example 为 .env 并填写你的 API Key：
bash
1
.env 文件内容示例：
env
12345
💡 获取 API Key：
智谱AI：开放平台控制台
硅基流动：个人中心 - API密钥
🚀 快速启动
bash
12
服务启动后访问：
📖 API 文档：http://localhost:8000/docs
🧪 接口测试：http://localhost:8000/redoc
📡 API 接口说明
🔹 执行智能任务（核心接口）
http
123456
响应示例：
json
12345678910111213141516
🔹 上传文档到知识库
http
1234
响应示例：
json
1234
🔹 直接查询知识库
http
123456
响应示例：
json
1234567
🔹 查看知识库状态
http
1
响应示例：
json
123
💡 使用示例
场景 1：自动生成竞品分析报告
bash
123
场景 2：上传产品手册后问答
bash
12345678
场景 3：多语言商务邮件生成
bash
123
⚙️ 核心模块说明
🧠 ManagerAgent（任务调度器）
接收用户自然语言任务
调用 LLM 智能拆解为 1~3 个子任务
按顺序调度专业 Agent 执行，传递上下文
汇总各子任务结果，生成最终报告
🤖 专业 Agent 集群
Agent
职责
可用工具
research_agent
信息调研与整理
crawl_tool（网页爬取+总结）
content_agent
商务内容创作
email_tool, minutes_tool
translate_agent
多语言精准翻译
translate_tool
knowledge_agent
知识库精准问答
knowledge_tool（RAG检索+严格约束回答）
📚 RAG 知识库流程
1234567891011
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
新功能请先开 Issue 讨论方案
代码提交前运行 black . 格式化
接口变更请同步更新 /docs 中的示例
📜 许可证
本项目采用 MIT License 开源，可自由使用、修改与分发。
🌟 Star 不迷路：如果本项目对你有帮助，欢迎点个 ⭐ 支持开源！
📬 问题反馈：遇到使用问题请提交 Issue，我们会尽快响应。