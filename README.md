# Multi-Agent 多智能体协同办公系统
基于 LangChain+LangGraph+智谱AI+硅基流动 构建的企业级智能助手，支持任务拆解、多Agent协同、RAG知识库问答、内容创作。

## 📝 更新日志 2026年5月8日
1.向量检索 top-3 扩大到 top-20，新增 Reranker 精排
2.总结类问题自动补充开头简介
3.优化 prompt，忽略后记等非正文内容

## 📝 更新日志 2026年4月22日
1.新增爬取网站功能
2.前端页面增加引导按钮

## 📝 更新日志 2026年4月19日
1. 新增可视化前端页面
2. 优化Dockerfile，提升构建速度
3. 优化AI推理逻辑，智能度大幅提升

## ✨ 核心功能
- 智能调研：网页爬取+结构化信息总结
- 内容创作：商务文案/邮件/会议纪要生成
- 多语言翻译：中英日韩专业翻译
- RAG问答：文档知识库检索+来源标注
- 任务编排：ManagerAgent自动拆解调度任务
- 开放API：RESTful接口，快速集成

## 🛠 技术栈
Python 3.11+ | FastAPI | LangChain | LangGraph | ZhipuAI | ChromaDB | Docker

## 🚀 快速启动
### 1. 本地部署
```bash
git clone https://github.com/gaoguangjin/multi_agent.git
cd multi_agent
# 配置环境变量
cp .env.example .env  # 填写ZHIPUAI_API_KEY
# 安装依赖&启动
pip install -r requirements.txt
python main.py
uv run uvicorn main:app --reload
```

### 2. Docker部署
```bash
docker-compose up -d
```

## 🌐 访问地址
- 在线演示：http://hlml.site:8000/
- 接口文档：http://localhost:8000/docs
- 接口测试：http://localhost:8000/redoc

## 📡 核心API
| 接口 | 方法 | 说明 |
| ---- | ---- | ---- |
| `/agent/run` | POST | 执行多Agent协同任务 |
| `/upload` | POST | 上传文档至知识库 |
| `/knowledge/query` | POST | 知识库精准问答 |
| `/knowledge/status` | GET | 查看知识库状态 |

## ⚠️ 注意事项
1. 需配置智谱AI API Key，支持免费额度
2. 硅基流动注册送限量API，可进一步优化
3. 单文件建议≤10MB，内存版Chroma重启数据丢失
