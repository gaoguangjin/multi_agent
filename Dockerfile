# 使用精简基础镜像，减少体积与攻击面
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（ChromaDB 向量化与文档解析需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# 1. 先复制依赖文件，利用 Docker 缓存层加速后续构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 复制完整项目代码
COPY . .

# 3. 预创建运行时目录（防止首次启动报错）
RUN mkdir -p uploads static

# 暴露端口
EXPOSE 8000

# 启动命令：单 worker 模式（2C2G 必选，多 worker 会 OOM）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]