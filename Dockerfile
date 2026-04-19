# ------------------------------
# 阶段 1: 依赖构建（用 UV，超快）
# ------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# 1. 替换 APT 源为清华源（解决 Debian 官方源慢/断连）
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

# 2. 安装编译依赖（仅构建阶段需要，运行时不需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. 安装 UV（官方脚本，比 pip 装快 10 倍）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. 仅复制依赖文件（利用 Docker 缓存！代码改了不重装依赖）
# 注意：你需要先在本地生成 pyproject.toml 和 uv.lock（见下方说明）
COPY pyproject.toml uv.lock ./

# 5. 用 UV 安装依赖到虚拟环境（--frozen 锁定版本，--no-dev 排除开发依赖）
RUN uv sync --frozen --no-dev

# ------------------------------
# 阶段 2: 运行时镜像（仅保留必要文件）
# ------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# 1. 同样替换 APT 源（运行时可能需要安装一些基础库）
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

# 2. 仅安装运行时依赖（比如 ChromaDB/PostgreSQL 运行时需要的库，不需要 gcc/build-essential）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 3. 创建非 root 用户（安全最佳实践）
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# 4. 从构建阶段复制虚拟环境（仅复制这一个目录！）
COPY --from=builder --chown=appuser /app/.venv /app/.venv

# 5. 复制项目代码（排除 .venv/__pycache__ 等，见 .dockerignore）
COPY --chown=appuser . .

# 6. 预创建运行时目录
RUN mkdir -p uploads static

# 7. 暴露端口
EXPOSE 8000

# 8. 启动命令（使用虚拟环境中的 uvicorn）
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]