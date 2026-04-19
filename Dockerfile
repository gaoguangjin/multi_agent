FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# 核心：国内源，速度飞起来
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .
RUN mkdir -p uploads static
RUN chmod -R 777 /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
