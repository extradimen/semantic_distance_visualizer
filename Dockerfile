# 使用Python 3.12官方镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    fonts-wqy-microhei \
    fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

# 复制整个项目目录（包括.venv虚拟环境）
# 注意：.venv 会被复制进去，模型缓存也会被复制（如果在项目目录中）
COPY . .

# 如果项目目录中有模型缓存，复制到标准位置
RUN if [ -d ".cache/huggingface" ]; then \
        mkdir -p /root/.cache/huggingface && \
        cp -r .cache/huggingface/* /root/.cache/huggingface/ 2>/dev/null || true; \
    fi

# 确保目录权限正确
RUN chmod +x start_service.sh stop_service.sh 2>/dev/null || true

# 创建必要的目录（如果不存在）
RUN mkdir -p uploads results /root/.cache/huggingface

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PATH="/app/.venv/bin:$PATH"
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

# 暴露端口
EXPOSE 5000

# 使用虚拟环境中的Python启动应用
CMD [".venv/bin/python", "app.py"]
