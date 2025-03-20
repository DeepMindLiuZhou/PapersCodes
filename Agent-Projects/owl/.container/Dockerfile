FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright \
    PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 安装系统依赖（合并为一个RUN命令减少层数）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ffmpeg libsm6 libxext6 xvfb xauth x11-utils \
    build-essential python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装uv工具
RUN pip install uv

# 复制构建项目文件
COPY pyproject.toml .
COPY README.md .
# 如果README.md不存在则创建
RUN if [ ! -f "README.md" ]; then echo -e "# OWL Project\n\n这是OWL项目的Docker环境。" > README.md; fi

# 创建虚拟环境并安装依赖
RUN uv venv .venv --python=3.10 && \
    . .venv/bin/activate && \
    uv pip install -e .

# 复制运行项目文件
COPY owl/ ./owl/
COPY licenses/ ./licenses/
COPY assets/ ./assets/
COPY README_zh.md .


# 创建启动脚本
RUN echo '#!/bin/bash\nxvfb-run --auto-servernum --server-args="-screen 0 1280x960x24" python "$@"' > /usr/local/bin/xvfb-python && \
    chmod +x /usr/local/bin/xvfb-python

# 创建欢迎脚本
RUN echo '#!/bin/bash\necho "欢迎使用OWL项目Docker环境！"\necho "Welcome to OWL Project Docker environment!"\necho ""\necho "可用的脚本 | Available scripts:"\nls -1 *.py | grep -v "__" | sed "s/^/- /"\necho ""\necho "运行示例 | Run examples:"\necho "  xvfb-python run.py                     # 运行默认脚本 | Run default script"\necho "  xvfb-python run_deepseek_example.py      # 运行DeepSeek示例 | Run DeepSeek example"\necho ""\necho "或者使用自定义查询 | Or use custom query:"\necho "  xvfb-python run.py \"你的问题 | Your question\""\necho ""' > /usr/local/bin/owl-welcome && \
    chmod +x /usr/local/bin/owl-welcome

# 设置工作目录
WORKDIR /app/owl

# 添加健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0 if __import__('os').path.exists('/app/owl') else 1)"

# 容器启动命令
CMD ["/bin/bash", "-c", "owl-welcome && /bin/bash"]