#!/bin/bash
# Docker镜像构建和导出脚本

set -e

echo "=========================================="
echo "语义距离可视化工具 - Docker镜像打包"
echo "=========================================="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装，请先安装Docker"
    exit 1
fi

# 项目目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "1. 检查项目文件..."
if [ ! -f "app.py" ]; then
    echo "错误: 未找到app.py，请确保在项目根目录运行此脚本"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "警告: 未找到.venv虚拟环境，将重新安装依赖"
    echo "建议先运行: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi

echo ""
echo "2. 检查模型缓存..."
MODEL_CACHE="$HOME/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
if [ -d "$MODEL_CACHE" ]; then
    echo "✓ 找到模型缓存: $MODEL_CACHE"
    # 复制模型缓存到项目目录（确保包含在镜像中）
    echo "  复制模型缓存到项目目录..."
    mkdir -p .cache/huggingface/hub
    # 如果已存在，先删除再复制（确保是最新的）
    if [ -d ".cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2" ]; then
        rm -rf .cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2
    fi
    cp -r "$MODEL_CACHE" .cache/huggingface/hub/ 2>/dev/null || true
    if [ -d ".cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2" ]; then
        echo "  ✓ 模型缓存已复制到项目目录"
    else
        echo "  ⚠ 模型缓存复制失败，但会尝试在构建时使用系统缓存"
    fi
else
    echo "⚠ 未找到模型缓存，构建时可能需要下载（需要网络）"
fi

echo ""
echo "3. 构建Docker镜像..."
IMAGE_NAME="semantic-visualizer"
IMAGE_TAG="latest"

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

if [ $? -ne 0 ]; then
    echo "错误: Docker镜像构建失败"
    exit 1
fi

echo ""
echo "4. 查看镜像信息..."
docker images "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "5. 导出镜像..."
EXPORT_FILE="semantic-visualizer-$(date +%Y%m%d-%H%M%S).tar.gz"
echo "正在导出到: $EXPORT_FILE"
docker save "${IMAGE_NAME}:${IMAGE_TAG}" | gzip > "$EXPORT_FILE"

if [ $? -eq 0 ]; then
    FILE_SIZE=$(du -h "$EXPORT_FILE" | cut -f1)
    echo "✓ 导出成功!"
    echo "  文件: $EXPORT_FILE"
    echo "  大小: $FILE_SIZE"
    echo ""
    echo "=========================================="
    echo "下一步操作："
    echo "1. 将 $EXPORT_FILE 传输到新服务器"
    echo "2. 在新服务器上运行:"
    echo "   docker load < $EXPORT_FILE"
    echo "   docker run -d -p 5000:5000 --name semantic-visualizer semantic-visualizer:latest"
    echo "=========================================="
else
    echo "错误: 镜像导出失败"
    exit 1
fi

