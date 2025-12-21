#!/bin/bash
# 停止语义可视化服务
# 用法: ./stop_service.sh [端口号]
# 例如: ./stop_service.sh 21304  # 停止端口 21304 的服务
#      ./stop_service.sh         # 停止默认端口 5000 的服务

# 获取端口参数，默认为 5000
PORT=${1:-5000}

# 验证端口号是否为有效数字
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "❌ 错误: 端口号必须是数字"
    echo "用法: ./stop_service.sh [端口号]"
    exit 1
fi

# 确定服务名称
SERVICE_NAME="semantic-visualizer"
if [ "$PORT" != "5000" ]; then
    SERVICE_NAME="${SERVICE_NAME}-${PORT}"
fi

echo "正在停止语义可视化服务 (端口: $PORT)..."

# 停止服务
sudo systemctl stop "${SERVICE_NAME}.service" 2>/dev/null

# 禁用服务（取消开机自启）
sudo systemctl disable "${SERVICE_NAME}.service" 2>/dev/null

# 检查服务是否已停止
if ! sudo systemctl is-active --quiet "${SERVICE_NAME}.service" 2>/dev/null; then
    echo "✅ 服务已成功停止！"
else
    echo "⚠️  警告: 服务可能仍在运行"
    echo "查看状态: sudo systemctl status $SERVICE_NAME"
fi
