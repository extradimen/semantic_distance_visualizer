#!/bin/bash
# 启动语义可视化服务
# 用法: ./start_service.sh [端口号]
# 例如: ./start_service.sh 21304  # 使用端口 21304
#      ./start_service.sh         # 使用默认端口 5000

# 获取端口参数，默认为 5000
PORT=${1:-5000}

# 验证端口号是否为有效数字
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "❌ 错误: 端口号必须是数字"
    echo "用法: ./start_service.sh [端口号]"
    exit 1
fi

# 验证端口号范围（1-65535）
if [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    echo "❌ 错误: 端口号必须在 1-65535 之间"
    exit 1
fi

echo "正在启动语义可视化服务..."
echo "使用端口: $PORT"

# 获取项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="semantic-visualizer"

# 获取当前用户（优先使用 SUDO_USER，如果没有则使用当前用户）
CURRENT_USER=${SUDO_USER:-$USER}
if [ -z "$CURRENT_USER" ]; then
    CURRENT_USER=$(whoami)
fi

# 创建临时服务文件，包含端口配置
TEMP_SERVICE_FILE="/tmp/${SERVICE_NAME}-${PORT}.service"
cat > "$TEMP_SERVICE_FILE" << EOF
[Unit]
Description=Semantic Distance Visualizer Web Application (Port $PORT)
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$SCRIPT_DIR/.venv/bin"
Environment="PORT=$PORT"
ExecStart=$SCRIPT_DIR/.venv/bin/python $SCRIPT_DIR/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 如果端口不是默认端口，使用带端口的服务名
if [ "$PORT" != "5000" ]; then
    SERVICE_NAME="${SERVICE_NAME}-${PORT}"
fi

# 复制服务文件到systemd目录
sudo cp "$TEMP_SERVICE_FILE" "/etc/systemd/system/${SERVICE_NAME}.service"

# 清理临时文件
rm -f "$TEMP_SERVICE_FILE"

# 重新加载systemd配置
sudo systemctl daemon-reload

# 启用服务（开机自启）
sudo systemctl enable "${SERVICE_NAME}.service"

# 启动服务
sudo systemctl start "${SERVICE_NAME}.service"

# 等待服务启动
sleep 2

# 检查服务状态
if sudo systemctl is-active --quiet "${SERVICE_NAME}.service"; then
    echo ""
    echo "✅ 服务已成功启动！"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "服务名称: $SERVICE_NAME"
    echo "运行端口: $PORT"
    echo "访问地址: http://localhost:$PORT"
    echo ""
    echo "常用命令:"
    echo "  查看状态: sudo systemctl status $SERVICE_NAME"
    echo "  查看日志: sudo journalctl -u $SERVICE_NAME -f"
    echo "  停止服务: sudo systemctl stop $SERVICE_NAME"
    echo "  重启服务: sudo systemctl restart $SERVICE_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 显示服务状态
    sudo systemctl status "${SERVICE_NAME}.service" --no-pager -l
else
    echo ""
    echo "❌ 服务启动失败！"
    echo "查看日志: sudo journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi
