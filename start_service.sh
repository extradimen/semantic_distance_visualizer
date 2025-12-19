#!/bin/bash
# 启动语义可视化服务

echo "正在启动语义可视化服务..."

# 复制服务文件到systemd目录
sudo cp semantic-visualizer.service /etc/systemd/system/

# 重新加载systemd配置
sudo systemctl daemon-reload

# 启用服务（开机自启）
sudo systemctl enable semantic-visualizer.service

# 启动服务
sudo systemctl start semantic-visualizer.service

# 检查服务状态
sudo systemctl status semantic-visualizer.service

echo ""
echo "服务已启动！"
echo "查看状态: sudo systemctl status semantic-visualizer"
echo "查看日志: sudo journalctl -u semantic-visualizer -f"
echo "停止服务: sudo systemctl stop semantic-visualizer"
echo "重启服务: sudo systemctl restart semantic-visualizer"

