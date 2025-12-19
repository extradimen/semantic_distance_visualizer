#!/bin/bash
# 停止语义可视化服务

echo "正在停止语义可视化服务..."

sudo systemctl stop semantic-visualizer.service
sudo systemctl disable semantic-visualizer.service

echo "服务已停止！"

