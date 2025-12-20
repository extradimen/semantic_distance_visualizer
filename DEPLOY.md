# 部署指南

## 在另一个服务器上部署

### 1. 克隆仓库

```bash
git clone https://github.com/extradimen/semantic_distance_visualizer.git
cd semantic_distance_visualizer
```

### 2. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 模型已包含在仓库中

**重要**：模型文件已经包含在 `.cache/huggingface/` 目录中，无需额外下载。

应用会自动从项目目录加载模型（通过 `app.py` 中的环境变量设置）。

### 4. 启动服务

#### 方式一：使用 systemd 服务（推荐）

```bash
# 安装并启动服务
./start_service.sh

# 查看服务状态
sudo systemctl status semantic-visualizer

# 查看日志
sudo journalctl -u semantic-visualizer -f
```

#### 方式二：直接运行

```bash
source .venv/bin/activate
python app.py
```

### 5. 访问应用

应用默认运行在 `http://localhost:5000`

如果需要外部访问，确保：
- 防火墙开放 5000 端口
- 或者使用反向代理（如 Nginx）

### 6. 停止服务

```bash
./stop_service.sh
```

## 注意事项

1. **模型文件**：模型文件（约 458MB）已包含在 Git 仓库中，克隆后即可使用
2. **Python 版本**：建议使用 Python 3.12 或更高版本
3. **系统依赖**：确保系统已安装中文字体（用于图表显示）
   ```bash
   sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
   ```
4. **内存要求**：建议至少 2GB 可用内存（模型加载需要约 400-500MB）

## 验证部署

1. 访问 `http://your-server-ip:5000`
2. 上传一个测试文件（CSV 或 Excel 格式，包含三列：分类、语言、词汇）
3. 检查是否能正常生成可视化结果

## 故障排查

### 模型加载失败
- 检查 `.cache/huggingface/hub/` 目录是否存在
- 检查 `app.py` 中的环境变量设置

### 中文显示问题
- 安装中文字体（见上方）
- 检查系统字体配置

### 服务无法启动
- 检查端口 5000 是否被占用：`sudo lsof -i :5000`
- 查看服务日志：`sudo journalctl -u semantic-visualizer -n 50`

