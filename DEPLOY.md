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

### 3. 模型加载

**模型加载策略**：
- **优先从 ModelScope 加载**：如果已安装 `modelscope`，应用会自动从 ModelScope（魔搭社区）下载模型，适合中国大陆用户，下载速度更快
- **自动回退**：如果 ModelScope 不可用或未安装，会自动回退到 Hugging Face 下载
- **本地缓存**：模型下载后会缓存在 `.cache/` 目录中，后续启动无需重新下载

**注意**：首次运行需要下载模型（约 470MB），请确保网络连接正常。

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

1. **模型下载**：
   - 首次运行时会自动下载模型（约 470MB）
   - 推荐安装 `modelscope` 以从 ModelScope 下载（中国大陆用户推荐）
   - 模型会缓存在 `.cache/` 目录中
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
- 检查网络连接是否正常
- 如果使用 ModelScope，检查是否已安装：`pip install modelscope`
- 检查 `.cache/modelscope/` 或 `.cache/huggingface/` 目录是否存在
- 查看应用日志了解具体错误信息

### 中文显示问题
- 安装中文字体（见上方）
- 检查系统字体配置

### 服务无法启动
- 检查端口 5000 是否被占用：`sudo lsof -i :5000`
- 查看服务日志：`sudo journalctl -u semantic-visualizer -n 50`

