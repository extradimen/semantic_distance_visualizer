# 词汇语义距离可视化工具

一个基于AI嵌入向量的跨文化语义分析Web平台，可以上传词汇列表，自动计算语义距离并生成多种可视化结果。

## 功能特性

- 📤 **文件上传**：支持多种格式的词汇文件（纯文本、CSV、Excel，三列格式：分类/语言/词汇）
- 🤖 **AI语义分析**：使用多语言Sentence Transformers模型计算语义嵌入
  - 优先从 ModelScope（魔搭社区）加载模型，适合中国大陆用户
  - 自动回退到 Hugging Face（如果 ModelScope 不可用）
- 📊 **多种可视化**：
  - 语义相似度热力图
  - 语义网络图（交互式，支持动态阈值调整）
  - 语义空间2D可视化（MDS降维）
- 🔗 **相似度分析**：自动找出最相似的词汇对
- 🌐 **跨语言支持**：支持中英文混合分析

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/extradimen/semantic_distance_visualizer.git
cd semantic_distance_visualizer
```

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate
```

### 3. 安装Python依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

### 4. 运行应用

#### 方式一：后台服务运行（推荐）

使用 systemd 服务，关闭终端后仍可运行，支持自定义端口：

```bash
# 启动服务（使用默认端口 5000）
./start_service.sh

# 启动服务（使用自定义端口，例如 21304）
./start_service.sh 21304

# 查看服务状态（默认端口）
sudo systemctl status semantic-visualizer

# 查看服务状态（自定义端口）
sudo systemctl status semantic-visualizer-21304

# 查看日志
sudo journalctl -u semantic-visualizer -f
# 或自定义端口
sudo journalctl -u semantic-visualizer-21304 -f

# 停止服务（默认端口）
./stop_service.sh

# 停止服务（自定义端口）
./stop_service.sh 21304

# 重启服务
sudo systemctl restart semantic-visualizer
# 或自定义端口
sudo systemctl restart semantic-visualizer-21304
```

**注意**：
- 服务会在后台运行，即使关闭终端也不会中断
- 服务会在系统重启后自动启动（已启用开机自启）
- 不同端口会创建不同的服务实例，可以同时运行多个实例

#### 方式二：前台运行（测试用）

```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 运行应用
python app.py
```

应用将在 `http://0.0.0.0:5000` 启动。

### 5. 访问应用

在浏览器中打开：`http://localhost:5000` 或 `http://你的服务器IP:5000`

## 文件格式

支持以下格式的词汇文件：

### 格式1：纯词汇（每行一个）
```
Trust
Commitment
Partnership
Negotiation
Cooperation
```

### 格式2：中英文对照（逗号分隔）
```
Trust,信任
Commitment,承诺
Partnership,伙伴关系
```

### 格式3：CSV/Excel格式（三列：分类、语言、词汇）

**CSV格式：**
```csv
category,language,word
情感,中文,信任
情感,中文,承诺
关系,英文,Partnership
```

**Excel格式：**
- 第一列：分类（category）
- 第二列：语言（language）
- 第三列：词汇（word）

系统会自动识别列名，支持中英文列名。

## 使用说明

1. **上传文件**：点击上传区域或拖拽文件到指定区域
2. **设置阈值**：调整网络图的相似度阈值（默认0.3），实时查看不同阈值下的网络结构
3. **开始分析**：点击"开始分析"按钮
4. **查看结果**：
   - 热力图：按（分类，语言）分组显示相似度
   - 网络图：交互式网络，节点大小表示连接数（度中心性），颜色表示分类
   - 散点图：2D语义空间可视化，颜色表示分类

## 模型加载

**模型加载策略**：
- **优先从 ModelScope 加载**：如果已安装 `modelscope`，应用会自动从 ModelScope（魔搭社区）下载模型，适合中国大陆用户，下载速度更快
- **自动回退**：如果 ModelScope 不可用或未安装，会自动回退到 Hugging Face 下载
- **本地缓存**：模型下载后会缓存在项目目录的 `.cache/` 文件夹中，后续启动无需重新下载

**注意**：首次运行需要下载模型（约 470MB），请确保网络连接正常。

## 项目结构

```
semantic_distance_visualizer/
├── app.py                 # Flask主应用
├── requirements.txt       # Python依赖
├── README.md             # 项目说明
├── start_service.sh      # 启动服务脚本（动态生成systemd服务配置）
├── stop_service.sh       # 停止服务脚本
├── templates/
│   └── index.html        # 前端页面
├── static/
│   ├── css/
│   │   └── style.css     # 样式文件
│   └── js/
│       └── main.js       # 前端脚本
├── uploads/              # 上传文件临时目录
└── results/              # 生成的可视化结果
```

## 技术栈

- **后端**：Flask
- **AI模型**：sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
  - 模型来源：优先从 [ModelScope](https://www.modelscope.cn/models/extradimen/paraphrase-multilingual-MiniLM-L12-v2) 加载
  - 备用来源：Hugging Face（如果 ModelScope 不可用）
- **可视化**：matplotlib, seaborn, networkx, vis-network
- **前端**：HTML5, CSS3, JavaScript

## 部署注意事项

1. **模型下载**：
   - 首次运行时会自动下载预训练模型（约470MB）
   - 如果已安装 `modelscope`，将从 ModelScope 下载（推荐，适合中国大陆用户）
   - 如果未安装 `modelscope` 或下载失败，将自动回退到 Hugging Face
   - 模型会缓存在项目目录的 `.cache/` 文件夹中

2. **Python 版本**：建议使用 Python 3.12 或更高版本

3. **系统依赖**：确保系统已安装中文字体（用于图表显示）
   ```bash
   sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
   ```

4. **内存要求**：建议至少 2GB 可用内存（模型加载需要约 400-500MB）

5. **端口配置**：
   - 应用默认运行在 5000 端口
   - 如果需要外部访问，确保防火墙开放 5000 端口
   - 或使用反向代理（如 Nginx）

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
- 检查虚拟环境是否正确配置

### 文件上传失败
- 检查文件格式是否正确（CSV/Excel，三列格式）
- 检查文件大小是否超过限制（默认16MB）
- 查看浏览器控制台错误信息

## 使用建议

- 建议词汇数量在2-100个之间
- 分析大量词汇可能需要较长时间，请耐心等待
- 热力图按（分类，语言）分组计算相似度
- 网络图和散点图按分类着色
- 网络图节点大小表示连接数（度中心性），越大表示该词汇在语义网络中越重要

## 许可证

MIT License
