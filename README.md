# 词汇语义距离可视化工具

一个基于AI嵌入向量的跨文化语义分析Web平台，可以上传词汇列表，自动计算语义距离并生成多种可视化结果。

## 功能特性

- 📤 **文件上传**：支持多种格式的词汇文件（纯文本、CSV、中英文对照）
- 🤖 **AI语义分析**：使用多语言Sentence Transformers模型计算语义嵌入
- 📊 **多种可视化**：
  - 语义相似度热力图
  - 语义网络图
  - 语义空间2D可视化（MDS降维）
- 🔗 **相似度分析**：自动找出最相似的词汇对
- 🌐 **跨语言支持**：支持中英文混合分析

## 安装步骤

### 1. 安装Python依赖

```bash
cd /home/ubuntu/semantic_distance_visualizer
pip install -r requirements.txt
```

### 2. 运行应用

#### 方式一：后台服务运行（推荐）

使用 systemd 服务，关闭终端后仍可运行：

```bash
# 启动服务
./start_service.sh

# 查看服务状态
sudo systemctl status semantic-visualizer

# 查看日志
sudo journalctl -u semantic-visualizer -f

# 停止服务
./stop_service.sh

# 重启服务
sudo systemctl restart semantic-visualizer
```

#### 方式二：前台运行（测试用）

```bash
source .venv/bin/activate
python app.py
```

应用将在 `http://0.0.0.0:5000` 启动。

### 3. 访问网站

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

### 格式3：CSV格式（多列）
```
Trust,信任,对合作伙伴意图与行为的正向期待
Commitment,承诺,双方在合作中投入资源与维持关系的意愿
```

系统会自动识别第一列作为词汇。

## 使用说明

1. **上传文件**：点击上传区域或拖拽文件到指定区域
2. **设置阈值**：调整网络图的相似度阈值（默认0.3）
3. **开始分析**：点击"开始分析"按钮
4. **查看结果**：等待分析完成后查看可视化结果和相似度表格

## 项目结构

```
semantic_distance_visualizer/
├── app.py                 # Flask主应用
├── requirements.txt       # Python依赖
├── README.md             # 项目说明
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
- **可视化**：matplotlib, seaborn, networkx
- **前端**：HTML5, CSS3, JavaScript

## 注意事项

- 首次运行时会自动下载预训练模型（约420MB），请确保网络连接正常
- 建议词汇数量在2-100个之间
- 分析大量词汇可能需要较长时间，请耐心等待

## 许可证

MIT License

