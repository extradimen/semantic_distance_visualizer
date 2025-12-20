# Docker 部署指南

## 打包Docker镜像（在当前服务器）

### 1. 确保项目完整
```bash
cd /home/ubuntu/semantic_distance_visualizer

# 确保虚拟环境存在且依赖已安装
source .venv/bin/activate
pip install -r requirements.txt

# 确保模型已下载（首次运行会自动下载）
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

### 2. 构建Docker镜像
```bash
# 构建镜像（包含所有依赖和模型）
docker build -t semantic-visualizer:latest .

# 查看镜像大小
docker images semantic-visualizer
```

### 3. 导出镜像
```bash
# 导出镜像为tar文件（可以传输到其他服务器）
docker save semantic-visualizer:latest | gzip > semantic-visualizer.tar.gz

# 或者使用更小的压缩
docker save semantic-visualizer:latest -o semantic-visualizer.tar
```

## 在新服务器上部署

### 1. 传输镜像文件
```bash
# 使用scp传输（替换为实际IP和路径）
scp semantic-visualizer.tar.gz user@new-server:/path/to/destination/
```

### 2. 加载镜像
```bash
# 在新服务器上加载镜像
docker load < semantic-visualizer.tar.gz
# 或者
docker load -i semantic-visualizer.tar
```

### 3. 运行容器

#### 方式1：使用docker run
```bash
docker run -d \
  --name semantic-visualizer \
  -p 5000:5000 \
  --restart unless-stopped \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/uploads:/app/uploads \
  semantic-visualizer:latest
```

#### 方式2：使用docker-compose（推荐）
```bash
# 复制docker-compose.yml到新服务器
# 然后运行
docker-compose up -d
```

### 4. 验证运行
```bash
# 查看容器状态
docker ps | grep semantic-visualizer

# 查看日志
docker logs semantic-visualizer

# 测试访问
curl http://localhost:5000
```

## 注意事项

1. **镜像大小**：包含虚拟环境和模型，镜像可能较大（约8-10GB），传输需要时间
2. **端口**：默认使用5000端口，确保防火墙开放
3. **数据持久化**：results和uploads目录建议挂载到宿主机
4. **资源要求**：建议至少4GB内存

## 常用命令

```bash
# 停止容器
docker stop semantic-visualizer

# 启动容器
docker start semantic-visualizer

# 重启容器
docker restart semantic-visualizer

# 查看日志
docker logs -f semantic-visualizer

# 进入容器
docker exec -it semantic-visualizer bash

# 删除容器
docker rm -f semantic-visualizer

# 删除镜像
docker rmi semantic-visualizer:latest
```

