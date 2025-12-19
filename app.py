from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import networkx as nx
from io import BytesIO
import base64
import pandas as pd
from datetime import datetime

# 配置中文字体
# 设置字体优先级列表，matplotlib会自动选择可用的字体
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei',  # 文泉驿微米黑
    'WenQuanYi Zen Hei',     # 文泉驿正黑
    'SimHei',                # 黑体
    'SimSun',                # 宋体
    'Microsoft YaHei',       # 微软雅黑
    'DejaVu Sans',           # 备用字体
    'Arial',
    'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 清除matplotlib字体缓存（如果需要）
try:
    import matplotlib.font_manager
    matplotlib.font_manager.fontManager.__init__()
except:
    pass

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# 加载预训练模型（使用多语言模型以支持中英文）
print("正在加载语义嵌入模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("模型加载完成！")

def parse_vocabulary(file_content):
    """解析上传的词汇文件，支持多种格式"""
    words = []
    lines = file_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # 支持多种格式：纯词汇、中英文对照、CSV等
        if ',' in line:
            parts = [p.strip() for p in line.split(',')]
            # 取第一个非空部分作为词汇
            word = parts[0] if parts[0] else (parts[1] if len(parts) > 1 else None)
            if word:
                words.append(word)
        elif '\t' in line:
            parts = [p.strip() for p in line.split('\t')]
            word = parts[0] if parts[0] else (parts[1] if len(parts) > 1 else None)
            if word:
                words.append(word)
        else:
            words.append(line)
    
    return words

def calculate_semantic_distances(words):
    """计算词汇间的语义距离"""
    print(f"正在计算 {len(words)} 个词汇的语义嵌入...")
    embeddings = model.encode(words, show_progress_bar=True)
    
    print("正在计算语义相似度矩阵...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # 转换为距离矩阵（距离 = 1 - 相似度）
    distance_matrix = 1 - similarity_matrix
    
    return embeddings, similarity_matrix, distance_matrix

def generate_heatmap(similarity_matrix, words, output_path):
    """生成语义相似度热力图"""
    # 增大图片尺寸，让每个图占满页面
    plt.figure(figsize=(20, 16))
    sns.heatmap(similarity_matrix, 
                xticklabels=words, 
                yticklabels=words,
                annot=True, 
                fmt='.2f',
                cmap='YlOrRd',
                cbar_kws={'label': '语义相似度'},
                square=True,
                annot_kws={'size': 8})
    plt.title('词汇语义相似度热力图', fontsize=20, pad=20, fontweight='bold')
    plt.xlabel('词汇', fontsize=14, fontweight='bold')
    plt.ylabel('词汇', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

def generate_network_graph(similarity_matrix, words, output_path, threshold=0.3):
    """生成语义网络图 - 学术专业风格"""
    G = nx.Graph()
    
    # 添加节点
    for word in words:
        G.add_node(word)
    
    # 添加边（只连接相似度高于阈值的词汇对）
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            similarity = similarity_matrix[i][j]
            if similarity > threshold:
                G.add_edge(words[i], words[j], weight=similarity)
    
    # 绘制网络图 - 学术专业风格，浅色背景
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    ax = fig.add_subplot(111, facecolor='white')
    
    # 使用力导向布局算法，优化参数
    pos = nx.spring_layout(G, k=2.5, iterations=150, seed=42)
    
    # 计算节点度中心性
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0
    
    # 绘制边 - 简洁专业风格
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # 使用简洁的边样式
    from matplotlib.collections import LineCollection
    
    edge_positions = []
    edge_alphas = []
    edge_widths = []
    
    for (u, v), weight in zip(edges, weights):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        edge_positions.append([(x1, y1), (x2, y2)])
        # 根据权重设置透明度和宽度
        edge_alphas.append(weight * 0.4 + 0.2)  # 透明度基于权重
        edge_widths.append(weight * 2.5 + 0.5)   # 宽度基于权重
    
    if edge_positions:
        # 使用统一的灰色，通过透明度区分权重
        edge_colors = [(0.4, 0.4, 0.4, alpha) for alpha in edge_alphas]
        lc = LineCollection(edge_positions, colors=edge_colors, linewidths=edge_widths, 
                           capstyle='round', alpha=0.6)
        ax.add_collection(lc)
    
    # 绘制节点 - 基于度中心性的大小和颜色
    node_sizes = []
    node_colors = []
    
    # 使用度中心性设置节点大小和颜色
    for node in G.nodes():
        degree = degrees.get(node, 0)
        # 节点大小：度越大，节点越大
        normalized_degree = (degree - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 0.5
        node_size = 300 + normalized_degree * 1200
        node_sizes.append(node_size)
        
        # 节点颜色：使用蓝色系，度越大颜色越深
        # 使用专业的蓝色渐变
        blue_intensity = 0.3 + normalized_degree * 0.5
        node_colors.append((0.2, 0.4, blue_intensity))
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, 
                           node_size=node_sizes, 
                           node_color=node_colors, 
                           alpha=0.8,
                           ax=ax,
                           edgecolors='#2c3e50',
                           linewidths=1.2)
    
    # 绘制标签 - 支持中文，清晰易读
    chinese_font = 'WenQuanYi Micro Hei'
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if 'WenQuanYi Micro Hei' not in available_fonts:
            if 'WenQuanYi Zen Hei' in available_fonts:
                chinese_font = 'WenQuanYi Zen Hei'
            else:
                chinese_font = 'sans-serif'
    except:
        chinese_font = 'sans-serif'
    
    # 绘制标签 - 简洁清晰
    for node, (x, y) in pos.items():
        ax.text(x, y, node, 
               fontsize=11, 
               fontfamily=chinese_font,
               ha='center', 
               va='center',
               color='#2c3e50',
               weight='500',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        alpha=0.85,
                        edgecolor='#bdc3c7',
                        linewidth=0.8))
    
    # 设置标题和样式
    ax.set_title(f'词汇语义网络图 (相似度阈值: {threshold})', 
                fontsize=18, 
                pad=20, 
                fontweight='600',
                color='#2c3e50', 
                family=chinese_font)
    
    # 设置坐标轴样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return output_path

def generate_mds_plot(embeddings, words, output_path):
    """使用MDS降维生成2D可视化"""
    from sklearn.manifold import MDS
    
    print("正在进行MDS降维...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    
    # 计算距离矩阵
    from sklearn.metrics.pairwise import euclidean_distances
    distance_matrix = euclidean_distances(embeddings)
    
    coords = mds.fit_transform(distance_matrix)
    
    # 增大图片尺寸
    plt.figure(figsize=(20, 16))
    plt.scatter(coords[:, 0], coords[:, 1], s=400, alpha=0.6, c=range(len(words)), cmap='viridis', edgecolors='black', linewidths=1)
    
    # 配置中文字体
    font_prop = fm.FontProperties()
    # 尝试使用系统中文字体
    chinese_font = 'WenQuanYi Micro Hei'
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if 'WenQuanYi Micro Hei' in available_fonts:
            font_prop.set_family('WenQuanYi Micro Hei')
        elif 'WenQuanYi Zen Hei' in available_fonts:
            font_prop.set_family('WenQuanYi Zen Hei')
        else:
            font_prop.set_family('sans-serif')
    except:
        font_prop.set_family('sans-serif')
    
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), 
                    fontsize=12, ha='center', va='center',
                    fontproperties=font_prop,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.title('词汇语义空间2D可视化 (MDS降维)', fontsize=20, pad=20, fontweight='bold')
    plt.xlabel('维度1', fontsize=14, fontweight='bold')
    plt.ylabel('维度2', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和生成可视化"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 读取文件内容
        file_content = file.read().decode('utf-8')
        words = parse_vocabulary(file_content)
        
        if len(words) < 2:
            return jsonify({'error': '至少需要2个词汇才能进行分析'}), 400
        
        if len(words) > 100:
            return jsonify({'error': '词汇数量不能超过100个'}), 400
        
        # 计算语义距离
        embeddings, similarity_matrix, distance_matrix = calculate_semantic_distances(words)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成可视化
        results = {}
        
        # 1. 热力图
        heatmap_path = os.path.join(app.config['RESULTS_FOLDER'], f'heatmap_{timestamp}.png')
        generate_heatmap(similarity_matrix, words, heatmap_path)
        results['heatmap'] = f'/results/heatmap_{timestamp}.png'
        
        # 2. 网络图
        network_path = os.path.join(app.config['RESULTS_FOLDER'], f'network_{timestamp}.png')
        threshold = float(request.form.get('threshold', 0.3))
        generate_network_graph(similarity_matrix, words, network_path, threshold)
        results['network'] = f'/results/network_{timestamp}.png'
        
        # 3. MDS 2D可视化
        mds_path = os.path.join(app.config['RESULTS_FOLDER'], f'mds_{timestamp}.png')
        generate_mds_plot(embeddings, words, mds_path)
        results['mds'] = f'/results/mds_{timestamp}.png'
        
        # 4. 生成相似度矩阵数据（JSON格式）
        similarity_data = {
            'words': words,
            'matrix': similarity_matrix.tolist(),
            'top_pairs': []
        }
        
        # 找出最相似的词汇对
        top_pairs = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                top_pairs.append({
                    'word1': words[i],
                    'word2': words[j],
                    'similarity': float(similarity_matrix[i][j])
                })
        top_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        similarity_data['top_pairs'] = top_pairs[:20]  # 前20对
        
        results['similarity_data'] = similarity_data
        results['word_count'] = len(words)
        
        return jsonify(results)
    
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

@app.route('/results/<filename>')
def get_result(filename):
    """返回生成的结果图片"""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

