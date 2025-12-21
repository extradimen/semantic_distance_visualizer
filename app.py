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
import pickle
import pandas as pd
from datetime import datetime

# 配置中文字体
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# 设置模型缓存目录为项目目录
project_root = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(project_root, '.cache')
os.makedirs(cache_dir, exist_ok=True)

# ModelScope 模型配置
modelscope_model_id = 'extradimen/paraphrase-multilingual-MiniLM-L12-v2'
modelscope_cache_dir = os.path.join(cache_dir, 'modelscope', 'hub')

# 加载预训练模型（优先从 ModelScope 加载，适用于中国大陆用户）
print("正在加载语义嵌入模型...")
try:
    # 尝试从 ModelScope 加载
    try:
        from modelscope import snapshot_download
        print(f"正在从 ModelScope 下载模型: {modelscope_model_id}")
        model_dir = snapshot_download(modelscope_model_id, cache_dir=modelscope_cache_dir)
        print(f"模型已下载到: {model_dir}")
        # 从本地路径加载模型
        model = SentenceTransformer(model_dir)
        print("✅ 模型已从 ModelScope 加载完成！")
    except ImportError:
        print("⚠️  ModelScope SDK 未安装，尝试从 Hugging Face 加载...")
        print("   提示: 如需从 ModelScope 加载，请运行: pip install modelscope")
        # 回退到 Hugging Face
        hf_cache_dir = os.path.join(cache_dir, 'huggingface')
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = hf_cache_dir
        os.environ['HF_HUB_CACHE'] = os.path.join(hf_cache_dir, 'hub')
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ 模型已从 Hugging Face 加载完成！")
    except Exception as e:
        print(f"⚠️  从 ModelScope 加载失败: {e}")
        print("   回退到 Hugging Face...")
        # 回退到 Hugging Face
        hf_cache_dir = os.path.join(cache_dir, 'huggingface')
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = hf_cache_dir
        os.environ['HF_HUB_CACHE'] = os.path.join(hf_cache_dir, 'hub')
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ 模型已从 Hugging Face 加载完成！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    raise

def parse_vocabulary_file(file_path, filename):
    """解析上传的词汇文件，支持CSV和Excel格式
    返回: list of dict, 每个dict包含 {'category': 分类, 'language': 语言, 'word': 词汇}
    """
    data = []
    
    # 根据文件扩展名选择解析方式
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        # Excel文件
        try:
            df = pd.read_excel(file_path, header=None)
        except Exception as e:
            raise ValueError(f"无法读取Excel文件: {str(e)}")
    elif filename.endswith('.csv'):
        # CSV文件
        try:
            df = pd.read_csv(file_path, header=None, encoding='utf-8')
        except UnicodeDecodeError:
            # 尝试其他编码
            df = pd.read_csv(file_path, header=None, encoding='gbk')
    else:
        raise ValueError("不支持的文件格式，请上传CSV或Excel文件")
    
    # 检查列数
    if df.shape[1] < 3:
        raise ValueError("文件必须包含至少3列：分类、语言、词汇")
    
    # 解析数据
    for idx, row in df.iterrows():
        category = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''
        language = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ''
        word = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ''
        
        # 跳过空行
        if not category and not language and not word:
            continue
        
        # 验证数据
        if not word:
            continue
        
        data.append({
            'category': category,
            'language': language,
            'word': word
        })
    
    if len(data) == 0:
        raise ValueError("文件中没有有效数据")
    
    return data

def calculate_semantic_distances(data_list):
    """计算词汇间的语义距离
    返回: embeddings_dict (词汇 -> 嵌入向量)
    """
    words = [item['word'] for item in data_list]
    print(f"正在计算 {len(words)} 个词汇的语义嵌入...")
    embeddings = model.encode(words, show_progress_bar=True)
    
    # 创建词汇到嵌入向量的字典
    embeddings_dict = {item['word']: emb for item, emb in zip(data_list, embeddings)}
    
    return embeddings_dict

def generate_heatmap(data_list, embeddings_dict, output_path):
    """生成语义相似度热力图 - 只计算同分类同语言的词汇间相似度，多行显示"""
    # 按分类和语言分组
    groups = {}
    for item in data_list:
        key = (item['category'], item['language'])
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    
    # 为每个分组生成热力图
    num_groups = len(groups)
    if num_groups == 0:
        raise ValueError("没有有效的数据分组")
    
    # 如果只有一个分组，直接生成单个热力图
    if num_groups == 1:
        group_key = list(groups.keys())[0]
        group_items = groups[group_key]
        words = [item['word'] for item in group_items]
        
        # 计算相似度矩阵
        embeddings = [embeddings_dict[item['word']] for item in group_items]
        similarity_matrix = cosine_similarity(embeddings)
        
        plt.figure(figsize=(max(20, len(words) * 0.8), max(16, len(words) * 0.7)))
        sns.heatmap(similarity_matrix, 
                    xticklabels=words, 
                    yticklabels=words,
                    annot=True, 
                    fmt='.2f',
                    cmap='YlOrRd',
                    cbar_kws={'label': '语义相似度'},
                    square=True,
                    annot_kws={'size': 8})
        plt.title(f'词汇语义相似度热力图 - {group_key[0]} ({group_key[1]})', 
                 fontsize=20, pad=20, fontweight='bold')
    else:
        # 多个分组，生成子图 - 纵向排列（多行）
        fig, axes = plt.subplots(num_groups, 1, figsize=(20, 16 * num_groups))
        if num_groups == 1:
            axes = [axes]
        
        for idx, (group_key, group_items) in enumerate(groups.items()):
            words = [item['word'] for item in group_items]
            embeddings = [embeddings_dict[item['word']] for item in group_items]
            similarity_matrix = cosine_similarity(embeddings)
            
            ax = axes[idx]
            sns.heatmap(similarity_matrix, 
                        xticklabels=words, 
                        yticklabels=words,
                        annot=True, 
                        fmt='.2f',
                        cmap='YlOrRd',
                        cbar_kws={'label': '语义相似度'},
                        square=True,
                        annot_kws={'size': 8},
                        ax=ax)
            ax.set_title(f'{group_key[0]} ({group_key[1]})', fontsize=18, fontweight='bold', pad=15)
            ax.set_xlabel('词汇', fontsize=14)
            ax.set_ylabel('词汇', fontsize=14)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        plt.suptitle('词汇语义相似度热力图（按分类和语言分组）', 
                    fontsize=22, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

def generate_network_graph(data_list, embeddings_dict, output_path, threshold=0.3):
    """生成语义网络图 - 按分类着色，节点大小基于度中心性"""
    G = nx.Graph()
    
    # 获取所有词汇
    words = [item['word'] for item in data_list]
    categories = [item['category'] for item in data_list]
    
    # 创建词汇到分类的映射
    word_to_category = {item['word']: item['category'] for item in data_list}
    
    # 获取所有唯一分类
    unique_categories = list(set(categories))
    
    # 添加节点
    for word in words:
        G.add_node(word)
    
    # 添加边（只连接相似度高于阈值的词汇对）
    embeddings = [embeddings_dict[word] for word in words]
    similarity_matrix = cosine_similarity(embeddings)
    
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
    
    # 绘制节点 - 按分类着色，大小基于度中心性
    node_sizes_ordered = []
    node_colors_ordered = []
    
    # 为每个分类分配颜色
    import matplotlib.cm as cm
    category_colors = {}
    if len(unique_categories) == 1:
        # 只有一个分类，使用单一颜色
        category_colors[unique_categories[0]] = (0.2, 0.4, 0.8)
    else:
        # 多个分类，使用不同颜色
        colors = cm.Set3(np.linspace(0, 1, len(unique_categories)))
        for idx, cat in enumerate(unique_categories):
            category_colors[cat] = colors[idx][:3]  # RGB only
    
    # 使用度中心性设置节点大小，按分类设置颜色
    for node in G.nodes():
        degree = degrees.get(node, 0)
        # 节点大小：度越大，节点越大
        normalized_degree = (degree - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 0.5
        node_size = 500 + normalized_degree * 2000  # 最小500，最大2500
        node_sizes_ordered.append(node_size)
        
        # 节点颜色：按分类
        category = word_to_category.get(node, unique_categories[0])
        node_colors_ordered.append(category_colors.get(category, (0.2, 0.4, 0.8)))
    
    # 绘制节点
    nodes = list(G.nodes())
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=nodes,
                           node_size=node_sizes_ordered, 
                           node_color=node_colors_ordered, 
                           alpha=0.8,
                           ax=ax,
                           edgecolors='#2c3e50',
                           linewidths=1.5)
    
    # 绘制标签 - 文字在节点内部，透明背景，颜色与节点匹配，大小动态适配
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
    
    # 计算文字大小和颜色的映射
    for idx, (node, (x, y)) in enumerate(pos.items()):
        degree = degrees.get(node, 0)
        node_size = node_sizes_ordered[idx]
        node_color = node_colors_ordered[idx]
        
        # 根据节点大小和文字长度动态计算字体大小
        base_font_size = 8 + (node_size / 2500) * 12  # 基础字体大小：8-20
        text_length = len(node)
        # 长文字适当减小字体
        if text_length > 8:
            font_size = base_font_size * (8 / text_length) * 0.9
        elif text_length > 5:
            font_size = base_font_size * 0.95
        else:
            font_size = base_font_size
        
        # 确保字体大小在合理范围内
        font_size = max(7, min(18, font_size))
        
        # 根据节点颜色计算文字颜色（确保对比度）
        r, g, b = node_color
        brightness = (r * 0.299 + g * 0.587 + b * 0.114)
        if brightness > 0.5:
            text_color = '#2c3e50'  # 深色文字
        else:
            text_color = '#ffffff'  # 浅色文字
        
        # 绘制文字，透明背景，在节点内部
        ax.text(x, y, node, 
               fontsize=font_size, 
               fontfamily=chinese_font,
               ha='center', 
               va='center',
               color=text_color,
               weight='600',
               # 不使用bbox，让文字直接在节点上
               zorder=10)  # 确保文字在节点上方
    
    # 添加图例
    if len(unique_categories) > 1:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=category_colors[cat], label=cat) 
                          for cat in unique_categories]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
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

def generate_mds_plot(data_list, embeddings_dict, output_path):
    """使用MDS降维生成2D可视化 - 按分类着色"""
    from sklearn.manifold import MDS
    
    words = [item['word'] for item in data_list]
    categories = [item['category'] for item in data_list]
    unique_categories = list(set(categories))
    
    # 获取嵌入向量
    embeddings = [embeddings_dict[word] for word in words]
    
    print("正在进行MDS降维...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    
    # 计算距离矩阵
    from sklearn.metrics.pairwise import euclidean_distances
    distance_matrix = euclidean_distances(embeddings)
    
    coords = mds.fit_transform(distance_matrix)
    
    # 为每个分类分配颜色
    import matplotlib.cm as cm
    category_colors = {}
    if len(unique_categories) == 1:
        category_colors[unique_categories[0]] = (0.2, 0.4, 0.8)
    else:
        colors = cm.Set3(np.linspace(0, 1, len(unique_categories)))
        for idx, cat in enumerate(unique_categories):
            category_colors[cat] = colors[idx][:3]
    
    # 按分类绘制散点
    plt.figure(figsize=(20, 16))
    for category in unique_categories:
        indices = [i for i, item in enumerate(data_list) if item['category'] == category]
        if indices:
            category_coords = coords[indices]
            category_words = [words[i] for i in indices]
            color = category_colors[category]
            plt.scatter(category_coords[:, 0], category_coords[:, 1], 
                       s=400, alpha=0.6, c=[color], 
                       edgecolors='black', linewidths=1, label=category)
    
    # 配置中文字体
    font_prop = fm.FontProperties()
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
    
    # 添加标签
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), 
                    fontsize=12, ha='center', va='center',
                    fontproperties=font_prop,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 添加图例
    if len(unique_categories) > 1:
        plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
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
        
        # 检查文件格式
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            return jsonify({'error': '只支持CSV和Excel文件（.csv, .xlsx, .xls）'}), 400
        
        # 保存临时文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f'temp_{timestamp}_{file.filename}'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)
        
        try:
            # 解析文件
            data_list = parse_vocabulary_file(temp_path, file.filename)
            
            if len(data_list) < 2:
                return jsonify({'error': '至少需要2个词汇才能进行分析'}), 400
            
            if len(data_list) > 200:
                return jsonify({'error': '词汇数量不能超过200个'}), 400
            
            # 计算语义嵌入
            embeddings_dict = calculate_semantic_distances(data_list)
            
            # 生成时间戳
            result_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 生成可视化
            results = {}
            
            # 1. 热力图（按分类和语言分组）
            heatmap_path = os.path.join(app.config['RESULTS_FOLDER'], f'heatmap_{result_timestamp}.png')
            generate_heatmap(data_list, embeddings_dict, heatmap_path)
            results['heatmap'] = f'/results/heatmap_{result_timestamp}.png'
            
            # 2. 网络图（按分类着色）
            network_path = os.path.join(app.config['RESULTS_FOLDER'], f'network_{result_timestamp}.png')
            threshold = float(request.form.get('threshold', 0.3))
            generate_network_graph(data_list, embeddings_dict, network_path, threshold)
            results['network'] = f'/results/network_{result_timestamp}.png'
            
            # 3. MDS 2D可视化（按分类着色）
            mds_path = os.path.join(app.config['RESULTS_FOLDER'], f'mds_{result_timestamp}.png')
            generate_mds_plot(data_list, embeddings_dict, mds_path)
            results['mds'] = f'/results/mds_{result_timestamp}.png'
            
            # 4. 生成相似度数据（所有词汇对）
            words = [item['word'] for item in data_list]
            embeddings = [embeddings_dict[word] for word in words]
            similarity_matrix = cosine_similarity(embeddings)
            
            similarity_data = {
                'words': words,
                'categories': [item['category'] for item in data_list],
                'languages': [item['language'] for item in data_list],
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
                        'category1': data_list[i]['category'],
                        'category2': data_list[j]['category'],
                        'similarity': float(similarity_matrix[i][j])
                    })
            top_pairs.sort(key=lambda x: x['similarity'], reverse=True)
            similarity_data['top_pairs'] = top_pairs[:20]  # 前20对
            
            results['similarity_data'] = similarity_data
            results['word_count'] = len(data_list)
            
            # 保存数据用于动态网络图生成
            try:
                network_data = {
                    'data_list': data_list,
                    'embeddings_dict': {k: v.tolist() for k, v in embeddings_dict.items()}
                }
                # 将数据序列化并编码
                pickled_data = pickle.dumps(network_data)
                encoded_data = base64.b64encode(pickled_data).decode('utf-8')
                results['network_data'] = encoded_data  # 前端可以存储这个用于动态生成
                print(f"网络数据已生成，大小: {len(encoded_data)} 字符")
            except Exception as e:
                print(f"生成网络数据失败: {str(e)}")
                import traceback
                traceback.print_exc()
                # 不中断流程，只是没有交互式网络图
            
            return jsonify(results)
        
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

@app.route('/network_data', methods=['POST'])
def get_network_data():
    """获取网络图数据（用于交互式可视化）"""
    try:
        # 检查请求内容类型
        if not request.is_json:
            print("错误: 请求不是JSON格式")
            return jsonify({'error': '请求必须是JSON格式'}), 400
        
        data = request.get_json()
        if not data:
            print("错误: 无法解析JSON数据")
            return jsonify({'error': '无法解析请求数据'}), 400
        
        threshold = float(data.get('threshold', 0.3))
        encoded_data = data.get('network_data')
        
        if not encoded_data:
            print("错误: 缺少network_data字段")
            return jsonify({'error': '缺少网络数据'}), 400
        
        # 解码数据
        import numpy as np
        
        try:
            pickled_data = base64.b64decode(encoded_data.encode('utf-8'))
            network_data = pickle.loads(pickled_data)
        except Exception as e:
            print(f"解码错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'数据解码失败: {str(e)}'}), 400
        
        # 恢复数据
        data_list = network_data['data_list']
        embeddings_dict = {k: np.array(v) for k, v in network_data['embeddings_dict'].items()}
        
        # 构建网络图
        words = [item['word'] for item in data_list]
        categories = [item['category'] for item in data_list]
        unique_categories = list(set(categories))
        
        # 创建词汇到分类的映射
        word_to_category = {item['word']: item['category'] for item in data_list}
        
        # 计算相似度矩阵
        embeddings = [embeddings_dict[word] for word in words]
        similarity_matrix = cosine_similarity(embeddings)
        
        # 构建节点数据
        nodes = []
        degrees = {}
        for i, word in enumerate(words):
            degree = sum(1 for j in range(len(words)) if i != j and similarity_matrix[i][j] > threshold)
            degrees[word] = degree
            category = word_to_category.get(word, unique_categories[0])
            nodes.append({
                'id': i,
                'label': word,
                'value': degree + 1,  # 节点大小
                'group': unique_categories.index(category),  # 分类组
                'category': category,
                'title': f'{word}\n分类: {category}\n连接数: {degree}'
            })
        
        # 构建边数据
        edges = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                similarity = similarity_matrix[i][j]
                if similarity > threshold:
                    edges.append({
                        'from': i,
                        'to': j,
                        'value': similarity,  # 边的权重
                        'width': similarity * 3,  # 边的宽度
                        'title': f'相似度: {similarity:.3f}'
                    })
        
        # 为每个分类分配颜色
        import matplotlib.cm as cm
        category_colors = {}
        if len(unique_categories) == 1:
            category_colors[unique_categories[0]] = '#3b82f6'
        else:
            colors = cm.Set3(np.linspace(0, 1, len(unique_categories)))
            for idx, cat in enumerate(unique_categories):
                # 转换为十六进制颜色
                r, g, b = [int(c * 255) for c in colors[idx][:3]]
                category_colors[cat] = f'#{r:02x}{g:02x}{b:02x}'
        
        # 为节点添加颜色
        for node in nodes:
            node['color'] = category_colors.get(node['category'], '#3b82f6')
        
        return jsonify({
            'nodes': nodes,
            'edges': edges,
            'categories': unique_categories,
            'category_colors': {cat: category_colors[cat] for cat in unique_categories}
        })
        
    except Exception as e:
        print(f"网络数据生成错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'生成失败: {str(e)}'}), 500

@app.route('/results/<filename>')
def get_result(filename):
    """返回生成的结果图片"""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
