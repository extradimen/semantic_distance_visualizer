from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import re
import random
from collections import defaultdict
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

# 配置中文字体 - 优先使用系统可用的中文字体
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',      # Noto Sans 中文字体（优先）
    'Noto Sans CJK JP',      # Noto Sans 日文字体（也支持中文）
    'WenQuanYi Micro Hei',   # 文泉驿微米黑
    'WenQuanYi Zen Hei',     # 文泉驿正黑
    'SimHei',                # 黑体
    'SimSun',                # 宋体
    'Microsoft YaHei',       # 微软雅黑
    'Source Han Sans SC',    # 思源黑体
    'DejaVu Sans',           # 备用字体
    'Arial',
    'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_matplotlib_cjk_font_family():
    """选择能渲染中文等 CJK 的字体名，供热力图/坐标轴等与 seaborn 联用。

    仅从已知 CJK 字体候选里解析，不把 rcParams 里的 DejaVu 等当作中文回退，
    否则会出现「方框 / missing glyph」。
    """
    try:
        available = {f.name for f in fm.fontManager.ttflist}
        for name in (
            'Noto Sans CJK SC',
            'Noto Sans CJK JP',
            'Noto Sans CJK HK',
            'Noto Sans CJK KR',
            'Source Han Sans SC',
            'Source Han Sans CN',
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Droid Sans Fallback',
            'SimHei',
            'SimSun',
            'Microsoft YaHei',
        ):
            if name in available:
                return name
    except Exception:
        pass
    return 'sans-serif'


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
# 与权重匹配的官方分词器（HF Hub）；用于修复部分 ModelScope 快照中 tokenizer 与权重不匹配的问题
HF_PARAPHRASE_MINILM_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def _latin_text_tokenizer_collapsed(st_model):
    """若不同英文词得到完全相同的 token id 序列，则英文会全部走 <unk>，向量塌缩为同一向量。"""
    tok = getattr(st_model, 'tokenizer', None)
    if tok is None:
        return False
    try:
        a = tok.encode('Trade', add_special_tokens=True)
        b = tok.encode('Advantage', add_special_tokens=True)
        return a == b
    except Exception:
        return False


def repair_paraphrase_multilingual_minilm_tokenizer(st_model, weights_dir=None):
    """ModelScope 部分快照中 tokenizer 与权重不匹配：英文全部变为 <unk>，向量塌缩，相似度恒≈1。

    做法：从 Hugging Face 拉取（或使用本地 HF 缓存）官方 tokenizer 文件，写入权重目录并
    **重新加载** SentenceTransformer（ST 5.x 不允许直接替换 tokenizer 属性）。
    weights_dir：SentenceTransformer 加载的本地目录；若为 None 则无法覆盖文件，仅告警。
    """
    if not _latin_text_tokenizer_collapsed(st_model):
        return st_model
    from transformers import AutoTokenizer

    print(
        '⚠️ 检测到本地 tokenizer 将英文误分为 <unk>（Trade 与 Advantage 的 token 序列相同），'
        '相似度会恒为 ~1。正在用 Hugging Face 官方 tokenizer 覆盖配置并重新加载模型…'
    )
    try:
        tok = AutoTokenizer.from_pretrained(HF_PARAPHRASE_MINILM_ID)
        if weights_dir and os.path.isdir(weights_dir):
            tj = os.path.join(weights_dir, 'tokenizer.json')
            if os.path.isfile(tj):
                bak = os.path.join(weights_dir, 'tokenizer.json.bak_before_hf_tokenizer')
                if not os.path.isfile(bak):
                    import shutil

                    shutil.copy2(tj, bak)
            tok.save_pretrained(weights_dir)
            st2 = SentenceTransformer(weights_dir)
            if _latin_text_tokenizer_collapsed(st2):
                print('⚠️ 覆盖 tokenizer 后英文仍塌缩，请检查 HF 缓存是否完整。')
            else:
                print('✅ tokenizer 已修复并重新加载，英文分词正常。')
            return st2
        print('⚠️ 未提供权重目录，无法写入 tokenizer 文件；请改用 Hugging Face 加载或指定本地路径。')
    except Exception as e:
        print(f'⚠️ 修复 tokenizer 失败（离线时请预先缓存 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2）: {e}')
    return st_model


# 加载预训练模型（优先从 ModelScope 加载，适用于中国大陆用户）
print("正在加载语义嵌入模型...")
try:
    # 若此前已缓存完整模型，离线环境直接使用本地目录，避免 snapshot_download / HF 联网失败
    _local_ms = os.path.join(
        modelscope_cache_dir,
        modelscope_model_id.replace("/", os.sep),
    )
    _st_weights_dir = None
    if os.path.isfile(os.path.join(_local_ms, "modules.json")):
        print(f"使用本地已缓存模型: {_local_ms}")
        model = SentenceTransformer(_local_ms)
        _st_weights_dir = _local_ms
        print("✅ 模型已从本地 ModelScope 缓存加载完成！")
    else:
        # 尝试从 ModelScope 加载
        try:
            from modelscope import snapshot_download
            print(f"正在从 ModelScope 下载模型: {modelscope_model_id}")
            model_dir = snapshot_download(modelscope_model_id, cache_dir=modelscope_cache_dir)
            print(f"模型已下载到: {model_dir}")
            # 从本地路径加载模型
            model = SentenceTransformer(model_dir)
            _st_weights_dir = model_dir
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

    model = repair_paraphrase_multilingual_minilm_tokenizer(model, _st_weights_dir)
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    raise

# ---------------------------- 论文实证分析辅助函数 ----------------------------

def _norm_col_name(c):
    return re.sub(r'\s+', '', str(c).strip().lower())


def canonical_language(lang):
    """将列名归一为 Chinese / English / Other（用于 Language Pair 汇总）。"""
    s = _norm_col_name(lang)
    if 'chinese' in s or s in ('中文', '汉语'):
        return 'Chinese'
    if 'english' in s or s == '英文':
        return 'English'
    return str(lang).strip() or 'Other'


def canonical_domain(category):
    """将 Class/Domain 粗分为 Trade / Intercultural；其余保留原标签便于表格展示。"""
    s = str(category).strip().lower()
    if not s:
        return 'Unknown'
    if 'trade' in s:
        return 'Trade'
    if 'intercultural' in s or 'crossculture' in s.replace(' ', '') or 'cross-culture' in s:
        return 'Intercultural'
    return str(category).strip()


def language_pair_label(lang1, lang2):
    c1, c2 = canonical_language(lang1), canonical_language(lang2)
    a, b = sorted([c1, c2])
    if a == 'Chinese' and b == 'Chinese':
        return 'Chinese–Chinese'
    if a == 'English' and b == 'English':
        return 'English–English'
    if {c1, c2} == {'Chinese', 'English'}:
        return 'Chinese–English'
    return f'{c1}–{c2}'


def domain_pair_type(d1, d2):
    """Trade–Trade / Intercultural–Intercultural / Trade–Intercultural"""
    x, y = canonical_domain(d1), canonical_domain(d2)
    if x == 'Trade' and y == 'Trade':
        return 'Trade–Trade'
    if x == 'Intercultural' and y == 'Intercultural':
        return 'Intercultural–Intercultural'
    if {x, y} == {'Trade', 'Intercultural'}:
        return 'Trade–Intercultural'
    return f'{x}–{y}'


def semantic_distance_matrix_from_embeddings(embeddings):
    """与论文一致：cosine distance = 1 - cosine_similarity。"""
    sim = cosine_similarity(embeddings)
    return 1.0 - sim


def _huggingface_host_reachable(timeout=2.5):
    """无网时避免 SentenceTransformer 对 huggingface.co 反复重试（否则会卡住很久）。"""
    try:
        import socket
        socket.create_connection(('huggingface.co', 443), timeout=timeout)
        return True
    except OSError:
        return False


def _hf_hub_snapshot_path(model_id, hub_root):
    """在 HF Hub 缓存目录查找 models--org--name/snapshots/<hash>。"""
    if not hub_root or not os.path.isdir(hub_root):
        return None
    repo_folder = 'models--' + model_id.replace('/', '--')
    snaps = os.path.join(hub_root, repo_folder, 'snapshots')
    if not os.path.isdir(snaps):
        return None
    for entry in sorted(os.listdir(snaps), reverse=True):
        sp = os.path.join(snaps, entry)
        if not os.path.isdir(sp):
            continue
        if os.path.isfile(os.path.join(sp, 'modules.json')):
            return sp
        if os.path.isfile(os.path.join(sp, 'config_sentence_transformers.json')):
            return sp
    return None


def get_sentence_transformer(model_name_or_path, cache_root=None):
    """加载 SentenceTransformer：本地目录、ModelScope、HF 本地缓存，最后才联网下载。"""
    if cache_root is None:
        cache_root = cache_dir
    # 已是本地目录且含 sentence-transformers 配置
    if os.path.isdir(model_name_or_path) and os.path.isfile(
        os.path.join(model_name_or_path, 'config_sentence_transformers.json')
    ):
        st = SentenceTransformer(model_name_or_path)
        return repair_paraphrase_multilingual_minilm_tokenizer(st, model_name_or_path)
    # ModelScope 风格缓存路径 extradimen/xxx
    ms_local = os.path.join(
        cache_root, 'modelscope', 'hub', model_name_or_path.replace('/', os.sep)
    )
    if os.path.isfile(os.path.join(ms_local, 'modules.json')):
        st = SentenceTransformer(ms_local)
        return repair_paraphrase_multilingual_minilm_tokenizer(st, ms_local)

    hf_hub_roots = [
        os.path.join(cache_root, 'huggingface', 'hub'),
        os.path.expanduser('~/.cache/huggingface/hub'),
    ]
    for hr in hf_hub_roots:
        snap = _hf_hub_snapshot_path(model_name_or_path, hr)
        if snap:
            st = SentenceTransformer(snap)
            return repair_paraphrase_multilingual_minilm_tokenizer(st, snap)

    if not _huggingface_host_reachable():
        raise RuntimeError(
            '无法连接 huggingface.co，且本地未找到该模型的 Hugging Face 缓存。'
            '请取消「多模型验证」勾选，或先在可联网环境下载模型到 ~/.cache/huggingface/hub。'
        )
    st = SentenceTransformer(model_name_or_path)
    wd = model_name_or_path if isinstance(model_name_or_path, str) and os.path.isdir(model_name_or_path) else None
    return repair_paraphrase_multilingual_minilm_tokenizer(st, wd)


def build_threshold_graph(data_list, similarity_matrix, threshold):
    """similarity > threshold 的加权无向图；边属性 weight=similarity，distance=1/weight。"""
    n = len(data_list)
    G = nx.Graph()
    for i in range(n):
        G.add_node(
            i,
            word=data_list[i]['word'],
            language=data_list[i].get('language', ''),
            domain=data_list[i].get('category', ''),
            submodule=data_list[i].get('submodule', ''),
        )
    for i in range(n):
        for j in range(i + 1, n):
            s = float(similarity_matrix[i, j])
            if s > threshold:
                w = max(s, 1e-9)
                G.add_edge(i, j, weight=w, similarity=s, distance=1.0 / w)
    return G


def compute_network_centralities(G):
    """返回 dict[node_id] -> weighted_degree, eigenvector, betweenness"""
    n_nodes = G.number_of_nodes()
    wd = {n: 0.0 for n in G.nodes()}
    for u, v, dat in G.edges(data=True):
        w = dat.get('weight', dat.get('similarity', 0))
        wd[u] += w
        wd[v] += w

    ev = {n: 0.0 for n in G.nodes()}
    if n_nodes > 0:
        try:
            ev_calc = nx.eigenvector_centrality_numpy(G, weight='weight')
            ev.update(ev_calc)
        except Exception:
            try:
                ev_calc = nx.eigenvector_centrality(G, weight='weight', max_iter=500)
                ev.update(ev_calc)
            except Exception:
                pass

    bw = {n: 0.0 for n in G.nodes()}
    if n_nodes > 1:
        try:
            bw_calc = nx.betweenness_centrality(G, weight='distance', normalized=True)
            bw.update(bw_calc)
        except Exception:
            try:
                bw_calc = nx.betweenness_centrality(G, normalized=True)
                bw.update(bw_calc)
            except Exception:
                pass

    return wd, ev, bw


def similarity_upper_tri_pearson(sim_a, sim_b):
    """两相似度矩阵上三角 Pearson 相关（词序须一致）。"""
    assert sim_a.shape == sim_b.shape
    iu = np.triu_indices(sim_a.shape[0], k=1)
    x = sim_a[iu].flatten()
    y = sim_b[iu].flatten()
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def mean_similarity_by_domain_groups(similarity_matrix, data_list, mask):
    """在给定节点 mask 下，计算 Trade–Trade / II / TI 平均相似度（仅 i<j 且两节点均在 mask）。"""
    idx = np.where(mask)[0]
    trade_m = []
    inter_m = []
    cross_m = []
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = int(idx[ii]), int(idx[jj])
            d1 = canonical_domain(data_list[i]['category'])
            d2 = canonical_domain(data_list[j]['category'])
            s = float(similarity_matrix[i, j])
            if d1 == 'Trade' and d2 == 'Trade':
                trade_m.append(s)
            elif d1 == 'Intercultural' and d2 == 'Intercultural':
                inter_m.append(s)
            elif {d1, d2} == {'Trade', 'Intercultural'}:
                cross_m.append(s)
    return (
        float(np.mean(trade_m)) if trade_m else float('nan'),
        float(np.mean(inter_m)) if inter_m else float('nan'),
        float(np.mean(cross_m)) if cross_m else float('nan'),
    )


def top_k_bridge_words(data_list, similarity_matrix, threshold, k=5, excluded_indices=None):
    """Bridge Score = cross-domain sim sum + betweenness；仅统计 sim>threshold 的跨 domain 边。"""
    n = len(data_list)
    G = build_threshold_graph(data_list, similarity_matrix, threshold)
    _, _, bw = compute_network_centralities(G)

    ex = set(excluded_indices or [])
    scores = []
    for i in range(n):
        if i in ex:
            continue
        di = canonical_domain(data_list[i]['category'])
        cd_sum = 0.0
        for j in range(n):
            if i == j or j in ex:
                continue
            dj = canonical_domain(data_list[j]['category'])
            if di == dj:
                continue
            s = float(similarity_matrix[i, j])
            if s > threshold:
                cd_sum += s
        b = float(bw.get(i, 0.0))
        scores.append((cd_sum + b, data_list[i]['word'], i))
    scores.sort(key=lambda x: -x[0])
    return [t[1] for t in scores[:k]]


def top_k_central_words(data_list, similarity_matrix, threshold, k=5, excluded_indices=None):
    """按加权度中心性排名。"""
    n = len(data_list)
    G = build_threshold_graph(data_list, similarity_matrix, threshold)
    wd, _, _ = compute_network_centralities(G)
    ex = set(excluded_indices or [])
    ranked = sorted(
        [(wd.get(i, 0.0), data_list[i]['word']) for i in range(n) if i not in ex],
        key=lambda x: -x[0],
    )
    return [t[1] for t in ranked[:k]]


def run_leave_one_submodule_out(data_list, similarity_matrix, threshold):
    """Leave-one-submodule-out：每次移除一个 submodule 标签对应的全部词。"""
    subs = sorted(set(str(item.get('submodule', '') or '') for item in data_list))
    rows = []
    for rem in subs:
        if rem == '':
            continue  # 不删除「空 submodule」作为一轮，避免清空全部无标签数据
        mask = np.array([str(item.get('submodule', '') or '') != rem for item in data_list])
        if mask.sum() < 2:
            continue
        tm, im, cm = mean_similarity_by_domain_groups(similarity_matrix, data_list, mask)
        sub_data = [data_list[i] for i in range(len(data_list)) if mask[i]]
        sub_sim = similarity_matrix[np.ix_(mask, mask)]
        bridges = top_k_bridge_words(sub_data, sub_sim, threshold, k=5)
        central = top_k_central_words(sub_data, sub_sim, threshold, k=5)
        rows.append({
            'Removed Submodule': rem,
            'Remaining N': int(mask.sum()),
            'Trade Mean': tm,
            'Intercultural Mean': im,
            'Cross-domain Mean': cm,
            'Top Bridge Concepts': ', '.join(bridges),
            'Top Central Concepts': ', '.join(central),
        })
    return pd.DataFrame(rows)


def run_random_subsampling(data_list, similarity_matrix, threshold, remove_ratio=0.2, n_iter=100, seed=42):
    """从 Trade / Intercultural 各类内各随机删相同比例，保持两类大致平衡。"""
    rng = random.Random(seed)
    trade_idx = [i for i, item in enumerate(data_list) if canonical_domain(item['category']) == 'Trade']
    inter_idx = [i for i, item in enumerate(data_list) if canonical_domain(item['category']) == 'Intercultural']

    trade_means, inter_means, cross_means = [], [], []
    bridge_counts = {}

    if not trade_idx or not inter_idx:
        return pd.DataFrame(), pd.DataFrame()

    n_remove_t = max(1, int(len(trade_idx) * remove_ratio))
    n_remove_i = max(1, int(len(inter_idx) * remove_ratio))

    for _ in range(n_iter):
        if len(trade_idx) - n_remove_t < 1 or len(inter_idx) - n_remove_i < 1:
            break
        kt = set(rng.sample(trade_idx, n_remove_t))
        ki = set(rng.sample(inter_idx, n_remove_i))
        removed = kt | ki
        mask = np.array([i not in removed for i in range(len(data_list))])
        if mask.sum() < 2:
            continue
        tm, im, cm = mean_similarity_by_domain_groups(similarity_matrix, data_list, mask)
        trade_means.append(tm)
        inter_means.append(im)
        cross_means.append(cm)

        sub_data = [data_list[i] for i in range(len(data_list)) if mask[i]]
        sub_sim = similarity_matrix[np.ix_(mask, mask)]
        top_b = top_k_bridge_words(sub_data, sub_sim, threshold, k=10)
        for w in top_b:
            bridge_counts[w] = bridge_counts.get(w, 0) + 1

    def _safe_mean_sd(arr):
        if not arr:
            return np.nan, np.nan
        a = np.asarray(arr, dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    tm_avg, tm_sd = _safe_mean_sd(trade_means)
    im_avg, im_sd = _safe_mean_sd(inter_means)
    cm_avg, cm_sd = _safe_mean_sd(cross_means)

    summary = pd.DataFrame([{
        'Iterations': len(trade_means),
        'Remove Ratio': remove_ratio,
        'Trade Mean Avg': tm_avg,
        'Trade Mean SD': tm_sd,
        'Intercultural Mean Avg': im_avg,
        'Intercultural Mean SD': im_sd,
        'Cross-domain Mean Avg': cm_avg,
        'Cross-domain Mean SD': cm_sd,
    }])

    stab = pd.DataFrame(
        [{'Concept': k, 'Appeared in Top 10 Times': v, 'Frequency': v / max(len(trade_means), 1)}
         for k, v in sorted(bridge_counts.items(), key=lambda x: -x[1])]
    )
    return summary, stab


def parse_vocabulary_file(file_path, filename):
    """解析上传的词汇文件，支持CSV和Excel格式
    格式：第一列 Class；可选 submodule / concept_id 列；其余列为各语言词汇。
    返回: list of dict，含 category, word, language, submodule, row_index, concept_id(可选)
    """
    data = []

    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        try:
            df = pd.read_excel(file_path, header=0)
        except Exception as e:
            raise ValueError(f"无法读取Excel文件: {str(e)}")
    elif filename.endswith('.csv'):
        try:
            df = pd.read_csv(file_path, header=0, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, header=0, encoding='gbk')
    else:
        raise ValueError("不支持的文件格式，请上传CSV或Excel文件")

    if df.shape[1] < 2:
        raise ValueError("文件必须包含至少2列：Class和至少一种语言的词汇列")

    columns = df.columns.tolist()
    class_col = columns[0]

    submodule_col = None
    concept_col = None
    reserved = set()

    for col in columns[1:]:
        nc = _norm_col_name(col)
        if nc in ('submodule', 'sub_module', '子模块'):
            submodule_col = col
            reserved.add(col)
        elif nc in ('concept_id', 'conceptid'):
            concept_col = col
            reserved.add(col)

    language_cols = [c for c in columns[1:] if c not in reserved]
    if not language_cols:
        raise ValueError("未找到语言词汇列（请确保除 Class/submodule/concept_id 外仍有语言列）")

    for idx, row in df.iterrows():
        category = str(row[class_col]).strip() if pd.notna(row[class_col]) else ''

        sub_val = ''
        if submodule_col is not None and pd.notna(row[submodule_col]):
            sub_val = str(row[submodule_col]).strip()

        if concept_col is not None and pd.notna(row[concept_col]):
            cid = str(row[concept_col]).strip()
        else:
            cid = ''

        if not category:
            continue

        for lang_col in language_cols:
            word = str(row[lang_col]).strip() if pd.notna(row[lang_col]) else ''
            if word:
                language = lang_col.strip()
                item = {
                    'category': category,
                    'word': word,
                    'language': language,
                    'submodule': sub_val,
                    'row_index': idx,
                }
                if cid:
                    item['concept_id'] = cid
                data.append(item)

    if len(data) == 0:
        raise ValueError("文件中没有有效数据")

    return data


def calculate_semantic_distances(data_list, st_model=None):
    """计算词汇语义嵌入。返回 (embeddings_dict, embeddings_ordered ndarray [n, dim])。"""
    if st_model is None:
        st_model = model
    words = [item['word'] for item in data_list]
    print(f"正在计算 {len(words)} 个词汇的语义嵌入...")
    embeddings = st_model.encode(words, show_progress_bar=True)
    emb_array = np.asarray(embeddings)
    embeddings_dict = {}
    for item, emb in zip(data_list, emb_array):
        embeddings_dict[item['word']] = emb
    return embeddings_dict, emb_array


def generate_heatmap(data_list, emb_array, output_path, heatmap_decimal_places=2):
    """生成语义相似度热力图 - 只计算同分类同语言的词汇间相似度，多行显示

    heatmap_decimal_places: 格内相似度显示小数位数（2–6），由前端传入。

    必须使用与 data_list 顺序一致的 emb_array（逐行 encode），不能再用「词字符串→向量」
    的字典：同一英文词（如 Trade）在多行重复出现时，字典只保留最后一次，会把不同行的词
    错误地映射成同一向量，热力图上就出现大片 1.0；中文多为复合词、表中重复字符串较少，
    往往看不出这一问题。
    """
    idx_of = {id(item): i for i, item in enumerate(data_list)}
    emb_array = np.asarray(emb_array)

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

    d = max(2, min(6, int(heatmap_decimal_places)))
    fmt_str = f'.{d}f'
    cjk_font = get_matplotlib_cjk_font_family()
    sans_chain = plt.rcParams.get('font.sans-serif', [])
    merged_sans = [cjk_font] + [x for x in sans_chain if x != cjk_font]

    with plt.rc_context({'font.sans-serif': merged_sans, 'axes.unicode_minus': False}):
        # 如果只有一个分组，直接生成单个热力图
        if num_groups == 1:
            group_key = list(groups.keys())[0]
            group_items = groups[group_key]
            words = [item['word'] for item in group_items]
            
            # 计算相似度矩阵（按节点在 data_list 中的位置取向量，避免同词覆盖）
            embeddings = emb_array[[idx_of[id(it)] for it in group_items]]
            similarity_matrix = cosine_similarity(embeddings)
            
            nw = len(words)
            heatmap_annot_font = 5 if nw > 20 else (6 if nw > 14 else 8)

            plt.figure(figsize=(max(20, len(words) * 0.8), max(16, len(words) * 0.7)))
            ax0 = sns.heatmap(similarity_matrix, 
                        xticklabels=words, 
                        yticklabels=words,
                        annot=True, 
                        fmt=fmt_str,
                        cmap='YlOrRd',
                        cbar_kws={'label': '语义相似度'},
                        square=True,
                        annot_kws={'size': heatmap_annot_font, 'family': cjk_font})
            plt.setp(ax0.get_xticklabels(), rotation=45, ha='right', fontsize=10, fontfamily=cjk_font)
            plt.setp(ax0.get_yticklabels(), rotation=0, fontsize=10, fontfamily=cjk_font)
            plt.title(f'词汇语义相似度热力图 - {group_key[0]} ({group_key[1]})', 
                     fontsize=20, pad=20, fontweight='bold', fontfamily=cjk_font)
        else:
            # 多个分组，生成子图 - 纵向排列（多行）
            fig, axes = plt.subplots(num_groups, 1, figsize=(20, 16 * num_groups))
            if num_groups == 1:
                axes = [axes]
            
            for idx, (group_key, group_items) in enumerate(groups.items()):
                words = [item['word'] for item in group_items]
                embeddings = emb_array[[idx_of[id(it)] for it in group_items]]
                similarity_matrix = cosine_similarity(embeddings)
                
                nw = len(words)
                heatmap_annot_font = 5 if nw > 20 else (6 if nw > 14 else 8)

                ax = axes[idx]
                sns.heatmap(similarity_matrix, 
                            xticklabels=words, 
                            yticklabels=words,
                            annot=True, 
                            fmt=fmt_str,
                            cmap='YlOrRd',
                            cbar_kws={'label': '语义相似度'},
                            square=True,
                            annot_kws={'size': heatmap_annot_font, 'family': cjk_font},
                            ax=ax)
                ax.set_title(f'{group_key[0]} ({group_key[1]})', fontsize=18, fontweight='bold', pad=15, fontfamily=cjk_font)
                ax.set_xlabel('词汇', fontsize=14, fontfamily=cjk_font)
                ax.set_ylabel('词汇', fontsize=14, fontfamily=cjk_font)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10, fontfamily=cjk_font)
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10, fontfamily=cjk_font)
            
            plt.suptitle('词汇语义相似度热力图（按分类和语言分组）', 
                        fontsize=22, fontweight='bold', y=0.995, fontfamily=cjk_font)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    return output_path


def _network_merge_edges_from_similarity(data_list, similarity_matrix, threshold):
    """按行对齐的相似度矩阵建无向边，同词字符串合并为一条边取最大权重。"""
    n = len(data_list)
    words = [item['word'] for item in data_list]
    edge_w = {}
    for i in range(n):
        for j in range(i + 1, n):
            if words[i] == words[j]:
                continue
            s = float(similarity_matrix[i, j])
            if s <= threshold:
                continue
            a, b = sorted([words[i], words[j]])
            edge_w[(a, b)] = max(edge_w.get((a, b), 0.0), s)
    return edge_w


def _network_mutual_topk_edges(edge_w, top_k_per_node):
    """每节点只保留权重最高的 k 条关联边，且仅当两端互为对方 top-k 时保留边（对称稀疏化）。"""
    if top_k_per_node <= 0 or not edge_w:
        return edge_w
    neigh = defaultdict(list)
    for (a, b), sim in edge_w.items():
        neigh[a].append((b, sim))
        neigh[b].append((a, sim))
    top_set = {}
    for u in neigh:
        neigh[u].sort(key=lambda x: -x[1])
        top_set[u] = {t[0] for t in neigh[u][:top_k_per_node]}
    new_w = {}
    for (a, b), sim in edge_w.items():
        if b in top_set.get(a, set()) and a in top_set.get(b, set()):
            new_w[(a, b)] = sim
    return new_w


def _network_cap_edge_count(edge_w, max_edges):
    if max_edges <= 0 or len(edge_w) <= max_edges:
        return edge_w
    items = sorted(edge_w.items(), key=lambda x: -x[1])
    return dict(items[:max_edges])


def _bezier_quad_samples(p0, p1, bend=0.12, n=28):
    """二次贝塞尔曲线采样点，用于绘制弯边。"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    mid = 0.5 * (p0 + p1)
    d = p1 - p0
    ln = np.linalg.norm(d)
    if ln < 1e-9:
        return np.vstack([p0, p1])
    perp = np.array([-d[1], d[0]], dtype=float)
    perp /= np.linalg.norm(perp)
    ctrl = mid + perp * (bend * ln)
    t = np.linspace(0.0, 1.0, n)
    om = 1.0 - t
    curve = (om ** 2)[:, np.newaxis] * p0 + (2 * om * t)[:, np.newaxis] * ctrl + (t ** 2)[:, np.newaxis] * p1
    return curve


def _spread_positions_from_centroid(pos, factor=1.5):
    """在力导向结果上把各点相对质心径向拉开，减轻「全挤在中间、四周空白」。"""
    if not pos or factor <= 1.0:
        return pos
    pts = np.array(list(pos.values()), dtype=float)
    c = pts.mean(axis=0)
    out = {}
    for n, p in pos.items():
        v = np.asarray(p, dtype=float) - c
        out[n] = tuple(c + factor * v)
    return out


def generate_network_graph_weighted(
    data_list,
    emb_matrix,
    output_path,
    threshold=0.3,
    centrality_type='degree',
    power=5,
    top_k_per_node=12,
    max_edges_total=1200,
    curved_edges=True,
    color_by_community=True,
    margin_frac=0.16,
    label_top_n=0,
):
    """生成语义网络图：稀疏边（互斥 top-k + 全局上限）、可选模块度社区着色、弯边与留白。

    emb_matrix 与 data_list 逐行对齐（避免同词字典覆盖）。label_top_n>0 时仅给中心性最高的前 N 个节点画文字。
    """
    G = nx.Graph()

    words = [item['word'] for item in data_list]
    categories = [item['category'] for item in data_list]
    languages = [item.get('language', '') for item in data_list]

    word_to_category = {}
    word_to_language = {}
    for item in data_list:
        w = item['word']
        word_to_category[w] = item['category']
        word_to_language[w] = item.get('language', '')

    unique_categories = list(set(categories))

    for word in words:
        G.add_node(word)

    similarity_matrix = cosine_similarity(np.asarray(emb_matrix))
    edge_w = _network_merge_edges_from_similarity(data_list, similarity_matrix, threshold)
    edge_w = _network_mutual_topk_edges(edge_w, top_k_per_node)
    edge_w = _network_cap_edge_count(edge_w, max_edges_total)

    for (a, b), sim in edge_w.items():
        G.add_edge(a, b, weight=sim)
    
    # 计算加权中心性
    if centrality_type == 'degree':
        # 加权度中心性：所有连接权重的和
        centrality = {}
        for node in G.nodes():
            total_weight = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
            centrality[node] = total_weight
        title_suffix = '加权度中心性'
    else:  # eigenvector
        # 加权特征向量中心性
        try:
            centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            for node in G.nodes():
                centrality.setdefault(node, 0.0)
        except Exception:
            # 如果迭代失败，使用度中心性作为备选
            centrality = {}
            for node in G.nodes():
                total_weight = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
                centrality[node] = total_weight
        title_suffix = '加权特征向量中心性'
    
    # 归一化中心性值（用于节点大小）
    if centrality and len(centrality) > 0:
        cent_values = list(centrality.values())
        max_centrality = max(cent_values)
        min_centrality = min(cent_values)
        centrality_range = max_centrality - min_centrality if max_centrality > min_centrality else 1
        
        # 调试信息
        print(f"节点数量: {len(centrality)}")
        print(f"中心性范围: min={min_centrality:.4f}, max={max_centrality:.4f}, range={centrality_range:.4f}")
        
        # 如果所有节点中心性为0（没有连接），给出警告
        if max_centrality == 0:
            print("警告: 所有节点中心性为0，可能阈值太高导致没有连接")
            centrality_range = 1  # 避免除以0
        elif centrality_range == 0:
            print("警告: 所有节点中心性相同，节点大小将相同")
            centrality_range = 1  # 避免除以0
    else:
        max_centrality = 1
        min_centrality = 0
        centrality_range = 1
        print("警告: 中心性字典为空")
    
    # 绘制网络图（略增大画布留白）
    fig = plt.figure(figsize=(22, 17), facecolor='white')
    ax = fig.add_subplot(111, facecolor='white')

    nn = G.number_of_nodes()
    if nn > 0:
        # k 越大节点间距越大；略大于默认 1/sqrt(n)，再对坐标做径向拉伸，避免团成一小团
        if nn > 1:
            kk = max(6.0 / np.sqrt(nn), 3.4)
        else:
            kk = None
        pos = nx.spring_layout(G, k=kk, iterations=360, seed=42, weight='weight')
        pos = _spread_positions_from_centroid(pos, factor=1.52)
    else:
        pos = {}

    # 边：弯线 + 低对比度，线宽随权重略变
    if G.number_of_edges() > 0:
        wmin = min(d['weight'] for _, _, d in G.edges(data=True))
        wmax = max(d['weight'] for _, _, d in G.edges(data=True))
        wr = (wmax - wmin) if wmax > wmin else 1.0
        for u, v, dat in G.edges(data=True):
            w = float(dat['weight'])
            lw = 0.35 + (w - wmin) / wr * 1.35
            alpha_e = 0.12 + (w - wmin) / wr * 0.18
            p0, p1 = pos[u], pos[v]
            if curved_edges:
                curve = _bezier_quad_samples(p0, p1, bend=0.14, n=32)
                ax.plot(
                    curve[:, 0],
                    curve[:, 1],
                    color=(0.42, 0.45, 0.5),
                    alpha=float(np.clip(alpha_e, 0.08, 0.35)),
                    linewidth=lw,
                    solid_capstyle='round',
                    zorder=1,
                )
            else:
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    color=(0.42, 0.45, 0.5),
                    alpha=float(np.clip(alpha_e, 0.08, 0.35)),
                    linewidth=lw,
                    solid_capstyle='round',
                    zorder=1,
                )

    node_sizes_ordered = []
    node_colors_ordered = []

    import matplotlib.cm as cm

    category_colors = {}
    if len(unique_categories) == 1:
        category_colors[unique_categories[0]] = (0.2, 0.4, 0.8)
    else:
        pal = cm.Set3(np.linspace(0, 1, len(unique_categories)))
        for idx, cat in enumerate(unique_categories):
            category_colors[cat] = pal[idx][:3]

    nodes = list(G.nodes())

    node_comm_index = {}
    if color_by_community and G.number_of_edges() > 0:
        try:
            from networkx.algorithms.community import greedy_modularity_communities

            comms = list(greedy_modularity_communities(G, weight='weight'))
            for ci, comm in enumerate(comms):
                for n in comm:
                    node_comm_index[n] = ci
        except Exception as ex:
            print(f'社区划分失败，回退按分类着色: {ex}')
            node_comm_index = {}
    n_comm = max(len(set(node_comm_index.values())), 1) if node_comm_index else 1
    comm_cmap = cm.get_cmap('tab20')

    if centrality:
        cent_values = list(centrality.values())
        print(f"中心性统计: min={min(cent_values):.4f}, max={max(cent_values):.4f}, range={centrality_range:.4f}")
        print(f"中心性值示例: {dict(list(centrality.items())[:5])}")

    label_nodes = set()
    if label_top_n and label_top_n > 0 and centrality:
        sorted_nodes = sorted(centrality.keys(), key=lambda x: centrality.get(x, 0), reverse=True)
        label_nodes = set(sorted_nodes[: min(label_top_n, len(sorted_nodes))])

    for node in nodes:
        cent_value = centrality.get(node, 0)
        if centrality_range > 0:
            normalized_cent = (cent_value - min_centrality) / centrality_range
            powered_normalized = normalized_cent ** power
        else:
            powered_normalized = 0.5
            print('警告: 所有节点中心性相同或为0，使用默认大小')

        node_size = 280 + powered_normalized * 4500
        node_sizes_ordered.append(node_size)

        if color_by_community and node in node_comm_index:
            ci = node_comm_index[node] % 20
            rgb = comm_cmap(ci / 19.0)[:3]
        else:
            category = word_to_category.get(node, unique_categories[0])
            rgb = category_colors.get(category, (0.2, 0.4, 0.8))
        node_colors_ordered.append(rgb)

    # 注意：部分 networkx 版本不接受 zorder，勿传入以免 draw_networkx_nodes 报错
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_size=node_sizes_ordered,
        node_color=node_colors_ordered,
        alpha=0.92,
        ax=ax,
        edgecolors='#334155',
        linewidths=1.1,
    )

    chinese_font = get_matplotlib_cjk_font_family()

    for idx, node in enumerate(nodes):
        if node not in pos:
            continue
        if label_top_n and label_top_n > 0 and node not in label_nodes:
            continue
        x, y = pos[node]
        node_size = node_sizes_ordered[idx]
        node_color = node_colors_ordered[idx]
        language = word_to_language.get(node, '')

        base_font_size = 7.5 + (node_size / 5000) * 12
        text_length = len(str(node))
        if text_length > 10:
            font_size = base_font_size * (10 / text_length) * 0.92
        else:
            font_size = base_font_size
        font_size = max(7, min(20, font_size))

        if sum(node_color) / 3 > 0.52:
            text_color = '#1e293b'
        else:
            text_color = '#f8fafc'

        if language == 'Chinese' or any('\u4e00' <= char <= '\u9fff' for char in str(node)):
            font_family = chinese_font
        else:
            font_family = 'DejaVu Sans'

        ax.text(
            x,
            y,
            node,
            fontsize=font_size,
            ha='center',
            va='center',
            color=text_color,
            fontweight='600',
            fontfamily=font_family,
            bbox=dict(
                boxstyle='round,pad=0.22',
                facecolor=node_color,
                alpha=0.88,
                edgecolor='#94a3b8',
                linewidth=0.35,
            ),
            zorder=4,
        )

    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        xr = max(xs) - min(xs) if xs else 1.0
        yr = max(ys) - min(ys) if ys else 1.0
        span = max(xr, yr, 1e-6)
        pad = margin_frac * span
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect('equal', adjustable='datalim')

    sub = []
    if top_k_per_node > 0:
        sub.append(f'每点最多{top_k_per_node}强边(互斥)')
    if max_edges_total > 0:
        sub.append(f'全局≤{max_edges_total}边')
    if color_by_community:
        sub.append('社区着色')
    if curved_edges:
        sub.append('弯边')
    extra = '；'.join(sub) if sub else '全阈值边'
    plt.title(
        f'语义网络图 - {title_suffix}\n（{extra}）',
        fontsize=18,
        pad=16,
        fontweight='bold',
        fontfamily=chinese_font,
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
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
    
    print("正在进行MDS降维 (cosine distance = 1 - cosine similarity)...")
    # sklearn 新版仅接受 normalized_stress in {True, False, 'auto'}；'stress' 会报错
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    distance_matrix = semantic_distance_matrix_from_embeddings(np.asarray(embeddings))
    coords = mds.fit_transform(distance_matrix)
    stress_2d = float(getattr(mds, 'stress_', np.nan))
    
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
    
    # 配置中文字体 - 使用和网络图相同的字体检测逻辑
    font_prop = fm.FontProperties()
    chinese_font_mds = 'sans-serif'
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        # 按优先级查找中文字体
        font_priority = [
            'Noto Sans CJK SC',
            'Noto Sans CJK JP',
            'Source Han Sans SC',
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'SimHei',
            'SimSun',
            'Microsoft YaHei'
        ]
        for font_name in font_priority:
            if font_name in available_fonts:
                chinese_font_mds = font_name
                break
        font_prop.set_family(chinese_font_mds)
        print(f'MDS图使用字体: {chinese_font_mds}')
    except Exception as e:
        print(f'MDS图字体检测失败: {e}, 使用默认字体')
        font_prop.set_family('sans-serif')
    
    # 创建词汇到分类的映射（用于文本框背景色）
    word_to_category = {item['word']: item['category'] for item in data_list}
    
    # 连接同一行的词（相同row_index）
    # 按row_index分组，找出同一行的词
    row_groups = {}
    for i, item in enumerate(data_list):
        row_idx = item.get('row_index', i)  # 如果没有row_index，使用索引作为fallback
        if row_idx not in row_groups:
            row_groups[row_idx] = []
        row_groups[row_idx].append(i)  # 存储索引
    
    # 绘制连接线（同一行的词）
    for row_idx, indices in row_groups.items():
        if len(indices) > 1:  # 只有同一行有多个词时才连接
            # 获取这些词的坐标
            row_coords = coords[indices]
            # 绘制连接线
            for i in range(len(indices) - 1):
                for j in range(i + 1, len(indices)):
                    plt.plot([row_coords[i, 0], row_coords[j, 0]], 
                            [row_coords[i, 1], row_coords[j, 1]], 
                            'k-', alpha=0.3, linewidth=1, zorder=0)  # 灰色细线，在底层
    
    # 添加标签（文本框背景色按分类填充）
    for i, word in enumerate(words):
        category = word_to_category.get(word, unique_categories[0])
        category_color = category_colors.get(category, (0.2, 0.4, 0.8))
        plt.annotate(word, (coords[i, 0], coords[i, 1]), 
                    fontsize=12, ha='center', va='center',
                    fontproperties=font_prop,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=category_color, alpha=0.8, edgecolor='gray'))
    
    # 添加图例
    if len(unique_categories) > 1:
        plt.legend(loc='upper right', fontsize=11, framealpha=0.9, prop=font_prop)
    
    plt.title('词汇语义空间2D可视化 (MDS降维)', fontsize=20, pad=20, fontweight='bold', fontfamily=chinese_font_mds)
    plt.xlabel('维度1', fontsize=14, fontweight='bold', fontfamily=chinese_font_mds)
    plt.ylabel('维度2', fontsize=14, fontweight='bold', fontfamily=chinese_font_mds)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    # 与论文式(14)一致：跨语言位移使用以下 2D 坐标计算 ||x_EN - x_CN||
    return output_path, stress_2d, np.asarray(coords)

def generate_mds_3d_plot(data_list, embeddings_dict, output_path):
    """使用MDS降维生成3D交互式可视化 - 按分类着色"""
    from sklearn.manifold import MDS
    import plotly.graph_objects as go
    import plotly.express as px
    
    words = [item['word'] for item in data_list]
    categories = [item['category'] for item in data_list]
    languages = [item.get('language', '') for item in data_list]
    unique_categories = list(set(categories))
    
    # 获取嵌入向量
    embeddings = [embeddings_dict[word] for word in words]
    
    print("正在进行3D MDS降维 (cosine distance = 1 - cosine similarity)...")
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    distance_matrix = semantic_distance_matrix_from_embeddings(np.asarray(embeddings))
    coords = mds.fit_transform(distance_matrix)
    stress_3d = float(getattr(mds, 'stress_', np.nan))
    
    # 为每个分类分配颜色
    import matplotlib.cm as cm
    category_colors = {}
    if len(unique_categories) == 1:
        category_colors[unique_categories[0]] = 'rgb(51, 102, 204)'
    else:
        colors = cm.Set3(np.linspace(0, 1, len(unique_categories)))
        for idx, cat in enumerate(unique_categories):
            r, g, b = colors[idx][:3]
            category_colors[cat] = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
    
    # 准备数据
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    z_coords = coords[:, 2]
    
    # 创建3D散点图
    fig = go.Figure()
    
    # 按分类绘制
    for category in unique_categories:
        indices = [i for i, item in enumerate(data_list) if item['category'] == category]
        if indices:
            fig.add_trace(go.Scatter3d(
                x=x_coords[indices],
                y=y_coords[indices],
                z=z_coords[indices],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=category_colors[category],
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
                text=[words[i] for i in indices],
                textposition='middle center',
                name=category,
                hovertemplate='<b>%{text}</b><br>' +
                            '分类: ' + category + '<br>' +
                            '语言: ' + [languages[i] for i in indices][0] + '<br>' +
                            'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))
    
    # 连接同一行的词
    row_groups = {}
    for i, item in enumerate(data_list):
        row_idx = item.get('row_index', i)
        if row_idx not in row_groups:
            row_groups[row_idx] = []
        row_groups[row_idx].append(i)
    
    # 绘制连接线
    for row_idx, indices in row_groups.items():
        if len(indices) > 1:
            for i in range(len(indices) - 1):
                for j in range(i + 1, len(indices)):
                    fig.add_trace(go.Scatter3d(
                        x=[x_coords[indices[i]], x_coords[indices[j]]],
                        y=[y_coords[indices[i]], y_coords[indices[j]]],
                        z=[z_coords[indices[i]], z_coords[indices[j]]],
                        mode='lines',
                        line=dict(color='gray', width=2, dash='dash'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text='语义空间3D可视化（MDS降维）',
            font=dict(size=20),
            x=0.5
        ),
        scene=dict(
            xaxis_title='维度1',
            yaxis_title='维度2',
            zaxis_title='维度3',
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # 保存为HTML文件
    fig.write_html(output_path)
    print(f"3D MDS图已保存: {output_path}")
    return output_path, stress_3d


def export_empirical_tables_xlsx(
    data_list,
    similarity_matrix,
    stress_2d,
    stress_3d,
    threshold,
    output_path,
    coords_mds2d=None,
):
    """生成论文实证汇总 Excel（多 sheet）。
    跨语言位移汇总（论文式14）：使用 MDS 二维坐标上 ||x_EN-x_CN||；配对表同时给出嵌入空间 D_ij=1-S_ij。
    """
    n = len(data_list)
    words = [item['word'] for item in data_list]
    dist_mat = 1.0 - similarity_matrix

    # -------- 1 Pairwise Similarity --------
    pair_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(similarity_matrix[i, j])
            d1 = canonical_domain(data_list[i]['category'])
            d2 = canonical_domain(data_list[j]['category'])
            pair_rows.append({
                'Word 1': words[i],
                'Word 2': words[j],
                'Language 1': data_list[i].get('language', ''),
                'Language 2': data_list[j].get('language', ''),
                'Domain 1': d1,
                'Domain 2': d2,
                'Submodule 1': data_list[i].get('submodule', ''),
                'Submodule 2': data_list[j].get('submodule', ''),
                'Similarity': sim,
                'Distance': float(dist_mat[i, j]),
                'Domain Pair': domain_pair_type(data_list[i]['category'], data_list[j]['category']),
                'Language Pair': language_pair_label(data_list[i].get('language', ''), data_list[j].get('language', '')),
            })
    df_pair = pd.DataFrame(pair_rows)

    def agg_sim_stats(sim_list):
        sim_list = [float(x) for x in sim_list if np.isfinite(x)]
        if not sim_list:
            return {'N': 0, 'Mean Similarity': np.nan, 'SD': np.nan, 'Min': np.nan, 'Max': np.nan, 'Mean Distance': np.nan}
        arr = np.array(sim_list)
        return {
            'N': len(sim_list),
            'Mean Similarity': float(np.mean(arr)),
            'SD': float(np.std(arr, ddof=0)),
            'Min': float(np.min(arr)),
            'Max': float(np.max(arr)),
            'Mean Distance': float(np.mean(1.0 - arr)),
        }

    # -------- 2 Domain Similarity Summary --------
    domain_buckets = {'Trade–Trade': [], 'Intercultural–Intercultural': [], 'Trade–Intercultural': []}
    for i in range(n):
        for j in range(i + 1, n):
            dp = domain_pair_type(data_list[i]['category'], data_list[j]['category'])
            if dp in domain_buckets:
                domain_buckets[dp].append(similarity_matrix[i, j])
    df_dom = pd.DataFrame([
        {'Domain Pair': k, **agg_sim_stats(domain_buckets[k])}
        for k in ['Trade–Trade', 'Intercultural–Intercultural', 'Trade–Intercultural']
    ])

    # -------- 3 Language Similarity Summary --------
    lang_buckets = {'Chinese–Chinese': [], 'English–English': [], 'Chinese–English': []}
    for i in range(n):
        for j in range(i + 1, n):
            lp = language_pair_label(data_list[i].get('language', ''), data_list[j].get('language', ''))
            if lp in lang_buckets:
                lang_buckets[lp].append(similarity_matrix[i, j])
    df_lang = pd.DataFrame([
        {'Language Pair': k, **agg_sim_stats(lang_buckets[k])}
        for k in ['Chinese–Chinese', 'English–English', 'Chinese–English']
    ])

    # -------- 4 Submodule internal similarity --------
    sub_dom_pairs = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            si = str(data_list[i].get('submodule', '') or '')
            sj = str(data_list[j].get('submodule', '') or '')
            if si and si == sj:
                dom = canonical_domain(data_list[i]['category'])
                if canonical_domain(data_list[j]['category']) != dom:
                    dom = 'Mixed'
                sub_dom_pairs[(si, dom)].append(similarity_matrix[i, j])
    df_sub_in = pd.DataFrame([
        {
            'Submodule': key[0],
            'Domain': key[1],
            'N Pairs': len(vals),
            'Mean Similarity': float(np.mean(vals)),
            'SD': float(np.std(vals, ddof=0)),
            'Min': float(np.min(vals)),
            'Max': float(np.max(vals)),
        }
        for key, vals in sorted(sub_dom_pairs.items())
        if vals
    ])

    # -------- 5 Submodule cross-domain (different submodule labels) --------
    cross_sub = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            si = str(data_list[i].get('submodule', '') or '')
            sj = str(data_list[j].get('submodule', '') or '')
            if si != sj:
                a, b = sorted([si, sj])
                dp = domain_pair_type(data_list[i]['category'], data_list[j]['category'])
                cross_sub[(a, b, dp)].append(similarity_matrix[i, j])
    df_cross_sub = pd.DataFrame([
        {
            'Submodule 1': key[0],
            'Submodule 2': key[1],
            'Domain Pair': key[2],
            'N': len(vals),
            'Mean Similarity': float(np.mean(vals)),
            'SD': float(np.std(vals, ddof=0)),
        }
        for key, vals in sorted(cross_sub.items())
        if vals
    ])

    # -------- 6 Centrality Ranking --------
    G = build_threshold_graph(data_list, similarity_matrix, threshold)
    wd, ev, bw = compute_network_centralities(G)
    cent_rows = []
    for i in range(n):
        cent_rows.append({
            'Word': words[i],
            'Language': data_list[i].get('language', ''),
            'Domain': canonical_domain(data_list[i]['category']),
            'Submodule': data_list[i].get('submodule', ''),
            'Weighted Degree': wd.get(i, 0.0),
            'Eigenvector Centrality': ev.get(i, 0.0),
            'Betweenness Centrality': bw.get(i, 0.0),
        })
    df_cent = pd.DataFrame(cent_rows)
    df_cent = df_cent.sort_values('Weighted Degree', ascending=False).reset_index(drop=True)
    df_cent.insert(0, 'Rank', range(1, len(df_cent) + 1))

    # -------- 7 Bridge Concept Ranking --------
    bridge_rows = []
    for i in range(n):
        di = canonical_domain(data_list[i]['category'])
        cd_links = 0
        cd_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            dj = canonical_domain(data_list[j]['category'])
            if di == dj:
                continue
            s = float(similarity_matrix[i, j])
            if s > threshold:
                cd_links += 1
                cd_sum += s
        cd_mean = cd_sum / cd_links if cd_links else 0.0
        bscore = cd_sum + float(bw.get(i, 0.0))
        bridge_rows.append({
            'Word': words[i],
            'Language': data_list[i].get('language', ''),
            'Domain': di,
            'Submodule': data_list[i].get('submodule', ''),
            'Cross-domain Links': cd_links,
            'Cross-domain Similarity Sum': cd_sum,
            'Cross-domain Similarity Mean': cd_mean,
            'Betweenness Centrality': float(bw.get(i, 0.0)),
            'Bridge Score': bscore,
        })
    df_bridge = pd.DataFrame(bridge_rows)
    df_bridge = df_bridge.sort_values('Bridge Score', ascending=False).reset_index(drop=True)
    df_bridge.insert(0, 'Rank', range(1, len(df_bridge) + 1))

    # -------- 8 Cross-linguistic Pair Distance --------
    groups = defaultdict(list)
    for idx, item in enumerate(data_list):
        cid = item.get('concept_id')
        key = cid if cid else item['row_index']
        groups[key].append(idx)

    xl_rows = []
    for key, idxs in groups.items():
        ens = [i for i in idxs if canonical_language(data_list[i]['language']) == 'English']
        zhs = [i for i in idxs if canonical_language(data_list[i]['language']) == 'Chinese']
        if not ens or not zhs:
            continue
        ei, zi = ens[0], zhs[0]
        sim = float(similarity_matrix[ei, zi])
        sem_dist = float(1.0 - sim)
        row = {
            'Concept ID': str(key),
            'English Term': words[ei],
            'Chinese Term': words[zi],
            'Domain': canonical_domain(data_list[ei]['category']),
            'Submodule': str(data_list[ei].get('submodule', '') or ''),
            'Similarity': sim,
            'Semantic Distance (1-cosine)': sem_dist,
        }
        if coords_mds2d is not None and len(coords_mds2d) > max(ei, zi):
            d_mds = float(
                np.linalg.norm(coords_mds2d[ei] - coords_mds2d[zi])
            )
            row['MDS-2D Cross-Ling Displacement'] = d_mds
        else:
            row['MDS-2D Cross-Ling Displacement'] = np.nan
        xl_rows.append(row)
    df_xl = pd.DataFrame(xl_rows)

    # -------- 9 Cross-linguistic Displacement by Domain（论文式14：MDS 二维位移）--------
    disp_key = 'MDS-2D Cross-Ling Displacement'
    disp = defaultdict(list)
    for r in xl_rows:
        v = r.get(disp_key)
        if v is not None and np.isfinite(v):
            disp[r['Domain']].append(float(v))
    df_disp = pd.DataFrame([
        {
            'Domain': dom,
            'N': len(vals),
            'Mean MDS-2D Displacement': float(np.mean(vals)),
            'SD': float(np.std(vals, ddof=0)),
            'Min': float(np.min(vals)),
            'Max': float(np.max(vals)),
        }
        for dom, vals in sorted(disp.items())
    ])

    # -------- 10 MDS Stress --------
    df_stress = pd.DataFrame([
        {'Projection': 'MDS 2D', 'Stress': stress_2d, 'Distance Type': '1 - cosine similarity'},
        {'Projection': 'MDS 3D', 'Stress': stress_3d, 'Distance Type': '1 - cosine similarity'},
    ])

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_pair.to_excel(writer, sheet_name='Pairwise Similarity', index=False)
        df_dom.to_excel(writer, sheet_name='Domain Similarity Summary', index=False)
        df_lang.to_excel(writer, sheet_name='Language Similarity Summary', index=False)
        df_sub_in.to_excel(writer, sheet_name='Submodule Similarity Summary', index=False)
        df_cross_sub.to_excel(writer, sheet_name='Submodule Cross-domain Sim', index=False)
        df_cent.to_excel(writer, sheet_name='Centrality Ranking', index=False)
        df_bridge.to_excel(writer, sheet_name='Bridge Concept Ranking', index=False)
        df_xl.to_excel(writer, sheet_name='Cross-linguistic Pair Distance', index=False)
        df_disp.to_excel(writer, sheet_name='XLing MDS2D Displacement', index=False)
        df_stress.to_excel(writer, sheet_name='MDS Stress Summary', index=False)

    print(f"实证表格已保存: {output_path}")
    return output_path


def run_cross_model_validation(
    data_list,
    baseline_similarity_matrix,
    model_names,
    threshold,
    baseline_model_label='paraphrase-multilingual-MiniLM-L12-v2 (baseline)',
):
    """多模型：先写当前 baseline 一行（相关=1），再对其余模型计算相关与统计。"""
    rows = []
    baseline = baseline_similarity_matrix
    mask = np.ones(len(data_list), dtype=bool)
    btm, bim, bcm = mean_similarity_by_domain_groups(baseline, data_list, mask)
    btops = top_k_bridge_words(data_list, baseline, threshold, k=5)
    rows.append({
        'Model': baseline_model_label,
        'Trade Mean': btm,
        'Intercultural Mean': bim,
        'Cross-domain Mean': bcm,
        'Top Bridge Concepts': ', '.join(btops),
        'Matrix Correlation with Baseline': 1.0,
    })

    seen = {baseline_model_label.lower()}
    for name in model_names:
        name = name.strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        try:
            mdl = get_sentence_transformer(name)
            _, emb = calculate_semantic_distances(data_list, mdl)
            sim = cosine_similarity(emb)
        except Exception as ex:
            rows.append({
                'Model': name,
                'Trade Mean': np.nan,
                'Intercultural Mean': np.nan,
                'Cross-domain Mean': np.nan,
                'Top Bridge Concepts': str(ex)[:200],
                'Matrix Correlation with Baseline': np.nan,
            })
            continue

        tm, im, cm = mean_similarity_by_domain_groups(sim, data_list, mask)
        tops = top_k_bridge_words(data_list, sim, threshold, k=5)
        corr = similarity_upper_tri_pearson(baseline, sim)
        rows.append({
            'Model': name,
            'Trade Mean': tm,
            'Intercultural Mean': im,
            'Cross-domain Mean': cm,
            'Top Bridge Concepts': ', '.join(tops),
            'Matrix Correlation with Baseline': corr,
        })
    return pd.DataFrame(rows)


def generate_similarity_excel(data_list, similarity_matrix, words, output_path):
    """生成包含所有词汇对相似度的Excel文件"""
    import pandas as pd
    
    # 创建所有词汇对的列表
    pairs = []
    languages = [item.get('language', '') for item in data_list]
    
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            word1 = words[i]
            word2 = words[j]
            lang1 = languages[i]
            lang2 = languages[j]
            similarity = float(similarity_matrix[i][j])
            
            # 判断配对类型
            if lang1 == 'Chinese' and lang2 == 'Chinese':
                pair_type = '中文-中文'
            elif lang1 == 'Chinese' and lang2 == 'English':
                pair_type = '中文-英文'
            elif lang1 == 'English' and lang2 == 'Chinese':
                pair_type = '英文-中文'
            elif lang1 == 'English' and lang2 == 'English':
                pair_type = '英文-英文'
            else:
                pair_type = f'{lang1}-{lang2}'
            
            pairs.append({
                '词汇1': word1,
                '语言1': lang1,
                '分类1': data_list[i]['category'],
                '词汇2': word2,
                '语言2': lang2,
                '分类2': data_list[j]['category'],
                '配对类型': pair_type,
                '相似度': similarity
            })
    
    # 创建DataFrame并按相似度排序
    df = pd.DataFrame(pairs)
    df = df.sort_values('相似度', ascending=False)
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # 从1开始编号
    
    # 保存为Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 所有配对
        df.to_excel(writer, sheet_name='所有配对', index=True, index_label='排名')
        
        # 按配对类型分组
        for pair_type in df['配对类型'].unique():
            df_type = df[df['配对类型'] == pair_type].copy()
            df_type = df_type.reset_index(drop=True)
            df_type.index = df_type.index + 1
            sheet_name = pair_type[:31]  # Excel工作表名称限制31个字符
            df_type.to_excel(writer, sheet_name=sheet_name, index=True, index_label='排名')
    
    print(f"相似度Excel文件已保存: {output_path}")
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
            
            # 计算语义嵌入（有序矩阵与词表一一对应，避免同词覆盖）
            embeddings_dict, emb_matrix = calculate_semantic_distances(data_list)
            
            # 生成时间戳
            result_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 生成可视化
            results = {}
            
            try:
                hdp = int(request.form.get('heatmap_decimal_places', 2))
            except (TypeError, ValueError):
                hdp = 2
            hdp = max(2, min(6, hdp))
            
            # 1. 热力图（按分类和语言分组）
            heatmap_path = os.path.join(app.config['RESULTS_FOLDER'], f'heatmap_{result_timestamp}.png')
            generate_heatmap(data_list, emb_matrix, heatmap_path, heatmap_decimal_places=hdp)
            results['heatmap'] = f'/results/heatmap_{result_timestamp}.png'
            results['heatmap_decimal_places'] = hdp
            
            # 2. 网络图（两个图：加权度中心性和加权特征向量中心性）
            threshold = float(request.form.get('threshold', 0.3))
            power = float(request.form.get('power', 5))
            try:
                ntk = int(request.form.get('network_top_k_per_node', '12'))
            except (TypeError, ValueError):
                ntk = 12
            try:
                nme = int(request.form.get('network_max_edges', '1200'))
            except (TypeError, ValueError):
                nme = 1200
            ntk = max(0, min(ntk, 500))
            nme = max(0, min(nme, 500000))
            curved_e = request.form.get('network_curved_edges', '1') not in ('0', 'false', 'off', 'no')
            comm_c = request.form.get('network_community_colors', '1') not in ('0', 'false', 'off', 'no')
            try:
                ltn = int(request.form.get('network_label_top_n', '0'))
            except (TypeError, ValueError):
                ltn = 0
            ltn = max(0, min(ltn, 2000))
            net_kw = dict(
                top_k_per_node=ntk,
                max_edges_total=nme,
                curved_edges=curved_e,
                color_by_community=comm_c,
                label_top_n=ltn,
            )
            
            # 加权度中心性网络图
            network_degree_path = os.path.join(app.config['RESULTS_FOLDER'], f'network_degree_{result_timestamp}.png')
            generate_network_graph_weighted(
                data_list, emb_matrix, network_degree_path, threshold, 'degree', power, **net_kw
            )
            results['network_degree'] = f'/results/network_degree_{result_timestamp}.png'
            
            # 加权特征向量中心性网络图
            network_eigen_path = os.path.join(app.config['RESULTS_FOLDER'], f'network_eigen_{result_timestamp}.png')
            generate_network_graph_weighted(
                data_list, emb_matrix, network_eigen_path, threshold, 'eigenvector', power, **net_kw
            )
            results['network_eigen'] = f'/results/network_eigen_{result_timestamp}.png'
            
            # 3. MDS 2D可视化（按分类着色）；距离矩阵为 1 - cosine similarity；坐标用于论文式(14)跨语言位移
            mds_path = os.path.join(app.config['RESULTS_FOLDER'], f'mds_{result_timestamp}.png')
            _, stress_2d, coords_mds2d = generate_mds_plot(data_list, embeddings_dict, mds_path)
            results['mds'] = f'/results/mds_{result_timestamp}.png'
            
            # 4. MDS 3D可视化（交互式）
            mds_3d_path = os.path.join(app.config['RESULTS_FOLDER'], f'mds_3d_{result_timestamp}.html')
            _, stress_3d = generate_mds_3d_plot(data_list, embeddings_dict, mds_3d_path)
            results['mds_3d'] = f'/results/mds_3d_{result_timestamp}.html'
            
            # 5. 生成相似度数据（所有词汇对）
            words = [item['word'] for item in data_list]
            similarity_matrix = cosine_similarity(emb_matrix)
            results['mds_stress_2d'] = stress_2d
            results['mds_stress_3d'] = stress_3d
            
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
            
            # 6. 生成Excel相似度数据文件
            excel_path = os.path.join(app.config['RESULTS_FOLDER'], f'similarity_data_{result_timestamp}.xlsx')
            generate_similarity_excel(data_list, similarity_matrix, words, excel_path)
            results['excel_download'] = f'/results/similarity_data_{result_timestamp}.xlsx'

            # 7. 论文实证表 empirical_tables
            empirical_path = os.path.join(
                app.config['RESULTS_FOLDER'], f'empirical_tables_{result_timestamp}.xlsx'
            )
            export_empirical_tables_xlsx(
                data_list,
                similarity_matrix,
                stress_2d,
                stress_3d,
                threshold,
                empirical_path,
                coords_mds2d=coords_mds2d,
            )
            results['empirical_tables_download'] = f'/results/empirical_tables_{result_timestamp}.xlsx'

            # 可选：Leave-one-submodule-out / Random subsampling（耗时，表单勾选）
            run_loo = request.form.get('run_loo') in ('1', 'true', 'on', 'yes')
            run_rs = request.form.get('run_rs') in ('1', 'true', 'on', 'yes')
            if run_loo or run_rs:
                rb_path = os.path.join(
                    app.config['RESULTS_FOLDER'], f'robustness_tables_{result_timestamp}.xlsx'
                )
                with pd.ExcelWriter(rb_path, engine='openpyxl') as writer:
                    if run_loo:
                        df_loo = run_leave_one_submodule_out(data_list, similarity_matrix, threshold)
                        df_loo.to_excel(writer, sheet_name='Leave-one-submodule-out', index=False)
                    if run_rs:
                        rr = float(request.form.get('remove_ratio', 0.2))
                        ni = int(request.form.get('n_iter', 100))
                        df_rs_sum, df_rs_bridge = run_random_subsampling(
                            data_list, similarity_matrix, threshold, remove_ratio=rr, n_iter=ni
                        )
                        df_rs_sum.to_excel(writer, sheet_name='Random Subsampling Summary', index=False)
                        df_rs_bridge.to_excel(writer, sheet_name='Bridge Concept Stability', index=False)
                results['robustness_tables_download'] = f'/results/robustness_tables_{result_timestamp}.xlsx'

            # 可选：多模型验证（每个模型单独加载，可能很慢）
            raw_models = request.form.get('cross_models', '') or ''
            extra_models = [
                m.strip()
                for m in re.split(r'[\n,]+', raw_models)
                if m.strip()
            ]
            if extra_models:
                cv_path = os.path.join(
                    app.config['RESULTS_FOLDER'], f'cross_model_validation_{result_timestamp}.xlsx'
                )
                df_cv = run_cross_model_validation(
                    data_list, similarity_matrix, extra_models, threshold
                )
                df_cv.to_excel(cv_path, sheet_name='Cross-model Validation Summary', index=False)
                results['cross_model_validation_download'] = (
                    f'/results/cross_model_validation_{result_timestamp}.xlsx'
                )
            
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
                'value': degree + 1,  # 节点大小基于度中心性（连接数），度越大节点越大
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
    # 从环境变量读取端口，默认为 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
