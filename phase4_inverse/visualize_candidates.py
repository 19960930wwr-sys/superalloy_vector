"""
Phase 4: 候选合金可视化验证
- 嵌入空间分布图
- 平行坐标图
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (DATA_DIR, ELEMENTS, PROCESS_COLS, TASKS, FIGURES_DIR,
                    RESULTS_DIR, EMBEDDINGS_DIR, TOKENIZER_DIR, MODELS_DIR,
                    EMBEDDING_DIM, SEED)


def load_candidates():
    """加载候选合金"""
    path = RESULTS_DIR / "inverse_design_candidates.json"
    with open(path) as f:
        data = json.load(f)
    return data['candidates']


def embedding_space_plot():
    """
    嵌入空间分布图
    将训练集合金和候选合金投影到同一2D空间
    """
    import torch
    from phase3_multitask.model_multitask import MultiTaskModel
    from phase3_multitask.dataset_alloy import load_alloy_dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    dataset = load_alloy_dataset('E_pa')
    
    # 加载模型获取共享特征
    input_dim = dataset.input_dim
    model = MultiTaskModel(input_dim).to(device)
    model.load_state_dict(
        torch.load(MODELS_DIR / "multitask_E_pa_fold0.pt", 
                  weights_only=True, map_location=device)
    )
    model.eval()
    
    # 获取训练集的共享特征
    print("Computing shared features for training set...")
    train_features = []
    train_tasks = []
    train_targets = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            inp = sample['input'].unsqueeze(0).to(device)
            feat = model.get_shared_features(inp)
            train_features.append(feat.cpu().numpy().flatten())
            train_tasks.append(sample['task_name'])
            train_targets.append(sample['target'])
    
    train_features = np.array(train_features)
    
    # 加载候选合金并计算其特征
    candidates = load_candidates()
    candidate_features = []
    
    embedding_matrix = np.load(EMBEDDINGS_DIR / "E_pa.npy")
    with open(TOKENIZER_DIR / "vocab.json") as f:
        vocab = json.load(f)
    
    from phase4_inverse.inverse_design import AlloyDesignProblem
    problem = AlloyDesignProblem(model, embedding_matrix, vocab, device)
    
    for cand in candidates:
        # 重建成分和工艺向量
        comp = np.zeros(len(ELEMENTS))
        for i, elem in enumerate(ELEMENTS):
            comp[i] = cand['composition'].get(elem, 0.0)
        
        proc = np.zeros(len(PROCESS_COLS))
        for i, col in enumerate(PROCESS_COLS):
            proc[i] = cand['process'].get(col, 0.0)
        
        inp = problem.compute_alloy_input(comp, proc)
        with torch.no_grad():
            feat = model.get_shared_features(inp.unsqueeze(0))
            candidate_features.append(feat.cpu().numpy().flatten())
    
    candidate_features = np.array(candidate_features)
    
    # 合并后做t-SNE
    all_features = np.vstack([train_features, candidate_features])
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    coords_all = tsne.fit_transform(all_features)
    
    n_train = len(train_features)
    coords_train = coords_all[:n_train]
    coords_cand = coords_all[n_train:]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 标记高性能区域
    # 蠕变数据按性能值着色
    creep_mask = np.array([t == 'creep' for t in train_tasks])
    other_mask = ~creep_mask
    
    # 其他数据用灰色
    ax.scatter(coords_train[other_mask, 0], coords_train[other_mask, 1],
              c='lightgray', s=20, alpha=0.3, label='Other alloys')
    
    # 蠕变数据用颜色映射
    if creep_mask.any():
        creep_targets = np.array(train_targets)[creep_mask]
        sc = ax.scatter(coords_train[creep_mask, 0], coords_train[creep_mask, 1],
                       c=creep_targets, cmap='RdYlGn', s=30, alpha=0.6,
                       label='Creep data (colored by life)')
        plt.colorbar(sc, ax=ax, label='Creep life')
    
    # 候选合金用星号标记
    ax.scatter(coords_cand[:, 0], coords_cand[:, 1],
              c='red', marker='*', s=300, zorder=10, edgecolors='black',
              linewidths=1, label='Candidates')
    
    # 标注候选编号
    for i, (x, y) in enumerate(coords_cand):
        ax.annotate(f'C{i+1}', (x, y), textcoords="offset points",
                   xytext=(8, 8), fontsize=9, fontweight='bold', color='red')
    
    ax.set_xlabel('t-SNE dim 1', fontsize=12)
    ax.set_ylabel('t-SNE dim 2', fontsize=12)
    ax.set_title('Embedding Space: Training Alloys vs Candidates', fontsize=14)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "embedding_space_candidates.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'embedding_space_candidates.png'}")


def parallel_coordinates_plot():
    """平行坐标图：展示候选合金的多维性能"""
    candidates = load_candidates()
    
    if not candidates:
        print("No candidates found.")
        return
    
    # 构建性能数据
    properties = list(TASKS.keys())
    data = []
    for cand in candidates:
        row = {'Candidate': f"C{cand['id']}"}
        for prop in properties:
            row[prop] = cand['predicted_properties'].get(prop, np.nan)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 归一化到[0,1]用于平行坐标图
    df_norm = df.copy()
    for col in properties:
        if col in df_norm.columns:
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()
            if col_max > col_min:
                df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
            else:
                df_norm[col] = 0.5
    
    # 绘制平行坐标图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(candidates)))
    
    x = range(len(properties))
    for i, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[p] for p in properties]
        ax.plot(x, values, '-o', color=colors[i], linewidth=2, markersize=6,
               label=f"C{candidates[i]['id']}", alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in properties], fontsize=10)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Parallel Coordinates: Candidate Alloy Properties', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 添加原始值标注
    for i, p in enumerate(properties):
        ax.annotate(f"[{df[p].min():.2f}, {df[p].max():.2f}]",
                   (i, -0.08), ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "parallel_coordinates_candidates.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'parallel_coordinates_candidates.png'}")


def composition_heatmap():
    """候选合金成分热力图"""
    candidates = load_candidates()
    
    # 构建成分矩阵
    comp_data = []
    for cand in candidates:
        row = {elem: cand['composition'].get(elem, 0) for elem in ELEMENTS}
        comp_data.append(row)
    
    df = pd.DataFrame(comp_data, index=[f"C{c['id']}" for c in candidates])
    
    # 只保留有值的列
    df = df.loc[:, (df > 0.01).any()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
               linewidths=0.5, cbar_kws={'label': 'at.%'})
    ax.set_title('Candidate Alloy Compositions (at.%)', fontsize=14)
    ax.set_ylabel('Candidate')
    ax.set_xlabel('Element')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "composition_heatmap_candidates.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'composition_heatmap_candidates.png'}")


def visualize_all():
    """运行所有可视化"""
    print("="*50)
    print("Candidate Alloy Visualization")
    print("="*50)
    
    try:
        parallel_coordinates_plot()
    except Exception as e:
        print(f"Parallel coordinates error: {e}")
    
    try:
        composition_heatmap()
    except Exception as e:
        print(f"Composition heatmap error: {e}")
    
    try:
        embedding_space_plot()
    except Exception as e:
        print(f"Embedding space plot error: {e}")


if __name__ == '__main__':
    visualize_all()
