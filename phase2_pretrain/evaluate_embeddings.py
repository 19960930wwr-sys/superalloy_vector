"""
Phase 2: 词向量内在评估
- t-SNE/UMAP降维可视化
- 元素类比推理
- 属性预测精度
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import ELEMENTS, EMBEDDINGS_DIR, FIGURES_DIR, RESULTS_DIR, TOKENIZER_DIR
from phase2_pretrain.element_attributes import get_element_attr_matrix, get_element_attr_df


# 元素分组（物理冶金学意义）
ELEMENT_GROUPS = {
    "gamma_prime_former": ['Al', 'Ti', 'Nb', 'Ta'],       # γ'形成元素
    "solid_solution": ['Mo', 'W', 'Re', 'Ru'],            # 固溶强化元素
    "grain_boundary": ['B', 'C', 'Zr', 'Hf'],             # 晶界强化元素
    "base_element": ['Ni', 'Co'],                          # 基体元素
    "other": ['Cr', 'Fe', 'Mn', 'Si', 'V', 'Ge', 'Ir', 'La', 'Y', 'Mg'],
}

GROUP_COLORS = {
    "gamma_prime_former": '#e74c3c',
    "solid_solution": '#3498db',
    "grain_boundary": '#2ecc71',
    "base_element": '#f39c12',
    "other": '#95a5a6',
}

GROUP_LABELS = {
    "gamma_prime_former": r"$\gamma'$ size",
    "solid_solution": "Solid Solution",
    "grain_boundary": "Grain Boundary",
    "base_element": "Base Element",
    "other": "Other",
}


def load_embeddings(name: str) -> tuple:
    """加载词向量和对应词表"""
    emb_path = EMBEDDINGS_DIR / f"{name}.npy"
    embeddings = np.load(emb_path)
    
    vocab_path = TOKENIZER_DIR / "vocab.json"
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    return embeddings, vocab


def get_element_vectors(embeddings: np.ndarray, vocab: dict) -> dict:
    """提取元素词向量"""
    element_vectors = {}
    for elem in ELEMENTS:
        if elem in vocab:
            idx = vocab[elem]
            if idx < embeddings.shape[0]:
                element_vectors[elem] = embeddings[idx]
    return element_vectors


def visualize_element_clustering(embeddings_dict: dict, save_name: str = "element_clustering"):
    """
    t-SNE可视化元素聚类
    Args:
        embeddings_dict: {'E_base': {...}, 'E_pa': {...}} 各模型的元素向量
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
    })

    n_models = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(7.8*n_models, 7.2))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, elem_vectors) in zip(axes, embeddings_dict.items()):
        # 准备数据
        elements_found = [e for e in ELEMENTS if e in elem_vectors]
        vectors = np.array([elem_vectors[e] for e in elements_found])
        
        # t-SNE降维
        if len(elements_found) > 5:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(elements_found)-1))
            coords = tsne.fit_transform(vectors)
        else:
            coords = vectors[:, :2]
        
        # 绘图
        for elem, (x, y) in zip(elements_found, coords):
            # 确定颜色
            color = GROUP_COLORS['other']
            for group, members in ELEMENT_GROUPS.items():
                if elem in members:
                    color = GROUP_COLORS[group]
                    break
            ax.scatter(x, y, c=color, s=150, zorder=5, edgecolor="white", linewidth=0.7)
            ax.annotate(elem, (x, y), textcoords="offset points", 
                       xytext=(6, 6), fontsize=15, fontweight='bold')
        
        ax.set_title(f'{model_name}', pad=12)
        ax.set_xlabel('t-SNE dim 1', labelpad=8)
        ax.set_ylabel('t-SNE dim 2', labelpad=8)
        ax.grid(True, alpha=0.28)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=GROUP_LABELS.get(group, group.replace('_', ' ').title()))
                      for group, color in GROUP_COLORS.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, 0.005), frameon=False,
               handlelength=1.2, columnspacing=1.6)
    
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    plt.savefig(FIGURES_DIR / f"{save_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / save_name}.png")


def analogy_test(elem_vectors: dict):
    """
    元素类比推理测试
    例如: Al - gamma' + solid_solution ≈ Mo?
    """
    # 定义类比对
    analogies = [
        ('Al', 'Ti', 'Mo', 'W'),       # Al:Ti :: Mo:W (同组元素)
        ('Al', 'Ni', 'Ti', 'Ni'),       # Al形成gamma' :: Ti也形成gamma'
        ('B', 'C', 'Zr', 'Hf'),         # 晶界强化元素对
        ('Mo', 'W', 'Re', 'Ru'),        # 固溶强化元素
    ]
    
    results = []
    print("\nAnalogy Test (a:b :: c:?):")
    for a, b, c, expected in analogies:
        if all(e in elem_vectors for e in [a, b, c, expected]):
            # target = vec(b) - vec(a) + vec(c)
            target = elem_vectors[b] - elem_vectors[a] + elem_vectors[c]
            
            # 找最近邻（排除a, b, c）
            best_sim = -float('inf')
            best_elem = None
            for elem, vec in elem_vectors.items():
                if elem in [a, b, c]:
                    continue
                sim = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec) + 1e-8)
                if sim > best_sim:
                    best_sim = sim
                    best_elem = elem
            
            correct = best_elem == expected
            results.append(correct)
            print(f"  {a}:{b} :: {c}:? -> predicted: {best_elem} "
                  f"(expected: {expected}) {'✓' if correct else '✗'}")
    
    accuracy = sum(results) / len(results) if results else 0
    print(f"  Accuracy: {accuracy:.2%}")
    return accuracy


def attribute_prediction_test(elem_vectors: dict):
    """
    属性预测精度测试
    冻结词向量，用线性层预测元素属性（留一法）
    """
    attr_df = get_element_attr_df()
    
    # 只用有词向量的元素
    elements_found = [e for e in ELEMENTS if e in elem_vectors]
    X = np.array([elem_vectors[e] for e in elements_found])
    Y = attr_df.loc[elements_found].values  # (N, 13)
    
    # 留一法交叉验证
    loo = LeaveOneOut()
    all_r2 = []
    
    for attr_idx, attr_name in enumerate(attr_df.columns):
        y = Y[:, attr_idx]
        preds = np.zeros_like(y)
        
        for train_idx, test_idx in loo.split(X):
            model = Ridge(alpha=1.0)
            model.fit(X[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X[test_idx])
        
        r2 = r2_score(y, preds)
        all_r2.append(r2)
    
    avg_r2 = np.mean(all_r2)
    print(f"\n  Attribute Prediction (Leave-One-Out):")
    print(f"  Average R²: {avg_r2:.4f}")
    for name, r2 in zip(attr_df.columns, all_r2):
        print(f"    {name}: R² = {r2:.4f}")
    
    return avg_r2, dict(zip(attr_df.columns, all_r2))


def evaluate_all_embeddings():
    """评估所有词向量"""
    results = {}
    embeddings_to_eval = {}
    
    # 尝试加载各类词向量
    for name in ['E_base', 'E_pa']:
        try:
            emb, vocab = load_embeddings(name)
            elem_vecs = get_element_vectors(emb, vocab)
            if elem_vecs:
                embeddings_to_eval[name] = elem_vecs
                print(f"\n{'='*50}")
                print(f"Evaluating: {name} ({len(elem_vecs)} elements found)")
                print(f"{'='*50}")
                
                # 类比测试
                analogy_acc = analogy_test(elem_vecs)
                
                # 属性预测
                avg_r2, attr_r2s = attribute_prediction_test(elem_vecs)
                
                results[name] = {
                    'analogy_accuracy': analogy_acc,
                    'avg_attr_r2': avg_r2,
                    'attr_r2_details': attr_r2s,
                }
        except FileNotFoundError:
            print(f"  {name} not found, skipping...")
    
    # Word2Vec词向量（不同格式）
    try:
        w2v_vocab_path = EMBEDDINGS_DIR / "E_w2v_vocab.json"
        w2v_vec_path = EMBEDDINGS_DIR / "E_w2v_vectors.npy"
        if w2v_vocab_path.exists() and w2v_vec_path.exists():
            with open(w2v_vocab_path, 'r', encoding='utf-8') as f:
                w2v_vocab = json.load(f)
            w2v_vectors = np.load(w2v_vec_path)
            
            elem_vecs = {}
            for elem in ELEMENTS:
                if elem in w2v_vocab:
                    elem_vecs[elem] = w2v_vectors[w2v_vocab[elem]]
            
            if elem_vecs:
                embeddings_to_eval['E_w2v'] = elem_vecs
                print(f"\n{'='*50}")
                print(f"Evaluating: E_w2v ({len(elem_vecs)} elements found)")
                print(f"{'='*50}")
                
                analogy_acc = analogy_test(elem_vecs)
                avg_r2, attr_r2s = attribute_prediction_test(elem_vecs)
                
                results['E_w2v'] = {
                    'analogy_accuracy': analogy_acc,
                    'avg_attr_r2': avg_r2,
                    'attr_r2_details': attr_r2s,
                }
    except Exception as e:
        print(f"  W2V evaluation error: {e}")
    
    # 可视化对比
    if embeddings_to_eval:
        visualize_element_clustering(embeddings_to_eval)
    
    # 保存结果
    # Convert numpy values to Python floats for JSON serialization
    json_results = {}
    for k, v in results.items():
        json_results[k] = {
            'analogy_accuracy': float(v['analogy_accuracy']),
            'avg_attr_r2': float(v['avg_attr_r2']),
            'attr_r2_details': {ak: float(av) for ak, av in v['attr_r2_details'].items()},
        }
    
    with open(RESULTS_DIR / "embedding_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nResults saved to {RESULTS_DIR / 'embedding_evaluation.json'}")
    return results


if __name__ == '__main__':
    evaluate_all_embeddings()
