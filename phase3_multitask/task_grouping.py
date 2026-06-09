"""
Phase 3: 数据驱动的任务相似度分析与自动分组

两种独立方法互相印证（M2 + M3），输出最终任务分组：

M2 (任务权重向量相似度):
    每个任务独立用 Ridge 回归（分类用LogReg）拟合 y_t = w_t · x + b
    任务相似度 = cos(w_i, w_j)
    物理含义：两个任务对成分/工艺向量的"敏感模式"是否一致

M3 (任务梯度相似度):
    在已训完的 baseline multitask 模型上，对每个任务取一个 batch，
    计算其相对共享层参数的梯度向量 g_t
    任务相似度 = cos(g_i, g_j)
    物理含义：两个任务在共享表征上的"优化方向"是否一致

输出（每种 embedding 一份）:
    output/figures/task_similarity_M2_{embedding}.png    (热图)
    output/figures/task_similarity_M3_{embedding}.png    (热图，可选)
    output/figures/task_dendrogram_M2_{embedding}.png    (层次聚类树状图)
    output/results/task_grouping_{embedding}.json        (自动分组结果)
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (TASKS, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, SEED,
                    MULTITASK_CONFIG)
from phase3_multitask.dataset_alloy import load_alloy_dataset, AlloyDataset
from phase3_multitask.model_multitask import MultiTaskModel


# ============================================================
# 公共工具
# ============================================================

def extract_features_targets_per_task(dataset: AlloyDataset
                                      ) -> Dict[str, Tuple[np.ndarray, np.ndarray, str]]:
    """对每个任务抽取所有样本的 (X, y_norm, task_type)。
    回归任务的 y 已经过 z-score 归一化（来自 dataset.normalize_target）。
    分类任务保持 0/1 标签。"""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}
    task_names = list(TASKS.keys())

    # 收集每个任务的样本
    for i in range(len(dataset)):
        item = dataset[i]
        task_name = item['task_name']
        x = item['input'].numpy()
        # __getitem__ 返回 target 已是 normalized（回归）或 0/1（分类）
        y = float(item['target'].item())
        if task_name not in out:
            out[task_name] = ([], [], TASKS[task_name]['type'])
        out[task_name][0].append(x)
        out[task_name][1].append(y)

    # 转为 numpy
    final = {}
    for tname, (xs, ys, ttype) in out.items():
        if len(xs) == 0:
            continue
        final[tname] = (np.array(xs, dtype=np.float32),
                        np.array(ys, dtype=np.float32),
                        ttype)
    return final


def cosine_similarity_matrix(vectors: Dict[str, np.ndarray]
                             ) -> Tuple[np.ndarray, List[str]]:
    """计算字典中所有向量两两余弦相似度。返回 (similarity_matrix, names)。"""
    names = list(vectors.keys())
    n = len(names)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            vi = vectors[names[i]]
            vj = vectors[names[j]]
            num = float(np.dot(vi, vj))
            den = float(np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12)
            M[i, j] = num / den
    return M, names


def plot_heatmap(matrix: np.ndarray, names: List[str], title: str,
                 save_path: Path):
    """绘制相似度热图"""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    ax.set_title(title)
    # 标注数值
    for i in range(len(names)):
        for j in range(len(names)):
            color = 'white' if abs(matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha='center', va='center', color=color, fontsize=9)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap: {save_path}")


def plot_dendrogram(matrix: np.ndarray, names: List[str], title: str,
                    save_path: Path):
    """绘制层次聚类树状图（基于 1 - similarity 作为距离）"""
    # 距离矩阵：1 - sim，截断到 [0, 2]
    dist = 1.0 - matrix
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)
    # 转为压缩距离向量
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='ward')

    fig, ax = plt.subplots(figsize=(9, 5))
    dendrogram(Z, labels=names, ax=ax, leaf_rotation=30,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(title)
    ax.set_ylabel('Distance (1 - similarity)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved dendrogram: {save_path}")
    return Z


def auto_group_from_linkage(Z: np.ndarray, names: List[str],
                            n_clusters_range: Tuple[int, int] = (2, 4),
                            distance_matrix: np.ndarray = None
                            ) -> Tuple[Dict[str, List[str]], int]:
    """从层次聚类结果自动选最优分组数（用 silhouette score）。

    返回 ({group_name: [tasks]}, best_k)
    """
    n = len(names)
    best_k, best_score = 2, -1.0
    if distance_matrix is None:
        distance_matrix = np.zeros((n, n))

    for k in range(n_clusters_range[0], min(n_clusters_range[1] + 1, n)):
        labels = fcluster(Z, t=k, criterion='maxclust')
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(distance_matrix, labels,
                                     metric='precomputed')
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_k = k

    final_labels = fcluster(Z, t=best_k, criterion='maxclust')
    groups: Dict[str, List[str]] = {}
    for tname, lbl in zip(names, final_labels):
        gkey = f"G{int(lbl)}"
        groups.setdefault(gkey, []).append(tname)

    print(f"  Best k = {best_k}, silhouette = {best_score:.4f}")
    return groups, best_k


# ============================================================
# M2: 任务权重向量相似度
# ============================================================

def compute_M2(dataset: AlloyDataset) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """对每个任务用 Ridge（分类用 LogReg）拟合 y = w·x + b，
    返回 (similarity_matrix, task_names, weight_vectors)。"""
    print("\n[M2] Computing task weight vectors via Ridge / LogReg ...")
    task_data = extract_features_targets_per_task(dataset)

    weight_vectors: Dict[str, np.ndarray] = {}
    for task_name, (X, y, ttype) in task_data.items():
        # 标准化特征（每个任务独立 scaler，但共享分布形态）
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        if ttype == 'regression':
            model = Ridge(alpha=1.0, random_state=SEED)
            model.fit(Xs, y)
            w = np.asarray(model.coef_, dtype=np.float32).flatten()
        else:
            # 二分类：LogReg
            try:
                model = LogisticRegression(C=1.0, max_iter=2000,
                                           random_state=SEED)
                model.fit(Xs, y.astype(int))
                w = np.asarray(model.coef_, dtype=np.float32).flatten()
            except Exception as e:
                print(f"  [WARN] LogReg failed for {task_name}: {e}")
                w = np.zeros(X.shape[1], dtype=np.float32)
        # 单位化
        norm = np.linalg.norm(w) + 1e-12
        weight_vectors[task_name] = w / norm
        print(f"  {task_name:15s}: |w|={np.linalg.norm(w):.4f}, n_samples={len(y)}")

    M, names = cosine_similarity_matrix(weight_vectors)
    return M, names, weight_vectors


# ============================================================
# M3: 任务梯度相似度（基于已训 baseline 模型）
# ============================================================

def compute_M3(dataset: AlloyDataset, embedding_name: str,
               n_folds: int = 1, batch_size: int = 256
               ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """对每个任务，在已训 baseline 上计算其相对共享层参数的梯度向量，
    返回 (similarity_matrix, task_names, gradient_vectors)。"""
    print("\n[M3] Computing task gradient vectors on baseline multitask model ...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = dataset.input_dim
    task_names_full = list(TASKS.keys())

    # 预先按任务收集样本索引
    task_indices: Dict[str, List[int]] = {t: [] for t in task_names_full}
    for i in range(len(dataset)):
        tn = dataset.data.iloc[i]['task']
        if tn in task_indices:
            task_indices[tn].append(i)

    # 跨折累积梯度向量（取多折平均，提升稳定性）
    grad_vectors_acc: Dict[str, List[np.ndarray]] = {t: [] for t in task_names_full}
    folds_used = 0

    for fold in range(n_folds):
        ckpt = MODELS_DIR / f"multitask_{embedding_name}_fold{fold}.pt"
        if not ckpt.exists():
            print(f"  [SKIP] Missing checkpoint: {ckpt}")
            continue
        model = MultiTaskModel(input_dim).to(device)
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()  # 仅用于梯度计算，dropout/BN固定
        folds_used += 1
        print(f"  Loaded {ckpt.name}")

        # 共享层参数
        backbone_params = list(model.backbone.parameters())

        for task_name in task_names_full:
            indices = task_indices[task_name]
            if len(indices) == 0:
                continue
            rng = np.random.RandomState(SEED + fold)
            chosen = rng.choice(indices, size=min(batch_size, len(indices)),
                                replace=False)

            xs, ys = [], []
            for idx in chosen:
                item = dataset[int(idx)]
                xs.append(item['input'])
                ys.append(item['target'])
            x_batch = torch.stack(xs).to(device)
            y_batch = torch.stack(ys).to(device)
            tid_val = task_names_full.index(task_name)
            tid_batch = torch.full((len(chosen),), tid_val,
                                   dtype=torch.long, device=device)

            # 前向 + 计算单任务损失
            preds = model(x_batch, tid_batch)
            loss_dict = model.compute_loss(preds, y_batch, tid_batch)
            loss = loss_dict['total_loss']

            model.zero_grad()
            loss.backward()

            grads = []
            for p in backbone_params:
                if p.grad is None:
                    grads.append(np.zeros(p.numel(), dtype=np.float32))
                else:
                    grads.append(p.grad.detach().cpu().numpy().flatten())
            g = np.concatenate(grads).astype(np.float32)
            n = np.linalg.norm(g) + 1e-12
            grad_vectors_acc[task_name].append(g / n)

    if folds_used == 0:
        print("  [WARN] No baseline checkpoints found, skipping M3.")
        return None, [], {}

    # 多折平均
    grad_vectors: Dict[str, np.ndarray] = {}
    for t, lst in grad_vectors_acc.items():
        if len(lst) == 0:
            continue
        avg = np.mean(np.stack(lst, axis=0), axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-12)
        grad_vectors[t] = avg

    M, names = cosine_similarity_matrix(grad_vectors)
    return M, names, grad_vectors


# ============================================================
# 主流程
# ============================================================

def run_task_grouping(embedding_name: str = 'E_pa', skip_m3: bool = False,
                      n_folds_for_m3: int = 1):
    print(f"\n{'='*60}")
    print(f"Task grouping analysis ({embedding_name})")
    print(f"{'='*60}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = load_alloy_dataset(embedding_name)
    print(f"Dataset size: {len(dataset)}, input_dim={dataset.input_dim}")

    # ---------------- M2 ----------------
    M2_mat, names2, _ = compute_M2(dataset)
    plot_heatmap(M2_mat, names2,
                 f'M2: Task Weight-Vector Similarity ({embedding_name})',
                 FIGURES_DIR / f'task_similarity_M2_{embedding_name}.png')

    dist2 = 1.0 - M2_mat
    np.fill_diagonal(dist2, 0.0)
    dist2 = np.clip(dist2, 0.0, 2.0)
    Z2 = plot_dendrogram(M2_mat, names2,
                         f'M2 Hierarchical Clustering ({embedding_name})',
                         FIGURES_DIR / f'task_dendrogram_M2_{embedding_name}.png')
    groups_M2, best_k_M2 = auto_group_from_linkage(
        Z2, names2, n_clusters_range=(2, 4), distance_matrix=dist2)

    # ---------------- M3 ----------------
    groups_M3, best_k_M3, M3_mat = None, None, None
    if not skip_m3:
        result = compute_M3(dataset, embedding_name, n_folds=n_folds_for_m3)
        if result[0] is not None:
            M3_mat, names3, _ = result
            plot_heatmap(M3_mat, names3,
                         f'M3: Task Gradient Similarity ({embedding_name})',
                         FIGURES_DIR / f'task_similarity_M3_{embedding_name}.png')
            dist3 = 1.0 - M3_mat
            np.fill_diagonal(dist3, 0.0)
            dist3 = np.clip(dist3, 0.0, 2.0)
            Z3 = plot_dendrogram(M3_mat, names3,
                                 f'M3 Hierarchical Clustering ({embedding_name})',
                                 FIGURES_DIR / f'task_dendrogram_M3_{embedding_name}.png')
            groups_M3, best_k_M3 = auto_group_from_linkage(
                Z3, names3, n_clusters_range=(2, 4), distance_matrix=dist3)

    # ---------------- 结果保存 ----------------
    out = {
        'embedding': embedding_name,
        'M2': {
            'similarity_matrix': M2_mat.tolist(),
            'task_names': names2,
            'best_k': int(best_k_M2),
            'groups': groups_M2,
        }
    }
    if M3_mat is not None:
        out['M3'] = {
            'similarity_matrix': M3_mat.tolist(),
            'task_names': names2,  # M3 任务顺序与 M2 相同（来自 TASKS）
            'best_k': int(best_k_M3),
            'groups': groups_M3,
        }

    # 决定最终采用的分组：优先 M3（梯度更直接反映共享表征），否则用 M2
    final_groups = groups_M3 if groups_M3 is not None else groups_M2
    out['final_groups'] = final_groups

    save_path = RESULTS_DIR / f'task_grouping_{embedding_name}.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved grouping JSON: {save_path}")

    print(f"\n{'='*60}")
    print(f"FINAL GROUPS ({embedding_name}):")
    for gname, tlist in final_groups.items():
        print(f"  {gname}: {tlist}")
    print(f"{'='*60}")
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, default='E_pa',
                        choices=['E_pa', 'E_base', 'E_w2v'])
    parser.add_argument('--skip_m3', action='store_true',
                        help='Skip M3 (gradient similarity) when no baseline checkpoint exists')
    parser.add_argument('--n_folds_m3', type=int, default=1,
                        help='Number of folds to average for M3 gradient (1-5)')
    args = parser.parse_args()
    run_task_grouping(args.embedding, skip_m3=args.skip_m3,
                      n_folds_for_m3=args.n_folds_m3)
