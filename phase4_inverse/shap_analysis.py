"""
Phase 4: SHAP解析
分析候选合金中各元素对性能的贡献
"""
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (ELEMENTS, PROCESS_COLS, TASKS, FIGURES_DIR, RESULTS_DIR,
                    EMBEDDINGS_DIR, TOKENIZER_DIR, MODELS_DIR, EMBEDDING_DIM, SEED)


def create_model_wrapper(model, embedding_matrix, vocab, device):
    """
    创建SHAP兼容的模型包装器
    输入：[24维成分 + 6维工艺] -> 输出：蠕变寿命预测
    """
    element_ids = [vocab.get(elem, 0) for elem in ELEMENTS]
    embed_dim = embedding_matrix.shape[1]
    emb_tensor = torch.tensor(embedding_matrix, dtype=torch.float32).to(device)
    
    # 工艺关键词向量
    process_kw_vecs = {}
    for kw in ['solution treatment', 'aging', 'creep']:
        if kw in vocab and vocab[kw] < embedding_matrix.shape[0]:
            process_kw_vecs[kw] = torch.tensor(
                embedding_matrix[vocab[kw]], dtype=torch.float32
            ).to(device)
    
    def predict_fn(X):
        """
        X: numpy array (n_samples, 30) -> [24 compositions + 6 process params]
        Returns: numpy array (n_samples,) creep life predictions
        """
        n_samples = X.shape[0]
        predictions = []
        
        model.eval()
        with torch.no_grad():
            for i in range(n_samples):
                comp = X[i, :24]
                proc = X[i, 24:]
                
                # 成分向量
                comp_vec = torch.zeros(embed_dim, device=device)
                total_w = 0.0
                for j, weight in enumerate(comp):
                    if weight > 0:
                        eid = element_ids[j]
                        if eid < emb_tensor.shape[0]:
                            comp_vec += weight * emb_tensor[eid]
                            total_w += weight
                if total_w > 0:
                    comp_vec /= total_w
                
                # 工艺向量
                proc_action_vec = torch.zeros(embed_dim, device=device)
                n_act = 0
                if proc[0] > 0 and 'solution treatment' in process_kw_vecs:
                    proc_action_vec += process_kw_vecs['solution treatment']
                    n_act += 1
                if proc[2] > 0 and 'aging' in process_kw_vecs:
                    proc_action_vec += process_kw_vecs['aging']
                    n_act += 1
                if proc[4] > 0 and 'creep' in process_kw_vecs:
                    proc_action_vec += process_kw_vecs['creep']
                    n_act += 1
                if n_act > 0:
                    proc_action_vec /= n_act
                
                proc_num = torch.tensor(proc, dtype=torch.float32, device=device)
                proc_num = proc_num / torch.tensor(
                    [1300, 48, 1000, 100, 1100, 500], dtype=torch.float32, device=device
                )
                
                inp = torch.cat([comp_vec, proc_action_vec, proc_num]).unsqueeze(0)
                preds = model(inp)
                
                if 'creep' in preds:
                    predictions.append(preds['creep'].cpu().item())
                else:
                    predictions.append(0.0)
        
        return np.array(predictions)
    
    return predict_fn


def run_shap_analysis():
    """运行SHAP分析"""
    import shap
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print("Loading model...")
    from phase3_multitask.model_multitask import MultiTaskModel
    
    embedding_matrix = np.load(EMBEDDINGS_DIR / "E_pa.npy")
    with open(TOKENIZER_DIR / "vocab.json") as f:
        vocab = json.load(f)
    
    input_dim = EMBEDDING_DIM * 2 + len(PROCESS_COLS)
    model = MultiTaskModel(input_dim).to(device)
    model.load_state_dict(
        torch.load(MODELS_DIR / "multitask_E_pa_fold0.pt",
                  weights_only=True, map_location=device)
    )
    model.eval()
    
    # 创建模型包装器
    predict_fn = create_model_wrapper(model, embedding_matrix, vocab, device)
    
    # 加载蠕变数据作为背景
    master = pd.read_csv(DATA_DIR / "master_table.csv")
    creep_data = master[master['task'] == 'creep'].copy()
    
    # 构建输入矩阵
    X_bg = np.zeros((len(creep_data), 30))
    for i, (_, row) in enumerate(creep_data.iterrows()):
        for j, elem in enumerate(ELEMENTS):
            X_bg[i, j] = row.get(elem, 0.0)
        for j, col in enumerate(PROCESS_COLS):
            X_bg[i, 24+j] = row.get(col, 0.0) if pd.notna(row.get(col)) else 0.0
    
    # 用子集作为背景数据（SHAP计算开销大）
    n_background = min(100, len(X_bg))
    bg_indices = np.random.choice(len(X_bg), n_background, replace=False)
    X_background = X_bg[bg_indices]
    
    # 加载候选合金
    candidates_path = RESULTS_DIR / "inverse_design_candidates.json"
    with open(candidates_path) as f:
        cand_data = json.load(f)
    candidates = cand_data['candidates']
    
    # 构建候选合金输入
    X_candidates = np.zeros((len(candidates), 30))
    for i, cand in enumerate(candidates):
        for j, elem in enumerate(ELEMENTS):
            X_candidates[i, j] = cand['composition'].get(elem, 0.0)
        for j, col in enumerate(PROCESS_COLS):
            X_candidates[i, 24+j] = cand['process'].get(col, 0.0)
    
    # 计算SHAP值
    print("Computing SHAP values...")
    explainer = shap.KernelExplainer(predict_fn, X_background)
    shap_values = explainer.shap_values(X_candidates, nsamples=100)
    
    # 特征名
    feature_names = ELEMENTS + PROCESS_COLS
    
    # 绘制SHAP摘要图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 元素SHAP值柱状图（只看元素部分）
    element_shap = shap_values[:, :24]
    mean_abs_shap = np.abs(element_shap).mean(axis=0)
    
    # 排序
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    top_k = 15  # 显示前15个元素
    
    colors = ['#e74c3c' if element_shap[:, idx].mean() > 0 else '#3498db' 
              for idx in sorted_idx[:top_k]]
    
    ax.barh(range(top_k), mean_abs_shap[sorted_idx[:top_k]], color=colors)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([ELEMENTS[idx] for idx in sorted_idx[:top_k]])
    ax.set_xlabel('Mean |SHAP value| (Contribution to Creep Life)', fontsize=12)
    ax.set_title('Element Importance for Creep Life Prediction\n(Red=Positive, Blue=Negative)', 
                fontsize=14)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_element_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'shap_element_importance.png'}")
    
    # 详细SHAP图（每个候选）
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, (ax, cand) in enumerate(zip(axes, candidates)):
        shap_vals = shap_values[i, :24]
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:8]
        
        colors = ['#e74c3c' if shap_vals[idx] > 0 else '#3498db' for idx in sorted_idx]
        ax.barh(range(len(sorted_idx)), shap_vals[sorted_idx], color=colors)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([ELEMENTS[idx] for idx in sorted_idx], fontsize=8)
        ax.set_title(f"C{cand['id']}", fontsize=10)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.invert_yaxis()
    
    plt.suptitle('SHAP Values per Candidate (Top 8 Elements)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_per_candidate.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'shap_per_candidate.png'}")
    
    # 保存SHAP结果
    shap_results = {
        'feature_names': feature_names,
        'mean_abs_shap': mean_abs_shap.tolist(),
        'shap_values_per_candidate': shap_values.tolist(),
    }
    with open(RESULTS_DIR / "shap_analysis_results.json", 'w') as f:
        json.dump(shap_results, f, indent=2)
    
    print(f"\nSHAP results saved to {RESULTS_DIR / 'shap_analysis_results.json'}")
    
    # 打印关键发现
    print("\nTop elements contributing to creep life:")
    for idx in sorted_idx[:10]:
        print(f"  {ELEMENTS[idx]}: mean|SHAP|={mean_abs_shap[idx]:.4f}, "
              f"avg direction={'positive' if element_shap[:, idx].mean() > 0 else 'negative'}")


if __name__ == '__main__':
    from config import DATA_DIR
    run_shap_analysis()
