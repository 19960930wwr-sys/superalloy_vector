"""
Phase 3: 模型评估与对比
对比：单任务 vs 多任务，E_base vs E_pa vs E_w2v
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TASKS, RESULTS_DIR, FIGURES_DIR


def load_results():
    """加载所有结果文件（多任务NN + 分组多任务NN + 单任务NN + ML基线）"""
    results = {}

    # 1. 神经网络多任务 / 单任务
    embedding_names = ['E_pa', 'E_base', 'E_w2v', 'E_attr', 'E_proc']

    for prefix in ['multitask', 'singletask']:
        for emb in embedding_names:
            key = f"{prefix}_{emb}"
            path = RESULTS_DIR / f"{key}_results.json"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    results[key] = json.load(f)

    # 2. 分组多任务（文件名格式：grouped-{group}_{embedding}_results.json）
    for path in RESULTS_DIR.glob("grouped-*_results.json"):
        stem = path.stem
        if stem.endswith("_results"):
            stem = stem[: -len("_results")]
        # stem = grouped-{group}_{emb}
        emb_found = None
        for emb in embedding_names:
            if stem.endswith('_' + emb):
                emb_found = emb
                model_id = stem[: -len('_' + emb)]  # grouped-{group}
                break
        if emb_found is None:
            continue
        key = f"{model_id}_{emb_found}"
        with open(path, 'r', encoding='utf-8') as f:
            results[key] = json.load(f)

    # 3. ML基线（文件名格式：ml-{algo}_{embedding}_results.json）
    for path in RESULTS_DIR.glob("ml-*_results.json"):
        stem = path.stem
        if stem.endswith("_results"):
            stem = stem[: -len("_results")]
        emb_found = None
        for emb in embedding_names:
            if stem.endswith('_' + emb):
                emb_found = emb
                model_id = stem[: -len('_' + emb)]  # ml-RFR
                break
        if emb_found is None:
            continue
        key = f"{model_id}_{emb_found}"
        with open(path, 'r', encoding='utf-8') as f:
            results[key] = json.load(f)

    return results


def create_comparison_table(results: dict) -> pd.DataFrame:
    """创建对比表格"""
    rows = []
    
    for config_name, task_results in results.items():
        parts = config_name.split('_', 1)
        model_type = parts[0]  # multitask or singletask
        embedding = parts[1] if len(parts) > 1 else ''
        
        for task_name, metrics in task_results.items():
            row = {
                'model': model_type,
                'embedding': embedding,
                'task': task_name,
                'task_type': TASKS[task_name]['type'] if task_name in TASKS else 'unknown',
            }
            row.update(metrics)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def plot_comparison(df: pd.DataFrame):
    """绘制对比图"""
    # 回归任务对比 (R²)
    reg_df = df[df['task_type'] == 'regression'].copy()
    if not reg_df.empty and 'r2_mean' in reg_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        reg_df['label'] = reg_df['model'] + '\n' + reg_df['embedding']
        pivot = reg_df.pivot_table(index='task', columns='label', values='r2_mean')
        
        pivot.plot(kind='bar', ax=ax, width=0.7)
        ax.set_ylabel('R² Score')
        ax.set_title('Regression Tasks: R² Comparison')
        ax.legend(title='Model Configuration', bbox_to_anchor=(1.05, 1))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "comparison_regression_r2.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: comparison_regression_r2.png")
    
    # 分类任务对比 (F1/AUC)
    cls_df = df[df['task_type'] == 'classification'].copy()
    if not cls_df.empty and 'f1_mean' in cls_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax, metric in zip(axes, ['f1_mean', 'auc_mean']):
            if metric in cls_df.columns:
                cls_df['label'] = cls_df['model'] + '\n' + cls_df['embedding']
                pivot = cls_df.pivot_table(index='task', columns='label', values=metric)
                pivot.plot(kind='bar', ax=ax, width=0.7)
                ax.set_ylabel(metric.replace('_mean', '').upper())
                ax.set_title(f'Classification: {metric.replace("_mean", "").upper()}')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "comparison_classification.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: comparison_classification.png")


def build_best_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """对每个 (task, embedding) 组合，选指标最优的模型。

    输出一张宽表，便于论文中引用 "每个任务×每个 embedding 下应选哪个训练方案"。
    评选规则：回归看 r2_mean 最大，分类看 f1_mean 最大。
    """
    records = []
    embeddings = sorted(df['embedding'].dropna().unique().tolist())
    for task_name in TASKS:
        ttype = TASKS[task_name]['type']
        metric_col = 'r2_mean' if ttype == 'regression' else 'f1_mean'
        for emb in embeddings:
            sub = df[(df['task'] == task_name) & (df['embedding'] == emb)].copy()
            if sub.empty or metric_col not in sub.columns:
                continue
            sub = sub.dropna(subset=[metric_col])
            if sub.empty:
                continue
            best_row = sub.loc[sub[metric_col].idxmax()]
            # 与同任务+embedding 下的 'multitask' 基准对比
            base = sub[sub['model'] == 'multitask']
            base_val = float(base[metric_col].iloc[0]) if not base.empty else float('nan')
            best_val = float(best_row[metric_col])
            records.append({
                'task': task_name,
                'task_type': ttype,
                'embedding': emb,
                'best_model': best_row['model'],
                'metric': metric_col.replace('_mean', ''),
                'best_value': round(best_val, 4),
                'multitask_value': round(base_val, 4) if base_val == base_val else None,
                'delta_vs_multitask': round(best_val - base_val, 4) if base_val == base_val else None,
            })
    best_df = pd.DataFrame(records)
    return best_df


def evaluate_all():
    """运行完整评估"""
    print("Loading results...")
    results = load_results()
    
    if not results:
        print("No results found. Please run training first.")
        return
    
    print(f"Found {len(results)} result files")
    
    # 创建对比表
    df = create_comparison_table(results)
    
    # 打印对比表
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    # 排序优先级：multitask -> grouped-* -> singletask -> ml-*
    def model_sort_key(model_name: str) -> int:
        if model_name == 'multitask':
            return 0
        if model_name.startswith('grouped-'):
            return 1
        if model_name == 'singletask':
            return 2
        if model_name.startswith('ml-'):
            return 3
        return 4

    for task_name in TASKS:
        task_df = df[df['task'] == task_name].copy()
        if task_df.empty:
            continue

        # 排序：先按 model_sort_key，再按 embedding
        task_df['__sort_model'] = task_df['model'].apply(model_sort_key)
        task_df = task_df.sort_values(['__sort_model', 'model', 'embedding']).drop(columns=['__sort_model'])

        print(f"\n--- {task_name} ({TASKS[task_name]['type']}) ---")
        if TASKS[task_name]['type'] == 'regression':
            cols = ['model', 'embedding', 'r2_mean', 'r2_std', 'rmse_mean', 'rmse_std']
        else:
            cols = ['model', 'embedding', 'f1_mean', 'f1_std', 'auc_mean', 'auc_std']

        available_cols = [c for c in cols if c in task_df.columns]
        print(task_df[available_cols].to_string(index=False))
    
    # 绘图
    plot_comparison(df)
    
    # 保存完整对比表
    df.to_csv(RESULTS_DIR / "full_comparison_table.csv", index=False)
    print(f"\nFull comparison table saved to {RESULTS_DIR / 'full_comparison_table.csv'}")

    # 生成"任务×embedding 最佳模型"映射表
    best_df = build_best_model_table(df)
    if not best_df.empty:
        best_path = RESULTS_DIR / "best_model_per_task_embedding.csv"
        best_df.to_csv(best_path, index=False)
        print("\n" + "=" * 80)
        print("BEST MODEL PER (task, embedding)")
        print("=" * 80)
        print(best_df.to_string(index=False))
        print(f"\nBest-model map saved to {best_path}")


if __name__ == '__main__':
    evaluate_all()
