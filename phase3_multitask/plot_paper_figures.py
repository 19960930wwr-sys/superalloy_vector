"""
Phase 3 论文图表生成
读取 output/results/ 下的所有 *_results.json + task_grouping_*.json
生成论文用的 4 张核心图：

1. fig_radar_best_per_task.png      - 雷达图：E_pa vs E_base 在 7 任务上各自最佳模型的性能
2. fig_heatmap_M3_compare.png       - 热图：M3 任务梯度相似度，左右对比 E_pa vs E_base
3. fig_grouped_vs_full_bar.png      - 柱状图：每任务 grouped MT vs full MT vs singletask
4. fig_delta_heatmap.png            - 热图：所有模型相对 full multitask 基线的 R²/F1 提升
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TASKS, RESULTS_DIR, FIGURES_DIR


TASK_ORDER = ['density', 'creep', 'liquidus', 'phase_class',
              'size', 'solidus', 'solvus']
TASK_LABELS = {
    'density': 'Density',
    'creep': 'Creep Life',
    'liquidus': 'Liquidus T',
    'phase_class': 'Phase Class',
    'size': r"$\gamma'$ Size",
    'solidus': 'Solidus T',
    'solvus': r"$\gamma'$ Solvus T",
}
EMB_COLORS = {'E_pa': '#d62728', 'E_base': '#1f77b4', 'E_w2v': '#2ca02c'}


# =====================================================
# 数据载入
# =====================================================

def load_full_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / 'full_comparison_table.csv')
    return df


def load_best_table() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / 'best_model_per_task_embedding.csv')


def get_metric(df: pd.DataFrame, model: str, embedding: str, task: str):
    """从 full table 取出 (model, embedding, task) 的核心指标
    回归: r2_mean; 分类: f1_mean
    """
    sub = df[(df['model'] == model) &
             (df['embedding'] == embedding) &
             (df['task'] == task)]
    if sub.empty:
        return None
    ttype = TASKS[task]['type']
    col = 'r2_mean' if ttype == 'regression' else 'f1_mean'
    val = sub.iloc[0].get(col)
    return None if pd.isna(val) else float(val)


# =====================================================
# 1. 雷达图：E_pa vs E_base 的最佳模型性能
# =====================================================

def plot_radar_best(best_df: pd.DataFrame, full_df: pd.DataFrame,
                    out_path: Path):
    angles = np.linspace(0, 2 * np.pi, len(TASK_ORDER), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    series = {
        'E_pa (best)': [best_df[(best_df['task'] == t) &
                                (best_df['embedding'] == 'E_pa')]
                        ['best_value'].iloc[0] for t in TASK_ORDER],
        'E_base (best)': [best_df[(best_df['task'] == t) &
                                  (best_df['embedding'] == 'E_base')]
                          ['best_value'].iloc[0] for t in TASK_ORDER],
        'E_pa (full MT)': [get_metric(full_df, 'multitask', 'E_pa', t)
                           for t in TASK_ORDER],
        'E_base (full MT)': [get_metric(full_df, 'multitask', 'E_base', t)
                             for t in TASK_ORDER],
    }
    style = {
        'E_pa (best)':       {'color': '#d62728', 'lw': 2.4, 'ls': '-'},
        'E_base (best)':     {'color': '#1f77b4', 'lw': 2.4, 'ls': '-'},
        'E_pa (full MT)':    {'color': '#d62728', 'lw': 1.5, 'ls': '--', 'alpha': 0.6},
        'E_base (full MT)':  {'color': '#1f77b4', 'lw': 1.5, 'ls': '--', 'alpha': 0.6},
    }
    for label, vals in series.items():
        v = vals + vals[:1]
        st = style[label]
        ax.plot(angles, v, label=label, **st)
        if 'best' in label:
            ax.fill(angles, v, color=st['color'], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.85', '0.9', '0.95'],
                       fontsize=9, color='gray')
    ax.set_rlabel_position(0)
    # 突出 0.85 阈值线
    ax.plot(angles, [0.85] * len(angles), color='black', ls=':', lw=1, alpha=0.5)
    ax.set_title('Best model per task: E_pa vs E_base (R² for reg / F1 for cls)',
                 fontsize=13, pad=20)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path.name}")


# =====================================================
# 2. M3 任务梯度相似度热图（双 embedding 并排）
# =====================================================

def plot_m3_compare(out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, emb in zip(axes, ['E_pa', 'E_base']):
        path = RESULTS_DIR / f'task_grouping_{emb}.json'
        if not path.exists():
            ax.set_visible(False)
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'M3' not in data:
            ax.set_visible(False)
            continue
        M = np.array(data['M3']['similarity_matrix'])
        names = data['M3']['task_names']
        labels = [TASK_LABELS.get(n, n) for n in names]
        sns.heatmap(M, ax=ax, annot=True, fmt='.2f',
                    xticklabels=labels, yticklabels=labels,
                    cmap='RdBu_r', center=0, vmin=-0.5, vmax=1.0,
                    square=True, cbar_kws={'label': 'cosine similarity'})
        ax.set_title(f'M3 Gradient Similarity ({emb})\nbest_k = {data["M3"]["best_k"]}',
                     fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path.name}")


# =====================================================
# 3. grouped MT vs full MT vs singletask 柱状图
# =====================================================

def plot_grouped_vs_full(full_df: pd.DataFrame, out_path: Path):
    """对每个 (task, embedding)，画 4 条柱：
    full-MT / grouped-MT (best of) / singletask / best-ml
    """
    rows = []
    for emb in ['E_pa', 'E_base']:
        for t in TASK_ORDER:
            ttype = TASKS[t]['type']
            metric = 'r2_mean' if ttype == 'regression' else 'f1_mean'
            sub = full_df[(full_df['embedding'] == emb) & (full_df['task'] == t)]
            if sub.empty:
                continue
            full_mt = sub[sub['model'] == 'multitask'][metric]
            st = sub[sub['model'] == 'singletask'][metric]
            grp = sub[sub['model'].str.startswith('grouped-')][metric]
            ml = sub[sub['model'].str.startswith('ml-')][metric]
            rows.append({
                'task': t, 'embedding': emb,
                'full_MT': float(full_mt.iloc[0]) if not full_mt.empty else np.nan,
                'grouped_MT': float(grp.max()) if not grp.empty else np.nan,
                'singletask': float(st.iloc[0]) if not st.empty else np.nan,
                'best_ML': float(ml.max()) if not ml.empty else np.nan,
            })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    width = 0.2
    x = np.arange(len(TASK_ORDER))

    for ax, emb in zip(axes, ['E_pa', 'E_base']):
        sub = df[df['embedding'] == emb].set_index('task').reindex(TASK_ORDER)
        ax.bar(x - 1.5 * width, sub['full_MT'], width, label='full multitask', color='#d62728')
        ax.bar(x - 0.5 * width, sub['grouped_MT'], width, label='grouped multitask', color='#ff7f0e')
        ax.bar(x + 0.5 * width, sub['singletask'], width, label='singletask NN', color='#2ca02c')
        ax.bar(x + 1.5 * width, sub['best_ML'], width, label='best ML baseline', color='#1f77b4')
        ax.axhline(y=0.85, color='black', ls=':', lw=1, alpha=0.5, label='R²=0.85 target')
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER],
                           rotation=30, ha='right', fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel('R² (regression) / F1 (classification)')
        ax.set_title(f'{emb}', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        if emb == 'E_pa':
            ax.legend(loc='lower right', fontsize=9)
    fig.suptitle('Per-task comparison: grouped MT vs full MT vs singletask vs ML',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path.name}")


# =====================================================
# 4. Delta heatmap：所有模型相对 full multitask 基线的指标提升
# =====================================================

def plot_delta_heatmap(full_df: pd.DataFrame, out_path: Path):
    """构造 (task × model) 矩阵：每格 = 该模型 - full-MT(同 embedding)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, emb in zip(axes, ['E_pa', 'E_base']):
        sub = full_df[full_df['embedding'] == emb]
        models = sorted(sub['model'].unique().tolist())
        # 排序：multitask -> grouped-* -> singletask -> ml-*
        def rk(m):
            if m == 'multitask': return (0, m)
            if m.startswith('grouped-'): return (1, m)
            if m == 'singletask': return (2, m)
            if m.startswith('ml-'): return (3, m)
            return (4, m)
        models = sorted(models, key=rk)

        mat = np.full((len(TASK_ORDER), len(models)), np.nan)
        for i, t in enumerate(TASK_ORDER):
            ttype = TASKS[t]['type']
            metric = 'r2_mean' if ttype == 'regression' else 'f1_mean'
            base_row = sub[(sub['task'] == t) & (sub['model'] == 'multitask')]
            if base_row.empty:
                continue
            base = float(base_row[metric].iloc[0])
            for j, m in enumerate(models):
                rec = sub[(sub['task'] == t) & (sub['model'] == m)]
                if rec.empty:
                    continue
                v = rec[metric].iloc[0]
                if pd.isna(v):
                    continue
                mat[i, j] = float(v) - base

        labels_y = [TASK_LABELS[t] for t in TASK_ORDER]
        sns.heatmap(mat, ax=ax, annot=True, fmt='.3f',
                    xticklabels=models, yticklabels=labels_y,
                    cmap='RdBu_r', center=0, vmin=-0.30, vmax=0.30,
                    cbar_kws={'label': 'Δ vs full multitask'},
                    linewidths=0.4, linecolor='white')
        ax.set_title(f'{emb}: model gain over full multitask', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path.name}")


# =====================================================
# main
# =====================================================

def plot_radar_best(best_df: pd.DataFrame, full_df: pd.DataFrame,
                    out_path: Path):
    """Redesigned Figure 6: full-MT radar plus direct Epa-minus-Ebase gains."""
    def mt_values(embedding: str):
        return [get_metric(full_df, 'multitask', embedding, t)
                for t in TASK_ORDER]

    pa_mt = np.array(mt_values('E_pa'))
    base_mt = np.array(mt_values('E_base'))
    mt_delta = pa_mt - base_mt

    angles = np.linspace(0, 2 * np.pi, len(TASK_ORDER), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(15.6, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.72)
    ax = fig.add_subplot(gs[0, 0], projection='polar')
    ax_delta = fig.add_subplot(gs[0, 1])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    radar_series = {
        r'$E_{\mathrm{pa}}$ full MT': (pa_mt.tolist(), '#c62828'),
        r'$E_{\mathrm{base}}$ full MT': (base_mt.tolist(), '#2f6fb0'),
    }
    for label, (vals, color) in radar_series.items():
        closed = vals + vals[:1]
        ax.plot(angles, closed, label=label, color=color, lw=3.0,
                marker='o', markersize=5.2)
        ax.fill(angles, closed, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], fontsize=12)
    ax.set_ylim(0.65, 1.0)
    ax.set_yticks([0.70, 0.80, 0.85, 0.90, 0.95])
    ax.set_yticklabels(['0.70', '0.80', '0.85', '0.90', '0.95'],
                       fontsize=10, color='dimgray')
    ax.set_rlabel_position(18)
    ax.plot(angles, [0.85] * len(angles), color='black', ls=':',
            lw=1.2, alpha=0.55)
    ax.set_title('Full multi-task performance', fontsize=14, pad=22)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=10, frameon=False)

    y = np.arange(len(TASK_ORDER))
    colors = ['#c62828' if v >= 0 else '#2f6fb0' for v in mt_delta]
    ax_delta.barh(y, mt_delta, height=0.56, color=colors, alpha=0.92)
    ax_delta.axvline(0, color='black', lw=1.0)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels([TASK_LABELS[t] for t in TASK_ORDER],
                             fontsize=11)
    ax_delta.invert_yaxis()

    xmin = min(-0.005, float(mt_delta.min()) - 0.01)
    xmax = max(0.095, float(mt_delta.max()) + 0.015)
    ax_delta.set_xlim(xmin, xmax)
    ax_delta.set_xlabel(
        r'Full MT metric difference: $E_{\mathrm{pa}} - E_{\mathrm{base}}$',
        fontsize=11)
    ax_delta.set_title('Representation gain under the same model', fontsize=14, pad=12)
    ax_delta.grid(axis='x', alpha=0.25)
    sns.despine(ax=ax_delta, left=True)

    for yi, value in zip(y, mt_delta):
        offset = 0.003 if value >= 0 else -0.003
        ha = 'left' if value >= 0 else 'right'
        ax_delta.text(value + offset, yi, f'{value:+.3f}',
                      va='center', ha=ha, fontsize=10)

    wins_mt = int((mt_delta > 0).sum())
    mean_gain = float(mt_delta.mean())
    ax_delta.text(0.98, 0.08,
                  rf'$E_{{\mathrm{{pa}}}}$ wins {wins_mt}/7 tasks' + '\n'
                  f'Mean gain = {mean_gain:+.3f}',
                  transform=ax_delta.transAxes, ha='right', va='bottom',
                  fontsize=10,
                  bbox=dict(boxstyle='round,pad=0.35',
                            fc='white', ec='0.80', alpha=0.95))

    fig.suptitle(r'Figure 6. $E_{\mathrm{pa}}$ improves the learned alloy representation',
                 fontsize=15, y=0.99)
    fig.savefig(out_path, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def main():
    full_df = load_full_table()
    best_df = load_best_table()
    print(f"Loaded full table: {len(full_df)} rows; best table: {len(best_df)} rows")

    plot_radar_best(best_df, full_df, FIGURES_DIR / 'fig_radar_best_per_task.png')
    plot_m3_compare(FIGURES_DIR / 'fig_heatmap_M3_compare.png')
    plot_grouped_vs_full(full_df, FIGURES_DIR / 'fig_grouped_vs_full_bar.png')
    plot_delta_heatmap(full_df, FIGURES_DIR / 'fig_delta_heatmap.png')
    print(f"\nAll paper figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
