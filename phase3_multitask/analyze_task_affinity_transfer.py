"""
Relate M3 task-affinity diagnostics to observed grouped-transfer gains.

For each task in each actually trained grouped multi-task model, this script
computes:

    delta = metric(grouped model, task) - metric(full MT, task)

where the metric is R2 for regression tasks and F1 for phase classification.
It then pairs delta with the task's average within-group M3 similarity:

    mean_M3(t) = mean_{k in G, k != t} M3(t, k)

The resulting table and figure support the interpretation that M3 is a
diagnostic indicator that must be empirically validated by grouped training,
rather than a deterministic predictor of positive transfer.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import FIGURES_DIR, RESULTS_DIR, TASKS  # noqa: E402


EMBEDDING_LABELS = {
    "E_pa": "PA-MLM-derived descriptor",
    "E_base": "Plain MLM-derived descriptor",
}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_column(task: str) -> str:
    return "r2_mean" if TASKS[task]["type"] == "regression" else "f1_mean"


def _load_similarity(embedding: str, method: str):
    data = _load_json(RESULTS_DIR / f"task_grouping_{embedding}.json")
    task_names = data[method]["task_names"]
    matrix = np.array(data[method]["similarity_matrix"], dtype=float)
    return task_names, matrix


def _similarity_value(task_a: str, task_b: str, task_names, matrix) -> float:
    ia = task_names.index(task_a)
    ib = task_names.index(task_b)
    return float(matrix[ia, ib])


def build_transfer_gain_table() -> pd.DataFrame:
    comparison_path = RESULTS_DIR / "full_comparison_table.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing {comparison_path}; run phase3_multitask.evaluate_models first."
        )

    df = pd.read_csv(comparison_path)
    grouped = df[df["model"].astype(str).str.startswith("grouped-")].copy()
    if grouped.empty:
        raise ValueError("No grouped rows found in full_comparison_table.csv")

    rows = []
    for (embedding, grouped_model), gdf in grouped.groupby(["embedding", "model"]):
        if embedding not in EMBEDDING_LABELS:
            continue
        tasks_in_group = gdf["task"].tolist()
        if len(tasks_in_group) < 2:
            continue

        sim_data = {}
        for method in ["M2", "M3"]:
            task_names, matrix = _load_similarity(embedding, method)
            pair_values = []
            for i, task_a in enumerate(tasks_in_group):
                for task_b in tasks_in_group[i + 1 :]:
                    pair_values.append(
                        _similarity_value(task_a, task_b, task_names, matrix)
                    )
            sim_data[method] = {
                "task_names": task_names,
                "matrix": matrix,
                "group_mean": float(np.mean(pair_values)) if pair_values else np.nan,
                "group_min": float(np.min(pair_values)) if pair_values else np.nan,
                "group_max": float(np.max(pair_values)) if pair_values else np.nan,
            }

        for _, row in gdf.iterrows():
            task = row["task"]
            metric_col = _metric_column(task)
            grouped_value = float(row[metric_col])

            base = df[
                (df["embedding"] == embedding)
                & (df["model"] == "multitask")
                & (df["task"] == task)
            ]
            if base.empty:
                continue
            full_mt_value = float(base.iloc[0][metric_col])
            others = [t for t in tasks_in_group if t != task]
            task_stats = {}
            for method in ["M2", "M3"]:
                task_names = sim_data[method]["task_names"]
                matrix = sim_data[method]["matrix"]
                task_pair_values = [
                    _similarity_value(task, other, task_names, matrix)
                    for other in others
                ]
                task_stats[method] = {
                    "task_mean": (
                        float(np.mean(task_pair_values))
                        if task_pair_values
                        else np.nan
                    ),
                    "task_min": (
                        float(np.min(task_pair_values))
                        if task_pair_values
                        else np.nan
                    ),
                }

            rows.append(
                {
                    "embedding": embedding,
                    "embedding_label": EMBEDDING_LABELS[embedding],
                    "grouped_model": grouped_model,
                    "task": task,
                    "group_tasks": ", ".join(tasks_in_group),
                    "metric": "R2" if metric_col == "r2_mean" else "F1",
                    "grouped_value": grouped_value,
                    "full_mt_value": full_mt_value,
                    "transfer_gain_delta": grouped_value - full_mt_value,
                    "task_mean_within_group_M2": task_stats["M2"]["task_mean"],
                    "task_min_within_group_M2": task_stats["M2"]["task_min"],
                    "task_mean_within_group_M3": task_stats["M3"]["task_mean"],
                    "task_min_within_group_M3": task_stats["M3"]["task_min"],
                    "group_mean_M2": sim_data["M2"]["group_mean"],
                    "group_min_M2": sim_data["M2"]["group_min"],
                    "group_max_M2": sim_data["M2"]["group_max"],
                    "group_mean_M3": sim_data["M3"]["group_mean"],
                    "group_min_M3": sim_data["M3"]["group_min"],
                    "group_max_M3": sim_data["M3"]["group_max"],
                    "n_tasks_in_group": len(tasks_in_group),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["embedding", "grouped_model", "task"]).reset_index(
            drop=True
        )
    return out


def _spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    if spearmanr is not None:
        res = spearmanr(x, y)
        return float(res.statistic), float(res.pvalue), len(x)
    # Fallback: Pearson correlation of average ranks.
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1]), np.nan, len(x)


def build_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in [("All grouped tasks", df)] + [
        (emb, g) for emb, g in df.groupby("embedding")
    ]:
        for x_col in [
            "task_mean_within_group_M2",
            "group_mean_M2",
            "task_mean_within_group_M3",
            "group_mean_M3",
        ]:
            rho, p, n = _spearman(
                sub[x_col].to_numpy(dtype=float),
                sub["transfer_gain_delta"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "subset": label,
                    "affinity_variable": x_col,
                    "spearman_rho": rho,
                    "p_value": p,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def plot_transfer_gain(df: pd.DataFrame, path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.9, 4.8), dpi=300)
    colors = {"E_pa": "#2F5DA8", "E_base": "#D64B3C"}
    markers = {"E_pa": "o", "E_base": "s"}

    for emb, sub in df.groupby("embedding"):
        ax.scatter(
            sub["task_mean_within_group_M3"],
            sub["transfer_gain_delta"],
            s=72,
            marker=markers.get(emb, "o"),
            color=colors.get(emb, "gray"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            label=EMBEDDING_LABELS.get(emb, emb),
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["task"],
                (
                    row["task_mean_within_group_M3"],
                    row["transfer_gain_delta"],
                ),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8.5,
                color="#333333",
            )

    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Mean within-group M3 similarity")
    ax.set_ylabel("Grouped model gain over full MT")
    ax.set_title("Task affinity vs. observed transfer gain")
    ax.grid(True, color="#DDDDDD", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    df = build_transfer_gain_table()
    corr = build_correlation_table(df)

    out_csv = RESULTS_DIR / "task_affinity_transfer_gain.csv"
    corr_csv = RESULTS_DIR / "task_affinity_transfer_correlation.csv"
    out_xlsx = RESULTS_DIR / "task_affinity_transfer_analysis.xlsx"
    fig_path = FIGURES_DIR / "task_affinity_transfer_gain.png"
    origin_dir = FIGURES_DIR / "origin"
    origin_dir.mkdir(parents=True, exist_ok=True)
    origin_xlsx = origin_dir / "task_affinity_transfer_gain_data.xlsx"

    df.to_csv(out_csv, index=False)
    corr.to_csv(corr_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="transfer_gain", index=False)
        corr.to_excel(writer, sheet_name="spearman", index=False)
    with pd.ExcelWriter(origin_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="plot_data", index=False)
        corr.to_excel(writer, sheet_name="spearman", index=False)
    plot_transfer_gain(df, fig_path)

    print(f"Saved: {out_csv}")
    print(f"Saved: {corr_csv}")
    print(f"Saved: {out_xlsx}")
    print(f"Saved: {origin_xlsx}")
    print(f"Saved: {fig_path}")
    print("\nTransfer-gain table:")
    print(df.to_string(index=False))
    print("\nSpearman correlations:")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
