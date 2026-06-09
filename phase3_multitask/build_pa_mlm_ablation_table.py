"""
Build the PA-MLM ablation summary table.

This table compares full multi-task downstream performance for:
  E_base : plain MLM
  E_attr : MLM + elemental-attribute regression
  E_proc : MLM + process-category prediction
  E_pa   : full PA-MLM
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR, TASKS  # noqa: E402


ABLATION_LABELS = {
    "E_base": "Plain MLM",
    "E_attr": "MLM + elemental attributes",
    "E_proc": "MLM + process category",
    "E_pa": "Full PA-MLM",
}


def _load_result(embedding: str) -> dict:
    path = RESULTS_DIR / f"multitask_{embedding}_results.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pa_mlm_ablation_table() -> pd.DataFrame:
    rows = []
    for emb, label in ABLATION_LABELS.items():
        result = _load_result(emb)
        for task_name in TASKS:
            metrics = result.get(task_name)
            if not metrics:
                continue
            task_type = TASKS[task_name]["type"]
            if task_type == "regression":
                rows.append(
                    {
                        "embedding": emb,
                        "variant": label,
                        "task": task_name,
                        "metric": "R2",
                        "value": metrics.get("r2_mean"),
                        "std": metrics.get("r2_std"),
                        "rmse": metrics.get("rmse_mean"),
                        "rmse_std": metrics.get("rmse_std"),
                    }
                )
            else:
                rows.append(
                    {
                        "embedding": emb,
                        "variant": label,
                        "task": task_name,
                        "metric": "F1",
                        "value": metrics.get("f1_mean"),
                        "std": metrics.get("f1_std"),
                        "auc": metrics.get("auc_mean"),
                        "auc_std": metrics.get("auc_std"),
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        order = {emb: i for i, emb in enumerate(ABLATION_LABELS)}
        task_order = {task: i for i, task in enumerate(TASKS)}
        df["embedding_order"] = df["embedding"].map(order)
        df["task_order"] = df["task"].map(task_order)
        df = df.sort_values(["task_order", "embedding_order"]).drop(
            columns=["embedding_order", "task_order"]
        )
    return df


def main():
    df = build_pa_mlm_ablation_table()
    out_csv = RESULTS_DIR / "pa_mlm_ablation_multitask_summary.csv"
    out_xlsx = RESULTS_DIR / "pa_mlm_ablation_multitask_summary.xlsx"
    df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="ablation_summary", index=False)
        if not df.empty:
            pivot = df.pivot_table(
                index=["task", "metric"],
                columns="variant",
                values="value",
                aggfunc="first",
            ).reset_index()
            pivot.to_excel(writer, sheet_name="pivot", index=False)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_xlsx}")
    if df.empty:
        print("No ablation results found yet. Run multitask training first.")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
