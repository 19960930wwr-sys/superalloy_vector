"""
Run PA-MLM ablation experiments end to end.

By default this script reuses existing E_base and E_pa outputs, trains the two
missing intermediate pretraining variants (E_attr and E_proc), runs full
multi-task downstream evaluation for E_attr and E_proc, then builds an
ablation summary table.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Run PA-MLM ablation experiments.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-pretrain", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-downstream", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_pretrain:
        from phase2_pretrain.train_pa_mlm_ablation import train_pa_mlm_variant

        for variant in ["attr", "proc"]:
            train_pa_mlm_variant(
                variant,
                num_epochs=args.epochs,
                max_samples=args.max_samples,
                force=args.force_pretrain,
                num_workers=args.num_workers,
            )

    if not args.skip_downstream:
        from phase3_multitask.train_multitask import train_multitask

        for embedding in ["E_attr", "E_proc"]:
            train_multitask(embedding)

    from phase3_multitask.build_pa_mlm_ablation_table import (
        build_pa_mlm_ablation_table,
    )
    from config import RESULTS_DIR

    df = build_pa_mlm_ablation_table()
    out_csv = RESULTS_DIR / "pa_mlm_ablation_multitask_summary.csv"
    out_xlsx = RESULTS_DIR / "pa_mlm_ablation_multitask_summary.xlsx"
    df.to_csv(out_csv, index=False)
    with __import__("pandas").ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="ablation_summary", index=False)
        if not df.empty:
            df.pivot_table(
                index=["task", "metric"],
                columns="variant",
                values="value",
                aggfunc="first",
            ).reset_index().to_excel(writer, sheet_name="pivot", index=False)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_xlsx}")


if __name__ == "__main__":
    main()
