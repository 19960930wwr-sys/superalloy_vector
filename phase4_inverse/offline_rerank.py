"""离线筛选 (无需重跑模型): 从 expanded 全量预测中按自定义阈值出 top-N"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, ELEMENTS

ALL_CSV = RESULTS_DIR / "forward_screen_all_predictions_E_pa_expanded.csv"

# 综合排序权重 (与主脚本一致)
SCORE_WEIGHTS = {
    'solvus':            (+1, 1.0),
    'density':           (-1, 1.0),
    'freezing_range':    (-1, 1.0),
    'processing_window': (+1, 1.0),
    'size':              (-1, 1.0),
    'creep':             (+1, 1.5),
    'phase_class':       (-1, 1.5),  # 用户强制 phase 安全 -> 加权
}


def fmt_label(row):
    parts = []
    for e in ['Co','Ni','Al','Ti','W','Ta','Mo','Nb','Cr','Re']:
        if e in row and row[e] > 0:
            parts.append(f"{e}{row[e]:g}")
    return "-".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solvus_min',   type=float, default=1150.0)
    ap.add_argument('--density_max',  type=float, default=8.9)
    ap.add_argument('--freeze_max',   type=float, default=60.0)
    ap.add_argument('--window_min',   type=float, default=100.0)
    ap.add_argument('--size_max',     type=float, default=500.0)
    ap.add_argument('--creep_min',    type=float, default=80.0)
    ap.add_argument('--phase_max',    type=float, default=0.7)
    ap.add_argument('--top_n',        type=int,   default=10)
    ap.add_argument('--tag',          type=str,   default='phaseSafe')
    args = ap.parse_args()

    df = pd.read_csv(ALL_CSV)
    print(f"[Load] {len(df):,} alloys from {ALL_CSV.name}")

    cons = [
        ('solvus',            '>', args.solvus_min),
        ('density',           '<', args.density_max),
        ('freezing_range',    '<', args.freeze_max),
        ('processing_window', '>', args.window_min),
        ('size',              '<', args.size_max),
        ('creep',             '>', args.creep_min),
        ('phase_class',       '<', args.phase_max),
    ]
    print("\n[Constraints]")
    mask = pd.Series(True, index=df.index)
    for col, op, thr in cons:
        sub = df[col] > thr if op == '>' else df[col] < thr
        mask &= sub
        print(f"  {col:18s} {op} {thr:>8.2f}  pass={int(sub.sum()):>7,d}")
    feas = df[mask].copy()
    print(f"\n[Joint pass] = {len(feas):,}")
    if feas.empty:
        print("  -> no candidates; relax thresholds.")
        return

    # 综合排序
    score = np.zeros(len(feas))
    for col, (sign, w) in SCORE_WEIGHTS.items():
        if col not in feas.columns: continue
        x = feas[col].values.astype(float)
        mu, sd = np.nanmean(x), np.nanstd(x)
        if sd < 1e-9: sd = 1.0
        score += sign * w * (x - mu) / sd
    feas['score'] = score
    top = feas.sort_values('score', ascending=False).head(args.top_n).reset_index(drop=True)
    top.insert(0, 'rank', np.arange(1, len(top) + 1))

    out_csv  = RESULTS_DIR / f"forward_screen_top{args.top_n}_E_pa_expanded_{args.tag}.csv"
    out_json = RESULTS_DIR / f"forward_screen_top{args.top_n}_E_pa_expanded_{args.tag}.json"
    feas_csv = RESULTS_DIR / f"forward_screen_feasible_E_pa_expanded_{args.tag}.csv"
    feas.to_csv(feas_csv, index=False)
    top.to_csv(out_csv, index=False)

    records = []
    for _, r in top.iterrows():
        records.append({
            'rank': int(r['rank']), 'label': fmt_label(r),
            'composition_at_pct': {e: float(r[e]) for e in ELEMENTS
                                   if e in r and r[e] > 0},
            'predictions': {
                'solvus_C':            float(r['solvus']),
                'density_g_cm3':       float(r['density']),
                'liquidus_C':          float(r['liquidus']),
                'solidus_C':           float(r['solidus']),
                'freezing_range_C':    float(r['freezing_range']),
                'processing_window_C': float(r['processing_window']),
                'gamma_prime_size':    float(r['size']),
                'creep_life_h':        float(r['creep']),
                'harmful_phase_prob':  float(r['phase_class']),
            },
            'score': float(r['score']),
        })
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'tag': args.tag,
            'constraints': {c: {'op': o, 'threshold': t} for c,o,t in cons},
            'n_feasible': int(len(feas)),
            'top_candidates': records,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n=== TOP {len(top)} (saved -> {out_csv.name}) ===")
    cols = ['rank','Co','Al','Ti','W','Ta','Mo','Nb','Re',
            'solvus','density','freezing_range','processing_window',
            'size','creep','phase_class','score']
    print(top[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == '__main__':
    main()
