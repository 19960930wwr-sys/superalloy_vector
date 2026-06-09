"""
基于训练集真实高 creep 合金的种子筛选 (seed-based screening)
=============================================================
策略:
  1. 从训练集 master_table.csv 抽取 creep 任务下满足
       T ∈ [Tmin, Tmax]°C  ∩  σ ∈ [σmin, σmax]MPa  ∩  life > life_min h
     的"已验证 seed 合金" (creep 直接采用真实实测值, 不再用模型预测)。
  2. 对每个 seed, 用 5 折多任务模型预测其余 5 项性能:
       solvus, density, phase_class, size, freezing_range.
  3. 按 "solvus↑  density↓  phase_class↓  size↓  freezing↓" 综合排序输出 top-N.
"""
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (DATA_DIR, RESULTS_DIR, ELEMENTS, MODELS_DIR, TASKS)
from phase4_inverse.forward_screen import (
    FeatureBuilder, load_models, TASK_PROCESS,
)
from phase3_multitask.dataset_alloy import load_alloy_dataset


# ===== 综合排序权重 (越大越好为正, 越小越好为负) =====
SCORE_WEIGHTS = {
    'solvus':             +1.0,   # 高
    'processing_window':  +1.0,   # 高 (= solidus - solvus, 工艺窗口越大越好)
    'density':            -1.0,   # 低
    'phase_class':        -1.5,   # 低 (避有害相)
    'size':               -1.0,   # 小
    'freezing_range':     -1.0,   # 小
}


def get_seeds(temp_lo, temp_hi, stress_lo, stress_hi, life_min):
    """从训练集 creep 任务中抽取满足条件的真实合金"""
    mt = pd.read_csv(DATA_DIR / "master_table.csv")
    crp = mt[mt['task'] == 'creep'].copy()
    seeds = crp[
        (crp['test_temp'].between(temp_lo, temp_hi)) &
        (crp['test_stress'].between(stress_lo, stress_hi)) &
        (crp['target'] > life_min)
    ].copy()
    seeds = seeds.reset_index(drop=True)
    seeds['creep_real'] = seeds['target']
    print(f"[seeds] T∈[{temp_lo},{temp_hi}]°C, σ∈[{stress_lo},{stress_hi}]MPa, "
          f"life>{life_min}h => {len(seeds)} 个真实合金")
    return seeds


@torch.no_grad()
def predict_seeds(seeds, builder, models, device, ds):
    """对每个 seed 用模型预测 5 项性能 (不含 creep, 因为有真实值)"""
    n = len(seeds)
    inp_no   = np.zeros((n, builder.input_dim), dtype=np.float32)
    inp_size = np.zeros((n, builder.input_dim), dtype=np.float32)
    for i, row in seeds.iterrows():
        comp = {e: float(row.get(e, 0.0) if pd.notna(row.get(e, np.nan)) else 0.0)
                for e in ELEMENTS}
        inp_no[i]   = builder.build_input(comp, None)
        inp_size[i] = builder.build_input(comp, TASK_PROCESS['size'])

    # 拼大 batch: [no(n), size(n)]
    big = np.concatenate([inp_no, inp_size], axis=0)
    big_t = torch.from_numpy(big).to(device)

    task_names = list(TASKS.keys())
    sums = {t: torch.zeros(big_t.shape[0], device=device) for t in task_names}
    for model in models:
        out = model(big_t)
        for t in task_names:
            sums[t] += out[t].squeeze(-1)
    avgs = {t: (sums[t] / len(models)).cpu().numpy() for t in task_names}

    # 反归一化 (用 ds.denormalize_target)
    res = pd.DataFrame()
    # 物性: 用 inp_no (前 n)
    for t in ['solvus', 'density', 'phase_class', 'liquidus', 'solidus']:
        res[t] = ds.denormalize_target(t, avgs[t][:n])
    # size: 用 inp_size (n ~ 2n)
    res['size'] = ds.denormalize_target('size', avgs['size'][n:2 * n])
    res['freezing_range']    = res['liquidus'] - res['solidus']
    res['processing_window'] = res['solidus']  - res['solvus']
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--temp_lo',   type=float, default=700)
    p.add_argument('--temp_hi',   type=float, default=820)
    p.add_argument('--stress_lo', type=float, default=700)
    p.add_argument('--stress_hi', type=float, default=900)
    p.add_argument('--life_min',  type=float, default=850)
    p.add_argument('--top_n',     type=int,   default=10)
    p.add_argument('--embedding_name', default='E_pa')
    # 软筛 (只参与排序, 不剔除; 设为 None 则不打分)
    p.add_argument('--solvus_min',   type=float, default=None,
                   help="可选硬过滤: solvus >= 该值才入候选")
    p.add_argument('--density_max',  type=float, default=None,
                   help="可选硬过滤: density <= 该值才入候选")
    p.add_argument('--window_min',   type=float, default=None,
                   help="可选硬过滤: processing_window >= 该值")
    p.add_argument('--phase_max',    type=float, default=None,
                   help="可选硬过滤: phase_class <= 该值")
    p.add_argument('--size_max',     type=float, default=None,
                   help="可选硬过滤: size <= 该值")
    p.add_argument('--freeze_max',   type=float, default=None,
                   help="可选硬过滤: freezing_range <= 该值")
    args = p.parse_args()

    # 1. seeds
    seeds = get_seeds(args.temp_lo, args.temp_hi, args.stress_lo,
                      args.stress_hi, args.life_min)
    if len(seeds) == 0:
        print("无 seed, 退出.")
        return

    # 2. 加载模型 + builder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = load_alloy_dataset(args.embedding_name)
    builder = FeatureBuilder(args.embedding_name)
    models = load_models(builder.input_dim, args.embedding_name, device)

    # 3. 模型预测其他 5 项
    print(f"[predict] 5 项性能 (除 creep 用真实值) ...")
    t0 = time.time()
    pred = predict_seeds(seeds, builder, models, device, ds)
    print(f"[predict] done in {time.time()-t0:.1f}s")

    # 4. 合并: seeds + 模型预测
    out = pd.concat([seeds.reset_index(drop=True), pred.reset_index(drop=True)],
                    axis=1)

    # 5. 可选硬过滤
    if args.solvus_min is not None:
        out = out[out['solvus'] >= args.solvus_min]
        print(f"[filter] solvus>={args.solvus_min} => {len(out)}")
    if args.density_max is not None:
        out = out[out['density'] <= args.density_max]
        print(f"[filter] density<={args.density_max} => {len(out)}")
    if args.window_min is not None:
        out = out[out['processing_window'] >= args.window_min]
        print(f"[filter] proc.window>={args.window_min} => {len(out)}")
    if args.phase_max is not None:
        out = out[out['phase_class'] <= args.phase_max]
        print(f"[filter] phase<={args.phase_max} => {len(out)}")
    if args.size_max is not None:
        out = out[out['size'] <= args.size_max]
        print(f"[filter] size<={args.size_max} => {len(out)}")
    if args.freeze_max is not None:
        out = out[out['freezing_range'] <= args.freeze_max]
        print(f"[filter] freezing<={args.freeze_max} => {len(out)}")
    if len(out) == 0:
        print("过滤后无候选, 放宽阈值再试.")
        return

    # 6. 综合打分 (z-score)
    score = np.zeros(len(out))
    for col, w in SCORE_WEIGHTS.items():
        if col not in out.columns: continue
        v = out[col].values.astype(float)
        mu, sd = np.nanmean(v), np.nanstd(v)
        if sd == 0 or np.isnan(sd):
            continue
        score += w * (v - mu) / sd
    out['score'] = score
    out = out.sort_values('score', ascending=False).reset_index(drop=True)

    # 7. 输出
    cols = (['score', 'creep_real', 'test_temp', 'test_stress'] +
            [c for c in ['solvus', 'processing_window', 'density',
                         'phase_class', 'size', 'freezing_range']
             if c in out.columns] +
            [e for e in ELEMENTS if e in out.columns])
    cols = [c for c in cols if c in out.columns]

    top = out.head(args.top_n)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 60)
    print(f"\n=== Top-{args.top_n} 候选 (按综合分排序) ===")
    print(top[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 8. 落盘
    tag = (f"seeds_T{int(args.temp_lo)}-{int(args.temp_hi)}"
           f"_S{int(args.stress_lo)}-{int(args.stress_hi)}"
           f"_life{int(args.life_min)}")
    out_path = RESULTS_DIR / f"forward_screen_seedscore_{tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n保存全表: {out_path}")
    top_path = RESULTS_DIR / f"forward_screen_seedtop{args.top_n}_{tag}.csv"
    top.to_csv(top_path, index=False)
    print(f"保存 top-{args.top_n}: {top_path}")


if __name__ == '__main__':
    main()
