"""检查 wider 空间下 creep@760/800 预测分布 + phase 矩阵"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

df = pd.read_csv(RESULTS_DIR / "forward_screen_all_predictions_E_pa_wider.csv")
n = len(df)
print(f"loaded {n:,} alloys (wider, creep@760/800)")

# creep 全空间分布
cv = df['creep'].values
print("\n[creep 760°C/800MPa 全空间预测]")
for q in [0, 5, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  p{q:>3d} = {np.percentile(cv, q):.1f} h")
for thr in [200, 400, 600, 800, 850, 1000, 1500, 2000]:
    cnt = int((cv > thr).sum())
    print(f"  creep > {thr:>4d} h: {cnt:>9,d}  ({100*cnt/n:.2f}%)")

# 5 项基础约束 (含 phase<0.5)
base = ((df['solvus']            > 1150.0) &
        (df['density']           <    8.9) &
        (df['freezing_range']    <   60.0) &
        (df['processing_window'] >  100.0) &
        (df['size']              <  500.0))
base_phase = base & (df['phase_class'] < 0.5)
print(f"\n[base (5 项, 不含 phase, 不含 creep)] pass = {int(base.sum()):,}")
print(f"[base ∩ phase<0.5]                       pass = {int(base_phase.sum()):,}")

print("\n[base ∩ phase<X ∩ creep>Y]")
hdr = ["phase\\creep"] + [f">{c}" for c in [200, 400, 600, 800, 850, 1000]]
print("  " + "  ".join(f"{h:>10s}" for h in hdr))
for pt in [0.3, 0.5, 0.7, 0.9, 0.99]:
    m_pc = df['phase_class'] < pt
    row = [f"{pt:<5.2f}"]
    for ct in [200, 400, 600, 800, 850, 1000]:
        m_cr = df['creep'] > ct
        cnt = int((base & m_pc & m_cr).sum())
        row.append(f"{cnt:,}")
    print("  " + "  ".join(f"{c:>10s}" for c in row))

# base ∩ phase<0.5 子集中 creep 上限
sub = df[base_phase]
if len(sub):
    cv2 = sub['creep'].values
    print(f"\n[base ∩ phase<0.5  n={len(sub):,}] creep 分布:")
    for q in [50, 75, 90, 95, 99, 100]:
        print(f"  p{q:>3d} = {np.percentile(cv2, q):.1f}")

# base 子集中 creep 最高的 200 个看 phase
top_creep = df[base].sort_values('creep', ascending=False).head(200)
if len(top_creep):
    print(f"\n[base 子集 creep top-200] creep range = "
          f"{top_creep['creep'].min():.1f} ~ {top_creep['creep'].max():.1f}")
    pc = top_creep['phase_class'].values
    print(f"  phase_class min/p10/p50/max = {pc.min():.3f}/"
          f"{np.percentile(pc,10):.3f}/{np.percentile(pc,50):.3f}/{pc.max():.3f}")
