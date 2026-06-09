"""定位 phase_safe 子集内各机械约束的可行边界"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

df = pd.read_csv(RESULTS_DIR / "forward_screen_all_predictions_E_pa_expanded.csv")
n = len(df)

for thr in [0.1, 0.3, 0.5]:
    sub = df[df['phase_class'] < thr]
    print(f"\n=== phase_class < {thr}  (n={len(sub):,}) ===")
    for col, hi_is_better in [('solvus', True), ('density', False),
                              ('freezing_range', False),
                              ('processing_window', True),
                              ('size', False), ('creep', True)]:
        v = sub[col].values
        print(f"  {col:20s}  min={v.min():>8.2f}  p50={np.percentile(v,50):>8.2f}  "
              f"p90={np.percentile(v,90):>8.2f}  p99={np.percentile(v,99):>8.2f}  "
              f"max={v.max():>8.2f}")

# leave-one-out: phase<0.5 + 6 项, 看每项独立放掉后能否非空
print("\n=== phase<0.5 + 6 项约束 leave-one-out ===")
m_pc     = df['phase_class']       < 0.5
m_sv     = df['solvus']            > 1150
m_dens   = df['density']           < 8.9
m_freeze = df['freezing_range']    < 60.0
m_window = df['processing_window'] > 100.0
m_size   = df['size']              < 500.0
m_creep  = df['creep']             > 200.0
all_m = {'phase<0.5': m_pc, 'solvus>1150': m_sv, 'density<8.9': m_dens,
         'freeze<60': m_freeze, 'window>100': m_window,
         'size<500': m_size, 'creep>200': m_creep}
full = m_pc & m_sv & m_dens & m_freeze & m_window & m_size & m_creep
print(f"  full intersect: {int(full.sum())}")
for drop in all_m:
    rest = pd.Series(True, index=df.index)
    for k, v in all_m.items():
        if k == drop: continue
        rest &= v
    print(f"  drop {drop:14s} -> {int(rest.sum()):,}")

# 把 creep 当作连续阈值, 在 phase<0.5 + 其余 4 项机械 + solvus>1150 子集里看 creep 分布
m_4mech_phase_sv = m_pc & m_sv & m_dens & m_freeze & m_window & m_size
sub = df[m_4mech_phase_sv]
print(f"\n=== phase<0.5 ∩ solvus>1150 ∩ {{density,freeze,window,size}} (n={len(sub):,}) ===")
if len(sub):
    cv = sub['creep'].values
    print(f"  creep:  min={cv.min():.1f}  p50={np.percentile(cv,50):.1f}  "
          f"p90={np.percentile(cv,90):.1f}  max={cv.max():.1f}")
    for thr in [50, 80, 100, 120, 150, 180, 200]:
        print(f"  creep > {thr}: {int((cv > thr).sum()):,}")
