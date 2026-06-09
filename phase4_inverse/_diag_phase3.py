"""phase 阈值 × creep 阈值 矩阵扫描 (其他 5 项硬约束保留)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

df = pd.read_csv(RESULTS_DIR / "forward_screen_all_predictions_E_pa_expanded.csv")

# 固定: solvus>1150, density<8.9, freeze<60, window>100, size<500
m_base = ((df['solvus']            > 1150.0) &
          (df['density']           <    8.9) &
          (df['freezing_range']    <   60.0) &
          (df['processing_window'] >  100.0) &
          (df['size']              <  500.0))
print(f"[base] solvus>1150 ∩ density<8.9 ∩ freeze<60 ∩ window>100 ∩ size<500: "
      f"{int(m_base.sum()):,}")

# 矩阵扫描
phase_thrs = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
creep_thrs = [50, 80, 100, 120, 150, 180, 200]

print("\n              " + "  ".join(f"creep>{c:<3d}" for c in creep_thrs))
for pt in phase_thrs:
    m_pc = df['phase_class'] < pt
    row = []
    for ct in creep_thrs:
        m_cr = df['creep'] > ct
        cnt = int((m_base & m_pc & m_cr).sum())
        row.append(f"{cnt:>8,d}")
    print(f"  phase<{pt:<5.2f}" + "  ".join(row))

# 在 base + creep>150 子集里看 phase 最低能到哪
sub = df[m_base & (df['creep'] > 150)]
print(f"\n=== base ∩ creep>150  (n={len(sub):,}) ===")
if len(sub):
    pc = sub['phase_class'].values
    print(f"  phase_class:  min={pc.min():.4f}  p10={np.percentile(pc,10):.4f}  "
          f"p50={np.percentile(pc,50):.4f}  max={pc.max():.4f}")

sub2 = df[m_base & (df['creep'] > 100)]
print(f"\n=== base ∩ creep>100  (n={len(sub2):,}) ===")
if len(sub2):
    pc = sub2['phase_class'].values
    print(f"  phase_class:  min={pc.min():.4f}  p10={np.percentile(pc,10):.4f}  "
          f"p50={np.percentile(pc,50):.4f}  max={pc.max():.4f}")
