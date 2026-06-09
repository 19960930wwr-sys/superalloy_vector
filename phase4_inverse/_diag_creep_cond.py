"""检查 creep 训练样本的 (test_temp, test_stress) 分布"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

mt = pd.read_csv(DATA_DIR / "master_table.csv")
crp = mt[mt['task'] == 'creep'].copy()
print(f"creep training samples = {len(crp)}")

for col in ['test_temp', 'test_stress']:
    v = crp[col].dropna().values
    print(f"\n{col}: n={len(v)}")
    for q in [0, 5, 25, 50, 75, 90, 95, 100]:
        print(f"  p{q:>3d} = {np.percentile(v, q):.1f}")

print("\n[联合分布: 760°C 附近 + 800 MPa 附近的样本数]")
for tt_lo, tt_hi in [(700, 820), (740, 780), (750, 770)]:
    for ts_lo, ts_hi in [(700, 900), (750, 850), (780, 820)]:
        sub = crp[(crp['test_temp']  >= tt_lo) & (crp['test_temp']  <= tt_hi) &
                  (crp['test_stress'] >= ts_lo) & (crp['test_stress'] <= ts_hi)]
        print(f"  T∈[{tt_lo},{tt_hi}] & σ∈[{ts_lo},{ts_hi}]: n={len(sub)}, "
              f"target range = [{sub['target'].min() if len(sub) else 'NA'}, "
              f"{sub['target'].max() if len(sub) else 'NA'}]")

# 二维 hist
print("\n[T x σ 联合分布 (训练集) Top-10 单元格]")
crp2 = crp.dropna(subset=['test_temp', 'test_stress'])
crp2['t_bin'] = (crp2['test_temp'] // 50).astype(int) * 50
crp2['s_bin'] = (crp2['test_stress'] // 100).astype(int) * 100
g = crp2.groupby(['t_bin', 's_bin']).size().reset_index(name='n').sort_values('n', ascending=False)
print(g.head(15).to_string(index=False))

# target>850h 在训练集出现频率
print("\n[训练集 creep_life > 850h 占比]")
n_above = int((crp['target'] > 850).sum())
print(f"  >850h: {n_above}/{len(crp)} ({100*n_above/len(crp):.1f}%)")
n_above_local = int(((crp['target'] > 850) &
                     (crp['test_temp'].between(700, 820)) &
                     (crp['test_stress'].between(700, 900))).sum())
print(f"  >850h ∩ T∈[700,820] ∩ σ∈[700,900]: {n_above_local}")
