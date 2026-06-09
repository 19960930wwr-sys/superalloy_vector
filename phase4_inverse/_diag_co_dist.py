"""检查训练集 creep 任务中 Co 含量分布 + 760°C 附近 Co-Ni 合金的 life 分布"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from config import DATA_DIR

mt = pd.read_csv(DATA_DIR / 'master_table.csv')
crp = mt[mt['task'] == 'creep'].copy()
crp['Co'] = crp['Co'].fillna(0.0)
crp['Ni'] = crp['Ni'].fillna(0.0)

print(f"creep 任务总样本: {len(crp)}")
print(f"\n[Co 含量分布]")
for lo, hi in [(0,5),(5,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,100)]:
    n = ((crp['Co'] >= lo) & (crp['Co'] < hi)).sum()
    print(f"  Co ∈ [{lo:>2d}, {hi:>3d}): {n}")

# 760°C 附近 (700-820) ∩ stress[700,900] ∩ 不同 Co 阈值的 life 分布
print(f"\n[T∈[700,820]°C ∩ σ∈[700,900]MPa 的样本中 Co 含量与 life]")
sub = crp[(crp['test_temp'].between(700,820)) & (crp['test_stress'].between(700,900))]
print(f"  total = {len(sub)}")
for co_lo in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
    s = sub[sub['Co'] >= co_lo]
    if len(s) == 0:
        print(f"  Co>={co_lo}: 0 个")
        continue
    print(f"  Co>={co_lo:>2d}: n={len(s):>2d}, "
          f"life range=[{s['target'].min():.0f}, {s['target'].max():.0f}], "
          f">850h: {(s['target']>850).sum()}, "
          f">500h: {(s['target']>500).sum()}, "
          f">200h: {(s['target']>200).sum()}")

# 列出所有 Co>=20 ∩ T∈[700,820] 的样本详情
print(f"\n[Co>=20 ∩ T∈[700,820] 详细]")
hi_co = crp[(crp['Co'] >= 20) & (crp['test_temp'].between(700,820))]
elems = ['Co','Ni','Al','Ti','W','Ta','Mo','Nb','Cr','Re','Hf','Ru']
print(f"  {'T':>5s} {'σ':>5s} {'life':>7s}  | "
      + " ".join(f"{e:>5s}" for e in elems))
for _, r in hi_co.iterrows():
    line = f"  {r['test_temp']:>5.0f} {r['test_stress']:>5.0f} {r['target']:>7.0f}  | "
    line += " ".join(f"{r.get(e,0.0):>5.2f}" for e in elems)
    print(line)

# Co-Ni 双基 (Co>=30 且 Ni<=50) 的全部 creep 样本
print(f"\n[全部 creep 任务中 Co>=30 ∩ Ni<=50 (真正 Co 基/Co-Ni 双基)]")
coni = crp[(crp['Co'] >= 30) & (crp['Ni'] <= 50)]
print(f"  共 {len(coni)} 个样本")
if len(coni) > 0:
    print(f"  T 范围: {coni['test_temp'].min():.0f} ~ {coni['test_temp'].max():.0f}")
    print(f"  σ 范围: {coni['test_stress'].min():.0f} ~ {coni['test_stress'].max():.0f}")
    print(f"  life: min={coni['target'].min():.0f}, max={coni['target'].max():.0f}, "
          f"median={coni['target'].median():.0f}")
    print(f"  >850h: {(coni['target']>850).sum()}, >500h: {(coni['target']>500).sum()}, "
          f">200h: {(coni['target']>200).sum()}")
