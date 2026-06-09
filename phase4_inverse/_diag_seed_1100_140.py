"""检查 1100°C/140MPa 附近 seed 数量分布"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from config import DATA_DIR

mt = pd.read_csv(DATA_DIR / 'master_table.csv')
crp = mt[mt['task'] == 'creep'].copy()

print(f"creep 任务总样本: {len(crp)}")
print(f"\n[1100°C/140MPa 附近 seed 分布]")
for tol in [(20, 20), (30, 30), (50, 30), (50, 50), (100, 50), (100, 80)]:
    dT, dS = tol
    sub = crp[(crp['test_temp'].between(1100-dT, 1100+dT)) &
              (crp['test_stress'].between(140-dS, 140+dS))]
    print(f"\n  T∈[{1100-dT},{1100+dT}], σ∈[{140-dS},{140+dS}]: total={len(sub)}")
    for th in [100, 200, 270, 300, 500, 700]:
        n = (sub['target'] > th).sum()
        print(f"    life>{th}h: {n}")
