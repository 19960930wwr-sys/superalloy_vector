"""检查 wider 空间下 creep@760/800 预测最高的合金成分共性"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

df = pd.read_csv(RESULTS_DIR / "forward_screen_all_predictions_E_pa_wider.csv")
print(f"loaded {len(df):,} alloys")

# 全空间 creep top-200 的成分统计
top = df.sort_values('creep', ascending=False).head(200)
print(f"\n[全空间 creep top-200 成分分布 (越贴上限说明扩大空间还有救)]")
print(f"  creep range: {top['creep'].min():.1f} ~ {top['creep'].max():.1f} h")
elems = ['Co', 'Ni', 'Al', 'Ti', 'W', 'Ta', 'Mo', 'Nb', 'Cr', 'Re', 'Hf']
print(f"\n  {'elem':<5s} {'min':>6s} {'p50':>6s} {'p90':>6s} {'max':>6s}    "
      f"wider 上限")
upper = {'Co':60,'Ni':30,'Al':13,'Ti':3.5,'W':6,'Ta':7,'Mo':3.5,
         'Nb':1.5,'Cr':5,'Re':4,'Hf':0.5}
for e in elems:
    if e not in top.columns:
        continue
    v = top[e].values
    print(f"  {e:<5s} {v.min():>6.2f} {np.percentile(v,50):>6.2f} "
          f"{np.percentile(v,90):>6.2f} {v.max():>6.2f}    upper={upper[e]:>5.1f}")

# 看 creep 最高的 top-20 详细
print(f"\n[creep top-20 详细]")
cols = ['Co','Ni','Al','Ti','W','Ta','Mo','Nb','Cr','Re','Hf',
        'creep','solvus','density','phase_class','size','freezing_range']
print(top.head(20)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# creep 排名 vs 元素值的相关
print(f"\n[全空间 creep 与各元素的 Pearson 相关 (越正 = 该元素越高 creep 越大)]")
for e in elems:
    if e not in df.columns: continue
    if df[e].nunique() <= 1: continue
    r = np.corrcoef(df[e].values, df['creep'].values)[0, 1]
    print(f"  {e:<5s} r = {r:>+.3f}")
