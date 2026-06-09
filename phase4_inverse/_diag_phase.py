"""诊断 phase_class 安全区 + 其他约束的交集 (基于 expanded 全量预测)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

CSV = RESULTS_DIR / "forward_screen_all_predictions_E_pa_expanded.csv"
df = pd.read_csv(CSV)
n = len(df)
print(f"loaded {n:,} alloys")

# --- 1. phase_class 全空间分布 ---
pc = df['phase_class'].values
print("\n[phase_class 全空间分布]")
for q in [0, 5, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  p{q:>3d} = {np.percentile(pc, q):.4f}")
for thr in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    cnt = int((pc < thr).sum())
    print(f"  phase_class < {thr}: {cnt:>7,d} ({100*cnt/n:.2f}%)")

# --- 2. 其他 5 项硬约束 (不含 solvus, 不含 phase) ---
m_dens   = df['density']           < 8.9
m_freeze = df['freezing_range']    < 60.0
m_window = df['processing_window'] > 100.0
m_size   = df['size']              < 500.0
m_creep  = df['creep']             > 200.0
m_5mech  = m_dens & m_freeze & m_window & m_size & m_creep
print(f"\n[5 项机械约束 (含 creep, 不含 solvus, 不含 phase)] pass = {int(m_5mech.sum()):,}")

# --- 3. phase_class 阈值 × solvus 阈值 双扫描 ---
print("\n[5 项机械 ∩ phase<X ∩ solvus>Y]")
hdr = ["solvus\\phase"] + [f"<{p:.2f}" for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]]
print("  " + "  ".join(f"{h:>10s}" for h in hdr))
for sv in [1100, 1120, 1140, 1150, 1160, 1180, 1220]:
    m_sv = df['solvus'] > sv
    row = [f"{sv}"]
    for thr in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        cnt = int((m_5mech & m_sv & (df['phase_class'] < thr)).sum())
        row.append(f"{cnt:,}")
    print("  " + "  ".join(f"{c:>10s}" for c in row))

# --- 4. 在 phase_class 最低的合金里看其他约束怎么样 ---
top_safe = df.sort_values('phase_class').head(200)
print(f"\n[phase_class 最低 200 个合金的其他指标]")
print(f"  phase_class range = {top_safe['phase_class'].min():.4f} ~ {top_safe['phase_class'].max():.4f}")
for col, op, thr in [('solvus','>',1220),('solvus','>',1150),('solvus','>',1100),
                     ('density','<',8.9),('freezing_range','<',60),
                     ('processing_window','>',100),('size','<',500),
                     ('creep','>',200)]:
    sub = top_safe[col] > thr if op == '>' else top_safe[col] < thr
    print(f"  {col} {op} {thr}: {int(sub.sum())}/200")

print(f"\n  组分均值: " + " ".join(
    f"{e}={top_safe[e].mean():.2f}" for e in ['Co','Ni','Al','Ti','W','Ta','Mo','Nb','Cr','Re']))
