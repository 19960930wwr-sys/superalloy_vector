"""验证: 文献合金 Co-7Al-8W-1Ta-4Ti 在训练集里的覆盖度 + 当前模型的预测"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import torch

from config import DATA_DIR, MODELS_DIR, ELEMENTS, TASKS
from phase3_multitask.dataset_alloy import load_alloy_dataset
from phase4_inverse.forward_screen import FeatureBuilder, load_models, TASK_PROCESS

# --- 文献合金: Co-7Al-8W-1Ta-4Ti (at%, Co bal.) ---
target_alloy = {'Co': 80.0, 'Al': 7.0, 'W': 8.0, 'Ta': 1.0, 'Ti': 4.0}
print("=== 目标合金: Co-7Al-8W-1Ta-4Ti ===")
print(f"  成分 (at%): {target_alloy}")
print(f"  总和: {sum(target_alloy.values()):.1f}")
print(f"  文献报道: 760°C / 800 MPa, life > 850 h")

# --- 1. 训练集里搜索类似的 Co-Al-W 合金 (任意 task) ---
mt = pd.read_csv(DATA_DIR / "master_table.csv")
mt['Co'] = mt['Co'].fillna(0.0)
mt['W']  = mt['W'].fillna(0.0)
mt['Al'] = mt['Al'].fillna(0.0)

print(f"\n[训练集中类 Co-Al-W γ' 合金搜索]")
similar = mt[(mt['Co'] >= 60) & (mt['W'] >= 5) & (mt['Al'] >= 5)]
print(f"  Co>=60 ∩ W>=5 ∩ Al>=5: 共 {len(similar)} 个 task 记录")
if len(similar) > 0:
    print(f"  分布在哪些任务: {similar['task'].value_counts().to_dict()}")
    # 列出样本
    elems = ['Co','Ni','Al','Ti','W','Ta','Mo','Nb','Cr','Re']
    show_cols = ['task', 'target', 'test_temp', 'test_stress'] + elems
    show_cols = [c for c in show_cols if c in similar.columns]
    print(similar[show_cols].head(20).to_string(index=False))

# 放宽: Co>=50
print(f"\n  Co>=50 ∩ W>=3 ∩ Al>=3: {len(mt[(mt['Co']>=50)&(mt['W']>=3)&(mt['Al']>=3)])}")
print(f"  Co>=40 ∩ W>=3 ∩ Al>=3: {len(mt[(mt['Co']>=40)&(mt['W']>=3)&(mt['Al']>=3)])}")
print(f"  Co>=30 ∩ W>=3:        {len(mt[(mt['Co']>=30)&(mt['W']>=3)])}")
print(f"  Co>=30 (任意):        {len(mt[mt['Co']>=30])}")

# --- 2. 模型对该合金的预测 ---
print(f"\n[模型对 Co-7Al-8W-1Ta-4Ti 的预测]")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = load_alloy_dataset('E_pa')
builder = FeatureBuilder('E_pa')
models = load_models(builder.input_dim, 'E_pa', device)

# 构造 3 套输入
inp_no   = builder.build_input(target_alloy, None)
inp_size = builder.build_input(target_alloy, TASK_PROCESS['size'])
inp_crp_lit = builder.build_input(target_alloy,
                                  {'test_temp': 760.0, 'test_stress': 800.0})
inp_crp_orig = builder.build_input(target_alloy, TASK_PROCESS['creep'])  # 1120/137

big = np.stack([inp_no, inp_size, inp_crp_lit, inp_crp_orig], axis=0)
big_t = torch.from_numpy(big).to(device)

with torch.no_grad():
    sums = {t: torch.zeros(4, device=device) for t in TASKS.keys()}
    for m in models:
        out = m(big_t)
        for t in TASKS.keys():
            sums[t] += out[t].squeeze(-1)
    avgs = {t: (sums[t] / len(models)).cpu().numpy() for t in TASKS.keys()}

# 反归一化
print(f"\n  --- 物性 (无工艺) ---")
for t in ['solvus', 'liquidus', 'solidus', 'density', 'phase_class']:
    val = ds.denormalize_target(t, avgs[t][:1])[0]
    print(f"    {t:<12s}: {val:.3f}")
sol = ds.denormalize_target('solidus', avgs['solidus'][:1])[0]
sv  = ds.denormalize_target('solvus',  avgs['solvus'][:1])[0]
liq = ds.denormalize_target('liquidus', avgs['liquidus'][:1])[0]
print(f"    processing window  = solidus - solvus = {sol-sv:.1f} °C")
print(f"    freezing range     = liquidus - solidus = {liq-sol:.1f} °C")

print(f"\n  --- 微结构 (1240°C/24h SHT + 1100°C/168h aging) ---")
print(f"    size: {ds.denormalize_target('size', avgs['size'][1:2])[0]:.2f}")

print(f"\n  --- 蠕变预测 ---")
crp_lit  = ds.denormalize_target('creep', avgs['creep'][2:3])[0]
crp_orig = ds.denormalize_target('creep', avgs['creep'][3:4])[0]
print(f"    @760°C/800MPa  (文献工况): {crp_lit:.1f} h    (文献报道: >850 h)")
print(f"    @1120°C/137MPa (你原工况): {crp_orig:.1f} h")
