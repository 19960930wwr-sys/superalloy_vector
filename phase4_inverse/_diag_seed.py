"""验证: 训练集里 760°C 附近 creep>850h 的真实样本 + 当前模型对它们的预测"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (DATA_DIR, MODELS_DIR, ELEMENTS, PROCESS_COLS,
                    TASKS, EMBEDDINGS_DIR, TOKENIZER_DIR)
from phase3_multitask.dataset_alloy import load_alloy_dataset
from phase3_multitask.model_multitask import MultiTaskModel

# --- 1. 取训练集里 760°C 附近且 creep>850h 的样本 ---
mt = pd.read_csv(DATA_DIR / "master_table.csv")
crp = mt[mt['task'] == 'creep'].copy()

# 760°C 附近 (700-820), creep>850h
hi = crp[(crp['test_temp'].between(700, 820)) & (crp['target'] > 850)]
print(f"训练集中 T∈[700,820]°C ∩ creep>850h 样本数 = {len(hi)}")
print()

# --- 2. 列出每个样本的关键元素 + 完整工艺 ---
elems_show = ['Co','Ni','Al','Ti','W','Ta','Mo','Nb','Cr','Re','Hf','Ru','C','B','Zr']
print(f"{'idx':>3s} {'T':>5s} {'σ':>5s} {'life':>7s}  | "
      + " ".join(f"{e:>5s}" for e in elems_show))
for i, (_, r) in enumerate(hi.iterrows()):
    line = f"{i:>3d} {r['test_temp']:>5.0f} {r['test_stress']:>5.0f} {r['target']:>7.0f}  | "
    line += " ".join(f"{r.get(e, 0.0):>5.2f}" for e in elems_show)
    print(line)

# --- 3. 列出非零元素 (检查是否含 wider 空间外的元素) ---
print(f"\n[每个样本的非零元素列表]")
all_elems = ELEMENTS
for i, (_, r) in enumerate(hi.iterrows()):
    nz = {e: r.get(e, 0.0) for e in all_elems if pd.notna(r.get(e, 0.0)) and r.get(e, 0.0) > 0}
    print(f"  #{i}: T={r['test_temp']:.0f} σ={r['test_stress']:.0f} life={r['target']:.0f}h "
          f"=> {nz}")

# --- 4. 用当前模型预测这些样本, 看是否能复现真值 ---
print(f"\n[当前 5 折模型对这些样本的预测]")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = load_alloy_dataset('E_pa')
input_dim = ds.input_dim

# 构造每个样本的输入向量 (与 dataset_alloy 一致)
def build_input_for_row(row):
    # 成分向量
    comp_vec = np.zeros(ds.embed_dim, dtype=np.float32)
    total = 0.0
    for e in ELEMENTS:
        w = row.get(e, 0.0)
        if pd.isna(w) or w <= 0: continue
        eid = ds.vocab.get(e, -1)
        if eid < 0 or eid >= ds.embedding_matrix.shape[0]: continue
        comp_vec += w * ds.embedding_matrix[eid].numpy()
        total += w
    if total > 0:
        comp_vec /= total

    # 工艺动作向量
    action = np.zeros(ds.embed_dim, dtype=np.float32)
    n_act = 0
    for kw, col in [('solution treatment', 'solution_temp'),
                    ('aging', 'aging_temp'),
                    ('creep', 'test_temp')]:
        if pd.notna(row.get(col, np.nan)) and kw in ds.vocab:
            vid = ds.vocab[kw]
            if vid < ds.embedding_matrix.shape[0]:
                action += ds.embedding_matrix[vid].numpy()
                n_act += 1
    if n_act > 0:
        action /= n_act

    # 工艺数值 z-score
    nums, has_any = [], False
    for col in PROCESS_COLS:
        val = row.get(col, np.nan)
        if pd.isna(val):
            nums.append(0.0)
        else:
            m = ds.process_means.get(col, 0.0)
            s = ds.process_stds.get(col, 1.0)
            m = 0.0 if pd.isna(m) else m
            s = 1.0 if (pd.isna(s) or s == 0) else s
            nums.append((val - m) / s)
            has_any = True
    nums = np.asarray(nums, dtype=np.float32)
    mask = np.array([1.0 if has_any else 0.0], dtype=np.float32)
    return np.concatenate([comp_vec, action, nums, mask])

# 加载 5 折模型
models = []
for fold in range(5):
    ckpt = MODELS_DIR / f"multitask_E_pa_fold{fold}.pt"
    if not ckpt.exists():
        continue
    m = MultiTaskModel(input_dim).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    models.append(m)
print(f"  loaded {len(models)} folds")

print(f"\n  {'idx':>3s} {'T':>4s} {'σ':>5s} {'真值':>6s} {'预测':>7s} {'误差%':>7s}")
with torch.no_grad():
    for i, (_, r) in enumerate(hi.iterrows()):
        x = torch.from_numpy(build_input_for_row(r)).unsqueeze(0).to(device)
        preds = []
        for m in models:
            out = m(x)
            preds.append(out['creep'].item())
        pred_z = np.mean(preds)
        # 反归一化
        pred = ds.denormalize_target('creep', np.array([pred_z]))[0]
        err = 100 * (pred - r['target']) / r['target']
        print(f"  {i:>3d} {r['test_temp']:>4.0f} {r['test_stress']:>5.0f} "
              f"{r['target']:>6.0f} {pred:>7.0f} {err:>+7.1f}%")
