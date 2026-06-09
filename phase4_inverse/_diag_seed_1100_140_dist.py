"""检查 1100/140 工况 seed 的各项性能预测分布,找硬约束瓶颈"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

from config import TASKS
from phase3_multitask.dataset_alloy import load_alloy_dataset
from phase4_inverse.forward_screen import FeatureBuilder, load_models, TASK_PROCESS
from phase4_inverse.seed_screen import get_seeds, predict_seeds

seeds = get_seeds(1080, 1120, 120, 160, 270)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = load_alloy_dataset('E_pa')
builder = FeatureBuilder('E_pa')
models = load_models(builder.input_dim, 'E_pa', device)
pred = predict_seeds(seeds, builder, models, device, ds)

print(f"\n[52 个 seed 的各项性能预测分布]")
for col, op, thr in [('solvus', '>=', 1220),
                     ('processing_window', '>=', 100),
                     ('density', '<=', 8.9),
                     ('phase_class', '<=', 0.5),
                     ('size', '<=', 500),
                     ('freezing_range', '<=', 60)]:
    v = pred[col].values
    pct = [np.percentile(v, p) for p in [5, 25, 50, 75, 95]]
    if op == '>=':
        passn = (v >= thr).sum()
    else:
        passn = (v <= thr).sum()
    print(f"  {col:<18s}: min={v.min():>7.2f}  p25={pct[1]:>7.2f}  "
          f"p50={pct[2]:>7.2f}  p75={pct[3]:>7.2f}  max={v.max():>7.2f}  "
          f"  通过 {op}{thr}: {passn}/52")

# 给一组宽松阈值看候选数
print(f"\n[逐渐放宽阈值的候选数]")
for cfg in [
    {'solvus':1220,'window':100,'density':8.9,'phase':0.5, 'size':500,'fr':60},
    {'solvus':1220,'window':80, 'density':8.9,'phase':0.5, 'size':500,'fr':60},
    {'solvus':1220,'window':60, 'density':8.9,'phase':0.7, 'size':600,'fr':80},
    {'solvus':1200,'window':50, 'density':9.0,'phase':0.7, 'size':600,'fr':80},
    {'solvus':1200,'window':30, 'density':9.0,'phase':0.9, 'size':800,'fr':100},
]:
    n = ((pred['solvus']>=cfg['solvus']) &
         (pred['processing_window']>=cfg['window']) &
         (pred['density']<=cfg['density']) &
         (pred['phase_class']<=cfg['phase']) &
         (pred['size']<=cfg['size']) &
         (pred['freezing_range']<=cfg['fr'])).sum()
    print(f"  {cfg}: {n}")
