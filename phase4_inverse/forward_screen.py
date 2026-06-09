"""
Phase 4: 正向筛选 (Forward Screening) - 基于已训练的 E_pa 多任务模型
=====================================================================
在指定 Co-Ni 基设计空间内，枚举所有合法成分组合，
通过 5 折多任务模型集成预测 7 个性能，应用硬约束筛选并综合排序，
输出 top-10 候选合金供实验验证。

【设计空间】(at%, 步长 0.5)
  Co: 40 ~ 60        (a)        Ni: 30  (固定 b)
  Al: 10.5 ~ 12.5    (c)        Cr: 5   (固定 i)
  Ti: 0 ~ 3          (d)        W:  0 ~ 3        (e)
  Ta: 2 ~ 4          (f)        Mo: 0 ~ 3.5      (g)
  Nb: 0 ~ 1          (h)        Re: 0 ~ 1.5      (j)
  约束: a + 30 + c + d + e + f + g + h + 5 + j = 100
       => a = 65 - (c + d + e + f + g + h + j)

【性能硬约束】
  γ' solvus           >  1220 °C
  density             <  8.9  g/cm³
  freezing range      <  60   °C   (= Tliquidus - Tsolidus)
  processing window   >  100  °C   (= Tsolidus - Tγ'solvus)
  γ' size             <  500       (after SHT 1240°C/24h + aging 1100°C/168h)
  creep life          >  200  h    (at 1120°C / 137 MPa)

【综合排序】
  6 项目标各自标准化为 z-score，按"越优越大"取符号后加权求和，
  默认等权 (可在 SCORE_WEIGHTS 调整)。

【备注】
  · oxidation 不在 7 项训练任务内，本脚本无法预测，需另行实验或扩任务。
  · phase_class 输出"有害相概率"，作为软排序项 (越低越优)。
"""
import sys
import json
import time
import argparse
from itertools import product
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (ELEMENTS, PROCESS_COLS, TASKS, MODELS_DIR, RESULTS_DIR,
                    EMBEDDINGS_DIR, TOKENIZER_DIR, EMBEDDING_DIM, SEED)
from phase3_multitask.dataset_alloy import load_alloy_dataset
from phase3_multitask.model_multitask import MultiTaskModel


# =====================================================================
# 1. 设计空间定义（3 档预设）
# =====================================================================
STEP = 0.5

# --- tight: 用户原始定义 ---
DESIGN_SPACE_TIGHT = {
    'Co': (40.0, 60.0, False),
    'Ni': (30.0, 30.0, True),
    'Al': (10.5, 12.5, False),
    'Ti': (0.0,  3.0,  False),
    'W':  (0.0,  3.0,  False),
    'Ta': (2.0,  4.0,  False),
    'Mo': (0.0,  3.5,  False),
    'Nb': (0.0,  1.0,  False),
    'Cr': (5.0,  5.0,  True),
    'Re': (0.0,  1.5,  False),
}

# --- expanded: 放宽高点 γ' 与 creep 驱动元素上限 (Ta/W/Re), Co 下限 ---
DESIGN_SPACE_EXPANDED = {
    'Co': (35.0, 60.0, False),   # 40→35
    'Ni': (30.0, 30.0, True),
    'Al': (10.5, 12.5, False),
    'Ti': (0.0,  3.0,  False),
    'W':  (0.0,  5.0,  False),   # 3→5
    'Ta': (2.0,  6.0,  False),   # 4→6
    'Mo': (0.0,  3.5,  False),
    'Nb': (0.0,  1.0,  False),
    'Cr': (5.0,  5.0,  True),
    'Re': (0.0,  3.0,  False),   # 1.5→3 (creep 第一驱动元素)
}

# --- wider: 介于 expanded 与 wide, 实际可跑 (~5e6 raw) ---
DESIGN_SPACE_WIDER = {
    'Co': (35.0, 60.0, False),   # 40→35 (与 expanded 一致)
    'Ni': (30.0, 30.0, True),
    'Al': (10.5, 13.0, False),   # 12.5→13
    'Ti': (0.0,  3.5,  False),   # 3→3.5
    'W':  (0.0,  6.0,  False),   # 5→6
    'Ta': (2.0,  7.0,  False),   # 6→7
    'Mo': (0.0,  3.5,  False),
    'Nb': (0.0,  1.5,  False),   # 1→1.5
    'Cr': (5.0,  5.0,  True),
    'Re': (0.0,  4.0,  False),   # 3→4
    'Hf': (0.0,  0.5,  False),   # 新增微量 Hf
}

# --- wide: 进一步放宽 Cr/Ni 浮动 + Hf 微量 (接近传统镂基) ---
DESIGN_SPACE_WIDE = {
    'Co': (30.0, 60.0, False),
    'Ni': (25.0, 40.0, False),
    'Al': (10.5, 13.0, False),
    'Ti': (0.0,  3.5,  False),
    'W':  (0.0,  6.0,  False),
    'Ta': (2.0,  8.0,  False),
    'Mo': (0.0,  3.5,  False),
    'Nb': (0.0,  1.5,  False),
    'Cr': (4.0,  8.0,  False),
    'Re': (0.0,  4.0,  False),
    'Hf': (0.0,  0.5,  False),   # 阶际杯强化辅助
}

DESIGN_SPACE_PRESETS = {
    'tight':    DESIGN_SPACE_TIGHT,
    'expanded': DESIGN_SPACE_EXPANDED,
    'wider':    DESIGN_SPACE_WIDER,
    'wide':     DESIGN_SPACE_WIDE,
}

# 运行时选定（默认 tight）
DESIGN_SPACE = DESIGN_SPACE_TIGHT
FREE_VARS = []          # 在 set_design_space() 计算
FIXED_SUM = 0.0
FIXED_VARS = {}
SUM_TARGET = 100.0
CO_LO, CO_HI = 40.0, 60.0


def set_design_space(name: str):
    """选择设计空间预设。同时重算 FREE_VARS / FIXED_SUM / Co 边界。"""
    global DESIGN_SPACE, FREE_VARS, FIXED_SUM, FIXED_VARS, CO_LO, CO_HI
    if name not in DESIGN_SPACE_PRESETS:
        raise ValueError(f"unknown design space: {name}")
    DESIGN_SPACE = DESIGN_SPACE_PRESETS[name]
    # Co 作为总和反算变量；其余为枚举变量
    FREE_VARS = [v for v in DESIGN_SPACE.keys() if v != 'Co']
    FIXED_VARS = {v: lo for v, (lo, hi, fixed) in DESIGN_SPACE.items() if fixed}
    FIXED_SUM = sum(FIXED_VARS.values())  # 不含 Co
    co_lo, co_hi, _ = DESIGN_SPACE['Co']
    CO_LO, CO_HI = co_lo, co_hi
    print(f"[Space] preset='{name}', free vars (non-Co)={FREE_VARS}")
    print(f"[Space] Co range = [{CO_LO}, {CO_HI}]; fixed sum (excl. Co) = {FIXED_SUM}")


def _arange_inclusive(lo: float, hi: float, step: float) -> np.ndarray:
    """[lo, hi] 含端点的网格 (用整数 round 防 float 精度漂移)"""
    n = int(round((hi - lo) / step)) + 1
    return np.array([round(lo + i * step, 2) for i in range(n)])


def enumerate_compositions() -> pd.DataFrame:
    """枚举所有满足总和约束 + Co∈[CO_LO,CO_HI] 的 (at%) 组合"""
    grids = {}
    for v in FREE_VARS:
        lo, hi, fixed = DESIGN_SPACE[v]
        if fixed:
            grids[v] = np.array([lo])
        else:
            grids[v] = _arange_inclusive(lo, hi, STEP)
    sizes = [len(grids[v]) for v in FREE_VARS]
    print(f"[Enumerate] grid sizes: "
          f"{dict(zip(FREE_VARS, sizes))}, raw = {np.prod(sizes):,}")

    rows = []
    for combo in product(*[grids[v] for v in FREE_VARS]):
        d = dict(zip(FREE_VARS, combo))
        co = SUM_TARGET - sum(d.values())
        co = round(co, 2)
        if co < CO_LO - 1e-6 or co > CO_HI + 1e-6:
            continue
        if abs(round(co / STEP) * STEP - co) > 1e-6:
            continue
        d['Co'] = co
        rows.append(d)

    df = pd.DataFrame(rows)
    print(f"[Enumerate] feasible compositions = {len(df):,}")
    return df


# =====================================================================
# 2. 输入向量构造（与 dataset_alloy.py 一致）
# =====================================================================
class FeatureBuilder:
    """构造与训练时一致的合金输入向量 (input_dim = 2*embed + 7)"""

    def __init__(self, embedding_name: str = 'E_pa'):
        # 复用 dataset 拿到 process 归一化统计 + target 反归一化统计
        self.dataset = load_alloy_dataset(embedding_name)
        self.embed_dim = self.dataset.embed_dim
        self.embedding_matrix = self.dataset.embedding_matrix.numpy()
        self.vocab = self.dataset.vocab
        self.element_ids = {e: self.vocab.get(e, -1) for e in ELEMENTS}
        self.input_dim = self.dataset.input_dim
        self.process_means = self.dataset.process_means
        self.process_stds = self.dataset.process_stds

        # 工艺动作词向量
        self.kw_vec = {}
        for kw in ['solution treatment', 'aging', 'creep']:
            if kw in self.vocab:
                vid = self.vocab[kw]
                if vid < self.embedding_matrix.shape[0]:
                    self.kw_vec[kw] = self.embedding_matrix[vid]

        print(f"[FeatureBuilder] input_dim={self.input_dim}, "
              f"embed_dim={self.embed_dim}, "
              f"process kw available={list(self.kw_vec.keys())}")

    # ---- 成分向量 ----
    def composition_vec(self, comp_at: dict) -> np.ndarray:
        v = np.zeros(self.embed_dim, dtype=np.float32)
        total = 0.0
        for elem, w in comp_at.items():
            if w <= 0:
                continue
            eid = self.element_ids.get(elem, -1)
            if eid < 0 or eid >= self.embedding_matrix.shape[0]:
                continue
            v += w * self.embedding_matrix[eid]
            total += w
        if total > 0:
            v /= total
        return v

    # ---- 工艺向量 ----
    def process_vec(self, process: Optional[dict]) -> np.ndarray:
        """process: {'solution_temp':..,'solution_time':..,
                    'aging_temp':..,'aging_time':..,
                    'test_temp':..,'test_stress':..}
        若为 None 则返回零工艺(has_process=0)"""
        action = np.zeros(self.embed_dim, dtype=np.float32)
        n_act = 0
        if process is None:
            process = {}

        if process.get('solution_temp') is not None and 'solution treatment' in self.kw_vec:
            action += self.kw_vec['solution treatment']; n_act += 1
        if process.get('aging_temp') is not None and 'aging' in self.kw_vec:
            action += self.kw_vec['aging']; n_act += 1
        if process.get('test_temp') is not None and 'creep' in self.kw_vec:
            action += self.kw_vec['creep']; n_act += 1
        if n_act > 0:
            action /= n_act

        # 数值参数 z-score
        nums = []
        has_any = False
        for col in PROCESS_COLS:
            val = process.get(col, None)
            if val is None:
                nums.append(0.0)
            else:
                m = self.process_means.get(col, 0.0)
                s = self.process_stds.get(col, 1.0)
                m = 0.0 if pd.isna(m) else m
                s = 1.0 if (pd.isna(s) or s == 0) else s
                nums.append((val - m) / s)
                has_any = True
        nums = np.asarray(nums, dtype=np.float32)
        mask = np.array([1.0 if has_any else 0.0], dtype=np.float32)
        return np.concatenate([action, nums, mask])

    def build_input(self, comp_at: dict, process: Optional[dict]) -> np.ndarray:
        return np.concatenate([self.composition_vec(comp_at), self.process_vec(process)])


# 各任务用什么工艺
TASK_PROCESS = {
    # 物性 (无工艺): 让模型走"无工艺均值预测"路径
    'density':     None,
    'liquidus':    None,
    'solidus':     None,
    'solvus':      None,
    'phase_class': None,
    # 微结构 (用户指定 SHT + aging)
    'size': {
        'solution_temp': 1240.0, 'solution_time': 24.0,
        'aging_temp':    1100.0, 'aging_time':    168.0,
    },
    # 蠕变 (用户指定测试条件)
    'creep': {
        'test_temp': 1120.0, 'test_stress': 137.0,
    },
}


# =====================================================================
# 3. 5 折集成推理
# =====================================================================
def load_models(input_dim: int, embedding_name: str, device, n_folds: int = 5):
    """加载 5 折 multitask_E_pa_fold{0..4}.pt"""
    models = []
    for fold in range(n_folds):
        ckpt = MODELS_DIR / f"multitask_{embedding_name}_fold{fold}.pt"
        if not ckpt.exists():
            print(f"  [WARN] missing {ckpt.name}, skip")
            continue
        m = MultiTaskModel(input_dim).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.eval()
        models.append(m)
    if not models:
        raise FileNotFoundError("No multitask checkpoints found.")
    print(f"[Ensemble] loaded {len(models)} folds")
    return models


@torch.no_grad()
def predict_all(builder: FeatureBuilder, comps_df: pd.DataFrame,
                models: list, device, batch: int = 4096) -> pd.DataFrame:
    """对每个合金构造 (no-process / size-process / creep-process) 三套输入，
    经 5 折集成预测后选取相应任务，再反归一化到原始量纲。"""
    n = len(comps_df)
    print(f"[Predict] building input tensors for {n:,} alloys × 3 process variants ...")

    # 三套工艺
    inp_no   = np.zeros((n, builder.input_dim), dtype=np.float32)
    inp_size = np.zeros((n, builder.input_dim), dtype=np.float32)
    inp_crp  = np.zeros((n, builder.input_dim), dtype=np.float32)

    for i, row in comps_df.iterrows():
        comp = {e: float(row.get(e, 0.0)) for e in ELEMENTS}
        inp_no[i]   = builder.build_input(comp, None)
        inp_size[i] = builder.build_input(comp, TASK_PROCESS['size'])
        inp_crp[i]  = builder.build_input(comp, TASK_PROCESS['creep'])

    # 拼成一个大 batch: [no(n), size(n), creep(n)]
    big = np.concatenate([inp_no, inp_size, inp_crp], axis=0)
    big_t = torch.from_numpy(big).to(device)

    # 7 个任务集成预测
    task_names = list(TASKS.keys())
    sums = {t: torch.zeros(big_t.shape[0], device=device) for t in task_names}

    print(f"[Predict] running ensemble inference (folds={len(models)}, batch={batch}) ...")
    t0 = time.time()
    for fold_i, model in enumerate(models):
        for s in range(0, big_t.shape[0], batch):
            chunk = big_t[s:s + batch]
            out = model(chunk)
            for t in task_names:
                if t in out:
                    sums[t][s:s + batch] += out[t]
        print(f"  fold {fold_i + 1}/{len(models)} done  "
              f"(elapsed {time.time() - t0:.1f}s)")

    mean_preds = {t: (sums[t] / len(models)).cpu().numpy() for t in task_names}

    # 取对应工艺位置的预测
    res = pd.DataFrame(index=comps_df.index)
    res = pd.concat([comps_df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)

    # rows: 0..n-1 = no-process; n..2n-1 = size; 2n..3n-1 = creep
    def slice_pred(t, group):
        if group == 'no':   return mean_preds[t][0:n]
        if group == 'size': return mean_preds[t][n:2 * n]
        if group == 'crp':  return mean_preds[t][2 * n:3 * n]
        raise ValueError

    # 反归一化 (回归任务)
    def denorm(t, vec):
        if TASKS[t]['type'] != 'regression':
            return vec
        return builder.dataset.denormalize_target(t, vec)

    res['density']     = denorm('density',  slice_pred('density',  'no'))
    res['liquidus']    = denorm('liquidus', slice_pred('liquidus', 'no'))
    res['solidus']     = denorm('solidus',  slice_pred('solidus',  'no'))
    res['solvus']      = denorm('solvus',   slice_pred('solvus',   'no'))
    res['phase_class'] = slice_pred('phase_class', 'no')  # 概率 [0,1]
    res['size']        = denorm('size',     slice_pred('size',     'size'))
    res['creep']       = denorm('creep',    slice_pred('creep',    'crp'))

    # 派生量
    res['freezing_range']    = res['liquidus'] - res['solidus']
    res['processing_window'] = res['solidus']  - res['solvus']
    return res


# =====================================================================
# 4. 硬约束筛选 + 综合排序
# =====================================================================
HARD_CONSTRAINTS = {
    # 列名 : (op, threshold, 描述)
    'solvus':            ('>',  1220.0, "γ' solvus > 1220 °C"),
    'density':           ('<',  8.9,    "density < 8.9 g/cm³"),
    'freezing_range':    ('<',  60.0,   "Tliq - Tsol < 60 °C"),
    'processing_window': ('>',  100.0,  "Tsol - Tγ'solvus > 100 °C"),
    'size':              ('<',  500.0,  "γ' size < 500"),
    'creep':             ('>',  200.0,  "creep life > 200 h @1120°C/137MPa"),
}

# 综合排序: 各指标 z-score 后乘以方向(越大越优=+1, 越小越优=-1)，再加权求和
SCORE_WEIGHTS = {
    'solvus':            (+1, 1.0),
    'density':           (-1, 1.0),
    'freezing_range':    (-1, 1.0),
    'processing_window': (+1, 1.0),
    'size':              (-1, 1.0),
    'creep':             (+1, 1.5),  # 蠕变略加权(直接关乎服役寿命)
    'phase_class':       (-1, 0.5),  # 软项: 有害相概率越低越好
}


def apply_constraints(df: pd.DataFrame, save_dir: Optional[Path] = None,
                      embedding_name: str = 'E_pa') -> pd.DataFrame:
    """应用 6 项硬约束。同时将每项约束独立通过的合金保存到
    forward_screen_pass-{constraint}_{embedding}.csv，便于单项倒查。"""
    mask = pd.Series(True, index=df.index)
    print("\n[Filter] applying hard constraints:")
    for col, (op, thr, desc) in HARD_CONSTRAINTS.items():
        if op == '>':
            sub = df[col] > thr
        else:
            sub = df[col] < thr
        mask &= sub
        n_pass = int(sub.sum())
        print(f"  {desc:50s}  pass={n_pass:>6d}/{len(df):,}")
        # 每项约束单独保存 (仅在有样本且 save_dir 可用时)
        if save_dir is not None and n_pass > 0:
            per_path = save_dir / f"forward_screen_pass-{col}_{embedding_name}.csv"
            df.loc[sub].sort_values(col,
                ascending=(op == '<')).to_csv(per_path, index=False)
            print(f"      -> {per_path.name}  ({n_pass} alloys saved)")
    kept = df[mask].copy()
    print(f"[Filter] joint pass = {len(kept):,} / {len(df):,}")
    return kept


def rank_candidates(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """对约束通过的合金做综合打分。若 df 为空 → 返回空表"""
    if df.empty:
        return df
    score = np.zeros(len(df), dtype=np.float64)
    for col, (sign, w) in SCORE_WEIGHTS.items():
        if col not in df.columns:
            continue
        x = df[col].values.astype(np.float64)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if sd < 1e-9:
            sd = 1.0
        z = (x - mu) / sd
        score += sign * w * z
    df = df.copy()
    df['score'] = score
    df = df.sort_values('score', ascending=False).head(top_n).reset_index(drop=True)
    df.insert(0, 'rank', np.arange(1, len(df) + 1))
    return df


# =====================================================================
# 5. 主流程
# =====================================================================
def format_alloy_label(row: pd.Series) -> str:
    """Co50Ni30Al12Ti1W2Ta3Mo1Nb0.5Cr5Re1 这种字符串"""
    parts = []
    for e in ['Co', 'Ni', 'Al', 'Ti', 'W', 'Ta', 'Mo', 'Nb', 'Cr', 'Re']:
        if e in row and row[e] > 0:
            v = row[e]
            parts.append(f"{e}{v:g}")
    return "-".join(parts)


def main(embedding_name: str = 'E_pa', top_n: int = 10,
         dry_run: bool = False, max_alloys: Optional[int] = None,
         space: str = 'tight', solvus_min: Optional[float] = None,
         creep_min: Optional[float] = None,
         creep_temp: Optional[float] = None,
         creep_stress: Optional[float] = None):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_design_space(space)
    # 允许 CLI 调整 solvus 阈值 (其余 5 项保持硬约束)
    if solvus_min is not None:
        op0, _, _ = HARD_CONSTRAINTS['solvus']
        HARD_CONSTRAINTS['solvus'] = (op0, float(solvus_min),
            f"\u03b3' solvus > {solvus_min:g} \u00b0C")
        print(f"[Constraint] override solvus threshold -> {solvus_min:g} \u00b0C")
    # 允许 CLI 调整 creep 工艺与阈值
    if creep_temp is not None:
        TASK_PROCESS['creep']['test_temp'] = float(creep_temp)
        print(f"[Process] creep test_temp -> {creep_temp:g} \u00b0C")
    if creep_stress is not None:
        TASK_PROCESS['creep']['test_stress'] = float(creep_stress)
        print(f"[Process] creep test_stress -> {creep_stress:g} MPa")
    if creep_min is not None:
        op0, _, _ = HARD_CONSTRAINTS['creep']
        tt = TASK_PROCESS['creep']['test_temp']
        ts = TASK_PROCESS['creep']['test_stress']
        HARD_CONSTRAINTS['creep'] = (op0, float(creep_min),
            f"creep life > {creep_min:g} h @{tt:g}\u00b0C/{ts:g}MPa")
        print(f"[Constraint] override creep threshold -> {creep_min:g} h")
    print(f"=== Forward Screen | embedding={embedding_name} | space={space} | device={device} ===")

    # 1) 枚举设计空间
    comps = enumerate_compositions()
    if max_alloys is not None and len(comps) > max_alloys:
        comps = comps.sample(max_alloys, random_state=SEED).reset_index(drop=True)
        print(f"[Sample] subsampled to {len(comps):,}")

    # 2) 构造 feature
    builder = FeatureBuilder(embedding_name)

    if dry_run:
        # 仅构造前 4 个样本输入向量做格式校验
        small = comps.head(4).reset_index(drop=True)
        for _, r in small.iterrows():
            comp = {e: float(r.get(e, 0.0)) for e in ELEMENTS}
            v = builder.build_input(comp, TASK_PROCESS['size'])
            assert v.shape == (builder.input_dim,)
        print(f"[Dry-run] feature shape OK: ({builder.input_dim},); skip inference.")
        return

    # 3) 加载 5 折集成
    models = load_models(builder.input_dim, embedding_name, device)

    # 4) 批量预测
    pred_df = predict_all(builder, comps, models, device)

    # 4.5) 先保存全量预测供后期检查/敷衡
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_pred_csv = RESULTS_DIR / f"forward_screen_all_predictions_{embedding_name}_{space}.csv"
    pred_df.to_csv(all_pred_csv, index=False)
    print(f"[Save] all {len(pred_df):,} predictions -> {all_pred_csv.name}")

    # 5) 硬约束过滤 (同时导出每项独立通过的子集)
    feasible = apply_constraints(pred_df, save_dir=RESULTS_DIR,
                                 embedding_name=f"{embedding_name}_{space}")

    # 6) 综合排序
    top = rank_candidates(feasible, top_n=top_n)

    # 7) 落盘 (文件名带 solvus + creep 阈值后缀)
    sv_thr = HARD_CONSTRAINTS['solvus'][1]
    cr_thr = HARD_CONSTRAINTS['creep'][1]
    cr_T   = TASK_PROCESS['creep']['test_temp']
    cr_S   = TASK_PROCESS['creep']['test_stress']
    tag = (f"{embedding_name}_{space}_solvus{int(sv_thr)}"
           f"_crp{int(cr_T)}C{int(cr_S)}MPa{int(cr_thr)}h")
    out_csv  = RESULTS_DIR / f"forward_screen_top{top_n}_{tag}.csv"
    out_json = RESULTS_DIR / f"forward_screen_top{top_n}_{tag}.json"
    feas_csv = RESULTS_DIR / f"forward_screen_feasible_{tag}.csv"

    feasible.to_csv(feas_csv, index=False)
    top.to_csv(out_csv, index=False)

    if not top.empty:
        records = []
        for _, r in top.iterrows():
            rec = {
                'rank': int(r['rank']),
                'label': format_alloy_label(r),
                'composition_at_pct': {e: float(r[e]) for e in ELEMENTS
                                       if e in r and r[e] > 0},
                'predictions': {
                    'solvus_C':            float(r['solvus']),
                    'density_g_cm3':       float(r['density']),
                    'liquidus_C':          float(r['liquidus']),
                    'solidus_C':           float(r['solidus']),
                    'freezing_range_C':    float(r['freezing_range']),
                    'processing_window_C': float(r['processing_window']),
                    'gamma_prime_size':    float(r['size']),
                    'creep_life_h':        float(r['creep']),
                    'harmful_phase_prob':  float(r['phase_class']),
                },
                'process': {
                    'size_test':  TASK_PROCESS['size'],
                    'creep_test': TASK_PROCESS['creep'],
                },
                'score': float(r['score']),
            }
            records.append(rec)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({
                'embedding':        embedding_name,
                'design_space':     {k: v for k, v in DESIGN_SPACE.items()},
                'hard_constraints': {k: {'op': v[0], 'threshold': v[1],
                                         'description': v[2]}
                                     for k, v in HARD_CONSTRAINTS.items()},
                'score_weights':    {k: {'sign': v[0], 'weight': v[1]}
                                     for k, v in SCORE_WEIGHTS.items()},
                'n_enumerated':     int(len(comps)),
                'n_feasible':       int(len(feasible)),
                'top_candidates':   records,
            }, f, indent=2, ensure_ascii=False)

    # 8) 打印 top
    print(f"\n=== TOP {len(top)} CANDIDATES (saved -> {out_csv.name}) ===")
    if top.empty:
        print("  没有任何合金满足全部 6 项硬约束。")
        print("  建议：")
        print("   · 放宽 creep > 200h (E_pa 训练集上限受限) 至 150h；")
        print("   · 或放宽 solvus > 1220°C 至 1200°C；")
        print("   · 或扩大设计空间 (尤其 Ta/Re 上限)。")
    else:
        cols_show = ['rank', 'Co', 'Ni', 'Al', 'Ti', 'W', 'Ta', 'Mo', 'Nb',
                     'Cr', 'Re', 'solvus', 'density', 'freezing_range',
                     'processing_window', 'size', 'creep',
                     'phase_class', 'score']
        print(top[cols_show].to_string(index=False,
                                       float_format=lambda x: f"{x:.3f}"))

    return top


# =====================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embedding', default='E_pa', choices=['E_pa', 'E_base', 'E_w2v'])
    ap.add_argument('--top_n', type=int, default=10)
    ap.add_argument('--space', default='tight',
                    choices=list(DESIGN_SPACE_PRESETS.keys()),
                    help="设计空间预设: tight(原始)/expanded(Ta≤6,Re≤3,W≤5)/wide(进一步放宽 Cr/Ni/Hf)")
    ap.add_argument('--dry_run', action='store_true',
                    help="只构建特征，不加载权重也不推理 (本地无 ckpt 时验证脚本用)")
    ap.add_argument('--max_alloys', type=int, default=None,
                    help="限制枚举数量 (调试用)")
    ap.add_argument('--solvus_min', type=float, default=None,
                    help="覆盖 solvus 阈值 (默认 1220, 推荐 1150 走模型可信区)")
    ap.add_argument('--creep_min', type=float, default=None,
                    help="覆盖 creep 阈值 h (默认 200, 760°C/800MPa 可试 850)")
    ap.add_argument('--creep_temp', type=float, default=None,
                    help="creep 测试温度 °C (默认 1120)")
    ap.add_argument('--creep_stress', type=float, default=None,
                    help="creep 测试应力 MPa (默认 137)")
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(embedding_name=args.embedding, top_n=args.top_n,
         dry_run=args.dry_run, max_alloys=args.max_alloys, space=args.space,
         solvus_min=args.solvus_min,
         creep_min=args.creep_min,
         creep_temp=args.creep_temp,
         creep_stress=args.creep_stress)
