"""
Model-driven exploratory screening using a task-optimized model panel.

This script is intentionally separate from the seed-based screening route.
It predicts all target properties, including creep life, and therefore
produces de novo hypotheses rather than experimentally anchored candidates.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, ELEMENTS, MODELS_DIR, RESULTS_DIR, SEED, TASKS  # noqa: E402
from phase3_multitask.dataset_alloy import load_alloy_dataset  # noqa: E402
from phase3_multitask.model_multitask import MultiTaskModel, SingleTaskModel  # noqa: E402
from phase4_inverse.forward_screen import FeatureBuilder  # noqa: E402


class TaskFeatureBuilder(FeatureBuilder):
    """Feature builder using the same task-filtered statistics as training."""

    def __init__(self, embedding_name: str = "E_pa", task_filter=None):
        self.dataset = load_alloy_dataset(embedding_name, task_filter=task_filter)
        self.embed_dim = self.dataset.embed_dim
        self.embedding_matrix = self.dataset.embedding_matrix.numpy()
        self.vocab = self.dataset.vocab
        self.element_ids = {e: self.vocab.get(e, -1) for e in ELEMENTS}
        self.input_dim = self.dataset.input_dim
        self.process_means = self.dataset.process_means
        self.process_stds = self.dataset.process_stds

        self.kw_vec = {}
        for kw in ["solution treatment", "aging", "creep"]:
            if kw in self.vocab:
                vid = self.vocab[kw]
                if vid < self.embedding_matrix.shape[0]:
                    self.kw_vec[kw] = self.embedding_matrix[vid]

        filter_text = "all tasks" if task_filter is None else task_filter
        print(f"[FeatureBuilder] filter={filter_text}, input_dim={self.input_dim}, "
              f"embed_dim={self.embed_dim}, "
              f"process kw available={list(self.kw_vec.keys())}")


TASK_PROCESS = {
    "density": None,
    "liquidus": None,
    "solidus": None,
    "solvus": None,
    "phase_class": None,
    "size": {
        "solution_temp": 1240.0,
        "solution_time": 24.0,
        "aging_temp": 1100.0,
        "aging_time": 168.0,
    },
    "creep": {
        "test_temp": 1100.0,
        "test_stress": 137.0,
    },
}


MODEL_PANEL = {
    "creep": {
        "kind": "singletask",
        "tasks": ["creep"],
        "metric": "R2=0.757",
        "reason": "best creep model; avoids grouped-paG2 negative transfer",
    },
    "density": {
        "kind": "grouped",
        "prefix": "grouped-paG1",
        "ckpt_prefix": "grouped-paG1",
        "tasks": ["density", "solidus", "solvus"],
        "metric": "R2=0.841",
        "reason": "grouped MTL improves density and preserves MTL evidence",
    },
    "liquidus": {
        "kind": "multitask",
        "tasks": list(TASKS.keys()),
        "metric": "R2=0.895",
        "reason": "full Epa MTL is essentially the best liquidus model",
    },
    "solidus": {
        "kind": "singletask",
        "tasks": ["solidus"],
        "metric": "R2=0.883",
        "reason": "best solidus model; affects processing window",
    },
    "solvus": {
        "kind": "singletask",
        "tasks": ["solvus"],
        "metric": "R2=0.901",
        "reason": "best solvus model and central hard constraint",
    },
    "size": {
        "kind": "singletask",
        "tasks": ["size"],
        "metric": "R2=0.953",
        "reason": "best size model",
    },
    "phase_class": {
        "kind": "multitask",
        "tasks": list(TASKS.keys()),
        "metric": "F1=0.930",
        "reason": "neural consistency check; SVC not checkpointed for inference",
    },
}


DESIGN_BOUNDS = {
    "Ni": (50.0, 68.0),
    "Co": (4.0, 18.0),
    "Al": (5.0, 7.0),
    "W": (4.0, 9.0),
    "Ta": (5.0, 8.5),
    "Mo": (0.0, 3.5),
    "Re": (2.0, 6.0),
    "Cr": (2.0, 7.0),
    "Ru": (0.0, 4.0),
    "Hf": (0.0, 0.5),
    "Ti": (0.0, 1.0),
    "Si": (0.0, 0.5),
    "Nb": (0.0, 1.0),
}


CONSTRAINTS = [
    ("creep", ">=", 270.0),
    ("solvus", ">=", 1220.0),
    ("density", "<=", 8.9),
    ("processing_window", ">=", 80.0),
    ("phase_class", "<=", 0.5),
    ("size", "<=", 500.0),
    ("freezing_range", "<=", 60.0),
]


SCORE_WEIGHTS = {
    "creep": (+1, 1.5),
    "solvus": (+1, 1.0),
    "density": (-1, 1.0),
    "processing_window": (+1, 1.0),
    "phase_class": (-1, 0.7),
    "size": (-1, 1.0),
    "freezing_range": (-1, 1.0),
}


def collate_single(batch):
    return {
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "raw_target": torch.tensor([item["raw_target"] for item in batch],
                                   dtype=torch.float32),
    }


def checkpoint_tasks(path: Path) -> List[str]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    tasks = sorted({
        k.split(".")[1]
        for k in state
        if k.startswith("task_heads.")
    })
    return tasks


def find_grouped_prefix(requested_prefix: str,
                        required_tasks: Iterable[str],
                        embedding: str = "E_pa") -> str:
    requested = MODELS_DIR / f"{requested_prefix}_{embedding}_fold0.pt"
    required = set(required_tasks)
    if requested.exists() and set(checkpoint_tasks(requested)) == required:
        return requested_prefix

    for path in MODELS_DIR.glob(f"grouped-*_{embedding}_fold0.pt"):
        tasks = set(checkpoint_tasks(path))
        if tasks == required:
            stem = path.stem
            return stem[: -len(f"_{embedding}_fold0")]

    raise FileNotFoundError(
        f"No grouped checkpoint found for tasks={sorted(required)}")


def train_single_checkpoint(task_name: str, embedding: str, fold: int,
                            device: torch.device, epochs: int,
                            patience: int, batch_size: int,
                            force: bool = False) -> Path:
    ckpt = MODELS_DIR / f"explore_singletask-{task_name}_{embedding}_fold{fold}.pt"
    if ckpt.exists() and not force:
        return ckpt

    dataset = load_alloy_dataset(embedding, task_filter=task_name)
    idx = np.arange(len(dataset))
    y = dataset.data["target"].values
    stratify = y if TASKS[task_name]["type"] == "classification" else None
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=SEED + fold,
        stratify=stratify,
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=min(batch_size, len(train_idx)),
        shuffle=True,
        drop_last=False,
        collate_fn=collate_single,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=min(batch_size, len(val_idx)),
        shuffle=False,
        collate_fn=collate_single,
    )

    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    model = SingleTaskModel(dataset.input_dim, TASKS[task_name]["type"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["input"].to(device)
            y_norm = batch["target"].to(device)
            pred = model(x)
            loss = model.compute_loss(pred, y_norm)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["input"].to(device)
                    y_norm = batch["target"].to(device)
                    pred = model(x)
                    losses.append(float(model.compute_loss(pred, y_norm).item()))
            val_loss = float(np.mean(losses)) if losses else float("inf")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience // 10:
                    break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
    torch.save(best_state, ckpt)
    print(f"[single] saved {ckpt.name}")
    return ckpt


def load_panel_models(builders: dict, embedding: str,
                      device: torch.device, train_epochs: int,
                      force_train: bool = False):
    panel = {}
    for task, spec in MODEL_PANEL.items():
        if spec["kind"] == "multitask":
            builder = builders["full"]
            models = []
            for fold in range(5):
                ckpt = MODELS_DIR / f"multitask_{embedding}_fold{fold}.pt"
                model = MultiTaskModel(builder.input_dim).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device,
                                                 weights_only=True))
                model.eval()
                models.append(model)
            panel[task] = {"models": models, "spec": spec, "builder": builder}
        elif spec["kind"] == "grouped":
            prefix = find_grouped_prefix(spec["ckpt_prefix"], spec["tasks"],
                                         embedding)
            builder = builders[f"grouped:{prefix}"]
            models = []
            for fold in range(5):
                ckpt = MODELS_DIR / f"{prefix}_{embedding}_fold{fold}.pt"
                model = MultiTaskModel(builder.input_dim,
                                       task_subset=spec["tasks"]).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device,
                                                 weights_only=True))
                model.eval()
                models.append(model)
            spec = dict(spec)
            spec["checkpoint_prefix"] = prefix
            panel[task] = {"models": models, "spec": spec, "builder": builder}
        elif spec["kind"] == "singletask":
            builder = builders[f"singletask:{task}"]
            models = []
            for fold in range(5):
                ckpt = train_single_checkpoint(
                    task, embedding, fold, device,
                    epochs=train_epochs, patience=60, batch_size=128,
                    force=force_train,
                )
                model = SingleTaskModel(builder.input_dim,
                                        TASKS[task]["type"]).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device,
                                                 weights_only=True))
                model.eval()
                models.append(model)
            panel[task] = {"models": models, "spec": spec, "builder": builder}
    return panel


def sample_compositions(n: int, seed: int, step: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    names = list(DESIGN_BOUNDS)
    lows = np.array([DESIGN_BOUNDS[k][0] for k in names], dtype=float)
    highs = np.array([DESIGN_BOUNDS[k][1] for k in names], dtype=float)

    # Oversample because normalization and Ni-domain filters remove rows.
    target = n
    while len(rows) < target:
        batch = max(20000, int((target - len(rows)) * 2.2))
        raw = rng.uniform(lows, highs, size=(batch, len(names)))
        raw = np.round(raw / step) * step
        sums = raw.sum(axis=1)
        keep = sums > 0
        raw = raw[keep]
        sums = sums[keep]
        normed = raw * (100.0 / sums[:, None])
        normed = np.round(normed / step) * step
        # Correct rounding drift through Ni.
        drift = 100.0 - normed.sum(axis=1)
        ni_idx = names.index("Ni")
        normed[:, ni_idx] += drift
        valid = np.ones(len(normed), dtype=bool)
        for i, name in enumerate(names):
            lo, hi = DESIGN_BOUNDS[name]
            valid &= normed[:, i] >= lo - 1e-8
            valid &= normed[:, i] <= hi + 1e-8
        normed = normed[valid]
        for vals in normed:
            row = {e: 0.0 for e in ELEMENTS}
            for name, val in zip(names, vals):
                row[name] = float(round(val, 4))
            rows.append(row)
            if len(rows) >= target:
                break

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    if len(df) > n:
        df = df.sample(n, random_state=seed).reset_index(drop=True)
    return df


def exclude_reported_compositions(candidates: pd.DataFrame,
                                  decimals: int = 2) -> pd.DataFrame:
    master = pd.read_csv(DATA_DIR / "master_table.csv")
    elems = [e for e in ELEMENTS if e in candidates.columns]

    def key_frame(df):
        arr = df[elems].fillna(0.0).round(decimals)
        return arr.astype(str).agg("|".join, axis=1)

    reported = set(key_frame(master))
    keys = key_frame(candidates)
    out = candidates.loc[~keys.isin(reported)].copy().reset_index(drop=True)
    print(f"[novelty] removed {len(candidates) - len(out)} exact reported compositions")
    return out


def build_inputs(builder: FeatureBuilder, comps: pd.DataFrame,
                 process: Optional[Dict[str, float]]) -> torch.Tensor:
    arr = np.zeros((len(comps), builder.input_dim), dtype=np.float32)
    for i, row in comps.iterrows():
        comp = {e: float(row.get(e, 0.0)) for e in ELEMENTS}
        arr[i] = builder.build_input(comp, process)
    return torch.from_numpy(arr)


@torch.no_grad()
def predict_task(task: str, entry: dict, x: torch.Tensor,
                 device: torch.device, batch_size: int) -> np.ndarray:
    spec = entry["spec"]
    builder = entry["builder"]
    preds = []
    x = x.to(device)
    for model in entry["models"]:
        chunks = []
        for s in range(0, len(x), batch_size):
            xb = x[s:s + batch_size]
            if spec["kind"] == "singletask":
                out = model(xb)
            else:
                out = model(xb)[task]
            chunks.append(out.detach().cpu().numpy())
        preds.append(np.concatenate(chunks))
    mean = np.mean(np.vstack(preds), axis=0)
    if TASKS[task]["type"] == "regression":
        mean = builder.dataset.denormalize_target(task, mean)
    return mean


def predict_panel(panel: dict, comps: pd.DataFrame, device: torch.device,
                  batch_size: int) -> pd.DataFrame:
    out = comps.copy()
    for task in ["creep", "density", "liquidus", "solidus",
                 "solvus", "size", "phase_class"]:
        print(f"[predict] {task}: {panel[task]['spec']['kind']}")
        builder = panel[task]["builder"]
        x = build_inputs(builder, comps, TASK_PROCESS[task])
        out[task] = predict_task(task, panel[task], x, device, batch_size)
    out["freezing_range"] = out["liquidus"] - out["solidus"]
    out["processing_window"] = out["solidus"] - out["solvus"]
    return out


def apply_constraints(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    mask = pd.Series(True, index=df.index)
    cascade = []
    for col, op, thr in CONSTRAINTS:
        if op == ">=":
            sub = df[col] >= thr
        elif op == "<=":
            sub = df[col] <= thr
        else:
            raise ValueError(op)
        mask &= sub
        cascade.append({
            "constraint": col,
            "op": op,
            "threshold": thr,
            "pass_individual": int(sub.sum()),
            "pass_cascade": int(mask.sum()),
        })
    return df[mask].copy(), cascade


def rank_candidates(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    score = np.zeros(len(df), dtype=float)
    for col, (sign, weight) in SCORE_WEIGHTS.items():
        vals = df[col].values.astype(float)
        sd = np.nanstd(vals)
        if sd < 1e-12:
            sd = 1.0
        score += sign * weight * ((vals - np.nanmean(vals)) / sd)
    out = df.copy()
    out["score"] = score
    out = out.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="E_pa", choices=["E_pa"])
    ap.add_argument("--n_candidates", type=int, default=150000)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--train_epochs", type=int, default=220)
    ap.add_argument("--force_train", action="store_true")
    ap.add_argument("--tag", default="task_optimized")
    args = ap.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    builders = {"full": TaskFeatureBuilder(args.embedding)}
    for task, spec in MODEL_PANEL.items():
        if spec["kind"] == "singletask":
            builders[f"singletask:{task}"] = TaskFeatureBuilder(
                args.embedding, task_filter=task)
        elif spec["kind"] == "grouped":
            prefix = find_grouped_prefix(spec["ckpt_prefix"], spec["tasks"],
                                         args.embedding)
            builders[f"grouped:{prefix}"] = TaskFeatureBuilder(
                args.embedding, task_filter=spec["tasks"])

    panel = load_panel_models(builders, args.embedding, device,
                              train_epochs=args.train_epochs,
                              force_train=args.force_train)
    panel_meta = {task: entry["spec"] for task, entry in panel.items()}

    t0 = time.time()
    comps = sample_compositions(args.n_candidates, SEED, args.step)
    comps = exclude_reported_compositions(comps)
    print(f"[space] {len(comps):,} novel sampled compositions")

    pred = predict_panel(panel, comps, device, args.batch_size)
    feasible, cascade = apply_constraints(pred)
    top = rank_candidates(feasible, args.top_n) if not feasible.empty else feasible

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_path = RESULTS_DIR / f"model_driven_exploratory_all_{args.tag}.csv"
    feasible_path = RESULTS_DIR / f"model_driven_exploratory_feasible_{args.tag}.csv"
    top_path = RESULTS_DIR / f"model_driven_exploratory_top{args.top_n}_{args.tag}.csv"
    meta_path = RESULTS_DIR / f"model_driven_exploratory_top{args.top_n}_{args.tag}.json"
    pred.to_csv(all_path, index=False)
    feasible.to_csv(feasible_path, index=False)
    top.to_csv(top_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "embedding": args.embedding,
            "n_sampled": int(len(comps)),
            "n_feasible": int(len(feasible)),
            "design_bounds": DESIGN_BOUNDS,
            "task_process": TASK_PROCESS,
            "model_panel": panel_meta,
            "constraints": [
                {"property": c, "op": o, "threshold": t}
                for c, o, t in CONSTRAINTS
            ],
            "cascade": cascade,
            "score_weights": SCORE_WEIGHTS,
            "elapsed_s": time.time() - t0,
        }, f, indent=2, ensure_ascii=False)

    print("\n[constraint cascade]")
    for row in cascade:
        print(f"  {row['constraint']:18s} {row['op']} {row['threshold']:8.2f}"
              f"  individual={row['pass_individual']:7d}"
              f"  cascade={row['pass_cascade']:7d}")

    if top.empty:
        print("\nNo feasible candidate found.")
        return

    show_cols = [
        "rank", "score", "Ni", "Co", "Al", "W", "Ta", "Mo", "Re", "Cr",
        "Ru", "Hf", "Ti", "Si", "Nb", "creep", "solvus", "density",
        "processing_window", "phase_class", "size", "freezing_range",
        "liquidus", "solidus",
    ]
    print(f"\n[top] saved -> {top_path}")
    print(top[show_cols].to_string(index=False,
                                   float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
