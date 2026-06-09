"""
Train PA-MLM ablation variants.

Variants:
  attr: MLM + elemental-attribute regression
  proc: MLM + process-category prediction
  full: MLM + both auxiliary objectives

The plain MLM baseline is trained by train_bert_base.py and saved as E_base.npy.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    BERT_CONFIG,
    ELEMENTS,
    EMBEDDINGS_DIR,
    MODELS_DIR,
    PRETRAIN_CONFIG,
    RESULTS_DIR,
    SEED,
)
from phase1_data.tokenizer import SuperalloyTokenizer  # noqa: E402
from phase2_pretrain.dataset_mlm import MLMDataset  # noqa: E402
from phase2_pretrain.element_attributes import get_element_attr_matrix  # noqa: E402
from phase2_pretrain.model_pa_mlm import PAMLMModel  # noqa: E402


VARIANTS = {
    "attr": {
        "embedding_name": "E_attr",
        "model_prefix": "pa_mlm_attr",
        "lambda_attr": 1.0,
        "lambda_process": 0.0,
        "label": "MLM + elemental attributes",
    },
    "proc": {
        "embedding_name": "E_proc",
        "model_prefix": "pa_mlm_proc",
        "lambda_attr": 0.0,
        "lambda_process": 0.5,
        "label": "MLM + process category",
    },
    "full": {
        "embedding_name": "E_pa",
        "model_prefix": "pa_mlm",
        "lambda_attr": 1.0,
        "lambda_process": 0.5,
        "label": "Full PA-MLM",
    },
}


def _save_history(history, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    rows = history.get("epochs", [])
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def train_pa_mlm_variant(
    variant: str,
    num_epochs: int | None = None,
    max_samples: int | None = None,
    suffix: str = "",
    force: bool = False,
    num_workers: int | None = None,
):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(VARIANTS)}")

    spec = dict(VARIANTS[variant])
    embedding_name = spec["embedding_name"] + suffix
    model_prefix = spec["model_prefix"] + suffix
    num_epochs = num_epochs or PRETRAIN_CONFIG["num_epochs"]
    num_workers = PRETRAIN_CONFIG.get("num_workers", 4) if num_workers is None else num_workers

    emb_path = EMBEDDINGS_DIR / f"{embedding_name}.npy"
    final_path = MODELS_DIR / f"{model_prefix}_final.pt"
    if emb_path.exists() and final_path.exists() and not force:
        print(f"[skip] Existing outputs found for {variant}: {emb_path.name}")
        return {
            "embedding_path": str(emb_path),
            "model_path": str(final_path),
            "skipped": True,
        }

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Variant: {variant} ({spec['label']})")
    print(f"Embedding output: {embedding_name}")
    print(f"Lambda_attr: {spec['lambda_attr']}")
    print(f"Lambda_process: {spec['lambda_process']}")

    print("Loading tokenizer...")
    tokenizer = SuperalloyTokenizer.load()
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")

    element_token_ids = [tokenizer.vocab.get(elem, -1) for elem in ELEMENTS]
    process_token_ids = list(tokenizer.get_process_id_set())
    element_attr_matrix = get_element_attr_matrix(normalize=True)
    print(f"Element attribute matrix: {element_attr_matrix.shape}")

    print("Creating PA-MLM dataset...")
    dataset = MLMDataset(tokenizer, pa_mode=True)
    if max_samples is not None and max_samples > 0 and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_samples)))
        print(f"Using subset for smoke/debug: {max_samples} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=PRETRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    model = PAMLMModel(
        vocab_size=vocab_size,
        element_token_ids=element_token_ids,
        process_token_ids=process_token_ids,
        element_attr_matrix=element_attr_matrix,
        config=BERT_CONFIG,
    ).to(device)
    model.set_tokenizer_vocab(tokenizer.vocab)
    model.lambda_attr = float(spec["lambda_attr"])
    model.lambda_process = float(spec["lambda_process"])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    use_amp = PRETRAIN_CONFIG.get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision (AMP fp16): {use_amp}")

    optimizer = AdamW(
        model.parameters(),
        lr=PRETRAIN_CONFIG["learning_rate"],
        weight_decay=0.01,
    )
    total_steps = max(1, len(dataloader) * num_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=PRETRAIN_CONFIG["learning_rate"],
        total_steps=total_steps,
        pct_start=PRETRAIN_CONFIG["warmup_ratio"],
    )

    history = {
        "variant": variant,
        "label": spec["label"],
        "embedding_name": embedding_name,
        "model_prefix": model_prefix,
        "lambda_attr": float(spec["lambda_attr"]),
        "lambda_process": float(spec["lambda_process"]),
        "num_epochs": int(num_epochs),
        "max_samples": max_samples,
        "epochs": [],
    }
    best_loss = float("inf")

    print("\nStarting PA-MLM ablation training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batches per epoch: {len(dataloader)}")
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {"total": 0.0, "mlm": 0.0, "attr": 0.0, "process": 0.0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"{variant} epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            is_element = batch["is_element"].to(device, non_blocking=True)
            is_process = batch["is_process"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                output = model(input_ids, attention_mask, labels, is_element, is_process)
                loss = output["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_losses["total"] += float(loss.item())
            epoch_losses["mlm"] += float(output.get("mlm_loss", torch.tensor(0.0)).item())
            epoch_losses["attr"] += float(output.get("attr_loss", torch.tensor(0.0)).item())
            epoch_losses["process"] += float(output.get("process_loss", torch.tensor(0.0)).item())
            num_batches += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "mlm": f"{output.get('mlm_loss', torch.tensor(0.0)).item():.4f}",
                }
            )

        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        row = {
            "epoch": epoch + 1,
            "total_loss": epoch_losses["total"],
            "mlm_loss": epoch_losses["mlm"],
            "element_attribute_loss": epoch_losses["attr"],
            "process_category_loss": epoch_losses["process"],
            "weighted_element_attribute_loss": spec["lambda_attr"] * epoch_losses["attr"],
            "weighted_process_category_loss": spec["lambda_process"] * epoch_losses["process"],
        }
        history["epochs"].append(row)
        print(
            f"  Epoch {epoch + 1}: total={epoch_losses['total']:.4f}, "
            f"mlm={epoch_losses['mlm']:.4f}, attr={epoch_losses['attr']:.4f}, "
            f"proc={epoch_losses['process']:.4f}"
        )

        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            torch.save(model.state_dict(), MODELS_DIR / f"{model_prefix}_best.pt")

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_losses["total"],
                    "variant": variant,
                },
                MODELS_DIR / f"{model_prefix}_epoch{epoch + 1}.pt",
            )

    torch.save(model.state_dict(), final_path)
    embeddings = model.get_embeddings()
    np.save(emb_path, embeddings)

    history["best_total_loss"] = best_loss
    history_json = MODELS_DIR / f"{model_prefix}_history.json"
    history_csv = RESULTS_DIR / f"{model_prefix}_history.csv"
    _save_history(history, history_json, history_csv)

    print(f"\nEmbeddings saved: {emb_path} shape={embeddings.shape}")
    print(f"Final model saved: {final_path}")
    print(f"History saved: {history_json}")
    return {
        "embedding_path": str(emb_path),
        "model_path": str(final_path),
        "history_path": str(history_json),
        "best_total_loss": float(best_loss),
        "skipped": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Train PA-MLM ablation variants.")
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--suffix", default="", help="Optional suffix for smoke tests.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    train_pa_mlm_variant(
        args.variant,
        num_epochs=args.epochs,
        max_samples=args.max_samples,
        suffix=args.suffix,
        force=args.force,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
