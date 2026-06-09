"""
Phase 3: 多任务训练脚本
5折交叉验证，不确定性加权多任务学习
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (MULTITASK_CONFIG, TASKS, MODELS_DIR, RESULTS_DIR, 
                    SEED, EMBEDDING_DIM, PROCESS_COLS)
from phase3_multitask.dataset_alloy import load_alloy_dataset, AlloyDataset
from phase3_multitask.model_multitask import MultiTaskModel


def collate_fn(batch):
    """自定义collate函数"""
    inputs = torch.stack([item['input'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])  # 已归一化（回归）或原始0/1（分类）
    raw_targets = torch.tensor([item['raw_target'] for item in batch], dtype=torch.float32)
    task_ids = torch.tensor([item['task_id'] for item in batch], dtype=torch.long)
    return {'input': inputs, 'target': targets, 'raw_target': raw_targets, 'task_id': task_ids}


def train_one_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        x = batch['input'].to(device)
        targets = batch['target'].to(device)
        task_ids = batch['task_id'].to(device)
        
        predictions = model(x, task_ids)
        loss_dict = model.compute_loss(predictions, targets, task_ids)
        loss = loss_dict['total_loss']
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, device, dataset):
    """评估模型（回归任务反归一化后计算原始量纲指标）"""
    model.eval()
    all_preds = {task: [] for task in TASKS}
    all_targets = {task: [] for task in TASKS}
    task_names = list(TASKS.keys())

    with torch.no_grad():
        for batch in dataloader:
            x = batch['input'].to(device)
            raw_targets = batch['raw_target']  # 原始单位
            task_ids = batch['task_id']

            predictions = model(x, task_ids.to(device))

            for tid in task_ids.unique():
                mask = task_ids == tid
                task_name = task_names[tid.item()]
                if task_name in predictions:
                    all_preds[task_name].extend(predictions[task_name].cpu().numpy().tolist())
                    all_targets[task_name].extend(raw_targets[mask].numpy().tolist())

    # 计算指标
    from sklearn.metrics import r2_score, mean_squared_error, f1_score, roc_auc_score
    metrics = {}

    for task_name in TASKS:
        if not all_preds[task_name]:
            continue

        preds = np.array(all_preds[task_name])
        targets = np.array(all_targets[task_name])

        if TASKS[task_name]['type'] == 'regression':
            # 预测从归一化空间反归一化到原始量纲
            preds_denorm = dataset.denormalize_target(task_name, preds)
            rmse = np.sqrt(mean_squared_error(targets, preds_denorm))
            r2 = r2_score(targets, preds_denorm) if len(targets) > 1 else 0.0
            metrics[task_name] = {'rmse': float(rmse), 'r2': float(r2)}
        else:
            pred_labels = (preds > 0.5).astype(int)
            f1 = f1_score(targets.astype(int), pred_labels, zero_division=0)
            try:
                auc = roc_auc_score(targets.astype(int), preds)
            except Exception:
                auc = 0.0
            metrics[task_name] = {'f1': float(f1), 'auc': float(auc)}

    return metrics


def train_multitask(embedding_name: str = 'E_pa',
                    task_subset=None, group_name: str = None):
    """训练多任务模型（5折交叉验证）

    Args:
        embedding_name: 词向量名称
        task_subset: 可选任务子集（list）。传入后仅训练该子集任务。
        group_name: 分组名称（训练子集时为输出文件命名）。
            None 且 task_subset!=None 时自动生成。
            None 且 task_subset==None 时使用默认 'multitask' 输出名。
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Embedding: {embedding_name}")

    # 准备任务子集 / 输出命名
    if task_subset is None:
        active_tasks = list(TASKS.keys())
        result_prefix = 'multitask'
        model_prefix = 'multitask'
    else:
        # 保持 TASKS 原顺序
        active_tasks = [t for t in TASKS.keys() if t in set(task_subset)]
        if len(active_tasks) < 2:
            raise ValueError(f"任务子集至少需要 2 个任务，当前：{active_tasks}")
        if group_name is None:
            group_name = '+'.join([t[:3] for t in active_tasks])
        result_prefix = f'grouped-{group_name}'
        model_prefix = f'grouped-{group_name}'
        print(f"Group: {group_name}, tasks: {active_tasks}")

    # 加载数据（子集过滤）
    dataset = load_alloy_dataset(embedding_name,
                                 task_filter=active_tasks if task_subset else None)
    input_dim = dataset.input_dim
    print(f"Dataset size: {len(dataset)}, Input dim: {input_dim}")
    
    # K折交叉验证
    kf = KFold(n_splits=MULTITASK_CONFIG['n_splits'], shuffle=True, random_state=SEED)
    
    all_fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{MULTITASK_CONFIG['n_splits']}")
        print(f"{'='*50}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset, batch_size=MULTITASK_CONFIG['batch_size'],
            shuffle=True, collate_fn=collate_fn, drop_last=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=MULTITASK_CONFIG['batch_size'],
            shuffle=False, collate_fn=collate_fn
        )
        
        # 创建模型（分组场景下传入 task_subset）
        model = MultiTaskModel(input_dim,
                               task_subset=active_tasks if task_subset else None
                               ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=MULTITASK_CONFIG['learning_rate'],
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MULTITASK_CONFIG['num_epochs']
        )
        
        best_val_score = -float('inf')  # 高越好（R²或F1）
        patience_counter = 0
        
        for epoch in range(MULTITASK_CONFIG['num_epochs']):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            
            # 每10个epoch评估
            if (epoch + 1) % 10 == 0:
                val_metrics = evaluate(model, val_loader, device, dataset)
                
                # 用平均R²/F1作为验证指标
                val_scores = []
                for task, m in val_metrics.items():
                    if 'r2' in m:
                        val_scores.append(m['r2'])
                    elif 'f1' in m:
                        val_scores.append(m['f1'])
                
                avg_score = float(np.mean(val_scores)) if val_scores else -float('inf')
                
                print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                      f"avg_val_score={avg_score:.4f}")
                
                # 基于val score的早停（保存验证最优模型，而非train最低）
                if avg_score > best_val_score:
                    best_val_score = avg_score
                    patience_counter = 0
                    torch.save(model.state_dict(), 
                             MODELS_DIR / f"{model_prefix}_{embedding_name}_fold{fold}.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= MULTITASK_CONFIG['patience'] // 10:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
        
        # 加载最优模型并评估
        model.load_state_dict(
            torch.load(MODELS_DIR / f"{model_prefix}_{embedding_name}_fold{fold}.pt",
                      weights_only=True)
        )
        fold_metrics = evaluate(model, val_loader, device, dataset)
        all_fold_metrics.append(fold_metrics)
        
        print(f"\nFold {fold+1} Results:")
        for task, m in fold_metrics.items():
            print(f"  {task}: {m}")
    
    # 汇总所有fold结果（仅针对 active_tasks）
    summary = {}
    for task in active_tasks:
        task_metrics_list = [fm.get(task, {}) for fm in all_fold_metrics if task in fm]
        if not task_metrics_list:
            continue
        
        if TASKS[task]['type'] == 'regression':
            rmses = [m['rmse'] for m in task_metrics_list if 'rmse' in m]
            r2s = [m['r2'] for m in task_metrics_list if 'r2' in m]
            summary[task] = {
                'rmse_mean': float(np.mean(rmses)) if rmses else 0,
                'rmse_std': float(np.std(rmses)) if rmses else 0,
                'r2_mean': float(np.mean(r2s)) if r2s else 0,
                'r2_std': float(np.std(r2s)) if r2s else 0,
            }
        else:
            f1s = [m['f1'] for m in task_metrics_list if 'f1' in m]
            aucs = [m['auc'] for m in task_metrics_list if 'auc' in m]
            summary[task] = {
                'f1_mean': float(np.mean(f1s)) if f1s else 0,
                'f1_std': float(np.std(f1s)) if f1s else 0,
                'auc_mean': float(np.mean(aucs)) if aucs else 0,
                'auc_std': float(np.std(aucs)) if aucs else 0,
            }
    
    # 保存结果
    result_path = RESULTS_DIR / f"{result_prefix}_{embedding_name}_results.json"
    with open(result_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS ({result_prefix} | {embedding_name}):")
    print(f"{'='*50}")
    for task, m in summary.items():
        print(f"  {task}: {m}")
    print(f"\nResults saved to {result_path}")
    
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, default='E_pa',
                       choices=['E_pa', 'E_base', 'E_w2v', 'E_attr', 'E_proc'])
    parser.add_argument('--tasks', type=str, default=None,
                       help='逗号分隔的任务子集名（如 "liquidus,solidus"）。None 表示全部 7 个任务。')
    parser.add_argument('--group_name', type=str, default=None,
                       help='分组名称，出现在输出文件名中。如 "meltpoint" → grouped-meltpoint_E_pa_results.json')
    args = parser.parse_args()
    
    task_subset = None
    if args.tasks:
        task_subset = [t.strip() for t in args.tasks.split(',') if t.strip()]
    
    train_multitask(args.embedding, task_subset=task_subset,
                    group_name=args.group_name)
