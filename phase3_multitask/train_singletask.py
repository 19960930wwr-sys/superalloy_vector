"""
Phase 3: 单任务基线训练
为每个任务独立训练模型（对照组）
"""
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (MULTITASK_CONFIG, TASKS, MODELS_DIR, RESULTS_DIR, 
                    SEED, EMBEDDING_DIM, PROCESS_COLS)
from phase3_multitask.dataset_alloy import load_alloy_dataset
from phase3_multitask.model_multitask import SingleTaskModel


def collate_fn(batch):
    """自定义collate函数"""
    inputs = torch.stack([item['input'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])  # 已归一化（回归）或原始0/1（分类）
    raw_targets = torch.tensor([item['raw_target'] for item in batch], dtype=torch.float32)
    return {'input': inputs, 'target': targets, 'raw_target': raw_targets}


def evaluate_single(model, dataloader, device, dataset, task_name, task_type):
    """评估单任务模型，返回(metrics_dict, score)"""
    from sklearn.metrics import r2_score, mean_squared_error, f1_score, roc_auc_score
    model.eval()
    all_preds = []
    all_raw_targets = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['input'].to(device)
            pred = model(x).cpu().numpy()
            all_preds.extend(pred.tolist())
            all_raw_targets.extend(batch['raw_target'].numpy().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_raw_targets)

    if task_type == 'regression':
        preds_denorm = dataset.denormalize_target(task_name, preds)
        rmse = float(np.sqrt(mean_squared_error(targets, preds_denorm)))
        r2 = float(r2_score(targets, preds_denorm)) if len(targets) > 1 else 0.0
        return {'rmse': rmse, 'r2': r2}, r2
    else:
        pred_labels = (preds > 0.5).astype(int)
        f1 = float(f1_score(targets.astype(int), pred_labels, zero_division=0))
        try:
            auc = float(roc_auc_score(targets.astype(int), preds))
        except Exception:
            auc = 0.0
        return {'f1': f1, 'auc': auc}, f1


def train_singletask(embedding_name: str = 'E_pa'):
    """训练所有单任务基线模型"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Embedding: {embedding_name}")
    
    all_results = {}
    
    for task_name, task_info in TASKS.items():
        print(f"\n{'='*50}")
        print(f"Training Single-Task: {task_name}")
        print(f"{'='*50}")
        
        # 加载该任务数据
        dataset = load_alloy_dataset(embedding_name, task_filter=task_name)
        input_dim = dataset.input_dim
        
        if len(dataset) < 10:
            print(f"  Skipping {task_name}: too few samples ({len(dataset)})")
            continue
        
        print(f"  Samples: {len(dataset)}, Input dim: {input_dim}")
        
        # K折交叉验证
        kf = KFold(n_splits=min(MULTITASK_CONFIG['n_splits'], len(dataset) // 5),
                   shuffle=True, random_state=SEED)
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(
                train_subset, batch_size=min(MULTITASK_CONFIG['batch_size'], len(train_idx)),
                shuffle=True, collate_fn=collate_fn, drop_last=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=len(val_idx),
                shuffle=False, collate_fn=collate_fn
            )
            
            # 创建单任务模型
            model = SingleTaskModel(input_dim, task_info['type']).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=MULTITASK_CONFIG['learning_rate'],
                weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=MULTITASK_CONFIG['num_epochs']
            )
            
            best_score = -float('inf')  # 高越好
            best_state = None
            patience_counter = 0
            
            for epoch in range(MULTITASK_CONFIG['num_epochs']):
                model.train()
                epoch_loss = 0
                n_batches = 0
                
                for batch in train_loader:
                    x = batch['input'].to(device)
                    target = batch['target'].to(device)
                    
                    pred = model(x)
                    loss = model.compute_loss(pred, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                scheduler.step()
                avg_loss = epoch_loss / max(n_batches, 1)
                
                # 每10个epoch评估一次，用val score做早停
                if (epoch + 1) % 10 == 0:
                    _, score = evaluate_single(
                        model, val_loader, device, dataset, task_name, task_info['type']
                    )
                    if score > best_score:
                        best_score = score
                        patience_counter = 0
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        patience_counter += 1
                        if patience_counter >= MULTITASK_CONFIG['patience'] // 10:
                            break
            
            # 评估（使用最优验证模型，原始量纲）
            if best_state is not None:
                model.load_state_dict(best_state)
            metrics, _ = evaluate_single(
                model, val_loader, device, dataset, task_name, task_info['type']
            )
            fold_metrics.append(metrics)
        
        # 汇总
        if task_info['type'] == 'regression':
            rmses = [m['rmse'] for m in fold_metrics]
            r2s = [m['r2'] for m in fold_metrics]
            all_results[task_name] = {
                'rmse_mean': float(np.mean(rmses)),
                'rmse_std': float(np.std(rmses)),
                'r2_mean': float(np.mean(r2s)),
                'r2_std': float(np.std(r2s)),
            }
        else:
            f1s = [m['f1'] for m in fold_metrics]
            aucs = [m['auc'] for m in fold_metrics]
            all_results[task_name] = {
                'f1_mean': float(np.mean(f1s)),
                'f1_std': float(np.std(f1s)),
                'auc_mean': float(np.mean(aucs)),
                'auc_std': float(np.std(aucs)),
            }
        
        print(f"  {task_name}: {all_results[task_name]}")
    
    # 保存结果
    result_path = RESULTS_DIR / f"singletask_{embedding_name}_results.json"
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll single-task results saved to {result_path}")
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, default='E_pa',
                       choices=['E_pa', 'E_base', 'E_w2v', 'E_attr', 'E_proc'])
    args = parser.parse_args()
    
    train_singletask(args.embedding)
