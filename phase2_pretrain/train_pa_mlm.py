"""
Phase 2: PA-MLM训练脚本
训练属性感知掩码语言模型
"""
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (PRETRAIN_CONFIG, BERT_CONFIG, EMBEDDINGS_DIR, 
                    MODELS_DIR, ELEMENTS, SEED)
from phase1_data.tokenizer import SuperalloyTokenizer
from phase2_pretrain.model_pa_mlm import PAMLMModel
from phase2_pretrain.dataset_mlm import MLMDataset
from phase2_pretrain.element_attributes import get_element_attr_matrix


def train_pa_mlm():
    """训练PA-MLM模型"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载分词器
    print("Loading tokenizer...")
    tokenizer = SuperalloyTokenizer.load()
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # 获取元素token ids（按ELEMENTS顺序）
    element_token_ids = [tokenizer.vocab.get(elem, -1) for elem in ELEMENTS]
    process_token_ids = list(tokenizer.get_process_id_set())
    
    # 加载元素属性矩阵
    element_attr_matrix = get_element_attr_matrix(normalize=True)
    print(f"Element attribute matrix: {element_attr_matrix.shape}")
    
    # 创建PA-MLM数据集
    print("Creating PA-MLM dataset...")
    dataset = MLMDataset(tokenizer, pa_mode=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=PRETRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=PRETRAIN_CONFIG.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    
    # 创建PA-MLM模型
    model = PAMLMModel(
        vocab_size=vocab_size,
        element_token_ids=element_token_ids,
        process_token_ids=process_token_ids,
        element_attr_matrix=element_attr_matrix,
        config=BERT_CONFIG,
    ).to(device)
    
    # 设置词表（用于反查token文本）
    model.set_tokenizer_vocab(tokenizer.vocab)
    
    print(f"PA-MLM Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 混合精度scaler
    use_amp = PRETRAIN_CONFIG.get('use_amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"Mixed precision (AMP fp16): {use_amp}")
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=PRETRAIN_CONFIG['learning_rate'],
        weight_decay=0.01,
    )
    
    total_steps = len(dataloader) * PRETRAIN_CONFIG['num_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=PRETRAIN_CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=PRETRAIN_CONFIG['warmup_ratio'],
    )
    
    # 训练循环
    print(f"\nStarting PA-MLM training...")
    print(f"  Epochs: {PRETRAIN_CONFIG['num_epochs']}")
    print(f"  Lambda_attr: {PRETRAIN_CONFIG['lambda_attr']}")
    print(f"  Lambda_process: {PRETRAIN_CONFIG['lambda_process']}")
    
    best_loss = float('inf')
    history = {'total_loss': [], 'mlm_loss': [], 'attr_loss': [], 'process_loss': []}
    
    for epoch in range(PRETRAIN_CONFIG['num_epochs']):
        model.train()
        epoch_losses = {'total': 0, 'mlm': 0, 'attr': 0, 'process': 0}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{PRETRAIN_CONFIG['num_epochs']}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            is_element = batch['is_element'].to(device, non_blocking=True)
            is_process = batch['is_process'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                output = model(input_ids, attention_mask, labels, is_element, is_process)
                loss = output['loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_losses['total'] += loss.item()
            epoch_losses['mlm'] += output.get('mlm_loss', torch.tensor(0)).item()
            epoch_losses['attr'] += output.get('attr_loss', torch.tensor(0)).item()
            epoch_losses['process'] += output.get('process_loss', torch.tensor(0)).item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mlm': f"{output.get('mlm_loss', torch.tensor(0)).item():.4f}",
            })
        
        # 记录历史
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        history['total_loss'].append(epoch_losses['total'])
        history['mlm_loss'].append(epoch_losses['mlm'])
        history['attr_loss'].append(epoch_losses['attr'])
        history['process_loss'].append(epoch_losses['process'])
        
        print(f"  Epoch {epoch+1}: total={epoch_losses['total']:.4f}, "
              f"mlm={epoch_losses['mlm']:.4f}, attr={epoch_losses['attr']:.4f}, "
              f"proc={epoch_losses['process']:.4f}")
        
        # 保存最优模型
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            torch.save(model.state_dict(), MODELS_DIR / "pa_mlm_best.pt")
        
        # 每10个epoch保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses['total'],
            }, MODELS_DIR / f"pa_mlm_epoch{epoch+1}.pt")
    
    # 保存最终模型和词向量
    torch.save(model.state_dict(), MODELS_DIR / "pa_mlm_final.pt")
    
    # 提取词嵌入
    embeddings = model.get_embeddings()
    np.save(EMBEDDINGS_DIR / "E_pa.npy", embeddings)
    
    # 保存训练历史
    with open(MODELS_DIR / "pa_mlm_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nPA-MLM embeddings saved: shape = {embeddings.shape}")
    print(f"Best total loss: {best_loss:.4f}")
    
    return model


if __name__ == '__main__':
    train_pa_mlm()
