"""
Phase 2: BERT基线训练（仅MLM任务）
从零训练小型BERT用于高温合金领域
"""
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (PRETRAIN_CONFIG, BERT_CONFIG, EMBEDDINGS_DIR, 
                    MODELS_DIR, TOKENIZER_DIR, SEED)
from phase1_data.tokenizer import SuperalloyTokenizer
from phase2_pretrain.model_bert import BertForMLM
from phase2_pretrain.dataset_mlm import MLMDataset


def train_bert_base():
    """训练BERT基线模型"""
    # 设置随机种子
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载分词器
    print("Loading tokenizer...")
    tokenizer = SuperalloyTokenizer.load()
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # 创建数据集
    print("Creating MLM dataset...")
    dataset = MLMDataset(tokenizer, pa_mode=False)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=PRETRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=PRETRAIN_CONFIG.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    
    # 创建模型
    model = BertForMLM(vocab_size, BERT_CONFIG).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
    
    # 学习率调度器
    total_steps = len(dataloader) * PRETRAIN_CONFIG['num_epochs']
    warmup_steps = int(total_steps * PRETRAIN_CONFIG['warmup_ratio'])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=PRETRAIN_CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=PRETRAIN_CONFIG['warmup_ratio'],
    )
    
    # 训练循环
    print(f"\nStarting BERT-base training...")
    print(f"  Epochs: {PRETRAIN_CONFIG['num_epochs']}")
    print(f"  Batch size: {PRETRAIN_CONFIG['batch_size']}")
    print(f"  Total steps: {total_steps}")
    
    best_loss = float('inf')
    
    for epoch in range(PRETRAIN_CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{PRETRAIN_CONFIG['num_epochs']}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                output = model(input_ids, attention_mask, labels)
                loss = output['loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 
                            'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
        
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
        
        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODELS_DIR / "bert_base_best.pt")
        
        # 每10个epoch保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, MODELS_DIR / f"bert_base_epoch{epoch+1}.pt")
    
    # 保存最终模型和词向量
    torch.save(model.state_dict(), MODELS_DIR / "bert_base_final.pt")
    
    # 提取词嵌入
    embeddings = model.get_embeddings()
    np.save(EMBEDDINGS_DIR / "E_base.npy", embeddings)
    print(f"\nBERT base embeddings saved: shape = {embeddings.shape}")
    print(f"Best loss: {best_loss:.4f}")
    
    return model


if __name__ == '__main__':
    train_bert_base()
