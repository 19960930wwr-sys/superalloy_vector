"""
Phase 2: MLM 数据集类
用于BERT预训练的Masked Language Model数据集
"""
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Set, Tuple
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (CORPUS_PROCESSED_DIR, TOKENIZER_DIR, PRETRAIN_CONFIG, 
                    BERT_CONFIG, SEED)
from phase1_data.tokenizer import SuperalloyTokenizer


class MLMDataset(Dataset):
    """
    Masked Language Model 数据集
    支持PA-MLM的差异化掩码策略
    """
    
    def __init__(self, tokenizer: SuperalloyTokenizer, 
                 corpus_file: str = None,
                 max_length: int = None,
                 mask_prob: float = None,
                 element_mask_prob: float = None,
                 process_mask_prob: float = None,
                 pa_mode: bool = False):
        """
        Args:
            tokenizer: 分词器
            corpus_file: 语料文件路径
            max_length: 最大序列长度
            mask_prob: 通用掩码概率
            element_mask_prob: 元素token掩码概率
            process_mask_prob: 工艺token掩码概率
            pa_mode: 是否启用PA-MLM模式（返回元素/工艺标识）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length or BERT_CONFIG['max_position_embeddings']
        self.mask_prob = mask_prob or PRETRAIN_CONFIG['mask_prob']
        self.element_mask_prob = element_mask_prob or PRETRAIN_CONFIG['element_mask_prob']
        self.process_mask_prob = process_mask_prob or PRETRAIN_CONFIG['process_mask_prob']
        self.pa_mode = pa_mode
        
        # 特殊token ids
        self.pad_id = tokenizer.vocab.get('[PAD]', 0)
        self.mask_id = tokenizer.vocab.get('[MASK]', 4)
        self.cls_id = tokenizer.vocab.get('[CLS]', 2)
        self.sep_id = tokenizer.vocab.get('[SEP]', 3)
        
        # 元素和工艺token id集合
        self.element_ids = tokenizer.get_element_id_set()
        self.process_ids = tokenizer.get_process_id_set()
        
        # 加载语料
        corpus_file = corpus_file or str(CORPUS_PROCESSED_DIR / "corpus_tokenized.jsonl")
        self.samples = self._load_corpus(corpus_file)
        
    def _load_corpus(self, corpus_file: str) -> List[List[int]]:
        """加载并编码语料"""
        samples = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = json.loads(line.strip())
                ids = self.tokenizer.encode(tokens, add_special=True)
                if len(ids) >= 5:  # 至少5个token（含CLS和SEP）
                    samples.append(ids)
        print(f"Loaded {len(samples)} samples for MLM training")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> dict:
        """
        Returns:
            input_ids: 被mask后的输入
            labels: 原始token id（非mask位置为-100）
            attention_mask: 注意力掩码
            is_element: 该位置是否为元素token（PA-MLM用）
            is_process: 该位置是否为工艺token（PA-MLM用）
        """
        original_ids = self.samples[idx].copy()
        
        # 截断到max_length
        if len(original_ids) > self.max_length:
            original_ids = original_ids[:self.max_length - 1] + [self.sep_id]
        
        input_ids = original_ids.copy()
        labels = [-100] * len(original_ids)  # -100表示不计算loss
        is_element = [False] * len(original_ids)
        is_process = [False] * len(original_ids)
        
        # 对每个位置决定是否mask（跳过CLS和SEP）
        for i in range(1, len(original_ids) - 1):
            token_id = original_ids[i]
            
            # 判断token类型并决定掩码概率
            if token_id in self.element_ids:
                prob = self.element_mask_prob if self.pa_mode else self.mask_prob
                is_element[i] = True
            elif token_id in self.process_ids:
                prob = self.process_mask_prob if self.pa_mode else self.mask_prob
                is_process[i] = True
            else:
                prob = self.mask_prob
            
            if random.random() < prob:
                labels[i] = original_ids[i]
                
                # 80% -> [MASK], 10% -> random, 10% -> keep
                r = random.random()
                if r < 0.8:
                    input_ids[i] = self.mask_id
                elif r < 0.9:
                    input_ids[i] = random.randint(6, len(self.tokenizer.vocab) - 1)
                # else: keep original
        
        # 填充到max_length
        padding_length = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + [self.pad_id] * padding_length
        labels = labels + [-100] * padding_length
        is_element = is_element + [False] * padding_length
        is_process = is_process + [False] * padding_length
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }
        
        if self.pa_mode:
            result['is_element'] = torch.tensor(is_element, dtype=torch.bool)
            result['is_process'] = torch.tensor(is_process, dtype=torch.bool)
        
        return result


def create_mlm_dataset(pa_mode: bool = False) -> MLMDataset:
    """创建MLM数据集的便捷函数"""
    tokenizer = SuperalloyTokenizer.load()
    return MLMDataset(tokenizer, pa_mode=pa_mode)


if __name__ == '__main__':
    # 测试
    print("Testing MLM Dataset...")
    tokenizer = SuperalloyTokenizer.load()
    dataset = MLMDataset(tokenizer, pa_mode=True)
    
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"labels shape: {sample['labels'].shape}")
    print(f"Masked positions: {(sample['labels'] != -100).sum().item()}")
    print(f"Element positions: {sample['is_element'].sum().item()}")
    print(f"Process positions: {sample['is_process'].sum().item()}")
