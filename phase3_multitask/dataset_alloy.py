"""
Phase 3: 合金数据集类
基于词向量构造合金表示，支持多任务学习
"""
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (DATA_DIR, ELEMENTS, PROCESS_COLS, TASKS, EMBEDDINGS_DIR,
                    TOKENIZER_DIR, EMBEDDING_DIM, SEED)


class AlloyDataset(Dataset):
    """
    合金多任务数据集
    使用词向量构造合金的向量化表示
    """
    
    def __init__(self, master_table: pd.DataFrame,
                 embedding_matrix: np.ndarray,
                 vocab: dict,
                 task_filter=None,
                 process_keywords_embeddings: dict = None,
                 task_target_stats: dict = None):
        """
        Args:
            master_table: 主数据表
            embedding_matrix: 词向量矩阵 (vocab_size, embed_dim)
            vocab: 词表 {token: id}
            task_filter: 如果指定，只保留该任务的数据；可传字符串（单任务）或 list/set/tuple（多任务子集）
            process_keywords_embeddings: 工艺关键词的词向量字典
            task_target_stats: 预计算的每任务target统计（mean/std），
                用于跨train/val一致归一化。格式: {task: {'mean':float,'std':float}}
        """
        self.embed_dim = embedding_matrix.shape[1]
        self.embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.vocab = vocab

        # 过滤任务（支持单任务字符串或任务子集列表）
        if task_filter is not None:
            if isinstance(task_filter, (list, tuple, set)):
                allowed = set(task_filter)
                self.data = master_table[master_table['task'].isin(allowed)].reset_index(drop=True)
            else:
                self.data = master_table[master_table['task'] == task_filter].reset_index(drop=True)
        else:
            self.data = master_table.reset_index(drop=True)

        # 获取元素token ids
        self.element_ids = {elem: vocab.get(elem, -1) for elem in ELEMENTS}

        # 工艺关键词嵌入
        self.process_kw_embeddings = process_keywords_embeddings or {}

        # 预计算工艺参数的归一化参数
        process_data = self.data[PROCESS_COLS].copy()
        self.process_means = process_data.mean()
        self.process_stds = process_data.std()
        self.process_stds[self.process_stds == 0] = 1.0

        # 任务类型映射
        self.task_types = {row['task']: row['task_type'] for _, row in
                         self.data[['task', 'task_type']].drop_duplicates().iterrows()}

        # ============ target按任务归一化（仅回归任务）============
        # 优先使用外部传入的统计（避免train/val泄露）
        if task_target_stats is not None:
            self.task_target_stats = task_target_stats
        else:
            self.task_target_stats = self._compute_target_stats(master_table)

    @staticmethod
    def _compute_target_stats(master_table: pd.DataFrame) -> dict:
        """按任务计算target的mean和std（仅对回归任务）"""
        stats = {}
        for task_name, task_info in TASKS.items():
            sub = master_table[master_table['task'] == task_name]
            if len(sub) == 0:
                continue
            if task_info['type'] == 'regression':
                vals = sub['target'].dropna().values.astype(np.float64)
                mean = float(vals.mean()) if len(vals) > 0 else 0.0
                std = float(vals.std()) if len(vals) > 1 else 1.0
                if std < 1e-8:
                    std = 1.0
                stats[task_name] = {'mean': mean, 'std': std}
            else:
                # 分类任务不归一化
                stats[task_name] = {'mean': 0.0, 'std': 1.0}
        return stats

    def normalize_target(self, task_name: str, raw_value: float) -> float:
        """将原始target转为归一化值"""
        if task_name not in self.task_target_stats:
            return raw_value
        s = self.task_target_stats[task_name]
        return (raw_value - s['mean']) / s['std']

    def denormalize_target(self, task_name: str, norm_value):
        """将归一化值转回原始量纲（支持scalar或numpy数组）"""
        if task_name not in self.task_target_stats:
            return norm_value
        s = self.task_target_stats[task_name]
        return norm_value * s['std'] + s['mean']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> dict:
        row = self.data.iloc[idx]

        # 1. 构造成分向量: c_i = Σ(w_j * E[element_j])
        composition_vec = self._compute_composition_vector(row)

        # 2. 构造工艺向量
        process_vec = self._compute_process_vector(row)

        # 3. 合并为总输入
        alloy_input = torch.cat([composition_vec, process_vec])

        # 4. 目标值（回归任务进行任务级z-score归一化，分类保持原始0/1）
        task_name = row['task']
        raw_target = float(row['target'])
        if self.task_types.get(task_name) == 'regression':
            target_val = self.normalize_target(task_name, raw_target)
        else:
            target_val = raw_target
        target = torch.tensor(target_val, dtype=torch.float32)

        # 5. 任务ID
        task_id = list(TASKS.keys()).index(task_name)

        # 6. 工艺存在掩码（标记该样本是否有工艺信息）
        has_process = not row[PROCESS_COLS].isna().all()

        return {
            'input': alloy_input,
            'target': target,
            'task_id': task_id,
            'task_name': task_name,
            'has_process': has_process,
            'raw_target': raw_target,  # 保留原始target供评估反归一化使用
        }
    
    def _compute_composition_vector(self, row) -> torch.Tensor:
        """
        计算成分加权词向量
        c_i = Σ(w_j * E[element_j]) / Σ(w_j)
        w_j 为元素j的at.%
        """
        weighted_sum = torch.zeros(self.embed_dim)
        total_weight = 0.0
        
        for elem in ELEMENTS:
            weight = row.get(elem, 0.0)
            if pd.notna(weight) and weight > 0:
                elem_id = self.element_ids.get(elem, -1)
                if elem_id >= 0 and elem_id < self.embedding_matrix.shape[0]:
                    weighted_sum += weight * self.embedding_matrix[elem_id]
                    total_weight += weight
        
        if total_weight > 0:
            weighted_sum = weighted_sum / total_weight
        
        return weighted_sum
    
    def _compute_process_vector(self, row) -> torch.Tensor:
        """
        计算工艺向量
        将工艺动作词向量与归一化数值参数组合，末尾追加has_process掩码位
        """
        # 检查该样本是否有任何工艺信息
        has_process_flag = 0.0
        for col in PROCESS_COLS:
            if pd.notna(row.get(col, np.nan)):
                has_process_flag = 1.0
                break
    
        # 工艺参数向量（归一化数值）
        process_values = []
        for col in PROCESS_COLS:
            val = row.get(col, np.nan)
            if pd.notna(val):
                mean_val = self.process_means.get(col, 0.0)
                std_val = self.process_stds.get(col, 1.0)
                if pd.isna(mean_val):
                    mean_val = 0.0
                if pd.isna(std_val) or std_val == 0:
                    std_val = 1.0
                normalized = (val - mean_val) / std_val
                process_values.append(normalized)
            else:
                # 无工艺信息时填0（归一化后的均值位置），mask位会告知模型忽略
                process_values.append(0.0)
    
        process_num_vec = torch.tensor(process_values, dtype=torch.float32)
    
        # 工艺动作词向量（基于有哪些工艺参数来确定）
        process_action_vec = torch.zeros(self.embed_dim)
        n_actions = 0
    
        # 如果有固溶相关参数 -> 加入solution treatment向量
        if pd.notna(row.get('solution_temp', np.nan)):
            st_vec = self._get_process_embedding('solution treatment')
            if st_vec is not None:
                process_action_vec = process_action_vec + st_vec
                n_actions += 1
    
        # 如果有时效相关参数 -> 加入aging向量
        if pd.notna(row.get('aging_temp', np.nan)):
            ag_vec = self._get_process_embedding('aging')
            if ag_vec is not None:
                process_action_vec = process_action_vec + ag_vec
                n_actions += 1
    
        # 如果有蠕变测试条件 -> 加入creep向量
        if pd.notna(row.get('test_temp', np.nan)):
            cr_vec = self._get_process_embedding('creep')
            if cr_vec is not None:
                process_action_vec = process_action_vec + cr_vec
                n_actions += 1
    
        if n_actions > 0:
            process_action_vec = process_action_vec / n_actions
    
        # has_process标志位（1维）
        mask_vec = torch.tensor([has_process_flag], dtype=torch.float32)
    
        # 拼接：工艺动作词向量 + 归一化数值参数 + has_process掩码位
        return torch.cat([process_action_vec, process_num_vec, mask_vec])
    
    def _get_process_embedding(self, keyword: str) -> Optional[torch.Tensor]:
        """获取工艺关键词的词向量"""
        if keyword in self.process_kw_embeddings:
            return torch.tensor(self.process_kw_embeddings[keyword], dtype=torch.float32)
        
        # 从词表查找
        token = keyword.lower()
        if token in self.vocab:
            idx = self.vocab[token]
            if idx < self.embedding_matrix.shape[0]:
                return self.embedding_matrix[idx]
        return None
    
    @property
    def input_dim(self) -> int:
        """输入向量总维度 = 成分向量(embed_dim) + 工艺动作(embed_dim) + 工艺数值(n_process_cols) + has_process掩码(1)"""
        return self.embed_dim + self.embed_dim + len(PROCESS_COLS) + 1


def load_alloy_dataset(embedding_name: str = 'E_pa', 
                       task_filter=None) -> AlloyDataset:
    """
    便捷加载函数
    Args:
        embedding_name: 使用哪个词向量 ('E_pa', 'E_base', 'E_w2v')
        task_filter: 任务过滤器，可传入：
            - None: 使用全部任务
            - str:  单任务过滤
            - list/tuple/set: 任务子集过滤（用于分组多任务训练）
    """
    # 加载主表
    master_path = DATA_DIR / "master_table.csv"
    master = pd.read_csv(master_path)

    # 加载词向量
    emb_path = EMBEDDINGS_DIR / f"{embedding_name}.npy"
    embeddings = np.load(emb_path)

    # 加载词表
    vocab_path = TOKENIZER_DIR / "vocab.json"
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # 预计算全局target统计（用完整master，避免task_filter后只有一任务导致其他任务无std）
    task_target_stats = AlloyDataset._compute_target_stats(master)

    return AlloyDataset(master, embeddings, vocab, task_filter,
                        task_target_stats=task_target_stats)


class MultiTaskBatchSampler:
    """
    多任务批次采样器
    按任务比例采样，确保每个batch中包含各任务的样本
    """
    
    def __init__(self, dataset: AlloyDataset, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 按任务分组索引
        self.task_indices = {}
        for idx in range(len(dataset)):
            task = dataset.data.iloc[idx]['task']
            if task not in self.task_indices:
                self.task_indices[task] = []
            self.task_indices[task].append(idx)
        
        # 计算总batch数
        self.n_batches = max(len(indices) for indices in self.task_indices.values()) // batch_size
    
    def __iter__(self):
        # 打乱每个任务的索引
        task_iters = {}
        for task, indices in self.task_indices.items():
            if self.shuffle:
                perm = np.random.permutation(len(indices))
                task_iters[task] = [indices[i] for i in perm]
            else:
                task_iters[task] = indices.copy()
        
        # 生成batch
        for _ in range(self.n_batches):
            batch = []
            # 从每个任务中按比例取样本
            for task, indices in task_iters.items():
                n_take = max(1, self.batch_size * len(indices) // sum(len(v) for v in self.task_indices.values()))
                taken = indices[:n_take]
                task_iters[task] = indices[n_take:] + taken  # 循环
                batch.extend(taken)
            
            # 截断到batch_size
            if len(batch) > self.batch_size:
                batch = batch[:self.batch_size]
            
            yield batch
    
    def __len__(self):
        return self.n_batches


if __name__ == '__main__':
    # 测试
    print("Testing AlloyDataset...")
    try:
        dataset = load_alloy_dataset('E_pa')
        print(f"Dataset size: {len(dataset)}")
        print(f"Input dimension: {dataset.input_dim}")
        
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input shape: {sample['input'].shape}")
        print(f"Target: {sample['target']}")
        print(f"Task: {sample['task_name']}")
    except Exception as e:
        print(f"Error (expected if embeddings not yet trained): {e}")
