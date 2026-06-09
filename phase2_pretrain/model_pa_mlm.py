"""
Phase 2: PA-MLM模型（属性感知掩码语言模型）
在标准MLM基础上，增加元素属性回归和工艺分类辅助任务
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (BERT_CONFIG, PRETRAIN_CONFIG, N_ELEMENT_ATTRS, ELEMENTS, 
                    EMBEDDINGS_DIR)
from phase2_pretrain.model_bert import BertEncoder
from phase2_pretrain.process_taxonomy import NUM_PROCESS_CATEGORIES


class PAMLMModel(nn.Module):
    """
    Property-Aware Masked Language Model (PA-MLM)
    
    在BERT MLM的基础上增加：
    1. 元素属性回归头：预测被mask元素的13维物理化学属性
    2. 工艺分类头：预测被mask工艺词的功能类别
    """
    
    def __init__(self, vocab_size: int, 
                 element_token_ids: list = None,
                 process_token_ids: list = None,
                 element_attr_matrix: np.ndarray = None,
                 config: dict = None):
        """
        Args:
            vocab_size: 词表大小
            element_token_ids: 元素token在词表中的id列表（按ELEMENTS顺序）
            process_token_ids: 工艺token的id列表
            element_attr_matrix: 元素属性矩阵 (N_elem, 13)，已标准化
            config: BERT配置
        """
        super().__init__()
        config = config or BERT_CONFIG
        hidden_size = config['hidden_size']
        
        # BERT编码器
        self.encoder = BertEncoder(vocab_size, config)
        
        # MLM预测头
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, vocab_size),
        )
        # 权重共享
        self.mlm_head[-1].weight = self.encoder.embeddings.token_embeddings.weight
        
        # 元素属性回归头
        self.element_attr_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, N_ELEMENT_ATTRS),
        )
        
        # 工艺分类头
        self.process_cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_PROCESS_CATEGORIES),
        )
        
        # 注册元素属性目标（不参与梯度）
        if element_attr_matrix is not None:
            self.register_buffer('element_attrs', 
                               torch.tensor(element_attr_matrix, dtype=torch.float32))
        else:
            self.register_buffer('element_attrs',
                               torch.zeros(len(ELEMENTS), N_ELEMENT_ATTRS))
        
        # 元素token_id到属性矩阵行索引的映射
        self.element_token_ids = element_token_ids or []
        self.element_id_to_idx = {}
        if element_token_ids:
            for idx, tid in enumerate(element_token_ids):
                self.element_id_to_idx[tid] = idx
        
        # 工艺token_id到类别的映射
        self.process_token_ids = process_token_ids or []
        
        # 损失函数
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.attr_loss_fn = nn.MSELoss()
        self.process_loss_fn = nn.CrossEntropyLoss()
        
        # 损失权重
        self.lambda_attr = PRETRAIN_CONFIG['lambda_attr']
        self.lambda_process = PRETRAIN_CONFIG['lambda_process']
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None,
                is_element: torch.Tensor = None,
                is_process: torch.Tensor = None,
                element_attr_labels: torch.Tensor = None,
                process_cls_labels: torch.Tensor = None) -> dict:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) MLM标签，-100为不计算
            is_element: (batch, seq_len) bool，标记元素位置
            is_process: (batch, seq_len) bool，标记工艺位置
            element_attr_labels: (N_masked_elements, 13) 元素属性目标
            process_cls_labels: (N_masked_process,) 工艺类别目标
        
        Returns:
            dict with losses and outputs
        """
        # 编码
        hidden_states = self.encoder(input_ids, attention_mask)
        
        # MLM损失
        mlm_logits = self.mlm_head(hidden_states)
        result = {
            'logits': mlm_logits,
            'hidden_states': hidden_states,
        }
        
        total_loss = torch.tensor(0.0, device=input_ids.device)
        
        if labels is not None:
            mlm_loss = self.mlm_loss_fn(
                mlm_logits.view(-1, mlm_logits.size(-1)), 
                labels.view(-1)
            )
            result['mlm_loss'] = mlm_loss
            total_loss = total_loss + mlm_loss
        
        # 元素属性回归损失
        if is_element is not None and labels is not None:
            # 找到被mask的元素位置
            masked_element_mask = is_element & (labels != -100)
            
            if masked_element_mask.any():
                # 获取这些位置的隐向量
                element_hidden = hidden_states[masked_element_mask]  # (N, hidden)
                # 预测属性
                attr_pred = self.element_attr_head(element_hidden)  # (N, 13)
                
                # 获取对应的真实属性
                masked_labels_at_elem = labels[masked_element_mask]  # token ids
                attr_targets = self._get_element_attrs(masked_labels_at_elem)
                
                if attr_targets is not None and len(attr_targets) > 0:
                    attr_loss = self.attr_loss_fn(attr_pred, attr_targets)
                    result['attr_loss'] = attr_loss
                    total_loss = total_loss + self.lambda_attr * attr_loss
        
        # 工艺分类损失
        if is_process is not None and labels is not None:
            masked_process_mask = is_process & (labels != -100)
            
            if masked_process_mask.any():
                process_hidden = hidden_states[masked_process_mask]
                process_logits = self.process_cls_head(process_hidden)
                
                # 获取工艺类别标签
                masked_labels_at_proc = labels[masked_process_mask]
                proc_targets = self._get_process_categories(masked_labels_at_proc)
                
                if proc_targets is not None and len(proc_targets) > 0:
                    valid_mask = proc_targets >= 0
                    if valid_mask.any():
                        process_loss = self.process_loss_fn(
                            process_logits[valid_mask], 
                            proc_targets[valid_mask]
                        )
                        result['process_loss'] = process_loss
                        total_loss = total_loss + self.lambda_process * process_loss
        
        result['loss'] = total_loss
        return result
    
    def _get_element_attrs(self, token_ids: torch.Tensor) -> torch.Tensor:
        """根据token_id获取对应元素的属性向量"""
        attrs = []
        for tid in token_ids.cpu().tolist():
            if tid in self.element_id_to_idx:
                idx = self.element_id_to_idx[tid]
                attrs.append(self.element_attrs[idx])
        
        if attrs:
            return torch.stack(attrs).to(token_ids.device)
        return None
    
    def _get_process_categories(self, token_ids: torch.Tensor) -> torch.Tensor:
        """根据工艺token_id获取类别标签"""
        from phase2_pretrain.process_taxonomy import TOKEN_TO_CATEGORY
        
        categories = []
        for tid in token_ids.cpu().tolist():
            # 反查token文本
            token_text = None
            for t, i in self.tokenizer_vocab.items() if hasattr(self, 'tokenizer_vocab') else []:
                if i == tid:
                    token_text = t
                    break
            
            if token_text and token_text in TOKEN_TO_CATEGORY:
                categories.append(TOKEN_TO_CATEGORY[token_text])
            else:
                categories.append(-1)
        
        return torch.tensor(categories, dtype=torch.long, device=token_ids.device)
    
    def set_tokenizer_vocab(self, vocab: dict):
        """设置词表（用于反查token文本）"""
        self.tokenizer_vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
    
    def _get_process_categories(self, token_ids: torch.Tensor) -> torch.Tensor:
        """根据工艺token_id获取类别标签"""
        from phase2_pretrain.process_taxonomy import TOKEN_TO_CATEGORY
        
        categories = []
        for tid in token_ids.cpu().tolist():
            token_text = self.id_to_token.get(tid, '') if hasattr(self, 'id_to_token') else ''
            if token_text in TOKEN_TO_CATEGORY:
                categories.append(TOKEN_TO_CATEGORY[token_text])
            else:
                categories.append(-1)
        
        return torch.tensor(categories, dtype=torch.long, device=token_ids.device)
    
    def get_embeddings(self) -> np.ndarray:
        """获取词嵌入矩阵"""
        return self.encoder.embeddings.token_embeddings.weight.detach().cpu().numpy()


if __name__ == '__main__':
    # 测试
    vocab_size = 10000
    model = PAMLMModel(vocab_size)
    
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[:, 5:10] = torch.randint(0, vocab_size, (batch_size, 5))
    is_element = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    is_process = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    output = model(input_ids, attention_mask, labels, is_element, is_process)
    print(f"PA-MLM Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total loss: {output['loss'].item():.4f}")
    print(f"MLM loss: {output.get('mlm_loss', 'N/A')}")
