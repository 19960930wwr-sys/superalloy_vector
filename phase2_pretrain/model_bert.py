"""
Phase 2: BERT模型定义
从零实现小型BERT用于高温合金领域预训练
"""
import sys
import math
import torch
import torch.nn as nn
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import BERT_CONFIG


class BertEmbeddings(nn.Module):
    """BERT嵌入层：Token + Position"""
    
    def __init__(self, vocab_size: int, config: dict = None):
        super().__init__()
        config = config or BERT_CONFIG
        hidden_size = config['hidden_size']
        max_pos = config['max_position_embeddings']
        dropout = config['hidden_dropout_prob']
        
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_pos, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
        # 注册position_ids buffer
        self.register_buffer('position_ids', torch.arange(max_pos).unsqueeze(0))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        
        token_emb = self.token_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)
        
        embeddings = token_emb + position_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, config: dict = None):
        super().__init__()
        config = config or BERT_CONFIG
        self.num_heads = config['num_attention_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads
        
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores + (1.0 - extended_mask.float()) * (-10000.0)
        
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.output(context)
        return output


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, config: dict = None):
        super().__init__()
        config = config or BERT_CONFIG
        hidden_size = config['hidden_size']
        intermediate_size = config['intermediate_size']
        dropout = config['hidden_dropout_prob']
        
        self.attention = MultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.attention_dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn_dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + self.attention_dropout(attn_output))
        
        # FFN with residual
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states + self.ffn_dropout(ffn_output))
        
        return hidden_states


class BertEncoder(nn.Module):
    """BERT编码器（多层Transformer）"""
    
    def __init__(self, vocab_size: int, config: dict = None):
        super().__init__()
        config = config or BERT_CONFIG
        
        self.embeddings = BertEmbeddings(vocab_size, config)
        self.layers = nn.ModuleList([
            TransformerBlock(config) 
            for _ in range(config['num_hidden_layers'])
        ])
        self.hidden_size = config['hidden_size']
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        hidden_states = self.embeddings(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states
    
    def get_token_embeddings(self) -> nn.Embedding:
        """获取token嵌入层（用于提取词向量）"""
        return self.embeddings.token_embeddings


class BertForMLM(nn.Module):
    """BERT + MLM预测头"""
    
    def __init__(self, vocab_size: int, config: dict = None):
        super().__init__()
        config = config or BERT_CONFIG
        hidden_size = config['hidden_size']
        
        self.encoder = BertEncoder(vocab_size, config)
        
        # MLM预测头
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, vocab_size),
        )
        
        # 权重共享：MLM输出层与embedding层共享权重
        self.mlm_head[-1].weight = self.encoder.embeddings.token_embeddings.weight
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None) -> dict:
        """
        Returns:
            dict with 'loss', 'logits', 'hidden_states'
        """
        hidden_states = self.encoder(input_ids, attention_mask)
        logits = self.mlm_head(hidden_states)
        
        result = {
            'logits': logits,
            'hidden_states': hidden_states,
        }
        
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            result['loss'] = loss
        
        return result
    
    def get_embeddings(self) -> torch.Tensor:
        """获取词嵌入矩阵"""
        return self.encoder.embeddings.token_embeddings.weight.detach().cpu().numpy()


if __name__ == '__main__':
    # 测试模型
    vocab_size = 10000
    model = BertForMLM(vocab_size)
    
    # 模拟输入
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[labels > 100] = -100  # 大部分位置不计算loss
    
    output = model(input_ids, attention_mask, labels)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Hidden states shape: {output['hidden_states'].shape}")
