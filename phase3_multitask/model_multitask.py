"""
Phase 3: 多任务学习模型
共享骨干网络 + 任务特定头
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TASKS, MULTITASK_CONFIG, EMBEDDING_DIM, PROCESS_COLS


class SharedBackbone(nn.Module):
    """共享骨干网络：多层全连接"""
    
    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.2):
        super().__init__()
        hidden_dims = hidden_dims or MULTITASK_CONFIG['hidden_dims']
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TaskHead(nn.Module):
    """任务特定头"""
    
    def __init__(self, input_dim: int, task_type: str):
        super().__init__()
        self.task_type = task_type
        
        if task_type == 'regression':
            self.head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1),
            )
        else:  # classification
            self.head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid(),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class MultiTaskModel(nn.Module):
    """
    多任务学习模型
    - 共享骨干：提取通用特征
    - 任务特定头：各任务独立预测
    - 不确定性加权：自动平衡各任务损失
    """
    
    def __init__(self, input_dim: int, config: dict = None,
                 task_subset=None):
        """
        Args:
            input_dim: 输入维度
            config: 超参配置
            task_subset: 可选任务子集（list/tuple/set）。仅为子集任务创建 head
                与 log_var；compute_loss/forward 也仅处理子集任务。None 表示全部 7 个任务。
        """
        super().__init__()
        config = config or MULTITASK_CONFIG
        
        # 任务子集（保持与 TASKS 一致的顺序，以适配全局 task_id 索引）
        if task_subset is None:
            self.active_tasks = list(TASKS.keys())
        else:
            allowed = set(task_subset)
            # 保持 TASKS 原顺序
            self.active_tasks = [t for t in TASKS.keys() if t in allowed]
        
        # 共享骨干
        self.backbone = SharedBackbone(
            input_dim, 
            config['hidden_dims'],
            config['dropout']
        )
        
        # 任务特定头（仅为 active_tasks 创建）
        self.task_heads = nn.ModuleDict()
        self.task_types = {}
        for task_name in self.active_tasks:
            task_info = TASKS[task_name]
            self.task_heads[task_name] = TaskHead(
                self.backbone.output_dim, 
                task_info['type']
            )
            self.task_types[task_name] = task_info['type']
        
        # 不确定性加权（Homoscedastic Uncertainty）
        # log_sigma^2 for each task，可学习参数
        self.log_vars = nn.ParameterDict({
            task_name: nn.Parameter(torch.zeros(1))
            for task_name in self.active_tasks
        })
        
        # 损失函数
        # SmoothL1 (Huber) 对离群点更稳健，改善高量纲任务梯度爆炸
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        # 分类任务loss放大系数：平衡多任务中分类被多个回归压制
        self.cls_amplify = 5.0
    
    def forward(self, x: torch.Tensor, task_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)
            task_ids: (batch,) 任务ID
        Returns:
            predictions: {task_name: (n_task_samples,)}
        """
        # 共享特征
        shared_features = self.backbone(x)
        
        # 如果没有task_ids，对所有任务做预测
        if task_ids is None:
            predictions = {}
            for task_name, head in self.task_heads.items():
                predictions[task_name] = head(shared_features)
            return predictions
        
        # 按任务分组预测
        predictions = {}
        task_names = list(TASKS.keys())
        for tid in task_ids.unique():
            mask = task_ids == tid
            task_name = task_names[tid.item()]
            # 跳过本模型未激活的任务（分组多任务场景下，子集外的任务不预测）
            if task_name not in self.task_heads:
                continue
            task_features = shared_features[mask]
            predictions[task_name] = self.task_heads[task_name](task_features)
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     targets: torch.Tensor, 
                     task_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算不确定性加权的多任务损失
        L_total = Σ_t (1/(2*σ_t²)) * L_t + log(σ_t)
        """
        task_names = list(TASKS.keys())
        total_loss = torch.tensor(0.0, device=targets.device)
        task_losses = {}
        
        for tid in task_ids.unique():
            mask = task_ids == tid
            task_name = task_names[tid.item()]
            
            # 跳过本模型未激活的任务（分组场景下子集外的任务无预测头）
            if task_name not in self.task_heads:
                continue
            if task_name not in predictions:
                continue
            
            pred = predictions[task_name]
            target = targets[mask]
            
            # 计算任务特定损失
            if self.task_types[task_name] == 'regression':
                loss = self.reg_loss(pred, target).mean()
            else:
                loss = self.bce_loss(pred, target).mean() * self.cls_amplify
            
            # 不确定性加权
            log_var = self.log_vars[task_name]
            precision = torch.exp(-log_var)
            weighted_loss = precision * loss + log_var
            
            total_loss = total_loss + weighted_loss.squeeze()
            task_losses[task_name] = loss.item()
        
        return {
            'total_loss': total_loss,
            'task_losses': task_losses,
        }
    
    def get_shared_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取共享特征（用于可视化和逆向设计）"""
        return self.backbone(x)


class SingleTaskModel(nn.Module):
    """单任务模型（基线对比用）"""
    
    def __init__(self, input_dim: int, task_type: str, config: dict = None):
        super().__init__()
        config = config or MULTITASK_CONFIG
        
        self.backbone = SharedBackbone(
            input_dim,
            config['hidden_dims'],
            config['dropout']
        )
        self.head = TaskHead(self.backbone.output_dim, task_type)
        self.task_type = task_type

        # SmoothL1对离群更稳健
        self.reg_loss = nn.SmoothL1Loss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.task_type == 'regression':
            return self.reg_loss(pred, target)
        else:
            return self.bce_loss(pred, target)


if __name__ == '__main__':
    # 测试
    input_dim = EMBEDDING_DIM * 2 + len(PROCESS_COLS)  # 成分向量 + 工艺向量
    print(f"Input dimension: {input_dim}")
    
    model = MultiTaskModel(input_dim)
    print(f"Multi-task model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟前向传播
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    task_ids = torch.randint(0, 7, (batch_size,))
    targets = torch.rand(batch_size)  # [0,1]范围，兼容分类任务
    
    predictions = model(x, task_ids)
    loss_dict = model.compute_loss(predictions, targets, task_ids)
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Task losses: {loss_dict['task_losses']}")
