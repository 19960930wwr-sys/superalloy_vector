"""
Phase 4: NSGA-II多目标逆向合金设计
基于训练好的多任务模型，搜索最优合金成分和工艺
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (ELEMENTS, PROCESS_COLS, TASKS, INVERSE_CONFIG, MODELS_DIR,
                    RESULTS_DIR, EMBEDDINGS_DIR, TOKENIZER_DIR, EMBEDDING_DIM, SEED)
from phase3_multitask.model_multitask import MultiTaskModel
from phase3_multitask.dataset_alloy import AlloyDataset


# 元素成分约束（at.%上下限）
ELEMENT_BOUNDS = {
    'Co': (0, 20),   'Al': (3, 15),   'W': (0, 12),   'Ni': (40, 80),
    'Ti': (0, 8),    'Cr': (3, 25),   'Ge': (0, 0),   'Ta': (0, 12),
    'B': (0, 0.5),   'Mo': (0, 8),    'Re': (0, 8),   'Nb': (0, 5),
    'Mn': (0, 0),    'Si': (0, 0),    'V': (0, 0),    'Fe': (0, 0),
    'Zr': (0, 0.5),  'Hf': (0, 2),    'Ru': (0, 6),   'Ir': (0, 0),
    'La': (0, 0),    'Y': (0, 0),     'Mg': (0, 0),   'C': (0, 0.5),
}

# 工艺参数约束
PROCESS_BOUNDS = {
    'solution_temp': (1100, 1350),    # °C
    'solution_time': (2, 48),          # h
    'aging_temp': (700, 1000),         # °C
    'aging_time': (1, 100),            # h
    'test_temp': (750, 1100),          # °C (蠕变测试温度)
    'test_stress': (100, 500),         # MPa
}


class AlloyDesignProblem:
    """合金设计优化问题定义"""
    
    def __init__(self, model: MultiTaskModel, embedding_matrix: np.ndarray,
                 vocab: dict, device: torch.device,
                 objectives: list = None, constraints: dict = None):
        """
        Args:
            model: 训练好的多任务模型（冻结）
            embedding_matrix: 词向量矩阵
            vocab: 词表
            device: 计算设备
            objectives: 优化目标 [(task_name, 'max'/'min'), ...]
            constraints: 约束条件 {task_name: (op, value), ...}
        """
        self.model = model
        self.model.eval()
        self.embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32).to(device)
        self.vocab = vocab
        self.device = device
        
        # 默认目标：最大化蠕变寿命 + 最大化γ'溶解温度
        self.objectives = objectives or [
            ('creep', 'max'),       # 最大化蠕变寿命
            ('solvus', 'max'),      # 最大化γ'溶解温度
        ]
        
        # 默认约束
        self.constraints = constraints or {
            'density': ('<=', 8.8),         # 密度 ≤ 8.8 g/cm³
            'phase_class': ('<=', 0.3),     # 有害相概率 ≤ 0.3
        }
        
        # 决策变量维度
        self.n_elements = len(ELEMENTS)
        self.n_process = len(PROCESS_COLS)
        self.n_var = self.n_elements + self.n_process
        
        # 元素token ids
        self.element_ids = [vocab.get(elem, 0) for elem in ELEMENTS]
        
        # 工艺关键词向量
        self.process_kw_vecs = {}
        for kw in ['solution treatment', 'aging', 'creep']:
            if kw in vocab and vocab[kw] < embedding_matrix.shape[0]:
                self.process_kw_vecs[kw] = embedding_matrix[vocab[kw]]
    
    def decode_individual(self, x: np.ndarray) -> tuple:
        """将优化变量解码为成分和工艺"""
        # 前n_elements个为成分（已归一化到[0,1]）
        comp_raw = x[:self.n_elements]
        
        # 将[0,1]映射到各元素的bounds范围
        composition = np.zeros(self.n_elements)
        for i, elem in enumerate(ELEMENTS):
            low, high = ELEMENT_BOUNDS[elem]
            composition[i] = low + comp_raw[i] * (high - low)
        
        # 归一化成分使总和为100
        total = composition.sum()
        if total > 0:
            composition = composition * 100.0 / total
        
        # 后n_process个为工艺参数
        process_raw = x[self.n_elements:]
        process_params = np.zeros(self.n_process)
        for i, col in enumerate(PROCESS_COLS):
            low, high = PROCESS_BOUNDS[col]
            process_params[i] = low + process_raw[i] * (high - low)
        
        return composition, process_params
    
    def compute_alloy_input(self, composition: np.ndarray, 
                           process_params: np.ndarray) -> torch.Tensor:
        """从成分和工艺构造模型输入向量"""
        embed_dim = self.embedding_matrix.shape[1]
        
        # 成分向量
        comp_vec = torch.zeros(embed_dim, device=self.device)
        total_weight = 0.0
        for i, (elem, weight) in enumerate(zip(ELEMENTS, composition)):
            if weight > 0:
                elem_id = self.element_ids[i]
                if elem_id < self.embedding_matrix.shape[0]:
                    comp_vec += weight * self.embedding_matrix[elem_id]
                    total_weight += weight
        if total_weight > 0:
            comp_vec /= total_weight
        
        # 工艺向量
        process_action_vec = torch.zeros(embed_dim, device=self.device)
        n_actions = 0
        if process_params[0] > 0:  # solution_temp
            if 'solution treatment' in self.process_kw_vecs:
                st_vec = torch.tensor(self.process_kw_vecs['solution treatment'], 
                                     dtype=torch.float32, device=self.device)
                process_action_vec += st_vec
                n_actions += 1
        if process_params[2] > 0:  # aging_temp
            if 'aging' in self.process_kw_vecs:
                ag_vec = torch.tensor(self.process_kw_vecs['aging'],
                                     dtype=torch.float32, device=self.device)
                process_action_vec += ag_vec
                n_actions += 1
        if process_params[4] > 0:  # test_temp
            if 'creep' in self.process_kw_vecs:
                cr_vec = torch.tensor(self.process_kw_vecs['creep'],
                                     dtype=torch.float32, device=self.device)
                process_action_vec += cr_vec
                n_actions += 1
        if n_actions > 0:
            process_action_vec /= n_actions
        
        # 归一化工艺数值参数
        process_num = torch.tensor(process_params, dtype=torch.float32, device=self.device)
        # 简单标准化
        process_num = process_num / torch.tensor(
            [1300, 48, 1000, 100, 1100, 500], dtype=torch.float32, device=self.device
        )
        
        # 拼接
        alloy_input = torch.cat([comp_vec, process_action_vec, process_num])
        return alloy_input
    
    def evaluate(self, X: np.ndarray) -> tuple:
        """
        评估种群
        Args:
            X: (pop_size, n_var) 种群
        Returns:
            F: (pop_size, n_obj) 目标值（pymoo最小化，所以max目标取负）
            G: (pop_size, n_constraints) 约束违背度
        """
        pop_size = X.shape[0]
        
        # 批量计算
        inputs = []
        compositions = []
        for i in range(pop_size):
            comp, proc = self.decode_individual(X[i])
            compositions.append(comp)
            inp = self.compute_alloy_input(comp, proc)
            inputs.append(inp)
        
        batch_input = torch.stack(inputs)
        
        # 模型预测
        with torch.no_grad():
            predictions = self.model(batch_input)
        
        # 计算目标值
        F = np.zeros((pop_size, len(self.objectives)))
        for j, (task_name, direction) in enumerate(self.objectives):
            if task_name in predictions:
                pred = predictions[task_name].cpu().numpy()
                F[:, j] = -pred if direction == 'max' else pred
        
        # 计算约束
        G = np.zeros((pop_size, len(self.constraints)))
        for j, (task_name, (op, value)) in enumerate(self.constraints.items()):
            if task_name in predictions:
                pred = predictions[task_name].cpu().numpy()
                if op == '<=':
                    G[:, j] = pred - value  # >0 表示违反
                elif op == '>=':
                    G[:, j] = value - pred
        
        return F, G, compositions


def run_nsga2():
    """运行NSGA-II优化"""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print("Loading multi-task model...")
    embedding_matrix = np.load(EMBEDDINGS_DIR / "E_pa.npy")
    with open(TOKENIZER_DIR / "vocab.json") as f:
        vocab = json.load(f)
    
    input_dim = EMBEDDING_DIM * 2 + len(PROCESS_COLS)
    model = MultiTaskModel(input_dim).to(device)
    model.load_state_dict(
        torch.load(MODELS_DIR / "multitask_E_pa_fold0.pt", weights_only=True,
                  map_location=device)
    )
    model.eval()
    
    # 创建问题
    problem_instance = AlloyDesignProblem(model, embedding_matrix, vocab, device)
    
    n_var = problem_instance.n_var
    n_obj = len(problem_instance.objectives)
    n_constr = len(problem_instance.constraints)
    
    class AlloyProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                n_ieq_constr=n_constr,
                xl=np.zeros(n_var),
                xu=np.ones(n_var),
            )
        
        def _evaluate(self, X, out, *args, **kwargs):
            F, G, _ = problem_instance.evaluate(X)
            out["F"] = F
            out["G"] = G
    
    # NSGA-II算法
    algorithm = NSGA2(
        pop_size=INVERSE_CONFIG['pop_size'],
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    
    print(f"\nRunning NSGA-II optimization...")
    print(f"  Variables: {n_var}")
    print(f"  Objectives: {n_obj}")
    print(f"  Constraints: {n_constr}")
    print(f"  Population: {INVERSE_CONFIG['pop_size']}")
    print(f"  Generations: {INVERSE_CONFIG['n_gen']}")
    
    problem = AlloyProblem()
    res = minimize(
        problem,
        algorithm,
        ('n_gen', INVERSE_CONFIG['n_gen']),
        seed=SEED,
        verbose=True,
    )
    
    # 从Pareto前沿选择候选
    print(f"\nPareto front size: {len(res.F)}")
    
    # 按拥挤距离选10个散布广泛的解
    n_candidates = min(INVERSE_CONFIG['n_candidates'], len(res.F))
    
    if len(res.F) <= n_candidates:
        selected_idx = list(range(len(res.F)))
    else:
        # 简单策略：均匀采样Pareto前沿
        indices = np.linspace(0, len(res.F) - 1, n_candidates, dtype=int)
        selected_idx = indices.tolist()
    
    # 解码候选合金
    candidates = []
    for i, idx in enumerate(selected_idx):
        comp, proc = problem_instance.decode_individual(res.X[idx])
        
        # 获取预测性能
        inp = problem_instance.compute_alloy_input(comp, proc)
        with torch.no_grad():
            preds = model(inp.unsqueeze(0))
        
        candidate = {
            'id': i + 1,
            'composition': {elem: float(f"{val:.2f}") for elem, val in zip(ELEMENTS, comp) if val > 0.01},
            'process': {col: float(f"{val:.1f}") for col, val in zip(PROCESS_COLS, proc)},
            'predicted_properties': {},
        }
        
        for task_name in TASKS:
            if task_name in preds:
                candidate['predicted_properties'][task_name] = float(preds[task_name].cpu().item())
        
        candidates.append(candidate)
        
        print(f"\nCandidate {i+1}:")
        print(f"  Composition: {candidate['composition']}")
        print(f"  Process: {candidate['process']}")
        print(f"  Predictions: {candidate['predicted_properties']}")
    
    # 保存结果
    result = {
        'objectives': [{'task': t, 'direction': d} for t, d in problem_instance.objectives],
        'constraints': {k: {'op': v[0], 'value': v[1]} for k, v in problem_instance.constraints.items()},
        'pareto_front_size': len(res.F),
        'candidates': candidates,
    }
    
    result_path = RESULTS_DIR / "inverse_design_candidates.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {result_path}")
    return candidates, res


if __name__ == '__main__':
    run_nsga2()
