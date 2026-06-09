"""
全局配置文件
高温合金领域词向量与多任务建模系统
"""
import os
from pathlib import Path

# ============ 路径配置 ============
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = PROJECT_ROOT / "corpus"
CODE_DIR = PROJECT_ROOT / "code"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 输出子目录
CORPUS_PROCESSED_DIR = OUTPUT_DIR / "corpus_processed"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# 创建所有输出目录
for d in [OUTPUT_DIR, CORPUS_PROCESSED_DIR, TOKENIZER_DIR, EMBEDDINGS_DIR,
          MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============ 元素与工艺定义 ============
# 24种高温合金元素（按数据集列出现顺序）
ELEMENTS = [
    'Co', 'Al', 'W', 'Ni', 'Ti', 'Cr', 'Ge', 'Ta', 'B', 'Mo',
    'Re', 'Nb', 'Mn', 'Si', 'V', 'Fe', 'Zr', 'Hf', 'Ru', 'Ir',
    'La', 'Y', 'Mg', 'C'
]

# 工艺参数列名（统一后）
PROCESS_COLS = [
    'solution_temp',   # 固溶温度 (°C)
    'solution_time',   # 固溶时间 (h)
    'aging_temp',      # 时效温度 (°C)
    'aging_time',      # 时效时间 (h)
    'test_temp',       # 测试温度 (°C)
    'test_stress',     # 测试应力 (MPa)
]

# 工艺动作关键词（用于分词器和PA-MLM）
PROCESS_KEYWORDS = [
    'solution treatment', 'solutioning', 'aging', 'ageing',
    'quenching', 'creep', 'rupture', 'annealing',
    'homogenization', 'precipitation', 'hardening',
    'heat treatment', 'hot isostatic pressing',
    'directional solidification', 'single crystal',
]

# 工艺条件关键词
PROCESS_CONDITION_KEYWORDS = [
    'temperature', 'time', 'stress', 'strain',
    'strain rate', 'cooling rate', 'heating rate',
    'pressure', 'duration',
]

# 7个任务定义
TASKS = {
    'density':     {'file': 'density.xlsx',     'type': 'regression', 'target': 'density'},
    'creep':       {'file': 'creep.xlsx',       'type': 'regression', 'target': 'creep_life'},
    'liquidus':    {'file': 'liquidus.xlsx',    'type': 'regression', 'target': 'liquidus_temp'},
    'phase_class': {'file': 'phase_class.xlsx', 'type': 'classification', 'target': 'phase_class'},
    'size':        {'file': 'size.xlsx',        'type': 'regression', 'target': 'gamma_prime_size'},
    'solidus':     {'file': 'solidus.xlsx',     'type': 'regression', 'target': 'solidus_temp'},
    'solvus':      {'file': 'solvus.xlsx',      'type': 'regression', 'target': 'solvus_temp'},
}

# ============ 模型超参数 ============
# 词向量维度
EMBEDDING_DIM = 128

# BERT配置（小型，适配语料规模）
BERT_CONFIG = {
    'hidden_size': 128,
    'num_hidden_layers': 6,
    'num_attention_heads': 8,
    'intermediate_size': 512,
    'max_position_embeddings': 128,  # 缩短至128，多数句子已足够
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
}

# 预训练超参数
PRETRAIN_CONFIG = {
    'batch_size': 64,           # RTX 4090 24GB可承载
    'learning_rate': 1e-3,       # 大batch对应更大lr
    'num_epochs': 15,            # 475万样本下15轮足以充分学习
    'warmup_ratio': 0.06,
    'mask_prob': 0.15,
    'element_mask_prob': 0.30,   # 元素token更高掩码概率
    'process_mask_prob': 0.30,   # 工艺token更高掩码概率
    'lambda_attr': 1.0,          # 属性回归损失权重
    'lambda_process': 0.5,       # 工艺分类损失权重
    'num_workers': 2,            # DataLoader并行加载
    'use_amp': True,             # 混合精度训练（fp16）
}

# Word2Vec超参数
W2V_CONFIG = {
    'vector_size': 128,
    'window': 5,
    'min_count': 5,
    'sg': 1,  # Skip-gram
    'workers': 8,
    'epochs': 30,
}

# 多任务学习超参数
MULTITASK_CONFIG = {
    'batch_size': 128,
    'learning_rate': 1e-3,
    'num_epochs': 500,
    'hidden_dims': [256, 128, 64],
    'dropout': 0.2,
    'n_splits': 5,  # K折交叉验证
    'patience': 60,  # 早停
}

# 逆向设计超参数
INVERSE_CONFIG = {
    'pop_size': 200,
    'n_gen': 200,
    'n_candidates': 10,
}

# 元素属性维度
N_ELEMENT_ATTRS = 13

# 随机种子
SEED = 42
