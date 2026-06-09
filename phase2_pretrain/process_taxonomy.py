"""
Phase 2: 工艺功能类别定义
定义工艺token到功能类别的映射
"""
import sys
from pathlib import Path
from typing import Dict, List
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PROCESS_KEYWORDS, PROCESS_CONDITION_KEYWORDS

# 工艺功能类别定义
PROCESS_CATEGORIES = {
    'heat_treatment': [        # 热处理动作
        'solution treatment', 'solutioning', 'annealing',
        'homogenization', 'aging', 'ageing',
        'heat treatment', 'precipitation', 'hardening',
    ],
    'cooling_method': [        # 冷却方式
        'quenching',
    ],
    'mechanical_test': [       # 力学测试
        'creep', 'rupture',
    ],
    'processing': [            # 加工制备
        'hot isostatic pressing',
        'directional solidification', 'single crystal',
    ],
    'condition_param': [       # 条件参数
        'temperature', 'time', 'stress', 'strain',
        'strain rate', 'cooling rate', 'heating rate',
        'pressure', 'duration',
    ],
}

# 反向映射：token -> category_id
TOKEN_TO_CATEGORY: Dict[str, int] = {}
CATEGORY_NAMES: List[str] = list(PROCESS_CATEGORIES.keys())

for cat_id, (cat_name, tokens) in enumerate(PROCESS_CATEGORIES.items()):
    for token in tokens:
        TOKEN_TO_CATEGORY[token.lower()] = cat_id

NUM_PROCESS_CATEGORIES = len(CATEGORY_NAMES)


def get_process_category(token: str) -> int:
    """获取工艺token对应的功能类别ID"""
    return TOKEN_TO_CATEGORY.get(token.lower(), -1)


def get_category_name(cat_id: int) -> str:
    """获取类别名称"""
    if 0 <= cat_id < len(CATEGORY_NAMES):
        return CATEGORY_NAMES[cat_id]
    return 'unknown'


def print_taxonomy():
    """打印工艺分类体系"""
    print("Process Taxonomy:")
    print(f"Total categories: {NUM_PROCESS_CATEGORIES}")
    print()
    for cat_id, (cat_name, tokens) in enumerate(PROCESS_CATEGORIES.items()):
        print(f"  [{cat_id}] {cat_name}:")
        for t in tokens:
            print(f"       - {t}")
        print()


if __name__ == '__main__':
    print_taxonomy()
    
    # 测试
    test_tokens = ['solution treatment', 'creep', 'temperature', 'quenching', 'aging']
    print("\nTest token -> category mapping:")
    for t in test_tokens:
        cat_id = get_process_category(t)
        print(f"  '{t}' -> [{cat_id}] {get_category_name(cat_id)}")
