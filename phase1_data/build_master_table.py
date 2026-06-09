"""
Phase 1: 构建主数据表
将7个数据集合并为统一格式的 master_table.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import DATA_DIR, ELEMENTS, PROCESS_COLS, TASKS, OUTPUT_DIR


def load_and_standardize(task_name: str, task_info: dict) -> pd.DataFrame:
    """加载单个数据集并标准化列名"""
    filepath = DATA_DIR / task_info['file']
    df = pd.read_excel(filepath)
    
    # 修正列名中的空格（如 'Al ' -> 'Al'）
    df.columns = df.columns.str.strip()
    
    # 提取目标列
    target_value = df['prop'].copy()
    df = df.drop(columns=['prop'])
    
    # 统一元素列
    for elem in ELEMENTS:
        if elem not in df.columns:
            df[elem] = 0.0
    
    # 提取元素成分
    elem_data = df[ELEMENTS].fillna(0.0).copy()
    
    # 处理工艺参数列
    process_data = pd.DataFrame(index=df.index, columns=PROCESS_COLS, dtype=float)
    
    if task_name == 'size':
        # size.xlsx: St->solution_temp, Sc->solution_time, Ac->aging_temp, At->aging_time
        if 'St' in df.columns:
            process_data['solution_temp'] = df['St']
        if 'Sc' in df.columns:
            process_data['solution_time'] = df['Sc']
        if 'Ac' in df.columns:
            process_data['aging_temp'] = df['Ac']
        if 'At' in df.columns:
            process_data['aging_time'] = df['At']
    elif task_name == 'creep':
        # creep.xlsx: Temperature->test_temp, Stress->test_stress
        if 'Temperature' in df.columns:
            process_data['test_temp'] = df['Temperature']
        if 'Stress' in df.columns:
            process_data['test_stress'] = df['Stress']
    
    # 构建结果DataFrame（先确定行数，避免pandas空表赋标量陷阱）
    n_rows = len(target_value)
    result = pd.DataFrame(index=range(n_rows))
    result['task'] = task_name
    for elem in ELEMENTS:
        result[elem] = elem_data[elem].values
    for col in PROCESS_COLS:
        result[col] = process_data[col].values
    result['target'] = target_value.values
    result['target_name'] = task_info['target']
    result['task_type'] = task_info['type']

    # 丢弃target为NaN的行（原始xlsx尾部可能有空行）
    result = result.dropna(subset=['target']).reset_index(drop=True)

    return result


def build_master_table():
    """合并所有数据集为统一主表"""
    all_dfs = []
    
    for task_name, task_info in TASKS.items():
        print(f"Processing {task_name}...")
        df = load_and_standardize(task_name, task_info)
        all_dfs.append(df)
        print(f"  -> {len(df)} samples, process cols with data: "
              f"{[c for c in PROCESS_COLS if df[c].notna().any()]}")
    
    master = pd.concat(all_dfs, ignore_index=True)
    
    # 保存
    output_path = DATA_DIR / "master_table.csv"
    master.to_csv(output_path, index=False)
    print(f"\nMaster table saved to {output_path}")
    print(f"Total samples: {len(master)}")
    print(f"\nSamples per task:")
    print(master['task'].value_counts().to_string())
    
    return master


if __name__ == '__main__':
    master = build_master_table()
