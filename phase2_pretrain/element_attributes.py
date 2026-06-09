"""
Phase 2: 元素属性矩阵
收集24种高温合金元素的13维物理化学属性
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import ELEMENTS, N_ELEMENT_ATTRS, EMBEDDINGS_DIR

# 24种元素的13维属性数据
# 属性: [原子序数, 电负性(Pauling), 原子半径(pm), 第一电离能(kJ/mol), 
#        熔点(K), 密度(g/cm3), 价电子数, 杨氏模量(GPa), 剪切模量(GPa),
#        磁矩(μB), 常见化合价, d电子数, Miedema电负性(V)]
ELEMENT_PROPERTIES = {
    'Co': [27, 1.88, 125, 760.4, 1768, 8.90, 9, 209, 75, 1.72, 2, 7, 5.10],
    'Al': [13, 1.61, 143, 577.5, 933, 2.70, 3, 70, 26, 0.00, 3, 0, 4.20],
    'W':  [74, 2.36, 139, 770.0, 3695, 19.25, 6, 411, 161, 0.00, 6, 4, 4.80],
    'Ni': [28, 1.91, 124, 737.1, 1728, 8.91, 10, 200, 76, 0.61, 2, 8, 5.20],
    'Ti': [22, 1.54, 147, 658.8, 1941, 4.51, 4, 116, 44, 0.00, 4, 2, 3.65],
    'Cr': [24, 1.66, 128, 652.9, 2180, 7.19, 6, 279, 115, 0.00, 3, 5, 4.65],
    'Ge': [32, 2.01, 122, 762.2, 1211, 5.32, 4, 103, 41, 0.00, 4, 0, 4.55],
    'Ta': [73, 1.50, 146, 761.0, 3290, 16.65, 5, 186, 69, 0.00, 5, 3, 4.05],
    'B':  [5, 2.04, 87, 800.6, 2349, 2.34, 3, 185, 0, 0.00, 3, 0, 4.75],
    'Mo': [42, 2.16, 139, 684.3, 2896, 10.28, 6, 329, 126, 0.00, 6, 4, 4.65],
    'Re': [75, 1.90, 137, 760.0, 3459, 21.02, 7, 463, 178, 0.00, 4, 5, 5.20],
    'Nb': [41, 1.60, 146, 652.1, 2750, 8.57, 5, 105, 38, 0.00, 5, 3, 4.00],
    'Mn': [25, 1.55, 127, 717.3, 1519, 7.47, 7, 198, 0, 0.00, 2, 5, 4.45],
    'Si': [14, 1.90, 117, 786.5, 1687, 2.33, 4, 130, 0, 0.00, 4, 0, 4.70],
    'V':  [23, 1.63, 134, 650.9, 2183, 6.11, 5, 128, 47, 0.00, 5, 3, 4.25],
    'Fe': [26, 1.83, 126, 762.5, 1811, 7.87, 8, 211, 82, 2.22, 3, 6, 4.93],
    'Zr': [40, 1.33, 160, 640.1, 2128, 6.51, 4, 88, 33, 0.00, 4, 2, 3.45],
    'Hf': [72, 1.30, 159, 658.5, 2506, 13.31, 4, 78, 30, 0.00, 4, 2, 3.55],
    'Ru': [44, 2.20, 134, 710.2, 2607, 12.37, 8, 447, 173, 0.00, 3, 6, 5.40],
    'Ir': [77, 2.20, 136, 880.0, 2719, 22.56, 9, 528, 210, 0.00, 4, 7, 5.55],
    'La': [57, 1.10, 187, 538.1, 1193, 6.16, 3, 37, 14, 0.00, 3, 1, 3.17],
    'Y':  [39, 1.22, 180, 600.0, 1799, 4.47, 3, 64, 26, 0.00, 3, 1, 3.20],
    'Mg': [12, 1.31, 160, 737.7, 923, 1.74, 2, 45, 17, 0.00, 2, 0, 3.45],
    'C':  [6, 2.55, 77, 1086.5, 3823, 2.27, 4, 1050, 0, 0.00, 4, 0, 5.70],
}


def get_element_attr_matrix(normalize: bool = True) -> np.ndarray:
    """
    获取元素属性矩阵 (24 x 13)
    
    Args:
        normalize: 是否Z-score标准化
    
    Returns:
        E_attr: shape (24, 13)
    """
    matrix = np.array([ELEMENT_PROPERTIES[elem] for elem in ELEMENTS], dtype=np.float32)
    
    if normalize:
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std == 0] = 1.0  # 避免除零
        matrix = (matrix - mean) / std
    
    return matrix


def get_element_attr_df() -> pd.DataFrame:
    """获取元素属性DataFrame（含列名）"""
    attr_names = [
        'atomic_number', 'electronegativity', 'atomic_radius', 
        'first_ionization', 'melting_point', 'density',
        'valence_electrons', 'youngs_modulus', 'shear_modulus',
        'magnetic_moment', 'common_valence', 'd_electrons',
        'miedema_electronegativity'
    ]
    
    matrix = np.array([ELEMENT_PROPERTIES[elem] for elem in ELEMENTS], dtype=np.float32)
    df = pd.DataFrame(matrix, index=ELEMENTS, columns=attr_names)
    return df


def save_element_attrs():
    """保存元素属性矩阵"""
    # 原始属性
    df = get_element_attr_df()
    df.to_csv(EMBEDDINGS_DIR / "element_attributes_raw.csv")
    
    # 标准化属性
    matrix_norm = get_element_attr_matrix(normalize=True)
    np.save(EMBEDDINGS_DIR / "element_attributes_normalized.npy", matrix_norm)
    
    print("Element attribute matrix shape:", matrix_norm.shape)
    print("\nElement attributes (first 5 elements):")
    print(df.head())
    
    return matrix_norm


if __name__ == '__main__':
    save_element_attrs()
