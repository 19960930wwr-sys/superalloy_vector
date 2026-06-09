"""
Phase 3: 基于经典机器学习算法的单任务基线训练
与神经网络单任务、神经网络多任务一起作三方对比
算法参考 model_selection.py 的配置（GridSearchCV + 5折CV）
"""
import sys
import json
import warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, LogisticRegression

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import MULTITASK_CONFIG, TASKS, RESULTS_DIR, SEED
from phase3_multitask.dataset_alloy import load_alloy_dataset

warnings.filterwarnings('ignore')

# 可选 XGBoost（依赖未安装时自动跳过）
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost not installed, XGB algorithm will be skipped.")


# =============== 算法 + 超参网格 ===============
def get_regressors():
    """回归算法列表: (name, model, param_grid)"""
    regs = [
        ("Ridge",
            Ridge(),
            {"alpha": [0.01, 0.1, 1, 10, 100]}),
        ("KNR",
            KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 7, 10]}),
        ("SVR",
            SVR(kernel='rbf', max_iter=200000),
            {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.01, 0.1]}),
        ("RFR",
            RandomForestRegressor(random_state=SEED, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}),
        ("GBR",
            GradientBoostingRegressor(random_state=SEED),
            {"n_estimators": [100, 200], "max_depth": [3, 5, 7],
             "learning_rate": [0.05, 0.1]}),
    ]
    if HAS_XGB:
        regs.append((
            "XGB",
            XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0,
                         tree_method='hist'),
            {"n_estimators": [100, 200], "max_depth": [3, 5, 7],
             "learning_rate": [0.05, 0.1]},
        ))
    return regs


def get_classifiers():
    """分类算法列表"""
    cls = [
        ("LogReg",
            LogisticRegression(max_iter=10000, n_jobs=-1),
            {"C": [0.01, 0.1, 1, 10]}),
        ("KNN",
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 10]}),
        ("SVC",
            SVC(probability=True, max_iter=200000),
            {"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1]}),
        ("RFC",
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}),
        ("GBC",
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": [100, 200], "max_depth": [3, 5]}),
    ]
    if HAS_XGB:
        cls.append((
            "XGB",
            XGBClassifier(random_state=SEED, n_jobs=-1, verbosity=0,
                          eval_metric='logloss', tree_method='hist'),
            {"n_estimators": [100, 200], "max_depth": [3, 5]},
        ))
    return cls


def extract_features_targets(dataset):
    """从 AlloyDataset 抽取所有样本的特征向量与原始目标值（绕开归一化）"""
    X_list, y_raw_list = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        X_list.append(item['input'].numpy())
        y_raw_list.append(item['raw_target'])
    return np.array(X_list, dtype=np.float32), np.array(y_raw_list, dtype=np.float32)


def train_singletask_ml(embedding_name: str = 'E_pa'):
    """对每个任务用多种 ML 算法做单任务基线（5折CV + GridSearch）"""
    np.random.seed(SEED)
    print(f"\n{'=' * 60}")
    print(f"Single-Task ML Baselines | Embedding: {embedding_name}")
    print(f"{'=' * 60}")

    all_results = {}  # {algo_name: {task_name: metric_summary}}

    for task_name, task_info in TASKS.items():
        print(f"\n--- Task: {task_name} ({task_info['type']}) ---")

        # 加载该任务子集
        dataset = load_alloy_dataset(embedding_name, task_filter=task_name)
        if len(dataset) < 10:
            print(f"  Skipping {task_name}: too few samples ({len(dataset)})")
            continue

        X, y = extract_features_targets(dataset)
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

        # 选择算法集
        if task_info['type'] == 'regression':
            algos = get_regressors()
            scoring = 'r2'
        else:
            algos = get_classifiers()
            scoring = 'f1'

        # K折
        n_splits = min(MULTITASK_CONFIG['n_splits'], max(2, len(X) // 5))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        for algo_name, model, param_grid in algos:
            fold_metrics = []
            for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                # 标准化特征
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)

                # GridSearchCV (内部3折，节省时间)
                try:
                    gs = GridSearchCV(model, param_grid, scoring=scoring,
                                      n_jobs=-1, cv=3)
                    gs.fit(X_tr_s, y_tr)
                    pred = gs.predict(X_va_s)
                except Exception as e:
                    print(f"    [{algo_name}] Fold {fold}: skipped ({e})")
                    continue

                if task_info['type'] == 'regression':
                    rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
                    r2 = float(r2_score(y_va, pred)) if len(y_va) > 1 else 0.0
                    fold_metrics.append({'rmse': rmse, 'r2': r2})
                else:
                    y_va_int = y_va.astype(int)
                    pred_int = (pred > 0.5).astype(int) if pred.dtype.kind == 'f' \
                        else pred.astype(int)
                    f1 = float(f1_score(y_va_int, pred_int, zero_division=0))
                    try:
                        if hasattr(gs, 'predict_proba'):
                            proba = gs.predict_proba(X_va_s)[:, 1]
                            auc = float(roc_auc_score(y_va_int, proba))
                        else:
                            auc = 0.0
                    except Exception:
                        auc = 0.0
                    fold_metrics.append({'f1': f1, 'auc': auc})

            if not fold_metrics:
                continue

            # 汇总该算法该任务的5折结果
            if task_info['type'] == 'regression':
                rmses = [m['rmse'] for m in fold_metrics]
                r2s = [m['r2'] for m in fold_metrics]
                summary = {
                    'rmse_mean': float(np.mean(rmses)),
                    'rmse_std': float(np.std(rmses)),
                    'r2_mean': float(np.mean(r2s)),
                    'r2_std': float(np.std(r2s)),
                }
            else:
                f1s = [m['f1'] for m in fold_metrics]
                aucs = [m['auc'] for m in fold_metrics]
                summary = {
                    'f1_mean': float(np.mean(f1s)),
                    'f1_std': float(np.std(f1s)),
                    'auc_mean': float(np.mean(aucs)),
                    'auc_std': float(np.std(aucs)),
                }

            all_results.setdefault(algo_name, {})[task_name] = summary
            print(f"  [{algo_name}] {summary}")

    # 每个算法独立保存一份 JSON，方便 evaluate_models 加载
    for algo_name, task_results in all_results.items():
        out_path = RESULTS_DIR / f"ml-{algo_name}_{embedding_name}_results.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(task_results, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print(f"All ML baselines done for {embedding_name}")
    print(f"{'=' * 60}")
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, default='E_pa',
                        choices=['E_pa', 'E_base', 'E_w2v', 'E_attr', 'E_proc'])
    args = parser.parse_args()
    train_singletask_ml(args.embedding)
