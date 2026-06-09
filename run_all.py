"""
一键运行脚本
按顺序执行：数据处理 -> 预训练 -> 多任务学习 -> 逆向设计
"""
import sys
import time
import argparse
from pathlib import Path

# 添加code目录到path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_phase1():
    """Phase 1: 数据标准化与知识结构化"""
    print("\n" + "="*60)
    print("PHASE 1: Data Standardization & Corpus Processing")
    print("="*60)
    
    # Step 1.2: 构建主表
    print("\n--- Building Master Table ---")
    from phase1_data.build_master_table import build_master_table
    build_master_table()
    
    # Step 1.3-1.4: 语料预处理和分词
    print("\n--- Building Corpus ---")
    from phase1_data.build_corpus import build_corpus
    build_corpus()
    
    # Step 2.1: 元素属性矩阵
    print("\n--- Saving Element Attributes ---")
    from phase2_pretrain.element_attributes import save_element_attrs
    save_element_attrs()


def run_phase2():
    """Phase 2: 词向量预训练"""
    print("\n" + "="*60)
    print("PHASE 2: Word Vector Pre-training")
    print("="*60)
    
    # Word2Vec基线
    print("\n--- Training Word2Vec ---")
    from phase2_pretrain.train_word2vec import train_word2vec
    train_word2vec()
    
    # BERT基线
    print("\n--- Training BERT Base ---")
    from phase2_pretrain.train_bert_base import train_bert_base
    train_bert_base()
    
    # PA-MLM
    print("\n--- Training PA-MLM ---")
    from phase2_pretrain.train_pa_mlm import train_pa_mlm
    train_pa_mlm()
    
    # 词向量评估
    print("\n--- Evaluating Embeddings ---")
    from phase2_pretrain.evaluate_embeddings import evaluate_all_embeddings
    evaluate_all_embeddings()


def run_phase3():
    """Phase 3: 多任务性能预测"""
    print("\n" + "="*60)
    print("PHASE 3: Multi-task Performance Prediction")
    print("="*60)
    
    # 多任务训练（使用PA词向量）
    print("\n--- Multi-task Training (E_pa) ---")
    from phase3_multitask.train_multitask import train_multitask
    train_multitask('E_pa')
    
    # 多任务训练（使用基线词向量）
    print("\n--- Multi-task Training (E_base) ---")
    train_multitask('E_base')
    
    # 单任务基线
    print("\n--- Single-task Baseline (E_pa) ---")
    from phase3_multitask.train_singletask import train_singletask
    train_singletask('E_pa')
    
    print("\n--- Single-task Baseline (E_base) ---")
    train_singletask('E_base')
    
    # 评估对比
    print("\n--- Model Evaluation ---")
    from phase3_multitask.evaluate_models import evaluate_all
    evaluate_all()


def run_phase4():
    """Phase 4: 逆向合金设计"""
    print("\n" + "="*60)
    print("PHASE 4: Inverse Alloy Design")
    print("="*60)
    
    # NSGA-II优化
    print("\n--- Running NSGA-II ---")
    from phase4_inverse.inverse_design import run_nsga2
    run_nsga2()
    
    # 可视化
    print("\n--- Visualizing Candidates ---")
    from phase4_inverse.visualize_candidates import visualize_all
    visualize_all()
    
    # SHAP分析
    print("\n--- SHAP Analysis ---")
    from phase4_inverse.shap_analysis import run_shap_analysis
    run_shap_analysis()


def main():
    parser = argparse.ArgumentParser(description='Superalloy Word Vector & Multi-task Learning System')
    parser.add_argument('--phase', type=int, nargs='+', default=[1, 2, 3, 4],
                       help='Which phases to run (1-4). Default: all')
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='Skip the pre-training phase (use existing embeddings)')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*60)
    print("Superalloy Domain Word Vector & Multi-task Learning System")
    print("="*60)
    print(f"Phases to run: {args.phase}")
    
    if 1 in args.phase:
        run_phase1()
    
    if 2 in args.phase and not args.skip_pretrain:
        run_phase2()
    
    if 3 in args.phase:
        run_phase3()
    
    if 4 in args.phase:
        run_phase4()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL DONE! Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
