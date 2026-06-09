"""
Phase 1: 语料预处理
- 读取27256篇文献txt
- 按句子分割
- 使用自定义分词器tokenize
- 构建词表
- 生成预训练语料
"""
import sys
import re
import json
from pathlib import Path
from tqdm import tqdm
from typing import List
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import CORPUS_DIR, CORPUS_PROCESSED_DIR, TOKENIZER_DIR
from phase1_data.tokenizer import SuperalloyTokenizer


def read_corpus_files() -> List[str]:
    """读取所有文献txt文件"""
    texts = []
    txt_files = list(CORPUS_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} text files in corpus")
    
    for fpath in tqdm(txt_files, desc="Reading corpus"):
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
                if text:
                    texts.append(text)
        except Exception as e:
            continue
    
    print(f"Successfully read {len(texts)} documents")
    return texts


def split_sentences(text: str) -> List[str]:
    """将文档文本分割为句子"""
    # 按句号、问号、感叹号分割，但保留缩写中的句点
    # 简单策略：按 '. ', '? ', '! ' 分割
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 过滤过短的句子（少于5个字符）
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def build_corpus():
    """构建预训练语料"""
    # Step 1: 读取所有文献
    documents = read_corpus_files()
    
    # Step 2: 分割句子
    print("\nSplitting documents into sentences...")
    all_sentences = []
    for doc in tqdm(documents, desc="Splitting"):
        sents = split_sentences(doc)
        all_sentences.extend(sents)
    print(f"Total sentences: {len(all_sentences)}")
    
    # Step 3: 初始化分词器并分词
    tokenizer = SuperalloyTokenizer()
    print("\nTokenizing sentences...")
    all_tokenized = []
    for sent in tqdm(all_sentences, desc="Tokenizing"):
        tokens = tokenizer.tokenize(sent)
        if len(tokens) >= 3:  # 至少3个token
            all_tokenized.append(tokens)
    print(f"Valid tokenized sentences: {len(all_tokenized)}")
    
    # Step 4: 构建词表
    print("\nBuilding vocabulary...")
    vocab = tokenizer.build_vocab(all_tokenized, min_freq=3)
    tokenizer.save()
    
    # Step 5: 保存处理后的语料
    # 保存为每行一个句子的token列表（JSON Lines格式）
    corpus_file = CORPUS_PROCESSED_DIR / "corpus_tokenized.jsonl"
    print(f"\nSaving tokenized corpus to {corpus_file}...")
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for tokens in tqdm(all_tokenized, desc="Saving"):
            f.write(json.dumps(tokens, ensure_ascii=False) + '\n')
    
    # 保存句子原文（用于参考）
    sentences_file = CORPUS_PROCESSED_DIR / "sentences.txt"
    with open(sentences_file, 'w', encoding='utf-8') as f:
        for sent in all_sentences:
            f.write(sent + '\n')
    
    # 统计信息
    stats = {
        'num_documents': len(documents),
        'num_sentences': len(all_sentences),
        'num_valid_tokenized': len(all_tokenized),
        'vocab_size': len(vocab),
        'avg_tokens_per_sentence': sum(len(t) for t in all_tokenized) / len(all_tokenized),
    }
    stats_file = CORPUS_PROCESSED_DIR / "corpus_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nCorpus Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    
    return tokenizer, all_tokenized


if __name__ == '__main__':
    build_corpus()
