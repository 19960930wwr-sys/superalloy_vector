"""
Phase 2: Word2Vec基线训练
使用gensim训练Skip-gram Word2Vec
"""
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (CORPUS_PROCESSED_DIR, EMBEDDINGS_DIR, W2V_CONFIG, 
                    ELEMENTS, SEED)


def load_tokenized_corpus() -> list:
    """加载分词后的语料"""
    corpus_file = CORPUS_PROCESSED_DIR / "corpus_tokenized.jsonl"
    sentences = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            tokens = json.loads(line.strip())
            sentences.append(tokens)
    print(f"Loaded {len(sentences)} sentences")
    return sentences


def train_word2vec():
    """训练Word2Vec模型"""
    from gensim.models import Word2Vec
    
    print("Loading tokenized corpus...")
    sentences = load_tokenized_corpus()
    
    print(f"\nTraining Word2Vec (Skip-gram)...")
    print(f"Config: {W2V_CONFIG}")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=W2V_CONFIG['vector_size'],
        window=W2V_CONFIG['window'],
        min_count=W2V_CONFIG['min_count'],
        sg=W2V_CONFIG['sg'],
        workers=W2V_CONFIG['workers'],
        epochs=W2V_CONFIG['epochs'],
        seed=SEED,
    )
    
    # 保存模型
    model_path = EMBEDDINGS_DIR / "word2vec.model"
    model.save(str(model_path))
    print(f"\nWord2Vec model saved to {model_path}")
    
    # 提取并保存词向量
    vocab = model.wv.key_to_index
    vectors = model.wv.vectors
    
    # 保存为numpy
    np.save(EMBEDDINGS_DIR / "E_w2v_vectors.npy", vectors)
    
    # 保存词表映射
    with open(EMBEDDINGS_DIR / "E_w2v_vocab.json", 'w') as f:
        json.dump(vocab, f)
    
    # 打印元素词向量信息
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Vector dimension: {vectors.shape[1]}")
    print(f"\nElement vectors found:")
    for elem in ELEMENTS:
        if elem in model.wv:
            # 找最相似的其他元素
            similar = model.wv.most_similar(elem, topn=3)
            print(f"  {elem}: most similar = {similar[:3]}")
        else:
            print(f"  {elem}: NOT in vocabulary")
    
    return model


if __name__ == '__main__':
    train_word2vec()
