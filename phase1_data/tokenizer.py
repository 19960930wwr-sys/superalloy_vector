"""
Phase 1: 高温合金领域自定义分词器
- 元素符号保持为独立token
- 工艺术语保持完整
- 基于WordPiece构建
"""
import sys
import re
from pathlib import Path
from typing import List, Set, Tuple
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import ELEMENTS, PROCESS_KEYWORDS, PROCESS_CONDITION_KEYWORDS, TOKENIZER_DIR


# 特殊token
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

# 构建元素token集合（大小写敏感）
ELEMENT_TOKENS = set(ELEMENTS)

# 构建工艺短语列表（按长度降序排，优先匹配长短语）
PROCESS_PHRASES = sorted(
    PROCESS_KEYWORDS + PROCESS_CONDITION_KEYWORDS,
    key=lambda x: len(x), reverse=True
)


class SuperalloyTokenizer:
    """高温合金领域定制分词器"""
    
    def __init__(self, vocab: dict = None, max_length: int = 256):
        self.max_length = max_length
        self.vocab = vocab or {}
        self.id2token = {v: k for k, v in self.vocab.items()}
        
        # 元素正则：匹配大写字母+可选小写字母（如Ni, Cr, Al, C, B）
        # 需要在完整上下文中识别
        self._element_pattern = re.compile(
            r'\b(' + '|'.join(sorted(ELEMENTS, key=len, reverse=True)) + r')\b'
        )
        
        # 数字模式：匹配数字（含小数点）
        self._number_pattern = re.compile(r'\b\d+\.?\d*\b')
        
        # 工艺短语模式（不区分大小写匹配，保留原文）
        self._process_patterns = [
            (re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE), phrase)
            for phrase in PROCESS_PHRASES
        ]
    
    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词:
        1. 先识别并保护工艺短语
        2. 识别元素符号
        3. 其余做基础分词
        """
        tokens = []
        
        # Step 1: 标记工艺短语位置
        protected_spans = []  # (start, end, token_text)
        
        for pattern, phrase in self._process_patterns:
            for m in pattern.finditer(text):
                # 检查是否与已有span重叠
                overlap = False
                for ps, pe, _ in protected_spans:
                    if m.start() < pe and m.end() > ps:
                        overlap = True
                        break
                if not overlap:
                    protected_spans.append((m.start(), m.end(), phrase.lower()))
        
        # Step 2: 按位置排序
        protected_spans.sort(key=lambda x: x[0])
        
        # Step 3: 逐段处理
        pos = 0
        for span_start, span_end, span_token in protected_spans:
            # 处理span之前的文本
            if pos < span_start:
                segment = text[pos:span_start]
                tokens.extend(self._tokenize_segment(segment))
            # 添加工艺短语token
            tokens.append(span_token)
            pos = span_end
        
        # 处理最后一段
        if pos < len(text):
            tokens.extend(self._tokenize_segment(text[pos:]))
        
        return tokens
    
    def _tokenize_segment(self, text: str) -> List[str]:
        """对非工艺短语的段落进行分词"""
        tokens = []
        # 按空格和标点分割，同时处理连字符分隔的元素（如Co-Al-W）
        words = re.findall(r"[A-Za-z]+(?:[''-][A-Za-z]+)*|\d+\.?\d*|[^\s\w]", text)
        
        for i, word in enumerate(words):
            # 跳过温度单位中的C（前面是°符号）
            if word == 'C' and i > 0 and tokens and tokens[-1] == '°':
                tokens.append('°c')  # 合并为温度单位
                tokens.pop(-2)  # 移除单独的°
                continue
            
            # 检查是否是元素符号（精确匹配）
            if word in ELEMENT_TOKENS:
                tokens.append(word)
            # 检查是否是纯数字
            elif re.match(r'^\d+\.?\d*$', word):
                tokens.append('[NUM]')
            # 检查是否是连字符连接的元素（如Ni-based, Co-Al-W）
            elif '-' in word:
                parts = word.split('-')
                for part in parts:
                    if part in ELEMENT_TOKENS:
                        tokens.append(part)
                    elif self._split_elements(part):
                        tokens.extend(self._split_elements(part))
                    else:
                        tokens.append(part.lower())
            # 检查是否包含元素符号（如NiCr -> Ni, Cr）
            elif re.match(r'^[A-Z][a-z]?(?:[A-Z][a-z]?)*$', word) and len(word) > 1:
                sub_tokens = self._split_elements(word)
                if sub_tokens:
                    tokens.extend(sub_tokens)
                else:
                    tokens.append(word.lower())
            else:
                # 普通单词，转小写
                if word.strip():
                    tokens.append(word.lower())
        
        return tokens
    
    def _split_elements(self, word: str) -> List[str]:
        """尝试将连续元素符号拆分（如NiCoCr -> [Ni, Co, Cr]）"""
        result = []
        i = 0
        while i < len(word):
            # 尝试匹配两字符元素
            if i + 1 < len(word):
                two_char = word[i:i+2]
                if two_char in ELEMENT_TOKENS:
                    result.append(two_char)
                    i += 2
                    continue
            # 尝试匹配单字符元素
            one_char = word[i]
            if one_char in ELEMENT_TOKENS:
                result.append(one_char)
                i += 1
            else:
                return []  # 无法完全拆分，返回空
        return result if result else []
    
    def build_vocab(self, corpus_tokens: List[List[str]], min_freq: int = 3):
        """根据语料构建词表"""
        from collections import Counter
        
        # 统计词频
        freq = Counter()
        for sent_tokens in corpus_tokens:
            freq.update(sent_tokens)
        
        # 构建词表：特殊token + 元素 + 工艺短语 + 高频词
        self.vocab = {}
        idx = 0
        
        # 特殊token
        for t in SPECIAL_TOKENS:
            self.vocab[t] = idx
            idx += 1
        
        # 数字占位符
        self.vocab['[NUM]'] = idx
        idx += 1
        
        # 元素token（始终包含）
        for elem in ELEMENTS:
            if elem not in self.vocab:
                self.vocab[elem] = idx
                idx += 1
        
        # 工艺短语token（始终包含）
        for phrase in PROCESS_PHRASES:
            token = phrase.lower()
            if token not in self.vocab:
                self.vocab[token] = idx
                idx += 1
        
        # 高频词
        for token, count in freq.most_common():
            if count >= min_freq and token not in self.vocab:
                self.vocab[token] = idx
                idx += 1
        
        self.id2token = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)}")
        return self.vocab
    
    def encode(self, tokens: List[str], add_special: bool = True) -> List[int]:
        """将token序列转为id序列"""
        ids = []
        if add_special:
            ids.append(self.vocab.get('[CLS]', 0))
        
        for t in tokens:
            ids.append(self.vocab.get(t, self.vocab.get('[UNK]', 1)))
        
        if add_special:
            ids.append(self.vocab.get('[SEP]', 0))
        
        # 截断
        if len(ids) > self.max_length:
            ids = ids[:self.max_length - 1] + [self.vocab.get('[SEP]', 0)]
        
        return ids
    
    def decode(self, ids: List[int]) -> List[str]:
        """将id序列转为token序列"""
        return [self.id2token.get(i, '[UNK]') for i in ids]
    
    def pad(self, ids: List[int]) -> List[int]:
        """填充到max_length"""
        pad_id = self.vocab.get('[PAD]', 0)
        if len(ids) < self.max_length:
            ids = ids + [pad_id] * (self.max_length - len(ids))
        return ids
    
    def is_element_token(self, token: str) -> bool:
        """判断是否为元素token"""
        return token in ELEMENT_TOKENS
    
    def is_process_token(self, token: str) -> bool:
        """判断是否为工艺token"""
        return token in [p.lower() for p in PROCESS_PHRASES]
    
    def get_element_id_set(self) -> Set[int]:
        """获取所有元素token的id集合"""
        return {self.vocab[e] for e in ELEMENTS if e in self.vocab}
    
    def get_process_id_set(self) -> Set[int]:
        """获取所有工艺token的id集合"""
        return {self.vocab[p.lower()] for p in PROCESS_PHRASES if p.lower() in self.vocab}
    
    def save(self, path: Path = None):
        """保存词表"""
        import json
        path = path or (TOKENIZER_DIR / "vocab.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Path = None):
        """加载词表"""
        import json
        path = path or (TOKENIZER_DIR / "vocab.json")
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        tokenizer = cls(vocab=vocab)
        print(f"Tokenizer loaded, vocab size: {len(vocab)}")
        return tokenizer


if __name__ == '__main__':
    # 测试分词器
    tokenizer = SuperalloyTokenizer()
    
    test_texts = [
        "The Ni-based superalloy contains 10.5 at.% Al and 5.2 at.% Ti for gamma prime strengthening.",
        "Solution treatment at 1300°C for 24h followed by aging at 850°C.",
        "Creep rupture life of the Co-Al-W alloy was measured at 900°C under 200 MPa stress.",
        "The microstructure after homogenization shows no sigma phase precipitation.",
    ]
    
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {tokens}")
        elem_tokens = [t for t in tokens if tokenizer.is_element_token(t)]
        proc_tokens = [t for t in tokens if tokenizer.is_process_token(t)]
        print(f"Elements found: {elem_tokens}")
        print(f"Process terms: {proc_tokens}")
