---
title: NLP 入门：文本预处理与词向量
date: 2026-07-03 17:00:00
categories: [自然语言处理]
tags: [NLP, 自然语言处理, 词向量, Word2Vec, 入门教程]
description: NLP 自然语言处理入门教程：分词、词干提取、TF-IDF、Word2Vec 与预训练词向量的完整讲解。
---

## 引言

> 对计算机来说，"你好"和"hello"只是两个字节序列，它不懂"你好"背后的含义。

**NLP（自然语言处理）** 要做的事就是让计算机理解人类语言。但文本数据有个麻烦——它不像图片那样天然有像素坐标结构。

本文带你走完 NLP 的标准流程：**原始文本 → 清洗 → 分词 → 向量化 → 词嵌入**，这是后续一切 NLP 任务的基础。

---

## 前置知识

- [机器学习入门：概念与分类全解](/2026/07/03/机器学习入门：概念与分类全解/)
- [Python 编程基础（一）：从零搭建你的AI开发环境](/2026/06/17/Python编程基础（一）：从零搭建你的AI开发环境/)

---

## 一、NLP 的核心挑战

### 1.1 语言 vs 计算机

| 语言 | 计算机 |
|:----:|:------:|
| "苹果很好吃" | 字节序列 |
| "我今天买了苹果手机" | 同一个词，不同意思 |
| "I am running" ~ "The running water" | 同一个词，不同词性 |
| "苹果很好吃" ~ "我吃了苹果" | "苹果"出现了位置变了 |

**三个核心问题：**

1. **分词** — 一句话拆成什么单位？
2. **歧义** — 同一个词在不同上下文什么意思？
3. **表示** — 怎么把词变成计算机能算的数字向量？

### 1.2 NLP 的典型任务

```
分词/词性标注 ─── 基础任务（管道上游）
     ↓
命名实体识别 (NER) ─── "2026年7月3日"是日期，"北京"是地点
     ↓
文本分类/情感分析 ─── 正面/负面/中性
     ↓
机器翻译/摘要/问答 ─── 复杂任务
```

---

## 二、文本预处理

### 2.1 分词（Tokenization）

分词是把文本拆成最小语义单位。

```python
# English — 简单，直接用空格+标点
text = "I don't like this movie at all."
tokens = ["I", "don't", "like", "this", "movie", "at", "all", "."]

# BPE (Byte Pair Encoding) — GPT/BERT 用的分词法
# "unbelievable" → ["un", "believ", "able"]
# 优点是能处理任何没见过的词
```

```python
# 安装
# pip install nltk spacy jieba

# NLTK
from nltk.tokenize import word_tokenize
tokens = word_tokenize("I don't like this movie at all.")
print(tokens)
# ['I', 'do', "n't", 'like', 'this', 'movie', 'at', 'all', '.']

# 中文分词 — jieba
import jieba
text = "我在北京清华大学读书，未来想要研究人工智能。"
tokens = list(jieba.cut(text))
print('/'.join(tokens))
# 我/在/北京/清华大学/读书/，/未来/想要/研究/人工/智能/。
```

### 2.2 文本清洗

```python
import re

def clean_text(text):
    """基础的文本清洗"""
    # 转小写
    text = text.lower()

    # 去 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 去 URL
    text = re.sub(r'http\S+|www\S+', '', text)

    # 去 @用户和 #话题
    text = re.sub(r'@\w+|#\w+', '', text)

    # 去特殊字符但保留标点
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)

    # 合并多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text
```

### 2.3 词干提取与词形还原

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "runner", "ran", "better"]

for w in words:
    print(f"{w:10s} → 词干: {stemmer.stem(w):10s} | 词形还原: {lemmatizer.lemmatize(w, pos='v'):10s}")

# 输出:
# running    → 词干: run       | 词形还原: run
# runner     → 词干: runner    | 词形还原: runner
# ran        → 词干: ran       | 词形还原: run
# better     → 词干: better    | 词形还原: better (词性修正后: good)
```

**区别：** 词干提取靠规则砍（"running" → "run"），词形还原则查词典（"ran" → "run"）。词形还原本质上更准确，但需要词性标注辅助。

### 2.4 停用词移除

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

tokens = ["this", "movie", "is", "really", "good"]
filtered = [t for t in tokens if t not in stop_words]
print(filtered)
# ['movie', 'really', 'good']
```

> ❗ 并非所有任务都适合去停用词。情感分析中 "not"、"but" 这类词很重要。

---

## 三、文本向量化

### 3.1 One-Hot 编码

```python
# 词表: ["爱", "很", "开心", "恨", "难过", "无聊"]
# 每个词是一个 6 维向量，只有一位是 1

"爱"    → [1, 0, 0, 0, 0, 0]
"开心"  → [0, 0, 1, 0, 0, 0]
"无聊"  → [0, 0, 0, 0, 0, 1]
```

**缺点：** 词表越大向量越长，且任意两个词的相似度都是 0。

### 3.2 TF-IDF

TF-IDF 评估一个词对一篇文档的重要程度：

```
TF(t, d) = 词 t 在文档 d 中出现的次数 / 文档 d 的总词数
IDF(t) = log(总文档数 / 包含词 t 的文档数)
TF-IDF = TF × IDF
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "I love deep learning and AI",
    "I love pizza and pasta",
    "Deep learning is fascinating",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

# 查看每个词的 TF-IDF 权重
feature_names = vectorizer.get_feature_names_out()
print(f"词表: {list(feature_names)}")
print(f"TF-IDF 矩阵形状: {tfidf_matrix.shape}")  # (3, 9)
```

TF-IDF 的优势是**能自动降低 "the"、"and" 等常见词的权重**，突出有区分度的词。

### 3.3 词向量（Word Embedding）

One-Hot 和 TF-IDF 的共同问题：**不包含语义信息。**

词向量把每个词映射到低维连续向量（通常 50-300 维），语义相似的词向量距离更近：

```
"国王" - "男人" + "女人" ≈ "女王"
```

```python
# Gensim Word2Vec
from gensim.models import Word2Vec

sentences = [
    ["i", "love", "deep", "learning"],
    ["deep", "learning", "is", "awesome"],
    ["i", "love", "nlp", "and", "computer", "vision"],
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
# vector_size: 词向量维度
# window:     上下文窗口大小
# min_count:  最小出现次数
# sg:         CBOW(0) / Skip-gram(1)

# 找相似词
similar = model.wv.most_similar('deep', topn=3)
print(f"与 'deep' 最相似的词: {similar}")
```

### 3.4 预训练词向量

自己训练词向量的时代已经过去了。现在直接用别人在大数据上训好的：

| 模型 | 维度 | 训练数据 | 特点 |
|:----:|:----:|:--------:|:----:|
| GloVe | 50/100/300 | 60 亿词 | 经典，基于全局共现 |
| FastText | 300 | 160 亿词 | 支持子词，能处理 OOV |
| **BERT** | 768 | 33 亿词 | **上下文相关**——里程碑式突破 |
| **GPT 系列** | 768-12288 | 海量 | 超大语言模型 |

**重点：BERT 之前，词向量是静态的**（"苹果"在"苹果很好吃"和"苹果手机"中向量一样）。BERT 之后，词向量变成**上下文相关**——同一个词在不同句子中向量不同。

```python
from transformers import AutoTokenizer, AutoModel
import torch

# 用 BERT 生成上下文相关的词向量
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

texts = [
    "I bought an apple.",
    "I bought an apple phone.",
]

for text in texts:
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)

    # 取 token 的向量（最后一层隐藏状态）
    embeddings = outputs.last_hidden_state  # [1, seq_len, 768]

    # "apple" 在这两个句子中的向量不同！
    print(f"'{text}' → 输出形状: {embeddings.shape}")
```

---

## 四、预处理流水线实战

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def nlp_pipeline(text):
    """完整的 NLP 预处理流水线"""
    # 1. 清洗
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.lower().strip()

    # 2. 分词
    tokens = nltk.word_tokenize(text)

    # 3. 去停用词和标点
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens
              if t not in stop_words and t.isalpha()]

    # 4. 词形还原
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens

# 测试
text = "The amazing movies were not really appreciated by the critics!!!"
print(nlp_pipeline(text))
# ['amazing', 'movie', 'really', 'appreciated', 'critic']
```

---

## 五、NLP 技术演进

```
2013  Word2Vec        ─── 词向量时代开始
2014  Seq2Seq + Attention ─── 机器翻译突破
2017  Transformer     ─── Attention Is All You Need
2018  BERT            ─── 双向上下文
          │
          ├── GPT (2018) ─── 单向自回归
          ├── RoBERTa (2019) ─── BERT 优化版
          └── ELECTRA (2020) ─── 更高效的预训练
          │
2019  GPT-2           ─── 后起之秀
2020  GPT-3           ─── 大模型爆发
2021-2026  GPT-3.5/4, Claude, Gemini, LLaMA, DeepSeek
```

现在 NLP 的主流范式是：**预训练 → 微调**。没有人再从零训词向量了。

---

## 六、总结

| 知识点 | 掌握 |
|--------|:----:|
| NLP 的核心挑战 | ✅ |
| 分词（中英文） | ✅ |
| 文本清洗与归一化 | ✅ |
| 词干提取 vs 词形还原 | ✅ |
| TF-IDF 原理 | ✅ |
| Word2Vec 词向量 | ✅ |
| 预训练词向量（BERT） | ✅ **核心** |
| 完整预处理流水线 | ✅ **实战** |

**下一步推荐：**
- [神经网络架构详解：RNN 与 LSTM](/2026/07/03/神经网络架构详解：RNN与LSTM/)
- [PyTorch 实战（二）：构建第一个全连接网络](/2026/07/03/PyTorch实战（二）：构建第一个全连接网络/)
- [实战：情感分析（IMDB）](/2026/07/03/实战：情感分析IMDB/)（即将发布）
