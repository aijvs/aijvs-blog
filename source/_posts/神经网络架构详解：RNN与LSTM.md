---
title: 神经网络架构详解：RNN 与 LSTM
date: 2026-07-03 15:00:00
categories:
  - 深度学习
tags:
  - RNN
  - LSTM
  - 序列模型
  - 进阶教程
  - 自然语言处理
  - PyTorch
cover: /img/posts/rnn-lstm-cover.png
description: RNN 循环神经网络与 LSTM 长短期记忆网络的完整教程：原理、梯度消失、门控机制与 PyTorch 情感分析实战。
---

## 引言

CNN 处理的是图片——有空间结构。那**文本、语音、股价、视频**呢？这些是序列数据，长度不固定，前后有依赖关系。

> "I **love** this movie" 和 "I don't **love** this movie"——同一个词 "love"，因为前面有个 "don't"，意思完全反了。

**RNN（循环神经网络）** 就是为序列数据设计的——它能记住前面看到的信息来影响当前的判断。

但 RNN 有个致命缺陷：**长期依赖问题**——看了 100 个词之后，第 1 个词的信息基本被遗忘了。这就是 **LSTM（长短期记忆网络）** 要解决的问题。

---

## 前置知识

- [深度学习入门指南：从零开始理解神经网络](/2026/05/25/深度学习入门指南：从零开始理解神经网络/)
- [PyTorch 实战（二）：构建第一个全连接网络](/2026/07/03/PyTorch实战（二）：构建第一个全连接网络/)

理解神经网络的基本概念就够了。

---

## 一、为什么需要 RNN？

### 传统网络的问题

```
传统神经网络（全连接 / CNN）:
输入 → [网络] → 输出

每个输入独立处理，没有"记忆"能力。
"I am from China, I speak ______"
→ 需要记住前面说了 "China" 才能预测 "Chinese"
→ 传统网络做不到
```

### RNN 的核心思想

**循环**：网络在处理每个输入时，不仅看当前输入，还看上一步的隐藏状态。

```
RNN:
               输出1      输出2      输出3
                ↑         ↑         ↑
     ┌───┐     ┌───┐     ┌───┐     ┌───┐
输入 →│ A │────▶│ A │────▶│ A │────▶│ A │
     └───┘     └─┬─┘     └─┬─┘     └─┬─┘
       ↑         │         │         │
       └─────────┴─────────┴─────────┘
       隐藏状态在时间步之间传递
```

---

## 二、RNN 原理

### 2.1 数学表达

```
h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b)
y_t = W_hy · h_t + b_y

其中：
h_t    = 时间步 t 的隐藏状态
x_t    = 时间步 t 的输入
h_{t-1}= 上一步的隐藏状态（记忆）
tanh   = 激活函数（压缩到 [-1, 1]）
```

### 2.2 PyTorch 实现

```python
import torch
import torch.nn as nn

# PyTorch 内置 RNN 层
rnn = nn.RNN(
    input_size=100,    # 每个时间步的输入维度（如词向量维度）
    hidden_size=128,   # 隐藏状态维度
    num_layers=2,      # 层数
    batch_first=True,  # 输入形状: (batch, seq_len, input_size)
)

# 输入: [batch=16, seq_len=10, input_size=100]
x = torch.randn(16, 10, 100)
output, h_n = rnn(x)

print(f"输出形状: {output.shape}")  # [16, 10, 128] — 每个时间步的隐藏状态
print(f"最终隐藏状态: {h_n.shape}")  # [2, 16, 128] — 最后一层的最终状态
```

### 2.3 手写一个简易 RNN

```python
class SimpleRNN(nn.Module):
    """手动实现的单层 RNN"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.shape

        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        outputs = []
        for t in range(seq_len):
            combined = torch.cat((x[:, t, :], hidden.squeeze(0)), dim=1)
            hidden = torch.tanh(self.i2h(combined)).unsqueeze(0)
            outputs.append(hidden)

        return torch.stack(outputs, dim=1).squeeze(0), hidden
```

---

## 三、RNN 的致命问题：梯度消失

### 3.1 问题根源

RNN 在每个时间步都要乘以权重 `W_hh`。经过 T 个时间步：

```
∂L/∂W ∝ W_hh^T
```

- 如果 `W_hh` 的特征值 < 1：**梯度消失** — 远程信息完全被遗忘
- 如果 `W_hh` 的特征值 > 1：**梯度爆炸** — 训练不稳定

**结果：** 标准 RNN 实际上只能记住大约 7-10 步之前的信息。

### 3.2 梯度消失的直观感受

> 读一段 100 个字的文章，读到第 100 个字时，第 1 个字的信息已经被稀释了 2^100 倍——远小于噪声。

---

## 四、LSTM：长短期记忆网络

LSTM（Long Short-Term Memory）通过**门控机制**解决了梯度消失问题。它引入了三个门和一个细胞状态：

### 4.1 LSTM 内部结构

```
                     ┌───────────────────────────┐
                     │        LSTM 单元           │
                     │                           │
  h_{t-1} ──────────┬─▶┌─────┐    ┌─────┐      │
                     │  │遗忘门│    │输入门│      │
  x_t ───────────────┼─▶└─────┘    └─────┘      │
                     │     │          │          │
                     │     ▼          ▼          │
                     │  ┌─────────────────┐      │
                     │  │     更新细胞状态  │      │
                     │  └─────────────────┘      │
                     │         │                 │
                     │         ▼                 │
                     │     ┌─────┐              │
                     │     │输出门│              │
                     │     └─────┘              │
                     │         │                 │
                     │    h_t, c_t               │
                     └───────────────────────────┘
```

**四个组件的作用：**

| 组件 | 公式 | 作用 |
|------|------|------|
| **遗忘门** | f_t = σ(W_f · [h_{t-1}, x_t] + b_f) | 决定丢弃哪些旧的记忆 |
| **输入门** | i_t = σ(W_i · [h_{t-1}, x_t] + b_i) | 决定存入哪些新信息 |
| **候选记忆** | Ĉ_t = tanh(W_c · [h_{t-1}, x_t] + b_c) | 生成新的候选记忆 |
| **输出门** | o_t = σ(W_o · [h_{t-1}, x_t] + b_o) | 决定输出哪些记忆 |

状态更新：

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ Ĉ_t    # 遗忘旧记忆 + 添加新记忆
h_t = o_t ⊙ tanh(C_t)                # 基于细胞状态输出
```

**关键创新**：细胞状态 C_t 的更新是**加法**而非乘法，梯度沿 C_t 传播时不会指数衰减——这就是 LSTM 能记住长程依赖的原因。

### 4.2 为什么 LSTM 比 RNN 好？

| RNN | LSTM |
|:---:|:----:|
| 一个 tanh 层 | 三个 σ 层 + 一个 tanh 层 |
| 梯度沿时间步乘法传播 → 指数衰减 | 梯度沿细胞状态加法传播 → 稳定 |
| 有效记忆 ~10 步 | 有效记忆 ~100+ 步 |
| 几乎被淘汰 | 仍是工业标准之一 |

> 2015 年后，LSTM 在大部分任务上又被 **Transformer** 超越（尤其 NLP）。但在时间序列预测、语音等任务上，LSTM 依然能打。

---

## 五、实战：情感分析

我们用 LSTM 对 IMDB 影评做二分类（正面/负面）。

### 5.1 数据准备

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 文本预处理
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for label, line in data_iter:
        yield tokenizer(line)

# 加载训练数据
train_iter = IMDB(split='train')

# 构建词表（只保留最常见的 25000 个词）
vocab = build_vocab_from_iterator(
    yield_tokens(train_iter),
    specials=['<unk>', '<pad>', '<bos>'],
    max_tokens=25000
)
vocab.set_default_index(vocab['<unk>'])

print(f"词表大小: {len(vocab)}")

# 文本 → 索引序列
def encode_text(text, max_len=200):
    tokens = tokenizer(text)[:max_len]
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

# 准备数据集
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, split, max_len=200):
        self.data = list(IMDB(split=split))
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        label = 1 if label == 'pos' else 0
        tokens = encode_text(text, self.max_len)
        return tokens, label

def collate_batch(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    padded = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
    return padded, torch.tensor(labels, dtype=torch.long), lengths

train_dataset = IMDBDataset('train')
test_dataset = IMDBDataset('test')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

print(f"训练集: {len(train_dataset)} 条")
print(f"测试集: {len(test_dataset)} 条")
```

### 5.2 定义 LSTM 模型

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<pad>'])
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths=None):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # 使用 pack_padded_sequence 避免填充部分被处理
        if lengths is not None:
            lengths = lengths.cpu()
            packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, cell) = self.lstm(packed)
        else:
            _, (hidden, cell) = self.lstm(embedded)

        # 取最后一层的最后一个时间步的隐藏状态
        last_hidden = hidden[-1]  # [batch, hidden_dim]

        out = self.dropout(last_hidden)
        out = self.fc(out)  # [batch, num_classes]
        return out
```

### 5.3 训练

```python
model = LSTMClassifier(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for texts, labels, lengths in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(texts, lengths)
        loss = criterion(output, labels)
        loss.backward()

        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        _, preds = output.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_acc = 100. * correct / total

    # 测试
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for texts, labels, lengths in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            _, preds = model(texts, lengths).max(1)
            test_total += labels.size(0)
            test_correct += preds.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | '
          f'Train: {train_acc:.2f}% | Test: {test_acc:.2f}%')

print("训练完成 ✅")
```

LSTM 情感分析模型通常 **5 个 epoch 达到 85-88% 的测试准确率**。

### 5.4 推理示例

```python
def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        tokens = encode_text(text).unsqueeze(0).to(device)
        output = model(tokens)
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = prob[0][pred].item()
        sentiment = "正面" if pred == 1 else "负面"
        print(f"文本: {text[:50]}...")
        print(f"情感: {sentiment} (置信度: {confidence:.2%})")
    return pred

predict_sentiment("This movie was absolutely amazing! The acting and plot were top-notch.")
predict_sentiment("Terrible waste of time. I regret watching this movie.")
```

输出示例：
```
文本: This movie was absolutely amazing! The acting...
情感: 正面 (置信度: 99.12%)
文本: Terrible waste of time. I regret watching this...
情感: 负面 (置信度: 98.45%)
```

---

## 六、LSTM vs GRU vs Transformer

| 模型 | 参数量 | 长期记忆 | 并行化 | 典型场景 |
|:----:|:------:|:--------:|:------:|:--------:|
| RNN | 少 | ❌ | ❌ | 已被淘汰 |
| LSTM | 中 | ✅ | ❌ | 时间序列、语音 |
| GRU | 少（比 LSTM 少 1/4） | ✅ | ❌ | 可替代 LSTM，效果相当 |
| Transformer | 多 | ✅✅ | ✅✅ | NLP 主流（BERT/GPT） |

> **GRU（门控循环单元）** 是 LSTM 的简化版——把遗忘门和输入门合并为"更新门"，去掉了细胞状态。效果和 LSTM 差不多，参数更少，训练更快。

---

## 七、总结

| 知识点 | 掌握 |
|--------|:----:|
| RNN 为什么需要"记忆" | ✅ |
| RNN 的循环机制和梯度消失问题 | ✅ |
| LSTM 的遗忘门/输入门/输出门 | ✅ **核心** |
| LSTM 如何解决长期依赖（细胞状态加法更新） | ✅ |
| PyTorch LSTM 情感分析实战 | ✅ **实战** |
| GRU / Transformer 对比 | ✅ |

**下一步推荐：**
- [NLP 入门：文本预处理与词向量](/2026/07/03/NLP入门：文本预处理与词向量/)（即将发布）
- [PyTorch 实战（一）：张量操作与自动微分](/2026/07/03/PyTorch实战（一）：张量操作与自动微分/)
- [神经网络架构详解：CNN 卷积神经网络](/2026/07/03/神经网络架构详解：CNN卷积神经网络/)
