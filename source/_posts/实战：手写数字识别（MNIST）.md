---
title: 实战：手写数字识别（MNIST）
date: 2026-07-11 10:00:00
categories: [实战项目]
tags: [PyTorch, MNIST, 手写数字识别, 入门教程, 卷积神经网络, 实战项目]
description: 从零实现 MNIST 手写数字识别，对比全连接网络和 CNN 的性能差异，完整代码可运行。
---

## 引子

你学了 PyTorch，懂了张量怎么算、梯度怎么传——但脑子里还是空的。

**"学完了理论，我不知道怎么组合起来。"**

这是我从后台看到最多的私信。所以这篇来了：一次完整的实战，从数据到模型到调优。

你能写出一个识别手写数字的程序，就说明你真正理解了深度学习的基础管线。

---

## 前置知识

- [PyTorch 实战（二）：构建第一个全连接网络](/2026/07/03/PyTorch实战（二）：构建第一个全连接网络/)
- [神经网络架构详解：CNN 卷积神经网络](/2026/07/03/神经网络架构详解：CNN卷积神经网络/)

看完这两篇再来，手感会好很多。

---

## 一、MNIST 数据集长什么样？

MNIST 是计算机视觉界的"Hello World"。

- 训练集：60,000 张 28×28 灰度手写数字（0-9）
- 测试集：10,000 张
- 每个像素 0-255，白色背景、黑色笔迹

```python
# 看一眼数据长什么样子
import torch
import torchvision
import matplotlib.pyplot as plt

train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
print(f"形状: {train_data.data.shape}")   # torch.Size([60000, 28, 28])
print(f"标签: {train_data.targets[:10]}") # tensor([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])
```

如果这行代码跑不了，说明你缺 `torchvision`：
```bash
pip install torchvision matplotlib
```

---

## 二、方案一：全连接网络（基线）

### 2.1 数据加载

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 预处理：转 Tensor 并归一化到 [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False)
```

> **为什么用 Normalize？** 原始像素值 0-255 方差很大，模型训练不稳定。归一化后均值为 0、方差为 1，梯度更新更平滑。

### 2.2 定义模型

```python
class FCModel(nn.Module):
    """3 层全连接网络"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [batch, 1, 28, 28] → [batch, 784]
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 交叉熵损失自带 softmax，这里不用加
```

### 2.3 训练函数

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
```

### 2.4 训练

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fc_model = FCModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.001)

epochs = 5
for epoch in range(1, epochs + 1):
    loss = train_one_epoch(fc_model, train_loader, optimizer, criterion, device)
    acc  = evaluate(fc_model, test_loader, device)
    print(f"Epoch {epoch}: loss={loss:.4f}, test_acc={acc:.4f}")
```

预期输出：
```
Epoch 1: loss=0.3372, test_acc=0.9372
Epoch 2: loss=0.1623, test_acc=0.9584
Epoch 3: loss=0.1134, test_acc=0.9650
Epoch 4: loss=0.0855, test_acc=0.9694
Epoch 5: loss=0.0673, test_acc=0.9739
```

5 轮训练，测试准确率 **97.4%**。已经不错了——但 CNN 能做得更好。

---

## 三、方案二：CNN（改进版）

全连接网络的缺点是**丢掉了空间结构**——28×28 的图片展平成 784 个独立像素。CNN 保留了 2D 结构。

```python
class CNNModel(nn.Module):
    """2 层卷积 + 2 层全连接"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28×28 → 28×28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14×14 → 14×14
        self.pool = nn.MaxPool2d(2, 2)     # 2×2 池化，宽高减半
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [b,1,28,28] → [b,32,14,14]
        x = self.pool(torch.relu(self.conv2(x)))  # [b,32,14,14] → [b,64,7,7]
        x = x.view(x.size(0), -1)                 # [b, 64*7*7]
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

一样的训练代码，唯一的区别是换模型：

```python
cnn_model = CNNModel().to(device)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

for epoch in range(1, epochs + 1):
    loss = train_one_epoch(cnn_model, train_loader, optimizer, criterion, device)
    acc  = evaluate(cnn_model, test_loader, device)
    print(f"Epoch {epoch}: loss={loss:.4f}, test_acc={acc:.4f}")
```

预期输出：
```
Epoch 1: loss=0.2022, test_acc=0.9684
Epoch 2: loss=0.0631, test_acc=0.9832
Epoch 3: loss=0.0422, test_acc=0.9867
Epoch 4: loss=0.0296, test_acc=0.9902
Epoch 5: loss=0.0227, test_acc=0.9914
```

**CNN 在 Epoch 1 就追上了全连接网络 5 轮的准确率，5 轮后 99.1%。**

差出来的 1.7% 就是"保住空间结构"带来的差距。

---

## 四、对比与解读

| 指标 | 全连接网络 | CNN |
|------|-----------|-----|
| 参数量 | 109,386 | 387,786 |
| 5 轮准确率 | 97.4% | 99.1% |
| 每轮训练时间（CPU） | ~15s | ~30s |
| 每轮训练时间（GPU） | ~3s | ~5s |

CNN 参数量虽然大，但在 GPU 上差距不大。**对于图像类任务，CNN 是绝对首选。**

### 为什么不是 100%？

99.1% 离 100% 差的那 0.9%，看这里——这些连人都认不出来：

![MNIST 难例](无法识别的样本，比如连笔、缺损、歪斜的 4/9 混淆)

**99.1% 说明不是模型问题，是数据本身的噪声。** 真实场景里，99% 精度已经足够部署。

---

## 五、完整代码（一键运行）

```python
"""mnist_demo.py - 手写数字识别完整代码"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=1000, shuffle=False)

# 2. 模型（CNN）
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 3. 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 6):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 测试
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, acc={correct/total:.4f}")
```

保存为 `mnist_demo.py`，直接 `python mnist_demo.py` 就能跑。

---

## 总结

你从这篇学到了三件事：

- **全连接网络能做 MNIST，但 CNN 更好**（97.4% → 99.1%）
- **深度学习的关键管线**：DataLoader → Model → Loss → Optimizer → Loop
- **归一化和卷积核**这些看起来很小的细节，对结果影响巨大

下一步可以尝试：
- 增加 Dropout 和数据增强，看看能不能冲 99.5%
- 换 ResNet-18，看看超大网络有没有过拟合
- 用这篇的代码，试着识别你自己的手写数字

下一篇我们来挑战 CIFAR-10——真正的彩色图片分类。
