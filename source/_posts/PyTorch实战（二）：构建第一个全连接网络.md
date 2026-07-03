---
title: PyTorch 实战（二）：构建第一个全连接网络
date: 2026-07-03 12:00:00
categories: [深度学习]
tags: [PyTorch, 深度学习, 神经网络, MNIST, 实战教程]
description: 从零实现 PyTorch 全连接网络：Dataset、DataLoader、nn.Module、训练循环、评估，全部代码可运行。
---

## 引言

上篇文章我们学了张量操作和自动微分——相当于拿到了砖头和水泥。这篇我们来盖第一栋房子：**用 PyTorch 的 nn.Module 构建全连接神经网络，完成手写数字识别**。

写完这篇文章的代码，你就走通了深度学习的完整流程：**数据加载 → 模型定义 → 训练 → 评估**。

---

## 前置知识

- [PyTorch 实战（一）：张量操作与自动微分](/2026/07/03/PyTorch实战（一）：张量操作与自动微分/)
- [机器学习入门：概念与分类全解](/2026/07/03/机器学习入门：概念与分类全解/)

尤其是 PyTorch（一）中的 `requires_grad`、`backward()`、梯度更新——这些是本文的基础。

---

## 一、项目结构总览

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  DataLoader   │────▶│   Model     │────▶│   Trainer    │
│  数据加载器    │     │  模型定义    │     │  训练循环    │
├──────────────┤     ├──────────────┤     ├──────────────┤
│ - MNIST 数据  │     │ - 网络层定义  │     │ - 前向传播   │
│ - 批量加载    │     │ - 前向传播   │     │ - 计算损失   │
│ - 数据增强   │     │ - 参数管理   │     │ - 反向传播   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                    │
                                              ┌─────▼──────┐
                                              │   Eval     │
                                              │  模型评估   │
                                              └────────────┘
```

---

## 二、数据准备

我们使用 **MNIST** 数据集——28×28 像素的手写数字图片（0-9）。

### 2.1 加载与预处理

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 超参数
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# 数据预处理：转 Tensor + 归一化
transform = transforms.Compose([
    transforms.ToTensor(),                              # PIL → Tensor，自动将 [0,255] 归一化到 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))          # 用 MNIST 的均值和标准差标准化
])

# 下载并加载训练集和测试集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建 DataLoader（自动批量、打乱、多线程加载）
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集: {len(train_dataset)} 张图片")
print(f"测试集: {len(test_dataset)} 张图片")

# 看一批数据长什么样
images, labels = next(iter(train_loader))
print(f"一批图像的形状: {images.shape}")  # [64, 1, 28, 28]
print(f"一批标签的形状: {labels.shape}")  # [64]
```

输出：
```
训练集: 60000 张图片
测试集: 10000 张图片
一批图像的形状: torch.Size([64, 1, 28, 28])
一批标签的形状: torch.Size([64])
```

### 2.2 可视化几个样本

```python
# 显示前 6 张图片
fig, axes = plt.subplots(2, 3, figsize=(8, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ax.set_title(f'标签: {labels[i].item()}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## 三、定义网络：nn.Module

PyTorch 中所有神经网络都继承自 `nn.Module`：

```python
class FullyConnectedNet(nn.Module):
    """三层全连接神经网络，用于 MNIST 分类"""

    def __init__(self):
        super().__init__()
        # 定义网络层
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层 → 隐藏层1
        self.fc2 = nn.Linear(128, 64)        # 隐藏层1 → 隐藏层2
        self.fc3 = nn.Linear(64, 10)         # 隐藏层2 → 输出层（10个数字）
        self.dropout = nn.Dropout(0.2)       # Dropout 防止过拟合

    def forward(self, x):
        """
        前向传播（只需定义这个，反向传播自动算）
        x: 输入张量 [batch_size, 1, 28, 28]
        """
        # 展平：将 [batch, 1, 28, 28] → [batch, 784]
        x = x.view(x.size(0), -1)

        # 隐藏层1: ReLU 激活
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 隐藏层2: ReLU 激活
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # 输出层：不做 softmax（CrossEntropyLoss 内部会做）
        x = self.fc3(x)

        return x

# 实例化模型
model = FullyConnectedNet()
print(model)
```

输出：
```
FullyConnectedNet(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=10, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
)
```

### 为什么输出层不做 Softmax？

`nn.CrossEntropyLoss` 内部包含了 `LogSoftmax + NLLLoss`，所以输出层直接输出**原始 logits** 即可。

如果你需要概率值（比如做推理展示），用 `F.softmax(model(x), dim=1)`。

---

## 四、训练循环

```python
# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"训练设备: {device}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练
train_losses = []
train_accs = []

for epoch in range(EPOCHS):
    model.train()  # 切换到训练模式（启用 Dropout）
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 1. 清零梯度
        optimizer.zero_grad()

        # 2. 前向传播
        output = model(data)

        # 3. 计算损失
        loss = criterion(output, target)

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 每 200 个 batch 打印一次
        if batch_idx % 200 == 199:
            avg_loss = running_loss / 200
            acc = 100. * correct / total
            print(f'Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | '
                  f'Loss: {avg_loss:.4f} | Acc: {acc:.2f}%')
            running_loss = 0.0

    # 记录每个 epoch 的数据
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f'═══ Epoch {epoch+1} 完成 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% ═══')
```

训练过程中你会看到类似这样的输出：
```
Epoch 1/5 | Batch 200/938 | Loss: 0.3825 | Acc: 87.50%
Epoch 1/5 | Batch 400/938 | Loss: 0.2562 | Acc: 91.25%
...
═══ Epoch 1 完成 | Loss: 0.3012 | Acc: 91.33% ═══
═══ Epoch 5 完成 | Loss: 0.0815 | Acc: 97.52% ═══
```

### 训练模式 vs 评估模式

```python
model.train()   # ✅ 训练时：Dropout 生效，BN 用 batch 统计量
model.eval()    # ✅ 评估时：Dropout 关闭，BN 用全局统计量
```

**忘记切换模式是新手最常见的 bug 之一。**

---

## 五、模型评估

```python
model.eval()  # 切换到评估模式（关闭 Dropout）
correct = 0
total = 0

# 禁用梯度追踪（节省内存，加快速度）
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

test_acc = 100. * correct / total
print(f'测试集准确率: {test_acc:.2f}%')
```

输出：
```
测试集准确率: 97.35%
```

### 查看错误分类的样本

```python
# 收集所有预测结果
model.eval()
all_images = []
all_preds = []
all_labels = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        _, preds = output.max(1)
        all_images.append(data.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(target)

images = torch.cat(all_images)
preds = torch.cat(all_preds)
labels = torch.cat(all_labels)

# 找出预测错的
wrong_mask = preds != labels
wrong_images = images[wrong_mask]
wrong_preds = preds[wrong_mask]
wrong_labels = labels[wrong_mask]

print(f"总共错 {wrong_mask.sum().item()} 张（准确率 {100-wrong_mask.sum().item()/len(labels)*100:.2f}%）")

# 显示前 9 个错误
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    if i < len(wrong_images):
        ax.imshow(wrong_images[i].squeeze(), cmap='gray')
        ax.set_title(f'预测: {wrong_preds[i].item()} | 真实: {wrong_labels[i].item()}', color='red')
        ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## 六、保存与加载模型

```python
# 保存（推荐：只保存参数，不保存结构）
torch.save(model.state_dict(), 'mnist_fc.pth')

# 加载
model = FullyConnectedNet()
model.load_state_dict(torch.load('mnist_fc.pth'))
model.eval()
print("模型加载完成 ✅")
```

---

## 七、完整代码

把所有代码合并成一个文件 `train_mnist.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FullyConnectedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=64, shuffle=False
    )

    # 模型
    model = FullyConnectedNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    for epoch in range(5):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} 完成')

    # 评估
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, preds = model(data).max(1)
            total += target.size(0)
            correct += preds.eq(target).sum().item()
    print(f'测试准确率: {100. * correct / total:.2f}%')

    # 保存
    torch.save(model.state_dict(), 'mnist_fc.pth')


if __name__ == '__main__':
    main()
```

---

## 八、常见问题

**Q：为什么我训练时 loss 不下降？**
A：（1）检查学习率是不是太大/太小（0.001 是安全值）；（2）检查数据归一化；（3）检查是否忘记 `optimizer.zero_grad()`。

**Q：全连接网络和卷积网络有什么区别？**
A：全连接把每个像素独立看待，忽略了图片的**空间结构**（相邻像素的关系）。CNN 用卷积核保留空间信息，效果更好——下一篇会讲。

**Q：Dropout 是什么原理？**
A：训练时随机让一部分神经元输出为 0（本文中 20%），迫使网络不依赖某个特定神经元，提高泛化能力。

---

## 总结

本文你完成了：

| 环节 | 技术 | 掌握 |
|------|------|------|
| 数据准备 | Dataset、DataLoader、Transform | ✅ |
| 模型定义 | nn.Module、nn.Linear、forward() | ✅ **核心** |
| 训练循环 | Zero_grad → Forward → Loss → Backward → Step | ✅ **核心** |
| 评估 | model.eval()、torch.no_grad() | ✅ |
| 保存/加载 | state_dict | ✅ |
| 完整流程 | 端到端训练一个识别手写数字的网络 | ✅ **里程碑** |

你已经从"了解深度学习"进入了**能实战深度学习的阶段**。

**下一步推荐：**
- [神经网络架构详解：CNN 卷积神经网络](/2026/07/03/神经网络架构详解：CNN卷积神经网络/)（即将发布）
- [实战：手写数字识别（MNIST）](/2026/07/03/实战：手写数字识别MNIST/)（即将发布）
