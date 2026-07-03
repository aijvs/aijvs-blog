---
title: 神经网络架构详解：CNN 卷积神经网络
date: 2026-07-03 14:00:00
categories:
  - 深度学习
tags:
  - CNN
  - 卷积神经网络
  - 进阶教程
  - 计算机视觉
  - PyTorch
cover: /img/posts/cnn-cover.png
description: CNN 卷积神经网络完整教程：卷积层、池化层、感受野的原理与 PyTorch 实现，附 CIFAR-10 分类实战。
---

## 引言

上一篇文章我们用全连接网络实现了手写数字识别——但你知道吗？如果换成人脸识别、自动驾驶场景，全连接网络基本不 work。

原因很简单：**全连接把 28×28 的图片展平成 784 个独立的像素，完全丢掉了空间结构信息。** 一张猫的图片，把像素随机打乱，全连接网络认不出来了。

**CNN（卷积神经网络）** 就是来解决这个问题的——它用卷积核在图片上滑动，保留空间结构，捕捉局部特征。这也是为什么 CNN 统治了计算机视觉领域近十年。

---

## 前置知识

- [PyTorch 实战（二）：构建第一个全连接网络](/2026/07/03/PyTorch实战（二）：构建第一个全连接网络/)
- [AI数学基础（一）：线性代数——让机器学会看向量](/2026/06/17/AI数学基础（一）：线性代数——让机器学会看向量/)

需要已经会用 PyTorch 搭网络、跑训练循环。

---

## 一、CNN 的核心思想

### 为什么全连接不行？

全连接层的参数量 = 输入维度 × 输出维度。对于一张 256×256 的彩色图片：

```
输入: 256 × 256 × 3 = 196,608
全连接层（128个神经元）: 196,608 × 128 ≈ 2500 万参数
```

**2500 万个参数就一层！** 三层下去直接上亿，根本训不动。

### CNN 的三大优势

| 全连接 | CNN |
|--------|-----|
| 每个像素独立 | **局部连接** — 只关注局部区域 |
| 不同位置要重复学习 | **权值共享** — 同一个卷积核扫过整张图 |
| 参数量爆炸 | **参数量可控** — 取决于卷积核大小而非图片大小 |

---

## 二、卷积层详解

### 2.1 卷积操作

想象你在图片上放了一个小窗口（3×3 像素），窗口在每个位置都和图片做**点积运算**：

```
输入图片 (5×5)          卷积核 (3×3)          输出特征图 (3×3)
┌──┬──┬──┬──┬──┐       ┌──┬──┬──┐          ┌───┬───┬───┐
│1 │2 │3 │4 │5 │       │1 │0 │-1│          │ ? │ ? │ ? │
├──┼──┼──┼──┼──┤       ├──┼──┼──┤          ├───┼───┼───┤
│6 │7 │8 │9 │10│       │2 │0 │-2│          │ ? │ ? │ ? │
├──┼──┼──┼──┼──┤       ├──┼──┼──┤          ├───┼───┼───┤
│11│12│13│14│15│       │1 │0 │-1│          │ ? │ ? │ ? │
├──┼──┼──┼──┼──┤       └──┴──┴──┘          └───┴───┴───┘
│16│17│18│19│20│
├──┼──┼──┼──┼──┤
│21│22│23│24│25│
└──┴──┴──┴──┴──┘

第一个位置计算:
1×1 + 2×0 + 3×(-1) + 6×2 + 7×0 + 8×(-2) + 11×1 + 12×0 + 13×(-1)
= 1 + 0 - 3 + 12 + 0 - 16 + 11 + 0 - 13 = -8
```

这个卷积核是一个**垂直边缘检测器** — 它能捕捉到图像中的垂直纹理。

### 2.2 关键参数

```python
import torch.nn as nn

# 卷积层定义
conv = nn.Conv2d(
    in_channels=3,     # 输入通道数（RGB图片=3，灰度图=1）
    out_channels=16,   # 输出通道数（用多少个卷积核）
    kernel_size=3,     # 卷积核大小（3×3）
    stride=1,          # 步长（每次移动几个像素）
    padding=1          # 填充（保持尺寸不变）
)
```

**参数对输出的影响：**

```
输出尺寸 = (输入尺寸 + 2×padding - kernel_size) / stride + 1

例：输入 32×32, kernel=3, padding=1, stride=1
输出 = (32 + 2 - 3) / 1 + 1 = 32  # 尺寸不变

例：输入 32×32, kernel=3, padding=0, stride=2
输出 = (32 + 0 - 3) / 2 + 1 = 15.5 → 15  # 下采样
```

### 2.3 感受野（Receptive Field）

卷积网络中，越深的层能看到输入图像上越大的区域：

| 层 | 卷积核 | 感受野 |
|:--:|:------:|:------:|
| 第1层卷积 | 3×3 | 3×3 |
| 第2层卷积 | 3×3 | 5×5 |
| 第3层卷积 | 3×3 | 7×7 |

**两层 3×3 卷积 = 一层 5×5 卷积**，但参数量更少（2×9 vs 25）。所以现代 CNN 倾向用多层小卷积核堆叠。

---

## 三、池化层

池化层对特征图做**下采样**，减少参数量，增加平移不变性。

```python
# 最大池化：取窗口内最大值
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 输出尺寸减半：32×32 → 16×16
```

```
最大池化 (2×2, stride=2):
┌──┬──┐    ┌──┐
│1 │5 │    │5 │
├──┼──┤ →  ├──┤
│2 │8 │    │8 │
└──┴──┘    └──┘
```

**平均池化**也常用，但在计算机视觉中最大池化更主流（保留最强的特征响应）。

---

## 四、经典 CNN 架构实战：CIFAR-10 分类

我们来搭建一个类似 **VGG 风格**的 CNN——堆叠卷积层 + 池化层，最后接全连接分类。

### 4.1 模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")


class SimpleCNN(nn.Module):
    """简版 VGG 风格 CNN"""

    def __init__(self, num_classes=10):
        super().__init__()

        # 特征提取器（卷积部分）
        self.features = nn.Sequential(
            # Block 1: 32×32 → 32×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32×32 → 16×16

            # Block 2: 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16×16 → 8×8

            # Block 3: 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8×8 → 4×4
        )

        # 分类器（全连接部分）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x


model = SimpleCNN().to(device)
print(model)
```

**参数量对比：**
```
SimpleCNN: 约 85 万参数
等效全连接网络: 图片 32×32×3=3072，一层 256 个神经元 → 78 万参数
但 CNN 有 6 层卷积，同样深度的全连接网络参数量会爆炸
```

### 4.2 数据加载（CIFAR-10）

CIFAR-10 是 32×32 的彩色图片（10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）。

```python
# 数据增强（提高泛化能力）
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # 随机水平翻转
    transforms.RandomCrop(32, padding=4),# 随机裁剪
    transforms.ColorJitter(0.1, 0.1),   # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),       # CIFAR-10 的 RGB 均值
        (0.2470, 0.2435, 0.2616)        # CIFAR-10 的 RGB 标准差
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

print(f"训练集: {len(train_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")
print(f"类别: {train_dataset.classes}")
```

### 4.3 训练

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = output.max(1)
        total += target.size(0)
        correct += preds.eq(target).sum().item()

    scheduler.step()

    train_acc = 100. * correct / total
    print(f'Epoch {epoch+1:2d}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | '
          f'Train Acc: {train_acc:.2f}%')

    # 每个 epoch 评估测试集
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, preds = model(data).max(1)
            test_total += target.size(0)
            test_correct += preds.eq(target).sum().item()
    test_acc = 100. * test_correct / test_total
    print(f'            Test  Acc: {test_acc:.2f}%')

print("训练完成 ✅")
```

用 GPU 训练 30 个 epoch 大约 5-10 分钟，最终测试准确率应达到 **80-85%**（对于这个简单的网络来说已经不错了）。

### 4.4 可视化卷积核

```python
# 查看第一层卷积核
first_conv = model.features[0]
weights = first_conv.weight.data.cpu()

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < weights.size(0):
        # 每个卷积核有 3 个通道（RGB），取平均值可视化
        kernel = weights[i].mean(dim=0)
        ax.imshow(kernel, cmap='viridis')
        ax.axis('off')
plt.suptitle('第一层 32 个卷积核可视化')
plt.show()
```

你会看到：有的卷积核学会检测边缘（黑白条纹），有的学会检测颜色块，有的是纹理模式——这就是 CNN 的"神经元"。

---

## 五、CNN 设计原则

### 5.1 经典设计模式

```
输入 → [Conv → ReLU]×N → Pool → [Conv → ReLU]×M → Pool → ... → FC → FC → Softmax
                              ↓                                ↓
                         逐步加深通道                      逐步减少神经元
                         逐步缩小尺寸
```

| 层类型 | 通道数变化 | 尺寸变化 |
|--------|-----------|---------|
| 浅层卷积 | 3 → 32 → 64 | 32×32 → 16×16 |
| 中层卷积 | 64 → 128 → 256 | 16×16 → 8×8 |
| 深层卷积 | 256 → 512 | 8×8 → 4×4 |
| 全连接 | 512×4×4 → 256 → 10 | 展平 |

### 5.2 经典 CNN 演进

| 模型 | 年份 | 特点 | 参数量 |
|:----:|:----:|------|:------:|
| LeNet-5 | 1998 | 第一个 CNN，手写数字识别 | 6 万 |
| AlexNet | 2012 | ImageNet 冠军，深度学习元年 | 6000 万 |
| VGG-16 | 2014 | 堆叠 3×3 小卷积核，简洁优雅 | 1.38 亿 |
| ResNet-50 | 2015 | 残差连接，解决了深层网络梯度消失 | 2500 万 |
| EfficientNet | 2019 | 神经架构搜索，效率最高 | 400 万-3000 万 |

**ResNet 的残差连接**是近年来最重要的创新之一：

```python
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接：跳过两层，直接加上输入
        return F.relu(out)
```

残差连接让梯度可以直接通过跳跃连接反向传播，解决了层数太深梯度消失的问题——这也是现在所有大模型（包括 Transformer）都在用残差连接的原因。

---

## 六、三张图看懂 CNN 流程

```
输入图片 (3×32×32)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                 特征提取器（卷积部分）                  │
├─────────────────────────────────────────────────────┤
│  Conv 3×32×32 → 32×32×32  (32个卷积核，提取边缘纹理) │
│  Conv 32×32×32 → 32×32×32  (第二层，提取简单形状)    │
│  Pool 32×32×32 → 32×16×16  (下采样，保留最强特征)    │
│  Conv 32×16×16 → 64×16×16  (更深，提取图案模式)     │
│  Conv 64×16×16 → 64×16×16                          │
│  Pool 64×16×16 → 64×8×8                            │
│  Conv 64×8×8 → 128×8×8   (最深，提取语义概念)      │
│  Conv 128×8×8 → 128×8×8                            │
│  Pool 128×8×8 → 128×4×4                            │
└──────────────────────┬──────────────────────────────┘
                       │ 展平: 128×4×4 = 2048
                       ▼
┌─────────────────────────────────────────────────────┐
│                 分类器（全连接部分）                    │
├─────────────────────────────────────────────────────┤
│            FC 2048 → 256   (特征 → 类别置信度)       │
│            FC 256 → 10                              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
                    [0.01, 0.02, 0.92, ...]  ← 输出概率
```

---

## 七、总结

本文你掌握了：

| 知识点 | 掌握 |
|--------|:----:|
| CNN 为什么比全连接更适合图片 | ✅ |
| 卷积操作原理（卷积核、步长、填充） | ✅ |
| 池化层的作用 | ✅ |
| 感受野与多层小卷积核 | ✅ |
| 用 PyTorch 搭建 CNN 完成 CIFAR-10 分类 | ✅ **实战** |
| 经典 CNN 架构演进（VGG/ResNet） | ✅ |
| 残差连接原理 | ✅ |

**下一步推荐：**
- [计算机视觉入门：图像处理基础](/2026/07/03/计算机视觉入门：图像处理基础/)（即将发布）
- [实战：图像分类（CIFAR-10）](/2026/07/03/实战：图像分类CIFAR-10/)（即将发布）
- [PyTorch 实战（一）：张量操作与自动微分](/2026/07/03/PyTorch实战（一）：张量操作与自动微分/)
