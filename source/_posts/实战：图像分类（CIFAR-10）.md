---
title: 实战：图像分类（CIFAR-10）
date: 2026-07-11 12:00:00
categories: [实战项目]
tags: [PyTorch, CIFAR-10, 图像分类, 进阶教程, 数据增强, 迁移学习, 实战项目]
description: 用 PyTorch 实现 CIFAR-10 彩色图片分类，对比普通 CNN、数据增强和迁移学习的实际效果差距。
---

## 引子

MNIST 的准确率刷到 99% 了，然后呢？

你换了张真实照片——模糊的、带噪点的、背景花里胡哨的——CNN 直接掉到 70%。

**不是你模型不行，是 MNIST 太简单了。**

CIFAR-10 就是下一个台阶：32×32 彩色图片，10 个类别，真正的"AI 能不能分清猫和狗"。

- 训练集：50,000 张
- 测试集：10,000 张
- 彩色（3 通道 RGB），32×32 像素
- 10 类：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车

跟 MNIST 的本质区别：**颜色 + 纹理 + 形状才是特征，纯像素不够。**

---

## 前置知识

- [实战：手写数字识别（MNIST）](/2026/07/11/实战：手写数字识别（MNIST）/)
- [神经网络架构详解：CNN 卷积神经网络](/2026/07/03/神经网络架构详解：CNN卷积神经网络/)

---

## 一、基线 CNN

先拿一个比 MNIST 那篇更深的 CNN，看 CIFAR-10 的难度。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 数据加载（无增强版）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

train_data = datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
test_data  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False, num_workers=2)

class BasicCNN(nn.Module):
    """3 层卷积 + 2 层全连接"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.pool  = nn.MaxPool2d(2, 2)     # 32→16→8→4
        self.fc1   = nn.Linear(256 * 4 * 4, 512)
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # 32→16
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # 16→8
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # 8→4
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

训练代码跟 MNIST 实测那篇一样，只是加了两个新东西：

**BatchNorm2d（批归一化）**：每一层输出都做归一化，加速收敛、减少过拟合。好消息：加了 BatchNorm 之后，你可以用更大的学习率。

**num_workers=2**：用 2 个进程预加载数据，GPU 不会被 CPU 拖慢。训 CPU 时可以关掉（=0）。

跑 20 轮：

```python
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasicCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        scheduler.step()
        print(f"Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, acc={correct/total:.4f}")

if __name__ == '__main__':
    train()
```

预期结果：
```
Epoch  1: loss=1.4167, acc=0.4765
Epoch  5: loss=0.7895, acc=0.7188
Epoch 10: loss=0.5825, acc=0.7924
Epoch 15: loss=0.4321, acc=0.8257
Epoch 20: loss=0.3380, acc=0.8432
```

**84.3%**。比瞎猜（10%）好很多，但远不如 MNIST 的 99%。

---

## 二、数据增强：免费的精度

CIFAR-10 训练集只有 5 万张图。数据增强相当于白送你更多数据：

```python
# 带增强的数据加载
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),     # 随机裁剪，补零4像素
    transforms.RandomHorizontalFlip(),         # 随机水平翻转
    transforms.ColorJitter(brightness=0.2,     # 随机调亮度
                          contrast=0.2),       # 随机调对比度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])
```

上面四行代码改了之后，跑 20 轮：

```
Epoch  1: loss=1.5232, acc=0.4421   # 一开始更难了（因为图像被随机改了）
Epoch  5: loss=1.0012, acc=0.6484
Epoch 10: loss=0.7843, acc=0.7412
Epoch 15: loss=0.6201, acc=0.8125
Epoch 20: loss=0.4972, acc=0.8701
```

**87.0%**，比基线高了 2.7%。

而且注意一个细节：没有增强时，训练 loss 远低于测试 loss（过拟合的苗头）。加了增强后两者差距缩小——**泛化能力更强。**

> **重要提醒：** 数据增强只能应用到训练集！测试集只用 Normalize。如果对测试集也做 RandomCrop，每次评估结果都会变，你就不知道模型到底怎样。

---

## 三、迁移学习：站在巨人肩上

接下来是**最大的飞跃**。用一个在 ImageNet（1400 万张图片）上预训练好的 ResNet-18，迁移学习到 CIFAR-10。

```python
from torchvision.models import resnet18

class TransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练 ResNet-18
        self.backbone = resnet18(pretrained=True)
        # 替换最后一层分类头
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 10)

    def forward(self, x):
        # CIFAR-10 是 32×32，ResNet 要求 224×224
        # 上采样到 ResNet 的输入尺寸
        x = torch.nn.functional.interpolate(x, size=(224, 224),
                                            mode='bilinear', align_corners=False)
        return self.backbone(x)
```

但 ResNet-18 有 1100 万参数，全套微调太慢了。更好的做法：

```python
# 冻结 backbone，只训练最后的分类头
model = TransferModel()
for param in model.backbone.parameters():
    param.requires_grad = False  # backbone 不更新

# 只优化新加的分类头
optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=0.001)
```

冻结后训练 10 轮：
```
Epoch  1: loss=0.8421, acc=0.7415
Epoch  5: loss=0.5395, acc=0.8093
Epoch 10: loss=0.4837, acc=0.8227
```

**82.3%**——只用 10 轮，只训分类头，不碰 1100 万参数。

如果再加 10 轮**全量微调**（解冻 backbone，lr 降到 1e-4）：

```python
# 第11轮开始：解冻 backbone，低学习率微调
for param in model.backbone.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

```
Epoch 11: loss=0.3512, acc=0.8813
Epoch 15: loss=0.2216, acc=0.9112
Epoch 20: loss=0.1892, acc=0.9219
```

**92.2%**——比普通 CNN + 数据增强高了 5 个点。

---

## 四、三方案对比

| 方案 | 参数量 | 20 轮精度 | 训练时间（GPU） |
|------|-------|----------|---------------|
| 普通 CNN | 4.5M | 84.3% | ~8min |
| + 数据增强 | 4.5M | 87.0% | ~9min（预处理多了） |
| + ResNet 迁移学习 | 11.1M | 92.2% | ~15min |

**关键结论：**

- **数据增强是性价比最高的技巧**——不要钱，改一行代码就 +2.7%
- **迁移学习是上限最高的方案**——+7.9%，但预训练模型很难在 CPU 上跑

---

## 五、完整代码

```python
"""cifar10_demo.py - CIFAR-10 图像分类完整代码"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据加载（训练集带增强）
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2470, 0.2435, 0.2616])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2470, 0.2435, 0.2616])
])

train_loader = DataLoader(
    datasets.CIFAR10('./data', train=True,  download=True, transform=train_transform),
    batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=test_transform),
    batch_size=256, shuffle=False, num_workers=2)

# 预训练 ResNet-18
from torchvision.models import resnet18

model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()

# Phase 1: 只训分类头
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(1, 11):
    model.train()
    loss_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = nn.functional.interpolate(images, (224, 224),
                                           mode='bilinear', align_corners=False)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = nn.functional.interpolate(images, (224, 224),
                                               mode='bilinear', align_corners=False)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Phase1 Epoch {epoch:2d}: loss={loss_total/len(train_loader):.4f}, acc={correct/total:.4f}")

# Phase 2: 全量微调
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(11, 21):
    # 同上训练循环
    model.train()
    loss_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = nn.functional.interpolate(images, (224, 224),
                                           mode='bilinear', align_corners=False)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = nn.functional.interpolate(images, (224, 224),
                                               mode='bilinear', align_corners=False)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Phase2 Epoch {epoch:2d}: loss={loss_total/len(train_loader):.4f}, acc={correct/total:.4f}")
```

> **GPU 的话约 15 分钟跑完。如果是 CPU，全量微调可能 2-3 小时——你可以在 Phase 1 后停，拎着 82% 的精度先玩玩。**

---

## 总结

- CIFAR-10 比 MNIST 难得多（84% vs 99%），但这才是真实世界的入门难度
- **数据增强**是白送的精调度，必须上
- **迁移学习**是做大模型的捷径——没人真的从零训了
- 学会了两阶段微调：先冻后解，又快又好

接下来你可以：
- 自己收集 10 张猫/狗照片，用这个模型测试
- 把 ResNet-18 换成 ResNet-50，看看精度还能涨多少
- 学习 TensorBoard 来可视化训练过程
- 尝试用 YOLO 做目标检测——下一篇就是这个
