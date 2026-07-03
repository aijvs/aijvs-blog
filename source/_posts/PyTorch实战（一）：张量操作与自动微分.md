---
title: PyTorch 实战（一）：张量操作与自动微分
date: 2026-07-03 11:00:00
categories:
  - 深度学习
tags:
  - PyTorch
  - 深度学习
  - 基础教程
  - 张量
  - 自动微分
cover: /img/posts/pytorch-tensor-cover.png
description: PyTorch 的核心基础：张量创建、索引运算、GPU加速，以及自动求导机制的完整讲解与实战。
---

## 引言

> 上一篇文章用了 KNN 做分类，KNN 并不需要梯度。但从神经网络开始，**自动求导（Autograd）** 就成了最核心的基础设施。

PyTorch 之所以成为深度学习领域的第一框架（没有之一），两个原因：

1. **动态计算图** — 边运行边构建，调试方便，灵活度高
2. **自动求导** — 你只需要定义前向传播，反向传播自动算好

本文从零开始，带你掌握 PyTorch 的两大核心：**张量操作** 和 **自动微分**。

---

## 前置知识

- [Python 编程基础（二）：变量、数据类型和基本运算](/2026/06/17/Python编程基础（二）：变量、数据类型和基本运算/)
- [Python 编程基础（三）：NumPy——AI工程师的第一件武器](/2026/06/17/Python编程基础（三）：NumPy——AI工程师的第一件武器/)
- [AI数学基础（一）：线性代数——让机器学会看向量](/2026/06/17/AI数学基础（一）：线性代数——让机器学会看向量/)

会用 NumPy 就够了，PyTorch 的 Tensor API 和 NumPy 几乎一模一样。

---

## 一、安装与配置

```bash
pip install torch torchvision torchaudio
```

验证安装：

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号:    {torch.cuda.get_device_name(0)}")
```

---

## 二、张量（Tensor）—— 深度学习的"数字容器"

张量是多维数组的泛化概念：

| 维度 | 名称 | 示例 |
|------|------|------|
| 0D | 标量 | 3.14 |
| 1D | 向量 | [1, 2, 3] |
| 2D | 矩阵 | [[1,2],[3,4]] |
| 3D | 3阶张量 | 一张彩色图片 (H×W×3) |
| 4D | 4阶张量 | 一批图片 (N×H×W×3) |

### 2.1 创建张量

```python
import torch
import numpy as np

# 从列表创建
a = torch.tensor([[1, 2], [3, 4]])
print(f"从列表创建:\n{a}")

# 从 NumPy 创建
b = torch.from_numpy(np.array([[5, 6], [7, 8]]))
print(f"从 NumPy 创建:\n{b}")

# 特殊张量
zeros = torch.zeros(2, 3)        # 全 0
ones = torch.ones(2, 3)          # 全 1
eye = torch.eye(3)               # 单位矩阵
rand = torch.randn(2, 4)         # 标准正态分布随机数
print(f"全零:\n{zeros}")
print(f"随机:\n{rand}")

# 指定数据类型和设备
x = torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
print(f"指定设备和类型: {x}")
```

### 2.2 张量属性

```python
x = torch.randn(3, 4, 5)
print(f"形状:     {x.shape}")
print(f"维度数:   {x.ndim}")
print(f"数据类型: {x.dtype}")
print(f"元素总数: {x.numel()}")
print(f"设备:     {x.device}")
```

输出：
```
形状:     torch.Size([3, 4, 5])
维度数:   3
数据类型: torch.float32
元素总数: 60
设备:     cpu
```

### 2.3 索引与切片（和 NumPy 一样）

```python
x = torch.randn(4, 4)
print(f"原始:\n{x}")

# 基本索引
print(f"第一行:    {x[0]}")
print(f"第二列:    {x[:, 1]}")
print(f"子矩阵:    {x[1:3, 1:3]}")

# 花式索引
indices = torch.tensor([0, 2, 3])
print(f"花式索引:  {x[indices]}")

# 布尔索引
print(f"大于0的值: {x[x > 0]}")
```

### 2.4 张量运算

```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# 算术运算
print(f"加法: \n{a + b}")
print(f"乘法（逐元素）: \n{a * b}")

# 矩阵乘法
print(f"矩阵乘法: \n{torch.mm(a, b)}")    # 2D 专用
print(f"矩阵乘法（@运算符）: \n{a @ b}")   # 更简洁

# 数学函数
print(f"求和:     {a.sum()}")
print(f"均值:     {a.mean()}")
print(f"标准差:   {a.std()}")
print(f"最大值:   {a.max()}")
print(f"转置:     \n{a.T}")

# 形状变换
c = torch.randn(2, 3, 4)
print(f"原始形状: {c.shape}")
print(f"展平:     {c.view(-1).shape}")    # view(-1) = 展平
print(f"重塑:     {c.view(6, 4).shape}")  # 重塑为 6×4
```

### 2.5 广播机制

和 NumPy 一样，PyTorch 会自动扩展维度不一致的运算：

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 形状 (2, 3)
b = torch.tensor([10, 20, 30])             # 形状 (3,)

print(a + b)  # b 自动广播为 (2, 3)
# 输出: [[11, 22, 33], [14, 25, 36]]
```

### 2.6 CPU 与 GPU 切换

```python
# 检查 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

# CPU 计算
%timeit x @ y

# 移到 GPU
x_gpu = x.to(device)
y_gpu = y.to(device)
%timeit x_gpu @ y_gpu
```

在 GPU 上，矩阵乘法通常比 CPU 快 **10-50 倍**（取决于显卡）。

---

## 三、自动微分（Autograd）—— 深度学习的引擎

自动微分是 PyTorch 最核心的机制。你只需要定义前向计算，PyTorch 会自动记录所有操作，并在你调用 `backward()` 时自动计算梯度。

### 3.1 基本原理

```python
# 创建需要计算梯度的张量
x = torch.tensor([2.0], requires_grad=True)

# 前向计算
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# 反向传播（自动计算梯度）
y.backward()

# 查看梯度 dy/dx = 2x + 3 = 2*2 + 3 = 7
print(f"梯度 dy/dx: {x.grad}")  # 输出: tensor([7.])
```

**关键理解**：PyTorch 会自动构建计算图，`backward()` 沿图反向传播梯度。

### 3.2 计算图可视化

```
x(2.0) ──┬──► x²(4.0) ──┐
         │               ├──► (+) ──► y(11.0)
         └──► 3x(6.0) ──┘
                + 1 ──────┘

backward() 反向传播:
dy/dx = 2x + 3 = 7
```

### 3.3 复杂例子：线性回归

让我们用自动微分手写一个线性回归：

```python
import torch
import matplotlib.pyplot as plt

# 生成合成数据：y = 2x + 1 + 噪音
torch.manual_seed(42)
X = torch.linspace(-1, 1, 100).reshape(-1, 1)
true_w, true_b = 2.0, 1.0
y = true_w * X + true_b + torch.randn_like(X) * 0.1

# 初始化参数
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练
learning_rate = 0.1
losses = []

for epoch in range(100):
    # 前向传播: y_pred = w*x + b
    y_pred = X @ w + b

    # 计算 MSE 损失
    loss = ((y_pred - y) ** 2).mean()

    # 反向传播
    loss.backward()

    # 梯度下降更新（关闭梯度追踪）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # 清零梯度（重要！）
        w.grad.zero_()
        b.grad.zero_()

    losses.append(loss.item())

print(f"真实参数: w={true_w}, b={true_b}")
print(f"学习结果: w={w.item():.4f}, b={b.item():.4f}")

# 绘制损失曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失')

plt.subplot(1, 2, 2)
plt.scatter(X.numpy(), y.numpy(), alpha=0.5)
plt.plot(X.numpy(), (X @ w + b).detach().numpy(), 'r-', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('拟合结果')
plt.show()
```

输出示例：
```
真实参数: w=2.0, b=1.0
学习结果: w=1.9892, b=1.0037
```

### 3.4 梯度清零：为什么每次都要做？

```python
# 错误示范
loss.backward()
w -= lr * w.grad
# ❌ 忘记 zero_() → 下次 backward() 时梯度会累加!

# 正确做法
loss.backward()
with torch.no_grad():
    w -= lr * w.grad
w.grad.zero_()  # ✅ 清零
```

PyTorch 默认会**累加梯度**，这是为了 RNN 等需要累积梯度的场景设计的。普通训练每次都要手动清零。

### 3.5 禁用梯度追踪

某些操作不需要梯度（如模型推理、参数更新），用 `torch.no_grad()` 避免构建计算图：

```python
# 推理时
with torch.no_grad():
    y_pred = model(X_test)  # 节省内存，速度更快

# 或全局禁用
torch.set_grad_enabled(False)
```

---

## 四、实用技巧

### 4.1 随机种子

```python
# 保证结果可复现
torch.manual_seed(42)        # CPU 随机数
torch.cuda.manual_seed(42)   # GPU 随机数
np.random.seed(42)           # NumPy 随机数
```

### 4.2 Tensor 与 NumPy 互转

```python
# Tensor → NumPy
tensor = torch.ones(3, 4)
numpy_arr = tensor.numpy()   # 共享内存！

# NumPy → Tensor
numpy_arr = np.ones((3, 4))
tensor = torch.from_numpy(numpy_arr)  # 也共享内存！

# 不共享内存的拷贝
tensor_clone = torch.tensor(numpy_arr)  # 独立副本
```

> **注意**：CPU 上的 Tensor 和 NumPy 数组共享内存，修改一个会影响另一个。

### 4.3 设备无关的代码

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
data = data.to(device)
labels = labels.to(device)
# 这样写，在任何设备上都能跑
```

---

## 五、总结

本文你掌握了：

| 技能 | 掌握程度 |
|------|---------|
| Tensor 创建与属性 | ✅ 基础 |
| 索引、切片、运算 | ✅ 基础 |
| 广播机制 | ✅ 理解 |
| GPU 加速切换 | ✅ 掌握 |
| **自动求导 Autograd** | ✅ **核心** |
| 手写线性回归 | ✅ 实战 |

**下一步推荐：**
- [PyTorch 实战（二）：构建第一个全连接网络](/2026/07/03/PyTorch实战（二）：构建第一个全连接网络/)（即将发布）
- [机器学习入门：概念与分类全解](/2026/07/03/机器学习入门：概念与分类全解/)
