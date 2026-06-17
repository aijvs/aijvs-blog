---
title: Python编程基础（三）：NumPy——AI工程师的第一件武器
date: 2026-06-17 14:00:00
categories: [基础教程]
tags: [Python, NumPy, 入门教程, 数据处理, AI基础]
description: 从Python原生列表到NumPy数组，效率提升100倍。学会用NumPy批量处理数据，这是通往AI开发的必经之路。
---

## 假设你是一个小学老师……

班上有50个学生，期末考试结束了，你手里有50张成绩单。

现在校长让你做几件事：算全班的平均分、找出最高分和最低分、看看有多少人及格、按成绩排个名次。

如果你一张一张地看，那得看50遍。而且下次月考你还得再这么干一遍。

你肯定不会这么傻。你会拿一张表，把50个分数写成一列，然后一眼扫过去就能找到最高分，拿个计算器一按就能算平均分。

你已经在不知不觉中使用了**"批量处理"**的思维方式——不是一次处理一个数据，而是一次处理一整批数据。

NumPy做的工作，就是这个。

## 为什么AI需要NumPy？

先看一个问题：如果你的大脑处理一个数字要1秒钟，那处理一百万个数字要多久？

如果一次只处理一个——一百万秒，大约11.5天。

但如果你的大脑升级了，可以同时处理一百万个数字——只要一秒。

从11.5天到1秒，这就是NumPy带来的效率提升。

在Python原生世界里，你要处理一百万个数字，就得用一个包含一百万个元素的列表，然后一个一个地算。Python的列表是"慢的"，因为它设计上就不是一个做数学计算的工具。

而NumPy的数组（array）是"快的"，因为它底层用C语言实现，对内存和计算做了大量优化。

让我们来感受一下差距：

```python
import time
import numpy as np

# 创建一百万个数字
size = 1000000
python_list = list(range(size))
numpy_array = np.arange(size)

# Python列表：逐个加1
start = time.time()
result_list = [x + 1 for x in python_list]
print(f"Python列表耗时：{time.time() - start:.4f}秒")

# NumPy数组：批量加1
start = time.time()
result_array = numpy_array + 1
print(f"NumPy数组耗时：{time.time() - start:.4f}秒")
```

在你的电脑上跑一下，你会看到NumPy的速度大约是Python列表的 **50到100倍**。

一百万个数字尚且如此，当你处理上亿的AI训练数据时，没有NumPy是不可想象的。

## 安装NumPy

如果你之前装了Anaconda，NumPy很可能已经装好了。验证一下：

```python
import numpy as np
print(np.__version__)
```

如果输出了版本号（比如 `1.26.4`），说明已经有了。

如果没有，安装很简单：

```python
pip install numpy
```

一行命令，搞定。

## ndarray：数组就像超级表格

NumPy的核心数据结构叫 **ndarray**（N-dimensional array，N维数组）。

你可以把它理解为一个"超级表格"：

- **一维数组** = 一排数字，像一个横向的表格
- **二维数组** = 一个有行有列的表格（就像Excel）
- **三维数组** = 多个表格叠在一起（像一个Excel工作簿有多个sheet）

下面来创建几个：

```python
import numpy as np

# 一维数组——一行数字
scores = np.array([85, 92, 78, 95, 88, 76, 90])
print(scores)
# 输出：[85 92 78 95 88 76 90]

# 二维数组——一个成绩表（3个学生，4门课）
grades = np.array([
    [85, 92, 78, 95],   # 学生1
    [88, 76, 90, 82],   # 学生2
    [72, 95, 88, 91]    # 学生3
])
print(grades)
# 输出：
# [[85 92 78 95]
#  [88 76 90 82]
#  [72 95 88 91]]
```

查看数组的形状（shape）：

```python
print(scores.shape)   # (7,)   —— 7个元素
print(grades.shape)   # (3, 4) —— 3行4列
```

## 创建数组的N种方式

除了直接 `np.array()` 手动输入，NumPy还提供了很多快捷方式：

### 从零开始创建

```python
# 全0数组——初始化一个空表格
zeros = np.zeros((3, 4))
print(zeros)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 全1数组
ones = np.ones((2, 3))
print(ones)
# [[1. 1. 1.]
#  [1. 1. 1.]]

# 指定范围的整数序列
nums = np.arange(0, 10, 2)  # 从0到10，步长2
print(nums)  # [0 2 4 6 8]
```

### 随机数组——AI训练经常用

```python
# 0到1之间的随机数
random_nums = np.random.rand(3, 3)
print(random_nums)

# 指定范围的随机整数
random_ints = np.random.randint(1, 100, size=5)
print(random_ints)
```

`np.random` 在AI里非常重要。神经网络的权重初始化、数据增强、随机采样……到处都离不开随机数。

### 等间隔数组

```python
# 从0到1，均匀分成5个点
points = np.linspace(0, 1, 5)
print(points)  # [0.   0.25 0.5  0.75 1.  ]
```

## 数组运算：买菜算账的智慧

NumPy的运算分两种，搞清楚它们的区别很重要：

### 逐元素运算——每个位置各自算

```python
prices = np.array([3.5, 5.0, 2.8, 4.2])  # 每种蔬菜的单价
quantities = np.array([2, 1, 3, 2])      # 每种蔬菜买了几斤

# 每种蔬菜各花了多少钱？
cost_per_item = prices * quantities
print(cost_per_item)
# [ 7.   5.   8.4  8.4]
```

这就像你去菜市场，西红柿3.5一斤买了2斤、白菜5.0一斤买了1斤……你不需要一个一个算，NumPy帮你一次性全算完。

**关键特性：两个数组形状相同，运算会自动对应每个位置的元素。**

标量和数组的运算也一样：

```python
prices = np.array([10, 20, 30, 40])
print(prices + 5)    # [15 25 35 45] —— 每个价格加5
print(prices * 0.8)  # [ 8 16 24 32] —— 八折
```

### 矩阵运算——真正的线性代数

```python
# 矩阵乘法（不是逐元素相乘！）
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 逐元素相乘（不是矩阵乘法）
print(A * B)
# [[ 5 12]
#  [21 32]]

# 矩阵乘法（真正的线性代数）
print(A @ B)
# [[19 22]
#  [43 50]]
```

`*` 是逐元素相乘，`@` 是矩阵乘法。这两个运算完全不同，千万别搞混。

矩阵乘法是AI的核心运算之一。神经网络的"前向传播"，本质上就是一堆矩阵乘法的嵌套。

## 索引和切片：像切蛋糕一样取数据

NumPy的索引语法和Python列表很像，但更强大：

```python
data = np.array([10, 20, 30, 40, 50, 60, 70, 80])

# 取单个元素
print(data[0])     # 10
print(data[-1])    # 80 —— 最后一个

# 切片
print(data[2:5])   # [30 40 50] —— 从第2个到第4个（左闭右开）
print(data[::2])   # [10 30 50 70] —— 每隔一个取一个
print(data[::-1])  # [80 70 60 50 40 30 20 10] —— 反过来
```

二维数组的索引：

```python
grades = np.array([
    [85, 92, 78, 95],
    [88, 76, 90, 82],
    [72, 95, 88, 91]
])

# 取第1个学生的第2门课成绩
print(grades[0, 1])  # 92

# 取所有学生的第3门课成绩
print(grades[:, 2])  # [78 90 88]

# 取前两个学生的所有成绩
print(grades[:2, :])
# [[85 92 78 95]
#  [88 76 90 82]]
```

**口诀：逗号前面管行，逗号后面管列。冒号 `:` 表示"全部"。**

## 常用函数：一个函数解决一整列数据

```python
scores = np.array([85, 92, 78, 95, 88, 76, 90, 82, 73, 96])

print(np.sum(scores))    # 855 —— 总分
print(np.mean(scores))   # 85.5 —— 平均分
print(np.max(scores))    # 96 —— 最高分
print(np.min(scores))    # 73 —— 最低分
print(np.std(scores))    # 7.24... —— 标准差
print(np.median(scores)) # 86.5 —— 中位数

# 有多少人及格（>=60）？
passed = scores >= 60
print(passed)            # [ True  True  True  True  True  True  True  True  True  True]
print(np.sum(passed))    # 10 —— True被当作1，False被当作0，加起来就是及格人数
```

注意最后一行——布尔值可以直接参与数学运算，True就是1，False就是0。这个特性非常实用。

`reshape` 是另一个常用函数：

```python
data = np.arange(12)  # [0 1 2 3 4 5 6 7 8 9 10 11]

# 把一维数组变成3行4列的二维数组
matrix = data.reshape(3, 4)
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 变成4行3列也行
print(data.reshape(4, 3))
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
```

## 实战场景：用NumPy分析学生成绩

让我们把前面学的知识串起来，做一个完整的成绩分析：

```python
import numpy as np

# 50个学生的5门课成绩
np.random.seed(42)  # 固定随机种子，让结果可复现
grades = np.random.randint(40, 100, size=(50, 5))

# 科目名称（纯展示用）
subjects = ["语文", "数学", "英语", "物理", "化学"]

# 每个学生的总分
total_scores = np.sum(grades, axis=1)

# 每个学生的平均分
avg_scores = np.mean(grades, axis=1)

# 每门课的平均分（全班的）
subject_avg = np.mean(grades, axis=0)
print("各科目平均分：")
for i, name in enumerate(subjects):
    print(f"  {name}：{subject_avg[i]:.1f}")

# 找到最高总分的学生
top_student = np.argmax(total_scores)
print(f"\n最高总分：第{top_student + 1}号学生，总分{total_scores[top_student]}")

# 统计各分数段人数
bins = np.array([0, 60, 70, 80, 90, 100])
hist, _ = np.histogram(avg_scores, bins=bins)
labels = ["不及格", "60-69", "70-79", "80-89", "90-100"]
print("\n分数段分布（按平均分）：")
for i, label in enumerate(labels):
    bar = "█" * hist[i]
    print(f"  {label}：{hist[i]}人 {bar}")

# 及格率
pass_rate = np.sum(avg_scores >= 60) / len(avg_scores) * 100
print(f"\n及格率：{pass_rate:.1f}%")
```

运行这段代码，你会得到类似这样的输出：

```
各科目平均分：
  语文：71.0
  数学：69.2
  英语：68.2
  物理：72.6
  化学：67.4

最高总分：第38号学生，总分447

分数段分布（按平均分）：
  不及格：2人 ██
  60-69：11人 ███████████
  70-79：22人 ████████████████████████
  80-89：14人 ████████████████
  90-100：1人 █

及格率：96.0%
```

你看，50个学生、5门课、250个数据点——用NumPy，几行代码就完成了统计、排序、分析。

如果用Python列表和循环来做同样的功能，代码量至少翻三倍，速度还会慢几十倍。

## 关键概念小结

| 概念 | 作用 | 类比 |
|------|------|------|
| `ndarray` | 存储数据的核心结构 | 超级表格 |
| `逐元素运算` | 对应位置各自计算 | 买每种菜各自算价钱 |
| `矩阵乘法 @` | 线性代数运算 | AI神经网络的基础 |
| `索引/切片` | 取出特定位置的值 | 切蛋糕 |
| `axis` | 指定运算方向 | axis=0竖着算，axis=1横着算 |
| `布尔索引` | 用条件筛选数据 | "把及格的人挑出来" |

`axis` 可能是最容易搞混的概念。记住：**axis=0 是跨行（竖着），axis=1 是跨列（横着）。** 或者更直白地记：`axis=0` 压缩行（让行消失），`axis=1` 压缩列（让列消失）。

## 下一站：Pandas

NumPy解决了"高效计算"的问题，但它有一个短板——**不能处理非数值数据**（比如名字、日期、类别标签）。

当你的数据里既有数字又有文字，而且需要做分组、筛选、合并这些操作时，你需要一个更强大的工具：**Pandas**。

NumPy是你的第一件武器，Pandas是第二件。两件配合使用，数据处理就游刃有余了。

不过在那之前，先确保你对NumPy的手感足够好——试着修改上面"成绩分析"的代码，看看能不能加一些自己的分析逻辑。

**真正的学习能力，从来不是看完一篇文章就掌握，而是动手改一改、试一试、犯一犯错误，然后才真正属于你。**

---


[上一篇](/2026/06/17/Python编程基础（二）：变量、数据类型和基本运算/)
[下一篇](/2026/06/17/AI数学基础（一）：线性代数——让机器学会看向量/)
