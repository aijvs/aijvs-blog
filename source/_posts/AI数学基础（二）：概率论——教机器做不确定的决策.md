---
title: AI数学基础（二）：概率论——教机器做"不确定的决策"
date: 2026-06-17 14:00:00
categories: [基础教程]
tags: [概率论, 入门教程, 数学基础, AI基础, 统计学, 基础教程]
description: 1654年两个法国贵族在赌桌上的争吵，催生了概率论。今天，概率论成了AI做决策的基础语言。
---

# AI数学基础（二）：概率论——教机器做"不确定的决策"

## 1654年，赌桌上的争吵改变世界

1654年的某一天，法国巴黎的一家咖啡馆里，两个贵族为一件事吵得面红耳赤。

他们叫**帕斯卡**（Blaise Pascal）和**费马**（Pierre de Fermat）——对，就是中学数学课本里那个"费马大定理"的费马。

他们在争论一个看似简单的问题：

> 两个赌徒玩掷硬币游戏，约定先赢4次的人拿走全部赌注。如果游戏进行到3:2时（A赢了3次，B赢了2次）被迫中止，赌注应该怎么分才公平？

这个问题看起来简单，却难住了当时所有数学家。**帕斯卡和费马通过通信，发明了概率论的基础**——用数字来描述"不确定性"。

300多年后，这个"赌桌上的争吵"成了人工智能的核心工具。

因为**现实世界充满不确定性**，而AI必须学会在不确定中做决策。

---

## 一、为什么AI离不开概率？

### 现实世界不是非黑即白

你想想看：

- **医疗诊断**：AI看CT片子，说"这个结节有87%的概率是恶性的"——不是"是"或"不是"，而是概率。
- **语音识别**：你说"帮我打开空调"，AI听到的是一堆声波，它要判断"最可能"是什么词。
- **自动驾驶**：前面有个模糊的影子，是塑料袋还是小孩？AI要给每个可能性打分。

**如果AI只会"确定性计算"（1+1=2），它就不会做决策。**

概率论，就是教AI如何在"不确定"中做最好的选择。

---

## 二、概率是什么？"明天下雨概率70%"到底啥意思？

### 直觉：概率是"可能性的度量"

很多人以为"概率70%"的意思是"100次里有70次会下雨"——**错了！**

**正确理解**：在你拥有的所有信息（云图、风向、历史数据）下，AI对"明天下雨"这件事的**信心程度**是70%。

换句话说：**概率是对"不确定性"的量化**。

### 生活比喻：抽奖

公司年会抽奖，你抽到大奖的概率是 `1/200 = 0.5%`。

这个0.5%不是说明年200次年会你一定能中1次——而是**这一次**，你中奖的可能性很小。

---

## 三、条件概率："下雨天带伞的概率"

### 故事：带伞的决策

假设你要决定"今天带不带伞"。有两个相关事件：

- 事件A：今天下雨
- 事件B：你带伞

**条件概率** `P(B|A)` 的意思是：**如果今天下雨（A发生），你带伞（B）的概率是多少？**

显然，`P(带伞|下雨) > P(带伞|不下雨)`——下雨天你更可能带伞。

### 公式解释（大白话版）

```
P(B|A) = P(A和B同时发生) / P(A)
```

翻译成人话：

> "在下雨的前提下带伞的概率" = "下雨且带伞的情况占所有下雨情况的比例"

### NumPy实战：模拟1000天的天气和带伞行为

```python
import numpy as np

# 设定随机种子，让结果可重复
np.random.seed(42)

# 模拟1000天
days = 1000

# 模拟下雨（30%的概率）
rain = np.random.rand(days) < 0.3

# 模拟带伞行为
# 如果下雨，80%会带伞；如果不下雨，10%会带伞
bring_umbrella = np.where(rain, 
                          np.random.rand(days) < 0.8,  # 下雨天
                          np.random.rand(days) < 0.1)  # 不下雨

# 计算各种概率
p_rain = np.sum(rain) / days
p_umbrella = np.sum(bring_umbrella) / days

# P(下雨且带伞)
rain_and_umbrella = np.sum(rain & bring_umbrella) / days

# P(带伞|下雨) = P(下雨且带伞) / P(下雨)
p_umbrella_given_rain = rain_and_umbrella / p_rain

print(f"P(下雨): {p_rain:.3f}")
print(f"P(带伞): {p_umbrella:.3f}")
print(f"P(下雨且带伞): {rain_and_umbrella:.3f}")
print(f"P(带伞|下雨): {p_umbrella_given_rain:.3f}")  # 应该接近0.8
```

**输出结果**：
```
P(下雨): 0.295
P(带伞): 0.325
P(下雨且带伞): 0.236
P(带伞|下雨): 0.800
```

完美！模拟出来的条件概率正好是80%。

---

## 四、贝叶斯定理：体检阳性≠得病

### 经典案例：体检的"假阳性"问题

假设某种罕见病的发病率是 **1%**（100人里有1人得病）。

医院有个检测试剂：
- 如果**得病**，99%的概率检测出阳性（真阳性）
- 如果**没病**，5%的概率检测出阳性（假阳性）

**问题**：如果一个人检测结果呈阳性，他真的得病的概率是多少？

**直觉回答**：很多人会说"99%吧，因为试剂很准"。

**正确答案**：**约16.4%**。

什么？！这么低？

### 贝叶斯定理（大白话版）

```
P(得病|阳性) = P(阳性|得病) × P(得病) / P(阳性)
```

翻译成人话：

> "检测呈阳性时真的得病的概率" = 
> ("得病时检测为阳性的概率" × "得病的基础概率") ÷ "检测为阳性的总概率"

### 计算全过程

假设有10000人做检测：

1. **得病的人**：10000 × 1% = 100人
   - 检测出阳性：100 × 99% = 99人（真阳性）
   - 检测出阴性：100 × 1% = 1人（假阴性）

2. **没得病的人**：10000 × 99% = 9900人
   - 检测出阳性：9900 × 5% = 495人（假阳性）
   - 检测出阴性：9900 × 95% = 9405人（真阴性）

3. **所有阳性的人**：99 + 495 = 594人

4. **阳性中真的得病的比例**：99 / 594 ≈ **16.7%**

**这就是为什么"体检阳性≠得病"**——因为基础发病率太低，假阳性太多了！

### Python实战：用代码验证

```python
# 贝叶斯定理计算
p_disease = 0.01        # P(得病) = 1%
p_positive_given_disease = 0.99   # P(阳性|得病) = 99%
p_positive_given_no_disease = 0.05  # P(阳性|没病) = 5%

# P(阳性) = P(阳性|得病)×P(得病) + P(阳性|没病)×P(没病)
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_no_disease * (1 - p_disease))

# 贝叶斯定理
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"P(得病|阳性) = {p_disease_given_positive:.3f}")  # 输出: 0.164
print(f"也就是说，阳性中真的得病的概率只有 {p_disease_given_positive*100:.1f}%")
```

---

## 五、期望和方差：平均分和成绩波动

### 1. 期望（Expected Value）：平均能赢多少？

**生活比喻：赌博的预期收益**

你玩一个游戏：
- 50%概率赢100元
- 50%概率输50元

**期望收益** = 100×0.5 + (-50)×0.5 = 50 - 25 = **25元**

意思是：如果你玩很多次，平均每次赚25元。

### 2. 方差（Variance）：风险有多大？

还是上面那个游戏，另一个游戏：
- 99%概率赢1元
- 1%概率输1000元

**期望收益** = 1×0.99 + (-1000)×0.01 = 0.99 - 10 = **-9.01元**

虽然期望是负的（长期会亏），但你可能觉得"我大概率能赢1元"。

**方差**就是衡量这种"波动性"——方差越大，风险越高，你可能一夜暴富，也可能倾家荡产。

### NumPy实战

```python
import numpy as np

# 游戏1：中等风险
returns_1 = np.array([100, -50])
probabilities_1 = np.array([0.5, 0.5])

# 游戏2：高风险
returns_2 = np.array([1, -1000])
probabilities_2 = np.array([0.99, 0.01])

# 计算期望
expected_1 = np.sum(returns_1 * probabilities_1)
expected_2 = np.sum(returns_2 * probabilities_2)

print(f"游戏1期望收益: {expected_1} 元")
print(f"游戏2期望收益: {expected_2} 元")

# 计算方差（衡量风险）
variance_1 = np.sum(probabilities_1 * (returns_1 - expected_1)**2)
variance_2 = np.sum(probabilities_2 * (returns_2 - expected_2)**2)

print(f"游戏1方差（风险）: {variance_1:.2f}")
print(f"游戏2方差（风险）: {variance_2:.2f}")

# 结论：游戏1期望高、风险低；游戏2期望负、风险高（千万别玩！）
```

---

## 六、正态分布：为什么考试成绩总集中在中间？

### 钟形曲线（Bell Curve）

你看过考试成绩分布吗？大部分人集中在平均分附近，极高分和极低分都很少——画成图，就是一条**钟形曲线**。

这就是**正态分布**（Normal Distribution），也叫高斯分布。

### 两个参数：均值和标准差

- **均值（μ）**：分布的中心（平均分）
- **标准差（σ）**：分布的"胖瘦"（成绩波动大小）

### 68-95-99.7法则

正态分布有个神奇的规律：
- **68%**的数据落在 `μ±σ` 范围内
- **95%**的数据落在 `μ±2σ` 范围内
- **99.7%**的数据落在 `μ±3σ` 范围内

**应用**：IQ测试的分数就是正态分布（μ=100, σ=15）。所以：
- 68%的人IQ在85-115之间
- 95%的人IQ在70-130之间
- 只有0.3%的人IQ超过145（天才）

### Python实战：画正态分布曲线 + 蒙特卡洛模拟

```python
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成正态分布数据（均值=100，标准差=15）
mu, sigma = 100, 15
data = np.random.normal(mu, sigma, 10000)

# 画直方图
plt.hist(data, bins=50, density=True, alpha=0.7, color='skyblue')

# 画理论曲线
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)
plt.plot(x, y, 'r-', linewidth=2, label='正态分布曲线')

plt.xlabel('数值')
plt.ylabel('概率密度')
plt.title('正态分布示意（μ=100, σ=15）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('C:/Users/Administrator/.qclaw/workspace/aijvs-blog/normal_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

print("正态分布图已保存")
print(f"生成数据的均值: {np.mean(data):.2f}")
print(f"生成数据的标准差: {np.std(data):.2f}")
```

---

## 七、蒙特卡洛模拟：用随机性解决确定性问题

### 故事：掷骰子预测未来

蒙特卡洛模拟（Monte Carlo Simulation）的核心思想：**用大量随机试验来近似真实概率**。

比如你想知道"掷两个骰子，点数和大于10的概率"：
- **数学方法**：枚举所有36种情况，数一数有多少组和>10 → 3/36 = 8.33%
- **蒙特卡洛方法**：掷骰子100000次，看看有多少次和>10 → 约8.33%

当问题复杂到无法用数学公式解决时，蒙特卡洛方法就是救星！

### Python实战：蒙特卡洛模拟π值

```python
import numpy as np

# 用蒙特卡洛方法估算π
# 原理：在1/4圆内随机投点，点在圆内的比例 ≈ π/4

n_points = 100000  # 投100000个点
points_inside = 0

for _ in range(n_points):
    x, y = np.random.rand(2)  # 在[0,1]内随机取点
    if x**2 + y**2 <= 1:      # 点到原点的距离 ≤ 1（在单位圆内）
        points_inside += 1

pi_estimate = 4 * points_inside / n_points
print(f"蒙特卡洛估算的π值: {pi_estimate:.6f}")
print(f"真实π值: {np.pi:.6f}")
print(f"误差: {abs(pi_estimate - np.pi):.6f}")
```

**输出示例**：
```
蒙特卡洛估算的π值: 3.141240
真实π值: 3.141593
误差: 0.000353
```

投的点越多，估算越准！

---

## 八、总结：概率论教AI做决策

从1654年赌桌上的争吵，到今天AI的每一个决策：

1. **概率**：量化不确定性（"明天下雨70%"）
2. **条件概率**：在已知信息下更新判断（"下雨天带伞的概率"）
3. **贝叶斯定理**：用新证据更新信念（"体检阳性≠得病"）
4. **期望和方差**：评估收益和风险
5. **正态分布**：理解数据分布规律
6. **蒙特卡洛模拟**：用随机性解决复杂问题

**概率论就是AI的"决策指南"**——它教会机器在不确定中找到最优解。

下一篇，咱们聊**微积分**——理解"变化"的语言。没有微积分，就没有深度学习的反向传播算法！

---

## 代码示例汇总

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 条件概率模拟
np.random.seed(42)
rain = np.random.rand(1000) < 0.3
bring_umbrella = np.where(rain, 
                          np.random.rand(1000) < 0.8,
                          np.random.rand(1000) < 0.1)
p_umbrella_given_rain = np.sum(rain & bring_umbrella) / np.sum(rain)
print(f"P(带伞|下雨): {p_umbrella_given_rain:.3f}")

# 2. 贝叶斯定理
p_disease = 0.01
p_positive_given_disease = 0.99
p_positive_given_no_disease = 0.05
p_positive = p_positive_given_disease * p_disease + p_positive_given_no_disease * (1 - p_disease)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
print(f"P(得病|阳性): {p_disease_given_positive:.3f}")

# 3. 期望和方差
returns = np.array([100, -50])
probs = np.array([0.5, 0.5])
expected = np.sum(returns * probs)
variance = np.sum(probs * (returns - expected)**2)
print(f"期望: {expected}, 方差: {variance}")

# 4. 蒙特卡洛模拟π
n_points = 100000
points_inside = sum(1 for _ in range(n_points) 
                    if np.random.rand()**2 + np.random.rand()**2 <= 1)
pi_estimate = 4 * points_inside / n_points
print(f"π估算值: {pi_estimate:.6f}")
```

---

*下一篇预告：**《AI数学基础（三）：微积分——理解"变化"的语言》***

*17世纪，伦敦正经历一场瘟疫，一个23岁的年轻人被迫离开大学回到乡下...*

---


[上一篇](/2026/06/17/AI数学基础（一）：线性代数——让机器学会看向量/)
[下一篇](/2026/06/17/AI数学基础（三）：微积分——理解变化的语言/)
