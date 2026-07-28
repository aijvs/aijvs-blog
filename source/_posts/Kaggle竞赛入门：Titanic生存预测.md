---
title: Kaggle 竞赛入门：Titanic 生存预测
date: 2026-07-29 09:00:00
categories: [竞赛实战]
tags: [Kaggle, 竞赛, 数据科学, 特征工程, scikit-learn, 入门教程]
description: 手把手完成你的第一个 Kaggle 竞赛项目：数据探索、特征工程、模型集成、提交评分，从 0.77 到 0.82 的完整优化路径。
---

## 引子

Kaggle 是数据科学界的"新东方"——刷名号、刷经验、刷 offer 的地方。

Titanic 是 Kaggle 的"Hello World"。2012 年开赛到今天，超过 3 万份参赛作品——新人在这里学到的不是调参，而是**一套完整的竞赛管线**。

简单的问题：给你 891 个乘客的信息（年龄、性别、船票等级等），预测谁能在海难中幸存。

但就这个看似简单的问题，涵盖了竞赛的 90% 的通用技能。

---

## 前置知识

- [机器学习入门：概念与分类全解](/2026/07/03/机器学习入门：概念与分类全解/)
- [Python 编程基础（三）：NumPy 快速入门](/2026/06/17/Python编程基础（三）：NumPy——AI工程师的第一件武器/)
- Pandas 基础（知道 DataFrame 怎么读写就够了）

---

## 一、加载数据

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print(train.shape)   # (891, 12)
print(train.columns)
```

看看前几行：
```
  PassengerId  Survived  Pclass  Name      Sex   Age  SibSp  Parch  Ticket  Fare Cabin Embarked
0           1         0       3  Braund..  male   22     1      0  A/5..   7.25  NaN         S
1           2         1       1  Cumings.. female  38     1      0  PC..   71.3  C85         C
2           3         1       3  Heikkin.. female  26     0      0  ST..   7.92  NaN         S
```

字段含义：

| 字段 | 含义 | 类型 |
|------|------|------|
| Survived | 是否幸存（1=是） | **目标变量** |
| Pclass | 船舱等级（1/2/3） | 类别型 |
| Sex | 性别 | 类别型 |
| Age | 年龄 | 数值型（**有缺失**） |
| SibSp | 兄弟姐妹/配偶数 | 数值型 |
| Parch | 父母/子女数 | 数值型 |
| Fare | 票价 | 数值型 |
| Embarked | 登船港口（C/Q/S） | 类别型（**有缺失**） |
| Cabin | 舱房号 | 类别型（**大量缺失**） |
| Name, Ticket | 名字和票号 | 高基数文本（需要特征提取） |

---

## 二、数据探索（EDA）

### 2.1 缺失值

```python
train.isnull().sum()
```

```
Age         177
Cabin       687
Embarked      2
```

- Cabin 缺了 77%——直接判定：**弃用 Cabin 列**。缺太多补了也是噪声
- Age 缺 177 条（20%）——需要填充
- Embarked 缺 2 条（0.2%）——填充众数

### 2.2 幸存率分布

```python
print(train['Survived'].value_counts(normalize=True))
```

```
0    0.616
1    0.384
```

大约 62% 的人遇难，38% 幸存。

### 2.3 核心变量分析

**性别：**

```python
train.groupby('Sex')['Survived'].mean()
```

```
female    0.742  ← 74% 的女性幸存
male      0.189  ← 只有 19% 的男性幸存
```

**等级：**

```python
train.groupby('Pclass')['Survived'].mean()
```

```
Pclass 1: 0.630  ← 头等舱
Pclass 2: 0.473
Pclass 3: 0.242  ← 三等舱
```

**年龄分布：**

```python
# 儿童幸存率远高于成人
train[train['Age'] < 12]['Survived'].mean()   # ~0.60
train[train['Age'] > 60]['Survived'].mean()   # ~0.27
```

**三个最强特征：Sex > Pclass > Age。**

---

## 三、特征工程

### 3.1 年龄填充

```python
# 按性别+等级分组，用中位数填充年龄
train['Age'] = train.groupby(['Sex', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median()))

# 分箱：把年龄分成 5 段
train['AgeBin'] = pd.cut(train['Age'],
    bins=[0, 12, 30, 50, 80],
    labels=['Child', 'Young', 'Middle', 'Elderly'])
```

为什么分箱？模型对"32 岁和 33 岁之间的差异"不感兴趣，但对"儿童 vs 成年人"的巨大差异感兴趣。

### 3.2 家庭人数

```python
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1  # +1 是自己

# 单人和超大家庭幸存率低，2-4 人家庭幸存率高
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
```

### 3.3 名字中提取称谓

```python
# Mr/Mrs/Miss/Dr/Reverend/...
train['Title'] = train['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)

# 合并稀有称谓
rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr',
               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
train['Title'] = train['Title'].replace(rare_titles, 'Rare')
train['Title'] = train['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
```

Title 是一个被低估的强特征——"Master"（小男孩）幸存率 >50%，"Mr"（成年男性）幸存率 ~15%。

### 3.4 票价分箱

```python
train['FareBin'] = pd.qcut(train['Fare'], 4,
    labels=['Low', 'Medium', 'High', 'VeryHigh'])
```

### 3.5 编码

```python
from sklearn.preprocessing import LabelEncoder

for col in ['Sex', 'Title', 'AgeBin', 'FareBin', 'Embarked']:
    train[col] = LabelEncoder().fit_transform(train[col].astype(str))
```

---

## 四、模型选择

Titanic 是小型表格数据集（几百条，十来个特征），不要上来就上 XGBoost——先试简单的。

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
            'FamilySize', 'IsAlone', 'Title', 'AgeBin', 'FareBin']
X = train[features]
y = train['Survived']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"{name:25s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

预期输出：
```
Logistic Regression       : 0.8214 (+/- 0.0256)
Random Forest             : 0.8283 (+/- 0.0187)
Gradient Boosting         : 0.8310 (+/- 0.0211)
```

三个模型都在 0.82-0.83 之间。差距不大——说明**特征是瓶颈**，不是模型。

---

## 五、集成与提分

### 5.1 投票集成

```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier([
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
], voting='soft')

scores = cross_val_score(voting, X, y, cv=kfold, scoring='accuracy')
print(f"Voting Ensemble: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

预期：**0.8347**——比最好的单模型涨了 0.3%。

### 5.2 Stacking

```python
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier([
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
], final_estimator=LogisticRegression(), cv=5)

scores = cross_val_score(stacking, X, y, cv=kfold, scoring='accuracy')
print(f"Stacking Ensemble: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

预期：**0.8372**——再涨 0.2-0.3%。

---

## 六、提交

```python
# 对测试集做同样的特征工程
# ...（同训练集的预处理流程，注意用训练集统计量填充）

predictions = voting.predict(test[features])

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
print(submission.head())
```

上传到 Kaggle：**Titanic → Submit Predictions → 上传 CSV**。

**预期公共榜分数：0.77 - 0.79。** 为什么比 CV 低？CV 是 5 折平均，Kaggle 的测试集是全量 unseen 的。

---

## 七、进阶优化（0.80+ 的来源）

下面这些技巧不是必须的，但它们把分数从 0.77 推到 0.80+：

### 7.1 交叉验证不一致检查

```python
# 如果 5 折 CV 的方差 > 0.03，说明特征不稳定
# 解决办法：检查是否有泄漏特征或数据分布不一致
```

### 7.2 极端值裁剪

```python
# Fare 有一个 512 的极端值
train['Fare'] = train['Fare'].clip(upper=200)
```

### 7.3 更细的 Cabin 特征

完全弃用 Cabin 可能太浪费。至少取出首字母：

```python
train['Deck'] = train['Cabin'].str[0]
# A/B/C/D/E 是高等级甲板 → 幸存率更高
```

### 7.4 XGBoost + 调参

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

Titanic 上 XGBoost 不一定比 Random Forest 好太多（数据太小），但训练 XGBoost 本身是一个重要的学习过程。

### 7.5 学习排名

| 公共榜分数 | 排名 | 对应的能力等级 |
|-----------|------|-------------|
| 0.78 | Top 50% | 基本管线完整 |
| 0.80 | Top 25% | 特征工程到位 |
| 0.81-0.82 | Top 10% | 集成 + 精细调参 |
| 0.83+ | Top 5% | 手工特征 + 领域知识 + 交叉验证调优 |

---

## 八、完整代码

```python
"""titanic.py - Kaggle Titanic 生存预测"""
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# 1. 加载
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
ids = test['PassengerId']

# 2. 合并预处理（统一处理 train + test）
all_data = pd.concat([train, test], keys=['train', 'test'], names=['source']).reset_index(level='source')

# 3. 特征工程
all_data['Age'] = all_data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
all_data['Embarked'] = all_data['Embarked'].fillna('S')
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)
all_data['Title'] = all_data['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
rare = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
all_data['Title'] = all_data['Title'].replace(rare, 'Rare')
all_data['Title'] = all_data['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})
all_data['AgeBin'] = pd.cut(all_data['Age'], bins=[0,12,30,50,80], labels=['Child','Young','Middle','Elderly'])
all_data['FareBin'] = pd.qcut(all_data['Fare'], 4, labels=['Low','Medium','High','VeryHigh'])

# 4. 编码
for col in ['Sex', 'Title', 'AgeBin', 'FareBin', 'Embarked']:
    all_data[col] = LabelEncoder().fit_transform(all_data[col].astype(str))

# 5. 拆分
features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title','AgeBin','FareBin']
X_train = all_data[all_data['source']=='train'][features]
y_train = train['Survived']
X_test  = all_data[all_data['source']=='test'][features]

# 6. 模型
model = VotingClassifier([
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
], voting='soft')
model.fit(X_train, y_train)

# 7. 预测 & 提交
preds = model.predict(X_test)
pd.DataFrame({'PassengerId': ids, 'Survived': preds}).to_csv('submission.csv', index=False)
print("submission.csv created ✅")
```

保存为 `titanic.py`，下载 Kaggle 数据集到同级目录，`python titanic.py` 即可得到可提交的 CSV。

---

## 总结

Titanic 教会你的不是 YOLO 或 Transformer——**它教的是数据科学的通用流程：**

- **EDA**：了解数据长什么样，缺失怎么办
- **特征工程**：从原始数据中榨取信息（Title 强于 Sex？）
- **交叉验证**：别光看训练精度，看泛化能力
- **集成**：三个臭皮匠顶一个诸葛亮

Kaggle 上，新手和高手之间差的就是特征工程和交叉验证之间的那层意识。

**下一步：**
- 挑战下一步竞赛：House Prices（同类型表格数据，特征工程更复杂）
- 或者跳到 Spaceship Titanic（同类型但数据无泄漏，需要更强特征工程）
- 学会用 Optuna 自动调参
