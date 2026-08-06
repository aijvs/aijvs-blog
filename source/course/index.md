---
title: 深度学习与 AI 实战教程 | AIJVS 课程中心
date: 2026-07-29 09:15:00
layout: page
comments: true
---

<style>
.hero-cta {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff !important;
  padding: 12px 28px;
  border-radius: 8px;
  display: inline-block;
  font-weight: 600;
  margin: 0.5rem 0.5rem 0.5rem 0;
  text-decoration: none !important;
  transition: transform .2s, box-shadow .2s;
}
.hero-cta:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(102,126,234,0.4); }
.hero-cta-secondary {
  background: #fff;
  color: #667eea !important;
  border: 2px solid #667eea;
  padding: 10px 26px;
  border-radius: 8px;
  display: inline-block;
  font-weight: 600;
  margin: 0.5rem 0.5rem 0.5rem 0;
  text-decoration: none !important;
  transition: transform .2s;
}
.hero-cta-secondary:hover { transform: translateY(-2px); }

.lesson-card {
  border: 1px solid #eee;
  border-radius: 10px;
  padding: 1rem 1.5rem;
  margin: 0.8rem 0;
  transition: box-shadow .2s;
}
.lesson-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
.lesson-card h4 { margin-top: 0; margin-bottom: 0.3rem; }
.lesson-card .tag {
  display: inline-block;
  background: #667eea15;
  color: #667eea;
  font-size: 0.8rem;
  padding: 2px 8px;
  border-radius: 4px;
  margin-right: 4px;
}
.lesson-card .summary-qa {
  color: #666;
  font-size: 0.85rem;
  margin: 0.4rem 0 0 0;
  padding-left: 0.8rem;
  border-left: 2px solid #e5e7eb;
}

.stage-card {
  border-left: 4px solid #667eea;
  padding: 0.8rem 1.2rem;
  margin: 0.6rem 0;
  background: #f8f9ff;
  border-radius: 0 8px 8px 0;
}
.stage-card h4 { margin: 0 0 0.3rem 0; }
.stage-card ul { margin: 0.3rem 0 0 1.2rem; padding: 0; }
.stage-card li { margin: 0.15rem 0; }

.attribution-box {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 1rem 1.5rem;
  margin: 1.5rem 0;
  background: #fafbfc;
  font-size: 0.9rem;
  line-height: 1.6;
}
</style>

# 深度学习与 AI 实战教程

<a href="#课程特色">🎓 课程特色</a> · <a href="#快速通道">🚀 快速通道</a> · <a href="#系统学习">📖 系统学习</a> · <a href="#学习建议">💡 学习建议</a>

---

<div style="text-align:center; margin: 1.5rem 0; padding: 1.5rem; background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%); border-radius: 12px;">
<p style="font-size:1.1rem; max-width:600px; margin:0 auto 1rem auto;">
<b>先做一个能用的东西，再搞懂它为什么 work。</b>
<br>这是本课程的教学哲学——借鉴自 <a href="https://course.fast.ai/" target="_blank">fast.ai</a> 创始人 Jeremy Howard 的 Top-down 教学法。
</p>
<div>
  <a href="/course/#快速通道" class="hero-cta">🚀 开始快速通道 →</a>
  <a href="/course/#系统学习" class="hero-cta-secondary">📖 从零系统学</a>
</div>
</div>

---

## 🎓 课程特色

### 本课程与其他教程有什么不同？

**传统教学模式（Bottom-up）：** 先学 Python → 再学数学 → 再学算法 → 第 30 个小时终于能跑一个模型。大多数人在这之前就放弃了。

**快速实践模式（Top-down）：** 第 1 课就训练并部署一个真实模型。不需要数学博士，不需要昂贵 GPU，不需要读完一本书。你先看到"这个东西真能做出来"，再有动力去啃背后的原理。

> 这个教学理念来自 **Jeremy Howard 和 Rachel Thomas 创立的 [fast.ai](https://www.fast.ai/)**（"Practical Deep Learning for Coders"）。fast.ai 的课程已被全球超过 600 万次观看，毕业生进入了 Google Brain、OpenAI、Tesla 等团队。本课程在借鉴其 Top-down 教学法的同时，全部适配中文场景，并使用 PyTorch 生态的中文主流框架和工具链。

### 不需要什么

| 以为需要的 | 实际不需要 |
|-----------|-----------|
| 大量的数学知识 | 高中水平足够了。遇到新数学概念，我们边用边学 |
| 海量数据集 | 用 50 张图片也能训练出可用的分类器 |
| 昂贵的 GPU | Google Colab / Kaggle Notebooks 免费 GPU 足够跑完所有项目 |
| 深度学习硕士学位 | 课程从头设计，唯一的前提是你会写一点 Python |

---

## 🚀 快速通道

**6 课，从实用出发。** 前 3 课动手做东西，后 3 课理解原理并深入真实世界。

### 第1课：训练并部署你的第一个图像分类器

<span class="tag">动手</span><span class="tag">CV</span><span class="tag">部署</span>

<details>
<summary>📝 课程概要</summary>
<p>用预训练模型让机器学会区分不同类别的图片——整个过程不到 50 行代码。然后把它部署成在线应用，生成一个公开链接。你会看到 2015 年还是科研前沿的技术，现在几行代码就能搞定。</p>

<p><b>本课将学会：</b></p>
<ul>
<li>如何加载预训练模型</li>
<li>什么是 fine-tune（微调），为什么它这么强大</li>
<li>训练完成后如何保存和导出模型</li>
<li>用 Hugging Face Spaces 将模型部署为 Web 应用</li>
</ul>

<p><b>学完后你能回答：</b></p>
<ul>
<li>深度学习模型训练的基本流程是什么？</li>
<li>迁移学习为什么让训练变得很快？</li>
<li>部署一个模型到云端需要几步？</li>
</ul>
</details>

<p><b>📖 核心文章：</b><a href="/2026/07/11/实战：图像分类（CIFAR-10）/">实战：图像分类（CIFAR-10）</a></p>
<p><b>📖 参考：</b><a href="/2026/07/03/计算机视觉入门：图像处理基础/">计算机视觉入门</a></p>
<p class="summary-qa">🔍 回顾自查：我能用预训练模型分类自己的图片吗？我知道模型是怎么导出的吗？我知道怎么让模型在浏览器里跑起来吗？</p>

---

### 第2课：用三代方案做一个情感分析系统

<span class="tag">动手</span><span class="tag">NLP</span><span class="tag">对比</span>

<details>
<summary>📝 课程概要</summary>
<p>输入一段影评判断好评还是差评。你亲手跑三个时代的方案：词袋模型（~68%准确率）→ LSTM（~86%）→ BERT（~93%）——亲眼看到技术进步带来的差距。</p>

<p><b>本课将学会：</b></p>
<ul>
<li>文本数据如何变成计算机能处理的数字</li>
<li>词袋模型为什么丢失词序信息</li>
<li>LSTM 如何记住上下文</li>
<li>预训练语言模型（BERT）为什么是目前的最优方案</li>
</ul>

<p><b>学完后你能回答：</b></p>
<ul>
<li>文本分类的完整 Pipeline 是怎样的？</li>
<li>"fine-tune"和"train from scratch"的区别？</li>
<li>什么样的任务用简单模型就够了？</li>
</ul>
</details>

<p><b>📖 核心文章：</b><a href="/2026/07/29/实战：情感分析系统/">实战：情感分析系统（BoW→LSTM→BERT）</a></p>
<p><b>📖 参考：</b><a href="/2026/07/03/NLP入门：文本预处理与词向量/">NLP 入门：文本预处理与词向量</a></p>
<p class="summary-qa">🔍 回顾自查：我能用代码把一段文本转成词向量吗？我知道什么时候该用简单模型什么时候该用 BERT 吗？</p>

---

### 第3课：实时目标检测——让你的摄像头认出东西

<span class="tag">动手</span><span class="tag">CV</span><span class="tag">部署</span>

<details>
<summary>📝 课程概要</summary>
<p>YOLOv8——目前最快最准的目标检测框架之一。一行代码下载预训练模型，三行代码开始摄像头实时检测。然后训练你自己的数据集（比如检测游戏画面、监控场景），导出到手机或边缘设备。</p>

<p><b>本课将学会：</b></p>
<ul>
<li>目标检测和图像分类的区别</li>
<li>YOLO 模型的"只看一次"思想</li>
<li>如何标注自己的数据集</li>
<li>模型导出（ONNX / TensorRT / NCNN）和跨平台部署</li>
</ul>

<p><b>学完后你能回答：</b></p>
<ul>
<li>目标检测模型是怎么同时完成"定位"和"分类"的？</li>
<li>YOLO 相比两阶段检测器（如 Faster R-CNN）有什么优缺点？</li>
<li>如何把训练好的模型部署到手机或树莓派？</li>
</ul>
</details>

<p><b>📖 核心文章：</b><a href="/2026/07/29/实战：目标检测（YOLOv8）/">实战：目标检测（YOLOv8）</a></p>
<p class="summary-qa">🔍 回顾自查：我能训练 YOLOv8 识别我自己的物体吗？我知道量化模型和全精度模型有什么区别吗？</p>

---

### 第4课：理解神经网络在学什么

<span class="tag">原理</span><span class="tag">CV</span><span class="tag">NLP</span>

<details>
<summary>📝 课程概要</summary>
<p>前 3 课你已经做出了模型。现在该拆开看看里面是什么了。CNN 的卷积核在学什么？为什么它对猫的耳朵和眼睛有独立的神经元？RNN 是怎么"记住"前文的？我们用可视化工具让你亲眼看到神经网络的内部运作。</p>

<p><b>本课将学会：</b></p>
<ul>
<li>CNN 的卷积层和池化层在做什么</li>
<li>RNN 和 LSTM 如何解决长期依赖问题</li>
<li>Grad-CAM：让模型告诉你它"看"了图片的哪个部分</li>
<li>混淆矩阵：模型在哪些类别上犯错了</li>
</ul>

<p><b>学完后你能回答：</b></p>
<ul>
<li>CNN 中的卷积核数量和感受野有什么关系？</li>
<li>LSTM 的门控机制是解决了什么具体问题？</li>
<li>我怎么判断模型是"猜对了"还是"真的学到了"？</li>
</ul>
</details>

<p><b>📖 核心文章：</b><a href="/2026/07/03/机器学习入门：概念与分类全解/">机器学习入门</a> · <a href="/2026/07/03/神经网络架构详解：CNN卷积神经网络/">CNN 详解</a> · <a href="/2026/07/03/神经网络架构详解：RNN与LSTM/">RNN/LSTM 详解</a></p>
<p class="summary-qa">🔍 回顾自查：我能解释卷积层对图像做了什么操作吗？我知道 LSTM 和普通 RNN 的关键区别吗？</p>

---

### 第5课：参加你的第一个 Kaggle 竞赛

<span class="tag">动手</span><span class="tag">竞赛</span><span class="tag">数据科学</span>

<details>
<summary>📝 课程概要</summary>
<p>Kaggle 是数据科学界的 GitHub。Titanic 生存预测是 Kaggle 最经典的入门竞赛。你从数据探索（EDA）开始，做特征工程，尝试多种模型（逻辑回归、随机森林、XGBoost），最后用投票集成和 Stacking 把多个模型的预测结合起来获得高分。这套管线适用于任何表格数据竞赛。</p>

<p><b>本课将学会：</b></p>
<ul>
<li>数据探索（EDA）的基本流程</li>
<li>缺失值处理、特征编码、特征组合</li>
<li>随机森林和 XGBoost 的理解与调参</li>
<li>模型集成：投票、平均、Stacking</li>
</ul>

<p><b>学完后你能回答：</b></p>
<ul>
<li>EDA 阶段我该看哪些关键指标？</li>
<li>特征工程对模型提升有多大？</li>
<li>集成学习为什么通常比单个模型好？</li>
</ul>
</details>

<p><b>📖 核心文章：</b><a href="/2026/07/29/Kaggle竞赛入门：Titanic生存预测/">Kaggle 竞赛入门：Titanic 生存预测</a></p>
<p class="summary-qa">🔍 回顾自查：我能独立完成一场 Kaggle 竞赛的完整流程吗？我知道怎么设计和比较特征吗？</p>

---

### 第6课：开源贡献——你的 GitHub 就是最好的简历

<span class="tag">职业</span><span class="tag">社区</span><span class="tag">Git</span>

<details>
<summary>📝 课程概要</summary>
<p>你学完了教程，跑完了代码——然后呢？开源社区是这个行业最强大的学习和求职网络。本课带你走完完整的 PR 流程：找 good first issue → Fork → 修改 → 提 PR → 被 Review → 合并。你会获得 GitHub 贡献记录——这在简历上比很多证书都管用。</p>

<p><b>本课将学会：</b></p>
<ul>
<li>如何在 GitHub 上找到适合新手的 Issue</li>
<li>Fork & PR 的完整 Git 工作流</li>
<li>如何编写高质量的 PR 描述</li>
<li>从文档修复到代码贡献的进阶路线</li>
</ul>

<p><b>学完后你能回答：</b></p>
<ul>
<li>我今天就能给哪个开源项目提 PR？</li>
<li>PR 被 reviewer 要求修改时怎么办？</li>
<li>我的 GitHub 贡献记录怎么写在简历上？</li>
</ul>
</details>

<p><b>📖 核心文章：</b><a href="/2026/07/29/开源项目贡献指南：如何参与AI开源社区/">开源项目贡献指南</a></p>
<p class="summary-qa">🔍 回顾自查：我提过至少一个 PR 了吗？我知道什么样的问题适合新手去贡献吗？</p>

---

## 📖 系统学习

按阶段推进的完整学习路径。建议搭配快速通道第 1–3 课同步进行——先有"我要解决什么问题"的目标感，学基础会快很多。

<div class="stage-card" style="border-left-color: #22c55e;">
<h4>阶段 0：打地基 <span style="color:#22c55e;font-weight:400;">（8 篇 · 预计 10~15 小时）</span></h4>
<p>Python 编程基础 + 三大数学工具：线性代数、概率论、微积分。不需要精通——先理解"它为什么有用"，后面的实战会帮你巩固。</p>
<ul>
<li>☐ <a href="/2026/06/17/Python编程基础（一）：从零搭建你的AI开发环境/">Python 环境搭建</a> — 装好 Python、Jupyter、PyTorch</li>
<li>☐ <a href="/2026/06/17/Python编程基础（二）：变量、数据类型和基本运算/">Python 基础：变量与数据类型</a></li>
<li>☐ <a href="/2026/06/17/Python编程基础（三）：NumPy——AI工程师的第一件武器/">NumPy：AI 工程师的第一件武器</a> — 所有 AI 框架的底层基石</li>
<li>☐ <a href="/2026/07/03/AI数学基础（一）：线性代数——让机器学会看向量/">线性代数：向量与矩阵</a> — 数据在计算机里的存在形式</li>
<li>☐ <a href="/2026/07/03/AI数学基础（二）：概率论——教机器做不确定的决策/">概率论：不确定性下的决策</a> — 模型为什么永远不会"100%确定"</li>
<li>☐ <a href="/2026/07/03/AI数学基础（三）：微积分——理解变化的语言/">微积分与导数</a> — 梯度下降到底在做什么</li>
<li>☐ <a href="/2026/07/03/深度学习入门指南：从零开始理解神经网络/">深度学习入门</a> — 神经网络全景概览</li>
<li>☐ <a href="/2026/07/03/PyTorch-vs-TensorFlow：2026年该选哪个框架/">PyTorch vs TensorFlow</a> — 框架选型对比</li>
</ul>
</div>

<div class="stage-card" style="border-left-color: #3b82f6;">
<h4>阶段 1：核心架构 <span style="color:#3b82f6;font-weight:400;">（7 篇 · 预计 10~15 小时）</span></h4>
<p>机器学习的核心概念 + 三大主流架构：全连接网络、CNN、RNN/LSTM。学完你就有能力看懂绝大多数 AI 应用的原理了。</p>
<ul>
<li>☐ <a href="/2026/07/03/机器学习入门：概念与分类全解/">机器学习核心概念</a> — 监督/无监督/半监督、过拟合、验证集</li>
<li>☐ <a href="/2026/07/03/PyTorch实战（一）：张量操作与自动微分/">PyTorch：张量与自动微分</a> — 框架最核心的两个抽象</li>
<li>☐ <a href="/2026/07/03/PyTorch实战（二）：构建第一个全连接网络/">PyTorch：第一个全连接网络</a></li>
<li>☐ <a href="/2026/07/03/神经网络架构详解：CNN卷积神经网络/">CNN 卷积神经网络</a> — 计算机视觉的主力架构</li>
<li>☐ <a href="/2026/07/03/神经网络架构详解：RNN与LSTM/">RNN 与 LSTM</a> — 序列数据处理的核心模型</li>
<li>☐ <a href="/2026/07/03/计算机视觉入门：图像处理基础/">计算机视觉基础</a> — 数据增强、图像变换、预处理</li>
<li>☐ <a href="/2026/07/03/NLP入门：文本预处理与词向量/">NLP 基础</a> — 向量化、分词、词嵌入</li>
</ul>
</div>

<div class="stage-card" style="border-left-color: #a855f7;">
<h4>阶段 2：实战项目 <span style="color:#a855f7;font-weight:400;">（6 篇 · 预计 15~20 小时）</span></h4>
<p>每个项目都是一个独立可运行的完整应用。建议在阶段 1 学到三分之二的时候就开始穿插做项目。</p>
<ul>
<li>☐ <a href="/2026/07/11/实战：手写数字识别（MNIST）/">手写数字识别（MNIST）</a> — 深度学习的 Hello World</li>
<li>☐ <a href="/2026/07/11/实战：图像分类（CIFAR-10）/">图像分类（CIFAR-10）</a> — 从预训练模型开始迁移学习</li>
<li>☐ <a href="/2026/07/29/实战：情感分析系统/">情感分析系统</a> — BoW→LSTM→BERT 三代方案实测对比</li>
<li>☐ <a href="/2026/07/29/实战：目标检测（YOLOv8）/">目标检测（YOLOv8）</a> — 实时检测 + 自定义数据集训练</li>
<li>☐ <a href="/2026/07/29/Kaggle竞赛入门：Titanic生存预测/">Kaggle Titanic 竞赛</a> — 数据探索→特征工程→模型集成完整管线</li>
<li>☐ <a href="/2026/07/29/开源项目贡献指南：如何参与AI开源社区/">开源贡献指南</a> — 从 Issue 到 PR 的完整流程</li>
</ul>
</div>

<div class="stage-card" style="border-left-color: #f59e0b;">
<h4>阶段 3：前沿技术 <span style="color:#f59e0b;font-weight:400;">（即将推出）</span></h4>
<p>从动手实践进入前沿领域：Transformer 架构、LLM 部署与微调、浏览器端推理等。</p>
<ul>
<li>⏳ Transformer 架构从零理解</li>
<li>⏳ LLM 部署指南：Ollama + vLLM + 量化</li>
<li>⏳ 用 LoRA 微调你的第一个模型</li>
<li>⏳ WebGPU：在浏览器里跑 AI 推理</li>
<li>⏳ AI 双周速递 & 论文解读</li>
</ul>
</div>

---

## 💡 学习建议

### 1. 融入 Top-down 教学法

不要试图"学完所有基础再动手"。先走快速通道第 1 课——30 分钟内你就会有一个能识别图片的模型在线上运行。<a href="https://aijvs.com/">看看别人用这个课程做出的东西</a>，你会更有动力。

### 2. 不要只看，要跑

读十遍不如跑一遍。每篇文章都有完整可运行的代码。设置好环境，打开 Jupyter Notebook 或者 Colab，一个字一个字打一遍。遇到错误就是最好的学习时刻——你会在修 bug 中学到最多东西。

### 3. 遇到问题怎么办？

- 每篇文章底部有评论区（Giscus），直接留言
- 先看评论区——大概率已经有人问过同样的问题
- 搜索引擎搜报错信息——90% 的问题前人已经遇到过并在 Stack Overflow / GitHub Issues 上有答案
- 在<a href="https://github.com/aijvs/aijvs-blog/discussions" target="_blank">GitHub Discussions</a>中提问

### 4. 每周坚持

每天 30 分钟 > 周末突击 5 小时。初学者最难的不是理解某个概念——而是两周不碰之后重新拾起的心理负担。每天动手写几行代码的习惯，比任何教材都重要。

---

## 致谢

本课程的教学理念借鉴自 **[Jeremy Howard](https://jeremy.fast.ai/) 和 [Rachel Thomas](https://rachel.fast.ai/) 创立的 [fast.ai](https://www.fast.ai/)** 的 "Practical Deep Learning for Coders" 课程（<a href="https://course.fast.ai/" target="_blank">course.fast.ai</a>）。

fast.ai 的教学理念是**Top-down（自上而下）**——从完整可用的解决方案开始，逐步深入到底层原理。这与传统大学课程"先学基础再学应用"的 Bottom-up 教学顺序相反。正如 Jeremy Howard 所说：

> "Nearly all technical subjects at university are taught 'bottom up': start with basic foundations, and gradually work up to complete useful solutions to real world problems. But we go 'top down': start with complete useful solutions to real world problems, and gradually work down to the basic foundations. Education experts recommend this approach for more effective learning."

fast.ai 课程已被超过 600 万次观看，其配套教材《Deep Learning for Coders with fastai and PyTorch》由 O'Reilly 出版，可在 <a href="https://github.com/fastai/fastbook" target="_blank">GitHub 免费阅读</a>。

本课程是对 fast.ai 教学理念的中文实践——在保留 Top-down 教学精髓的同时，全部内容适配中文开发者场景，使用 PyTorch 生态的主流工具链。

<div class="attribution-box">
<b>关于版权与许可：</b> 截至 2026 年 7 月，本课程所引用的 fast.ai 课程（course.fast.ai）采用 <a href="https://github.com/fastai/course22/blob/master/LICENSE" target="_blank">Apache 2.0 许可</a>（或类似开放许可）。本课程的学习结构借鉴 fast.ai 的 Top-down 教学法，所有具体的技术讲解、代码示例和文章内容均为 AIJVS 原创，以 CC-BY 4.0 许可发布。如有任何版权问题，请联系 <a href="mailto:aijvscom@gmail.com">aijvscom@gmail.com</a>。
</div>

---

**准备好了？从<a href="/2026/07/11/实战：图像分类（CIFAR-10）/">快速通道第 1 课</a>开始——5 分钟训练你的第一个模型。**
