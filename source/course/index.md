---
title: AIJVS 课程中心
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
  padding: 1.2rem 1.5rem;
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
</style>

# AIJVS 课程中心

不用学完所有理论再动手。你可以在 **5 分钟内训练并部署第一个模型**，然后再回头搞懂它为什么 work。

<div style="text-align:center; margin: 2rem 0;">
  <a href="/course/#快速通道" class="hero-cta">🚀 开始快速通道 →</a>
  <a href="/course/#系统学习" class="hero-cta-secondary">📖 从零系统学</a>
</div>

---

## 两条路线

| | 🚀 快速通道 | 📖 系统学习 |
|---|---|---|
| 适合谁 | 你写过代码，想先看看 AI 能做啥 | 你是纯新手，或者喜欢一步步来 |
| 教学法 | **Top-down**: 先做出东西，再讲原理 | **Bottom-up**: 从 Python 和数学基础开始 |
| 第 1 课之后 | 你已经有一个能用的模型了 | 你熟悉了 Python 环境搭建 |
| 共几课 | 6 课 | 23+ 篇文章，4 个阶段 |
| 预估耗时 | 3~5 小时 | 40~60 小时 |

> **建议：** 走快速通道第 1~2 课找感觉，再切换到系统学习补基础。这是最有效率的路。

---

## 🚀 快速通道

从实用出发，每次动手做个东西。

<div class="lesson-card">
<h4>第1课：训练并部署你的第一个图像分类器</h4>
<p>用预训练模型，5 分钟让机器学会区分猫和狗。再花 5 分钟部署到线上，生成链接发朋友圈。</p>
<p><span class="tag">动手</span><span class="tag">图像分类</span></p>
<p>📖 <a href="/2026/07/11/实战：图像分类（CIFAR-10）/"><strong>核心文章</strong>：实战：图像分类（CIFAR-10）</a><br>
📖 参考：<a href="/2026/07/03/计算机视觉入门：图像处理基础/">计算机视觉入门</a></p>
</div>

<div class="lesson-card">
<h4>第2课：用三代方案做一个情感分析系统</h4>
<p>输入一段影评，判断是好评还是差评。你能亲眼看到：词袋模型（68%）、LSTM（86%）、BERT（93%）——三个时代的技术差距在哪。</p>
<p><span class="tag">动手</span><span class="tag">NLP</span></p>
<p>📖 <a href="/2026/07/29/实战：情感分析系统/"><strong>核心文章</strong>：实战：情感分析系统</a><br>
📖 参考：<a href="/2026/07/03/NLP入门：文本预处理与词向量/">NLP入门：文本预处理与词向量</a></p>
</div>

<div class="lesson-card">
<h4>第3课：实时目标检测——让你的摄像头认出东西</h4>
<p>YOLOv8 一行代码下载预训练模型，三行代码跑摄像头实时检测。训练自定义数据集，导出到手机/树莓派。</p>
<p><span class="tag">动手</span><span class="tag">计算机视觉</span></p>
<p>📖 <a href="/2026/07/29/实战：目标检测（YOLOv8）/"><strong>核心文章</strong>：实战：目标检测（YOLOv8）</a></p>
</div>

<div class="lesson-card">
<h4>第4课：拆开神经网络看看里面是什么</h4>
<p>CNN 为什么擅长看图？RNN/LSTM 为什么记得上下文？先从感性的例子理解，再深入数学原理。</p>
<p><span class="tag">理解原理</span></p>
<p>📖 <a href="/2026/07/03/机器学习入门：概念与分类全解/">机器学习入门：概念与分类全解</a><br>
📖 <a href="/2026/07/03/神经网络架构详解：CNN卷积神经网络/">CNN 卷积神经网络详解</a><br>
📖 <a href="/2026/07/03/神经网络架构详解：RNN与LSTM/">RNN 与 LSTM 详解</a></p>
</div>

<div class="lesson-card">
<h4>第5课：参加你的第一个 Kaggle 竞赛</h4>
<p>Titanic 是 Kaggle 的 Hello World。学数据探索、特征工程、模型集成——一套通用竞赛管线，任何表格数据比赛都能复用。</p>
<p><span class="tag">动手</span><span class="tag">竞赛</span></p>
<p>📖 <a href="/2026/07/29/Kaggle竞赛入门：Titanic生存预测/"><strong>核心文章</strong>：Kaggle 竞赛入门：Titanic 生存预测</a></p>
</div>

<div class="lesson-card">
<h4>第6课：参与开源——你的 GitHub 就是简历</h4>
<p>从提 Issue 到合 PR，走过完整流程。你会获得两个绿色方块，和一句能在面试时说的话："我有开源贡献。"</p>
<p><span class="tag">职业</span><span class="tag">社区</span></p>
<p>📖 <a href="/2026/07/29/开源项目贡献指南：如何参与AI开源社区/"><strong>核心文章</strong>：开源项目贡献指南</a></p>
</div>

---

## 📖 系统学习

完整的 AI 学习路径，从零到能用。

<div class="stage-card" style="border-left-color: #22c55e;">
<h4>阶段 0：打地基 <span style="color:#22c55e;font-weight:400;">（8 篇 · 预计 10~15 小时）</span></h4>
<p>Python 编程 + AI 数学基础。不用精通，够用就行。</p>
<ul>
<li>☐ <a href="/2026/06/17/Python编程基础（一）：从零搭建你的AI开发环境/">Python 环境搭建与基础语法</a></li>
<li>☐ <a href="/2026/06/17/Python编程基础（二）：变量、数据类型和基本运算/">变量、数据类型和基本运算</a></li>
<li>☐ <a href="/2026/06/17/Python编程基础（三）：NumPy——AI工程师的第一件武器/">NumPy：AI 工程师的第一件武器</a></li>
<li>☐ <a href="/2026/07/03/AI数学基础（一）：线性代数——让机器学会看向量/">线性代数：让机器学会看向量</a></li>
<li>☐ <a href="/2026/07/03/AI数学基础（二）：概率论——教机器做不确定的决策/">概率论：教机器做不确定的决策</a></li>
<li>☐ <a href="/2026/07/03/AI数学基础（三）：微积分——理解变化的语言/">微积分：理解变化的语言</a></li>
<li>☐ <a href="/2026/07/03/深度学习入门指南：从零开始理解神经网络/">深度学习入门指南</a></li>
<li>☐ <a href="/2026/07/03/PyTorch-vs-TensorFlow：2026年该选哪个框架/">PyTorch vs TensorFlow：该选哪个框架</a></li>
</ul>
</div>

<div class="stage-card" style="border-left-color: #3b82f6;">
<h4>阶段 1：核心概念 <span style="color:#3b82f6;font-weight:400;">（7 篇 · 预计 10~15 小时）</span></h4>
<p>机器学习入门、三个主流架构（CNN / RNN / Transformer）、NLP 与 CV 基础。</p>
<ul>
<li>☐ <a href="/2026/07/03/机器学习入门：概念与分类全解/">机器学习入门：概念与分类</a></li>
<li>☐ <a href="/2026/07/03/PyTorch实战（一）：张量操作与自动微分/">PyTorch 实战（一）：张量与自动微分</a></li>
<li>☐ <a href="/2026/07/03/PyTorch实战（二）：构建第一个全连接网络/">PyTorch 实战（二）：全连接网络</a></li>
<li>☐ <a href="/2026/07/03/神经网络架构详解：CNN卷积神经网络/">CNN 卷积神经网络</a></li>
<li>☐ <a href="/2026/07/03/神经网络架构详解：RNN与LSTM/">RNN 与 LSTM</a></li>
<li>☐ <a href="/2026/07/03/计算机视觉入门：图像处理基础/">计算机视觉入门</a></li>
<li>☐ <a href="/2026/07/03/NLP入门：文本预处理与词向量/">NLP 入门：文本预处理与词向量</a></li>
</ul>
</div>

<div class="stage-card" style="border-left-color: #a855f7;">
<h4>阶段 2：实战项目 <span style="color:#a855f7;font-weight:400;">（6 篇 · 预计 15~20 小时）</span></h4>
<p>写完代码跑起来的项目，每篇一个完整应用。</p>
<ul>
<li>☐ <a href="/2026/07/11/实战：手写数字识别（MNIST）/">手写数字识别（MNIST）</a></li>
<li>☐ <a href="/2026/07/11/实战：图像分类（CIFAR-10）/">图像分类（CIFAR-10）</a></li>
<li>☐ <a href="/2026/07/29/实战：情感分析系统/">情感分析系统（BoW→LSTM→BERT）</a></li>
<li>☐ <a href="/2026/07/29/实战：目标检测（YOLOv8）/">目标检测（YOLOv8）</a></li>
<li>☐ <a href="/2026/07/29/Kaggle竞赛入门：Titanic生存预测/">Kaggle Titanic 竞赛</a></li>
<li>☐ <a href="/2026/07/29/开源项目贡献指南：如何参与AI开源社区/">开源项目贡献指南</a></li>
</ul>
</div>

<div class="stage-card" style="border-left-color: #f59e0b;">
<h4>阶段 3：前沿技术 <span style="color:#f59e0b;font-weight:400;">（即将推出）</span></h4>
<p>Transformer 原理、LLM 部署与微调、WebGPU 推理、AI 双周速递、论文解读。</p>
<ul>
<li>⏳ Transformer 架构从零理解</li>
<li>⏳ LLM 部署指南：Ollama + vLLM + 量化</li>
<li>⏳ 用 LoRA 微调你的第一个模型</li>
<li>⏳ WebGPU：在浏览器里跑 AI 推理</li>
<li>⏳ AI 双周速递 & 论文解读</li>
</ul>
</div>

---

## 学习建议

1. **不要试图学完再动手。** 先走快速通道第 1 课，30 分钟就有一个能用的模型。再回头补基础——有了"我要解决什么问题"的目标感，学基础会快得多。
2. **不要跳过代码。** 读十遍不如跑一遍。每篇文章都有完整代码，复制粘贴到你自己的环境里跑一次。
3. **遇到问题怎么办？**
   - 每篇文章底部有评论区（Giscus），直接留言
   - 先看评论区有没有人问过同样的问题
   - 搜索引擎搜 error 信息（90% 的问题别人已经遇到过）

---

**准备好了？从[快速通道第 1 课](/2026/07/11/实战：图像分类（CIFAR-10）/)开始，5 分钟训练你的第一个模型。**
