---
title: WebGPU实战：在浏览器里跑大模型（零成本本地推理）
date: 2026-08-06 13:00:00
categories:
  - 实战项目
tags:
  - WebGPU
  - 浏览器
  - Transformers.js
  - 部署
  - 进阶
description: 用 WebGPU + Transformers.js 在浏览器里直接运行大模型：无需服务器、无需安装、隐私全本地。从原理到可运行代码，20分钟让AI在你的浏览器里跑起来。
---

> **阅读前置**：本篇是 LLM 系列的实战篇。理论部分见 [大语言模型（LLM）原理](/2026/08/06/大语言模型（LLM）原理：从训练到推理的完整流程/)。需要基础 HTML/JavaScript 知识，Python 不是必需。

---

## 为什么要在浏览器里跑 AI？

先看一组真实数据（来自 2026 年 WebGPU 基准测试，Apple M4 Max + Chrome）：

| 指标 | 数值 |
|---|---|
| 模型 | Gemma-4-E2B（20 亿参数） |
| 预填充速度 | **4676 token/秒** |
| 解码速度 | 73.9 token/秒 |
| 初始化时间 | 1.1 秒 |
| 模型大小 | 200 MB |
| GPU 内存 | 1.8 GB |

**你的浏览器，就是一台 AI 推理服务器。** 这意味着：

1. **零成本**：不用买 GPU，不用租云服务器
2. **隐私安全**：数据不出本地，聊天记录不会上传
3. **离线可用**：模型加载后断网也能用
4. **无需安装**：打开网页就是 AI 应用

这正是 [AI双周速递·创刊号](/2026/08/06/AI双周速递·创刊号：Kimi%20K3%20登顶全球开源王座，DeepSeek%20V4-Flash%20正式版上线/) 里说的"AI 应用爆发"的一个缩影——当推理成本趋近于零，应用形态就被彻底改变。

---

## 一、WebGPU 是什么？

**WebGPU** 是浏览器的新一代 GPU 接口（类似 WebGL 的下一代，但更接近现代图形 API），2023 年起在 Chrome/Edge/Firefox 逐步可用，2026 年已全面普及。

关键点：

- **GPU 加速**：利用显卡并行计算，比 CPU 快 10-100 倍
- **AI 推理**：虽然不是为 AI 设计的，但大模型的矩阵运算天然适合 GPU
- **WASM 配合**：WebAssembly 处理模型解析，WebGPU 负责计算

```
浏览器 → Transformers.js (推理引擎) → WebGPU (GPU加速) → 显卡
```

---

## 二、Transformers.js：浏览器里的 HuggingFace

**Transformers.js**（HuggingFace 官方）把 Python 的 transformers 库搬到了 JavaScript。它支持：

- **数百个模型**：Llama、Gemma、Phi、Qwen 等开源模型
- **多任务**：文本生成、分类、翻译、图像识别、语音识别
- **自动选择后端**：有 WebGPU 用 GPU，没有就退到 WASM/CPU

> 小技巧：最新版需用 `npm i @huggingface/transformers`，或从 CDN 引入。

---

## 三、实战：20 行代码跑文本生成

### 3.1 用 CDN（最简单，纯 HTML）

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>浏览器里的 AI</title>
</head>
<body>
  <h1>🦄 浏览器本地大模型</h1>
  <p>首次加载需要下载模型（约200MB），之后可离线使用</p>
  <textarea id="input" rows="3" cols="60">请用一句话介绍人工智能</textarea>
  <br>
  <button onclick="run()">生成</button>
  <pre id="output">等待中...</pre>

  <script type="module">
    import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
    
    // 创建文本生成流水线（device: 'webgpu' 启用GPU加速）
    const generator = await pipeline('text-generation', 'onnx-community/Qwen2.5-0.5B-Instruct', {
      device: 'webgpu',   // 用 WebGPU 加速；不支持则去掉这行自动降级
      dtype: 'q4',        // 4位量化，模型更小更快
    });

    async function run() {
      const text = document.getElementById('input').value;
      const output = document.getElementById('output');
      output.textContent = '生成中...';
      
      const result = await generator(text, {
        max_new_tokens: 100,
        do_sample: true,
        temperature: 0.7,
      });
      
      output.textContent = result[0].generated_text;
    }
  </script>
</body>
</html>
```

**保存为 `index.html`，双击打开**（或 `npx serve`），等待模型下载即可使用。

### 3.2 关键参数解读

| 参数 | 作用 |
|---|---|
| `device: 'webgpu'` | 启用 GPU 加速，速度提升 10-100 倍 |
| `dtype: 'q4'` | 4 位量化，200MB 模型只要 50MB |
| `max_new_tokens` | 最多生成多少 token |
| `temperature` | 随机性（0=确定，1=发散） |

### 3.3 运行效果

```
输入: 请用一句话介绍人工智能
输出: 人工智能是让计算机模拟人类智能行为的技术，
      包括学习、推理、感知和自然语言处理等方面。
```

**注意**：首次运行会下载模型（浏览器缓存到本地），之后断网也能用。想换模型？把 `onnx-community/Qwen2.5-0.5B-Instruct` 换成任意支持的模型名（如 `Xenova/llama-3.2-1b`）即可。

---

## 四、进阶：做个本地 AI 助手页面

把生成结果流式输出（像 ChatGPT 一样打字效果）：

```javascript
// 流式生成
const stream = await generator(text, {
  max_new_tokens: 200,
  stream: true,          // 开启流式
  callback_function: (chunk) => {
    const token = chunk[0].output_token_text;
    output.textContent += token;  // 逐字追加显示
  }
});
```

加上 CSS 聊天框样式，你就有了一个**完全本地运行、数据不出设备**的 AI 聊天助手——可以部署到 Cloudflare Pages / Vercel / GitHub Pages 上免费发布。

---

## 五、WebGPU 能跑多大模型？

| 设备 | 可运行模型 | 说明 |
|---|---|---|
| 手机（旗舰） | 1-4B 量化 | 200MB-1GB |
| 笔记本（集显） | 4-8B 量化 | 1-3GB |
| 台式机（独显） | 8-32B 量化 | 3-10GB |
| 工作站 | 70B 量化 | 需 40GB+ 显存 |

> 大规模模型（如 Kimi K3 的 2.8 万亿）不适合浏览器——但绝大多数日常场景（写作、翻译、摘要、代码补全），**10B 以内的模型完全够用**。这就是 [LLM 原理](/2026/08/06/大语言模型（LLM）原理：从训练到推理的完整流程/) 里说的"蒸馏 + 量化让模型变小变快"的落地。

---

## 六、动手挑战（做完发评论区）

1. **改造**：把上面的代码加一个"加载进度条"（`generator` 支持 `progress_callback`）
2. **换模型**：试试图像分类 `Xenova/vit-base-patch16-224`，用 WebGPU 识别本地图片
3. **发布**：部署到 Cloudflare Pages，把链接发到评论区——让全世界用你的本地 AI

### 配套资源

- [AI导航](/nav/) 里有 WebGPU 相关工具站（搜索 "WebGPU"）
- [开源项目贡献指南](/2026/07/29/开源项目贡献指南：如何参与AI开源社区/) 教你如何给 Transformers.js 提 PR

---

## 七、总结

| 知识点 | 一句话 |
|---|---|
| WebGPU | 浏览器 GPU 接口，让 AI 推理本地化 |
| Transformers.js | HuggingFace 的 JS 版，一行代码调用模型 |
| 量化 | q4 让模型体积缩小 4 倍 |
| 应用场景 | 隐私敏感、离线、低成本场景的最佳方案 |

### 联动路径

- 理论：[大语言模型（LLM）原理](/2026/08/06/大语言模型（LLM）原理：从训练到推理的完整流程/) → [Transformer深度解读](/2026/08/06/Transformer深度解读：从注意力机制到现代大模型的基石/)
- 实战：本篇 → [PyTorch实战](/2026/07/03/PyTorch实战（一）：张量操作与自动微分/) → [图像分类（CIFAR-10）](/2026/07/11/实战：图像分类（CIFAR-10）/)
- 资讯：看看最近哪些模型支持浏览器端 → [AI双周速递·创刊号](/2026/08/06/AI双周速递·创刊号：Kimi%20K3%20登顶全球开源王座，DeepSeek%20V4-Flash%20正式版上线/)

---

**思考题**：如果每个人的浏览器都能跑 AI，云厂商的推理业务会受影响吗？什么场景必须用云端大模型（提示：Kimi K3 的 100 万 token 上下文）？评论区见。
