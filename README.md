# AIJVS 博客

> 免费开源的深度学习和 AI 学习网站，让每个人都参与到 AI 中来！

🌐 **网站**: https://aijvs.com

📝 **博客框架**: Hexo

🎨 **主题**: Butterfly

📦 **部署**: Cloudflare Pages

---

## 🚀 快速开始

### 环境要求

- Node.js >= 16
- npm >= 6

### 安装依赖

```bash
npm install
```

### 本地开发

```bash
# 启动本地服务器
hexo server

# 访问 http://localhost:4000
```

### 生成静态文件

```bash
hexo generate
# 或
hexo g
```

### 清理缓存

```bash
hexo clean
```

---

## 📝 创建新文章

```bash
# 创建新文章
hexo new post "文章标题"

# 文章位于 source/_posts/ 目录
```

### Front Matter 模板

```markdown
---
title: 文章标题
date: 2026-05-25 17:00:00
categories:
  - 分类名
tags:
  - 标签 1
  - 标签 2
---

# 正文内容
```

---

## 🎨 主题配置

主题配置文件位于：`themes/butterfly/_config.butterfly.yml`

常用配置项：
- 导航菜单
- 社交链接
- 评论系统
- 搜索功能
- 深色模式

---

## 📦 部署到 Cloudflare Pages

### 1. 推送到 GitHub

```bash
git init
git add .
git commit -m "Initial commit"

# 在 GitHub 创建仓库后
git remote add origin https://github.com/你的用户名/aijvs-blog.git
git branch -M main
git push -u origin main
```

### 2. Cloudflare Pages 配置

1. 访问 https://pages.cloudflare.com/
2. 点击 "Create a project"
3. 选择 "Connect to Git"
4. 选择 `aijvs-blog` 仓库
5. 配置构建设置：
   - **Framework preset**: Hexo
   - **Build command**: `hexo generate`
   - **Build output directory**: `public`
   - **Root directory**: `/`
6. 点击 "Save and Deploy"

### 3. 绑定自定义域名

1. 在 Cloudflare Pages 项目中进入 "Custom domains"
2. 点击 "Add a custom domain"
3. 输入 `aijvs.com`
4. 按照提示配置 DNS（CNAME 记录）

---

## 📁 项目结构

```
aijvs-blog/
├── _config.yml              # 站点配置
├── package.json             # 依赖管理
├── themes/
│   └── butterfly/           # Butterfly 主题
│       └── _config.butterfly.yml  # 主题配置
├── source/
│   ├── _posts/              # 博客文章
│   ├── categories/          # 分类页
│   ├── tags/                # 标签页
│   ├── archives/            # 归档页
│   ├── about/               # 关于页
│   └── img/                 # 图片资源
├── public/                  # 生成的静态文件
└── README.md
```

---

## 🔧 常用命令

| 命令 | 说明 |
|------|------|
| `hexo new post "标题"` | 创建新文章 |
| `hexo new page "页面名"` | 创建新页面 |
| `hexo generate` | 生成静态文件 |
| `hexo server` | 本地预览 |
| `hexo clean` | 清理缓存 |
| `hexo deploy` | 部署（如配置） |

---

## 🎯 内容规划

### 基础教程
- [ ] Python 编程基础
- [ ] 数学基础系列
- [ ] 机器学习入门

### 进阶内容
- [ ] 深度学习框架对比
- [ ] 神经网络详解
- [ ] 计算机视觉
- [ ] 自然语言处理

### 实战项目
- [ ] 手写数字识别
- [ ] 图像分类项目
- [ ] 文本分类项目
- [ ] Kaggle 竞赛实战

---

## 🤝 贡献指南

欢迎贡献内容！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

MIT License

---

## 🔗 友情链接

- [Hexo 官方文档](https://hexo.io/zh-cn/docs/)
- [Butterfly 主题文档](https://butterfly.js.org/)
- [Cloudflare Pages 文档](https://developers.cloudflare.com/pages/)

---

**AIJVS - 让每个人都参与到 AI 学习中来！**
