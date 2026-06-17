# AIJVS 博客部署指南

本文档详细说明如何将 AIJVS 博客部署到 Cloudflare Pages。

---

## 📋 部署前检查清单

- [ ] 已安装 Node.js 和 npm
- [ ] 已在 GitHub 创建账号
- [ ] 已在 Cloudflare 创建账号
- [ ] aijvs.com 域名已添加到 Cloudflare
- [ ] 本地测试通过（`hexo server` 可正常访问）

---

## 🚀 第一步：推送到 GitHub

### 1.1 初始化 Git 仓库

```bash
cd /home/admin/openclaw/workspace/aijvs-blog

git init
git add .
git commit -m "Initial commit - AIJVS blog"
```

### 1.2 创建 GitHub 仓库

访问 https://github.com/new

- **Repository name**: `aijvs-blog`
- **Description**: AIJVS 官方博客 - 免费开源的深度学习和 AI 学习网站
- **Public**（公开仓库）
- 不要初始化 README（我们已经有了）

### 1.3 推送代码

```bash
git branch -M main
git remote add origin https://github.com/你的用户名/aijvs-blog.git
git push -u origin main
```

---

## ☁️ 第二步：Cloudflare Pages 部署

### 2.1 登录 Cloudflare

访问 https://pages.cloudflare.com/

使用你的 Cloudflare 账号登录。

### 2.2 创建项目

1. 点击 **"Create a project"**
2. 选择 **"Connect to Git"**

### 2.3 连接 GitHub

1. 点击 **"Authorize Cloudflare Pages"**（如首次使用）
2. 选择你的 GitHub 账号
3. 授权 Cloudflare 访问你的仓库

### 2.4 选择仓库

在仓库列表中找到并选择 `aijvs-blog`

### 2.5 配置构建设置

```
┌─────────────────────────────────────────┐
│ Build settings                          │
├─────────────────────────────────────────┤
│ Framework preset: Hexo                  │
│ Build command: hexo generate            │
│ Build output directory: public          │
│ Root directory: /                       │
│ NODE_VERSION: 18                        │
└─────────────────────────────────────────┘
```

**重要配置说明：**

| 配置项 | 值 | 说明 |
|--------|-----|------|
| Framework preset | Hexo | 自动配置 Hexo 环境 |
| Build command | `hexo generate` | 生成静态文件 |
| Output directory | `public` | Hexo 输出目录 |
| Root directory | `/` | 项目根目录 |
| NODE_VERSION | `18` | Node.js 版本 |

### 2.6 保存并部署

1. 点击 **"Save and Deploy"**
2. 等待构建完成（通常 1-3 分钟）
3. 构建成功后会显示预览 URL（类似 `aijvs-blog.xxxx.workers.dev`）

---

## 🌐 第三步：绑定自定义域名

### 3.1 在 Cloudflare Pages 中添加域名

1. 进入你的 Pages 项目
2. 点击 **"Custom domains"** 标签
3. 点击 **"Add a custom domain"**
4. 输入：`aijvs.com`
5. 点击 **"Add domain"**

### 3.2 配置 DNS

Cloudflare 会自动检测域名并配置。如需手动配置：

#### 方案 A：根域名（aijvs.com）

```
类型：CNAME
名称：@
内容：aijvs-blog.xxxx.workers.dev
Proxy: 开启（橙色云朵）
```

#### 方案 B：子域名（www.aijvs.com）

```
类型：CNAME
名称：www
内容：aijvs-blog.xxxx.workers.dev
Proxy: 开启（橙色云朵）
```

### 3.3 SSL 证书

Cloudflare 会自动提供 HTTPS 证书，无需手动配置。

证书生效时间：通常 5-15 分钟

---

## ✅ 第四步：验证部署

### 4.1 访问网站

- 测试 URL: `https://aijvs-blog.xxxx.workers.dev`
- 正式域名：`https://aijvs.com`

### 4.2 检查清单

- [ ] 首页正常显示
- [ ] 文章列表可见
- [ ] 分类/标签页面正常
- [ ] 搜索功能可用
- [ ] 移动端适配正常
- [ ] HTTPS 证书生效

---

## 🔄 后续更新

### 自动部署

Cloudflare Pages 已配置自动部署：

```
推送代码到 GitHub → 自动触发构建 → 自动部署
```

### 更新文章流程

```bash
# 1. 创建新文章
hexo new post "新文章标题"

# 2. 编辑文章内容
# 在 source/_posts/ 目录编辑

# 3. 本地测试
hexo clean && hexo generate && hexo server

# 4. 提交并推送
git add .
git commit -m "Add: 新文章标题"
git push

# 5. 等待 Cloudflare 自动部署（1-3 分钟）
```

---

## 🔧 故障排查

### 问题 1：构建失败

**错误信息**: `Build failed`

**解决方案：**

1. 查看构建日志
2. 检查 Node.js 版本设置（应为 18）
3. 确认 `package.json` 包含所有依赖
4. 本地运行 `hexo generate` 测试

### 问题 2：域名不解析

**症状**: 访问 aijvs.com 显示错误

**解决方案：**

1. 检查 DNS 配置是否正确
2. 等待 DNS 传播（最多 48 小时）
3. 检查 Cloudflare SSL 证书状态
4. 清除浏览器缓存

### 问题 3：404 错误

**症状**: 页面显示 404

**解决方案：**

1. 检查 `public` 目录是否生成
2. 确认构建命令正确
3. 检查文章 front matter 格式

### 问题 4：样式丢失

**症状**: 页面显示正常但无样式

**解决方案：**

1. 清除浏览器缓存
2. 检查主题是否正确安装
3. 确认 `hexo generate` 生成 CSS 文件

---

## 📊 性能优化建议

### 1. 启用 CDN 缓存

在 Cloudflare 中配置：

- **Caching Level**: Standard
- **Browser Cache TTL**: 7 days
- **Auto Minify**: 启用 HTML、CSS、JS

### 2. 图片优化

```bash
# 安装图片压缩插件
npm install hexo-image-compress --save
```

### 3. 启用懒加载

Butterfly 主题已默认启用图片懒加载。

---

## 📈 监控与分析

### 1. Cloudflare Analytics

访问 Cloudflare Dashboard → Analytics 查看：

- 访问量统计
- 带宽使用
- 缓存命中率

### 2. 添加统计工具

#### Google Analytics

在主题配置中添加：

```yaml
# themes/butterfly/_config.butterfly.yml
analytics:
  google:
    enable: true
    id: UA-XXXXXXXXX-X
```

#### Umami（隐私友好）

```yaml
analytics:
  umami:
    enable: true
    src: https://umami.example.com/script.js
    data_website_id: xxxxxxxx
```

---

## 💰 费用说明

### Cloudflare Pages 免费额度

| 资源 | 免费额度 | 说明 |
|------|---------|------|
| 网站数量 | 无限 | 可部署多个项目 |
| 请求次数 | 无限 | 无限制 |
| 带宽 | 无限 | 无限制 |
| 构建次数 | 500 次/月 | 对博客足够 |
| 构建并发 | 1 | 同时 1 个构建 |

**结论**: 个人博客完全够用，无需付费。

---

## 🎯 最佳实践

### 1. 提交信息规范

```bash
# 新增文章
git commit -m "Add: 文章标题"

# 修改配置
git commit -m "Config: 修改主题配置"

# 修复问题
git commit -m "Fix: 修复搜索功能"

# 更新内容
git commit -m "Update: 更新关于页面"
```

### 2. 分支管理

- `main`: 生产分支（自动部署）
- `dev`: 开发分支（测试用）
- `feature/*`: 功能分支

### 3. 备份策略

- GitHub 仓库自动备份
- 定期导出 `source/_posts/` 目录
- 保存配置文件到本地

---

## 📞 获取帮助

- [Hexo 官方文档](https://hexo.io/zh-cn/docs/)
- [Butterfly 主题文档](https://butterfly.js.org/)
- [Cloudflare Pages 文档](https://developers.cloudflare.com/pages/)
- [GitHub Issues](https://github.com/aijvs/aijvs-blog/issues)

---

**最后更新**: 2026-05-25
