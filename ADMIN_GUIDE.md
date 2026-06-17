# AIJVS 博客管理后台部署指南

## 📋 方案说明

由于 Hexo 是静态博客生成器，我们提供两种管理后台方案：

### 方案 A：hexo-admin 插件（推荐用于本地/服务器部署）
- ✅ 功能完整
- ✅ 支持文章管理、配置管理
- ❌ 需要 Node.js 服务器环境

### 方案 B：Cloudflare Pages + 静态管理界面（推荐用于 Cloudflare 部署）
- ✅ 无需服务器
- ✅ 免费托管
- ⚠️ 功能简化（文章管理需通过 GitHub）

---

## 🚀 方案 A：hexo-admin 完整后台

### 1. 安装 hexo-admin

```bash
cd /home/admin/openclaw/workspace/aijvs-blog
npm install hexo-admin --save
```

### 2. 配置 hexo-admin

在 `_config.yml` 末尾添加：

```yaml
# hexo-admin 配置
admin:
  username: admin
  password_hash: $2a$10$JLpY7FqZKvNn5XqQ8u9H5OZKjJ5XqQ8u9H5OZKjJ5XqQ8u9H5O
  secret: aijvs_blog_secret_key_2026
  deployCommand: hexo generate
```

### 3. 生成密码哈希

```bash
node -e "console.log(require('bcryptjs').hashSync('你的密码', 10))"
```

### 4. 启动管理后台

```bash
# 本地运行
hexo server -p 4000

# 访问管理后台
http://localhost:4000/admin
```

### 5. 部署到服务器（可选）

```bash
# 使用 PM2 管理
npm install -g pm2
pm2 start "hexo server --admin" --name aijvs-admin
pm2 save
pm2 startup
```

---

## ☁️ 方案 B：Cloudflare Pages 静态管理

### 1. 访问管理页面

```
https://aijvs.com/admin/
```

### 2. 默认登录信息

```
用户名：admin
密码：admin123
```

### 3. 功能说明

| 功能 | 状态 | 说明 |
|------|------|------|
| 文章列表 | ✅ | 查看所有文章 |
| 新建文章 | ✅ | 创建 Markdown 文章 |
| 编辑文章 | ✅ | 修改现有文章 |
| 删除文章 | ✅ | 删除文章 |
| 分类管理 | ✅ | 自动统计 |
| 标签管理 | ✅ | 自动统计 |
| 配置管理 | ⏳ | 计划中 |
| 评论管理 | ⏳ | 需集成评论系统 |
| 媒体库 | ⏳ | 计划中 |

---

## 🔐 安全建议

### 修改默认密码

1. 登录管理后台
2. 进入设置页面
3. 修改密码

### 启用 HTTPS

Cloudflare Pages 默认启用 HTTPS，无需额外配置。

### 限制访问（可选）

在 Cloudflare Dashboard 设置访问规则：

1. 进入 **Security** → **WAF**
2. 添加规则限制特定 IP 访问 `/admin/*`

---

## 📝 使用工作流

### 发布新文章的流程

#### 方式 1：通过管理后台（推荐）

1. 访问 `https://aijvs.com/admin/`
2. 登录后点击「新建文章」
3. 填写标题、分类、标签、内容
4. 点击「保存」
5. 文章自动保存到 `source/_posts/` 目录
6. Cloudflare Pages 自动检测并重新构建

#### 方式 2：通过 GitHub

1. 在 GitHub 仓库创建新文件
2. 路径：`source/_posts/文章标题.md`
3. 使用 Front Matter 格式：

```markdown
---
title: 文章标题
date: 2026-05-26 00:00:00
categories:
  - 分类名
tags:
  - 标签 1
  - 标签 2
---

# 正文内容
```

4. 提交后 Cloudflare Pages 自动构建

#### 方式 3：本地 Hexo

```bash
# 创建新文章
hexo new post "文章标题"

# 编辑文章
code source/_posts/文章标题.md

# 生成并预览
hexo clean && hexo generate && hexo server

# 推送到 GitHub
git add .
git commit -m "Add: 文章标题"
git push
```

---

## 🛠️ 故障排查

### 问题 1：无法访问管理后台

**症状**: 访问 `/admin/` 显示 404

**解决方案**:
```bash
# 确认 admin 页面存在
ls source/admin/index.html

# 重新生成
hexo clean && hexo generate
```

### 问题 2：登录失败

**症状**: 提示用户名或密码错误

**解决方案**:
1. 检查浏览器控制台是否有错误
2. 清除浏览器缓存
3. 确认 API 端点正常

### 问题 3：文章保存后不显示

**症状**: 保存成功但网站不显示新文章

**解决方案**:
```bash
# 检查文章格式
cat source/_posts/文章标题.md

# 重新生成
hexo clean && hexo generate

# 推送 GitHub 触发 Cloudflare 构建
git add . && git commit -m "Update" && git push
```

---

## 📊 统计数据

管理后台会自动统计：

- 📝 文章总数
- 📂 分类数量
- 🏷️ 标签数量

数据来源于 `source/_posts/` 目录下的文章 Front Matter。

---

## 🔗 相关文档

- [Hexo 官方文档](https://hexo.io/zh-cn/docs/)
- [Butterfly 主题文档](https://butterfly.js.org/)
- [hexo-admin 插件](https://github.com/jaredly/hexo-admin)
- [Cloudflare Pages 文档](https://developers.cloudflare.com/pages/)

---

*最后更新：2026-05-26*
