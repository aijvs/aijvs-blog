# AIJVS 博客 Cloudflare Pages 配置指南

## 🌐 配置自定义域名

### 步骤 1：添加 blog.aijvs.com 到 Cloudflare Pages

1. 访问 https://pages.cloudflare.com/
2. 进入 `aijvs-blog` 项目
3. 点击 **"Custom domains"** 标签
4. 点击 **"Add a custom domain"**
5. 输入：`blog.aijvs.com`
6. 点击 **"Add domain"**

### 步骤 2：配置 DNS

Cloudflare 会自动创建 CNAME 记录：

```
类型：CNAME
名称：blog
内容：aijvs-blog.[你的 subdomain].workers.dev
Proxy: 开启（橙色云朵）
```

### 步骤 3：验证 SSL 证书

等待 SSL 证书生效（通常 5-15 分钟）

---

## 🔒 限制后台访问

### 方式 1：Cloudflare Access（推荐）

1. 访问 https://one.dash.cloudflare.com/
2. 点击 **"Access"** → **"Applications"**
3. 点击 **"Add an application"**
4. 选择 **"Self-hosted"**
5. 配置：
   - **Name**: AIJVS Admin
   - **Subdomain**: `blog`
   - **Domain**: `aijvs.com`
   - **Path**: `/admin/*`
6. 添加访问策略：
   - 允许特定邮箱/用户访问
   - 或允许所有人访问（仅密码保护）

### 方式 2：Cloudflare Workers 访问控制

创建 Workers 脚本检查域名：

```javascript
// 在 Cloudflare Workers 中创建
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)
  
  // 只允许 blog.aijvs.com 访问 /admin/*
  if (url.pathname.startsWith('/admin/')) {
    if (url.hostname !== 'blog.aijvs.com') {
      return new Response('Forbidden: Admin access denied', { status: 403 })
    }
  }
  
  return fetch(request)
}
```

### 方式 3：Cloudflare Rules（最简单）

1. 访问 https://dash.cloudflare.com/
2. 选择 `aijvs.com` 域名
3. 进入 **"Security"** → **"WAF"**
4. 点击 **"Create rule"**
5. 配置规则：

```
Rule name: Block Admin from Main Domain

If expression:
(http.request.uri.path contains "/admin") and (http.host eq "aijvs.com")

Then:
Block
```

---

## ✅ 验证配置

### 测试 1：访问 blog.aijvs.com/admin

```
应该：正常显示登录页面
```

### 测试 2：访问 aijvs.com/admin

```
应该：403 Forbidden 或页面拒绝访问
```

---

## 📋 完整访问控制方案

| 层面 | 配置 | 说明 |
|------|------|------|
| **DNS** | blog.aijvs.com CNAME | 独立子域名 |
| **Cloudflare Pages** | Custom domain | 绑定子域名 |
| **前端验证** | JavaScript 域名检查 | 第一层保护 |
| **Cloudflare Rules** | WAF 规则 | 第二层保护 |
| **Cloudflare Access** | 访问策略 | 第三层保护（可选）|

---

## 🔐 推荐的完整配置

```
1. 添加 blog.aijvs.com 到 Cloudflare Pages
2. 配置 DNS CNAME 记录
3. 创建 Cloudflare WAF 规则阻止 aijvs.com/admin
4. （可选）启用 Cloudflare Access 添加额外认证
```

---

*最后更新：2026-05-26*
