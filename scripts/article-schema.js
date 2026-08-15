'use strict';

/**
 * Article Schema 增强脚本 — GEO 优化
 * Butterfly 主题自带 BlogPosting JSON-LD（headline/url/image/date/author），
 * 但缺 description / keywords / articleSection / publisher / mainEntityOfPage。
 * 本脚本在 after_render:html 阶段：
 *   1. 找到主题注入的 BlogPosting JSON-LD
 *   2. 解析并补齐 GEO 关键字段（description、keywords、articleSection、publisher、mainEntityOfPage、inLanguage）
 *   3. 写回 HTML
 * 仅处理文章页（page.layout === 'post'），列表页不动。
 */

const SITE = 'https://aijvs.com';

hexo.extend.filter.register('after_render:html', function (str, data) {
  // 仅处理文章页
  if (!data || !data.page) return str;
  if (data.page.layout !== 'post' && !(data.page.is_post && data.page.is_post())) return str;

  const page = data.page;
  const title = page.title;
  const url = page.permalink || (SITE + page.path);

  // 摘要：优先 description，否则取内容前 200 字符
  let description = page.description || '';
  if (!description && page.content) {
    const plain = page.content
      .replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    description = plain.substring(0, 200);
  }

  const categories = (page.categories && page.categories.data ? page.categories.data : [])
    .map((c) => (typeof c === 'string' ? c : c.name))
    .filter(Boolean);
  const tags = (page.tags && page.tags.data ? page.tags.data : [])
    .map((t) => (typeof t === 'string' ? t : t.name))
    .filter(Boolean);

  const datePublished = page.date ? page.date.toISOString() : '';
  const dateModified = page.updated ? page.updated.toISOString() : datePublished;

  // 完整 Article Schema（覆盖主题的简化版）
  const schema = {
    '@context': 'https://schema.org',
    '@type': 'BlogPosting',
    headline: title,
    description: description,
    datePublished: datePublished,
    dateModified: dateModified,
    author: {
      '@type': 'Person',
      name: 'AIJVS',
      url: SITE
    },
    publisher: {
      '@type': 'Organization',
      name: 'AIJVS',
      logo: {
        '@type': 'ImageObject',
        url: SITE + '/img/logo.svg'
      }
    },
    mainEntityOfPage: {
      '@type': 'WebPage',
      '@id': url
    },
    url: url,
    inLanguage: 'zh-CN'
  };
  if (categories.length) schema.articleSection = categories[0];
  if (tags.length) schema.keywords = tags.join(', ');
  // image: 复用主题已有的或 avatar
  const imgMatch = str.match(/"image"\s*:\s*"([^"]+)"/);
  if (imgMatch) schema.image = imgMatch[1];

  const jsonld =
    '<script type="application/ld+json">' +
    JSON.stringify(schema)
      .replace(/</g, '\\u003c')
      .replace(/>/g, '\\u003e') +
    '</script>';

  // 1. 删除主题注入的 BlogPosting（保留 WebSite）
  const blogPostingRe =
    /<script type="application\/ld\+json">[\s\S]*?"@type"\s*:\s*"BlogPosting"[\s\S]*?<\/script>/;
  if (blogPostingRe.test(str)) {
    str = str.replace(blogPostingRe, '');
  }

  // 2. 注入到 </head> 前
  if (str.includes('</head>')) {
    return str.replace('</head>', jsonld + '\n</head>');
  }
  return str;
});
