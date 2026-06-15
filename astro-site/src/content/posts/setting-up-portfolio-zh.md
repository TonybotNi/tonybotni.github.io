---
title: "搭建你的学术作品集"
date: "2026-04-28"
description: "一份关于如何克隆、配置并部署全新学术作品集的完整指南。"
author: "Claude Shannon"
tags:
  - "Documentation"
  - "Setup"
  - "Astro"
  - "Portfolio"
---

# 搭建你的学术作品集

欢迎来到你的全新学术作品集！本模板基于 Astro 构建，专为学者、研究人员和专业人士设计，帮助你打造简洁、高性能、可定制的网站，用于展示工作成果、发表论文与个人思考。

在本指南中，我将带你完成从零到上线的全部流程——从克隆仓库，到配置站点，再到撰写第一篇内容。

## 1. 克隆仓库

首先，你需要将仓库克隆到本地。可以使用标准 Git，也可以使用 GitHub CLI。

**方式 A：使用 Git**
在终端中运行以下命令：

```bash
git clone https://github.com/rubzip/academic-portfolio-astro.git my-portfolio
cd my-portfolio
```

**方式 B：使用 GitHub CLI（推荐）**
如果你希望立即以本仓库为模板创建自己的远程仓库，GitHub CLI 会非常方便：

```bash
gh repo create my-portfolio --template="rubzip/academic-portfolio-astro" --clone
cd my-portfolio
```

接下来安装项目依赖。本项目使用 `npm`（请确保已安装 Node.js >= 22.12.0）：

```bash
npm install
```

依赖安装完成后，即可启动本地开发服务器：

```bash
npm run dev
```

站点应已在 `http://localhost:4321` 运行。你对代码所做的修改会自动在浏览器中热更新。

## 2. 全局配置

作品集的核心配置集中在 `src/config/site.ts` 文件中。这是你应当首先编辑的文件。

```typescript
// src/config/site.ts
export const SITE: SiteConfig = {
    website: "https://your-domain.com/",
    author: "Your Name",
    desc: "Your personal academic portfolio.",
    title: "Your Name",
    ogImage: "your-image.webp",
    postPerPage: 5,
    favicon: "/favicon.svg",
    lang: "en",
};
```

请务必将 `website`、`author`、`desc`、`title` 等字段更新为你的个人信息。

### 分析统计配置

如需追踪页面访问量，可在同一文件中配置 Google Analytics 4 或 Umami：

```typescript
export const ANALYTICS: AnalyticsConfig = {
    ga4Id: "G-XXXXXXXXXX", // 填入你的 Google Analytics 4 Measurement ID
    umami: {
        websiteId: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        src: "https://cloud.umami.is/script.js", 
    }
};
```

分析脚本默认懒加载，不会对 Core Web Vitals 造成负面影响！

## 3. 主题与外观

本模板开箱即用，视觉效果出色。我们使用集中式主题引擎，配置位于 `src/config/site.ts` 与 `src/config/themes.ts`。

在 `site.ts` 中，你可以开关深色模式，并选择偏好的浅色与深色主题：

```typescript
export const THEME_CONFIG: ThemeConfig = {
    lightAndDark: true, // 启用月亮/太阳切换
    themeLight: "light_notepad", 
    themeDark: "dark_notepad",
};
```

本地服务器运行时，可访问内置开发者工具 `http://localhost:4321/dev-tools` 预览所有可用配色！若要自定义配色，只需在 `src/config/themes.ts` 的 `THEMES` 中新增一项即可。

## 4. 内容管理

作品集完全由 Markdown（`.md`）文件驱动，便于版本控制与自由写作。所有内容集合位于 `src/content/`。

主要集合包括：
- **`posts/`**：博客文章（就像本篇）。
- **`projects/`**：展示软件、硬件或研究项目。
- **`publications/`**：列出学术论文。
- **`talks/`**：归档会议报告与幻灯片。
- **`teaching/`**：列出授课或曾授课程。

### 禁用页面

若不需要默认提供的全部内容集合，可在 `src/config/pages.ts` 中禁用。

例如，若不想使用 `talks` 集合，可在 `src/config/pages.ts` 中设置：

```typescript
export const PAGES: PagesConfig = {
    // 页面默认配置，可覆盖任意属性
    talks: {
        title: "Talks & Presentations",
        subtitle: "Public lectures, colloquia, and conference presentations.",
        isActive: false, // 禁用 talks 集合
    },
    ...
};
```

此配置会同时关闭导航栏中的 talks 页面与对应内容集合。

### 撰写新文章

在 `src/content/posts` 目录下新增 `.md` 文件即可创建新文章。`src/example_contents/` 中为各集合提供了示例。frontmatter 示例如下：

```yaml
---
title: "My New Research Idea"
date: "2026-05-01"
description: "A brief exploration of a novel concept."
author: "Your Name"
tags:
  - "Research"
  - "Theory"
---
```

Markdown 中可直接使用 $\LaTeX$ 数学公式，由 KaTeX 提供支持！

$$ E = mc^2 $$

## 5. 导航与社交链接

若要增删导航栏链接，或更新左侧边栏的社交媒体图标，请编辑：

- `src/config/navigation.ts`：控制顶部导航栏。
- `src/config/social.ts`：控制左侧个人资料区的图标。

## 6. 构建与部署

内容与配置满意后，可构建生产版本：

```bash
npm run build
```

这会在 `dist/` 目录生成高度优化的静态 HTML 文件。可将该目录部署到 GitHub Pages、Vercel、Netlify 或任意静态托管服务。

---

以上就是全部步骤！你现在可以开始向世界发布你的学术旅程了。
