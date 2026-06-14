# tonybotni.github.io

Personal academic homepage built with [Astro](https://astro.build) (`academic-portfolio-astro` template).

## Live site

- **URL:** https://tonybotni.github.io/
- **Deploy:** GitHub Actions on push to `master` (see [.github/workflows/deploy.yml](.github/workflows/deploy.yml))

## Project layout

| Path | Purpose |
|------|---------|
| [astro-site/](astro-site/) | Current site source (edit content & config here) |
| [archive/legacy-jekyll/](archive/legacy-jekyll/) | Previous Academic Pages / Jekyll site (archived) |
| [archive/assets/](archive/assets/) | Misc source assets kept for reference |

## Local development

Requires Node.js **>= 22.12**.

```bash
cd astro-site
npm install
npm run dev
```

Open http://localhost:4321

## Production build

```bash
cd astro-site
npm run build
npm run preview
```

## GitHub Pages setup

In the repository **Settings → Pages**, set **Build and deployment → Source** to **GitHub Actions**.

After pushing to `master`, the workflow builds `astro-site/dist` and publishes it.
