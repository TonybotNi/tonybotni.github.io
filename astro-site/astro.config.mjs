// @ts-check
import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import sitemap from '@astrojs/sitemap';
import { visit } from 'unist-util-visit';

import tailwindcss from '@tailwindcss/vite';

/** Wraps every <table> in a <div class="table-container"> for overflow scrolling */
function rehypeWrapTables() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName === 'table' && parent && typeof index === 'number') {
        const wrapper = {
          type: 'element',
          tagName: 'div',
          properties: { className: ['table-container'] },
          children: [node],
        };
        parent.children.splice(index, 1, wrapper);
      }
    });
  };
}

// https://astro.build/config
export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeWrapTables],
  },
  build: {
    inlineStylesheets: 'always'
  },
  vite: {
    plugins: [tailwindcss()],
  },
  site: 'https://tonybotni.github.io',
  base: '/',
  integrations: [sitemap()],
});