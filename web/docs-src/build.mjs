// saetta.dev/docs: renders docs/*.md, ordered by docs/nav.json, into static pages under web/docs.
//
//   node web/docs-src/build.mjs           write web/docs/** and web/sitemap.xml
//   node web/docs-src/build.mjs --check   exit 1 if any of them is out of date (writes nothing)
//
// nav.json is the only list of pages: { "sections": [ { "title", "pages": [ { "slug", "title",
// "file"? } ] } ] }. Slug "index" is /docs/, any other slug is /docs/<slug>/ from docs/<slug>.md
// ("file" names another source, relative to docs/). A page whose file does not exist yet renders
// a "being written" placeholder. Relative links to a page's .md file become its route, links into
// web/ become site paths, and links to any other file in the repository point at it on GitHub.
// The header and footer are the main page's (web/index.html), with its in-page links made
// absolute, so the docs keep the site's navigation and theme toggle.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Marked } from "marked";
import GithubSlugger from "github-slugger";
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import c from "highlight.js/lib/languages/c";
import diff from "highlight.js/lib/languages/diff";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(HERE, "..");
const ROOT = path.resolve(WEB, "..");
const DOCS = path.join(ROOT, "docs");
const OUT = path.join(WEB, "docs");
const SITE = "https://saetta.dev";
const REPO = "https://github.com/antonellof/saetta";
const CHECK = process.argv.includes("--check");

for (const [name, lang] of Object.entries({ bash, c, diff, javascript, json, markdown, python, typescript, xml })) {
  hljs.registerLanguage(name, lang);
}
const ALIASES = { sh: "bash", shell: "bash", zsh: "bash", console: "bash", js: "javascript", mjs: "javascript", ts: "typescript", html: "xml", jsonl: "json", md: "markdown", py: "python" };
const LABELS = { bash: "shell", javascript: "js", typescript: "ts", xml: "html" };

// ---- pages -----------------------------------------------------------------------------------

const nav = JSON.parse(fs.readFileSync(path.join(DOCS, "nav.json"), "utf8")).sections;
const pages = [];
for (const section of nav) {
  for (const p of section.pages) {
    const file = path.posix.normalize(path.posix.join("docs", p.file ?? `${p.slug}.md`)); // repo-relative
    const route = p.slug === "index" ? "/docs/" : `/docs/${p.slug}/`;
    const out = p.slug === "index" ? "index.html" : `${p.slug}/index.html`;
    pages.push({ ...p, section: section.title, file, route, out });
  }
}
const byFile = new Map(pages.map((p) => [p.file, p]));

// ---- links -----------------------------------------------------------------------------------

function normalise(p) {
  const out = [];
  for (const part of p.split("/")) {
    if (part === "..") out.pop();
    else if (part && part !== ".") out.push(part);
  }
  return out.join("/");
}

// A link in the Markdown file `from` (repo-relative) as the site uses it.
function resolveHref(href, from, { image = false } = {}) {
  if (!href || /^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("#") || href.startsWith("//")) return href;
  const i = href.indexOf("#");
  const raw = i < 0 ? href : href.slice(0, i);
  const hash = i < 0 ? "" : href.slice(i);
  const repoPath = raw.startsWith("/") ? normalise(raw.slice(1)) : normalise(path.posix.join(path.posix.dirname(from), raw));
  const page = byFile.get(repoPath);
  if (page) return page.route + hash;
  if (repoPath === "web" || repoPath.startsWith("web/")) return "/" + repoPath.slice(4) + hash;
  if (image) return `${REPO}/raw/main/${repoPath}`;
  const full = path.join(ROOT, repoPath);
  const isDir = fs.existsSync(full) ? fs.statSync(full).isDirectory() : !path.posix.extname(repoPath);
  const kind = isDir ? "tree" : "blob";
  return `${REPO}/${kind}/main/${repoPath}${hash}`;
}

// ---- Markdown --------------------------------------------------------------------------------

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const textOf = (html) =>
  html.replace(/<[^>]+>/g, "").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").trim();

function render(md, from) {
  const slugger = new GithubSlugger();
  const headings = [];
  let title = null;
  const marked = new Marked({
    gfm: true,
    renderer: {
      heading({ tokens, depth }) {
        const inner = this.parser.parseInline(tokens);
        const text = textOf(inner);
        if (depth === 1 && title === null) {
          title = text;
          return `<h1>${inner}</h1>\n`;
        }
        const id = slugger.slug(text);
        if (depth === 2 || depth === 3) headings.push({ depth, text, id });
        return `<h${depth} id="${id}"><a class="anchor" href="#${id}" aria-hidden="true" tabindex="-1">#</a>${inner}</h${depth}>\n`;
      },
      code({ text, lang }) {
        const name = (lang || "").trim().split(/\s+/)[0].toLowerCase();
        const key = ALIASES[name] ?? name;
        const body = key && hljs.getLanguage(key) ? hljs.highlight(text, { language: key, ignoreIllegals: true }).value : esc(text);
        const label = name ? LABELS[key] ?? key : "text";
        return `<div class="code doc-code"><div class="code-head"><span>${esc(label)}</span><button class="copy" type="button">Copy</button></div><pre><code>${body}</code></pre></div>\n`;
      },
      link({ href, title: t, tokens }) {
        const to = resolveHref(href, from);
        const ext = /^https?:/.test(to);
        return `<a href="${esc(to)}"${t ? ` title="${esc(t)}"` : ""}${ext ? ' rel="noopener"' : ""}>${this.parser.parseInline(tokens)}</a>`;
      },
      image({ href, title: t, text }) {
        return `<img src="${esc(resolveHref(href, from, { image: true }))}" alt="${esc(text)}"${t ? ` title="${esc(t)}"` : ""} loading="lazy">`;
      },
      table(token) {
        const cell = (c) => {
          const align = c.align ? ` style="text-align:${c.align}"` : "";
          const tag = c.header ? "th" : "td";
          return `<${tag}${align}>${this.parser.parseInline(c.tokens)}</${tag}>`;
        };
        const head = `<tr>${token.header.map(cell).join("")}</tr>`;
        const rows = token.rows.map((r) => `<tr>${r.map(cell).join("")}</tr>`).join("\n");
        return `<div class="doc-table"><table><thead>${head}</thead><tbody>${rows}</tbody></table></div>\n`;
      },
    },
  });
  const html = marked.parse(md);
  return { html, headings, title };
}

// The first paragraph's text, for the description meta tag.
function describe(md) {
  for (const block of md.split(/\n\s*\n/)) {
    const b = block.trim();
    if (!b || /^(#|```|\||<|[-*] |\d+\. )/.test(b)) continue;
    const t = b.replace(/!\[[^\]]*\]\([^)]*\)/g, "").replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/[`*_]/g, "").replace(/\s+/g, " ").trim();
    return t.length > 200 ? t.slice(0, 197).replace(/\s+\S*$/, "") + "..." : t;
  }
  return "";
}

// ---- the site's header and footer -------------------------------------------------------------

const indexHtml = fs.readFileSync(path.join(WEB, "index.html"), "utf8");
const absolute = (html) => html.replace(/href="#top"/g, 'href="/"').replace(/href="#([^"]+)"/g, 'href="/#$1"');
const headerHtml = absolute(indexHtml.match(/<header class="nav">[\s\S]*?<\/header>/)[0]).replace(
  /<a href="\/docs\/">Docs<\/a>/,
  '<a href="/docs/" aria-current="page">Docs</a>',
);
const footerHtml = absolute(indexHtml.match(/<footer class="footer">[\s\S]*?<\/footer>/)[0]);
const themeScript = indexHtml.match(/<script>\s*\/\/ theme before paint[\s\S]*?<\/script>/)[0];
const fontLinks = indexHtml.match(/<link rel="preconnect"[\s\S]*?<link rel="stylesheet" href="\/styles.css">/)[0];

// ---- page template ---------------------------------------------------------------------------

function sidebar(current) {
  return nav
    .map(
      (s) =>
        `<div class="ds-group"><p class="ds-title">${esc(s.title)}</p><ul>` +
        s.pages
          .map((p) => {
            const pg = pages.find((x) => x.slug === p.slug);
            const on = pg === current ? ' aria-current="page"' : "";
            return `<li><a href="${pg.route}"${on}>${esc(p.title)}</a></li>`;
          })
          .join("") +
        `</ul></div>`,
    )
    .join("\n");
}

function toc(headings) {
  if (headings.length < 2) return "";
  return (
    `<nav class="doc-toc" aria-label="On this page"><p class="ds-title">On this page</p><ul>` +
    headings.map((h) => `<li class="d${h.depth}"><a href="#${h.id}">${esc(h.text)}</a></li>`).join("") +
    `</ul></nav>`
  );
}

function pageHtml(page, i) {
  const src = path.join(ROOT, page.file);
  const exists = fs.existsSync(src);
  const md = exists ? fs.readFileSync(src, "utf8") : "";
  let { html, headings, title } = exists ? render(md, page.file) : { html: "", headings: [], title: null };
  if (!exists) {
    html = `<h1>${esc(page.title)}</h1>\n<p class="doc-wip">This page is being written. Until it is here, the <a href="${REPO}#readme">README</a> covers the essentials.</p>\n`;
  } else if (title === null) {
    html = `<h1>${esc(page.title)}</h1>\n` + html;
  }
  const description = (exists && describe(md)) || `${page.title}: Saetta documentation.`;
  const fullTitle = page.slug === "index" ? "Saetta documentation" : `${page.title} · Saetta docs`;
  const url = SITE + page.route;
  const prev = pages[i - 1];
  const next = pages[i + 1];
  const pager =
    `<nav class="doc-pager" aria-label="Previous and next">` +
    (prev ? `<a class="prev" href="${prev.route}"><span>Previous</span>${esc(prev.title)}</a>` : "<span></span>") +
    (next ? `<a class="next" href="${next.route}"><span>Next</span>${esc(next.title)}</a>` : "<span></span>") +
    `</nav>`;
  const edit = exists ? `<p class="doc-edit"><a href="${REPO}/blob/main/${page.file}" rel="noopener">Edit this page on GitHub</a></p>` : "";
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${esc(fullTitle)}</title>
  <meta name="description" content="${esc(description)}">
  <link rel="canonical" href="${url}">
  <meta name="robots" content="index, follow">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <meta property="og:type" content="article">
  <meta property="og:site_name" content="Saetta">
  <meta property="og:title" content="${esc(fullTitle)}">
  <meta property="og:description" content="${esc(description)}">
  <meta property="og:image" content="https://www.saetta.dev/assets/og-card.png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="Saetta, local coding agent">
  <meta property="og:url" content="${url}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${esc(fullTitle)}">
  <meta name="twitter:description" content="${esc(description)}">
  <meta name="twitter:image" content="https://www.saetta.dev/assets/og-card.png">
  ${themeScript}
  ${fontLinks}
  <link rel="stylesheet" href="/docs.css">
</head>
<body class="docs">
  <!-- generated by web/docs-src/build.mjs from ${page.file}: edit the Markdown, not this file -->
  ${headerHtml}

  <div class="ds-bar"><button class="ds-open" type="button" aria-expanded="false" aria-controls="ds-nav">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
    Menu</button><span class="ds-crumb">${esc(page.section)} <span aria-hidden="true">/</span> ${esc(page.title)}</span></div>
  <div class="docs-shell">
    <aside class="ds" id="ds-nav" aria-label="Documentation">
      <div class="ds-head"><input class="ds-filter" type="search" placeholder="Filter pages" aria-label="Filter pages"><button class="ds-close icon-btn" type="button" aria-label="Close menu">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg></button></div>
      ${sidebar(page)}
      <p class="ds-none" hidden>No matching pages.</p>
    </aside>
    <main class="doc" id="top">
      <article class="doc-body">
${html}      </article>
      ${edit}
      ${pager}
    </main>
    ${toc(headings)}
  </div>
  <div class="ds-scrim" hidden></div>

  ${footerHtml}

  <script src="/docs.js" defer></script>
</body>
</html>
`;
}

// ---- sitemap ---------------------------------------------------------------------------------

function sitemap() {
  const current = fs.readFileSync(path.join(WEB, "sitemap.xml"), "utf8");
  const rootLine = current.match(/^\s*<url><loc>https:\/\/saetta\.dev\/<\/loc>.*<\/url>$/m)?.[0].trim() ?? `<url><loc>${SITE}/</loc></url>`;
  const lines = [rootLine, ...pages.map((p) => `<url><loc>${SITE}${p.route}</loc></url>`)];
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${lines.map((l) => "  " + l).join("\n")}\n</urlset>\n`;
}

// ---- write or check --------------------------------------------------------------------------

const outputs = new Map();
pages.forEach((p, i) => outputs.set(path.join(OUT, p.out), pageHtml(p, i)));
outputs.set(path.join(WEB, "sitemap.xml"), sitemap());

function walk(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)]));
}
const extra = walk(OUT).filter((f) => !outputs.has(f));
const rel = (f) => path.relative(ROOT, f);

if (CHECK) {
  const stale = [...outputs].filter(([f, s]) => !fs.existsSync(f) || fs.readFileSync(f, "utf8") !== s).map(([f]) => f);
  if (stale.length || extra.length) {
    for (const f of stale) console.error(`stale: ${rel(f)}`);
    for (const f of extra) console.error(`not generated: ${rel(f)}`);
    console.error("web/docs is out of date: run node web/docs-src/build.mjs");
    process.exit(1);
  }
  console.log(`web/docs is up to date (${pages.length} pages)`);
} else {
  for (const f of extra) fs.rmSync(f);
  for (const [f, s] of outputs) {
    fs.mkdirSync(path.dirname(f), { recursive: true });
    if (!fs.existsSync(f) || fs.readFileSync(f, "utf8") !== s) fs.writeFileSync(f, s);
  }
  // folders left empty by a removed page
  for (const d of fs.readdirSync(OUT, { withFileTypes: true })) {
    const p = path.join(OUT, d.name);
    if (d.isDirectory() && walk(p).length === 0) fs.rmSync(p, { recursive: true });
  }
  const missing = pages.filter((p) => !fs.existsSync(path.join(ROOT, p.file)));
  console.log(`wrote ${pages.length} pages to web/docs` + (missing.length ? ` (${missing.length} being written: ${missing.map((p) => p.slug).join(", ")})` : ""));
}
