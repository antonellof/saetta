# Saetta docs

The Markdown files in this folder are the documentation, published at [saetta.dev/docs](https://saetta.dev/docs/). They read on GitHub as they are; the site renders the same files.

## How the site is built

- [`nav.json`](nav.json) lists the pages, in sections, in the order of the sidebar and of the previous/next links. Slug `index` is `/docs/`; slug `mcp` is `/docs/mcp/`, from `docs/mcp.md`. A page can name another source with `"file"` (the changelog is `../CHANGELOG.md`). This README is not in the list, so it is not published.
- A `nav.json` entry whose file does not exist yet renders a "This page is being written" placeholder.
- Relative links between the `.md` files become site routes (`mcp.md#oauth-sign-in` is `/docs/mcp/#oauth-sign-in`). Links into `web/` (such as `../web/assets/code/tree.png`) become site paths (`/assets/code/tree.png`). Links to any other file in the repository point at it on GitHub.
- `h2` and `h3` headings feed the "On this page" list; their ids are GitHub's, so `#anchors` work on both.
- Fenced code blocks are highlighted at build time (highlight.js) and get a copy button.

The generator is [`web/docs-src/build.mjs`](../web/docs-src/build.mjs). It writes static pages to `web/docs/<slug>/index.html` with the main page's header and footer, `web/styles.css` and `web/docs.css`, and adds the pages to `web/sitemap.xml`. The site stays a static folder: Vercel serves `web/` with no build step, so the generated pages are committed.

## After editing a page

```sh
cd web/docs-src && npm ci && cd ../..     # once: marked, highlight.js, github-slugger
node web/docs-src/build.mjs               # regenerate web/docs and web/sitemap.xml
node web/docs-src/build.mjs --check       # exit 1 if web/docs is out of date
python3 -m http.server -d web 8000        # preview at http://localhost:8000/docs/
```

Commit the Markdown and the regenerated `web/docs` together; `--check` is the test that they match.

## Writing

Plain, specific sentences; examples as code blocks; every number as the README states it. Nothing internal: these pages describe the released program.
