# Project page

Jekyll site served by GitHub Pages (no build step needed — GitHub builds on push).

## Layout

| File | What it holds |
|---|---|
| `_data/paper.yml` | Title, venue, authors, links, abstract, footer — **edit this first for a new paper** |
| `index.html` | Section order only |
| `_includes/sections/*.html` | One file per section (header, abstract, method, video, code) |
| `_includes/method/overview.html` | Clickable overview figure (SVG); blocks with `data-step` select a demo step |
| `_includes/method/steps.html` | Title + description per demo step |
| `_includes/method/algorithms.html` | Pseudo-code per step; `data-l` ids are the lines the animation highlights |
| `assets/js/wgr-demo.js` | Method animation (scene, planner simulation, drawing) |
| `_layouts/default.html` | Shared frame: `<head>`, theme toggle, footer |
| `assets/css/style.css` | All styles; light/dark colors are the tokens at the top |
| `assets/js/theme.js` | Light/dark toggle (default: light) |

## New paper

1. Copy this repo.
2. Edit `_data/paper.yml`.
3. Edit or replace the section files in `_includes/sections/`; add/remove them in `index.html`.
4. If the method demo doesn't fit the new paper, drop `method/` includes and `wgr-demo.js`, or adapt them.

`others/` is local-only (excluded from the site and git) — keep paper drafts there.

## Local preview

Needs Ruby ≥ 3.0 (`brew install ruby`), then:

```sh
gem install jekyll
jekyll serve        # http://localhost:4000
```

Opening `index.html` directly in a browser will not work — the page has to be built.
