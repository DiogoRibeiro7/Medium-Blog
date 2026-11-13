# Displayed equations for the Medium articles

Medium does not render LaTeX. The displayed equations of the Medium manuscripts are therefore published as images, and this folder holds them together with their LaTeX source.

- `manifest.json` maps each article (`<series-or-theme>/<article-folder>`) to its equations, in order of appearance: `{"name": "eq01", "tex": "..."}`.
- `render_equations.py` compiles each article's equations with `pdflatex` (one `standalone` page per equation), rasterises them at 300 dpi and centres each one on a 1600-pixel canvas. Because every image has the same width, Medium scales them all by the same factor, so the maths appears at about the size of the body text. Tags such as `\tag{2}` are printed to the right as `(2)`.
- `<series-or-theme>/<article-folder>/eqNN.png` are the rendered images. Manuscripts link to them through `raw.githubusercontent.com` at a pinned commit.

Inline maths in the manuscripts is written directly as Unicode text, so it needs no images.

## Run

Requirements: `pdflatex` with amsmath, amssymb, mathtools, bm, xcolor and standalone; `pdftoppm` (poppler); Python 3.11+ with Pillow.

```bash
python render_equations.py                      # every article
python render_equations.py housing/urban-affordability-cascade   # one article
```

The CI workflow renders every equation in the manifest and fails if any does not compile or any image is missing.
