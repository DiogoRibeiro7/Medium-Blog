"""Render the displayed equations of the Medium articles as PNG images.

Medium does not render LaTeX, so each displayed equation in a manuscript is
replaced by an image link, and its LaTeX source is kept in ``manifest.json``:

    {"<series-or-theme>/<article-folder>": [{"name": "eq01", "tex": "..."}, ...], ...}

For every article this script compiles one ``standalone`` LaTeX document with
one page per equation, rasterises it at a fixed resolution and centres each
equation on a canvas of fixed width. Every image therefore has the same scale
once Medium fits it to the column, and the maths appears at about the size of
the body text. Equation tags (``\\tag{2}``) are written to the right as ``(2)``.

Usage:
    python render_equations.py                 # render everything
    python render_equations.py SLUG [...]      # render selected articles

Requires pdflatex (with amsmath, amssymb, mathtools, bm, xcolor, standalone),
pdftoppm (poppler) and Pillow.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT: Path = Path(__file__).resolve().parent
MANIFEST: Path = ROOT / "manifest.json"
DPI: int = 300            # 12 pt at 300 dpi gives an em of about 50 px
CANVAS_WIDTH: int = 1600  # Medium fits this to the column, so all equations share one scale
SIDE_MARGIN: int = 40
VERTICAL_PAD: int = 28
INK: str = "292929"       # Medium's body-text colour

PREAMBLE = r"""\documentclass[12pt,border=4pt,multi=eqn]{standalone}
\usepackage{amsmath,amssymb,mathtools,bm}
\usepackage{xcolor}
\newenvironment{eqn}{\color[HTML]{%s}}{}
\begin{document}
""" % INK


def prepare(tex: str) -> str:
    """Turn one manuscript equation into a page body for the standalone document."""
    tag = re.search(r"\\tag\*?\{([^}]*)\}", tex)
    body = re.sub(r"\\tag\*?\{[^}]*\}|\\label\{[^}]*\}|\\nonumber|\\notag", "", tex).strip()
    body = body.replace("€", r"\text{\texteuro}")
    body = body.rstrip()
    # top-level line breaks need an environment that accepts them
    depth = 0
    top_level_break = False
    for token in re.finditer(r"\\begin\{[^}]*\}|\\end\{[^}]*\}|\\\\", body):
        t = token.group(0)
        if t.startswith("\\begin"):
            depth += 1
        elif t.startswith("\\end"):
            depth -= 1
        elif depth == 0:
            top_level_break = True
    if top_level_break:
        env = "aligned" if "&" in body else "gathered"
        body = f"\\begin{{{env}}}{body}\\end{{{env}}}"
    if tag:
        body = f"{body}\\qquad\\text{{({tag.group(1)})}}"
    return f"\\begin{{eqn}}$\\displaystyle {body}$\\end{{eqn}}\n"


def compile_pages(bodies: list[str], workdir: Path) -> list[Path]:
    """Compile the pages and return one PNG per page, in order."""
    src = workdir / "eqs.tex"
    src.write_text(PREAMBLE + "".join(bodies) + "\\end{document}\n", encoding="utf-8")
    run = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", src.name],
        cwd=workdir, capture_output=True, text=True, errors="replace",
    )
    if run.returncode != 0:
        tail = "\n".join(line for line in run.stdout.splitlines() if line.startswith("!") or line.startswith("l."))
        raise RuntimeError(tail or run.stdout[-2000:])
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "eqs.pdf", "page"], cwd=workdir, check=True)
    pages = sorted(workdir.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[1]))
    if len(pages) != len(bodies):
        raise RuntimeError(f"expected {len(bodies)} pages, got {len(pages)}")
    return pages


def place(page: Path, target: Path) -> None:
    """Crop the page to its ink and centre it on the fixed-width canvas."""
    img = Image.open(page).convert("RGB")
    grey = img.convert("L").point(lambda v: 255 if v < 250 else 0)
    box = grey.getbbox()
    if box:
        img = img.crop(box)
    usable = CANVAS_WIDTH - 2 * SIDE_MARGIN
    if img.width > usable:
        scale = usable / img.width
        img = img.resize((usable, max(1, round(img.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (CANVAS_WIDTH, img.height + 2 * VERTICAL_PAD), "white")
    canvas.paste(img, ((CANVAS_WIDTH - img.width) // 2, VERTICAL_PAD))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, optimize=True)


def render_article(slug: str, equations: list[dict]) -> None:
    """Render all equations of one article, falling back to one compile per equation on error."""
    out_dir = ROOT / slug
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        bodies = [prepare(eq["tex"]) for eq in equations]
        try:
            pages = compile_pages(bodies, work)
            for eq, page in zip(equations, pages, strict=True):
                place(page, out_dir / f"{eq['name']}.png")
            return
        except RuntimeError:
            pass
        failures = []
        for eq, body in zip(equations, bodies, strict=True):
            sub = work / eq["name"]
            sub.mkdir()
            try:
                (page,) = compile_pages([body], sub)
                place(page, out_dir / f"{eq['name']}.png")
            except RuntimeError as err:
                failures.append(f"{slug}/{eq['name']}: {err}")
        if failures:
            raise RuntimeError("\n".join(failures))


def main(argv: list[str]) -> int:
    for tool in ("pdflatex", "pdftoppm"):
        if shutil.which(tool) is None:
            print(f"missing {tool}", file=sys.stderr)
            return 2
    manifest: dict[str, list[dict]] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    slugs = argv or sorted(manifest)
    errors = []
    for slug in slugs:
        try:
            render_article(slug, manifest[slug])
            print(f"{slug}: {len(manifest[slug])} equations")
        except RuntimeError as err:
            errors.append(str(err))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
