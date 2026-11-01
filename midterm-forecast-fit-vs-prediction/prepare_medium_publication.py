"""Prepare a self-contained, publication-oriented copy of the frozen study.

The scientific source, its forecast lock, and numerical evidence are never
rewritten. Medium does not natively typeset LaTeX, so the release copy replaces
display equations with reproducible PNG images and short inline expressions
with readable Unicode. Building the PNGs requires a local LaTeX installation;
the CI check validates the committed release without installing TeX.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final


ROOT: Final[Path] = Path(__file__).resolve().parent
SOURCE: Final[Path] = ROOT / "article.md"
PACKAGE: Final[Path] = ROOT / "publication"
OUTPUT: Final[Path] = PACKAGE / "medium_ready.md"
EQUATIONS: Final[Path] = PACKAGE / "equations"
REPOSITORY: Final[str] = "DiogoRibeiro7/Medium-Blog"
ARTICLE_DIRECTORY: Final[str] = "midterm-forecast-fit-vs-prediction"
# A permanent GitHub revision containing the manuscript, figures and lock.
FROZEN_PUBLIC_REF: Final[str] = "3a409879187f3d6b4c98f4de247f1f413add8249"

DISPLAY_MATH: Final[re.Pattern[str]] = re.compile(r"(?ms)^\\\[\n(.*?)\n\\\]$")
INLINE_MATH: Final[re.Pattern[str]] = re.compile(r"\\\((.*?)\\\)")
FIGURE: Final[re.Pattern[str]] = re.compile(r"!\[([^\]]+)\]\((figures/[^)\s]+\.png)\)")
MARKDOWN_LINK: Final[re.Pattern[str]] = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
TABLE_FIGURES: Final[tuple[str, ...]] = (
    "historical-election-examples.png",
    "forward-validation-comparison.png",
    "estimator-sensitivity.png",
    "paired-resampling-comparison.png",
)
TABLE_FIGURE_REF: Final[str] = "e31ce5cd4b0ec9f64c4e2977168514785796393e"
REFERENCES: Final[re.Pattern[str]] = re.compile(r"(?m)^(\d+)\. .+")

# All inline math expressions in the locked original are explicit. Reject new
# expressions until a human chooses their publication-friendly wording.
INLINE_TEXT: Final[dict[str, str]] = {
    "R^2=0.8934": "R² = 0.8934",
    "R^2=0.8465": "R² = 0.8465",
    "R^2": "R²",
    "0\\leq R^2\\leq1": "0 ≤ R² ≤ 1",
    "R^2=r_{xy}^2": "R² = rₓᵧ²",
    "R^2=0.89": "R² = 0.89",
    "|r_{xy}|\\simeq0.943": "|rₓᵧ| ≈ 0.943",
    "t": "t",
    "Y_t": "Yₜ",
    "G_t": "Gₜ",
    "S_t": "Sₜ",
    "X": "X",
    "y\\in\\mathbb R^{18}": "y ∈ ℝ¹⁸",
    "X^\\top X": "XᵀX",
    "G_{2026}=-8": "G₂₀₂₆ = −8",
    "S_{2026}=220": "S₂₀₂₆ = 220",
    "18-3=15": "18 − 3 = 15",
    "Y_t-\\widehat Y_t": "Yₜ − Ŷₜ",
    "i": "i",
    "\\widehat y_{(-i),i}": "ŷ₍₋ᵢ₎,ᵢ",
    "h_{ii}": "hᵢᵢ",
    "H=X(X^\\top X)^{-1}X^\\top": "H = X(XᵀX)⁻¹Xᵀ",
    "\\operatorname{E}(Y\\mid G,S)": "E(Y | G, S)",
    "x_0=(1,G_0,S_0)^\\top": "x₀ = (1, G₀, S₀)ᵀ",
    "\\widehat\\sigma^2": "σ̂²",
    "n-p": "n − p",
    "n=18": "n = 18",
    "p=3": "p = 3",
    "q_{\\alpha/2,\\,n-p}": "q(α/2, n − p)",
    "(1-\\alpha)": "(1 − α)",
    "x_0": "x₀",
    "\\widehat y_0=-32.32": "ŷ₀ = −32.32",
    "3/18=1/6": "3/18 = 1/6",
    "1-h_{ii}": "1 − hᵢᵢ",
    "e_i": "eᵢ",
    "106.998+2.70846G-0.534787S": "106.998 + 2.70846G − 0.534787S",
    "G": "G",
    "\\nu=4": "ν = 4",
    "\\alpha=2": "α = 2",
    "T": "T",
    "(\\sum w)^2/\\sum w^2": "(Σw)² / Σw²",
    "j": "j",
    "e_{j,t}=\\widehat Y_{j,t}-Y_t": "eⱼ,ₜ = Ŷⱼ,ₜ − Yₜ",
    "d_{j,t}": "dⱼ,ₜ",
}

EQUATION_CAPTIONS: Final[tuple[str, ...]] = (
    "Definition of historical R-squared",
    "Two-predictor linear model",
    "Least-squares coefficient solution",
    "Estimated House seat-change equation",
    "Forecast from the original 2026 inputs",
    "Leave-one-election-out root mean squared error",
    "Exact leave-one-out residual identity",
    "Model-based future-election prediction interval",
    "Cook's distance influence diagnostic",
    "Four-degree-of-freedom Student-t regression",
    "Student-t negative log likelihood",
    "Ridge penalized regression objective",
    "Recency-weighted least-squares objective",
    "Paired squared-error differential",
    "Exact paired mean squared-error identity",
)


def extract_equations(source: str) -> tuple[str, ...]:
    """Return display-math bodies, verifying the editorial equation count."""
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    equations = tuple(DISPLAY_MATH.findall(source))
    if len(equations) != len(EQUATION_CAPTIONS):
        raise ValueError(f"Expected {len(EQUATION_CAPTIONS)} display equations, found {len(equations)}")
    return equations


def build_release_markdown(source: str) -> str:
    """Transform the manuscript without modifying its frozen scientific source."""
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if not source.startswith("# The Forecast Is Not the Fit:"):
        raise ValueError("Unexpected article title")
    if "final pre-election manuscript" not in source:
        raise ValueError("The manuscript has not received editorial sign-off")
    if "31 October 2026" not in source or "1 November 2026" not in source:
        raise ValueError("The information and registration dates must remain visible")

    equations = extract_equations(source)
    if len(FIGURE.findall(source)) != 13 + len(TABLE_FIGURES):
        raise ValueError("Expected thirteen original evidence figures and four table images")

    equation_index = iter(range(1, len(equations) + 1))

    def replace_equation(match: re.Match[str]) -> str:
        index = next(equation_index)
        caption = EQUATION_CAPTIONS[index - 1]
        return f"![Equation {index:02d}: {caption}.](equations/equation_{index:02d}.png)"

    rendered = DISPLAY_MATH.sub(replace_equation, source)

    def replace_inline(match: re.Match[str]) -> str:
        raw = match.group(1)
        if raw not in INLINE_TEXT:
            raise ValueError(f"Unknown inline expression: {raw!r}")
        return INLINE_TEXT[raw]

    rendered = INLINE_MATH.sub(replace_inline, rendered)

    def replace_figure(match: re.Match[str]) -> str:
        alt, relative = match.groups()
        if not (ROOT / relative).is_file():
            raise ValueError(f"Missing figure: {relative}")
        # Pin the scientific figure to the existing merged pre-election revision.
        ref = TABLE_FIGURE_REF if Path(relative).name in TABLE_FIGURES else FROZEN_PUBLIC_REF
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{ref}/{ARTICLE_DIRECTORY}/{relative}"
        return f"![{alt}]({url})"

    rendered = FIGURE.sub(replace_figure, rendered)

    def replace_link(match: re.Match[str]) -> str:
        label, target = match.groups()
        if target.startswith(("https://", "http://", "mailto:")):
            return match.group(0)
        if not (ROOT / target).exists():
            raise ValueError(f"Missing reproducibility companion: {target}")
        kind = "tree" if (ROOT / target).is_dir() else "blob"
        url = f"https://github.com/{REPOSITORY}/{kind}/{FROZEN_PUBLIC_REF}/{ARTICLE_DIRECTORY}/{target}"
        return f"[{label}]({url})"

    rendered = MARKDOWN_LINK.sub(replace_link, rendered)
    editorial_line = next(
        (line for line in rendered.splitlines() if line.startswith("*Editorial status:")),
        None,
    )
    if editorial_line is None:
        raise ValueError("Missing editorial-status marker")
    rendered = rendered.replace(editorial_line, "*Pre-election publication edition prepared 1 November 2026. No 2026 outcome is used.*")
    if INLINE_MATH.search(rendered) or DISPLAY_MATH.search(rendered):
        raise ValueError("Unconverted LaTeX remains in the publication copy")
    return rendered


def render_equation_png(number: int, latex: str, destination: Path) -> None:
    """Typeset one display equation with a cropped white PDF-to-PNG export."""
    if not isinstance(number, int) or number < 1:
        raise ValueError("Equation number must be a positive integer")
    if not isinstance(latex, str) or not latex.strip():
        raise ValueError("Equation body must be nonempty LaTeX")
    source = (
        "\\documentclass[border=8pt]{standalone}\n"
        "\\usepackage{amsmath,amssymb}\n"
        "\\begin{document}\n$\\displaystyle\n"
        + latex
        + "\n$\n\\end{document}\n"
    )
    with tempfile.TemporaryDirectory(prefix="midterm-equation-") as temporary:
        working = Path(temporary)
        tex_file = working / "equation.tex"
        tex_file.write_text(source, encoding="utf-8")
        compiled = subprocess.run(
            ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", "equation.tex"],
            cwd=working,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if compiled.returncode:
            raise RuntimeError(f"Equation {number} did not compile:\n{compiled.stdout[-2500:]}")
        exported = subprocess.run(
            ["pdftoppm", "-f", "1", "-singlefile", "-r", "220", "-png", "equation.pdf", "equation"],
            cwd=working,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if exported.returncode:
            raise RuntimeError(f"Equation {number} PNG rendering failed: {exported.stderr}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(working / "equation.png", destination)


def check_release(source: str, rendered: str) -> tuple[int, int, int]:
    """Validate the publication text, bibliography, image assets and fidelity."""
    expected = build_release_markdown(source)
    if rendered != expected:
        raise ValueError("Publication Markdown is stale relative to the final manuscript")
    figures = FIGURE.findall(source)
    if len(set(path for _, path in figures)) != 13 + len(TABLE_FIGURES):
        raise ValueError("Figure references are missing or duplicated")
    if {Path(path).name for _, path in figures if Path(path).name in TABLE_FIGURES} != set(TABLE_FIGURES):
        raise ValueError("Required table images are missing")
    references = [int(item) for item in REFERENCES.findall(source)]
    if references != list(range(1, 20)):
        raise ValueError("Expected nineteen consecutively numbered references")
    for number in range(1, len(EQUATION_CAPTIONS) + 1):
        image_path = EQUATIONS / f"equation_{number:02d}.png"
        if not image_path.is_file() or image_path.stat().st_size < 500:
            raise ValueError(f"Equation image missing or too small: {image_path}")
        with image_path.open("rb") as binary:
            if binary.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Equation asset is not a PNG: {image_path}")
    word_count = len(rendered.split())
    if word_count < 8000:
        raise ValueError(f"Publication copy is too short: {word_count} words")
    return word_count, len(figures), len(EQUATION_CAPTIONS)


def main() -> int:
    """Build the publication copy, or verify its committed form in CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="Render equations and write the final Medium copy")
    mode.add_argument("--check", action="store_true", help="Check committed copy and images without TeX")
    args = parser.parse_args()

    manuscript = SOURCE.read_text(encoding="utf-8")
    publication = build_release_markdown(manuscript)
    if args.build:
        for index, equation in enumerate(extract_equations(manuscript), start=1):
            render_equation_png(index, equation, EQUATIONS / f"equation_{index:02d}.png")
        PACKAGE.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(publication, encoding="utf-8")
    if not OUTPUT.is_file():
        raise ValueError(f"Missing publication manuscript: {OUTPUT}")
    words, figures, equations = check_release(manuscript, OUTPUT.read_text(encoding="utf-8"))
    print(f"Medium copy verified: {words} words, {figures} figures, {equations} rendered equations, 19 references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
