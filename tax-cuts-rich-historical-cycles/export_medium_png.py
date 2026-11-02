"""Render tracked SVG evidence charts as Medium-uploadable PNGs.

SVG is the analytical source of truth; PNG is a derived publication artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

FIGURES: tuple[str, ...] = (
    "sensitivity",
    "historical_top_rate",
    "tax_rate_and_receipts",
    "growth_and_investment",
    "household_gini",
    "cbo_2018_distribution",
    "nonresidential_investment",
    "fiscal_balance_public_debt",
    "revenue_outlays",
    "receipt_composition",
    "outlay_composition",
)
DEFAULT_ROOT: Path = Path(__file__).resolve().parent


def render_figures(root: Path, output_dir: Path, width: int = 2200) -> list[Path]:
    """Render all curated SVG files to PNG at the requested pixel width.

    Raises ValueError for a nonpositive width and FileNotFoundError for missing SVGs.
    CairoSVG is imported only when publication export is requested.
    """
    if width <= 0:
        raise ValueError("PNG width must be positive")
    import cairosvg  # Optional dependency, installed only in the export CI step.

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for stem in FIGURES:
        svg_path: Path = root / "results" / f"{stem}.svg"
        if not svg_path.is_file():
            raise FileNotFoundError(f"Missing tracked figure: {svg_path}")
        png_path: Path = output_dir / f"{stem}.png"
        cairosvg.svg2png(
            url=str(svg_path), write_to=str(png_path), output_width=width
        )
        if not png_path.is_file() or png_path.stat().st_size == 0:
            raise RuntimeError(f"Empty raster output: {png_path}")
        rendered.append(png_path)
    return rendered


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI PNG exporter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "publication_png")
    parser.add_argument("--width", type=int, default=2200)
    args = parser.parse_args(argv)
    for png in render_figures(DEFAULT_ROOT, args.out, args.width):
        print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
