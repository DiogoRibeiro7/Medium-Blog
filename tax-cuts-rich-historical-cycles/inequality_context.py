"""Validate and visualize the Census household money-income Gini series."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


DATA_FILE = Path(__file__).parent / "data" / "household_gini_1967_2018.csv"


@dataclass(frozen=True)
class GiniPoint:
    """Census CPS pretax household money-income Gini for a calendar year."""

    year: int
    household_money_income_gini: float

    def __post_init__(self) -> None:
        if type(self.year) is not int or not 1967 <= self.year <= 2018:
            raise ValueError("Year outside the Census comparison window")
        if not isinstance(self.household_money_income_gini, (float, int)) or not math.isfinite(self.household_money_income_gini):
            raise TypeError("Gini must be finite and numeric")
        if not 0 <= self.household_money_income_gini <= 1:
            raise ValueError("Gini must lie in [0, 1]")


def load_gini(path: Path = DATA_FILE) -> list[GiniPoint]:
    """Load a frozen complete annual series without bridging missing years."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["year", "household_money_income_gini"]:
            raise ValueError("Unexpected Gini data schema")
        rows = [
            GiniPoint(int(row["year"]), float(row["household_money_income_gini"]))
            for row in reader
        ]
    if [row.year for row in rows] != list(range(1967, 2019)):
        raise ValueError("Expected 52 consecutive unique calendar years")
    return rows


def render_gini(rows: list[GiniPoint]) -> str:
    """Draw two segments to expose the 1993 CPS comparability break."""
    if [row.year for row in rows] != list(range(1967, 2019)):
        raise ValueError("Expected complete 1967–2018 series")
    left, right, top, bottom = 100., 1000., 100., 465.

    def px(year: int) -> float:
        return left + (year - 1967) * (right - left) / 51

    def py(value: float) -> float:
        return bottom - (value - 0.35) * (bottom - top) / 0.20

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 650" role="img" aria-labelledby="title desc">',
        '<title id="title">U.S. Census household pretax money-income Gini, 1967–2018</title>',
        '<desc id="desc">Annual descriptive series plotted in two segments across the 1993 CPS survey break.</desc>',
        '<rect width="1100" height="650" fill="white"/>',
        '<text x="550" y="39" text-anchor="middle" font-size="22" font-family="sans-serif" font-weight="bold">Household money-income inequality, 1967–2018</text>',
        '<text x="550" y="64" text-anchor="middle" font-size="14" font-family="sans-serif">Census CPS / FRED GINIALLRH · pretax money income · Gini ratio</text>',
    ]
    for value in (0.35, 0.40, 0.45, 0.50, 0.55):
        y = py(value)
        svg.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#ddd"/>')
        svg.append(f'<text x="88" y="{y+5:.2f}" font-family="sans-serif" text-anchor="end" font-size="13">{value:.2f}</text>')
    for year in (1967, 1977, 1987, 1997, 2007, 2017):
        svg.append(f'<text x="{px(year):.2f}" y="496" text-anchor="middle" font-size="13" font-family="sans-serif">{year}</text>')
    for segment in (rows[:26], rows[26:]):
        points = " ".join(f"{px(row.year):.2f},{py(row.household_money_income_gini):.2f}" for row in segment)
        svg.append(f'<polyline points="{points}" fill="none" stroke="#245f99" stroke-width="3"/>')
    break_x = (px(1992) + px(1993)) / 2
    svg.append(f'<line x1="{break_x:.2f}" x2="{break_x:.2f}" y1="90" y2="470" stroke="#b54a46" stroke-width="2" stroke-dasharray="6,5"/>')
    svg.append(f'<text x="{break_x+12:.2f}" y="115" font-size="13" font-family="sans-serif" fill="#933">1993 survey redesign / discontinuity</text>')
    svg.extend([
        '<text x="550" y="543" text-anchor="middle" font-size="13" font-family="sans-serif">The jump between 1992 and 1993 is not interpreted as an economic change.</text>',
        '<text x="550" y="567" text-anchor="middle" font-size="13" font-family="sans-serif">Money income excludes some capital gains and noncash transfers; not after-tax resources or wealth.</text>',
        '<text x="550" y="591" text-anchor="middle" font-size="13" font-family="sans-serif">Descriptive series only: this plot does not estimate tax-policy effects.</text>',
        '</svg>',
    ])
    return "\n".join(svg) + "\n"


def main() -> None:
    """Regenerate the versioned source-linked figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "household_gini.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_gini(load_gini()), encoding="utf-8")


if __name__ == "__main__":
    main()
