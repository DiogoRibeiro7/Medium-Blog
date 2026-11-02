"""Validate and visualize OMB federal outlays by broad function."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


DATA_FILE = Path(__file__).parent / "data" / "omb_outlays_by_function_1964_2018.csv"
YEARS = list(range(1964, 2019))
COLUMNS = ["year", "national_defense_pct_gdp", "human_resources_pct_gdp", "physical_resources_pct_gdp", "net_interest_pct_gdp", "other_functions_pct_gdp", "undistributed_offsets_pct_gdp", "total_outlays_pct_gdp"]


@dataclass(frozen=True)
class OutlayMix:
    """OMB fiscal-year superfunction shares of nominal GDP, in percent."""

    year: int
    defense: float
    human: float
    physical: float
    interest: float
    other: float
    offsets: float
    total: float

    def __post_init__(self) -> None:
        if type(self.year) is not int or self.year not in YEARS:
            raise ValueError("Year outside frozen fiscal window")
        values = (self.defense, self.human, self.physical, self.interest, self.other, self.offsets, self.total)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise TypeError("Every share must be finite and numeric")
        if any(v < 0 for v in values[:5]) or self.total < 0 or self.offsets > 0:
            raise ValueError("Unexpected superfunction sign")
        if abs(sum(values[:6]) - self.total) > 0.301:
            raise ValueError("Outlay categories fail the OMB rounding check")


def load_outlays(path: Path = DATA_FILE) -> list[OutlayMix]:
    """Read the six superfunctions and total, preserving the source precision."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Unexpected OMB outlay schema")
        rows = [OutlayMix(int(r["year"]), *(float(r[column]) for column in COLUMNS[1:])) for r in reader]
    if [r.year for r in rows] != YEARS:
        raise ValueError("Expected 55 consecutive unique fiscal years")
    return rows


def render_outlays(rows: list[OutlayMix]) -> str:
    """Draw four main functions; retain other and offsets in the CSV."""
    if [r.year for r in rows] != YEARS:
        raise ValueError("Expected ordered complete 1964–2018 observations")
    left, right, top, bottom = 95., 1010., 110., 444.

    def px(year: int) -> float:
        return left+(year-1964)*(right-left)/54

    def py(value: float) -> float:
        return bottom-value*(bottom-top)/18

    series = [("Human resources", "#378466", [r.human for r in rows]), ("National defense", "#245f91", [r.defense for r in rows]), ("Net interest", "#b75b3b", [r.interest for r in rows]), ("Physical resources", "#8060a7", [r.physical for r in rows])]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 650" role="img" aria-labelledby="title desc">',
        '<title id="title">U.S. federal outlays by broad function, 1964–2018</title>',
        '<desc id="desc">OMB fiscal 2027 historical Table 3.1; four selected broad federal outlay functions as percent of GDP; other functions and negative offsetting receipts are in the linked CSV.</desc>',
        '<rect width="1100" height="650" fill="white"/>',
        '<text x="550" y="38" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">What did federal spending finance?</text>',
        '<text x="550" y="66" text-anchor="middle" font-family="sans-serif" font-size="14">OMB FY2027 historical Table 3.1 · fiscal 1964–2018 · share of nominal GDP</text>',
    ]
    for value in (0, 3, 6, 9, 12, 15, 18):
        y = py(value)
        svg.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#ddd"/>')
        svg.append(f'<text x="82" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{value}%</text>')
    for year in (1964, 1974, 1984, 1994, 2004, 2014, 2018):
        svg.append(f'<text x="{px(year):.2f}" y="469" text-anchor="middle" font-family="sans-serif" font-size="13">{year}</text>')
    for title, color, values in series:
        points = " ".join(f"{px(r.year):.2f},{py(value):.2f}" for r, value in zip(rows, values, strict=True))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    for index, (title, color, _) in enumerate(series):
        x, y = 120+(index%2)*500, 513+(index//2)*25
        svg.append(f'<line x1="{x}" x2="{x+36}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        svg.append(f'<text x="{x+45}" y="{y+5}" font-family="sans-serif" font-size="14">{title}</text>')
    svg.extend([
        '<text x="550" y="576" text-anchor="middle" font-family="sans-serif" font-size="14">Other functions and negative undistributed offsetting receipts remain in the CSV.</text>',
        '<text x="550" y="602" text-anchor="middle" font-family="sans-serif" font-size="14">Functional shares are descriptive; they do not identify tax-law effects or policy opportunity costs.</text>',
        '<text x="550" y="629" text-anchor="middle" font-family="sans-serif" font-size="12">Source: OMB, FY2027 Historical Tables, Table 3.1 (2026 vintage).</text>',
        '</svg>',
    ])
    return "\n".join(svg)+"\n"


def main() -> None:
    """Regenerate the versioned spending-by-function figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "outlay_composition.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_outlays(load_outlays()), encoding="utf-8")


if __name__ == "__main__":
    main()
