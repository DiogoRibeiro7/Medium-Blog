"""Reproducible descriptive GDP growth and investment chronology."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "real_growth_private_investment_1964_2018.csv"


@dataclass(frozen=True)
class MacroPoint:
    """One BEA observation; both fields are percentages with different bases."""

    year: int
    real_gdp_growth_pct: float
    private_investment_share_pct: float

    def __post_init__(self) -> None:
        if type(self.year) is not int or not 1964 <= self.year <= 2018:
            raise ValueError("Year outside the evidence window")
        for item in (self.real_gdp_growth_pct, self.private_investment_share_pct):
            if not isinstance(item, (float, int)) or not math.isfinite(item):
                raise TypeError("Macroeconomic percentages must be finite numeric values")
        if not -30 < self.real_gdp_growth_pct < 30:
            raise ValueError("Implausible annual growth value")
        if not 0 < self.private_investment_share_pct < 100:
            raise ValueError("Investment share must lie in (0, 100)")


def load_macro(path: Path = DATA_FILE) -> list[MacroPoint]:
    """Read and structurally validate the frozen BEA evidence."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["year", "real_gdp_growth_pct", "private_investment_share_pct"]:
            raise ValueError("Unexpected macro source schema")
        rows = [
            MacroPoint(
                int(row["year"]),
                float(row["real_gdp_growth_pct"]),
                float(row["private_investment_share_pct"]),
            )
            for row in reader
        ]
    if [row.year for row in rows] != list(range(1964, 2019)):
        raise ValueError("Expected 55 unique, ordered, consecutive calendar years")
    return rows


def render_macro(rows: list[MacroPoint]) -> str:
    """Render two independent scale panels; deliberately fit no regression."""
    if [r.year for r in rows] != list(range(1964, 2019)):
        raise ValueError("Complete macro data required")
    left, right = 105.0, 985.0

    def xpos(year: int) -> float:
        return left + (year - 1964) * (right - left) / 54

    def ypos(value: float, low: float, high: float, top: float, bottom: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    result = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1090 755" role="img" aria-labelledby="title desc">',
        '<title id="title">Real GDP growth and private domestic investment share, 1964–2018</title>',
        '<desc id="desc">Two revised BEA annual national-account series shown descriptively with distinct vertical scales.</desc>',
        '<rect width="1090" height="755" fill="white"/>',
        '<text x="545" y="38" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Output growth and private investment, 1964–2018</text>',
        '<text x="545" y="61" text-anchor="middle" font-family="sans-serif" font-size="13">BEA annual accounts · no tax-policy counterfactual or causal attribution</text>',
    ]
    for heading, values, low, high, top, bottom, colour in (
        ("Real GDP annual growth (%)", [r.real_gdp_growth_pct for r in rows], -4., 8., 110., 315., "#245f99"),
        ("Gross private domestic investment / GDP (%)", [r.private_investment_share_pct for r in rows], 10., 22., 390., 595., "#9e5c29"),
    ):
        result.append(f'<text x="{left}" y="{top-20}" font-family="sans-serif" font-size="17" font-weight="bold">{heading}</text>')
        for index in range(5):
            value = low + index * (high - low) / 4
            y = ypos(value, low, high, top, bottom)
            result.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#ddd"/>')
            result.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.0f}</text>')
        coords = " ".join(f"{xpos(1964+i):.2f},{ypos(value, low, high, top, bottom):.2f}" for i, value in enumerate(values))
        result.append(f'<polyline points="{coords}" fill="none" stroke="{colour}" stroke-width="3" stroke-linejoin="round"/>')
    for year in (1965, 1975, 1985, 1995, 2005, 2015):
        result.append(f'<text x="{xpos(year):.2f}" y="620" text-anchor="middle" font-family="sans-serif" font-size="13">{year}</text>')
    result.extend([
        '<text x="545" y="670" text-anchor="middle" font-family="sans-serif" font-size="13">Investment includes inventories and residential construction, not only business fixed capital.</text>',
        '<text x="545" y="691" text-anchor="middle" font-family="sans-serif" font-size="13">A rate of GDP growth is not the same measure as an investment share of GDP.</text>',
        '<text x="545" y="712" text-anchor="middle" font-family="sans-serif" font-size="13">Historical co-movement is not an estimated effect of any tax reform.</text>',
        '</svg>',
    ])
    return "\n".join(result) + "\n"


def main() -> None:
    """Regenerate versioned historical descriptive figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "growth_and_investment.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_macro(load_macro()), encoding="utf-8")


if __name__ == "__main__":
    main()
