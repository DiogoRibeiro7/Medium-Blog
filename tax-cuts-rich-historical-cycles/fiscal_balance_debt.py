"""Validate and visualize OMB/FRED fiscal balance and publicly held debt."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


DATA_FILE = Path(__file__).parent / "data" / "federal_balance_public_debt_1964_2018.csv"
YEARS = list(range(1964, 2019))


@dataclass(frozen=True)
class FiscalObservation:
    """Annual OMB/FRED federal balance and publicly held debt, each as % GDP."""

    year: int
    balance_pct: float
    debt_pct: float

    def __post_init__(self) -> None:
        if type(self.year) is not int or self.year not in YEARS:
            raise ValueError("Year outside 1964–2018")
        for value in (self.balance_pct, self.debt_pct):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise TypeError("Fiscal percentages must be finite numbers")
        if not -50 <= self.balance_pct <= 50 or not 0 <= self.debt_pct <= 250:
            raise ValueError("Fiscal observation outside sanity bounds")


def load_fiscal(path: Path = DATA_FILE) -> list[FiscalObservation]:
    """Read the full frozen window without interpolating missing observations."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["year", "federal_surplus_deficit_gdp_pct", "debt_held_public_gdp_pct"]:
            raise ValueError("Unexpected OMB/FRED fiscal CSV schema")
        rows = [FiscalObservation(int(r["year"]), float(r["federal_surplus_deficit_gdp_pct"]), float(r["debt_held_public_gdp_pct"])) for r in reader]
    if [r.year for r in rows] != YEARS:
        raise ValueError("Expected 55 consecutive, unique annual fiscal observations")
    return rows


def render_fiscal(rows: list[FiscalObservation]) -> str:
    """Create a deterministic two-panel descriptive chart with distinct axes."""
    if [r.year for r in rows] != YEARS:
        raise ValueError("Expected complete ordered fiscal series")
    left, right = 100., 1010.

    def px(year: int) -> float:
        return left + (year - 1964) * (right-left)/54

    def py(value: float, minimum: float, maximum: float, top: float, bottom: float) -> float:
        return bottom - (value-minimum) * (bottom-top)/(maximum-minimum)

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 790" role="img" aria-labelledby="title desc">',
        '<title id="title">Federal budget balance and debt held by the public, 1964–2018</title>',
        '<desc id="desc">Two separate annual OMB/FRED percentage-of-GDP line charts, with fiscal/calendar-year alignment caveat, without causal attribution to tax reforms.</desc>',
        '<rect width="1100" height="790" fill="white"/>',
        '<text x="550" y="37" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Federal deficits and accumulated public debt, 1964–2018</text>',
        '<text x="550" y="63" text-anchor="middle" font-family="sans-serif" font-size="14">OMB/FRED · federal fiscal data as shares of annual nominal GDP</text>',
    ]
    panels = [
        ("Federal surplus (+) / deficit (−), % GDP", [r.balance_pct for r in rows], -12., 4., 118., 345., "#bd6539", [-10, -5, 0]),
        ("Debt held by the public, % GDP", [r.debt_pct for r in rows], 0., 100., 430., 662., "#286b9b", [0, 25, 50, 75, 100]),
    ]
    for title, values, low, high, top, bottom, color, ticks in panels:
        svg.append(f'<text x="{left}" y="{top-16}" font-family="sans-serif" font-size="17" font-weight="bold">{title}</text>')
        for value in ticks:
            y = py(value, low, high, top, bottom)
            svg.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#ddd"/>')
            svg.append(f'<text x="87" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{value}%</text>')
        points = " ".join(f"{px(r.year):.2f},{py(value,low,high,top,bottom):.2f}" for r, value in zip(rows, values, strict=True))
        svg.append(f'<polyline points="{points}" stroke="{color}" stroke-width="3" fill="none"/>')
        for year in (1964, 1974, 1984, 1994, 2004, 2014, 2018):
            svg.append(f'<text x="{px(year):.2f}" y="{bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="13">{year}</text>')
    svg.extend([
        '<text x="550" y="725" text-anchor="middle" font-family="sans-serif" font-size="14">Fiscal-year numerators and calendar-year GDP denominators are not exactly aligned.</text>',
        '<text x="550" y="748" text-anchor="middle" font-family="sans-serif" font-size="14">Deficits depend on spending, taxes, the business cycle, and interest; no tax-cut effect is identified.</text>',
        '<text x="550" y="774" text-anchor="middle" font-family="sans-serif" font-size="12">Source: OMB/FRED FYFSGDA188S and FYPUGDA188S; frozen November 2026 vintage.</text>',
        '</svg>',
    ])
    return "\n".join(svg) + "\n"


def main() -> None:
    """Regenerate the versioned fiscal-balance and debt comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "fiscal_balance_public_debt.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_fiscal(load_fiscal()), encoding="utf-8")


if __name__ == "__main__":
    main()
