"""Verify the OMB receipts-minus-outlays identity and render both series."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).parent
RECEIPTS = ROOT / "data" / "federal_receipts_gdp_1964_2018.csv"
OUTLAYS = ROOT / "data" / "federal_net_outlays_gdp_1964_2018.csv"
BALANCE = ROOT / "data" / "federal_balance_public_debt_1964_2018.csv"
YEARS = list(range(1964, 2019))


@dataclass(frozen=True)
class FiscalComponents:
    """Annual federal aggregates, each as a share of nominal annual GDP."""

    year: int
    receipts: float
    outlays: float
    balance: float

    def __post_init__(self) -> None:
        if type(self.year) is not int or self.year not in YEARS:
            raise ValueError("Unexpected fiscal year")
        for value in (self.receipts, self.outlays, self.balance):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise TypeError("Fiscal shares must be finite numbers")
        # Published ratios differ by at most 0.00013 percentage points.
        if abs(self.receipts - self.outlays - self.balance) > 0.00015:
            raise ValueError("Federal balance accounting identity fails")


def _series(path: Path, columns: list[str], value_col: str) -> list[tuple[int, float]]:
    """Strictly parse a complete and uniquely keyed 1964–2018 series."""
    if not isinstance(path, Path):
        raise TypeError("Input file path must be pathlib.Path")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError("Unexpected fiscal series schema")
        observations = [(int(r["year"]), float(r[value_col])) for r in reader]
    if [year for year, _ in observations] != YEARS:
        raise ValueError("Expected complete 55-year fiscal series")
    return observations


def load_components(receipts_path: Path = RECEIPTS, outlays_path: Path = OUTLAYS, balance_path: Path = BALANCE) -> list[FiscalComponents]:
    """Verify that three official series reconcile without balancing adjustments."""
    receipts = _series(receipts_path, ["year", "receipts_pct_gdp"], "receipts_pct_gdp")
    outlays = _series(outlays_path, ["year", "federal_net_outlays_gdp_pct"], "federal_net_outlays_gdp_pct")
    balance = _series(balance_path, ["year", "federal_surplus_deficit_gdp_pct", "debt_held_public_gdp_pct"], "federal_surplus_deficit_gdp_pct")
    return [FiscalComponents(year, r, o, b) for (year, r), (_, o), (_, b) in zip(receipts, outlays, balance, strict=True)]


def render_components(rows: list[FiscalComponents]) -> str:
    """Draw paired time series with a documented accounting interpretation."""
    if [r.year for r in rows] != YEARS:
        raise ValueError("Expected the ordered 1964–2018 window")
    left, right, top, bottom = 100., 1010., 110., 468.

    def px(year: int) -> float:
        return left + (year-1964)*(right-left)/54

    def py(value: float) -> float:
        return bottom-(value-12)*(bottom-top)/15

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 640" role="img" aria-labelledby="title desc">',
        '<title id="title">U.S. federal receipts and net outlays as percentages of GDP, 1964–2018</title>',
        '<desc id="desc">Annual OMB/FRED federal receipts and net outlays, whose difference equals the annual budget balance. Observed movements do not identify tax-policy effects.</desc>',
        '<rect width="1100" height="640" fill="white"/>',
        '<text x="550" y="38" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Federal receipts and net outlays, 1964–2018</text>',
        '<text x="550" y="65" text-anchor="middle" font-family="sans-serif" font-size="14">OMB/FRED · percent of annual nominal GDP · federal fiscal-year numerators</text>',
    ]
    for value in (12, 15, 18, 21, 24, 27):
        y = py(value)
        svg.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#ddd"/>')
        svg.append(f'<text x="86" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{value}%</text>')
    for year in (1964, 1974, 1984, 1994, 2004, 2014, 2018):
        svg.append(f'<text x="{px(year):.2f}" y="495" text-anchor="middle" font-family="sans-serif" font-size="13">{year}</text>')
    for series, color in (([r.receipts for r in rows], "#2779a3"), ([r.outlays for r in rows], "#bc6938")):
        points = " ".join(f"{px(r.year):.2f},{py(v):.2f}" for r, v in zip(rows, series, strict=True))
        svg.append(f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width="3"/>')
    svg.extend([
        '<line x1="310" x2="355" y1="523" y2="523" stroke="#2779a3" stroke-width="4"/>',
        '<text x="363" y="529" font-family="sans-serif" font-size="15">Federal receipts</text>',
        '<line x1="600" x2="645" y1="523" y2="523" stroke="#bc6938" stroke-width="4"/>',
        '<text x="653" y="529" font-family="sans-serif" font-size="15">Federal net outlays</text>',
        '<text x="550" y="563" text-anchor="middle" font-family="sans-serif" font-size="14">Receipts − outlays = surplus (or deficit); independently checked for all 55 years.</text>',
        '<text x="550" y="587" text-anchor="middle" font-family="sans-serif" font-size="14">An accounting gap is not the causal effect of any tax reform.</text>',
        '<text x="550" y="613" text-anchor="middle" font-family="sans-serif" font-size="13">Source: OMB / FRED FYFRGDA188S, FYONGDA188S, FYFSGDA188S (2026 vintage).</text>',
        '</svg>',
    ])
    return "\n".join(svg) + "\n"


def main() -> None:
    """Regenerate the stored receipts/outlays decomposition figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revenue_outlays.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_components(load_components()), encoding="utf-8")


if __name__ == "__main__":
    main()
