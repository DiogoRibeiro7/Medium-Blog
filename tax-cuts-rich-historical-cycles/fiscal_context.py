"""Visual chronology of statutory top rates and federal receipts/GDP.

The two series measure different aggregates. This module does not fit any
relationship or identify an effect of tax legislation on government revenue.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from historical_rates import Observation, load_rates


RECEIPTS_FILE = Path(__file__).parent / "data" / "federal_receipts_gdp_1964_2018.csv"


@dataclass(frozen=True)
class FiscalPoint:
    """FRED receipts-to-annual-GDP percentage for one labelled year."""

    year: int
    receipts_pct_gdp: float

    def __post_init__(self) -> None:
        if type(self.year) is not int or not 1964 <= self.year <= 2018:
            raise ValueError("Invalid historical year")
        if not isinstance(self.receipts_pct_gdp, (float, int)) or not math.isfinite(self.receipts_pct_gdp):
            raise TypeError("Receipts/GDP must be finite and numeric")
        if not 0 < self.receipts_pct_gdp < 100:
            raise ValueError("Receipts/GDP must lie in (0, 100)")


def load_receipts(path: Path = RECEIPTS_FILE) -> list[FiscalPoint]:
    """Validate a frozen, strictly continuous historical transcription."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["year", "receipts_pct_gdp"]:
            raise ValueError("Unexpected receipts source schema")
        rows = [FiscalPoint(int(row["year"]), float(row["receipts_pct_gdp"])) for row in reader]
    if [row.year for row in rows] != list(range(1964, 2019)):
        raise ValueError("Missing, reordered, or duplicate fiscal years")
    return rows


def render_comparison(rates: list[Observation], receipts: list[FiscalPoint]) -> str:
    """Draw a two-panel SVG with independent scales and common chronology."""
    if [row.year for row in rates] != [row.year for row in receipts] or len(rates) != 55:
        raise ValueError("Source chronologies must have the same 55 years")
    left, right = 100.0, 980.0

    def px(year: int) -> float:
        return left + (year - 1964) * (right - left) / 54

    def py(value: float, low: float, high: float, top: float, bottom: float) -> float:
        return bottom - (value - low) * (bottom - top) / (high - low)

    result = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 730" role="img" aria-labelledby="title description">',
        '<title id="title">U.S. statutory top tax bracket rate versus federal receipts-to-GDP, 1964–2018</title>',
        '<desc id="description">Two descriptive panels with independent scales. No causal relationship is claimed.</desc>',
        '<rect width="1080" height="730" fill="white"/>',
        '<text x="540" y="38" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="22">Top rate and total federal receipts, 1964–2018</text>',
        '<text x="540" y="61" text-anchor="middle" font-family="sans-serif" font-size="13">Two different measures · independent vertical scales · no causal attribution</text>',
    ]
    for title, values, domain, box, stroke in (
        ("Highest statutory individual rate (%) — IRS", [r.top_rate_pct for r in rates], (0., 100.), (110., 305.), "#225f9e"),
        ("All federal receipts (% of GDP) — OMB/FRED", [r.receipts_pct_gdp for r in receipts], (10., 22.), (390., 585.), "#a14d24"),
    ):
        top, bottom = box
        lo, hi = domain
        result.append(f'<text x="{left}" y="{top-20}" font-family="sans-serif" font-size="17" font-weight="bold">{title}</text>')
        for tick in range(5):
            val = lo + tick * (hi - lo) / 4
            yy = py(val, lo, hi, top, bottom)
            result.append(f'<line x1="{left}" x2="{right}" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#ddd"/>')
            result.append(f'<text x="{left-12}" y="{yy+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{val:.0f}</text>')
        coords = " ".join(f"{px(1964+i):.2f},{py(val, lo, hi, top, bottom):.2f}" for i, val in enumerate(values))
        result.append(f'<polyline points="{coords}" stroke="{stroke}" stroke-width="3" stroke-linejoin="round" fill="none"/>')
    for year in (1965, 1975, 1985, 1995, 2005, 2015):
        result.append(f'<text x="{px(year):.2f}" y="614" text-anchor="middle" font-family="sans-serif" font-size="13">{year}</text>')
    result.extend([
        '<text x="540" y="653" text-anchor="middle" font-family="sans-serif" font-size="13">Fiscal receipts contain payroll, corporate and other taxes; the top rate is a statutory bracket rate.</text>',
        '<text x="540" y="674" text-anchor="middle" font-family="sans-serif" font-size="13">Receipts have a fiscal-year numerator and a calendar-year GDP denominator.</text>',
        '<text x="540" y="695" text-anchor="middle" font-family="sans-serif" font-size="13">The chart does not estimate tax-cut effects, revenue-maximizing rates or self-financing.</text>',
        '</svg>',
    ])
    return "\n".join(result) + "\n"


def main() -> None:
    """Generate the frozen descriptive visual for offline reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "tax_rate_and_receipts.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_comparison(load_rates(), load_receipts()), encoding="utf-8")


if __name__ == "__main__":
    main()
