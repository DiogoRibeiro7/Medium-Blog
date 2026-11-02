"""Validate and plot OMB federal receipt categories, 1964–2018."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from xml.sax.saxutils import escape
from pathlib import Path


DATA = Path(__file__).parent / "data" / "omb_receipts_by_source_1964_2018.csv"
YEARS = list(range(1964, 2019))
COLUMNS = ["year", "individual_income_pct_gdp", "corporate_income_pct_gdp", "social_insurance_pct_gdp", "excise_pct_gdp", "other_pct_gdp", "total_receipts_pct_gdp"]


@dataclass(frozen=True)
class ReceiptMix:
    """Five OMB tax categories plus independently displayed total, % GDP."""

    year: int
    individual: float
    corporate: float
    social_insurance: float
    excise: float
    other: float
    total: float

    def __post_init__(self) -> None:
        if type(self.year) is not int or self.year not in YEARS:
            raise ValueError("Unexpected OMB fiscal year")
        values = (self.individual, self.corporate, self.social_insurance, self.excise, self.other, self.total)
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise TypeError("Receipt shares must be finite numeric values")
        if any(not 0 <= v <= 100 for v in values):
            raise ValueError("Receipt shares must be percentages")
        # OMB publishes each of the five categories and total to one decimal.
        if abs(sum(values[:5]) - self.total) > 0.251:
            raise ValueError("Categories do not reconcile with displayed OMB total")


def load_mix(path: Path = DATA) -> list[ReceiptMix]:
    """Load exactly 55 distinct annual observations from the frozen OMB table."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Unexpected OMB receipt-mix schema")
        rows = [ReceiptMix(int(r["year"]), *(float(r[col]) for col in COLUMNS[1:])) for r in reader]
    if [r.year for r in rows] != YEARS:
        raise ValueError("Expected complete 1964–2018 series without duplicates")
    return rows


def render_mix(rows: list[ReceiptMix]) -> str:
    """Draw three key receipt sources without conflating levels and rates."""
    if [r.year for r in rows] != YEARS:
        raise ValueError("Expected ordered 1964–2018 series")
    left, right, top, bottom = 100., 1010., 108., 440.

    def px(year: int) -> float:
        return left + (year - 1964) * (right-left) / 54

    def py(value: float) -> float:
        return bottom - value * (bottom-top) / 11

    series = [("Individual income taxes", "#245f91", [r.individual for r in rows]), ("Corporate income taxes", "#c46538", [r.corporate for r in rows]), ("Social insurance & retirement", "#32876d", [r.social_insurance for r in rows])]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 630" role="img" aria-labelledby="title desc">',
        '<title id="title">U.S. federal receipts by source, 1964–2018</title>',
        '<desc id="desc">Annual shares of nominal GDP from OMB fiscal 2027 historical Table 2.3. Three largest tax categories shown; excise and other categories are in the accompanying data.</desc>',
        '<rect width="1100" height="630" fill="white"/>',
        '<text x="550" y="38" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Which taxes financed the federal government?</text>',
        '<text x="550" y="65" text-anchor="middle" font-family="sans-serif" font-size="14">OMB FY2027 historical table · federal fiscal years 1964–2018 · percent of nominal GDP</text>',
    ]
    for value in (0, 2, 4, 6, 8, 10):
        y = py(value)
        svg.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#ddd"/>')
        svg.append(f'<text x="87" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{value}%</text>')
    for year in (1964, 1974, 1984, 1994, 2004, 2014, 2018):
        svg.append(f'<text x="{px(year):.2f}" y="465" text-anchor="middle" font-family="sans-serif" font-size="13">{year}</text>')
    for name, color, values in series:
        points = " ".join(f"{px(r.year):.2f},{py(value):.2f}" for r, value in zip(rows, values, strict=True))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    for x, (name, color, _) in zip((100, 420, 720), series, strict=True):
        svg.append(f'<line x1="{x}" x2="{x+36}" y1="506" y2="506" stroke="{color}" stroke-width="4"/>')
        svg.append(f'<text x="{x+44}" y="511" font-family="sans-serif" font-size="14">{escape(name)}</text>')
    svg.extend([
        '<text x="550" y="548" text-anchor="middle" font-family="sans-serif" font-size="14">Excise and other receipts are included in the CSV, but omitted from this three-line figure.</text>',
        '<text x="550" y="574" text-anchor="middle" font-family="sans-serif" font-size="14">Shares are not tax rates on wealthy households and do not isolate effects of tax reforms.</text>',
        '<text x="550" y="604" text-anchor="middle" font-family="sans-serif" font-size="12">Source: Office of Management and Budget, Historical Tables FY2027, Table 2.3.</text>',
        '</svg>',
    ])
    return "\n".join(svg) + "\n"


def main() -> None:
    """Regenerate the source-linked receipt-composition figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "receipt_composition.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_mix(load_mix()), encoding="utf-8")


if __name__ == "__main__":
    main()