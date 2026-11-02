"""Compare BEA nonresidential fixed investment with gross private investment."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


BASE = Path(__file__).parent
NONRES_FILE = BASE / "data" / "nonresidential_fixed_investment_share_1964_2018.csv"
GROSS_FILE = BASE / "data" / "real_growth_private_investment_1964_2018.csv"
YEARS = list(range(1964, 2019))


@dataclass(frozen=True)
class InvestmentPoint:
    """BEA percent-of-nominal-GDP series for a calendar year."""

    year: int
    nonresidential_fixed_pct: float
    gross_private_pct: float

    def __post_init__(self) -> None:
        if type(self.year) is not int or self.year not in YEARS:
            raise ValueError("Year outside frozen 1964–2018 window")
        for value in (self.nonresidential_fixed_pct, self.gross_private_pct):
            if type(value) not in (float, int) or not math.isfinite(value):
                raise TypeError("Shares must be finite numbers")
            if not 0 <= value <= 100:
                raise ValueError("Investment shares must lie between 0 and 100")


def load_investment(nonres_path: Path = NONRES_FILE, gross_path: Path = GROSS_FILE) -> list[InvestmentPoint]:
    """Join two independently sourced annual series with strict year checks."""
    if not isinstance(nonres_path, Path) or not isinstance(gross_path, Path):
        raise TypeError("Input paths must be pathlib.Path")

    def read(path: Path, expected: list[str], column: str) -> list[tuple[int, float]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected:
                raise ValueError("Unexpected investment CSV schema")
            result = [(int(row["year"]), float(row[column])) for row in reader]
        if [year for year, _ in result] != YEARS:
            raise ValueError("Expected 55 consecutive unique observations")
        return result

    fixed = read(nonres_path, ["year", "nonresidential_fixed_investment_share_pct"], "nonresidential_fixed_investment_share_pct")
    gross = read(gross_path, ["year", "real_gdp_growth_pct", "private_investment_share_pct"], "private_investment_share_pct")
    return [InvestmentPoint(y, fixed_share, gross_share) for (y, fixed_share), (_, gross_share) in zip(fixed, gross, strict=True)]


def render_investment(rows: list[InvestmentPoint]) -> str:
    """Create deterministic SVG with both series and explicit measurement caveats."""
    if [row.year for row in rows] != YEARS:
        raise ValueError("Requires full 1964–2018 series")
    left, right, upper, lower = 95., 1010., 105., 435.

    def px(year: int) -> float:
        return left + (year - 1964) * (right-left) / 54

    def py(share: float) -> float:
        return lower - (share - 8) * (lower-upper) / 14

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 640" role="img" aria-labelledby="title desc">',
        '<title id="title">U.S. private investment shares of GDP, 1964–2018</title>',
        '<desc id="desc">Two descriptive annual BEA series: gross private domestic investment and the nonresidential fixed subset, excluding housing and inventories.</desc>',
        '<rect width="1100" height="640" fill="white"/>',
        '<text x="550" y="38" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="22">Business-oriented investment and the wider private aggregate</text>',
        '<text x="550" y="65" text-anchor="middle" font-family="sans-serif" font-size="14">U.S. BEA / FRED · share of nominal GDP · annual 1964–2018</text>',
    ]
    for value in (8, 10, 12, 14, 16, 18, 20, 22):
        y = py(value)
        svg.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#ddd"/>')
        svg.append(f'<text x="82" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{value}%</text>')
    for year in (1964, 1974, 1984, 1994, 2004, 2014, 2018):
        svg.append(f'<text x="{px(year):.2f}" y="463" text-anchor="middle" font-family="sans-serif" font-size="13">{year}</text>')
    for values, color in (([r.gross_private_pct for r in rows], "#3579a6"), ([r.nonresidential_fixed_pct for r in rows], "#c16d33")):
        points = " ".join(f"{px(r.year):.2f},{py(value):.2f}" for r, value in zip(rows, values, strict=True))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    svg.extend([
        '<line x1="195" x2="240" y1="501" y2="501" stroke="#3579a6" stroke-width="4"/>',
        '<text x="249" y="506" font-family="sans-serif" font-size="14">Gross private domestic investment</text>',
        '<line x1="610" x2="655" y1="501" y2="501" stroke="#c16d33" stroke-width="4"/>',
        '<text x="664" y="506" font-family="sans-serif" font-size="14">Private nonresidential fixed investment</text>',
        '<text x="550" y="549" text-anchor="middle" font-family="sans-serif" font-size="14">Nonresidential excludes residential construction and inventories, but includes structures and IP.</text>',
        '<text x="550" y="574" text-anchor="middle" font-family="sans-serif" font-size="14">Percent-of-GDP shares vary with both investment and GDP. No tax-reform effect is identified.</text>',
        '<text x="550" y="602" text-anchor="middle" font-family="sans-serif" font-size="13">Sources: BEA/FRED A008RE1A156NBEA and existing gross investment series; revised NIPA data.</text>',
        '</svg>',
    ])
    return "\n".join(svg) + "\n"


def main() -> None:
    """Regenerate versioned nonresidential fixed investment chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "results" / "nonresidential_investment.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_investment(load_investment()), encoding="utf-8")


if __name__ == "__main__":
    main()
