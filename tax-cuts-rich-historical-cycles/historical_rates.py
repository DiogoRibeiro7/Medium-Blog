"""Validate and visualize IRS top statutory rate history (1964–2018).

This historical series is descriptive, not an estimate of policy effects.
It deliberately does not combine statutory individual tax rates with
corporate rates, realized average burdens, or outcome series.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


DATA_FILE = Path(__file__).parent / "data" / "top_marginal_rate_1964_2018.csv"


@dataclass(frozen=True)
class Observation:
    """One tax-year's highest statutory federal regular-income-tax rate."""

    year: int
    top_rate_pct: float

    def __post_init__(self) -> None:
        if type(self.year) is not int or not 1964 <= self.year <= 2018:
            raise ValueError("Year must be an integer within 1964–2018")
        if not isinstance(self.top_rate_pct, (int, float)) or not math.isfinite(self.top_rate_pct):
            raise TypeError("Rate must be finite and numeric")
        if not 0 < self.top_rate_pct <= 100:
            raise ValueError("Rate must lie in (0, 100]")


def load_rates(path: Path = DATA_FILE) -> list[Observation]:
    """Read the frozen source transcription and enforce annual continuity."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["year", "top_rate_pct"]:
            raise ValueError("Unexpected source-data schema")
        observations = [
            Observation(year=int(row["year"]), top_rate_pct=float(row["top_rate_pct"]))
            for row in reader
        ]
    if [row.year for row in observations] != list(range(1964, 2019)):
        raise ValueError("Historical tax years must be unique, sorted, and complete")
    return observations


def render_svg(observations: list[Observation]) -> str:
    """Build a deterministic, accessible SVG with annotated policy turning points."""
    if len(observations) != 55 or observations[0].year != 1964 or observations[-1].year != 2018:
        raise ValueError("Expected the complete 1964–2018 historical series")
    width, height = 1000, 580
    left, right, top, bottom = 95, 950, 85, 480

    def x(year: int) -> float:
        return left + (year - 1964) * (right - left) / (2018 - 1964)

    def y(rate: float) -> float:
        return bottom - rate * (bottom - top) / 100

    points = " ".join(f"{x(row.year):.2f},{y(row.top_rate_pct):.2f}" for row in observations)
    content = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 580" role="img" aria-labelledby="title desc">',
        '<title id="title">U.S. highest statutory federal individual income-tax rate, 1964–2018</title>',
        '<desc id="desc">IRS regular-income-tax highest bracket rate; descriptive annual series, not effective taxation.</desc>',
        '<rect width="1000" height="580" fill="white"/>',
        '<text x="500" y="38" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="22">Top statutory U.S. individual income-tax rate, 1964–2018</text>',
        '<text x="500" y="62" text-anchor="middle" font-family="sans-serif" font-size="13">IRS / FRED IITTRHB · highest regular-tax bracket · percent</text>',
    ]
    for rate in range(0, 101, 20):
        yy = y(float(rate))
        content.extend([
            f'<line x1="{left}" x2="{right}" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#ddd"/>',
            f'<text x="80" y="{yy + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{rate}%</text>',
        ])
    for year in (1965, 1975, 1985, 1995, 2005, 2015):
        xx = x(year)
        content.append(f'<text x="{xx:.2f}" y="502" text-anchor="middle" font-family="sans-serif" font-size="13">{year}</text>')
    content.append(f'<polyline fill="none" stroke="#225f9e" stroke-width="3" stroke-linejoin="round" points="{points}"/>')
    for year, label, label_y in ((1965, "1965: 70%", 135), (1982, "1982: 50%", 250), (1988, "1988: 28%", 360), (2003, "2003: 35%", 325), (2018, "2018: 37%", 300)):
        observed = next(row for row in observations if row.year == year)
        xx, yy = x(year), y(observed.top_rate_pct)
        content.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="4.5" fill="#153d66"/>')
        content.append(f'<text x="{xx:.2f}" y="{label_y}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>')
    content.extend([
        '<text x="500" y="533" text-anchor="middle" font-family="sans-serif" font-size="12">Statutory rates differ from effective burdens; bracket bases and earned-income rules changed over time.</text>',
        '<text x="500" y="550" text-anchor="middle" font-family="sans-serif" font-size="12">Descriptive historical evidence only; rate changes do not identify macroeconomic causal effects.</text>',
        '</svg>',
    ])
    return "\n".join(content) + "\n"


def main() -> None:
    """Validate the annual transcription and write the source-linked chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results" / "historical_top_rate.svg",
    )
    args = parser.parse_args()
    rows = load_rates()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_svg(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
