"""Render CBO's selected 2018 income-group estimates without causal attribution."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DATA_FILE = Path(__file__).parent / "data" / "cbo_2018_distribution.csv"
GROUPS = ("Lowest quintile", "Middle quintile", "Highest quintile")


@dataclass(frozen=True)
class HouseholdIncome:
    """Rounded CBO group-average income in constant 2018 dollars."""

    group: str
    before: int
    after: int

    def __post_init__(self) -> None:
        if self.group not in GROUPS:
            raise ValueError("Unexpected income group")
        if type(self.before) is not int or type(self.after) is not int:
            raise TypeError("Income values must be integer 2018 dollars")
        if self.before < 0 or self.after < 0:
            raise ValueError("Income values cannot be negative")


def load_income(path: Path = DATA_FILE) -> list[HouseholdIncome]:
    """Read the fixed three-group subset, requiring exactly the published schema."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "income_group", "before_transfers_taxes_usd_2018", "after_transfers_taxes_usd_2018"
        ]:
            raise ValueError("Unexpected CBO income schema")
        rows = [HouseholdIncome(row["income_group"], int(row["before_transfers_taxes_usd_2018"]), int(row["after_transfers_taxes_usd_2018"])) for row in reader]
    if tuple(row.group for row in rows) != GROUPS:
        raise ValueError("Expected one row for each selected quintile in order")
    return rows


def render_income(rows: list[HouseholdIncome]) -> str:
    """Return a deterministic paired-bar SVG emphasizing concepts, not policies."""
    if tuple(row.group for row in rows) != GROUPS:
        raise ValueError("Expected the three CBO quintiles in order")
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 570" role="img" aria-labelledby="title desc">',
        '<title id="title">Selected CBO household income quintiles, 2018</title>',
        '<desc id="desc">Paired bars compare 2018 average household income before and after means-tested transfers and federal taxes for three selected quintiles; not a tax-law effect.</desc>',
        '<rect width="1100" height="570" fill="white"/>',
        '<text x="550" y="38" text-anchor="middle" font-size="23" font-family="sans-serif" font-weight="bold">Household resources before and after transfers and taxes</text>',
        '<text x="550" y="64" text-anchor="middle" font-size="15" font-family="sans-serif">CBO estimates · selected 2018 quintiles · average 2018 dollars</text>',
        '<rect x="270" y="87" width="18" height="16" fill="#3579a6"/>',
        '<text x="295" y="101" font-family="sans-serif" font-size="14">Before means-tested transfers and federal taxes</text>',
        '<rect x="680" y="87" width="18" height="16" fill="#d17b36"/>',
        '<text x="705" y="101" font-family="sans-serif" font-size="14">After transfers and taxes</text>',
    ]
    start, scale = 230, 700 / 350000
    for index, row in enumerate(rows):
        y = 158 + index * 104
        svg.append(f'<text x="210" y="{y+29}" text-anchor="end" font-family="sans-serif" font-size="16">{row.group}</text>')
        for value, offset, color in ((row.before, 0, "#3579a6"), (row.after, 36, "#d17b36")):
            width = value * scale
            svg.append(f'<rect x="{start}" y="{y+offset}" width="{width:.2f}" height="27" fill="{color}"/>')
            svg.append(f'<text x="{start+width+9:.2f}" y="{y+offset+19}" font-family="sans-serif" font-size="14">${value:,}</text>')
    svg.extend([
        '<text x="550" y="495" text-anchor="middle" font-family="sans-serif" font-size="14">Before includes social insurance; after adds means-tested transfers and subtracts federal taxes.</text>',
        '<text x="550" y="519" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold">The gap between bars is NOT an estimate of the effect of the 2017 tax act.</text>',
        '<text x="550" y="544" text-anchor="middle" font-family="sans-serif" font-size="13">Source: Congressional Budget Office, The Distribution of Household Income, 2018 (2021); rounded estimates.</text>',
        '</svg>',
    ])
    return "\n".join(svg) + "\n"


def main() -> None:
    """Regenerate the committed descriptive SVG without external dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "cbo_2018_distribution.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_income(load_income()), encoding="utf-8")


if __name__ == "__main__":
    main()
