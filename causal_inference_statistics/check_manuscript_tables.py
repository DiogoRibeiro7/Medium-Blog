"""Check reported four-decimal table values against archived Monte Carlo CSVs.

Usage: python causal_inference_statistics/check_manuscript_tables.py --parts 1 2 3 4 5

This is a conservative numeric consistency screen, not a full semantic proof:
it checks four-decimal table cells against numeric CSV cells after rounding.
"""
from __future__ import annotations

import argparse
import csv
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELL = re.compile(r"^-?\d[\d,]*\.\d{4}$")


def audit(part: int) -> list[str]:
    """Return manuscript numeric table cells unsupported by archived output."""
    folder = ROOT / f"part{part}"
    manuscript = (folder / "article.md").read_text(encoding="utf-8")
    with (folder / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: set[Decimal] = set()
    for row in rows:
        for raw in row.values():
            try:
                values.add(Decimal(str(raw)).quantize(Decimal("0.0001")))
            except (InvalidOperation, TypeError):
                continue
    errors: list[str] = []
    for lineno, line in enumerate(manuscript.splitlines(), start=1):
        if not (line.startswith("|") and line.endswith("|")):
            continue
        if "True treatment effect" in line or "True total effect" in line:
            continue  # Structural truths may be specified only in experiment source.
        for text in line.strip("|").split("|"):
            cell = text.strip()
            if CELL.fullmatch(cell):
                value = Decimal(cell.replace(",", ""))
                if value not in values:
                    errors.append(f"part {part}, line {lineno}: table cell {cell} not found in CSV")
    return errors


def main() -> int:
    """Run a conservative numeric-table consistency check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", type=int, required=True)
    args = parser.parse_args()
    failures = 0
    for part in args.parts:
        if not 1 <= part <= 20:
            parser.error("Part must be 1–20")
        problems = audit(part)
        print(f"Part {part}: {'PASS' if not problems else 'REVIEW'}")
        for issue in problems:
            print(issue)
        failures += len(problems)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
