"""Post hoc: the sentence each operator-report total was read from, for the months named.

    python diagnose.py --months 2021-07 --sources sources --out results/diagnosis.txt

Not part of the registered analysis; it shows the text behind a parsed value that disagrees with both other
counts, so that a misreading of the PDF can be told apart from a figure printed in the report.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from feed.http import get
from feed.sources import REPORT_PATTERNS, pdf_text


def contexts(text: str, width: int = 120) -> list[str]:
    t = re.sub(r"\s+", " ", text)
    out = []
    for i, p in enumerate(REPORT_PATTERNS):
        for m in re.finditer(p, t, flags=re.I):
            out.append(f"pattern {i}: ...{t[max(0, m.start() - width) : m.end() + width]}...")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--months", nargs="+", required=True)
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    ap.add_argument("--out", type=Path, default=Path("results/diagnosis.txt"))
    a = ap.parse_args()
    rows = {r["month"]: r for r in csv.DictReader(open(a.sources / "citibike_operator_reports.csv", encoding="utf-8"))}
    lines = ["Post hoc: where each operator-report total was read (not part of the registered analysis).", ""]
    for m in a.months:
        r = rows.get(m)
        if not r:
            lines.append(f"{m}: no report link")
            continue
        try:
            text = pdf_text(get(r["url"]), pages=6)
        except Exception as e:  # noqa: BLE001 - recorded, not fatal
            lines.append(f"{m}: download failed ({str(e)[:80]})")
            continue
        lines.append(f"{m}: parsed total {r['trips']} (pattern {r['pattern']}), {r['url']}")
        lines += [f"    {c}" for c in contexts(text)] or ["    no pattern found"]
        lines.append("")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
