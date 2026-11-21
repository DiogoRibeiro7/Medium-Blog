"""Compare a regenerated summary with the committed one.

The text must match exactly; each number may differ by at most one unit in its
last printed digit, which absorbs rounding differences between machines (the
synthetic-control optimiser and floating-point sums are not bit-identical across
CPUs)::

    python check_summary.py committed.txt regenerated.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def split(line: str) -> tuple[list[str], list[str]]:
    return NUMBER.split(line), NUMBER.findall(line)


def close(a: str, b: str) -> bool:
    if a == b:
        return True
    decimals = max(len(x.split(".")[1]) if "." in x else 0 for x in (a, b))
    x, y = float(a.replace(",", "")), float(b.replace(",", ""))
    return abs(x - y) <= 10 ** -decimals * 1.000001


def main(argv: list[str]) -> int:
    old = Path(argv[1]).read_text().splitlines()
    new = Path(argv[2]).read_text().splitlines()
    problems = []
    if len(old) != len(new):
        problems.append(f"line count {len(old)} != {len(new)}")
    for i, (a, b) in enumerate(zip(old, new), start=1):
        ta, na = split(a)
        tb, nb = split(b)
        if ta != tb or len(na) != len(nb) or not all(close(x, y) for x, y in zip(na, nb)):
            problems.append(f"line {i}:\n  - {a}\n  + {b}")
    if problems:
        print("\n".join(problems))
        return 1
    print(f"{len(new)} lines match within one unit of the last printed digit.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
