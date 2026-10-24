"""Compare a fresh results/summary.txt with the committed one, number by number.

Floating-point kernels differ slightly between machines, so a coverage count
can move by a replication or a distance tie can break the other way. The
check allows an absolute difference of 0.006 (about one Monte Carlo standard
error of a coverage estimate) and requires the text around the numbers to be
identical.

Usage: python check_reproduction.py committed_summary.txt results/summary.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

NUMBER = re.compile(r"[-+]?\d+\.\d+|[-+]?\d+")
TOLERANCE = 0.006


def main(old_path: str, new_path: str) -> int:
    old = Path(old_path).read_text().splitlines()
    new = Path(new_path).read_text().splitlines()
    if len(old) != len(new):
        print(f"line count differs: {len(old)} vs {len(new)}")
        return 1
    worst, failures = 0.0, 0
    for i, (a, b) in enumerate(zip(old, new), start=1):
        if NUMBER.sub("#", a) != NUMBER.sub("#", b):
            print(f"line {i}: text differs\n  {a}\n  {b}")
            failures += 1
            continue
        for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)):
            diff = abs(float(x) - float(y))
            worst = max(worst, diff)
            if diff > TOLERANCE:
                print(f"line {i}: {x} vs {y}\n  {a}\n  {b}")
                failures += 1
    print(f"largest absolute difference: {worst:.4f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
