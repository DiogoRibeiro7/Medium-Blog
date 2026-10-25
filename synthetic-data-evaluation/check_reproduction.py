"""Compare a fresh results/summary.txt with the committed one, number by number.

Floating-point kernels differ slightly between machines (a tree split that is
a near-tie can go the other way), so a coverage count can move by a
replication and a printed value can round to the neighbouring digit. A number
passes if it differs from the committed one by at most 0.006 (about one Monte
Carlo standard error of a coverage estimate), by at most one unit in its last
printed digit, or, for numbers of magnitude 1 or more, by at most 1%
(interval-length ratios under the Raghunathan-Reiter-Rubin rule are
heavy-tailed averages). The text around the
numbers must be identical.

Usage: python check_reproduction.py committed_summary.txt results/summary.txt
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

NUMBER = re.compile(r"[-+]?\d+\.\d+|[-+]?\d+")
TOLERANCE = 0.006
RELATIVE = 0.01


def close(x: str, y: str) -> bool:
    a, b = float(x), float(y)
    decimals = len(x.split(".")[1]) if "." in x else 0
    last_digit = 1.01 * 10.0 ** (-decimals)
    relative = RELATIVE * max(abs(a), abs(b)) if max(abs(a), abs(b)) >= 1 else 0.0
    return abs(a - b) <= max(TOLERANCE, last_digit, relative)


def main(old_path: str, new_path: str) -> int:
    old = Path(old_path).read_text().splitlines()
    new = Path(new_path).read_text().splitlines()
    if len(old) != len(new):
        print(f"line count differs: {len(old)} vs {len(new)}")
        return 1
    worst, failures = 0.0, 0
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"

    def report(message: str) -> None:
        print(message)
        if in_actions and failures < 10:  # also as annotations, readable through the API
            print("::error title=summary mismatch::" + message.replace("\n", " | "))

    for i, (a, b) in enumerate(zip(old, new), start=1):
        if NUMBER.sub("#", a) != NUMBER.sub("#", b):
            report(f"line {i}: text differs\n  {a}\n  {b}")
            failures += 1
            continue
        for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)):
            diff = abs(float(x) - float(y))
            worst = max(worst, diff)
            if not close(x, y):
                report(f"line {i}: {x} vs {y}\n  {a}\n  {b}")
                failures += 1
    print(f"largest absolute difference: {worst:.4f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
