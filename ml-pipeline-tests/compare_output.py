"""Compare a fresh run of analysis.py with the committed results/output.txt.

Counts and every word must match exactly. Decimal numbers that come out of a
trained model may move in their last printed digit on another CPU, so each one
may differ by up to two units in its last decimal place (0.709 may become
0.707 or 0.711, 8.5 may become 8.3 or 8.7). Anything larger fails the check.

    python compare_output.py results/output.txt fresh_output.txt
"""

import re
import sys
from decimal import Decimal
from pathlib import Path

DECIMAL = re.compile(r"^-?\d+\.\d+$")
TOKEN = re.compile(r"-?\d[\d,]*\.\d+(?:e[+-]\d+)?|-?\d[\d,]*|\S")


def tokens(line: str) -> list[str]:
    return TOKEN.findall(line)


def close(a: str, b: str) -> bool:
    if a == b:
        return True
    if not (DECIMAL.match(a) and DECIMAL.match(b)):
        return False
    places = max(len(a.split(".")[1]), len(b.split(".")[1]))
    return abs(Decimal(a) - Decimal(b)) <= 2 * Decimal(10) ** -places


def main(committed: str, fresh: str) -> int:
    old = Path(committed).read_text().splitlines()
    new = Path(fresh).read_text().splitlines()
    if len(old) != len(new):
        print(f"line count differs: {len(old)} committed, {len(new)} fresh")
        return 1
    bad = 0
    for i, (a, b) in enumerate(zip(old, new, strict=True), start=1):
        ta, tb = tokens(a), tokens(b)
        if len(ta) != len(tb) or not all(close(x, y) for x, y in zip(ta, tb, strict=True)):
            bad += 1
            print(f"line {i}:\n  committed: {a}\n  fresh:     {b}")
    print(f"{bad} of {len(old)} lines differ beyond the last printed digit")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
