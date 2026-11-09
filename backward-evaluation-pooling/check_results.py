"""Compare regenerated summary files with the committed ones, number by number.

    python check_results.py results/sims_summary.txt results/premise_summary.txt results/forward_summary.txt

The text must match exactly once numbers are masked; each number must agree within an
absolute tolerance of 0.05 or a relative tolerance of 2%. Maximum-likelihood fits and their
bootstrap replicates can move in the last digit across machines (BLAS, CPU), so an exact
text comparison is too strict for them.
"""

from __future__ import annotations

import re
import subprocess
import sys

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def committed(path: str) -> str:
    return subprocess.run(["git", "show", f"HEAD:./{path}"], capture_output=True, text=True, check=True).stdout


def compare(old: str, new: str, abs_tol: float = 0.05, rel_tol: float = 0.02) -> list[str]:
    problems = []
    old_lines, new_lines = old.splitlines(), new.splitlines()
    if len(old_lines) != len(new_lines):
        return [f"line count {len(old_lines)} -> {len(new_lines)}"]
    for i, (a, b) in enumerate(zip(old_lines, new_lines), start=1):
        if a == b:
            continue
        if NUM.sub("#", a) != NUM.sub("#", b):
            problems.append(f"line {i}: text differs\n  - {a}\n  + {b}")
            continue
        for x, y in zip(NUM.findall(a), NUM.findall(b)):
            fx, fy = float(x), float(y)
            if abs(fx - fy) > max(abs_tol, rel_tol * max(abs(fx), abs(fy))):
                problems.append(f"line {i}: {x} -> {y}\n  - {a}\n  + {b}")
                break
    return problems


def main(paths: list[str]) -> int:
    bad = 0
    for path in paths:
        with open(path) as f:
            new = f.read()
        old = committed(path)
        if old != new:
            print(f"{path}: differs from the committed version")
        for p in compare(old, new):
            print(f"{path}: {p}")
            bad += 1
    if bad:
        print(f"{bad} difference(s) beyond tolerance")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
