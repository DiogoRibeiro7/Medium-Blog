"""Compare a fresh results/summary.json with a reference copy, within tolerances.

The fits are deterministic on one machine, but floating-point arithmetic in XLA can differ in
the last bits between processors, and a non-convex optimisation can carry such differences
forward. So instead of requiring identical files, this check requires every reported number to
agree with the committed run within a tolerance that is far smaller than any difference the
article draws a conclusion from: graph statistics exactly, average distortion and MAP within
0.003 or 3%, worst-case distortion and radii within 25%, losses within 10%.

Usage:  python check_results.py REFERENCE.json results/summary.json
"""

from __future__ import annotations

import json
import math
import sys

TOL = {"D_avg": (0.003, 0.03), "MAP": (0.003, 0.03), "D_avg_1500": (0.003, 0.03), "D_avg_4500": (0.003, 0.03),
       "MAP_1500": (0.003, 0.03), "MAP_4500": (0.003, 0.03), "worst": (0.0, 0.25), "radius_max": (0.0, 0.25),
       "loss": (0.0, 0.10), "log_u": (0.1, 0.0),
       "sibling_over_parent": (0.03, 0.0), "sibling_over_parent_min": (0.03, 0.0), "sibling_over_parent_max": (0.03, 0.0)}


def walk(a, b, path, bad):
    if isinstance(a, dict):
        if set(a) != set(b):
            bad.append(f"{path}: keys differ")
            return
        for k in a:
            walk(a[k], b[k], f"{path}/{k}", bad)
    elif isinstance(a, list):
        if len(a) != len(b):
            bad.append(f"{path}: lengths differ")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", bad)
    elif isinstance(a, float) or isinstance(b, float):
        key = path.rsplit("/", 1)[-1]
        abs_tol, rel_tol = TOL.get(key, (0.0, 0.0))
        if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            bad.append(f"{path}: {a} vs {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} vs {b!r}")


def main() -> int:
    ref, new = (json.load(open(p)) for p in sys.argv[1:3])
    bad: list[str] = []
    walk(ref, new, "", bad)
    for line in bad[:50]:
        print(line)
    print(f"{len(bad)} value(s) outside tolerance")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
