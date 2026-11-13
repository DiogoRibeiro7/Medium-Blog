"""Descriptive numbers the article quotes that evaluate.py does not print (not part of the registered test).

python details.py --results results --out results/details.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median, pstdev

from feed import detect as D
from feed import regimes as R

MOTIVATING = "citibike / DOT table"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/details.txt"))
    a = ap.parse_args()
    pairs = json.loads((a.results / "pairs.json").read_text(encoding="utf-8"))
    L = ["Descriptive numbers quoted in the article (not part of the registered test).", ""]

    tests = [n for n in pairs if n != MOTIVATING]
    closer = [n for n in tests if pairs[n]["err_segment_rule"] < pairs[n]["err_full_rule"]]
    full_within = [n for n in tests if pairs[n]["within_full_rule"]]
    L.append(
        f"Later segment, test pairs: segment rule closer than the full-history rule in {len(closer)} of "
        f"{len(tests)} ({', '.join(closer)})"
    )
    L.append(
        f"Later segment, test pairs: full-history rule also within a factor of two in {len(full_within)} of "
        f"{len(tests)}"
    )
    L.append("")

    with open(a.results / "windows.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for label, keep in (("test pairs", lambda n: n != MOTIVATING), ("motivating pair", lambda n: n == MOTIVATING)):
        sel = [r for r in rows if keep(r["pair"])]
        obs = [float(r["observed"]) if r["observed"] else D.GRID[-1] for r in sel]
        lt = [math.log(o / float(r["pred_trailing"])) for o, r in zip(obs, sel, strict=True)]
        le = [math.log(o / float(r["pred_expanding"])) for o, r in zip(obs, sel, strict=True)]
        within = sum(
            D.within_factor_two(float(r["pred_trailing"]), float(r["observed"]) if r["observed"] else None) for r in sel
        )
        L.append(f"Forward windows, {label}: {len(sel)}; trailing rule within a factor of two in {within}")
        L.append(
            f"  log(observed / trailing): median {median(lt):+.3f}, mean {sum(lt) / len(lt):+.3f}, "
            f"standard deviation {pstdev(lt):.3f} (a factor of {math.exp(pstdev(lt)):.1f})"
        )
        L.append(f"  log(observed / expanding): median {median(le):+.3f}, standard deviation {pstdev(le):.3f}")
        L.append(
            f"  misses with the observed value above twice the trailing rule: {sum(x > math.log(2) for x in lt)}; "
            f"below half of it: {sum(x < -math.log(2) for x in lt)}"
        )
        L.append("")

    n_trail, n_ahead = R.TRAIL, R.AHEAD
    pos = 0.95 * (n_trail - 1)
    allowed = math.floor(D.BUDGET * n_ahead)
    need = math.ceil(D.TARGET * n_ahead)
    L.append(
        f"Order statistics: the 95th percentile of {n_trail} deviations sits at position {pos:.2f} of "
        f"0..{n_trail - 1}: between the deviations ranked {n_trail - math.floor(pos)} and {n_trail - math.ceil(pos)} "
        f"from the top"
    )
    L.append(
        f"Over {n_ahead} months the matched budget allows {allowed} clean alarm(s), and 90% detection needs "
        f"{need} of {n_ahead} months caught, so at most {n_ahead - need} months may be missed"
    )
    (a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
