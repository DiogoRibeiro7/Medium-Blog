"""What the method recovers when the answer is known: synthetic sales with and without lock-in.

    python validate.py     # writes results/validation.txt

Builds synthetic Price Paid Data (tests/synth.py) twice: with owners whose home's value is below
the price they paid selling at 40% of the normal monthly rate, and with no such effect. The
synthetic market has the same boom, bust and recovery shape as England and Wales, and the
planted lock-in is the only reason for its resale rate to change. Each is run through the same
build and analysis as the real data, and the curve and counterfactuals are compared with the
truth.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

import analysis as A  # noqa: E402
import build as B  # noqa: E402
import synth  # noqa: E402

HOMES = 20000
A.MIN_BIN_STOCK = 1000


def run(lock_in: float, tmp: Path) -> dict:
    ppd = tmp / f"pp_{lock_in}.csv"
    synth.make_ppd(ppd, homes_per_area=HOMES, seed=23, lock_in=lock_in)
    data = tmp / f"data_{lock_in}"
    B.build(str(ppd), data, memory_limit="2GB", threads=2, min_area_pairs=100)
    hc = pd.read_csv(data / "hazard_cells.csv.gz")
    curve = A.lock_in_curve(hc, 2008, 2012)
    eff = curve.set_index("m_bin")
    ref = float(eff.ref_rate.iloc[0])
    cf = A.counterfactual(hc[(hc.year >= 2007) & (hc.year <= 2012)], curve).set_index("year")
    fall = cf.rate[2007] - cf.rate[2009]
    narrow = (fall - (cf.rate_without_lock_in[2007] - cf.rate_without_lock_in[2009])) / fall
    wide = (fall - (cf.rate_without_position[2007] - cf.rate_without_position[2009])) / fall
    rel = {b: float(eff.effect.get(b, np.nan) / ref) for b in (-5, -3, -1, 1, 5, 8)}
    return {"lock_in": lock_in, "ref_rate": ref, "rel": rel, "fall": fall,
            "rate_2007": cf.rate[2007], "rate_2009": cf.rate[2009], "narrow": narrow, "wide": wide,
            "share_under_2009": cf.share_under[2009]}


def main() -> int:
    lines = ["VALIDATION ON SYNTHETIC SALES (tests/synth.py, 4 areas x 20,000 homes, seed 23)", "=" * 76]
    with tempfile.TemporaryDirectory() as d:
        for li in (0.4, 1.0):
            r = run(li, Path(d))
            label = ("planted: owners below their purchase price sell at 40% of the normal rate"
                     if li < 1 else "null: no effect of the purchase price")
            lines.append("")
            lines.append(label)
            lines.append(f"  resale rate 2007 {100 * r['rate_2007']:.2f}%, 2009 {100 * r['rate_2009']:.2f}% "
                         f"(fall {100 * r['fall'] / r['rate_2007']:.0f}%); share under water at the start of 2009 "
                         f"{100 * r['share_under_2009']:.0f}%")
            lines.append("  estimated 2008-2012 curve, relative to owners at 0 to +2%: " + "; ".join(
                f"[{2 * b:+d}%] {100 * v:+.0f}%" for b, v in r["rel"].items()))
            if r["fall"] / r["rate_2007"] > 0.05:
                lines.append(f"  share of the 2007-2009 fall attributed: narrow {100 * r['narrow']:.0f}%, "
                             f"wide {100 * r['wide']:.0f}% (true share: 100%, the planted effect is the only cause)")
            else:
                vals = list(r["rel"].values())
                lines.append(f"  no fall to attribute; the curve varies by {100 * (max(vals) - min(vals)):.0f} points "
                             f"across the bands shown")
    text = "\n".join(lines) + "\n"
    (HERE / "results" / "validation.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
