"""Descriptive summaries derived from the saved results (no new estimation).

    python post.py

Reads ``results/forward_scores.csv.gz`` and ``results/forward_decisions.csv`` and writes
``results/post_summary.txt``:

* decisions: how often the walk stopped and how often the check switched, by error distribution;
* the period after COVID-19 (forecasts from January 2021): gains of B and C over E on each test
  series, the window in which the check was designed to act;
* attribution: the plain rule's total loss on the series where it loses, Gaussian versus Student-t,
  and how much of it the check removes under each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
from backward_eval import mean_and_se


def main() -> None:
    sc = pd.read_csv(common.OUT / "forward_scores.csv.gz", parse_dates=["date"])
    dec = pd.read_csv(common.OUT / "forward_decisions.csv")
    lines = ["Descriptive summaries derived from forward_scores.csv.gz and forward_decisions.csv.", ""]

    lines.append("Decisions on the test series (yearly re-selections, 1990 onwards)")
    for errors in ("t", "gaussian"):
        d = dec[(dec.set == "test") & (dec.errors == errors)]
        st = d[d.stopped]
        lines.append(f"  {errors}: {len(d)} decisions; walk stopped in {len(st)}; check switched in {int(d.check_switched.sum())}; "
                     f"median z_end when stopped {st.z_end12.median():+.2f}; z_end > 0 in {int((st.z_end12 > 0).sum())}")
    for errors in ("t", "gaussian"):
        d = dec[(dec.set == "dev") & (dec.errors == errors)]
        lines.append(f"  development set, {errors}: {len(d)} decisions; walk stopped in {int(d.stopped.sum())}; "
                     f"check switched in {int(d.check_switched.sum())}")
    lines.append("")

    lines.append("After COVID-19: gain over E of B and C for forecasts from 2021-01 (nats per month, Newey-West se)")
    for which, keys in (("test", list(common.TEST)), ("dev", list(common.DEV))):
        for errors in ("t", "gaussian"):
            sub = sc[(sc.set == which) & (sc.errors == errors) & (sc.date >= "2021-01-01")]
            parts = []
            tot = {"B": [], "C": []}
            for k in keys:
                g = sub[sub.series == k]
                if not len(g):
                    continue
                mb, sb = mean_and_se((g.E - g.B).to_numpy())
                mc, scc = mean_and_se((g.E - g.C).to_numpy())
                tot["B"].append(mb)
                tot["C"].append(mc)
                parts.append(f"{common.SHORT[k]} B {mb:+.4f} ({sb:.4f}) C {mc:+.4f} ({scc:.4f}) n {len(g)}")
            lines.append(f"  [{which}, {errors}] mean over series: B {np.mean(tot['B']):+.4f}, C {np.mean(tot['C']):+.4f}")
            lines += ["    " + p for p in parts]
    lines.append("")

    lines.append("Attribution on the test series (scoring without Mar-Dec 2020): total gain of B and C over E "
                 "on the series where B loses under each error distribution")
    gf = pd.read_csv(common.OUT / "forward_gains.csv")
    for errors in ("gaussian", "t"):
        g = gf[(gf.set == "test") & (gf.errors == errors) & (gf.scoring == "no_covid")]
        tab = g.pivot(index="series", columns="scheme", values="gain")
        neg = tab[tab.B < 0]
        lines.append(f"  {errors}: B loses on {len(neg)} series, total {neg.B.sum():+.4f}; C total there {neg.C.sum():+.4f} "
                     f"(removed {100 * (1 - neg.C.sum() / neg.B.sum()):.0f}%); mean over all ten: B {tab.B.mean():+.4f}, "
                     f"C {tab.C.mean():+.4f}, AveW {tab.AveW.mean():+.4f}")
    (common.OUT / "post_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
