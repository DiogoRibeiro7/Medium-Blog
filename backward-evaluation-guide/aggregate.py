"""Combine the shards of ``daily.py``; the verdict on T1 (PLAN.md) and the readings.

    python aggregate.py

Writes ``results/daily_summary.txt``, ``results/daily_lastrow.csv`` (today's model on the
past, per series and family), ``results/daily_mean_matrix.csv`` (the matrices averaged over
the series of a domain), ``results/daily_monitor.csv`` (monitoring, averaged by date) and
``results/daily_forward.csv`` (deployed log loss per series).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SH = RES / "shards"
DOMAINS = ("elec", "dev_web", "test_web")
DOM = {"elec": "electricity, 25 countries", "dev_web": "Wikipedia, 304 development articles",
       "test_web": "Wikipedia, 245 test articles"}
FAMILIES = ("AR", "SLM", "ETS", "GBM", "GBMn")
PARTS = ("excess", "level", "spread", "structure")
EXAMPLES = (("elec", "IT"), ("dev_web", "country:Ukraine"), ("test_web", "sport:Formula One"))


def ttest(d) -> tuple[float, float, float, float]:
    d = np.asarray(d, float)
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    tt = m / se if se > 0 else 0.0
    return m, se, float(student_t.sf(tt, len(d) - 1)), float(student_t.cdf(tt, len(d) - 1))


def fp(p: float, digits: int = 3) -> str:
    """A p-value for the summary. Values below 1e-6 are reported as such: their mantissa moves
    between runs (LightGBM differs across machines in the fourth decimal) and means nothing."""
    return "<1e-6" if p < 1e-6 else f"{p:.{digits}g}"


def group(domain: str, key: str) -> str:
    return "country" if domain == "elec" else key.split(":", 1)[0] if ":" in key else "earlier"


def read(kind: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted(SH.glob(f"*_{kind}.csv.gz"))], ignore_index=True)


def main() -> None:
    cells, mon, fwd = read("cells"), read("monitor"), read("forward")
    lines = ["Daily data: train-period x test-period matrices (blocks of 364 days, 28-day gap, Student-t(5)),",
             "monitoring, and the forward check of PLAN.md. Log loss in nats per day; parts sum to the excess.", ""]

    # today's model (last block) on every earlier block
    last = cells[cells.train == cells.k - 1]
    per = last.groupby(["domain", "series", "family"])[list(PARTS)].mean().reset_index()
    per.to_csv(RES / "daily_lastrow.csv", index=False, float_format="%.5f")
    lines.append("=== Today's model (fitted on the last 364 days) on every earlier year: mean over years, then over series ===")
    for dom in DOMAINS:
        x = per[per.domain == dom]
        lines.append(f"[{DOM[dom]}]")
        for fam in FAMILIES:
            y = x[x.family == fam]
            m = y[list(PARTS)].mean()
            se = y.excess.std(ddof=1) / np.sqrt(len(y))
            lines.append(f"  {fam:>4}: excess {m.excess:+.3f} (se {se:.3f}) = level {m.level:.3f} + spread {m.spread:.3f} "
                         f"+ structure {m.structure:+.3f}; series with excess > 0: {100 * (y.excess > 0).mean():.0f}%")
    lines.append("")

    # every cell, not only today's model
    lines.append("=== All cells (every year's model on every other year): mean over cells, then over series ===")
    allc = cells.groupby(["domain", "series", "family"])[list(PARTS)].mean().reset_index()
    for dom in DOMAINS:
        x = allc[allc.domain == dom]
        lines.append(f"[{DOM[dom]}] " + "; ".join(
            f"{fam} {x[x.family == fam].excess.mean():+.3f} = {x[x.family == fam].level.mean():.3f} + "
            f"{x[x.family == fam].spread.mean():.3f} + {x[x.family == fam].structure.mean():+.3f}" for fam in FAMILIES))
    lines.append("")

    mm = cells.groupby(["domain", "family", "train_label", "test_label"])[list(PARTS)].mean().reset_index()
    mm.to_csv(RES / "daily_mean_matrix.csv", index=False, float_format="%.5f")
    ex = pd.concat([cells[(cells.domain == d) & (cells.series == k)] for d, k in EXAMPLES], ignore_index=True)
    ex.to_csv(RES / "daily_example_cells.csv", index=False, float_format="%.5f")

    # monitoring
    mon["date"] = pd.to_datetime(mon.date)
    mg = mon.groupby(["domain", "date"]).agg(**{p: (p, "mean") for p in PARTS},
                                             share_over=("excess", lambda v: float((v > 0.05).mean())),
                                             n=("excess", "size")).reset_index()
    mg.to_csv(RES / "daily_monitor.csv", index=False, float_format="%.5f")
    lines.append("=== Monitoring (seasonal linear model, last 364 days on the 364 days before a 28-day gap) ===")
    for dom in DOMAINS:
        x = mg[mg.domain == dom].set_index("date")
        top = x.excess.nlargest(3)
        lines.append(f"[{DOM[dom]}] {len(x)} origins, {x.index.min():%Y-%m} to {x.index.max():%Y-%m}; mean excess "
                     f"{x.excess.mean():+.3f}; largest: " + "; ".join(
                         f"{d:%Y-%m-%d} {v:+.3f} (level {x.loc[d, 'level']:.3f}, spread {x.loc[d, 'spread']:.3f}, "
                         f"structure {x.loc[d, 'structure']:+.3f}; {100 * x.loc[d, 'share_over']:.0f}% of series above 0.05)"
                         for d, v in top.items()))
        by_year = x.groupby(x.index.year)[list(PARTS)].mean()
        lines.append("  by year of the origin: " + "; ".join(
            f"{y} {r.excess:+.3f} ({r.level:.3f}/{r.spread:.3f}/{r.structure:+.3f})" for y, r in by_year.iterrows()))
    lines.append("")

    # the forward check (T1)
    fwd["L"] = fwd.sum_ll / fwd.n
    w = fwd.pivot_table(index=["domain", "series"], columns="family", values="L").reset_index()
    w["group"] = [group(d, s) for d, s in zip(w.domain, w.series)]
    w.to_csv(RES / "daily_forward.csv", index=False, float_format="%.6f")
    lines.append("=== Forward check (PLAN.md, T1): deployed log loss, GBM minus GBMn (positive: GBMn better) ===")
    for dom in DOMAINS:
        x = w[w.domain == dom]
        r = ttest(x.GBM - x.GBMn)
        verdict = ""
        if dom == "test_web":
            verdict = (" -> SUPPORTED" if r[0] > 0 and r[2] < 0.05 else " -> REVERSED" if r[0] < 0 and r[3] < 0.05
                       else " -> NOT SUPPORTED")
        lines.append(f"[{DOM[dom]}] mean L: GBM {x.GBM.mean():.4f}, GBMn {x.GBMn.mean():.4f}, SLM {x.SLM.mean():.4f}; "
                     f"GBM - GBMn {r[0]:+.4f} (se {r[1]:.4f}; p>0 {fp(r[2])}; p<0 {fp(r[3])}); GBMn better in "
                     f"{int((x.GBM > x.GBMn).sum())} of {len(x)}{verdict}")
        if dom != "elec":
            g = (x.GBM - x.GBMn).groupby(x.group).mean()
            lines.append(f"  by list: {int((g > 0).sum())} of {len(g)} positive (" + ", ".join(
                f"{k} {v:+.4f}" for k, v in g.items()) + ")")
        s = ttest(x.SLM - x.GBMn)
        lines.append(f"  SLM - GBMn {s[0]:+.4f} (se {s[1]:.4f}; p>0 {fp(s[2])}; p<0 {fp(s[3])}); GBMn better than SLM in "
                     f"{int((x.SLM > x.GBMn).sum())} of {len(x)}")
        # does the reading say in advance who gains? (exploratory)
        lr = per[(per.domain == dom) & (per.family == "GBM")].set_index("series").excess
        gain = (x.set_index("series").GBM - x.set_index("series").GBMn).reindex(lr.index)
        rho, p = spearmanr(lr, gain)
        lines.append(f"  exploratory: Spearman(GBM's excess on the past, forward gain of GBMn) {rho:+.2f} (p {fp(p, 2)})")
    (RES / "daily_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
