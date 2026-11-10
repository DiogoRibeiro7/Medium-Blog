"""Exploratory follow-up (not pre-registered), from ``results/`` and the data.

    python post.py

1. When FWD and BWD disagree: how often each one's pick does better over the next quarter.
2. Does the backward choice lose less on articles whose level moves most between years?
   Drift = standard deviation of the yearly means of log(1 + views), over the mean within-year
   standard deviation; Spearman correlation with L(FWD) - L(BWD), set A.
3. Does the gap depend on how much past the backward score has? Gap by decision year.
4. Which candidate the backward score ranks first when the forward backtest picks the seasonal linear model.

Writes ``results/post_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common


def drift(s: pd.Series) -> float:
    g = s.groupby(s.index.year)
    g = [x for _, x in g if len(x) >= 300]
    means = np.array([x.mean() for x in g])
    within = np.mean([x.std() for x in g])
    return float(means.std(ddof=1) / within)


def main() -> None:
    loss = pd.read_csv(common.OUT / "series_losses.csv")
    dec = pd.read_csv(common.OUT / "decisions.csv.gz")
    out = ["Exploratory follow-up (not pre-registered).", ""]

    out.append("1. Decisions where FWD and BWD pick different candidates (set A, B)")
    for (dom, st), d in dec.groupby(["domain", "set"]):
        f = d[d.proc == "FWD"].set_index(["series", "date"])
        b = d[d.proc == "BWD"].set_index(["series", "date"])
        j = f[["choice", "realised_choice"]].join(b[["choice", "realised_choice"]], rsuffix="_b")
        dis = j[j.choice != j.choice_b]
        diff = dis.realised_choice_b - dis.realised_choice
        out.append(f"   {dom} set {st}: disagree in {100 * len(dis) / len(j):.0f}% of {len(j)} decisions; "
                   f"BWD's pick better next quarter in {100 * (diff < 0).mean():.0f}% of those; "
                   f"mean next-quarter loss of BWD's pick minus FWD's {diff.mean():+.4f} nats/day "
                   f"(median {diff.median():+.4f})")
    out.append("")

    out.append("2. Level drift between years and the gap L(FWD) - L(BWD), set A")
    for dom in ("dev_web", "test_web"):
        data = common.load(dom)
        w = loss[(loss.domain == dom) & (loss.set == "A")].pivot_table(index="series", columns="proc", values="L")
        dr = pd.Series({k: drift(data[k]) for k in w.index})
        gap = w.FWD - w.BWD
        rho, p = spearmanr(dr, gap)
        q = pd.qcut(dr, 4, labels=["lowest", "second", "third", "highest"])
        by = gap.groupby(q, observed=True).agg(["mean", "count"])
        out.append(f"   {dom}: Spearman(drift, gap) {rho:+.2f} (p {p:.2g}); median drift {dr.median():.2f}")
        out.append("     gap by drift quartile: " + "; ".join(f"{k} {r['mean']:+.4f} ({int(r['count'])})" for k, r in by.iterrows()))
        share = (gap > 0).groupby(q, observed=True).mean()
        out.append("     share of articles where BWD did better, by quartile: " + "; ".join(f"{k} {100 * v:.0f}%" for k, v in share.items()))
    out.append("")

    out.append("3. Next-quarter loss of BWD's pick minus FWD's pick, by decision year (set A)")
    for dom in common.DOMAINS:
        d = dec[(dec.domain == dom) & (dec.set == "A")]
        f = d[d.proc == "FWD"].set_index(["series", "date"]).realised_choice
        b = d[d.proc == "BWD"].set_index(["series", "date"]).realised_choice
        g = (b - f).rename("d").reset_index()
        g["year"] = pd.to_datetime(g.date).dt.year
        r = g.groupby("year").d.mean()
        out.append(f"   {dom}: " + "; ".join(f"{y} {v:+.4f}" for y, v in r.items()))
    out.append("")

    out.append("4. When FWD picks SLM_t: what BWD picks instead (set A, share of those decisions)")
    for dom in common.DOMAINS:
        d = dec[(dec.domain == dom) & (dec.set == "A")]
        f = d[d.proc == "FWD"].set_index(["series", "date"]).choice
        b = d[d.proc == "BWD"].set_index(["series", "date"]).choice
        x = b[f == "SLM_t"].value_counts(normalize=True)
        out.append(f"   {dom}: " + "; ".join(f"{k} {100 * v:.0f}%" for k, v in x.items()))
    out.append("")

    out.append("5. Mean backward score minus mean forward score of each candidate (set A; how much worse each looks on the past)")
    for dom in common.DOMAINS:
        d = dec[(dec.domain == dom) & (dec.set == "A")]
        f = d[d.proc == "FWD"]
        b = d[d.proc == "BWD"]
        cols = [c for c in d.columns if c.startswith("score:") and d[c].notna().any()]
        out.append(f"   {dom}: " + "; ".join(f"{c[6:]} {b[c].mean() - f[c].mean():+.3f}" for c in cols))
    (common.OUT / "post_summary.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
