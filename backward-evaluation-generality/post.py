"""Descriptive summaries derived from the committed results (no new estimation).

    python post.py

Reads ``results/series_gains.csv``, ``results/decisions.csv`` and ``results/periods_mean.csv`` and
writes ``results/post_summary.txt``:

* per cell, the share of series in which pooling over windows beat the plain and the checked rule,
  and in which the checked rule beat the plain rule;
* the number of series in which the walk never stopped (so that B equals E);
* the share of decisions in which the walk stopped and the check switched, by year;
* electricity: gains over the expanding window by 91-day period from 2019 (seasonal linear model).
"""

from __future__ import annotations

import pandas as pd

import common
from models import FAMILIES


def main() -> None:
    g = pd.read_csv(common.OUT / "series_gains.csv")
    dec = pd.read_csv(common.OUT / "decisions.csv", parse_dates=["date"])
    pm = pd.read_csv(common.OUT / "periods_mean.csv", parse_dates=["period"])
    lines = ["Descriptive summaries derived from series_gains.csv, decisions.csv and periods_mean.csv.", ""]

    lines.append("Share of series (Student-t / Gaussian): AveW > B; AveW > C0; C0 > B; series with B = E (walk never stopped)")
    for dom in common.DOMAINS:
        for fam in FAMILIES:
            parts = []
            for dist in ("t", "gauss"):
                w = g[(g.domain == dom) & (g.family == fam) & (g.dist == dist)].pivot_table(index="series", columns="scheme", values="G")
                parts.append(f"{dist}: {100 * (w.AveW > w.B).mean():.0f}%, {100 * (w.AveW > w.C0).mean():.0f}%, "
                             f"{100 * (w.C0 > w.B).mean():.0f}%, {int((w.B == 0).sum())} of {len(w)}")
            lines.append(f"  {dom} {fam}: " + " | ".join(parts))
    lines.append("")

    lines.append("The check (c = 0) under each distribution: series where B loses / gains, share of losses removed, share of gains kept")
    for dom in common.DOMAINS:
        for fam in FAMILIES:
            parts = []
            for dist in ("gauss", "t"):
                w = g[(g.domain == dom) & (g.family == fam) & (g.dist == dist)].pivot_table(index="series", columns="scheme", values="G")
                neg, pos = w.B < 0, w.B > 0
                rem = 100 * (1 - w.C0[neg].sum() / w.B[neg].sum()) if neg.any() else float("nan")
                kept = 100 * w.C0[pos].sum() / w.B[pos].sum() if pos.any() else float("nan")
                parts.append(f"{dist}: {int(neg.sum())}/{int(pos.sum())}, {rem:.0f}% removed, {kept:.0f}% kept")
            lines.append(f"  {dom} {fam}: " + " | ".join(parts))
    lines.append("")

    lines.append("Decisions by year: share of series-decisions where the walk stopped / where the check (c = 0) switched")
    dec["year"] = dec.date.dt.year
    for dom in common.DOMAINS:
        for dist in ("t", "gauss"):
            d = dec[(dec.domain == dom) & (dec.dist == dist)].groupby("year").agg(
                stop=("stopped_at", lambda s: (s > 0).mean()), sw=("switch_c0", "mean"))
            lines.append(f"  {dom} {dist}: " + "; ".join(f"{y} {100 * r.stop:.0f}/{100 * r.sw:.0f}" for y, r in d.iterrows()))
    lines.append("")

    lines.append("Blocks back at which the walk stopped (share of stopped decisions), Student-t")
    for dom in common.DOMAINS:
        d = dec[(dec.domain == dom) & (dec.dist == "t") & (dec.stopped_at > 0)]
        vc = d.stopped_at.value_counts(normalize=True).sort_index()
        lines.append(f"  {dom}: " + "; ".join(f"{int(k)} block{'s' if k > 1 else ''} {100 * v:.0f}%" for k, v in vc.items()))
    lines.append("")

    lines.append("Electricity, seasonal linear model, Student-t: gain over E by 91-day period (mean across series), from 2019")
    q = pm[(pm.domain == "elec") & (pm.family == "SLM") & (pm.dist == "t")].pivot(index="period", columns="scheme", values="ll")
    q = q[q.index >= "2019-01-01"]
    for per, r in q.iterrows():
        lines.append(f"  {per:%Y-%m-%d}: " + "; ".join(f"{k} {r.E - r[k]:+.4f}" for k in ("B", "C0", "P0.25", "AveW")))
    (common.OUT / "post_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
