"""Descriptive analyses added after the results (PLAN.md, "Descriptive analyses added after the results").

python details.py --sources sources --results results --out results/details.txt

Not part of the registered test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import AHTS, weeks_used
from feed import erlang as E
from feed import forecast as F
from feed import schedule as S
from feed.calls import evaluation_weeks, load, low_days

AHT = AHTS[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/details.txt"))
    a = ap.parse_args()
    c = load(a.sources / "hourly_calls.csv")
    weeks = weeks_used(c)
    last = evaluation_weeks()[0] - pd.Timedelta(days=7)
    model, hol = F.fit_gbm(c, last)
    actual = [F.simple(c, w, "oracle") for w in weeks]
    fc = {
        "two_weeks": [F.simple(c, w, "two_weeks") for w in weeks],
        "mean4": [F.simple(c, w, "mean4") for w in weeks],
        "gbm": [F.gbm_forecast(model, hol, c, w) for w in weeks],
    }
    act = np.concatenate(actual)
    L = ["Descriptive analyses added after the results (not part of the registered test). AHT 5 minutes.", ""]
    js: dict = {}

    L.append("1. Bias of the forecasts over the test weeks: total forecast / total calls - 1")
    js["bias"] = {}
    for n, f in fc.items():
        b = float(np.concatenate(f).sum() / act.sum() - 1)
        js["bias"][n] = b
        L.append(f"   {n}: {b:+.4f}")
    y = pd.Series(c.calls, index=c.hours)
    yearly = y.groupby(y.index.year).sum()
    test = y[(y.index >= weeks[0]) & (y.index < weeks[-1] + pd.Timedelta(days=7))]
    js["hour_profile_test"] = (test.groupby(test.index.hour).sum() / test.sum()).tolist()
    L.append("   calls by year: " + ", ".join(f"{k} {int(v)}" for k, v in yearly.items()))
    L.append("")

    need = [E.requirement(x, AHT) for x in actual]
    L.append("2. Where the constraint cost sits: excess agent-hours with the oracle, by hour of day, as a share of")
    L.append("   each regime's total excess over the test weeks; and the share of excess in hours 07:00 to 10:59")
    js["excess_by_hour"] = {}
    duals = {}
    for regime in ("R1", "R2", "R3", "R4"):
        ex = np.zeros(24)
        pos = np.zeros(24)
        val = np.zeros(24)
        for nd in need:
            on, _ = S.staffing(regime, nd)
            e = np.maximum(on - nd, 0)
            ex += e.reshape(7, 24).sum(axis=0)
            if regime in ("R1", "R4"):
                sp = S.shadow_prices(regime, nd)
                pos += (sp > 1e-9).reshape(7, 24).sum(axis=0)
                val += sp.reshape(7, 24).sum(axis=0)
        share = ex / ex.sum()
        js["excess_by_hour"][regime] = share.tolist()
        L.append(
            f"   {regime}: morning 07-10 {share[7:11].sum():.3f}; night 00-05 {share[0:6].sum():.3f}; "
            f"evening 17-23 {share[17:24].sum():.3f}; peak hour {int(np.argmax(share))}:00"
        )
        if regime in ("R1", "R4"):
            duals[regime] = {"positive_weeks": (pos / len(need)).tolist(), "mean_value": (val / len(need)).tolist()}
    L.append("")
    L.append(
        "3. Binding hours: share of test weeks in which an hour of the day has a positive shadow price on some day,"
    )
    L.append("   averaged over the 7 days, and the mean shadow price summed over the week, oracle requirements")
    for regime, d in duals.items():
        pw = np.array(d["positive_weeks"]) / 7
        top = np.argsort(-pw)[:5]
        L.append(
            f"   {regime}: hours most often binding: "
            + ", ".join(f"{h}:00 {pw[h]:.2f}" for h in top)
            + f"; binding hours per week on average {np.array(d['positive_weeks']).sum():.1f} of 168"
        )
    js["duals"] = duals
    L.append("")

    low = low_days(c)
    gap_weeks = sorted({(d - pd.Timedelta(days=d.dayofweek)).normalize() for d in low.index})
    L.append(
        "4. The gradient-boosted model refitted without the training weeks that hold the low days "
        f"({', '.join(str(w.date()) for w in gap_weeks)})"
    )
    m2, _ = F.fit_gbm(c, last, exclude=gap_weeks)
    g2 = [F.gbm_forecast(m2, hol, c, w) for w in weeks]
    w1 = float(np.abs(np.concatenate(fc["gbm"]) - act).sum() / act.sum())
    w2 = float(np.abs(np.concatenate(g2) - act).sum() / act.sum())
    b2 = float(np.concatenate(g2).sum() / act.sum() - 1)
    L.append(f"   WAPE {w1:.4f} -> {w2:.4f}; bias {js['bias']['gbm']:+.4f} -> {b2:+.4f}")
    js["gbm_without_gap"] = {"wape": w2, "bias": b2, "regimes": {}}
    for regime in S.REGIMES:
        tot = {}
        for name, f in (("as registered", fc["gbm"]), ("without the gap weeks", g2)):
            ex = sh = nd_tot = 0.0
            for fw, nd in zip(f, need, strict=True):
                on, _ = S.staffing(regime, E.requirement(fw, AHT))
                ex += np.maximum(on - nd, 0).sum()
                sh += np.maximum(nd - on, 0).sum()
                nd_tot += nd.sum()
            tot[name] = ((ex + sh) / nd_tot, (ex + 3 * sh) / nd_tot)
        js["gbm_without_gap"]["regimes"][regime] = tot
        r, g = tot["as registered"], tot["without the gap weeks"]
        L.append(f"   {regime}: cost k=1 {r[0]:.4f} -> {g[0]:.4f}; k=3 {r[1]:.4f} -> {g[1]:.4f}")
    L.append("")

    with open(a.results / "cells.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if float(r["aht"]) == AHT]
    L.append(
        "5. Order of the three forecasts by cost under each regime, against their order by WAPE (gbm, mean4, two_weeks)"
    )
    for k in ("1", "3"):
        for regime in S.REGIMES:
            sel = [r for r in rows if r["regime"] == regime and r["k"] == k and r["forecast"] != "oracle"]
            order = [r["forecast"] for r in sorted(sel, key=lambda r: float(r["cost"]))]
            L.append(f"   k {k}, {regime}: {', '.join(order)}")
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (a.out.parent / "details.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
