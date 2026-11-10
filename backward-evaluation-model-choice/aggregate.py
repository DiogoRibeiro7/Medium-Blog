"""Combine the shards and compute H1-H3 (PREREGISTRATION.md) and the secondary analyses.

    python aggregate.py

Writes ``results/series_losses.csv`` (mean log loss per series, set and procedure),
``results/choices.csv`` (share of decisions in which each procedure picked each candidate),
``results/decision_quality.csv`` and ``results/summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as student_t

import common
from choice import SETS


def test(d) -> tuple[float, float, float, float]:
    """Mean, se, p(mean > 0), p(mean < 0) across series."""
    d = np.asarray(d, dtype=float)
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    tt = m / se if se > 0 else 0.0
    return m, se, float(student_t.sf(tt, len(d) - 1)), float(student_t.cdf(tt, len(d) - 1))


def fmt(r) -> str:
    return f"{r[0]:+.4f} (se {r[1]:.4f}; p>0 {r[2]:.3g}; p<0 {r[3]:.3g})"


def verdict(r, positive="SUPPORTED", negative="REVERSED") -> str:
    if r[0] > 0 and r[2] < 0.05:
        return positive
    if r[0] < 0 and r[3] < 0.05:
        return negative
    return "NOT SUPPORTED"


def groups_sign(w: pd.DataFrame, col: pd.Series) -> str:
    g = col.groupby(w["group"]).mean()
    return f"{int((g > 0).sum())} of {len(g)} lists positive (" + ", ".join(f"{k} {v:+.4f}" for k, v in g.items()) + ")"


def main() -> None:
    sh = common.OUT / "shards"
    loss = pd.concat([pd.read_csv(f) for f in sorted(sh.glob("*_losses.csv"))], ignore_index=True)
    dec = pd.concat([pd.read_csv(f) for f in sorted(sh.glob("*_decisions.csv.gz"))], ignore_index=True)
    loss["L"] = loss.sum_ll / loss.n
    loss["group"] = [common.group_of(d, s) for d, s in zip(loss.domain, loss.series)]
    loss.to_csv(common.OUT / "series_losses.csv", index=False, float_format="%.6f")
    keep = dec[dec.set.isin(("A", "B"))].dropna(axis=1, how="all")
    keep.to_csv(common.OUT / "decisions.csv.gz", index=False, float_format="%.4f",
                compression={"method": "gzip", "mtime": 0})

    # FIX: candidate with the lowest mean log loss on the development set (25 load series and
    # 304 articles pooled), per set; computed from the development shards only
    fix = {}
    for st in SETS:
        dv = loss[loss.domain.isin(("dev_elec", "dev_web")) & (loss.set == st) & loss.proc.str.startswith("CAND:")]
        fix[st] = dv.groupby("proc").L.mean().idxmin()

    lines = ["Model choice by forward backtest (FWD) and backward evaluation (BWD); PREREGISTRATION.md.",
             "L = mean log loss over the evaluation days (nats per day; lower is better). Differences are across series;",
             "p>0 / p<0 = one-sided paired t-test p-values.", ""]
    lines.append("Series: " + ", ".join(f"{d} {loss[loss.domain == d].series.nunique()}" for d in common.DOMAINS))
    lines.append("FIX (best candidate on the development set, all 329 series pooled): " + ", ".join(f"set {k}: {v[5:]}" for k, v in fix.items()))
    lines.append("")

    W = {}
    for dom in common.DOMAINS:
        for st in SETS:
            w = loss[(loss.domain == dom) & (loss.set == st)].pivot_table(index=["series", "group"], columns="proc", values="L")
            w = w.reset_index().set_index("series")
            w["FIX"] = w[fix[st]]
            W[(dom, st)] = w

    w, wb = W[("test_web", "A")], W[("test_web", "B")]
    h1 = test(w.FWD - w.BWD)
    h2 = test((w.FWD - w.BWD) - (wb.loc[w.index, "FWD"] - wb.loc[w.index, "BWD"]))
    h3a, h3b = test(w.BWD - w.ENS), test(w.FWD - w.ENS)
    lines.append("=== PRE-REGISTERED VERDICTS (test set, 245 articles) ===")
    lines.append(f"H1 L(FWD) - L(BWD), set A: {fmt(h1)} -> {verdict(h1)}")
    lines.append(f"   by list: {groups_sign(w, w.FWD - w.BWD)}")
    lines.append(f"H2 [L(FWD) - L(BWD)]_A - [L(FWD) - L(BWD)]_B: {fmt(h2)} -> {verdict(h2)}")
    lines.append(f"   set B alone, L(FWD) - L(BWD): {fmt(test(wb.FWD - wb.BWD))}")
    h3 = "SUPPORTED" if (h3a[0] > 0 and h3a[2] < 0.05 and h3b[0] > 0 and h3b[2] < 0.05) else "NOT SUPPORTED"
    lines.append(f"H3 L(BWD) - L(ENS): {fmt(h3a)}; L(FWD) - L(ENS): {fmt(h3b)} -> {h3}")
    lines.append(f"   by list, BWD - ENS: {groups_sign(w, w.BWD - w.ENS)}")
    lines.append(f"   by list, FWD - ENS: {groups_sign(w, w.FWD - w.ENS)}")
    lines.append("")

    procs = ["FWD", "BWD", "BWD1", "HYB", "ENS", "FIX", "ORACLE"]
    lines.append("=== Mean log loss by domain, set and procedure (and paired differences against FWD) ===")
    for dom in common.DOMAINS:
        for st in SETS:
            x = W[(dom, st)]
            cands = [c for c in x.columns if str(c).startswith("CAND:")]
            lines.append(f"[{dom} / set {st}] {len(x)} series")
            lines.append("  mean L: " + "; ".join(f"{p} {x[p].mean():.4f}" for p in procs) + " | candidates: "
                         + "; ".join(f"{c[5:]} {x[c].mean():.4f}" for c in cands))
            lines.append("  FWD minus: " + "; ".join(f"{p} {fmt(test(x.FWD - x[p]))}" for p in ("BWD", "BWD1", "HYB", "ENS", "FIX")))
    lines.append("")

    lines.append("=== Decisions: share choosing the period's best candidate; mean regret; share changing choice; "
                 "Spearman(score, next-period loss) ===")
    qual = []
    for (dom, st, proc), d in dec.groupby(["domain", "set", "proc"]):
        cands = SETS[st]
        hit = float((d.choice == d.best).mean())
        regret = float((d.realised_choice - d.realised_best).mean())
        d = d.sort_values(["series", "date"])
        change = float((d.groupby("series").choice.shift() != d.choice)[d.groupby("series").cumcount() > 0].mean())
        sc = d[[f"score:{c}" for c in cands]].to_numpy()
        rl = d[[f"real:{c}" for c in cands]].to_numpy()
        rhos = [spearmanr(a, b)[0] for a, b in zip(sc, rl) if np.ptp(a) > 0 and np.ptp(b) > 0]
        qual.append({"domain": dom, "set": st, "proc": proc, "hit": hit, "regret": regret, "change": change,
                     "spearman": float(np.nanmean(rhos))})
    q = pd.DataFrame(qual)
    q.to_csv(common.OUT / "decision_quality.csv", index=False, float_format="%.4f")
    for dom in common.DOMAINS:
        for st in SETS:
            r = q[(q.domain == dom) & (q.set == st)].set_index("proc")
            lines.append(f"  {dom} set {st}: " + "; ".join(
                f"{p} hit {r.loc[p, 'hit']:.2f}, regret {r.loc[p, 'regret']:.4f}, change {r.loc[p, 'change']:.2f}, rho {r.loc[p, 'spearman']:+.2f}"
                for p in ("FWD", "BWD", "BWD1", "HYB")))
    lines.append("")

    lines.append("=== Which candidate each procedure picked (share of decisions), set A ===")
    ch = dec[dec.set == "A"].groupby(["domain", "proc"]).choice.value_counts(normalize=True).rename("share").reset_index()
    ch.to_csv(common.OUT / "choices.csv", index=False, float_format="%.4f")
    for dom in common.DOMAINS:
        for p in ("FWD", "BWD"):
            c = ch[(ch.domain == dom) & (ch.proc == p)]
            lines.append(f"  {dom} {p}: " + "; ".join(f"{r.choice} {100 * r.share:.0f}%" for r in c.itertuples()))
    lines.append("")

    lines.append("=== Test set by list, set A: mean L of FWD, BWD, ENS ===")
    for grp, x in w.groupby("group"):
        lines.append(f"  {grp} ({len(x)}): FWD {x.FWD.mean():.4f}; BWD {x.BWD.mean():.4f}; ENS {x.ENS.mean():.4f}; "
                     f"FWD - BWD {(x.FWD - x.BWD).mean():+.4f}")
    (common.OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
