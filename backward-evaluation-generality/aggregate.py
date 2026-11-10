"""Combine the shards and compute the pre-registered findings F1-F3 (PREREGISTRATION.md).

    python aggregate.py

Writes ``results/series_gains.csv`` (gain over E per series, family, distribution and scheme, all
days and without March-June 2020), ``results/periods_mean.csv`` (cross-series mean log loss by
91-day period), ``results/decisions.csv``, ``results/cells.csv`` and ``results/summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import common
from models import FAMILIES

SCHEMES = ("E", "B", "C0", "C1645", "P0.25", "AveW")


def mean_test(d: np.ndarray) -> tuple[float, float, float]:
    """Mean, standard error and one-sided p-value for mean > 0 across series."""
    d = np.asarray(d, dtype=float)
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    p = float(student_t.sf(m / se, len(d) - 1)) if se > 0 else float("nan")
    return m, se, p


def load_shards() -> tuple[pd.DataFrame, pd.DataFrame]:
    sh = common.OUT / "shards"
    per = pd.concat([pd.read_csv(f) for f in sorted(sh.glob("*_periods.csv.gz"))], ignore_index=True)
    dec = pd.concat([pd.read_csv(f) for f in sorted(sh.glob("*_decisions.csv"))], ignore_index=True)
    return per, dec


def gains(per: pd.DataFrame) -> pd.DataFrame:
    tot = per.groupby(["domain", "series", "family", "dist", "scheme"])[["sum_ll", "n", "sum_ll_nocovid", "n_nocovid"]].sum()
    tot["ll"] = tot.sum_ll / tot.n
    tot["ll_nocovid"] = tot.sum_ll_nocovid / tot.n_nocovid
    tot = tot.reset_index()
    e = tot[tot.scheme == "E"].set_index(["domain", "series", "family", "dist"])[["ll", "ll_nocovid"]]
    tot = tot.join(e, on=["domain", "series", "family", "dist"], rsuffix="_E")
    tot["G"] = tot.ll_E - tot.ll
    tot["G_nocovid"] = tot.ll_nocovid_E - tot.ll_nocovid
    tot["group"] = [common.group_of(d, s) for d, s in zip(tot.domain, tot.series)]
    return tot[["domain", "group", "series", "family", "dist", "scheme", "n", "ll", "ll_E", "G", "G_nocovid"]]


def wide(g: pd.DataFrame, dist: str, col: str = "G") -> pd.DataFrame:
    return g[g.dist == dist].pivot_table(index=["domain", "series", "family"], columns="scheme", values=col)


def cell_findings(gt: pd.DataFrame, gg: pd.DataFrame, check: str = "C0") -> dict:
    """F1-F3 for one cell; gt, gg: series x scheme gains with Student-t and Gaussian distributions."""
    f1a = mean_test(gt["AveW"] - gt["B"])
    f1b = mean_test(gt["AveW"] - gt[check])
    neg, pos = gg["B"] < 0, gg["B"] > 0
    sB_neg, sC_neg = gg.loc[neg, "B"].sum(), gg.loc[neg, check].sum()
    sB_pos, sC_pos = gg.loc[pos, "B"].sum(), gg.loc[pos, check].sum()
    f2_loss = (sC_neg >= 0.5 * sB_neg) if neg.any() else None
    f2_gain = (sC_pos >= 0.5 * sB_pos) if pos.any() else None
    parts = [x for x in (f2_loss, f2_gain) if x is not None]
    f3a = mean_test(gt["B"] - gg["B"])
    f3b = mean_test((gg[check] - gg["B"]) - (gt[check] - gt["B"]))
    return {
        "n": len(gt),
        "F1_AveW_minus_B": f1a, "F1_AveW_minus_C": f1b, "F1": f1a[0] > 0 and f1b[0] > 0,
        "F2_sum_B_losers": sB_neg, "F2_sum_C_losers": sC_neg, "F2_losers": int(neg.sum()),
        "F2_sum_B_gainers": sB_pos, "F2_sum_C_gainers": sC_pos, "F2_gainers": int(pos.sum()),
        "F2_loss_part": f2_loss, "F2_gain_part": f2_gain, "F2": (all(parts) if parts else None),
        "F3_tB_minus_gB": f3a, "F3_check_gain_g_minus_t": f3b, "F3": f3a[0] > 0 and f3b[0] > 0,
    }


def verdict(results: list[bool | None]) -> str:
    testable = [r for r in results if r is not None]
    if not testable:
        return "not testable"
    k, m = sum(testable), len(testable)
    if (m == 8 and k >= 6) or (m < 8 and k >= 0.75 * m):
        return f"TRAVELS ({k} of {m} cells)"
    if (m == 8 and k >= 3) or (m < 8 and k >= 0.375 * m):
        return f"TRAVELS PARTLY ({k} of {m} cells)"
    return f"DOES NOT TRAVEL ({k} of {m} cells)"


def fmt(t: tuple[float, float, float]) -> str:
    return f"{t[0]:+.4f} (se {t[1]:.4f}, p {t[2]:.3g})"


def main() -> None:
    per, dec = load_shards()
    g = gains(per)
    g.to_csv(common.OUT / "series_gains.csv", index=False, float_format="%.6f")
    dec.to_csv(common.OUT / "decisions.csv", index=False, float_format="%.4f")
    pm = per.assign(ll=per.sum_ll / per.n).groupby(["domain", "family", "dist", "scheme", "period"]).ll.mean().reset_index()
    pm.to_csv(common.OUT / "periods_mean.csv", index=False, float_format="%.6f")

    lines = ["Generality study: pre-registered findings F1-F3 (PREREGISTRATION.md).",
             "G = mean log loss of the expanding window minus that of the scheme (nats per day; positive = better than E).",
             "Means are across series; se = standard error across series; p = one-sided paired t-test p-value.", ""]
    nser = g.groupby("domain").series.nunique()
    lines.append("Series: " + ", ".join(f"{d} {n}" for d, n in nser.items()))
    lines.append("")

    cells = []
    for variant, check in (("primary (C0)", "C0"), ("secondary (C1645)", "C1645")):
        res = {"F1": [], "F2": [], "F3": []}
        lines.append(f"=== Findings, checked rule {variant}; F1 and F3a/F3b with Student-t, F2 with Gaussian; all days scored ===")
        for dom in common.DOMAINS:
            for fam in FAMILIES:
                gt = wide(g, "t").xs((dom, fam), level=("domain", "family"))
                gg = wide(g, "gauss").xs((dom, fam), level=("domain", "family")).loc[gt.index]
                c = cell_findings(gt, gg, check)
                for f in res:
                    res[f].append(c[f])
                cells.append({"variant": check, "domain": dom, "family": fam, **{k: (v if not isinstance(v, tuple) else v[0])
                                                                               for k, v in c.items()}})
                lines.append(f"[{dom} / {fam}] {c['n']} series")
                lines.append(f"  F1 {c['F1']}: AveW - B {fmt(c['F1_AveW_minus_B'])}; AveW - {check} {fmt(c['F1_AveW_minus_C'])}")
                rem = 100 * (1 - c["F2_sum_C_losers"] / c["F2_sum_B_losers"]) if c["F2_losers"] else float("nan")
                kept = 100 * c["F2_sum_C_gainers"] / c["F2_sum_B_gainers"] if c["F2_gainers"] else float("nan")
                lines.append(f"  F2 {c['F2']}: B loses on {c['F2_losers']} series (sum {c['F2_sum_B_losers']:+.4f}, {check} "
                             f"{c['F2_sum_C_losers']:+.4f}, {rem:.0f}% removed); B gains on {c['F2_gainers']} "
                             f"(sum {c['F2_sum_B_gainers']:+.4f}, {check} {c['F2_sum_C_gainers']:+.4f}, {kept:.0f}% kept)")
                lines.append(f"  F3 {c['F3']}: G_t(B) - G_g(B) {fmt(c['F3_tB_minus_gB'])}; "
                             f"check gain Gaussian minus Student-t {fmt(c['F3_check_gain_g_minus_t'])}")
        lines.append("")
        for f, r in res.items():
            lines.append(f"  VERDICT {f} [{variant}]: {verdict(r)}")
        lines.append("")
    pd.DataFrame(cells).to_csv(common.OUT / "cells.csv", index=False, float_format="%.6f")

    lines.append("=== Mean gain over E by domain, family and distribution (all days; share of series with G > 0) ===")
    for dom in common.DOMAINS:
        for dist in ("t", "gauss"):
            w = wide(g, dist).xs(dom, level="domain")
            for fam in FAMILIES:
                x = w.xs(fam, level="family")
                lines.append(f"  {dom} {dist:5s} {fam:3s}: " + "; ".join(
                    f"{k} {x[k].mean():+.4f} ({100 * (x[k] > 0).mean():.0f}%)" for k in SCHEMES if k != "E"))
    lines.append("")

    lines.append("=== F1 with Gaussian distributions (secondary): mean of AveW - B and AveW - C0 ===")
    for dom in common.DOMAINS:
        w = wide(g, "gauss").xs(dom, level="domain")
        for fam in FAMILIES:
            x = w.xs(fam, level="family")
            lines.append(f"  {dom} {fam}: AveW - B {fmt(mean_test(x['AveW'] - x['B']))}; AveW - C0 {fmt(mean_test(x['AveW'] - x['C0']))}")
    lines.append("")

    lines.append("=== Electricity without March-June 2020 in the scoring (secondary), Student-t ===")
    w = wide(g, "t", "G_nocovid").xs("elec", level="domain")
    for fam in FAMILIES:
        x = w.xs(fam, level="family")
        lines.append(f"  {fam}: " + "; ".join(f"{k} {x[k].mean():+.4f}" for k in SCHEMES if k != "E")
                     + f"; AveW - B {fmt(mean_test(x['AveW'] - x['B']))}")
    lines.append("")

    lines.append("=== Wikipedia by group (secondary), Student-t: mean gain over E ===")
    gw = g[(g.domain == "web") & (g.dist == "t")]
    for grp in ("country", "element"):
        for fam in FAMILIES:
            x = gw[(gw.group == grp) & (gw.family == fam)].pivot_table(index="series", columns="scheme", values="G")
            lines.append(f"  {grp} {fam} ({len(x)}): " + "; ".join(f"{k} {x[k].mean():+.4f}" for k in SCHEMES if k != "E"))
    lines.append("")

    lines.append("=== Absolute: mean log loss of the expanding window by family and distribution (nats per day) ===")
    for dom in common.DOMAINS:
        e = g[(g.domain == dom) & (g.scheme == "E")].groupby(["family", "dist"]).ll.mean()
        lines.append(f"  {dom}: " + "; ".join(f"{fam} {dist} {e[(fam, dist)]:.4f}" for fam in FAMILIES for dist in ("gauss", "t")))
    lines.append("")

    lines.append("=== Decisions (every 91 days) ===")
    for dom in common.DOMAINS:
        for dist in ("t", "gauss"):
            d = dec[(dec.domain == dom) & (dec.dist == dist)]
            st = d[d.stopped_at > 0]
            lines.append(f"  {dom} {dist}: {len(d)} decisions; walk stopped in {len(st)} ({100 * len(st) / len(d):.0f}%); "
                         f"check switched c=0 {int(d.switch_c0.sum())}, c=1.645 {int(d.switch_c1645.sum())}; "
                         f"median z_end when stopped {st.z_end.median():+.2f}")
    (common.OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
