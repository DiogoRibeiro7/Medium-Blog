"""The pre-registered forward test (PREREGISTRATION.md) and the in-sample development set.

    python forward.py --workers 2

AR(3) by maximum likelihood, blocks 60 months, gap 12, z_crit 2.498, re-selection every 12
months, one-step forecasts. Test series from January 1990, development series (US) from January
1985. Primary: Student-t(5) errors, March-December 2020 not scored. Secondary: Gaussian errors,
all months, r = 24, c = 0 and 2.498.

Writes ``results/forward_scores.csv.gz`` (one row per origin), ``results/forward_gains.csv``,
``results/forward_decisions.csv`` and ``results/forward_summary.txt``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import common
from backward_eval import ARStudentT, embed_ar, mean_and_se
from episode import criteria, run_fast, run_generic

ORDER = ("E", "B", "C", "P0.25", "PC0.25", "AveW")
RS, CS = (common.R_PRIMARY, common.R_ALT), (common.C_ALT[0], common.C_PRIMARY, common.C_ALT[1])


def one(args):
    key, s, which, errors = args
    X, z, t = embed_ar(s.to_numpy(), common.AR_ORDER)
    dates = s.index[t]
    first = common.FIRST_ORIGIN_TEST if which == "test" else common.FIRST_ORIGIN_DEV
    origins = np.flatnonzero(dates >= pd.Timestamp(first))
    kw = dict(reselect=common.RESELECT, folds=common.FOLDS, seed=7, rs=RS, cs=CS,
              primary=(common.R_PRIMARY, common.C_PRIMARY))
    if errors == "gaussian":
        outs, sel = run_fast(X, z, origins, common.BLOCK, common.MAX_BACK, common.Z_CRIT, common.GAP, **kw)
    else:
        outs, sel = run_generic(X, z, origins, lambda: ARStudentT(df=common.T_DF), common.BLOCK, common.MAX_BACK,
                                common.Z_CRIT, common.GAP, **kw)
    sc = pd.DataFrame({"series": key, "set": which, "errors": errors, "date": dates[origins]})
    for k, o in outs.items():
        sc[k] = o.nlpd
    dec = []
    for d in sel.decisions:
        st = d.stopped_at is not None
        ze = d.z_end[common.R_PRIMARY]
        dec.append({"series": key, "set": which, "errors": errors, "date": f"{dates[d.origin]:%Y-%m}",
                    "stopped": st, "rows_B": d.rows_B, "z_star": d.z_star if st else np.nan,
                    "z_end12": ze, "z_end24": d.z_end[common.R_ALT],
                    "check_switched": bool(st and ze > common.C_PRIMARY)})
    return sc, pd.DataFrame(dec)


def gains(sc: pd.DataFrame, keep_covid: bool) -> pd.DataFrame:
    recs = []
    for (key, which, errors), g in sc.groupby(["series", "set", "errors"]):
        if not keep_covid:
            g = g[~g.date.between(pd.Timestamp(common.COVID[0]), pd.Timestamp(common.COVID[1]))]
        for k in [c for c in g.columns if c not in ("series", "set", "errors", "date")]:
            m, se = mean_and_se((g["E"] - g[k]).to_numpy())
            recs.append({"series": key, "set": which, "group": common.GROUP[key], "errors": errors, "scheme": k,
                         "scoring": "all" if keep_covid else "no_covid", "gain": m, "se": se, "n": len(g)})
    return pd.DataFrame(recs)


def table(gf: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {u: dict(zip(g.scheme, g.gain)) for u, g in gf.groupby("series")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    common.OUT.mkdir(exist_ok=True)
    test, dev = common.load("test"), common.load("dev")
    tasks = [(k, s, w, e) for w, data in (("test", test), ("dev", dev)) for k, s in data.items()
             for e in ("t", "gaussian")]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(one, tasks))
    sc = pd.concat([r[0] for r in res], ignore_index=True)
    dec = pd.concat([r[1] for r in res], ignore_index=True)
    sc.to_csv(common.OUT / "forward_scores.csv.gz", index=False, float_format="%.6g")
    dec.to_csv(common.OUT / "forward_decisions.csv", index=False, float_format="%.4g")
    gf = pd.concat([gains(sc, False), gains(sc, True)], ignore_index=True)
    gf.to_csv(common.OUT / "forward_gains.csv", index=False, float_format="%.6g")

    variants = sorted(k for k in gf.scheme.unique() if k.startswith("C["))
    test_keys = [k for k in common.TEST if k in test]
    dev_keys = [k for k in common.DEV if k in dev]
    lines = ["Pre-registered forward test. AR(3), blocks 60 months, gap 12, z_crit 2.498, re-selection every 12 months; "
             "end-of-episode check r = 12, c = 1.645. Test series from 1990-01, development (US) series from 1985-01.",
             "Gain = mean log loss of E minus that of the scheme (nats per month; positive = better than E), "
             "Newey-West standard error in brackets.",
             f"Test series available: {len(test_keys)} of {len(common.TEST)}"
             + ("" if len(test_keys) == len(common.TEST) else f" (dropped: {sorted(set(common.TEST) - set(test_keys))})"),
             ""]
    for errors in ("t", "gaussian"):
        for scoring in ("no_covid", "all"):
            sub = gf[(gf.errors == errors) & (gf.scoring == scoring)]
            title = "Student-t(5)" if errors == "t" else "Gaussian"
            role = "PRIMARY" if (errors, scoring) == ("t", "no_covid") else "secondary"
            lines.append(f"== {title} errors, scoring {'without Mar-Dec 2020' if scoring == 'no_covid' else 'all months'} [{role}] ==")
            for keys, label in ((test_keys, "test"), (dev_keys, "development, in-sample")):
                for k in keys:
                    g = sub[sub.series == k].set_index("scheme")
                    lines.append(f"  {common.SHORT[k]} [{common.GROUP[k]}]: " + "; ".join(
                        f"{s} {g.loc[s, 'gain']:+.4f} ({g.loc[s, 'se']:.4f})" for s in ORDER))
                tab = table(sub[sub.series.isin(keys)])
                cr = criteria(tab, keys, "C")
                lines.append(f"  Criteria on the {label} series: B gains on {[common.SHORT[u] for u in cr['B_gains_on']]}, "
                             f"loses on {[common.SHORT[u] for u in cr['B_loses_on']]}")
                lines.append(f"    H1 losses removed: sum G(B) {cr['sum_B_where_B_loses']:+.4f}, sum G(C) {cr['sum_C_there']:+.4f} -> {cr['H1']}")
                lines.append(f"    H2 gains kept: sum G(B) {cr['sum_B_where_B_gains']:+.4f}, sum G(C) {cr['sum_C_there_']:+.4f} -> {cr['H2']}")
                lines.append(f"    H3 competitive: mean G(C) {cr['mean']['C']:+.4f}; AveW {cr['mean']['AveW']:+.4f}; "
                             f"P0.25 {cr['mean']['P0.25']:+.4f} -> {cr['H3']}")
                lines.append(f"    Verdict: {cr['verdict'].upper()}" + ("" if label == "test" else " (in-sample, not part of the verdict)"))
                lines.append("    Mean gain: " + "; ".join(f"{s} {cr['mean'][s]:+.4f}" for s in ORDER))
                for grp in ("A", "B"):
                    ks = [k for k in keys if common.GROUP[k] == grp]
                    if ks:
                        lines.append(f"    Mean gain, group {grp} ({len(ks)} series): " + "; ".join(
                            f"{s} {np.mean([tab[k][s] for k in ks]):+.4f}" for s in ORDER))
                if label == "test":
                    for v in variants:
                        cv = criteria(tab, keys, v)
                        lines.append(f"    Variant {v}: H1 {cv['H1']} ({cv['sum_C_there']:+.4f}), H2 {cv['H2']} "
                                     f"({cv['sum_C_there_']:+.4f}), mean {cv['mean'][v]:+.4f}, verdict {cv['verdict']}")
            lines.append("")
    lines.append("S3: decisions in which the check switched the plain rule to the expanding window (Student-t errors)")
    for keys, label in ((test_keys, "test"), (dev_keys, "development")):
        for k in keys:
            d = dec[(dec.series == k) & (dec.errors == "t")]
            sw = d[d.check_switched].date.tolist()
            lines.append(f"  {common.SHORT[k]} ({label}): walk stopped in {int(d.stopped.sum())} of {len(d)} decisions; "
                         f"check switched {len(sw)}: {', '.join(sw) if sw else '-'}")
    (common.OUT / "forward_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
