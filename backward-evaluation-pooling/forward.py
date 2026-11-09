"""The pre-registered forward test on eight new US series (plus the in-sample pair).

    python forward.py [--workers 2]

AR(3), Gaussian errors, least squares; blocks of 60 months, gap 12, up to 8 blocks back,
z_crit 2.498; re-chosen every 12 months; one-step forecasts from January 1985. Primary scoring
leaves March-December 2020 out of the averages; secondary scoring keeps every month. The
Student-t(5) robustness check runs E, B, EWP, P0.5 and AveW with Student-t errors.

Writes ``results/forward_scores.csv.gz`` (per series, scheme and month), ``results/forward_gains.csv``,
``results/forward_weights.csv`` and ``results/forward_summary.txt``.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backward-evaluation"))

from backward_eval import ARStudentT, embed_ar, mean_and_se  # noqa: E402
from common import COVID, FIRST_ORIGIN, IN_SAMPLE, SHORT, TEST, load  # noqa: E402
from pooling import SCHEMES, criteria, run_fast, run_generic  # noqa: E402

P, BLOCK, GAP, MAX_BACK = 3, 60, 12, 8
Z_CRIT = float(norm.ppf(1 - 0.05 / MAX_BACK))
OUT = Path(__file__).parent / "results"


def one(args: tuple[str, pd.Series, bool]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sid, s, student = args
    X, z, t = embed_ar(s.to_numpy(), P)
    dates = s.index[t]
    origins = np.flatnonzero(dates >= FIRST_ORIGIN)
    if student:
        outs = run_generic(X, z, origins, ARStudentT, BLOCK, MAX_BACK, Z_CRIT, GAP, reselect=12, folds=5, seed=7)
        sel = []
    else:
        outs, sel = run_fast(X, z, origins, BLOCK, MAX_BACK, Z_CRIT, GAP, reselect=12, folds=5, seed=7)
    frames = [pd.DataFrame({"series": sid, "errors": "student_t5" if student else "gaussian", "scheme": k,
                            "date": dates[origins], "nlpd": o.nlpd, "sqerr": o.sqerr, "pit": o.pit,
                            "weight": o.weight}) for k, o in outs.items()]
    w = pd.DataFrame([{"series": sid, "date": dates[x.origin], "rows": x.rows, "stopped": x.stopped,
                       "z_star": x.z_star, "weight": norm.cdf(x.z_star - Z_CRIT) if x.stopped else 0.0}
                      for x in sel])
    return pd.concat(frames, ignore_index=True), w


def gains(df: pd.DataFrame, keep_covid: bool) -> pd.DataFrame:
    rows = []
    for (sid, err), g in df.groupby(["series", "errors"]):
        piv = g.pivot(index="date", columns="scheme", values="nlpd")
        if not keep_covid:
            piv = piv[~((piv.index >= COVID[0]) & (piv.index <= COVID[1]))]
        for k in piv.columns:
            d = (piv["E"] - piv[k]).to_numpy()
            m, se = mean_and_se(d)
            rows.append({"series": sid, "errors": err, "scheme": k, "scoring": "all" if keep_covid else "no_covid",
                         "gain": m, "se": se, "n": len(d)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    data = load()
    ids = [*TEST, *IN_SAMPLE]
    tasks = [(sid, data[sid], False) for sid in ids] + [(sid, data[sid], True) for sid in ids]
    scores, weights = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for sc, w in ex.map(one, tasks):
            scores.append(sc)
            if len(w):
                weights.append(w)
    df = pd.concat(scores, ignore_index=True)
    wf = pd.concat(weights, ignore_index=True)
    df.to_csv(OUT / "forward_scores.csv.gz", index=False, float_format="%.6g")
    wf.to_csv(OUT / "forward_weights.csv", index=False, float_format="%.4g")
    gdf = pd.concat([gains(df, False), gains(df, True)], ignore_index=True)
    gdf.to_csv(OUT / "forward_gains.csv", index=False, float_format="%.6g")

    lines = [f"Pre-registered forward test. AR({P}), blocks {BLOCK} months, gap {GAP}, z_crit {Z_CRIT:.3f}, origins from "
             f"{FIRST_ORIGIN[:7]}. Gain = mean log loss of E minus that of the scheme (nats per month; positive = better "
             "than the expanding window), Newey-West standard error in brackets.", ""]
    for err in ("gaussian", "student_t5"):
        for scoring in ("no_covid", "all"):
            tag = f"{'Gaussian' if err == 'gaussian' else 'Student-t(5)'} errors, scoring {'without Mar-Dec 2020' if scoring == 'no_covid' else 'all months'}"
            lines.append(f"== {tag} ==")
            sub = gdf[(gdf.errors == err) & (gdf.scoring == scoring)]
            table: dict[str, dict[str, float]] = {}
            for sid in ids:
                r = sub[sub.series == sid].set_index("scheme")
                table[sid] = r["gain"].to_dict()
                parts = "; ".join(f"{k} {r.loc[k, 'gain']:+.4f} ({r.loc[k, 'se']:.4f})" for k in SCHEMES if k in r.index)
                lines.append(f"  {SHORT[sid]}{' [in-sample]' if sid in IN_SAMPLE else ''}: {parts}")
            if err == "gaussian":
                cr = criteria(table, list(TEST))
                lines.append(f"  Criteria on the eight test series: B gains on {[SHORT[u] for u in cr['units_B_gains']]}, "
                             f"loses on {[SHORT[u] for u in cr['units_B_loses']]}")
                lines.append(f"    C1 gains kept: sum G(B) {cr['sum_G_B_where_B_gains']:+.4f}, sum G(EWP) {cr['sum_G_EWP_there']:+.4f} "
                             f"-> {cr['C1_gains_kept']}")
                lines.append(f"    C2 losses removed: sum G(B) {cr['sum_G_B_where_B_loses']:+.4f}, sum G(EWP) {cr['sum_G_EWP_there_']:+.4f} "
                             f"-> {cr['C2_losses_removed']}")
                lines.append(f"    C3 weighting matters: mean G(EWP) {cr['mean_G_EWP']:+.4f}; "
                             + ", ".join(f"{k} {v:+.4f}" for k, v in cr["mean_G_fixed"].items())
                             + f"; best {cr['best_fixed']} -> {cr['C3_weighting_matters']}")
                verdict = cr["C1_gains_kept"] is not False and cr["C2_losses_removed"] is not False and cr["C3_weighting_matters"]
                lines.append(f"    Verdict ({'primary' if scoring == 'no_covid' else 'secondary'} scoring): "
                             f"{'SUPPORTED' if verdict else 'NOT SUPPORTED'}")
                means = {k: np.mean([table[u][k] for u in TEST]) for k in SCHEMES}
                lines.append("    Mean gain over the eight: " + "; ".join(f"{k} {v:+.4f}" for k, v in means.items()))
                ins = criteria(table, list(IN_SAMPLE))
                lines.append(f"  In-sample pair: C1 {ins['C1_gains_kept']} (sum G(B) {ins['sum_G_B_where_B_gains']:+.4f}, "
                             f"G(EWP) {ins['sum_G_EWP_there']:+.4f}); C2 {ins['C2_losses_removed']} "
                             f"(sum G(B) {ins['sum_G_B_where_B_loses']:+.4f}, G(EWP) {ins['sum_G_EWP_there_']:+.4f})")
            else:
                means = {k: np.mean([table[u][k] for u in TEST if k in table[u]]) for k in ("B", "EWP", "P0.5", "AveW")}
                lines.append("  Mean gain over the eight: " + "; ".join(f"{k} {v:+.4f}" for k, v in means.items()))
            lines.append("")
    lines.append("Backward walk, Gaussian errors: share of yearly selections that stopped; median z* and EWP weight when stopped")
    for sid in ids:
        w = wf[wf.series == sid]
        st = w[w.stopped]
        lines.append(f"  {SHORT[sid]}: stopped {100 * w.stopped.mean():.0f}% of {len(w)}; median z* "
                     f"{st.z_star.median():.2f}; median weight {st.weight.median():.2f}; weight > 0.9 in "
                     f"{(st.weight > 0.9).sum()} of {len(st)}")
    (OUT / "forward_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
