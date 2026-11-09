"""Part 3 of the study: US industrial production growth and CPI inflation, 1960-2026.

    python application.py

1. The train-period x test-period matrix of an AR(3) with Gaussian errors fitted on
   five-year blocks (12-month gap), for each series.
2. The backward rule at the last origin: how far back does a model of 2021-2026 work?
3. A forward (pseudo out-of-sample) comparison of window schemes from January 1985,
   scored one month ahead, with the backward rule as one of the schemes.

Data: ``data/fred_monthly.csv`` from ``fetch.py`` (latest-vintage FRED series, not real
time vintages). Writes ``results/application_*.csv`` and ``results/application_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from backward_eval import ARGaussian, ARStudentT, backward_window, embed_ar, mean_and_se, transfer_matrix
from schemes import run_all, run_generic

HERE = Path(__file__).parent
OUT = HERE / "results"
P, BLOCK, GAP, MAX_BACK = 3, 60, 12, 8
Z_CRIT = float(norm.ppf(1 - 0.05 / MAX_BACK))
EXTRA_Z = (1.0, 1.645, 2.0, 3.0)
ROLLING = (60, 120, 240, 360)
START, FIRST_ORIGIN = "1960-01-01", "1985-01-01"
COVID = ("2020-03-01", "2020-12-01")  # months left out of scoring in the robustness check
PERIODS = {"1985-2007": ("1985-01-01", "2007-12-01"), "2008-2019": ("2008-01-01", "2019-12-01"),
           "2020-2026": ("2020-01-01", "2026-12-01"), "1985-2026": ("1985-01-01", "2026-12-01")}


def load() -> dict[str, pd.Series]:
    df = pd.read_csv(HERE / "data" / "fred_monthly.csv", parse_dates=["date"]).set_index("date")
    ip = (100 * np.log(df["INDPRO"]).diff()).loc[START:].dropna()
    cpi = (1200 * np.log(df["CPIAUCSL"]).diff()).loc[START:].dropna()
    return {"ip_growth": ip.rename("ip_growth"), "cpi_inflation": cpi.rename("cpi_inflation")}


LABEL = {"ip_growth": "Industrial production growth (% per month)",
         "cpi_inflation": "CPI inflation (% per year, monthly changes annualised)"}


def matrix_part(name: str, s: pd.Series, lines: list[str]) -> pd.DataFrame:
    X, z, t = embed_ar(s.to_numpy(), P)
    dates = s.index[t]
    res = transfer_matrix(X, z, ARGaussian, BLOCK, gap=GAP, folds=5, seed=2026)
    labels = [f"{dates[a]:%Y-%m}" for a, b in res.blocks]
    ends = [f"{dates[b - 1]:%Y-%m}" for a, b in res.blocks]
    covid = (dates >= COVID[0]) & (dates <= COVID[1])
    E = res.excess("nlpd")
    E_nc = res.excess("nlpd", keep=~covid)
    S = res.excess("sqerr")
    C = res.mean_matrix("cover90")
    K = res.k
    recs = []
    for i in range(K):
        for j in range(K):
            m, se = res.paired(i, j) if i != j else (0.0, np.nan)
            recs.append({"series": name, "train_block": i, "test_block": j,
                         "train_start": labels[i], "train_end": ends[i],
                         "test_start": labels[j], "test_end": ends[j],
                         "excess_nlpd": E[i, j], "excess_nlpd_no_covid": E_nc[i, j],
                         "sqerr_ratio": S[i, j], "cover90": C[i, j],
                         "z": m / se if i != j and se > 0 else np.nan})
    lines.append(f"[{name}] {LABEL[name]}: {len(s)} months {s.index[0]:%Y-%m} to {s.index[-1]:%Y-%m}; "
                 f"AR({P}); {K} blocks of {BLOCK} months; gap {GAP}")
    lines.append("  blocks: " + ", ".join(f"{a}..{b}" for a, b in zip(labels, ends)))
    last = K - 1
    lines.append("  last row, today's model on past blocks (excess NLPD / 90% coverage / z), oldest first:")
    for j in range(K - 1):
        m, se = res.paired(last, j)
        lines.append(f"    {labels[j]}..{ends[j]}: {E[last, j]:+.3f} / {C[last, j]:.2f} / {m / se:+.1f}"
                     f"   (without Mar-Dec 2020: {E_nc[last, j]:+.3f})")
    A = E - E.T
    iu = np.triu_indices(K, 1)
    order = np.argsort(-np.abs(A[iu]))[:5]
    lines.append("  largest asymmetries |E[i,j] - E[j,i]| (train block -> test block, excess both ways):")
    for o in order:
        i, j = iu[0][o], iu[1][o]
        lines.append(f"    {labels[i]} model on {labels[j]}: {E[i, j]:+.3f}; {labels[j]} model on {labels[i]}: {E[j, i]:+.3f}")
    nat_sd = [np.sqrt(np.mean(res.native['sqerr'][a:b])) for a, b in res.blocks]
    lines.append("  native one-step RMSE by block: " + " ".join(f"{v:.2f}" for v in nat_sd))
    return pd.DataFrame(recs)


def rule_today(name: str, s: pd.Series, lines: list[str]) -> None:
    X, z, t = embed_ar(s.to_numpy(), P)
    dates = s.index[t]
    for zc in (Z_CRIT, 1.645):
        ch = backward_window(X, z, len(z), ARGaussian, BLOCK, MAX_BACK, zc, gap=GAP, folds=5,
                             rng=np.random.default_rng(11))
        start = dates[len(z) - ch.rows] if ch.rows < len(z) else dates[0]
        stats = ", ".join(f"{m:+.2f} (z {zz:+.1f})" for m, zz in ch.stats)
        lines.append(f"  backward rule at the last origin, z > {zc:.2f}: window {ch.rows} months "
                     f"(from {start:%Y-%m}); stopped at block {ch.stopped_at}; per block back: {stats}")


def heavy_tail_check(name: str, s: pd.Series, lines: list[str]) -> pd.DataFrame:
    """Is the picture an artefact of Gaussian tails? Repeat the last row and the rule with
    an AR(3) with Student-t(5) errors fitted by maximum likelihood."""
    X, z, t = embed_ar(s.to_numpy(), P)
    dates = s.index[t]
    res = transfer_matrix(X, z, ARStudentT, BLOCK, gap=GAP, folds=5, seed=2026)
    E = res.excess("nlpd")
    C = res.mean_matrix("cover90")
    K = res.k
    labels = [f"{dates[a]:%Y-%m}" for a, b in res.blocks]
    lines.append("  Student-t(5) errors, last row (excess NLPD / 90% coverage / z), oldest first: " + "; ".join(
        f"{labels[j]} {E[K - 1, j]:+.3f}/{C[K - 1, j]:.2f}/{res.paired(K - 1, j)[0] / res.paired(K - 1, j)[1]:+.1f}"
        for j in range(K - 1)))
    for when in ("2010-01-01", "2022-01-01"):
        o = int(np.searchsorted(dates, pd.Timestamp(when)))
        for lab, fac in (("Gaussian", ARGaussian), ("Student-t", ARStudentT)):
            ch = backward_window(X, z, o, fac, BLOCK, MAX_BACK, Z_CRIT, gap=GAP, folds=5,
                                 rng=np.random.default_rng(11))
            stats = ", ".join(f"{m:+.2f} (z {zz:+.1f})" for m, zz in ch.stats)
            lines.append(f"  rule at {when[:7]}, {lab}: window {ch.rows} months; per block back: {stats}")
    recs = [{"series": name, "train_block": i, "test_block": j, "train_start": labels[i], "test_start": labels[j],
             "excess_nlpd": E[i, j], "cover90": C[i, j]} for i in range(K) for j in range(K)]
    return pd.DataFrame(recs)


def forward_part(name: str, s: pd.Series, lines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, z, t = embed_ar(s.to_numpy(), P)
    dates = s.index[t]
    origins = np.flatnonzero(dates >= FIRST_ORIGIN)
    outs, details = run_all(X, z, origins, ROLLING, BLOCK, MAX_BACK, Z_CRIT, GAP, reselect=12, folds=5,
                            seed=7, avew_min=BLOCK, avew_step=12, roc_min=BLOCK // 2, cv_len=BLOCK,
                            extra_z=EXTRA_Z)
    long = []
    for k, o in outs.items():
        long.append(pd.DataFrame({"series": name, "scheme": k, "date": dates[origins], "nlpd": o.nlpd,
                                  "sqerr": o.sqerr, "pit": o.pit, "cover90": o.cover90, "window": o.window}))
    fwd = pd.concat(long, ignore_index=True)
    win = []
    for k, det in details.items():
        for o, rows, stop in det:
            win.append({"series": name, "rule": k, "date": dates[o], "rows": rows, "stopped": stop})
    wdf = pd.DataFrame(win)

    lines.append(f"  forward comparison, origins {dates[origins[0]]:%Y-%m} to {dates[origins[-1]]:%Y-%m} "
                 f"({len(origins)} forecasts); mean NLPD minus expanding (Newey-West s.e.); MSE; coverage")
    base = fwd[fwd.scheme == "expanding"].set_index("date")
    covid = (base.index >= COVID[0]) & (base.index <= COVID[1])
    for per, (a, b) in PERIODS.items():
        sel = (base.index >= a) & (base.index <= b)
        for drop_covid in (False, True):
            if drop_covid and not ((per == "1985-2026") or (per == "2020-2026")):
                continue
            keep = sel & (~covid if drop_covid else True)
            parts = []
            for k in outs:
                d = fwd[fwd.scheme == k].set_index("date")
                diff = (d["nlpd"] - base["nlpd"])[keep]
                m, se = mean_and_se(diff.to_numpy())
                parts.append((k, m, se, d["sqerr"][keep].mean(), d["cover90"][keep].mean()))
            parts.sort(key=lambda r: r[1])
            tag = per + (" without Mar-Dec 2020" if drop_covid else "")
            lines.append(f"    {tag}: " + "; ".join(f"{k} {m:+.3f} ({se:.3f}) mse {ms:.3f} cov {cv:.2f}"
                                                   for k, m, se, ms, cv in parts))
    # the same comparison for the main schemes with Student-t(5) errors
    tout = run_generic(X, z, origins, ARStudentT, BLOCK, MAX_BACK, Z_CRIT, GAP, rolling=(120,), reselect=12,
                       folds=5, seed=7, cv_len=BLOCK)
    tl = []
    for k, o in tout.items():
        tl.append(pd.DataFrame({"series": name, "scheme": k, "date": dates[origins], "nlpd": o.nlpd,
                                "sqerr": o.sqerr, "pit": o.pit, "cover90": o.cover90, "window": o.window}))
    tdf = pd.concat(tl, ignore_index=True)
    tdf.to_csv(OUT / f"application_forward_t_{name}.csv", index=False, float_format="%.6g")
    tb = tdf[tdf.scheme == "expanding"].set_index("date")
    for per, (a, b) in PERIODS.items():
        for drop_covid in (False, True):
            if drop_covid and per not in ("1985-2026", "2020-2026"):
                continue
            keep = (tb.index >= a) & (tb.index <= b)
            if drop_covid:
                keep &= ~((tb.index >= COVID[0]) & (tb.index <= COVID[1]))
            parts = []
            for k in tout:
                d = tdf[tdf.scheme == k].set_index("date")
                m, se = mean_and_se((d["nlpd"] - tb["nlpd"])[keep].to_numpy())
                parts.append((k, m, se, d["cover90"][keep].mean()))
            parts.sort(key=lambda r: r[1])
            tag = per + (" without Mar-Dec 2020" if drop_covid else "")
            lines.append(f"    Student-t(5), {tag}: " + "; ".join(f"{k} {m:+.3f} ({se:.3f}) cov {cv:.2f}"
                                                                for k, m, se, cv in parts))
    bw = wdf[wdf.rule == "backward"]
    lines.append("    backward rule window (months) at each January: " + " ".join(
        f"{d:%Y}:{r}" for d, r in zip(bw.date, bw.rows) if d.month == 1 and d.year % 3 == 0))
    return fwd, wdf


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = load()
    lines = ["Real-data application: train-period x test-period matrices and forward comparison", ""]
    mats, mats_t, fwds, wins = [], [], [], []
    for name, s in data.items():
        mats.append(matrix_part(name, s, lines))
        rule_today(name, s, lines)
        mats_t.append(heavy_tail_check(name, s, lines))
        f, w = forward_part(name, s, lines)
        fwds.append(f)
        wins.append(w)
        lines.append("")
    pd.concat(mats).to_csv(OUT / "application_matrix.csv", index=False, float_format="%.6g")
    pd.concat(mats_t).to_csv(OUT / "application_matrix_t.csv", index=False, float_format="%.6g")
    pd.concat(fwds).to_csv(OUT / "application_forward.csv", index=False, float_format="%.6g")
    pd.concat(wins).to_csv(OUT / "application_windows.csv", index=False)
    (OUT / "application_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
