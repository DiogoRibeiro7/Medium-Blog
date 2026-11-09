"""The warranted spread: what fundamentals justify, and what is left over.

    python warranted.py

The TPI may buy only against spread rises that country fundamentals do not justify. This script
makes that call the way a data scientist would have to: as a prediction problem with a model
choice, and with the model choice reported rather than hidden.

For nine euro-area sovereigns (France, Italy, Spain, Belgium, Austria, Netherlands, Finland,
Portugal, Ireland) the quarterly 10-year spread to Germany is regressed on fundamentals relative
to Germany, lagged one quarter (debt and net lending over the last four quarters in % of GDP,
real growth on a year earlier, the current account in % of GDP). The specification grid crosses

- six feature sets: debt; debt and balance; debt, balance and growth; plus the current account;
  plus the square of the debt gap; plus the debt gap times the German yield (debt costs more to
  carry when rates are higher);
- with and without the German 10-year yield as a regressor;
- with and without country fixed effects (liquidity, size, institutions);
- five estimation windows: 2000-2007 (before the crisis); 2000-2019 without 2010Q2-2012Q4
  ("normal times"); 2013-2019 (after the OMT announcement); 2000Q1-2024Q2 (everything before
  France's snap election); and a real-time expanding window that uses only data up to four
  quarters before each prediction.

That is 6 x 2 x 2 x 5 = 120 models. Each gives a warranted spread for every country-quarter and a
residual (actual minus warranted). A quarter is flagged "unwarranted" by a model when its
residual exceeds the 95th percentile of that model's own residuals in its estimation window.

Checks of what the residual measures:
- mean reversion: the change in the spread over the next four quarters regressed on the residual
  (a residual that is panic should shrink; one that is a mis-measured fundamental should not);
- the OMT episode: the residual and the warranted spread of Italy, Spain, Portugal and Ireland
  in 2012Q2 and a year later.

Writes ``results/warranted_panel.csv.gz``, ``results/warranted_specs.csv`` and
``results/warranted_summary.txt``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import NAMES, PANEL, RESULTS, quarterly_panel, write

FEATURES: dict[str, list[str]] = {
    "debt": ["debt_gap"],
    "debt_balance": ["debt_gap", "balance_gap"],
    "fiscal_growth": ["debt_gap", "balance_gap", "growth_gap"],
    "plus_current_account": ["debt_gap", "balance_gap", "growth_gap", "ca_gap"],
    "nonlinear_debt": ["debt_gap", "debt_gap_sq", "balance_gap", "growth_gap"],
    "rate_dependent_debt": ["debt_gap", "debt_x_bund", "balance_gap", "growth_gap"],
}
WINDOWS: dict[str, tuple[str, str, tuple[str, str] | None]] = {
    "pre_crisis": ("2000Q1", "2007Q4", None),
    "normal_times": ("2000Q1", "2019Q4", ("2010Q2", "2012Q4")),
    "post_omt": ("2013Q1", "2019Q4", None),
    "long": ("2000Q1", "2024Q2", None),
    "real_time": ("2000Q1", "", None),
}
REALTIME_LAG = 4  # quarters between the last estimation quarter and the prediction
REALTIME_MIN = 24  # quarters of data before the first real-time prediction
FLAG_Q = 0.95
LATEST_FUND = "2026Q2"


def design() -> pd.DataFrame:
    p = quarterly_panel()
    p = p[p["geo"].isin(PANEL)].copy()
    p = p.sort_values(["geo", "quarter"])
    lagged = ["debt_gap", "balance_gap", "growth_gap", "ca_gap"]
    for c in lagged:
        p[c] = p.groupby("geo")[c].shift(1)
    p["debt_gap_sq"] = p["debt_gap"] ** 2 / 100
    p["debt_x_bund"] = p["debt_gap"] * p["bund"] / 10
    return p.reset_index(drop=True)


def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def matrix(d: pd.DataFrame, feats: list[str], bund: bool, fe: bool) -> np.ndarray:
    cols = [d[f].to_numpy(float) for f in feats]
    if bund:
        cols.append(d["bund"].to_numpy(float))
    if fe:
        for g in PANEL:
            cols.append((d["geo"] == g).to_numpy(float))
    else:
        cols.append(np.ones(len(d)))
    return np.column_stack(cols)


def in_window(q: pd.Series, start: str, end: str, hole: tuple[str, str] | None) -> pd.Series:
    s, e = pd.Period(start, "Q"), pd.Period(end, "Q")
    keep = (q >= s) & (q <= e)
    if hole:
        keep &= ~((q >= pd.Period(hole[0], "Q")) & (q <= pd.Period(hole[1], "Q")))
    return keep


def run_spec(d: pd.DataFrame, fname: str, bund: bool, fe: bool, wname: str) -> tuple[pd.DataFrame, dict]:
    feats = FEATURES[fname]
    need = feats + (["bund"] if bund else [])
    ok = d[need].notna().all(axis=1)
    usable = d[ok].copy()
    have_y = usable["spread"].notna()
    out = usable[["geo", "quarter", "spread"]].copy()
    out["warranted"] = np.nan
    out["threshold"] = np.nan
    meta = dict(features=fname, bund=bund, fe=fe, window=wname)
    if wname != "real_time":
        start, end, hole = WINDOWS[wname]
        train = have_y & in_window(usable["quarter"], start, end, hole)
        X = matrix(usable[train], feats, bund, fe)
        y = usable.loc[train, "spread"].to_numpy(float)
        beta = fit(X, y)
        res = y - X @ beta
        out["warranted"] = matrix(usable, feats, bund, fe) @ beta
        out["threshold"] = np.quantile(res, FLAG_Q)
        meta.update(n=int(train.sum()), r2=float(1 - res.var() / y.var()), resid_sd=float(res.std()),
                    coef_debt=float(beta[0]), threshold=float(np.quantile(res, FLAG_Q)))
    else:
        quarters = sorted(usable["quarter"].unique())
        first = quarters[0] + REALTIME_MIN + REALTIME_LAG
        Xall = matrix(usable, feats, bund, fe)
        yall = usable["spread"].to_numpy(float)
        qv = usable["quarter"].to_numpy()
        coefs = []
        for t in quarters:
            if t < first:
                continue
            train = (qv <= t - REALTIME_LAG) & have_y.to_numpy()
            beta = fit(Xall[train], yall[train])
            res = yall[train] - Xall[train] @ beta
            now = qv == t
            out.loc[out.index[now], "warranted"] = Xall[now] @ beta
            out.loc[out.index[now], "threshold"] = np.quantile(res, FLAG_Q)
            coefs.append(beta[0])
        meta.update(n=int(have_y.sum()), r2=np.nan, resid_sd=np.nan, coef_debt=float(np.median(coefs)), threshold=np.nan)
    out["residual"] = out["spread"] - out["warranted"]
    out["flag"] = out["residual"] > out["threshold"]
    out["spec"] = f"{fname}|{'bund' if bund else 'nobund'}|{'fe' if fe else 'pooled'}|{wname}"
    return out, meta


def mean_reversion(panel: pd.DataFrame) -> pd.DataFrame:
    """Per spec: slope of the four-quarter-ahead spread change on the residual."""
    rows = []
    for spec, f in panel.groupby("spec"):
        f = f.sort_values(["geo", "quarter"]).copy()
        f["ahead"] = f.groupby("geo")["spread"].shift(-4) - f["spread"]
        f["dw"] = f.groupby("geo")["warranted"].shift(-4) - f["warranted"]
        for label, mask in (("all", f["quarter"] >= pd.Period("2006Q1", "Q")),
                            ("crisis_2010_2012", (f["quarter"] >= pd.Period("2010Q2", "Q")) & (f["quarter"] <= pd.Period("2012Q4", "Q")))):
            g = f[mask].dropna(subset=["ahead", "residual", "dw"])
            if len(g) < 20:
                continue
            X = np.column_stack([np.ones(len(g)), g["residual"], g["dw"]])
            b = fit(X, g["ahead"].to_numpy(float))
            rows.append(dict(spec=spec, sample=label, n=len(g), slope_residual=b[1], slope_warranted=b[2]))
    return pd.DataFrame(rows)


def run() -> str:
    d = design()
    outs, metas = [], []
    for fname, bund, fe, wname in itertools.product(FEATURES, (False, True), (False, True), WINDOWS):
        o, m = run_spec(d, fname, bund, fe, wname)
        outs.append(o)
        m["spec"] = o["spec"].iloc[0]
        metas.append(m)
    panel = pd.concat(outs, ignore_index=True)
    specs = pd.DataFrame(metas)
    panel["window"] = panel["spec"].str.split("|").str[3]
    out = panel.copy()
    out["quarter"] = out["quarter"].astype(str)
    out.to_csv(RESULTS / "warranted_panel.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}, float_format="%.4f")
    specs.to_csv(RESULTS / "warranted_specs.csv", index=False, float_format="%.4f")
    mr = mean_reversion(panel)
    mr.to_csv(RESULTS / "warranted_mean_reversion.csv", index=False, float_format="%.4f")

    headline = panel[panel["window"] != "pre_crisis"]
    lines = [
        f"Models: {len(specs)} ({len(FEATURES)} feature sets x German yield yes/no x fixed effects yes/no x {len(WINDOWS)} windows); "
        f"headline uses the {specs['window'].ne('pre_crisis').sum()} models outside the 2000-2007 window",
        f"Panel: {', '.join(PANEL)}; quarterly 2000Q1-{panel['quarter'].max()}; fundamentals lagged one quarter, latest fundamentals {LATEST_FUND}",
        "",
        "Fit in the estimation window (fixed windows): R2 median [min, max]; residual sd median; debt-gap slope median (pp of spread per 10 pp of debt gap):",
    ]
    for w, f in specs[specs["window"] != "real_time"].groupby("window", sort=False):
        lines.append(f"  {w}: R2 {f.r2.median():.2f} [{f.r2.min():.2f}, {f.r2.max():.2f}]; resid sd {f.resid_sd.median():.2f}; debt slope x10 {10 * f.coef_debt.median():.3f}; flag threshold median {f.threshold.median():.2f}")
    rt = specs[specs["window"] == "real_time"]
    lines.append(f"  real_time: debt slope x10 median over time {10 * rt.coef_debt.median():.3f}")

    def dist(sub: pd.DataFrame, col: str) -> str:
        v = sub[col].dropna()
        return f"median {v.median():+.2f}, 10-90% [{v.quantile(.1):+.2f}, {v.quantile(.9):+.2f}], min {v.min():+.2f}, max {v.max():+.2f} (n={len(v)})"

    lines += ["", "France, by quarter (headline models): actual spread; warranted; residual; share of models flagging unwarranted"]
    fr = headline[headline["geo"] == "FR"]
    for q in [pd.Period(x, "Q") for x in ("2019Q4", "2023Q4", "2024Q2", "2024Q3", "2024Q4", "2025Q2", "2025Q4", "2026Q1", "2026Q2", "2026Q3")]:
        f = fr[fr["quarter"] == q]
        if f.empty or f["spread"].isna().all():
            continue
        lines.append(f"  {q}: actual {f.spread.iloc[0]:.2f}; warranted median {f.warranted.median():.2f} [10-90% {f.warranted.quantile(.1):.2f}, {f.warranted.quantile(.9):.2f}]; "
                     f"residual median {f.residual.median():+.2f}; flagged {f.flag.mean():.0%} of {f.flag.notna().sum()}")
    last_q = fr.dropna(subset=["spread", "warranted"])["quarter"].max()
    f = fr[fr["quarter"] == last_q]
    lines.append(f"  latest ({last_q}) residual: {dist(f, 'residual')}")
    for w, g in f.groupby("window"):
        lines.append(f"    window {w}: warranted median {g.warranted.median():.2f}, residual median {g.residual.median():+.2f}, flagged {g.flag.mean():.0%}")
    f = f.assign(fe=f["spec"].str.split("|").str[2], bund=f["spec"].str.split("|").str[1])
    for col, label in (("fe", "country effects"), ("bund", "German yield")):
        for k, g in f.groupby(col):
            lines.append(f"    {label} {k}: warranted median {g.warranted.median():.2f}, residual median {g.residual.median():+.2f}, flagged {g.flag.sum()} of {len(g)}")
    pre = panel[(panel["geo"] == "FR") & (panel["quarter"] == last_q) & (panel["window"] == "pre_crisis")]
    lines.append(f"    window pre_crisis (excluded from headline): warranted median {pre.warranted.median():.2f}, residual median {pre.residual.median():+.2f}, flagged {pre.flag.mean():.0%}")
    lines.append(f"  October 2026 daily peak 1.78 (22 Oct, documented) against the latest warranted median {f.warranted.median():.2f}: excess {1.78 - f.warranted.median():+.2f}; "
                 f"share of headline models whose threshold it would cross (residual > threshold): {((1.78 - f.warranted) > f.threshold).mean():.0%}")
    lines.append(f"  Excess over warranted needed to be flagged by half of the headline models: {(f.threshold.median()):.2f} above warranted, i.e. a spread near {f.warranted.median() + f.threshold.median():.2f}")

    lines += ["", "The OMT episode (headline models), 2012Q2 -> 2013Q2: actual, warranted median, residual median, share flagged"]
    for g in ("IT", "ES", "PT", "IE"):
        a = headline[(headline["geo"] == g) & (headline["quarter"] == pd.Period("2012Q2", "Q"))]
        b = headline[(headline["geo"] == g) & (headline["quarter"] == pd.Period("2013Q2", "Q"))]
        lines.append(f"  {NAMES[g]}: actual {a.spread.iloc[0]:.2f} -> {b.spread.iloc[0]:.2f} ({b.spread.iloc[0] - a.spread.iloc[0]:+.2f}); "
                     f"warranted {a.warranted.median():.2f} -> {b.warranted.median():.2f} ({b.warranted.median() - a.warranted.median():+.2f}); "
                     f"residual {a.residual.median():+.2f} -> {b.residual.median():+.2f} ({b.residual.median() - a.residual.median():+.2f}); "
                     f"flagged {a.flag.mean():.0%} -> {b.flag.mean():.0%}")
    peak = headline[(headline["quarter"] >= pd.Period("2010Q1", "Q")) & (headline["quarter"] <= pd.Period("2012Q4", "Q"))]
    pk = peak.groupby(["geo", "quarter"])["flag"].mean().groupby("geo").max()
    lines.append("  Highest share of headline models flagging any quarter of 2010-2012: " + ", ".join(f"{NAMES[g]} {pk[g]:.0%}" for g in ("IT", "ES", "PT", "IE", "BE", "FR", "AT", "NL", "FI")))

    lines += ["", "Mean reversion: slope of the next four quarters' spread change on the residual (controlling for the change in the warranted spread); median [10-90%] across headline models"]
    mr_h = mr[~mr["spec"].str.endswith("pre_crisis")]
    for s, f in mr_h.groupby("sample"):
        lines.append(f"  {s}: residual slope {f.slope_residual.median():+.2f} [{f.slope_residual.quantile(.1):+.2f}, {f.slope_residual.quantile(.9):+.2f}]; warranted slope {f.slope_warranted.median():+.2f}; n median {int(f.n.median())}")

    lines += ["", "Out-of-sample error for France, 2024Q3 onward (headline fixed-window models): mean absolute residual"]
    for w, g in fr[(fr["quarter"] >= pd.Period("2024Q3", "Q")) & (fr["window"] != "real_time")].groupby("window"):
        lines.append(f"  {w}: {g.residual.abs().mean():.2f} pp (mean residual {g.residual.mean():+.2f})")
    return write("warranted_summary.txt", lines)


if __name__ == "__main__":
    print(run())
