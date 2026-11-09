"""The warranted spread: what fundamentals justify, and what is left over.

    python warranted.py

The TPI may buy only against spread rises that country fundamentals do not justify. This script
makes that call the way a data scientist would have to: as a prediction problem with a model
choice, and with the model choice reported rather than hidden.

For nine euro-area sovereigns (France, Italy, Spain, Belgium, Austria, Netherlands, Finland,
Portugal, Ireland) the quarterly 10-year spread to Germany is regressed on fundamentals relative
to Germany, lagged one quarter (debt and net lending over the last four quarters in % of GDP,
real growth on a year earlier, the current account in % of GDP, and the net international
investment position in % of GDP, the previous end-year value). Greece is predicted out of sample
(with the average country effect in fixed-effects models). The specification grid crosses

- eight feature sets: debt; debt and balance; debt, balance and growth; plus the current account;
  plus the square of the debt gap; plus the debt gap times the German yield; plus external debt
  (net international investment position); plus both external debt and the current account;
- with and without the German 10-year yield as a regressor;
- with and without country fixed effects (liquidity, size, institutions);
- five estimation windows: 2000-2007 (before the crisis); 2000-2019 without 2010Q2-2012Q4
  ("normal times"); 2013-2019 (after the OMT announcement); 2000Q1-2024Q2 (everything before
  France's snap election); and a real-time expanding window that uses only data up to four
  quarters before each prediction.

That is 8 x 2 x 2 x 5 = 160 models. Each gives a warranted spread for every country-quarter and a
residual (actual minus warranted). A quarter is flagged "unwarranted" by a model when its
residual exceeds the 95th percentile of that model's own residuals in its estimation window.

Reported for Portugal, Greece and Ireland: actual and warranted spreads, residuals and flags by
quarter, the ranking of the three by actual and by warranted spread, the role of the external
feature sets, the OMT episode, and a real-time check (how much lower Portugal's warranted spread
would be with the debt figure published in April 2010 instead of today's revised one).

Writes ``results/warranted_panel.csv.gz``, ``results/warranted_specs.csv`` and
``results/warranted_summary.txt``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import NAMES, PANEL, PREDICT, RESULTS, TRIO, annual_panel, quarterly_panel, write

FEATURES: dict[str, list[str]] = {
    "debt": ["debt_gap"],
    "debt_balance": ["debt_gap", "balance_gap"],
    "fiscal_growth": ["debt_gap", "balance_gap", "growth_gap"],
    "plus_current_account": ["debt_gap", "balance_gap", "growth_gap", "ca_gap"],
    "nonlinear_debt": ["debt_gap", "debt_gap_sq", "balance_gap", "growth_gap"],
    "rate_dependent_debt": ["debt_gap", "debt_x_bund", "balance_gap", "growth_gap"],
    "plus_external_debt": ["debt_gap", "balance_gap", "growth_gap", "niip_gap"],
    "external_and_current_account": ["debt_gap", "balance_gap", "growth_gap", "ca_gap", "niip_gap"],
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
    p = p[p["geo"].isin(PREDICT)].copy()
    p = p.sort_values(["geo", "quarter"])
    lagged = ["debt_gap", "balance_gap", "growth_gap", "ca_gap", "niip_gap"]
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
        # countries outside the estimation panel (Greece) get the average country effect
        outside = (~d["geo"].isin(PANEL)).to_numpy(float) / len(PANEL)
        for g in PANEL:
            cols.append((d["geo"] == g).to_numpy(float) + outside)
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
    have_y = usable["spread"].notna() & usable["geo"].isin(PANEL)
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
                    coef_debt=float(beta[0]), threshold=float(np.quantile(res, FLAG_Q)),
                    coef_balance=float(beta[feats.index("balance_gap")]) if "balance_gap" in feats else np.nan)
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
        meta.update(n=int(have_y.sum()), r2=np.nan, resid_sd=np.nan, coef_debt=float(np.median(coefs)), threshold=np.nan, coef_balance=np.nan)
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


# Eurostat news release 55/2010, 22 April 2010 (first notification of 2009 data)
REALTIME_2009 = {"PT": (76.8, -9.4), "EL": (115.1, -13.6), "IE": (64.0, -14.3)}
SHOW_Q = ("2009Q4", "2010Q2", "2010Q4", "2011Q1", "2011Q2", "2011Q3", "2011Q4", "2012Q1", "2012Q2", "2012Q4", "2013Q2")


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
    panel["features"] = panel["spec"].str.split("|").str[0]
    out = panel.drop(columns="features").copy()
    out["quarter"] = out["quarter"].astype(str)
    out.to_csv(RESULTS / "warranted_panel.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}, float_format="%.4f")
    specs.to_csv(RESULTS / "warranted_specs.csv", index=False, float_format="%.4f")
    mr = mean_reversion(panel[panel["geo"].isin(PANEL)])
    mr.to_csv(RESULTS / "warranted_mean_reversion.csv", index=False, float_format="%.4f")

    head = panel[panel["window"] != "pre_crisis"]
    n_head = head["spec"].nunique()
    lines = [
        f"Models: {len(specs)} ({len(FEATURES)} feature sets x German yield yes/no x fixed effects yes/no x {len(WINDOWS)} windows); "
        f"headline uses the {n_head} models outside the 2000-2007 window",
        f"Estimation panel: {', '.join(PANEL)}; Greece predicted out of sample; quarterly 2000Q1-{panel['quarter'].max()}; fundamentals lagged one quarter",
        "",
        "Fit in the estimation window (fixed windows): R2 median; debt-gap slope median (pp of spread per 10 pp of debt gap); external-debt slope median (pp per 10 pp of NIIP gap, external sets):",
    ]
    for w, f in specs[specs["window"] != "real_time"].groupby("window", sort=False):
        lines.append(f"  {w}: R2 {f.r2.median():.2f} [{f.r2.min():.2f}, {f.r2.max():.2f}]; debt slope x10 {10 * f.coef_debt.median():.3f}")
    ext = [k for k in FEATURES if "external" in k]
    coefs = []
    for k in ext:
        for bund, fe in itertools.product((False, True), (False, True)):
            for w in ("normal_times", "post_omt", "long"):
                need = FEATURES[k] + (["bund"] if bund else [])
                start, end, hole = WINDOWS[w]
                dd = d[d[need].notna().all(axis=1) & d["spread"].notna() & d["geo"].isin(PANEL)]
                dd = dd[in_window(dd["quarter"], start, end, hole)]
                beta = fit(matrix(dd, FEATURES[k], bund, fe), dd["spread"].to_numpy(float))
                coefs.append(beta[FEATURES[k].index("niip_gap")])
    lines.append(f"  NIIP-gap slope x10 (external sets, fixed windows other than 2000-2007): median {10 * np.median(coefs):+.3f}, range [{10 * min(coefs):+.3f}, {10 * max(coefs):+.3f}] (negative: a more negative position raises the spread)")

    lines += ["", "Portugal, Greece and Ireland by quarter (headline models): actual; warranted median [10-90%]; residual median; share flagged"]
    for g in TRIO:
        lines.append(f"  {NAMES[g]}:")
        f = head[head["geo"] == g]
        for qs in SHOW_Q:
            h = f[f["quarter"] == pd.Period(qs, "Q")]
            if h.empty or h["spread"].isna().all():
                continue
            lines.append(f"    {qs}: actual {h.spread.iloc[0]:.2f}; warranted {h.warranted.median():.2f} [{h.warranted.quantile(.1):.2f}, {h.warranted.quantile(.9):.2f}]; "
                         f"residual {h.residual.median():+.2f}; flagged {h.flag.mean():.0%}")
    lines += ["", "Peak residual, 2010-2012 (median model) and first quarter flagged by at least half of the headline models:"]
    for g in TRIO:
        f = head[(head["geo"] == g) & (head["quarter"] >= pd.Period("2008Q1", "Q")) & (head["quarter"] <= pd.Period("2013Q4", "Q"))]
        med = f.groupby("quarter")["residual"].median()
        share = f.groupby("quarter")["flag"].mean()
        first = share[share >= 0.5].index.min()
        lines.append(f"  {NAMES[g]}: peak residual {med.max():+.2f} in {med.idxmax()}; first quarter flagged by half: {first}; quarters flagged by half in 2010-2012: {int((share.loc[pd.Period('2010Q1', 'Q'):pd.Period('2012Q4', 'Q')] >= 0.5).sum())} of 12")

    lines += ["", "Ranking (1 = highest spread) by actual and by median warranted spread, Portugal / Greece / Ireland:"]
    for qs in SHOW_Q:
        q = pd.Period(qs, "Q")
        h = head[(head["quarter"] == q) & head["geo"].isin(TRIO)]
        act = h.groupby("geo")["spread"].first().rank(ascending=False)
        war = h.groupby("geo")["warranted"].median().rank(ascending=False)
        if act.isna().any():
            continue
        lines.append(f"  {qs}: actual " + ", ".join(f"{NAMES[g]} {int(act[g])}" for g in TRIO) + "; warranted " + ", ".join(f"{NAMES[g]} {int(war[g])}" for g in TRIO))
    pt_ie = head[head["geo"].isin(("PT", "IE"))]
    act = pt_ie.groupby(["quarter", "geo"])["spread"].first().unstack()
    war = pt_ie.groupby(["quarter", "geo"])["warranted"].median().unstack()
    span = act.loc[pd.Period("2010Q1", "Q"):pd.Period("2012Q4", "Q")]
    wspan = war.loc[span.index]
    lines.append(f"  Quarters of 2010-2012 in which Portugal's actual spread exceeded Ireland's: {int((span['PT'] > span['IE']).sum())} of {len(span)}; "
                 f"in which Portugal's warranted spread exceeded Ireland's: {int((wspan['PT'] > wspan['IE']).sum())} of {len(span)}")

    lines += ["", "Does external debt explain Portugal's excess? Residual median in 2011Q4 and 2012Q1 by feature set (headline windows):"]
    for k in FEATURES:
        f = head[(head["geo"] == "PT") & (head["features"] == k) & head["quarter"].isin([pd.Period("2011Q4", "Q"), pd.Period("2012Q1", "Q")])]
        g = f.groupby("quarter")["residual"].median()
        fl = f.groupby("quarter")["flag"].mean()
        lines.append(f"  {k}: 2011Q4 {g.iloc[0]:+.2f} (flagged {fl.iloc[0]:.0%}); 2012Q1 {g.iloc[1]:+.2f} (flagged {fl.iloc[1]:.0%})")

    a = annual_panel()
    lines += ["", "Real-time check: 2009 debt and balance (% of GDP) published in April 2010 against today's vintage, and the effect on the warranted spread "
              "(median slopes of the headline fixed-window linear models)"]
    lin = specs[(specs["window"].isin(["normal_times", "post_omt", "long"])) & (~specs["features"].isin(["nonlinear_debt", "rate_dependent_debt"]))]
    s_debt = float(lin["coef_debt"].median())
    s_bal = float(lin["coef_balance"].median())
    lines.append(f"  slopes: debt {s_debt:+.4f} pp of spread per pp of GDP; balance {s_bal:+.4f}")
    for g, (rt_d, rt_b) in REALTIME_2009.items():
        row = a[(a["geo"] == g) & (a["year"] == 2009)].iloc[0]
        dd_, db_ = float(row["debt"]) - rt_d, float(row["balance"]) - rt_b
        lines.append(f"  {NAMES[g]}: debt {rt_d:.1f} then, {row['debt']:.1f} now ({dd_:+.1f}); balance {rt_b:.1f} then, {row['balance']:.1f} now ({db_:+.1f}); "
                     f"warranted spread with the April 2010 figures lower by about {s_debt * dd_ + s_bal * db_:.2f} pp")

    lines += ["", "The OMT episode (headline models), 2012Q2 -> 2013Q2: actual, warranted median, residual median, share flagged"]
    for g in ("IT", "ES", "PT", "IE", "EL"):
        a1 = head[(head["geo"] == g) & (head["quarter"] == pd.Period("2012Q2", "Q"))]
        b1 = head[(head["geo"] == g) & (head["quarter"] == pd.Period("2013Q2", "Q"))]
        lines.append(f"  {NAMES[g]}: actual {a1.spread.iloc[0]:.2f} -> {b1.spread.iloc[0]:.2f}; warranted {a1.warranted.median():.2f} -> {b1.warranted.median():.2f}; "
                     f"residual {a1.residual.median():+.2f} -> {b1.residual.median():+.2f} ({b1.residual.median() - a1.residual.median():+.2f}); flagged {a1.flag.mean():.0%} -> {b1.flag.mean():.0%}")

    lines += ["", "Mean reversion (estimation panel): slope of the next four quarters' spread change on the residual; median [10-90%] across headline models"]
    mr_h = mr[~mr["spec"].str.endswith("pre_crisis")]
    for s_, f in mr_h.groupby("sample"):
        lines.append(f"  {s_}: residual slope {f.slope_residual.median():+.2f} [{f.slope_residual.quantile(.1):+.2f}, {f.slope_residual.quantile(.9):+.2f}]; warranted slope {f.slope_warranted.median():+.2f}")
    return write("warranted_summary.txt", lines)


if __name__ == "__main__":
    print(run())
