"""Counterfactual 10-year government bond yields for the euro-area crisis.

    python yields.py

Two counterfactuals for what each country's 10-year yield would have been without the crisis:

- ``no_fragmentation`` (CF-A): every country keeps its 2003-2007 spread to the GDP-weighted
  euro-area average yield. The crisis is read as a redistribution of yields around that average.
- ``safe_asset`` (CF-B): Germany's yield follows a synthetic control built from ten-year yields of
  advanced economies outside the euro area (non-negative weights summing to one, fitted on
  demeaned 1999-2009 data), and every other country keeps its 2003-2007 spread to Germany.

The crisis window is January 2010 to December 2014, before the ECB announced its public-sector
purchase programme in January 2015. Gaps outside the window are reported but not attributed to
the crisis. Placebo runs repeat the synthetic control for every donor country.

Writes ``results/yields_*.csv`` and ``results/yields_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

PROGRAMME = ("EL", "IE", "PT", "CY", "ES")
CORE = ("DE", "FR", "NL", "FI", "AT")
OTHERS = ("IT", "BE")
LEDGER = PROGRAMME + CORE + OTHERS
EA_AVERAGE = ("AT", "BE", "DE", "EL", "ES", "FI", "FR", "IE", "IT", "LU", "NL", "PT")
DONORS = ("US", "GB", "CA", "SE", "NO", "CH", "JP", "AU", "NZ")
PRE_SPREAD = ("2003-01", "2007-12")
PRE_FIT = ("1999-01", "2009-12")
WINDOW = ("2010-01", "2014-12")
WINDOWS = {"main 2010-2014": WINDOW, "acute 2010-2012": ("2010-01", "2012-12"), "to QE 2010-2015": ("2010-01", "2015-12")}
WEIGHT_YEAR = 2009


def load() -> tuple[pd.DataFrame, pd.Series]:
    """Monthly yields in percent, one column per country; GDP weights for the euro-area average."""
    e = pd.read_csv(DATA / "yields_monthly.csv", keep_default_na=False, na_values=[""])
    e = e.pivot_table(index="time", columns="geo", values="value")
    e = e.drop(columns=[c for c in DONORS if c in e])
    f = pd.read_csv(DATA / "fred_long_rates.csv", keep_default_na=False, na_values=[""])
    f = f[f["country"].isin(DONORS)]
    f["time"] = f["date"].str[:7]
    f = f.pivot_table(index="time", columns="country", values="value")
    y = e.join(f, how="outer").sort_index()
    y = y.loc["1995-01":]
    g = pd.read_csv(DATA / "gdp.csv", keep_default_na=False, na_values=[""])
    g = g[g["time"] == WEIGHT_YEAR].set_index("geo")["value"]
    w = g.reindex(EA_AVERAGE).dropna()
    return y, w / w.sum()


def ea_average(y: pd.DataFrame, w: pd.Series) -> pd.Series:
    """GDP-weighted average over the countries with a yield in each month (Greece from 2001)."""
    cols = [c for c in w.index if c in y]
    v = y[cols]
    ww = v.notna().mul(w[cols], axis=1)
    return (v.fillna(0) * ww).sum(axis=1) / ww.sum(axis=1)


def synthetic(target: pd.Series, donors: pd.DataFrame, fit: tuple[str, str]) -> tuple[pd.Series, pd.Series, float]:
    """Demeaned synthetic control: non-negative weights summing to one."""
    t, d = target.loc[fit[0] : fit[1]], donors.loc[fit[0] : fit[1]]
    ok = t.notna() & d.notna().all(axis=1)
    t, d = t[ok], d[ok]
    tm, dm = t - t.mean(), d - d.mean()
    k = d.shape[1]

    def loss(wv: np.ndarray) -> float:
        return float(((tm.to_numpy() - dm.to_numpy() @ wv) ** 2).mean())

    res = minimize(loss, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda wv: wv.sum() - 1}], options={"ftol": 1e-12, "maxiter": 2000})
    wv = pd.Series(np.clip(res.x, 0, None), index=d.columns)
    wv = wv / wv.sum()
    path = t.mean() + (donors - d.mean()).mul(wv, axis=1).sum(axis=1, min_count=len(wv))
    return path, wv, float(np.sqrt(loss(wv.to_numpy())))


def placebo(y: pd.DataFrame, fit: tuple[str, str], window: tuple[str, str]) -> pd.DataFrame:
    """Synthetic control for Germany and for every donor, from the remaining donors."""
    rows = []
    for unit in ("DE",) + DONORS:
        pool = [c for c in DONORS if c != unit]
        path, _, pre_rmse = synthetic(y[unit], y[pool], fit)
        gap = y[unit] - path
        win = gap.loc[window[0] : window[1]]
        rows.append({"unit": unit, "pre_rmse": pre_rmse, "window_mean_gap": win.mean(), "window_rmse": float(np.sqrt((win**2).mean())), "ratio": float(np.sqrt((win**2).mean())) / pre_rmse})
    return pd.DataFrame(rows)


def counterfactuals(y: pd.DataFrame, w: pd.Series) -> tuple[pd.DataFrame, pd.Series, float]:
    avg = ea_average(y, w)
    spread_avg = (y[list(LEDGER)].sub(avg, axis=0)).loc[PRE_SPREAD[0] : PRE_SPREAD[1]].mean()
    spread_de = (y[list(LEDGER)].sub(y["DE"], axis=0)).loc[PRE_SPREAD[0] : PRE_SPREAD[1]].mean()
    de_syn, weights, pre_rmse = synthetic(y["DE"], y[list(DONORS)], PRE_FIT)
    rows = []
    for c in LEDGER:
        rows.append(pd.DataFrame({"geo": c, "time": y.index, "actual": y[c].to_numpy(), "no_fragmentation": (avg + spread_avg[c]).to_numpy(), "safe_asset": (de_syn + spread_de[c]).to_numpy(), "ea_average": avg.to_numpy()}))
    cf = pd.concat(rows, ignore_index=True)
    return cf, weights, pre_rmse


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    y, w = load()
    cf, weights, pre_rmse = counterfactuals(y, w)
    cf.to_csv(RESULTS / "yields_monthly_cf.csv", index=False, float_format="%.4f")
    weights.rename("weight").to_csv(RESULTS / "yields_synthetic_weights.csv", float_format="%.4f")
    pl = placebo(y, PRE_FIT, WINDOW)
    pl.to_csv(RESULTS / "yields_placebo.csv", index=False, float_format="%.4f")

    cf["year"] = cf["time"].str[:4].astype(int)
    cf["gap_no_fragmentation"] = cf["actual"] - cf["no_fragmentation"]
    cf["gap_safe_asset"] = cf["actual"] - cf["safe_asset"]
    annual = cf.groupby(["geo", "year"])[["actual", "no_fragmentation", "safe_asset", "gap_no_fragmentation", "gap_safe_asset"]].mean().reset_index()
    annual.to_csv(RESULTS / "yields_annual.csv", index=False, float_format="%.4f")

    lines = [
        f"Euro-area average: GDP-weighted ({WEIGHT_YEAR} GDP) over {', '.join(EA_AVERAGE)}; pre-crisis spreads averaged {PRE_SPREAD[0]} to {PRE_SPREAD[1]}",
        f"Synthetic Germany, fitted {PRE_FIT[0]} to {PRE_FIT[1]} on demeaned yields, pre-fit RMSE {pre_rmse:.3f} pp; weights: " + ", ".join(f"{k} {v:.3f}" for k, v in weights.sort_values(ascending=False).items() if v > 0.001),
        "",
        "Pre-crisis spreads (pp), 2003-2007, to the euro-area average and to Germany:",
    ]
    for c in LEDGER:
        s = cf[(cf["geo"] == c) & cf["time"].between(*PRE_SPREAD)]
        lines.append(f"  {c}: {(s['actual'] - s['ea_average']).mean():+.3f} / {(s['actual'] - cf[(cf['geo'] == 'DE') & cf['time'].between(*PRE_SPREAD)]['actual'].to_numpy()).mean():+.3f}")
    for name, (a, b) in WINDOWS.items():
        lines += ["", f"Mean yield gap, actual minus counterfactual (pp), {name} ({a} to {b}): no-fragmentation / safe-asset"]
        for c in LEDGER:
            s = cf[(cf["geo"] == c) & cf["time"].between(a, b)]
            lines.append(f"  {c}: {s['gap_no_fragmentation'].mean():+.3f} / {s['gap_safe_asset'].mean():+.3f}")
    lines += ["", "Gaps after the window, 2015-2026 (pp), not attributed to the crisis: no-fragmentation / safe-asset"]
    for c in LEDGER:
        s = cf[(cf["geo"] == c) & (cf["year"] >= 2015)]
        lines.append(f"  {c}: {s['gap_no_fragmentation'].mean():+.3f} / {s['gap_safe_asset'].mean():+.3f}")
    peak = cf.loc[cf.groupby("geo")["actual"].idxmax(), ["geo", "time", "actual"]].set_index("geo").reindex(LEDGER)
    lines += ["", "Peak monthly yield (percent): " + ", ".join(f"{c} {r.actual:.2f} ({r.time})" for c, r in peak.iterrows())]
    de = pl[pl["unit"] == "DE"].iloc[0]
    rank = int((pl["ratio"] >= de["ratio"]).sum())
    lines += [
        "",
        f"Placebo, synthetic control for Germany and each donor, window {WINDOW[0]} to {WINDOW[1]}:",
    ]
    for r in pl.sort_values("ratio", ascending=False).itertuples(index=False):
        lines.append(f"  {r.unit}: pre-fit RMSE {r.pre_rmse:.3f}, mean window gap {r.window_mean_gap:+.3f}, window RMSE / pre RMSE {r.ratio:.2f}")
    lines.append(f"  Germany ranks {rank} of {len(pl)} by the ratio (permutation p = {rank / len(pl):.2f})")
    text = "\n".join(lines) + "\n"
    (RESULTS / "yields_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
