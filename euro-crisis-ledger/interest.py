"""From yield gaps to euros: the interest account of the euro-crisis ledger.

    python interest.py

Reads ``results/yields_annual.csv`` (``yields.py``) and Eurostat general-government interest
payable (D.41) and debt (EDP debt), 1995-2025.

The implicit interest rate i_t (interest payable over average debt) moves towards the market
yield x_t as debt is refinanced: i_t - i_{t-1} = rho (x_t - i_{t-1}) + e_t. rho, the share of the
debt repriced each year, is estimated on the five core countries, which borrowed in the market
throughout (pooled with country intercepts, 1999-2025).

Two calculations, for each counterfactual:

- **Market repricing** (all countries): the crisis-window yield gap g_t (actual minus
  counterfactual, 2010-2014, zero afterwards) passes into the implicit rate as
  d_t = (1 - rho) d_{t-1} + rho g_t, and the interest effect is d_t times average debt. Gaps
  from 2010 to 2014 keep acting until 2025 through debt issued in those years.
- **Actual against counterfactual** (all countries; the relevant one for programme countries,
  which stopped borrowing in the market and paid official rates instead): actual interest
  minus the interest that a counterfactual implicit rate, simulated from 2009 with the
  counterfactual yields, would have produced on the actual debt. Two versions: counterfactual
  yields in the window only, and in every year from 2010 (no crisis at all).

Positive numbers are extra interest paid because of the crisis; negative numbers are savings.
Writes ``results/interest_*.csv`` and ``results/interest_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

PROGRAMME = ("EL", "IE", "PT", "CY", "ES")
CORE = ("DE", "FR", "NL", "FI", "AT")
COUNTRIES = PROGRAMME + CORE + ("IT", "BE")
CFS = ("no_fragmentation", "safe_asset")
WINDOW = (2010, 2014)
LAST = 2025
RHO_FIT = (1999, 2025)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g = pd.read_csv(DATA / "gov_main.csv", keep_default_na=False, na_values=[""])
    interest = g[g["na_item"] == "D41PAY"].pivot_table(index="time", columns="geo", values="value")
    d = pd.read_csv(DATA / "debt.csv", keep_default_na=False, na_values=[""])
    debt = d.pivot_table(index="time", columns="geo", values="value")
    gdp = pd.read_csv(DATA / "gdp.csv", keep_default_na=False, na_values=[""]).pivot_table(index="time", columns="geo", values="value")
    a = pd.read_csv(RESULTS / "yields_annual.csv", keep_default_na=False, na_values=[""])
    return interest, debt, gdp, a.pivot_table(index="year", columns="geo", values="actual"), a


def implicit_rate(interest: pd.DataFrame, debt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg = (debt + debt.shift(1)) / 2
    return interest / avg * 100, avg


def estimate_rho(rate: pd.DataFrame, x: pd.DataFrame, countries: tuple[str, ...], years: tuple[int, int]) -> tuple[float, float, dict[str, float]]:
    """Pooled pass-through with country intercepts, and country-by-country estimates."""
    rows = []
    for c in countries:
        rows.append(pd.DataFrame({"di": rate[c] - rate[c].shift(1), "gap": x[c] - rate[c].shift(1), "c": c}).loc[years[0] : years[1]].dropna())
    df = pd.concat(rows)
    z = pd.get_dummies(df["c"], dtype=float)
    z["gap"] = df["gap"]
    b, *_ = np.linalg.lstsq(z.to_numpy(), df["di"].to_numpy(), rcond=None)
    e = df["di"].to_numpy() - z.to_numpy() @ b
    cov = np.linalg.inv(z.T.to_numpy() @ z.to_numpy()) * (e @ e / (len(df) - z.shape[1]))
    each = {}
    for c in countries:
        s = df[df["c"] == c]
        zz = np.column_stack([np.ones(len(s)), s["gap"]])
        each[c] = float(np.linalg.lstsq(zz, s["di"].to_numpy(), rcond=None)[0][1])
    return float(b[-1]), float(np.sqrt(cov[-1, -1])), each


def repricing(gap: pd.Series, rho: float, years: range) -> pd.Series:
    """d_t = (1 - rho) d_{t-1} + rho g_t, with g_t the window gap (zero outside the window)."""
    d, out = 0.0, {}
    for y in years:
        g = float(gap.get(y, 0.0)) if WINDOW[0] <= y <= WINDOW[1] else 0.0
        d = (1 - rho) * d + rho * g
        out[y] = d
    return pd.Series(out)


def simulate(start: float, x: pd.Series, rho: float, years: range) -> pd.Series:
    i, out = start, {}
    for y in years:
        i = i + rho * (float(x[y]) - i)
        out[y] = i
    return pd.Series(out)


def run(rho_by: dict[str, float], rate: pd.DataFrame, avg_debt: pd.DataFrame, interest: pd.DataFrame, gdp: pd.DataFrame, x: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    years = range(WINDOW[0], LAST + 1)
    rows = []
    for c in COUNTRIES:
        a = annual[annual["geo"] == c].set_index("year")
        rho = rho_by[c]
        for cf in CFS:
            gap = a[f"gap_{cf}"]
            d = repricing(gap, rho, years)
            xcf = x[c].copy()
            for y in range(WINDOW[0], WINDOW[1] + 1):
                xcf[y] = x[c][y] - gap[y]
            sim_cf = simulate(float(rate.loc[WINDOW[0] - 1, c]), xcf, rho, years)
            # no crisis at all: counterfactual yields in every year from 2010, not only in the window
            xfull = x[c].copy()
            for y in years:
                if y in gap.index and not np.isnan(gap[y]):
                    xfull[y] = x[c][y] - gap[y]
            sim_full = simulate(float(rate.loc[WINDOW[0] - 1, c]), xfull, rho, years)
            sim_act = simulate(float(rate.loc[WINDOW[0] - 1, c]), x[c], rho, years)
            for y in years:
                D = float(avg_debt.loc[y, c])
                rows.append({
                    "geo": c, "cf": cf, "year": y, "rho": rho, "gap": float(gap.get(y, np.nan)), "repricing_pp": d[y],
                    "market_effect": d[y] / 100 * D, "actual_rate": float(rate.loc[y, c]), "cf_rate": sim_cf[y], "model_rate": sim_act[y],
                    "actual_effect": (float(rate.loc[y, c]) - sim_cf[y]) / 100 * D, "model_error": (float(rate.loc[y, c]) - sim_act[y]) / 100 * D,
                    "cf_rate_full": sim_full[y], "actual_effect_full": (float(rate.loc[y, c]) - sim_full[y]) / 100 * D,
                    "interest": float(interest.loc[y, c]), "avg_debt": D, "gdp": float(gdp.loc[y, c]),
                })
    return pd.DataFrame(rows)


def totals(r: pd.DataFrame, col: str, upto: int = LAST) -> pd.DataFrame:
    s = r[r["year"] <= upto].groupby(["geo", "cf"]).agg(eur_bn=(col, lambda v: v.sum() / 1e3)).reset_index()
    pct = r[r["year"] <= upto].assign(p=lambda d: d[col] / d["gdp"] * 100).groupby(["geo", "cf"])["p"].sum().rename("pct_gdp_years").reset_index()
    return s.merge(pct, on=["geo", "cf"])


def main() -> int:
    interest, debt, gdp, x, annual = load()
    rate, avg_debt = implicit_rate(interest, debt)
    rho, se, each = estimate_rho(rate, x, CORE, RHO_FIT)
    rho_pre, se_pre, _ = estimate_rho(rate, x, CORE + PROGRAMME[1:] + ("IT", "BE"), (1999, 2009))
    variants = {"pooled": {c: rho for c in COUNTRIES}, "country": {c: each.get(c, rho) for c in COUNTRIES}, "pre-crisis": {c: rho_pre for c in COUNTRIES}, "low 0.10": {c: 0.10 for c in COUNTRIES}, "high 0.30": {c: 0.30 for c in COUNTRIES}}
    main_r = run(variants["pooled"], rate, avg_debt, interest, gdp, x, annual)
    main_r.to_csv(RESULTS / "interest_by_year.csv", index=False, float_format="%.4f")
    rows = []
    for name, rb in variants.items():
        r = main_r if name == "pooled" else run(rb, rate, avg_debt, interest, gdp, x, annual)
        for col in ("market_effect", "actual_effect", "actual_effect_full"):
            t = totals(r, col)
            t["method"], t["rho_variant"] = col, name
            rows.append(t)
        w = totals(r, "market_effect", WINDOW[1])
        w["method"], w["rho_variant"] = "market_effect_to_2014", name
        rows.append(w)
    tot = pd.concat(rows, ignore_index=True)
    tot.to_csv(RESULTS / "interest_totals.csv", index=False, float_format="%.4f")

    lines = [
        f"Pass-through rho (share of debt repriced per year), core countries pooled {RHO_FIT[0]}-{RHO_FIT[1]}: {rho:.3f} (SE {se:.3f}); pre-crisis 1999-2009, eleven euro countries: {rho_pre:.3f} (SE {se_pre:.3f})",
        "Country estimates (core, 1999-2025): " + ", ".join(f"{c} {v:.3f}" for c, v in each.items()),
        f"Implied share of a 2014 rate gap still acting in {LAST}: {(1 - rho) ** (LAST - WINDOW[1]):.2f}",
        "",
        "Implicit interest rate on general government debt (percent), 2009 / 2012 / 2014 / 2019 / 2025:",
    ]
    for c in COUNTRIES:
        lines.append(f"  {c}: " + " / ".join(f"{rate.loc[y, c]:.2f}" for y in (2009, 2012, 2014, 2019, 2025)))
    for method, label in (("market_effect", "Market repricing"), ("actual_effect", "Actual against counterfactual (window yields)"), ("actual_effect_full", "Actual against a no-crisis counterfactual (counterfactual yields in every year from 2010)")):
        lines += ["", f"{label}, cumulative 2010-{LAST}, EUR bn (sum of annual % of GDP), pooled rho; no-fragmentation | safe-asset:"]
        t = tot[(tot["method"] == method) & (tot["rho_variant"] == "pooled")]
        for c in COUNTRIES:
            a = t[(t["geo"] == c) & (t["cf"] == "no_fragmentation")].iloc[0]
            b = t[(t["geo"] == c) & (t["cf"] == "safe_asset")].iloc[0]
            lines.append(f"  {c}: {a.eur_bn:+.1f} ({a.pct_gdp_years:+.2f}) | {b.eur_bn:+.1f} ({b.pct_gdp_years:+.2f})")
    t = tot[(tot["method"] == "market_effect_to_2014") & (tot["rho_variant"] == "pooled")]
    lines += ["", "Market repricing, 2010-2014 only, EUR bn; no-fragmentation | safe-asset:"]
    for c in COUNTRIES:
        lines.append(f"  {c}: {t[(t.geo == c) & (t.cf == 'no_fragmentation')].eur_bn.iloc[0]:+.1f} | {t[(t.geo == c) & (t.cf == 'safe_asset')].eur_bn.iloc[0]:+.1f}")
    lines += ["", "Sensitivity of the market-repricing total 2010-2025 to rho, EUR bn, no-fragmentation | safe-asset:"]
    for c in COUNTRIES:
        parts = []
        for name in variants:
            t = tot[(tot["method"] == "market_effect") & (tot["rho_variant"] == name) & (tot["geo"] == c)]
            parts.append(f"{name} {t[t.cf == 'no_fragmentation'].eur_bn.iloc[0]:+.1f}|{t[t.cf == 'safe_asset'].eur_bn.iloc[0]:+.1f}")
        lines.append(f"  {c}: " + "; ".join(parts))
    me = main_r[main_r["cf"] == "no_fragmentation"].groupby("geo")["model_error"].sum() / 1e3
    lines += ["", "Model check: actual interest minus interest simulated from actual yields, cumulative 2010-2025, EUR bn (official lending, restructuring and model error):"]
    lines.append("  " + ", ".join(f"{c} {me[c]:+.1f}" for c in COUNTRIES))
    for grp, members in (("core", CORE), ("programme", PROGRAMME)):
        for cf in CFS:
            for method in ("market_effect", "actual_effect", "actual_effect_full"):
                t = tot[(tot["method"] == method) & (tot["rho_variant"] == "pooled") & (tot["cf"] == cf) & tot["geo"].isin(members)]
                lines.append(f"Total {grp}, {cf}, {method}: EUR {t.eur_bn.sum():+.1f} bn")
    text = "\n".join(lines) + "\n"
    (RESULTS / "interest_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
