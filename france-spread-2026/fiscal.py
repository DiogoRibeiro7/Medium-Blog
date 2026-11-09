"""Levels and pass-through: France in 2025-2026 against Italy and Spain in 2011.

    python fiscal.py

1. Levels. Debt, interest, the implicit interest rate (interest payable over average debt), the
   market 10-year yield, the gap between them, the primary balance, nominal growth and the
   primary balance that would hold the debt ratio constant, for France in 2025 (and at the
   latest monthly yield) against Italy and Spain in 2011 and France in 2011.
2. Pass-through. The implicit rate moves towards the market yield as debt is refinanced:
   i_t - i_{t-1} = rho (x_t - i_{t-1}) + a + e_t. rho is estimated country by country and pooled
   (country intercepts), 1999-2025. A structural check compares France's rho with the share of
   its debt that is refinanced or newly issued each year (1 / average maturity + deficit / debt).
3. Backtests. rho is fitted on years before a test period only; the implicit rate is then
   simulated from the last fitted year with the actual yields and compared with the actual rate:
   Italy and Spain 2011-2019 (fit 1999-2010) and France 2020-2025 (fit 1999-2019). A second
   simulation holds Italy's and Spain's yields at their 2012 level from 2013 (with each
   country's full-sample rho), to show what the bill would have become if the 2011-2012 yields
   had persisted.

Writes ``results/fiscal_levels.csv``, ``results/fiscal_backtest.csv`` and
``results/fiscal_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, annual, monthly_yields, write

RHO_FIT = (1999, 2025)
POOL = ("FR", "IT", "ES", "DE", "NL", "AT", "FI", "BE")
# Average residual maturity of French negotiable state debt at the end of 2025: 8 years and
# 184 days (Agence France Tresor, monthly bulletin, July 2026).
FR_MATURITY_2025 = 8 + 184 / 365


def estimate_rho(a: pd.DataFrame, countries: tuple[str, ...], years: tuple[int, int]) -> tuple[float, float, dict[str, tuple[float, float]]]:
    """Pooled rho with country intercepts, its standard error, and (rho, se) by country."""
    rows = []
    for c in countries:
        s = a[a["geo"] == c].set_index("year").sort_index()
        f = pd.DataFrame({"di": s["implicit"].diff(), "gap": s["yield"] - s["implicit"].shift(), "c": c})
        rows.append(f.loc[years[0] : years[1]].dropna())
    df = pd.concat(rows)
    z = pd.get_dummies(df["c"], dtype=float)
    z["gap"] = df["gap"]
    zz = z.to_numpy()
    b, *_ = np.linalg.lstsq(zz, df["di"].to_numpy(), rcond=None)
    e = df["di"].to_numpy() - zz @ b
    cov = np.linalg.inv(zz.T @ zz) * (e @ e / (len(df) - zz.shape[1]))
    each = {}
    for c in countries:
        s = df[df["c"] == c]
        x = np.column_stack([np.ones(len(s)), s["gap"]])
        bb, *_ = np.linalg.lstsq(x, s["di"].to_numpy(), rcond=None)
        ee = s["di"].to_numpy() - x @ bb
        cc = np.linalg.inv(x.T @ x) * (ee @ ee / max(len(s) - 2, 1))
        each[c] = (float(bb[1]), float(np.sqrt(cc[1, 1])))
    return float(b[-1]), float(np.sqrt(cov[-1, -1])), each


def simulate(start: float, x: np.ndarray, rho: float) -> np.ndarray:
    """Implicit rate path from ``start`` with market yields ``x`` (one per year)."""
    out = np.empty(len(x))
    i = start
    for t, xt in enumerate(x):
        i = i + rho * (xt - i)
        out[t] = i
    return out


def stabilising_primary(d: float, i: float, g: float) -> float:
    """Primary balance (percent of GDP) that keeps the debt ratio d (percent) constant."""
    return d * (i - g) / (100 + g)


def backtest(a: pd.DataFrame, c: str, fit: tuple[int, int], test: tuple[int, int], hold_from: int | None = None, rho_hold: float | None = None) -> pd.DataFrame:
    _, _, each = estimate_rho(a, (c,), fit)
    rho = each[c][0]
    s = a[a["geo"] == c].set_index("year").sort_index()
    years = np.arange(test[0], test[1] + 1)
    x = s.loc[years, "yield"].to_numpy()
    pred = simulate(float(s.loc[test[0] - 1, "implicit"]), x, rho)
    out = pd.DataFrame({"geo": c, "year": years, "yield": x, "actual": s.loc[years, "implicit"].to_numpy(), "predicted": pred, "rho": rho})
    if hold_from is not None:
        xh = np.where(years >= hold_from, s.loc[hold_from - 1, "yield"], x)
        out["held"] = simulate(float(s.loc[test[0] - 1, "implicit"]), xh, rho if rho_hold is None else rho_hold)
        out["held_int_gdp"] = out["held"] / 100 * s.loc[years, "avg_debt"].to_numpy() / s.loc[years, "gdp"].to_numpy() * 100
        out["actual_int_gdp"] = s.loc[years, "int_gdp"].to_numpy()
    return out


def run() -> str:
    a = annual()
    m = monthly_yields()
    lines: list[str] = []
    latest = m.dropna(subset=["FR", "DE"]).index.max()
    lines.append(f"Latest monthly 10-year yields ({latest}): " + ", ".join(f"{c} {m.loc[latest, c]:.2f}" for c in ("FR", "DE", "IT", "ES")))
    lines.append(f"Spreads to Germany ({latest}): " + ", ".join(f"{c} {m.loc[latest, c] - m.loc[latest, 'DE']:+.2f}" for c in ("FR", "IT", "ES")))

    # 1. levels
    rows = []
    cases = [("FR", 2025), ("IT", 2011), ("ES", 2011), ("FR", 2011), ("IT", 2025), ("ES", 2025)]
    for c, y in cases:
        r = a[(a["geo"] == c) & (a["year"] == y)].iloc[0]
        rows.append(
            dict(case=f"{c} {y}", geo=c, year=y, debt_gdp=r.debt_gdp, int_gdp=r.int_gdp, implicit=r.implicit, market=r["yield"],
                 gap=r["yield"] - r.implicit, primary=r.primary, balance=r.balance_gdp, g=r.g,
                 pb_star_implicit=stabilising_primary(r.debt_gdp, r.implicit, r.g), pb_star_market=stabilising_primary(r.debt_gdp, r["yield"], r.g))
        )
    fr25 = a[(a["geo"] == "FR") & (a["year"] == 2025)].iloc[0]
    xl = float(m.loc[latest, "FR"])
    rows.append(dict(case=f"FR 2025 at {latest} yield", geo="FR", year=2025, debt_gdp=fr25.debt_gdp, int_gdp=fr25.int_gdp, implicit=fr25.implicit,
                     market=xl, gap=xl - fr25.implicit, primary=fr25.primary, balance=fr25.balance_gdp, g=fr25.g,
                     pb_star_implicit=stabilising_primary(fr25.debt_gdp, fr25.implicit, fr25.g), pb_star_market=stabilising_primary(fr25.debt_gdp, xl, fr25.g)))
    lv = pd.DataFrame(rows)
    lv.to_csv(RESULTS / "fiscal_levels.csv", index=False)
    lines.append("")
    lines.append("Levels: debt % GDP / interest % GDP / implicit rate / market yield / gap / primary balance % GDP / overall balance / nominal growth / debt-stabilising primary balance at the implicit rate | at the market yield")
    for r in lv.itertuples():
        lines.append(f"  {r.case}: {r.debt_gdp:.1f} / {r.int_gdp:.2f} / {r.implicit:.2f} / {r.market:.2f} / {r.gap:+.2f} / {r.primary:+.2f} / {r.balance:+.2f} / {r.g:+.2f} / {r.pb_star_implicit:+.2f} | {r.pb_star_market:+.2f}")

    # 2. pass-through
    rho, se, each = estimate_rho(a, POOL, RHO_FIT)
    lines.append("")
    lines.append(f"Pass-through rho, pooled {', '.join(POOL)} with country intercepts, {RHO_FIT[0]}-{RHO_FIT[1]}: {rho:.3f} (SE {se:.3f})")
    lines.append("  by country: " + ", ".join(f"{c} {v[0]:.3f} ({v[1]:.3f})" for c, v in each.items()))
    rfr = each["FR"][0]
    share = 100 / FR_MATURITY_2025 / 100 + (-fr25.balance_gdp) / fr25.debt_gdp
    lines.append(f"Structural check, France 2025: 1 / average maturity ({FR_MATURITY_2025:.2f} years) = {1 / FR_MATURITY_2025:.3f}; deficit / debt = {-fr25.balance_gdp / fr25.debt_gdp:.3f}; sum {share:.3f} against estimated rho {rfr:.3f}")
    gap0 = xl - fr25.implicit
    lines.append(f"Half-life of a gap between market yield and implicit rate at France's rho: {np.log(0.5) / np.log(1 - rfr):.1f} years; share of today's gap still open after 5 and 10 years: {(1 - rfr) ** 5:.2f}, {(1 - rfr) ** 10:.2f}")
    lines.append(f"France: gap between the {latest} yield and the 2025 implicit rate {gap0:+.2f} pp, of which German yield level {float(m.loc[latest, 'DE']) - fr25.implicit:+.2f} and French spread {xl - float(m.loc[latest, 'DE']):+.2f}")

    # 3. backtests
    bt = pd.concat([
        backtest(a, "IT", (1999, 2010), (2011, 2019), hold_from=2013, rho_hold=each["IT"][0]),
        backtest(a, "ES", (1999, 2010), (2011, 2019), hold_from=2013, rho_hold=each["ES"][0]),
        backtest(a, "FR", (1999, 2019), (2020, 2025)),
    ], ignore_index=True)
    bt.to_csv(RESULTS / "fiscal_backtest.csv", index=False)
    lines.append("")
    lines.append("Backtests: rho fitted before the test period, implicit rate simulated with actual yields; error = predicted - actual (pp)")
    for c, f in bt.groupby("geo", sort=False):
        err = f["predicted"] - f["actual"]
        lines.append(f"  {c} {f.year.min()}-{f.year.max()}: rho {f.rho.iloc[0]:.3f}; mean error {err.mean():+.2f}, RMSE {np.sqrt((err ** 2).mean()):.2f}, max abs {err.abs().max():.2f}; "
                     + "actual/predicted " + ", ".join(f"{int(y)} {ac:.2f}/{pr:.2f}" for y, ac, pr in zip(f.year, f.actual, f.predicted)))
        if "held" in f and f["held"].notna().any():
            last = f.iloc[-1]
            lines.append(f"    if yields had stayed at their 2012 level from 2013 (full-sample rho {each[c][0]:.3f}): implicit rate {last.year:.0f} {last.held:.2f} (actual {last.actual:.2f}); interest % GDP {last.held_int_gdp:.2f} (actual {last.actual_int_gdp:.2f})")
    for c in ("IT", "ES"):
        s = a[a["geo"] == c].set_index("year")
        lines.append(f"  {c}: implicit rate 2010 {s.loc[2010, 'implicit']:.2f}, peak {s.loc[2010:2016, 'implicit'].max():.2f} in {int(s.loc[2010:2016, 'implicit'].idxmax())}; market yield peak annual {s.loc[2010:2016, 'yield'].max():.2f} in {int(s.loc[2010:2016, 'yield'].idxmax())}; interest % GDP 2010 {s.loc[2010, 'int_gdp']:.2f}, peak {s.loc[2010:2016, 'int_gdp'].max():.2f} in {int(s.loc[2010:2016, 'int_gdp'].idxmax())}")
    return write("fiscal_summary.txt", lines)


if __name__ == "__main__":
    print(run())
