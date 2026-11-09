"""The slow bill: France's interest burden under yield, growth and primary-balance scenarios.

    python stress.py

From the 2025 accounts (debt ratio, implicit interest rate), the implicit rate moves towards the
annual average 10-year yield at France's estimated pass-through rho, and the debt ratio follows

    d_t = (A (1 + i_t / 200) - pb_t) / (1 - i_t / 200),   A = d_{t-1} / (1 + g_t / 100),

so that interest is charged on average debt within the year (i_t and g_t in percent, pb_t the
primary balance in percent of GDP). Interest as a share of GDP is i_t times average debt.

Scenarios (all assumptions, stated in the summary):

- Yields from 2027: hold at the latest observed level; back to the 2025 average; German yield
  held with the French spread back to its 2025 average; and the latest level plus the rise in
  Italy's spread to Germany in 2011-2014 relative to 2010, applied to 2027-2030.
  2026 uses the observed monthly average to date, with the latest month carried to December.
- Primary balance: 2026 from the government's deficit forecast (5.4% of GDP) less modelled
  interest; then either held at that level ("no adjustment") or the 2027 deficit target of
  5.0% followed by an improvement of 0.5 point of GDP a year up to +1.0% ("plan").
- Nominal growth: 2026 from the national accounts to date; 2.5% a year from 2027 (1.5% and
  3.5% as alternatives).

A Monte Carlo adds uncertainty: annual yields follow a random walk from the latest level with
the standard deviation of France's annual yield changes, 1999-2025, and nominal growth is drawn
around 2.5% with the standard deviation of 1999-2019 growth. 20,000 paths, fixed seed.

Writes ``results/stress_paths.csv``, ``results/stress_mc.csv`` and ``results/stress_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, annual, monthly_yields, quarterly, write
from fiscal import POOL, RHO_FIT, estimate_rho

START = 2025
END = 2036
DEFICIT_2026 = 5.4  # government forecast, PLF 2027 (1 October 2026)
DEFICIT_2027 = 5.0  # PLF 2027 target
G_BASE, G_LOW, G_HIGH = 2.5, 1.5, 3.5
PB_STEP, PB_CAP = 0.5, 1.0
SEED = 20261109
N_PATHS = 20_000


def step(d_prev: float, i: float, g: float, pb: float) -> tuple[float, float]:
    """One year of debt dynamics: returns (debt ratio, interest as % of GDP)."""
    a = d_prev / (1 + g / 100)
    d = (a * (1 + i / 200) - pb) / (1 - i / 200)
    return d, i / 100 * (a + d) / 2


def project(d0: float, i0: float, yields: np.ndarray, growth: np.ndarray, rho: float, primary: str, years: np.ndarray) -> pd.DataFrame:
    """Debt, implicit rate and interest paths. ``primary`` is 'held' or 'plan'."""
    d, i = d0, i0
    rows = []
    pb_held = prev_pb = None
    for y, x, g in zip(years, yields, growth):
        i = i + rho * (x - i)
        if y == 2026:
            # deficit fixed at the forecast: solve for the primary balance given modelled interest
            a = d / (1 + g / 100)
            d_new = (a + DEFICIT_2026) / 1.0
            ig = i / 100 * (a + d_new) / 2
            pb = ig - DEFICIT_2026
            d = a - pb + ig
            pb_held = pb
        else:
            if primary == "held":
                pb = pb_held
            elif y == 2027:
                a = d / (1 + g / 100)
                ig = i / 100 * (a + a + DEFICIT_2027) / 2
                pb = ig - DEFICIT_2027
            else:
                pb = min(prev_pb + PB_STEP, PB_CAP)
            d, ig = step(d, i, g, pb)
        prev_pb = pb
        rows.append(dict(year=int(y), yield_=x, growth=g, implicit=i, primary=pb, debt_gdp=d, int_gdp=ig, balance=pb - ig))
    return pd.DataFrame(rows).rename(columns={"yield_": "yield"})


def project_many(d0: float, i0: float, X: np.ndarray, G: np.ndarray, rho: float, primary: str, years: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``project`` over paths (rows of X and G). Returns interest % GDP and debt % GDP."""
    n, T = X.shape
    d = np.full(n, d0)
    i = np.full(n, i0)
    IG, D = np.empty((n, T)), np.empty((n, T))
    pb_held = prev = None
    for k, y in enumerate(years):
        g = G[:, k]
        i = i + rho * (X[:, k] - i)
        a = d / (1 + g / 100)
        if y == 2026:
            d = a + DEFICIT_2026
            ig = i / 100 * (a + d) / 2
            pb = ig - DEFICIT_2026
            pb_held = pb
        else:
            if primary == "held":
                pb = pb_held
            elif y == 2027:
                pb = i / 100 * (a + a + DEFICIT_2027) / 2 - DEFICIT_2027
            else:
                pb = np.minimum(prev + PB_STEP, PB_CAP)
            d = (a * (1 + i / 200) - pb) / (1 - i / 200)
            ig = i / 100 * (a + d) / 2
        prev = pb
        IG[:, k], D[:, k] = ig, d
    return IG, D


def inputs() -> dict:
    a = annual()
    m = monthly_yields()
    q = quarterly()
    fr = a[a["geo"] == "FR"].set_index("year")
    rho, _, each = estimate_rho(a, POOL, RHO_FIT)
    latest = m.dropna(subset=["FR"]).index.max()
    y26 = m.loc[m.index.year == 2026, "FR"]
    y26_avg = float((y26.sum() + y26.iloc[-1] * (12 - len(y26))) / 12)
    de26 = m.loc[m.index.year == 2026, "DE"]
    qf = q[q["geo"] == "FR"].set_index("quarter")["gdp"]
    have = [p for p in qf.index if p.startswith("2026")]
    prev = [p.replace("2026", "2025") for p in have]
    g26 = 100 * (qf.loc[have].sum() / qf.loc[prev].sum() - 1)
    it = a[a["geo"] == "IT"].set_index("year")
    de = a[a["geo"] == "DE"].set_index("year")
    it_spread = (it["yield"] - de["yield"]).loc[2010:2014]
    shock = (it_spread.loc[2011:2014] - it_spread.loc[2010]).to_numpy()
    fr_spread_2025 = float(fr.loc[2025, "yield"] - de.loc[2025, "yield"])
    dy = fr["yield"].loc[1999:2025].diff().dropna()
    gg = fr["g"].loc[1999:2019].dropna()
    qi = q[q["geo"] == "FR"].set_index("quarter")["interest"]
    h1 = float(qi.loc[[x for x in have if x <= "2026-Q2"]].sum() / qi.loc[[x for x in prev if x <= "2025-Q2"]].sum())
    return dict(
        h1_ratio=h1, int26_check=float(fr.loc[START, "interest"]) * h1 / 1000,
        d0=float(fr.loc[START, "debt_gdp"]), i0=float(fr.loc[START, "implicit"]), int0=float(fr.loc[START, "int_gdp"]),
        gdp0=float(fr.loc[START, "gdp"]), rho=each["FR"][0], rho_pooled=rho, latest=str(latest), x_latest=float(m.loc[latest, "FR"]),
        de_latest=float(m.loc[latest, "DE"]), y26=y26_avg, months26=len(y26), de26=float(de26.mean()), g26=float(g26), quarters26=len(have),
        x2025=float(fr.loc[2025, "yield"]), fr_spread_2025=fr_spread_2025, it_shock=shock, sd_dy=float(dy.std()), sd_g=float(gg.std()),
        it_2012_int=float(it.loc[2012, "int_gdp"]),
    )


def yield_path(kind: str, p: dict, years: np.ndarray) -> np.ndarray:
    x = np.full(len(years), p["x_latest"])
    x[years == 2026] = p["y26"]
    later = years >= 2027
    if kind == "back_to_2025":
        x[later] = p["x2025"]
    elif kind == "spread_back":
        x[later] = p["de_latest"] + p["fr_spread_2025"]
    elif kind == "italy_2011":
        for k, s in enumerate(p["it_shock"]):
            x[years == 2027 + k] += s
    return x


def run() -> str:
    p = inputs()
    years = np.arange(START + 1, END + 1)
    growth = {"base": np.where(years == 2026, p["g26"], G_BASE), "low": np.where(years == 2026, p["g26"], G_LOW), "high": np.where(years == 2026, p["g26"], G_HIGH)}
    paths = []
    for yk in ("hold", "back_to_2025", "spread_back", "italy_2011"):
        for pk in ("plan", "held"):
            for gk in ("base", "low", "high"):
                for rk, rho in (("france", p["rho"]), ("pooled", p["rho_pooled"]), ("low", 0.10), ("high", 0.30)):
                    f = project(p["d0"], p["i0"], yield_path(yk, p, years), growth[gk], rho, pk, years)
                    f[["yields", "primary_path", "growth_path", "rho_case"]] = yk, pk, gk, rk
                    paths.append(f)
    P = pd.concat(paths, ignore_index=True)
    # euros: nominal GDP path from 2025
    gdp = {}
    for gk, gv in growth.items():
        level = p["gdp0"]
        for y, g in zip(years, gv):
            level *= 1 + g / 100
            gdp[(gk, int(y))] = level
    P["gdp_bn"] = [gdp[(g, int(y))] / 1000 for g, y in zip(P["growth_path"], P["year"])]
    P["interest_bn"] = P["int_gdp"] / 100 * P["gdp_bn"]
    P.to_csv(RESULTS / "stress_paths.csv", index=False)

    lines = [
        f"Start {START}: debt {p['d0']:.1f}% of GDP, implicit rate {p['i0']:.2f}%, interest {p['int0']:.2f}% of GDP (EUR {p['int0'] / 100 * p['gdp0'] / 1000:.1f} bn)",
        f"Pass-through: France rho {p['rho']:.3f}; pooled {p['rho_pooled']:.3f}; alternatives 0.10 and 0.30",
        f"Yields: latest month {p['latest']} France {p['x_latest']:.2f}, Germany {p['de_latest']:.2f}; 2026 average {p['y26']:.2f} ({p['months26']} months observed, latest carried forward); 2025 average {p['x2025']:.2f}; French spread 2025 {p['fr_spread_2025']:+.2f}",
        "Italy's spread to Germany, rise over its 2010 level in 2011-2014 (pp): " + ", ".join(f"{v:+.2f}" for v in p["it_shock"]),
        f"Nominal growth: 2026 {p['g26']:.2f}% ({p['quarters26']} quarters against the same quarters of 2025); {G_BASE}% from 2027 (alternatives {G_LOW}, {G_HIGH})",
        f"Primary balance: 2026 from a deficit of {DEFICIT_2026}% of GDP; 'plan' = deficit {DEFICIT_2027}% in 2027 then +{PB_STEP} pp a year up to {PB_CAP:+.1f}%; 'held' = 2026 level",
        "",
        "Interest % of GDP (EUR bn) and debt % of GDP, France rho, base growth: 2027 / 2030 / 2036",
    ]
    base = P[(P.growth_path == "base") & (P.rho_case == "france")]
    r26 = base[(base.yields == "hold") & (base.primary_path == "plan") & (base.year == 2026)].iloc[0]
    lines.insert(-2, f"Check, 2026: modelled interest EUR {r26.interest_bn:.1f} bn ({r26.int_gdp:.2f}% of GDP), debt {r26.debt_gdp:.1f}% of GDP; 2025 interest scaled by the growth of first-half interest, 2026 over 2025 ({p['h1_ratio']:.3f}): EUR {p['int26_check']:.1f} bn")
    for (yk, pk), f in base.groupby(["yields", "primary_path"], sort=False):
        f = f.set_index("year")
        lines.append(f"  yields {yk}, primary {pk}: " + " / ".join(f"{f.loc[y, 'int_gdp']:.2f} ({f.loc[y, 'interest_bn']:.0f})" for y in (2027, 2030, 2036))
                     + f"; debt {f.loc[2030, 'debt_gdp']:.1f} / {f.loc[2036, 'debt_gdp']:.1f}; implicit rate 2030 {f.loc[2030, 'implicit']:.2f}, 2036 {f.loc[2036, 'implicit']:.2f}")
    # decomposition of the rise in interest/GDP to 2030 and 2036, plan, base growth
    def v(yk, pk="plan", gk="base", rk="france", year=2030):
        return float(P[(P.yields == yk) & (P.primary_path == pk) & (P.growth_path == gk) & (P.rho_case == rk) & (P.year == year)]["int_gdp"].iloc[0])
    for year in (2030, 2036):
        lines.append("")
        lines.append(f"Decomposition of the change in interest % of GDP, {START} to {year} (plan, base growth, France rho):")
        lines.append(f"  already locked in by 2025 yields (yields back to 2025): {v('back_to_2025', year=year) - p['int0']:+.2f}")
        lines.append(f"  rise in German yields since 2025 (French spread back to 2025): {v('spread_back', year=year) - v('back_to_2025', year=year):+.2f}")
        lines.append(f"  wider French spread (yields held): {v('hold', year=year) - v('spread_back', year=year):+.2f}")
        lines.append(f"  total with yields held: {v('hold', year=year) - p['int0']:+.2f} (to {v('hold', year=year):.2f})")
        lines.append(f"  levers, yields held: no adjustment instead of plan {v('hold', 'held', year=year) - v('hold', year=year):+.2f}; growth 1.5% instead of 2.5% {v('hold', gk='low', year=year) - v('hold', year=year):+.2f}; growth 3.5% {v('hold', gk='high', year=year) - v('hold', year=year):+.2f}; an Italian 2011 spread shock in 2027-2030 {v('italy_2011', year=year) - v('hold', year=year):+.2f}")
        lines.append(f"  pass-through sensitivity, yields held: pooled {v('hold', rk='pooled', year=year):.2f}, rho 0.10 {v('hold', rk='low', year=year):.2f}, rho 0.30 {v('hold', rk='high', year=year):.2f}")

    # Monte Carlo
    rng = np.random.default_rng(SEED)
    out = []
    for pk in ("plan", "held"):
        X = np.tile(yield_path("hold", p, years), (N_PATHS, 1))
        shocks = rng.normal(0, p["sd_dy"], X.shape)
        shocks[:, years == 2026] = 0
        X = np.maximum(X + np.cumsum(shocks, axis=1), 0)
        G = np.tile(growth["base"], (N_PATHS, 1)) + rng.normal(0, p["sd_g"], X.shape) * (years != 2026)
        IG, D = project_many(p["d0"], p["i0"], X, G, p["rho"], pk, years)
        res = np.stack([IG, D], axis=2)
        for k, y in enumerate(years):
            q = np.percentile(res[:, k, 0], [5, 25, 50, 75, 95])
            qd = np.percentile(res[:, k, 1], [5, 50, 95])
            out.append(dict(primary_path=pk, year=int(y), p5=q[0], p25=q[1], p50=q[2], p75=q[3], p95=q[4], debt_p5=qd[0], debt_p50=qd[1], debt_p95=qd[2],
                            prob_above_4=float((res[:, k, 0] > 4).mean()), prob_above_it2012=float((res[:, k, 0] > p["it_2012_int"]).mean())))
    mc = pd.DataFrame(out)
    mc.to_csv(RESULTS / "stress_mc.csv", index=False)
    lines.append("")
    lines.append(f"Monte Carlo ({N_PATHS} paths, seed {SEED}): yield shocks sd {p['sd_dy']:.2f} pp a year (random walk from the latest level), growth sd {p['sd_g']:.2f} pp around {G_BASE}%")
    for pk, f in mc.groupby("primary_path", sort=False):
        f = f.set_index("year")
        for y in (2030, 2036):
            r = f.loc[y]
            lines.append(f"  primary {pk}, {y}: interest % GDP median {r.p50:.2f}, 50% band {r.p25:.2f}-{r.p75:.2f}, 90% band {r.p5:.2f}-{r.p95:.2f}; debt median {r.debt_p50:.1f} (90% {r.debt_p5:.1f}-{r.debt_p95:.1f}); P(interest > 4% of GDP) {r.prob_above_4:.2f}; P(interest > Italy 2012, {p['it_2012_int']:.2f}%) {r.prob_above_it2012:.2f}")
    return write("stress_summary.txt", lines)


if __name__ == "__main__":
    print(run())
