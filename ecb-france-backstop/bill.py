"""The bill: France's interest burden to 2036 with and without an ECB cap on the spread.

    python bill.py

The debt dynamics are those of the France-2026 analysis in this repository (france-spread-2026,
stress.py): from the 2025 accounts the implicit interest rate moves towards the annual 10-year
yield at France's estimated pass-through rho, and

    d_t = (A (1 + i_t / 200) - pb_t) / (1 - i_t / 200),   A = d_{t-1} / (1 + g_t / 100).

The 10-year yield is split as German yield + warranted spread + unwarranted residual, with the
warranted spread taken from warranted.py (median of the headline models in the latest quarter)
and the trigger level as the spread at which half of those models would flag the excess. Yields
from 2027 (2026 uses the observed months, latest carried forward):

- held: the latest month;
- spread at warranted: German yield (latest) + warranted spread; this is the most a TPI can
  legitimately aim at, since purchases stop when tensions reflect fundamentals;
- spread zero: German yield alone, an upper bound on what any spread cap could do;
- rates back to 2025: German yield at its 2025 average + warranted spread (the part of the bill
  that moves with euro-area-wide rates, which the ECB sets for twenty countries by inflation);
- panic: latest level plus the rise in Italy's spread in 2011-2014 over its 2010 level,
  applied to 2027-2030;
- panic capped at trigger: the panic path with the spread limited to the trigger level;
- panic capped at warranted: the panic path brought back to the warranted spread.

Primary balance: 2026 from the government's deficit forecast (5.4% of GDP); the 2027 target of
5.0% and then +0.5 point of GDP a year to +1.0% ("plan"). Nominal growth 2.5% from 2027.

Writes ``results/bill_paths.csv`` and ``results/bill_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, annual, monthly_yields, quarterly_fiscal, write

START, END = 2025, 2036
DEFICIT_2026, DEFICIT_2027 = 5.4, 5.0
G_BASE = 2.5
PB_STEP, PB_CAP = 0.5, 1.0
POOL = ("FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE", "DE")
RHO_FIT = (2000, 2025)


def estimate_rho(a: pd.DataFrame, c: str, years: tuple[int, int]) -> float:
    """Pass-through of the 10-year yield to the implicit rate: Delta i = rho (x - i_{-1}) + a."""
    f = a[a["geo"] == c].set_index("year").loc[years[0]:years[1]]
    i, x = f["implicit"], f["yield"]
    di = i.diff()
    gap = (x - i.shift())
    ok = di.notna() & gap.notna()
    X = np.column_stack([gap[ok], np.ones(ok.sum())])
    beta, *_ = np.linalg.lstsq(X, di[ok].to_numpy(), rcond=None)
    return float(beta[0])


def project(d0: float, i0: float, yields: np.ndarray, growth: np.ndarray, rho: float, years: np.ndarray) -> pd.DataFrame:
    d, i = d0, i0
    rows, prev = [], None
    for y, x, g in zip(years, yields, growth):
        i = i + rho * (x - i)
        a = d / (1 + g / 100)
        if y == 2026:
            d_new = a + DEFICIT_2026
            ig = i / 100 * (a + d_new) / 2
            pb = ig - DEFICIT_2026
            d = d_new
        else:
            if y == 2027:
                pb = i / 100 * (a + a + DEFICIT_2027) / 2 - DEFICIT_2027
            else:
                pb = min(prev + PB_STEP, PB_CAP)
            d = (a * (1 + i / 200) - pb) / (1 - i / 200)
            ig = i / 100 * (a + d) / 2
        prev = pb
        rows.append(dict(year=int(y), yield_=x, implicit=i, primary=pb, debt_gdp=d, int_gdp=ig))
    return pd.DataFrame(rows).rename(columns={"yield_": "yield"})


def warranted_inputs() -> dict:
    p = pd.read_csv(RESULTS / "warranted_panel.csv.gz")
    p = p[p["window"] != "pre_crisis"]
    fr = p[p["geo"] == "FR"].dropna(subset=["spread", "warranted"])
    last = fr["quarter"].max()
    f = fr[fr["quarter"] == last]
    return dict(quarter=last, warranted=float(f["warranted"].median()), trigger=float(f["warranted"].median() + f["threshold"].median()),
                w10=float(f["warranted"].quantile(.1)), w90=float(f["warranted"].quantile(.9)))


def run() -> str:
    a = annual()
    m = monthly_yields()
    fr = a[a["geo"] == "FR"].set_index("year")
    de = a[a["geo"] == "DE"].set_index("year")
    it = a[a["geo"] == "IT"].set_index("year")
    rho = estimate_rho(a, "FR", RHO_FIT)
    latest = m.dropna(subset=["FR"]).index.max()
    x_latest, de_latest = float(m.loc[latest, "FR"]), float(m.loc[latest, "DE"])
    y26 = m.loc[m.index.year == 2026, "FR"]
    y26_avg = float((y26.sum() + y26.iloc[-1] * (12 - len(y26))) / 12)
    q = quarterly_fiscal()
    qf = q[q["geo"] == "FR"].set_index("quarter")["gdp"]
    have = [p for p in qf.index if str(p).startswith("2026") and pd.notna(qf.loc[p])]
    prev = [p.replace("2026", "2025") for p in have]
    g26 = float(100 * (qf.loc[have].sum() / qf.loc[prev].sum() - 1))
    it_spread = (it["yield"] - de["yield"]).loc[2010:2014]
    shock = (it_spread.loc[2011:2014] - it_spread.loc[2010]).to_numpy()
    w = warranted_inputs()
    d0, i0, int0, gdp0 = float(fr.loc[START, "debt_gdp"]), float(fr.loc[START, "implicit"]), float(fr.loc[START, "int_gdp"]), float(fr.loc[START, "gdp"])
    de2025 = float(de.loc[2025, "yield"])

    years = np.arange(START + 1, END + 1)
    growth = np.where(years == 2026, g26, G_BASE)
    later = years >= 2027

    def path(spread: np.ndarray | float, bund: float = de_latest) -> np.ndarray:
        x = np.full(len(years), y26_avg)
        x[later] = bund + (spread[later] if isinstance(spread, np.ndarray) else spread)
        return x

    s_latest = x_latest - de_latest
    panic = np.full(len(years), s_latest)
    for k, s in enumerate(shock):
        panic[years == 2027 + k] += s
    scen = {
        "held": path(s_latest),
        "spread_at_warranted": path(w["warranted"]),
        "spread_zero": path(0.0),
        "rates_back_to_2025": path(w["warranted"], bund=de2025),
        "panic": path(panic),
        "panic_capped_at_trigger": path(np.minimum(panic, w["trigger"])),
        "panic_capped_at_warranted": path(np.minimum(panic, w["warranted"])),
    }
    paths = []
    gdp = gdp0 * np.cumprod(1 + growth / 100)
    for k, x in scen.items():
        for rk, r in (("france", rho), ("low", 0.10), ("high", 0.30)):
            f = project(d0, i0, x, growth, r, years)
            f["scenario"], f["rho_case"] = k, rk
            f["interest_bn"] = f["int_gdp"] / 100 * gdp / 1000
            paths.append(f)
    P = pd.concat(paths, ignore_index=True)
    P.to_csv(RESULTS / "bill_paths.csv", index=False, float_format="%.4f")

    def v(s: str, y: int, col: str = "int_gdp", rk: str = "france") -> float:
        return float(P[(P.scenario == s) & (P.year == y) & (P.rho_case == rk)][col].iloc[0])

    lines = [
        f"Start {START}: debt {d0:.1f}% of GDP, implicit rate {i0:.2f}%, interest {int0:.2f}% of GDP (EUR {int0 / 100 * gdp0 / 1000:.1f} bn); pass-through rho {rho:.3f} (France, {RHO_FIT[0]}-{RHO_FIT[1]})",
        f"Yields: latest {latest} France {x_latest:.2f}, Germany {de_latest:.2f}, spread {s_latest:.2f}; 2026 average {y26_avg:.2f}; German 2025 average {de2025:.2f}",
        f"Warranted spread ({w['quarter']}, headline models): median {w['warranted']:.2f} (10-90% {w['w10']:.2f}-{w['w90']:.2f}); trigger level (half of models flag) {w['trigger']:.2f}",
        "Italy 2011-2014 spread rise over 2010 (pp): " + ", ".join(f"{s:+.2f}" for s in shock),
        f"Nominal growth 2026 {g26:.2f}% ({len(have)} quarters), then {G_BASE}%; primary balance: deficit {DEFICIT_2026}% (2026), {DEFICIT_2027}% (2027), then +{PB_STEP} pp a year to {PB_CAP:+.1f}%",
        "",
        "Interest % of GDP (EUR bn) 2027 / 2030 / 2036; debt % of GDP 2030 / 2036; France rho",
    ]
    for k in scen:
        lines.append(f"  {k}: " + " / ".join(f"{v(k, y):.2f} ({v(k, y, 'interest_bn'):.0f})" for y in (2027, 2030, 2036))
                     + f"; debt {v(k, 2030, 'debt_gdp'):.1f} / {v(k, 2036, 'debt_gdp'):.1f}")
    for y in (2030, 2036):
        lines += ["", f"Rise in interest % of GDP, {START} to {y} ({int0:.2f} -> {v('held', y):.2f}), and what each lever changes:"]
        lines.append(f"  total with yields held: {v('held', y) - int0:+.2f}")
        lines.append(f"  of which the whole French spread over Germany (held minus spread zero): {v('held', y) - v('spread_zero', y):+.2f}")
        lines.append(f"  of which the part above the warranted spread (held minus spread at warranted): {v('held', y) - v('spread_at_warranted', y):+.2f}")
        lines.append(f"  euro-area rates back to 2025 (spread at warranted): {v('rates_back_to_2025', y) - v('spread_at_warranted', y):+.2f}")
        lines.append(f"  already locked in, at warranted spread and 2025 German yield: {v('rates_back_to_2025', y) - int0:+.2f}")
        lines.append(f"  a panic like Italy's in 2011-2014: {v('panic', y) - v('held', y):+.2f}; capped at the trigger {v('panic_capped_at_trigger', y) - v('held', y):+.2f}; capped at warranted {v('panic_capped_at_warranted', y) - v('held', y):+.2f}")
        lines.append(f"  EUR bn, {y}: held {v('held', y, 'interest_bn'):.0f}; spread zero {v('spread_zero', y, 'interest_bn'):.0f}; panic {v('panic', y, 'interest_bn'):.0f}; panic capped at trigger {v('panic_capped_at_trigger', y, 'interest_bn'):.0f}; capped at warranted {v('panic_capped_at_warranted', y, 'interest_bn'):.0f}")
        lines.append(f"  pass-through sensitivity (held): rho 0.10 {v('held', y, rk='low'):.2f}, rho 0.30 {v('held', y, rk='high'):.2f}; panic minus held at rho 0.30 {v('panic', y, rk='high') - v('held', y, rk='high'):+.2f}")
    cum = P[(P.rho_case == "france") & (P.year >= 2027)].groupby("scenario")["interest_bn"].sum()
    lines += ["", "Cumulative interest 2027-2036, EUR bn: " + "; ".join(f"{k} {cum[k]:.0f}" for k in scen)]
    lines.append(f"  panic avoided by a cap at the trigger: {cum['panic'] - cum['panic_capped_at_trigger']:.0f}; at warranted: {cum['panic'] - cum['panic_capped_at_warranted']:.0f}; whole spread (held minus zero): {cum['held'] - cum['spread_zero']:.0f}")
    return write("bill_summary.txt", lines)


if __name__ == "__main__":
    print(run())
