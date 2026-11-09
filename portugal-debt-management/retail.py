"""Savings certificates (Certificados de Aforro): what the retail wave of 2022-2026 cost.

    python retail.py

The stock of savings certificates rose from about EUR 12 billion at the end of 2021 to EUR 45
billion by September 2026 (BPstat, direct State debt). Series E, sold until 2 June 2023, paid a base
rate of three-month Euribor plus one point, between 0% and 3.5%, plus a loyalty premium of 0.5
points from the second to the fifth year. Series F, sold from 5 June 2023, pays three-month
Euribor between 0% and 2.5%, plus 0.25 points from the second year (rules in ``documented.csv``).

Monthly net increases in the stock are taken as new subscriptions: from January 2022 to May 2023
they are series E, from June 2023 series F (net flows include capitalised interest and
redemptions, so the vintages are approximate). Each vintage is then paid its series' rate each
month, with three-month Euribor of the previous month, and compared with what the same money would
have cost in the market:

- rolled at twelve-month Euribor (a short-term proxy, close to Treasury bill yields);
- fixed at the five-year Portuguese yield of the month of subscription (BPstat).

A savings certificate is closer to the first: the holder can redeem it at any time after three
months. Interest on savings certificates is taxed at source at 28%, which the State collects back;
a net-of-tax line applies that to the certificates only (market debt held by non-residents pays no
Portuguese tax, so this is the most favourable case for the certificates). A counterfactual also prices the series F vintages at the series E formula, to measure
what closing series E saved, before any change in demand.

Writes ``results/retail_*.csv`` and ``results/retail_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, bp_series, euribor, pt_curve, write

E_START, E_END, F_START = pd.Period("2022-01", "M"), pd.Period("2023-05", "M"), pd.Period("2023-06", "M")
WITHHOLDING = 0.28  # personal income tax withheld at source on savings-certificate interest


def e_rate(e3m: float, age: int) -> float:
    return min(3.5, max(0.0, e3m + 1.0)) + (0.5 if 12 <= age < 60 else 1.0 if age >= 60 else 0.0)


def f_rate(e3m: float, age: int) -> float:
    prem = 0.0 if age < 12 else 0.25 if age < 60 else 0.5 if age < 108 else 1.0
    return min(2.5, max(0.0, e3m)) + prem


def run() -> str:
    w = bp_series({12561438: "ca", 12561434: "ct"})
    w.index = pd.PeriodIndex(w.index, freq="M")
    w = w.groupby(level=0).last()
    e3, e12 = euribor("3m"), euribor("12m")
    curve = pt_curve()
    last = w["ca"].dropna().index.max()
    flows = w["ca"].diff()
    vint = flows.loc[E_START:last].clip(lower=0.0)
    series = pd.Series(np.where(vint.index <= E_END, "E", "F"), index=vint.index)
    months = pd.period_range(E_START, last, freq="M")
    rows = []
    for t in months:
        e_prev = float(e3.loc[:t - 1].iloc[-1])
        m12 = float(e12.loc[:t].iloc[-1])
        for v, amt in vint.loc[:t].items():
            if amt <= 0:
                continue
            age = (t - v).n
            s = series[v]
            paid = e_rate(e_prev, age) if s == "E" else f_rate(e_prev, age)
            alt_e = e_rate(e_prev, age)
            y5 = float(curve.loc[v, 5.0])
            rows.append(dict(month=t, vintage=v, series=s, amount=amt, rate=paid, euribor12=m12, pt5_at_issue=y5,
                             cost=paid / 1200 * amt, cost_short=m12 / 1200 * amt, cost_fixed=y5 / 1200 * amt,
                             cost_if_e=alt_e / 1200 * amt))
    v = pd.DataFrame(rows)
    m = v.groupby(["month", "series"]).agg(amount=("amount", "sum"), cost=("cost", "sum"), cost_short=("cost_short", "sum"),
                                         cost_fixed=("cost_fixed", "sum"), cost_if_e=("cost_if_e", "sum")).reset_index()
    m["rate"] = m.cost * 1200 / m.amount
    out = m.assign(month=m.month.astype(str))
    out.to_csv(RESULTS / "retail_monthly.csv", index=False, float_format="%.4f")
    rates = pd.DataFrame({"euribor3m": e3, "euribor12m": e12, "pt2y": curve[2.0], "pt5y": curve[5.0]}).loc["2021-01":last]
    rates["series_e_new"] = [min(3.5, max(0.0, x + 1.0)) for x in e3.shift(1).loc["2021-01":last]]
    rates["series_f_new"] = [min(2.5, max(0.0, x)) for x in e3.shift(1).loc["2021-01":last]]
    rates.reset_index(names="month").assign(month=lambda d: d.month.astype(str)).to_csv(RESULTS / "retail_rates.csv", index=False, float_format="%.4f")
    stock = w.loc["2010-01":last].reset_index(names="month").assign(month=lambda d: d.month.astype(str))
    stock.to_csv(RESULTS / "retail_stock.csv", index=False, float_format="%.1f")

    lines = [f"Savings certificates: stock EUR {w.loc['2021-12', 'ca'] / 1000:.1f} bn at end-2021, {w.loc['2023-05', 'ca'] / 1000:.1f} bn in May 2023, "
             f"{w.loc[last, 'ca'] / 1000:.1f} bn in {last}; Treasury certificates {w.loc['2021-12', 'ct'] / 1000:.1f} bn -> {w.loc[last, 'ct'] / 1000:.1f} bn",
             f"  net inflows taken as series E (Jan 2022 - May 2023): EUR {vint.loc[:E_END].sum() / 1000:.1f} bn; series F (Jun 2023 - {last}): EUR {vint.loc[F_START:].sum() / 1000:.1f} bn",
             f"  largest monthly inflow {flows.loc[E_START:].max() / 1000:.2f} bn in {flows.loc[E_START:].idxmax()}; inflow in the month series E closed (May 2023) {flows.loc['2023-05'] / 1000:.2f} bn, "
             f"June 2023 {flows.loc['2023-06'] / 1000:.2f} bn, July 2023 {flows.loc['2023-07'] / 1000:.2f} bn", ""]
    lines.append("Rates paid on new subscriptions and market rates (%), selected months: series E formula; series F formula; 12-month Euribor; Portuguese 2-year; 5-year")
    for mm in ("2022-06", "2022-12", "2023-05", "2023-12", "2024-06", "2024-12", "2025-06", "2025-12", str(last)):
        r = rates.loc[mm]
        lines.append(f"  {mm}: E {r.series_e_new:.2f}; F {r.series_f_new:.2f}; Euribor 12m {r.euribor12m:.2f}; PT 2y {r.pt2y:.2f}; PT 5y {r.pt5y:.2f}")
    lines += ["", "Cost of the 2022-2026 vintages against market funding (EUR million, cumulative to the last month; positive = savings certificates cost more)"]
    for s_, f in m.groupby("series"):
        lines.append(f"  series {s_}: paid {f.cost.sum():,.0f}; rolled at 12-month Euribor {f.cost_short.sum():,.0f} ({f.cost.sum() - f.cost_short.sum():+,.0f}); "
                     f"fixed at the 5-year yield {f.cost_fixed.sum():,.0f} ({f.cost.sum() - f.cost_fixed.sum():+,.0f})")
    for y in (2022, 2023, 2024, 2025, 2026):
        f = m[m.month.map(lambda p: p.year) == y]
        if f.empty:
            continue
        lines.append(f"  {y}: paid {f.cost.sum():,.0f}; against 12-month Euribor {f.cost.sum() - f.cost_short.sum():+,.0f}; "
                     f"against the 5-year yield {f.cost.sum() - f.cost_fixed.sum():+,.0f}; average rate paid {f.cost.sum() * 1200 / f.amount.sum():.2f}% (12-month Euribor {f.cost_short.sum() * 1200 / f.amount.sum():.2f}%)")
    lines.append("Net of the 28% tax withheld on certificate interest (market alternatives untaxed): " + "; ".join(
        f"series {s_}: {f.cost.sum() * (1 - WITHHOLDING):,.0f} against 12-month Euribor {f.cost.sum() * (1 - WITHHOLDING) - f.cost_short.sum():+,.0f}, "
        f"against the 5-year yield {f.cost.sum() * (1 - WITHHOLDING) - f.cost_fixed.sum():+,.0f}" for s_, f in m.groupby("series")))
    fv = m[m.series == "F"]
    lines += ["", f"Closing series E: the series F vintages priced at the series E formula would have cost {fv.cost_if_e.sum():,.0f} instead of {fv.cost.sum():,.0f} "
                  f"(EUR {fv.cost_if_e.sum() - fv.cost.sum():,.0f} m less, before any change in demand)"]
    return write("retail_summary.txt", lines)


if __name__ == "__main__":
    print(run())
