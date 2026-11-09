"""Bond exchanges and buybacks: where the "interest saved" goes.

    python exchanges.py

The Treasury agency (IGCP) bought back high-coupon bonds close to maturity, paying cash or new
bonds that mature later. Bought at market prices, a bond with a coupon above the market yield
costs more than its face value. The interest that stops being paid on the old bond is therefore
matched, in present value, by the premium paid for it; in the government accounts the interest
line falls and the debt (measured at face value) rises. What changes for real is the redemption
profile, and, for cash buybacks, the return given up on the cash used.

For each operation documented in IGCP releases and press reports (``documented.csv``; amounts per
line, coupons and maturities as published), the bought bond is priced at the Portuguese yield for
its residual maturity in that month (BPstat; annual coupons, no accrued interest) and the bonds
sold at the yield for their maturity (or the published yield). Reported per operation:

- premium paid: (price - 100) x nominal, the extra face value of debt needed to pay for it;
- interest no longer paid to the old maturity: (coupon - yield of the bonds sold) x nominal x
  years, the "saving" in the accounts (for cash buybacks, against the deposit facility rate as the
  return on the cash used);
- the difference, which is the term-structure part (positive when the bonds sold yield less than
  the bond bought would have, at its own maturity);
- years by which the redemption was pushed back.

Writes ``results/exchanges.csv`` and ``results/exchanges_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, bp_series, pt_curve, pt_yield, write

# date, kind, bought: [(nominal EUR m, coupon %, maturity)], sold: [(maturity, yield % or None)], note
OPERATIONS = [
    ("2017-08-30", "exchange", [(426.7, 4.45, "2018-06-15"), (401.0, 4.75, "2019-06-14"), (908.0, 4.80, "2020-06-15")], [("2022-10-17", None)],
     "EUR 1,735.7 m bought (2018 line implied by the total); EUR 1,793 m of 2022 bonds sold"),
    ("2018-02-28", "buyback", [(150.0, 4.75, "2019-06-14"), (100.0, 3.85, "2021-04-15")], [], "February 2018 buybacks"),
    ("2018-10-31", "buyback", [(428.0, 4.75, "2019-06-14")], [], "October 2018 buyback of the June 2019 line"),
    ("2019-01-30", "exchange", [(702.0, 4.80, "2020-06-15")], [("2028-10-17", None)], "June 2020 line into October 2028"),
    ("2019-05-22", "exchange", [(742.0, 3.85, "2021-04-15")], [("2026-07-21", None)], "April 2021 line into 2026"),
    ("2019-07-24", "exchange", [(256.0, 4.80, "2020-06-15"), (541.0, 3.85, "2021-04-15")], [("2026-07-21", None), ("2028-10-17", None)],
     "2020 and 2021 lines into 2026 and 2028 (split between them not published; equal weights)"),
    ("2022-07-06", "exchange", [(538.0, 4.95, "2023-10-25"), (225.0, 5.65, "2024-02-15")], [("2028-10-17", 1.64), ("2035-10-12", None)],
     "EUR 531 m of OT 2.125% Oct 2028 sold at 102.88 (yield 1.64%) and EUR 232 m of OT 0.9% Oct 2035"),
    ("2022-11-30", "buyback", [(700.0, 4.95, "2023-10-25"), (700.0, 5.65, "2024-02-15")], [],
     "EUR 1,400 m across the two lines (split not published; equal halves)"),
]


def price(coupon: float, y: float, years: float) -> float:
    """Clean price of an annual-coupon bond with `years` to maturity (fractional first period)."""
    n = int(np.ceil(years - 1e-9))
    times = years - np.arange(n)[::-1]  # time to each remaining coupon date
    times = times[times > 0]
    disc = (1 + y / 100) ** -times
    accrued = coupon * (1 - (years - np.floor(years))) if years % 1 else 0.0
    return float((coupon * disc).sum() + 100 * disc[-1] - accrued)


def run() -> str:
    curve = pt_curve()
    mro = bp_series({12504591: "dfr"})["dfr"]  # ECB deposit facility rate (BPstat)
    mro.index = pd.PeriodIndex(mro.index, freq="M")
    mro = mro.groupby(level=0).last()
    rows = []
    for date, kind, bought, sold, note in OPERATIONS:
        d = pd.Timestamp(date)
        m = pd.Period(d, "M")
        if sold:
            y_new = np.mean([y if y is not None else pt_yield(curve, m, (pd.Timestamp(t) - d).days / 365.25) for t, y in sold])
            new_years = np.mean([(pd.Timestamp(t) - d).days / 365.25 for t, _ in sold])
        else:
            y_new = float(mro.loc[:m].iloc[-1])
            new_years = 0.0
        for nominal, cpn, mat in bought:
            years = (pd.Timestamp(mat) - d).days / 365.25
            y_old = pt_yield(curve, m, years)
            p_old = price(cpn, y_old, years)
            premium = (p_old - 100) / 100 * nominal
            saved = (cpn - y_new * p_old / 100) / 100 * nominal * years
            rows.append(dict(date=date, kind=kind, nominal=nominal, coupon=cpn, maturity=mat, years_left=years, yield_old=y_old, price_old=p_old,
                             premium=premium, yield_new=y_new, interest_saved=saved, term_part=saved - premium,
                             years_deferred=max(0.0, new_years - years) if kind == "exchange" else 0.0, note=note))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exchanges.csv", index=False, float_format="%.4f")

    lines = ["Documented exchanges and buybacks: nominal bought, the price paid, and where the interest 'saving' goes (EUR million)"]
    for (date, kind), f in df.groupby(["date", "kind"], sort=False):
        lines.append(f"  {date} {kind}: nominal {f.nominal.sum():,.0f}; coupons {', '.join(f'{c:.2f}% {mt[:7]}' for c, mt in zip(f.coupon, f.maturity))}; "
                     f"price {', '.join(f'{p:.2f}' for p in f.price_old)} (yields {', '.join(f'{y:.2f}' for y in f.yield_old)}%); "
                     f"premium {f.premium.sum():,.1f}; interest no longer paid to the old maturity {f.interest_saved.sum():,.1f} "
                     f"against {'new bonds at' if kind == 'exchange' else 'the deposit rate of'} {f.yield_new.iloc[0]:.2f}%; difference {f.term_part.sum():+,.1f}"
                     + (f"; redemption pushed back {np.average(f.years_deferred, weights=f.nominal):.1f} years" if kind == "exchange" else ""))
    t = df.sum(numeric_only=True)
    lines += ["", f"All documented operations: nominal {t.nominal:,.0f}; premium paid {t.premium:,.0f}; interest no longer paid {t.interest_saved:,.0f}; "
                  f"difference {t.term_part:+,.0f} ({t.term_part / t.interest_saved:.0%} of the interest line)"]
    ex = df[df.kind == "exchange"]
    lines.append(f"  exchanges only: nominal {ex.nominal.sum():,.0f}; premium {ex.premium.sum():,.0f}; interest no longer paid {ex.interest_saved.sum():,.0f}; "
                 f"redemptions pushed back {np.average(ex.years_deferred, weights=ex.nominal):.1f} years on average")
    by_year = df.assign(y=df.maturity.str[:4]).groupby("y").nominal.sum()
    lines.append("  nominal removed from each redemption year: " + ", ".join(f"{y} {v:,.0f}" for y, v in by_year.items()))
    return write("exchanges_summary.txt", lines)


if __name__ == "__main__":
    print(run())
