"""Early repayments to the European lenders: the EFSF in 2019 and the EFSM in 2022-2025.

    python european.py

The European loans were cheaper than the IMF's, so the arithmetic is closer. For each repayment the
saving at par is (loan rate - replacement yield) x amount x years to the loan's due date, where the
replacement is a Portuguese bond of the same residual maturity issued in the month of the
repayment (BPstat yields; ``common.pt_curve``). Only average loan rates are public: about 1.7-1.8%
for the EFSF in 2019 and about 2% for the EFSM in 2025, as reported from IGCP data
(``documented.csv``); the rates of the individual tranches repaid are not. So the script reports,
for each repayment, the saving at the published average rate and the break-even loan rate (the
replacement yield): the repayment saves money at par only if the tranche carried more than that.

From 2026 Portugal's EFSM repayments are to be made "on net present value terms at the prevailing
EFSM market funding rates": Portugal pays the market value of the loan, not its face value. The
coupon difference then disappears from the saving, which shrinks to the gap between Portugal's
own funding rate and the EU's for the same maturity.

Writes ``results/european.csv`` and ``results/european_summary.txt``.
"""

from __future__ import annotations

import pandas as pd

from common import RESULTS, pt_curve, pt_yield, write

# repayment month, lender, amount (EUR bn), due date(s) of the tranches repaid with amounts, published average rate
REPAYMENTS = [
    ("2019-10", "EFSF", [(1.0, "2025-08"), (1.0, "2025-12")], (1.7, 1.8), "EUR 2 bn: the tranche due August 2025 in full, part of one due December 2025 (amounts split equally here)"),
    ("2023-11", "EFSM", [(1.5, None)], (2.0, 2.0), "EUR 1.5 bn; tranche not published"),
    ("2025-12", "EFSM", [(1.8, "2028-12"), (0.7, "2031-04")], (2.0, 2.0), "EUR 1.8 bn of the loan due 5 December 2028 and EUR 0.7 bn of the loan due 22 April 2031"),
]
EFSF_GOVERNMENT_ESTIMATE = 120  # EUR million, cumulative interest saving announced for the 2019 repayment
HORIZONS_UNKNOWN = (3.0, 5.0, 10.0)  # years tried when the tranche is not published


def run() -> str:
    curve = pt_curve()
    rows, lines = [], []
    lines.append("Early repayments to the European lenders: saving at par = (loan rate - matched Portuguese yield) x amount x years")
    for month, lender, parts, (r_lo, r_hi), note in REPAYMENTS:
        m = pd.Period(month, "M")
        lines.append(f"  {month} {lender}: {note}")
        for amount, due in parts:
            if due is None:
                for h in HORIZONS_UNKNOWN:
                    y = pt_yield(curve, m, h)
                    save_lo, save_hi = (r_lo - y) / 100 * amount * h * 1000, (r_hi - y) / 100 * amount * h * 1000
                    rows.append(dict(month=month, lender=lender, amount=amount, due="unknown", years=h, yield_matched=y, rate_low=r_lo, rate_high=r_hi,
                                     saving_low=save_lo, saving_high=save_hi))
                    lines.append(f"    if {h:.0f} years to due: matched yield {y:.2f}% (break-even loan rate); at {r_lo}% the saving is {save_lo:+,.0f} m")
                continue
            years = (pd.Period(due, "M") - m).n / 12
            y = pt_yield(curve, m, years)
            save_lo, save_hi = (r_lo - y) / 100 * amount * years * 1000, (r_hi - y) / 100 * amount * years * 1000
            rows.append(dict(month=month, lender=lender, amount=amount, due=due, years=years, yield_matched=y, rate_low=r_lo, rate_high=r_hi,
                             saving_low=save_lo, saving_high=save_hi))
            lines.append(f"    EUR {amount:.1f} bn due {due} ({years:.1f} years): matched yield {y:.2f}% (break-even); saving at {r_lo}%: {save_lo:+,.0f} m"
                         + (f", at {r_hi}%: {save_hi:+,.0f} m" if r_hi != r_lo else ""))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "european.csv", index=False, float_format="%.4f")

    efsf = df[df.lender == "EFSF"]
    implied = (EFSF_GOVERNMENT_ESTIMATE / 1000 / (efsf.amount * efsf.years).sum() * 100) + (efsf.yield_matched * efsf.amount * efsf.years).sum() / (efsf.amount * efsf.years).sum()
    lines += ["", f"EFSF 2019: saving at the published average rates {efsf.saving_low.sum():,.0f}-{efsf.saving_high.sum():,.0f} m against the government's estimate of about {EFSF_GOVERNMENT_ESTIMATE} m;",
              f"  the government figure is consistent with a rate on the tranches repaid of about {implied:.2f}% (or with a costlier replacement)",
              f"  replacing with a ten-year bond instead: yield {pt_yield(curve, pd.Period('2019-10', 'M'), 10):.2f}%"]
    d25 = df[(df.lender == "EFSM") & (df.month == "2025-12")]
    lines += ["", f"EFSM December 2025: break-even rates {', '.join(f'{r.yield_matched:.2f}% ({r.due})' for r in d25.itertuples())}; at the reported average of about 2% the "
              f"saving at par is {d25.saving_low.sum():+,.0f} m, i.e. a cost unless the tranches carried more than the break-even"]
    for months, label in (("2023-11", "November 2023"), ("2025-12", "December 2025"), ("2026-10", "October 2026")):
        mm = pd.Period(months, "M")
        if mm in curve.index:
            lines.append(f"  Portuguese yields, {label}: " + ", ".join(f"{k:g}y {curve.loc[mm, k]:.2f}%" for k in (2.0, 3.0, 5.0, 7.0, 10.0)))
    return write("european_summary.txt", lines)


if __name__ == "__main__":
    print(run())
