"""Every number in the article.

python mortgage_numbers.py > output.txt

Also writes the tables behind the figures to results/.
"""

from pathlib import Path

import pandas as pd

import mortgage_core as mc

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def head(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def eur(x):
    return f"{x:,.0f} euros"


def summary(r):
    """Share of complete cohorts where variable was cheaper, mean extra cost of fixing, and payment rises."""
    full = r[r.complete]
    late = r[(r.index >= "2019-01") & (r.index <= "2021-12")]
    return (
        f"variable cheaper {100 * (full.extra_fixed > 0).mean():5.1f}%, mean extra cost of fixing"
        f" {mc.mean(full.extra_fixed):7,.0f}; largest rise {100 * r.rise.max():5.1f}%,"
        f" mean rise 2019-2021 starts {100 * mc.mean(late.rise):5.1f}%"
    )


def main():
    RES.mkdir(exist_ok=True)
    m = mc.load()
    loans = mc.Loans(m)
    income = m.income()
    r = mc.all_cohorts(loans, income=income, hicp=m.hicp)
    e12, par10 = loans.euribor, loans.par

    # ------------------------------------------------------------------------------------------------ 1
    head("1. Data and conventions")
    print(f"loans started every month from {m.months[0]} to {m.months[-1]}: {len(m.months)}")
    print(f"loan: {eur(mc.PRINCIPAL)} over {mc.TERM // 12} years, equal monthly instalments")
    print(f"variable: 12-month Euribor (floored at 0) + {mc.MARGIN:.1f} points, reset every 12 months")
    print(
        f"fixed: 10-year par yield of AAA euro-area government bonds (floored at 0) + {mc.MARGIN:.1f} points, ten years"
    )
    print(
        f"compared over the first {mc.HORIZON} months,"
        f" or to {mc.LAST} for loans started after {r[r.complete].index[-1]}"
    )
    print("the replay's rates against the ECB's published euro-area averages for new housing loans:")
    for month, kind, published, label in mc.ECB_CHECKS:
        replayed = (mc.floored(par10[month]) if kind == "fixed" else mc.floored(e12[month])) + mc.MARGIN
        print(f"  {month} {kind:8s} ({label}): published {published:.2f}%, replay {replayed:.2f}%")

    # ------------------------------------------------------------------------------------------------ 2
    head("2. Rates, 1999-2026")
    e, p = e12.loc[mc.FIRST : mc.LAST], par10.loc[mc.FIRST : mc.LAST]
    print(f"12-month Euribor: {e.iloc[0]:.2f}% in {e.index[0]}; highest {e.max():.2f}% ({e.idxmax()});")
    print(f"  lowest {e.min():.2f}% ({e.idxmin()}); {mc.LAST}: {e.iloc[-1]:.2f}%; below zero in {(e < 0).sum()} months")
    print(f"  first month below zero {e[e < 0].index[0]}, last {e[e < 0].index[-1]}")
    print(f"  2022-01 {e['2022-01']:.2f}%, 2023-10 {e['2023-10']:.2f}%")
    print(f"10-year AAA par yield: {p.iloc[0]:.2f}% in {p.index[0]}; highest {p.max():.2f}% ({p.idxmax()});")
    print(f"  lowest {p.min():.2f}% ({p.idxmin()}); {mc.LAST}: {p.iloc[-1]:.2f}%; below zero in {(p < 0).sum()} months")
    pd.DataFrame({"euribor12": e, "par10": p}).round(4).to_csv(RES / "rates.csv", index_label="month")

    # ------------------------------------------------------------------------------------------------ 3
    head("3. The ten-year bill: loans started 1999-01 to the last month with ten years of data")
    full = r[r.complete]
    x = full.extra_fixed
    print(f"start months: {len(full)} ({full.index[0]} to {full.index[-1]})")
    print(f"variable cheaper over ten years in {(x > 0).sum()} ({100 * (x > 0).mean():.1f}%)")
    print(
        f"extra interest paid by fixing: mean {eur(mc.mean(x))}, median {eur(x.median())},"
        f" quartiles {eur(x.quantile(0.25))} and {eur(x.quantile(0.75))}"
    )
    print(f"  as a share of the variable loan's interest: mean {100 * mc.mean(x / full.interest_var):.1f}%")
    print(f"largest gain from going variable: {eur(x.max())} ({x.idxmax()})")
    print(f"start months where fixing paid: {', '.join(x[x < 0].index[:3])} ... {', '.join(x[x < 0].index[-3:])}")
    print(f"  largest gain from fixing among them: {eur(-x.min())} ({x.idxmin()})")
    print("by year of start (mean over the start months; partial horizons after 2016-11):")
    print("  year  extra cost of fixing  rate variable  rate fixed  biggest payment rise  months")
    for y, g in r.groupby(r.index.str[:4]):
        print(
            f"  {y}  {mc.mean(g.extra_fixed):12,.0f}  {mc.mean(g.rate_var):12.2f}%  {mc.mean(g.rate_fix):9.2f}%"
            f"  {100 * mc.mean(g.rise):12.1f}%  {g.months.min():4d}-{g.months.max():3d}"
        )
    r.round(5).to_csv(RES / "cohorts.csv", index_label="start")

    # ------------------------------------------------------------------------------------------------ 4
    head("4. Why variable won: what the yield curve expected and what happened")
    gap = full.par_start - full.euribor_avg
    print(
        f"10-year par yield at the start minus the average 12-month Euribor over the next ten years:"
        f" mean {mc.mean(gap):.2f} points; positive in {(gap > 0).sum()} of {len(gap)} start months"
    )
    errors = mc.forecast_errors(m)
    print(f"forecast errors of the yield curve, start months {errors.index[0]} to {errors.index[-1]} ({len(errors)}):")
    print("  years ahead: mean error (realised minus implied), share of start months where Euribor came in lower")
    for h in range(1, 10):
        col = errors[f"h{h}"]
        print(f"  {h}: {mc.mean(col):6.2f} points, {100 * (col < 0).mean():5.1f}%")
    errors.round(4).to_csv(RES / "forecast_errors.csv", index_label="start")

    # ------------------------------------------------------------------------------------------------ 5
    head("5. The payment shock")
    print(f"largest rise in the variable payment, any start month: {100 * r.rise.max():.1f}% ({r.rise.idxmax()})")
    w = r.loc[r.rise.idxmax()]
    print(
        f"  from {w.pay_var0:.0f} to {w.pay_var_max:.0f} euros a month per 100,000 borrowed, in {w.peak_month};"
        f" payment-to-income from {100 * mc.START_SHARE:.0f}% to {100 * w.share_var_max:.1f}%"
    )
    print(f"  extra paid in the worst twelve months: {eur(w.extra_peak_year)}")
    for a, b in [("2003-01", "2005-12"), ("2015-01", "2018-12"), ("2019-01", "2021-12"), ("2022-01", "2022-06")]:
        g = r.loc[a:b]
        print(
            f"starts {a} to {b}: mean rise {100 * mc.mean(g.rise):.1f}%, largest {100 * g.rise.max():.1f}%;"
            f" peak months {g.peak_month.min()} to {g.peak_month.max()};"
            f" payment-to-income at worst, mean {100 * mc.mean(g.share_var_max):.1f}%"
        )
    print(f"start months with a rise above 25%: {(r.rise > 0.25).sum()}; above 50%: {(r.rise > 0.5).sum()}")
    wage, hicp = m.wage, m.hicp
    prices = mc.mean(hicp.loc["2023-01":"2023-12"]) / mc.mean(hicp.loc["2021-01":"2021-12"]) - 1
    print(
        f"2021 to 2023: compensation per employee {100 * (wage['2023'] / wage['2021'] - 1):+.1f}%,"
        f" consumer prices (annual averages) {100 * prices:+.1f}%"
    )
    s = r.share_var_max.dropna()
    print(
        f"start months where payment-to-income passed 40%:"
        f" {(s > 0.40).sum()} ({s[s > 0.40].index[0]} to {s[s > 0.40].index[-1]})"
    )
    print(
        f"fixed loans started 2019-2021: payment-to-income at the end of 2025, mean"
        f" {100 * mc.mean(r.loc['2019-01':'2021-12'].share_fix_end):.1f}%"
    )
    print(
        f"fixed loans started 2019-2021: the payment in {mc.LAST}, in prices of the start month,"
        f" is {100 * mc.mean(r.loc['2019-01':'2021-12'].real_fix_end):.1f}% of the first payment"
    )
    paths = {}
    for start in ["2005-01", "2015-01", "2021-10"]:
        n = min(mc.HORIZON, loans.available(start))
        months = loans.months[loans.months.index(start) : loans.months.index(start) + n]
        paths[f"{start}|variable"] = pd.Series(loans.variable(start, n).payment.to_numpy(), index=months)
        paths[f"{start}|fixed"] = pd.Series(loans.fixed(start, n).payment.to_numpy(), index=months)
        v = paths[f"{start}|variable"]
        print(
            f"  start {start}: variable {v.iloc[0]:.0f} -> highest {v.max():.0f} ({v.idxmax()});"
            f" fixed {paths[f'{start}|fixed'].iloc[0]:.0f}"
        )
    pd.DataFrame(paths).round(3).to_csv(RES / "payment_paths.csv", index_label="month")

    # ------------------------------------------------------------------------------------------------ 6
    head("6. Premium and payout")
    c = r.loc[: full.index[-1]]
    print(
        f"complete horizons: the premium (extra interest paid by fixing) averaged {eur(mc.mean(c.extra_fixed))}"
        f" over ten years, {eur(mc.mean(c.extra_fixed) / 10)} a year"
    )
    shock = r.loc["2015-01":"2022-06"]
    print(
        f"starts 2015-01 to 2022-06 (horizons to {mc.LAST}):"
        f" fixing saved {eur(-mc.mean(shock.extra_fixed))} on average;"
        f" saved money in {(shock.extra_fixed < 0).sum()} of {len(shock)} start months"
    )
    print(
        f"  the worst twelve months cost a variable borrower {eur(mc.mean(shock.extra_peak_year))}"
        " more than the first payment, on average"
    )
    calm = full[full.rise < 0.10]
    print(
        f"complete horizons with a payment rise under 10%: {len(calm)} of {len(full)};"
        f" fixing cost {eur(mc.mean(calm.extra_fixed))} on average"
    )
    hit = r[r.rise >= 0.25]
    print(
        f"start months with a rise of 25% or more: {len(hit)};"
        f" fixing cost {eur(mc.mean(hit.extra_fixed))} on average (negative = saved)"
    )

    # ------------------------------------------------------------------------------------------------ 7
    head("7. Checks: other settings")
    variants = [
        ("base case", {}),
        ("fixed loans 0.25 points dearer", {"margin_fix": mc.MARGIN + 0.25}),
        ("fixed loans 0.50 points dearer", {"margin_fix": mc.MARGIN + 0.5}),
        ("margins of 1.5 points", {"margin_var": 1.5, "margin_fix": 1.5}),
        ("6-month Euribor, reset every 6 months", {"index": "IRT_M6", "reset": 6}),
        ("3-month Euribor, reset every 3 months", {"index": "IRT_M3", "reset": 3}),
        ("all euro-area government bonds", {"bonds": "CGB_EA"}),
        ("no floor at zero", {"floor": False}),
        ("fixed for 30 years (30-year par yield)", {"fixed_years": 30}),
    ]
    for name, kw in variants:
        rv = mc.all_cohorts(mc.Loans(m, **kw))
        print(f"  {name:40s} {summary(rv)}")

    # ------------------------------------------------------------------------------------------------ 8
    head(f"8. A loan started in {mc.LAST}")
    raw, fwd, fixed_rate = mc.new_loan_scenarios(m)
    dem, _, _ = mc.new_loan_scenarios(m, demean=True)
    first_var = mc.floored(e12[mc.LAST]) + mc.MARGIN
    print(f"variable starts at {first_var:.2f}%, fixed at {fixed_rate:.2f}%")
    print("12-month Euribor implied by the yield curve at each annual reset: " + ", ".join(f"{v:.2f}" for v in fwd))
    p0 = mc.payment(mc.PRINCIPAL, first_var, mc.TERM)
    print(
        f"first variable payment {p0:.0f} euros a month per 100,000;"
        f" fixed {mc.payment(mc.PRINCIPAL, fixed_rate, mc.TERM):.0f}"
    )
    for name, s in [("past errors as they were", raw), ("past errors without their average", dem)]:
        print(f"{name} ({len(s)} scenarios, from start months {s.index[0]} to {s.index[-1]}):")
        print(
            f"  variable cheaper in {100 * (s.extra_fixed > 0).mean():.0f}%; extra cost of fixing: median"
            f" {eur(s.extra_fixed.median())}, 10th percentile {eur(mc.quantile(s.extra_fixed, 10))},"
            f" 90th {eur(mc.quantile(s.extra_fixed, 90))}"
        )
        print(
            f"  biggest payment rise: median {100 * s.rise.median():.0f}%,"
            f" above 25% in {100 * (s.rise > 0.25).mean():.0f}%,"
            f" above 50% in {100 * (s.rise > 0.5).mean():.0f}%, largest {100 * s.rise.max():.0f}%"
            f" ({p0 * (1 + s.rise.max()):.0f} euros a month)"
        )
    pd.concat([raw.assign(kind="raw"), dem.assign(kind="demeaned")]).round(4).to_csv(
        RES / "new_loan.csv", index_label="origin"
    )
    key = {"fixed_rate_now": fixed_rate, "variable_rate_now": first_var}
    pd.Series(key).round(4).to_csv(RES / "key_numbers.csv", header=["value"], index_label="name")


if __name__ == "__main__":
    main()
