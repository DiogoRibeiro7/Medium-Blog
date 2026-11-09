"""Early repayment of the IMF loan, 2015-2018: what it saved, in euros, and what it changed.

    python imf.py

The IMF lent Portugal SDR 22.942 billion between May 2011 and April 2014 under an Extended Fund
Facility. Each purchase was due in twelve equal semi-annual repurchases from 4.5 to 10 years after
it was made, so the original schedule ran to 2024. Portugal repaid it all by December 2018.

Two paths are compared month by month, in SDR and then in euros:

- A, the original schedule: purchases dated by the months in which the BPstat stock of non-euro
  loans of the State jumps (scaled to the documented total), repaid as the EFF rule prescribes;
- B, what happened: the BPstat stock converted to SDR, repaid in the order of the instalments'
  due dates.

Charges on IMF credit are the SDR interest rate (rebuilt from its basket, see ``common.sdr``) plus
a margin of 100 basis points (60 from November 2024) plus a level-based surcharge of 200 basis
points on credit above a threshold: 300% of the old quota (SDR 1,029.7 million) until 17 February
2016 and 187.5% of the new quota (SDR 2,060.1 million) after. The time-based surcharge (100 basis
points on above-threshold credit outstanding more than 36 months, 51 for the EFF after 2016) is
left out of the baseline, because its base depends on conventions the public data cannot pin
down; it is reported as a sensitivity. The baseline reproduces the 243 basis points average spread
that IGCP data implied at the end of 2016 (``documented.csv``).

Every instalment repaid before its due date is replaced by a euro bond issued that month, maturing
on the instalment's original due date, at the Portuguese yield for that residual maturity (the
matched replacement: it removes the IMF cost without changing the maturity of the debt). The
saving is then split into:

- interest: IMF charges avoided minus coupons on the replacement bonds;
- exchange rate: the euro value of the SDR principal on its original due date minus its euro value
  when it was repaid (positive when the SDR appreciated against the euro afterwards).

Ex ante, the same arithmetic is done with the SDR rate and the exchange rate frozen at the month
of each repayment, as a treasury could have done when deciding. A second replacement, ten-year
bonds, measures what lengthening the debt added after the original due dates (the counterfactual
then refinances on the due date at the realised Portuguese yield for the remaining years; months
after the last observation keep the last yield).

Writes ``results/imf_*.csv`` and ``results/imf_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, bp_series, pt_curve, pt_yield, sdr, write

IGCP_ALL_IN = 4.4  # %, cost of the IMF loan reported from IGCP data in mid-2017 (documented.csv)
TOTAL_PURCHASES_SDR = 22.942  # bn, IMF press release 14/173 (April 2014)
OLD_QUOTA, NEW_QUOTA = 1.0297, 2.0601  # bn SDR (2,305.7% of quota = SDR 23.742 bn; quota after the 14th review)
NEW_RULES = pd.Period("2016-02", "M")
MARGIN_CUT = pd.Period("2024-11", "M")
LEVEL_SURCHARGE, TIME_SURCHARGE = 2.0, 1.0
END = pd.Period("2024-12", "M")
JUMP = 0.3  # bn SDR: a monthly rise in the stock above this is a purchase


def threshold(m: pd.Period) -> float:
    return 3.0 * OLD_QUOTA if m < NEW_RULES else 1.875 * NEW_QUOTA


def time_trigger(m: pd.Period) -> int:
    return 36 if m < NEW_RULES else 51


def purchases(stock_sdr: pd.Series) -> pd.Series:
    d = stock_sdr.diff().loc["2011-04":"2014-06"]
    p = d[d > JUMP]
    return p * TOTAL_PURCHASES_SDR / p.sum()


def schedule(p: pd.Series) -> pd.DataFrame:
    rows = []
    for m, amt in p.items():
        for k in range(12):
            rows.append(dict(purchase=m, due=m + 54 + 6 * k, sdr=amt / 12))
    return pd.DataFrame(rows).sort_values(["due", "purchase"]).reset_index(drop=True)


def rate_of_charge(m: pd.Period, credit: float, aged: float, sdr_rate: float, with_time: str | None) -> float:
    """Average annual rate (percent) on the whole credit outstanding."""
    if credit <= 0:
        return 0.0
    margin = 0.6 if m >= MARGIN_CUT else 1.0
    above = max(0.0, credit - threshold(m))
    extra = LEVEL_SURCHARGE * above
    if with_time == "high":
        extra += TIME_SURCHARGE * min(aged, above)
    elif with_time == "low":
        extra += TIME_SURCHARGE * max(0.0, aged - threshold(m))
    return sdr_rate + margin + extra / credit


def marginal_rate(m: pd.Period, credit: float, sdr_rate: float) -> float:
    margin = 0.6 if m >= MARGIN_CUT else 1.0
    return sdr_rate + margin + (LEVEL_SURCHARGE if credit > threshold(m) else 0.0)


def paths(with_time: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = sdr()
    eur = bp_series({12561444: "imf_eur"})["imf_eur"]
    eur.index = pd.PeriodIndex(eur.index, freq="M")
    months = pd.period_range("2011-04", END, freq="M")
    eur = eur.reindex(months).fillna(0.0)
    raw = eur / s["eur_per_sdr"].reindex(months) / 1000
    p = purchases(raw)
    inst = schedule(p)
    # A: original schedule
    a_out = pd.Series(0.0, index=months)
    for m in months:
        a_out[m] = p[p.index <= m].sum() - inst[inst["due"] <= m]["sdr"].sum()
    # B: observed stock, never above A before the last purchase, never rising after it
    b = raw.copy()
    last = p.index.max()
    b[b.index <= last] = a_out[a_out.index <= last]
    after = b[b.index > last]
    b[b.index > last] = np.minimum(after.cummin().to_numpy(), a_out[a_out.index > last].to_numpy())
    b[b < 0.05] = 0.0
    # consume instalments in due-date order as B repays
    inst["paid"] = pd.NaT
    remaining = inst["sdr"].to_numpy().copy()
    paid_month = [None] * len(inst)
    repaid = (b.shift(1, fill_value=0.0) - b).clip(lower=0.0)
    repaid[repaid.index <= last] = 0.0
    j = 0
    for m, amt in repaid.items():
        while amt > 1e-9 and j < len(inst):
            take = min(amt, remaining[j])
            remaining[j] -= take
            amt -= take
            if remaining[j] <= 1e-9:
                paid_month[j] = m
                j += 1
    inst["paid"] = paid_month
    # aged credit (FIFO vintages) for each path, for the time-based sensitivity
    def aged(out: pd.Series, m: pd.Period) -> float:
        repaid_total = p.sum() - out[m]
        acc, res = 0.0, 0.0
        for pm, amt in p.items():
            left = max(0.0, amt - max(0.0, repaid_total - acc))
            acc += amt
            if (m - pm).n > time_trigger(m):
                res += left
        return res

    rows = []
    for m in months:
        r = s.loc[m, "sdr_rate"]
        e = s.loc[m, "eur_per_sdr"]
        ra = rate_of_charge(m, a_out[m], aged(a_out, m) if with_time else 0.0, r, with_time)
        rb = rate_of_charge(m, b[m], aged(b, m) if with_time else 0.0, r, with_time)
        rows.append(dict(month=m, sdr_rate=r, eur_per_sdr=e, credit_a=a_out[m], credit_b=b[m], rate_a=ra, rate_b=rb,
                         charges_a_eur=ra / 100 / 12 * a_out[m] * e * 1000, charges_b_eur=rb / 100 / 12 * b[m] * e * 1000))
    return pd.DataFrame(rows).set_index("month"), inst


def replacement(inst: pd.DataFrame, path: pd.DataFrame, tenor: float | None = None) -> pd.DataFrame:
    """One euro bond for each instalment repaid early (matched to its due date, or `tenor` years)."""
    curve = pt_curve()
    last_curve = curve.index.max()
    early = inst[inst["paid"].notna() & (inst["paid"] < inst["due"])].copy()
    rows = []
    for r in early.itertuples():
        tau = r.paid
        e_tau = path.loc[tau, "eur_per_sdr"]
        e_due = path.loc[r.due, "eur_per_sdr"] if r.due in path.index else np.nan
        years = (r.due - tau).n / 12
        eur = r.sdr * e_tau * 1000
        cpn_matched = pt_yield(curve, tau, years)
        cpn = cpn_matched if tenor is None else pt_yield(curve, tau, tenor)
        # charges avoided on this instalment, at the marginal rate of the B path each month
        avoided = 0.0
        for m in pd.period_range(tau + 1, r.due, freq="M"):
            if m not in path.index:
                break
            credit = path.loc[m, "credit_b"] + r.sdr  # had it not been repaid
            avoided += marginal_rate(m, credit, path.loc[m, "sdr_rate"]) / 100 / 12 * r.sdr * path.loc[m, "eur_per_sdr"] * 1000
        ex_ante = (marginal_rate(tau, path.loc[tau, "credit_b"] + r.sdr, path.loc[tau, "sdr_rate"]) - cpn_matched) / 100 * eur * years
        row = dict(paid=tau, due=r.due, years=years, sdr=r.sdr, eur=eur, coupon=cpn, coupon_matched=cpn_matched,
                   avoided=avoided, coupons=cpn / 100 * eur * years, fx=r.sdr * (e_due - e_tau) * 1000, ex_ante=ex_ante)
        if tenor is not None:
            # after the due date: the counterfactual refinances the euro value of the principal
            end = tau + int(round(tenor * 12))
            post_cf, post_b = 0.0, 0.0
            if years < tenor:
                m_ref = min(r.due, last_curve)
                y_ref = pt_yield(curve, m_ref, (end - r.due).n / 12)
                post_years = (end - r.due).n / 12
                post_cf = y_ref / 100 * (r.sdr * e_due * 1000) * post_years
                post_b = cpn / 100 * eur * post_years
                row.update(refi_yield=y_ref)
            row.update(post_due_cf=post_cf, post_due_b=post_b, end=end)
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> str:
    path, inst = paths()
    rep = replacement(inst, path)
    rep10 = replacement(inst, path, tenor=10.0)
    path.reset_index().assign(month=lambda d: d["month"].astype(str)).to_csv(RESULTS / "imf_paths.csv", index=False, float_format="%.4f")
    rep.assign(paid=rep["paid"].astype(str), due=rep["due"].astype(str)).to_csv(RESULTS / "imf_replacement.csv", index=False, float_format="%.4f")

    lines = []
    lines.append(f"Purchases: SDR {TOTAL_PURCHASES_SDR:.3f} bn in {inst['purchase'].nunique()} months, {inst['purchase'].min()} to {inst['purchase'].max()}; "
                 f"original schedule ends {inst['due'].max()}")
    peak = path["credit_b"].max()
    lines.append(f"Peak credit SDR {peak:.2f} bn; threshold for the level-based surcharge SDR {3 * OLD_QUOTA:.3f} bn to January 2016, {1.875 * NEW_QUOTA:.3f} bn after")
    for m in ("2014-12", "2016-12", "2017-12", "2018-06"):
        r = path.loc[m]
        lines.append(f"  {m}: credit SDR {r.credit_b:.2f} bn (schedule {r.credit_a:.2f}); SDR rate {r.sdr_rate:.2f}%; average rate of charge {r.rate_b:.2f}% "
                     f"(spread over the SDR rate {100 * (r.rate_b - r.sdr_rate):.0f} bp); euro per SDR {r.eur_per_sdr:.4f}")
    by_year = rep.groupby(rep["paid"].map(lambda x: x.year))
    lines += ["", "Early repayments (instalments paid before their due date), by year of payment: SDR bn; euro bn; average years early; replacement coupon (matched, %)"]
    for y, f in by_year:
        lines.append(f"  {y}: SDR {f.sdr.sum():.2f}; EUR {f.eur.sum() / 1000:.2f}; {np.average(f.years, weights=f.sdr):.1f} years early; coupon {np.average(f.coupon, weights=f.eur):.2f}")
    lines.append(f"  total: SDR {rep.sdr.sum():.2f} bn, EUR {rep.eur.sum() / 1000:.2f} bn, {np.average(rep.years, weights=rep.sdr):.1f} years early on average")

    tot_avoided, tot_cpn, tot_fx, tot_ante = rep.avoided.sum(), rep.coupons.sum(), rep.fx.sum(), rep.ex_ante.sum()
    lines += ["", "Saving with matched replacement (EUR million, undiscounted, to each instalment's original due date):",
              f"  IMF charges avoided {tot_avoided:,.0f}; replacement coupons {tot_cpn:,.0f}; interest saving {tot_avoided - tot_cpn:,.0f}",
              f"  exchange rate on the principal (euro value at due date minus at repayment) {tot_fx:+,.0f}",
              f"  total ex post {tot_avoided - tot_cpn + tot_fx:,.0f}",
              f"  ex ante (SDR rate and exchange rate frozen at each repayment month) {tot_ante:,.0f}"]
    by_y = rep.assign(y=rep["paid"].map(lambda x: x.year)).groupby("y")
    lines.append("  by year of repayment, ex post interest / exchange rate / ex ante: " +
                 "; ".join(f"{y} {(f.avoided - f.coupons).sum():,.0f} / {f.fx.sum():+,.0f} / {f.ex_ante.sum():,.0f}" for y, f in by_y))
    allin = ((IGCP_ALL_IN - rep.coupon_matched) / 100 * rep.eur * rep.years).sum()
    lines.append(f"  ex ante at the {IGCP_ALL_IN}% all-in cost reported from IGCP data in 2017 instead of the contractual charges: {allin:,.0f}")
    dec18 = rep[rep["paid"] == pd.Period("2018-12", "M")]
    y10_dec18 = pt_curve().loc[pd.Period("2018-12", "M"), 10.0]
    alt = ((dec18.apply(lambda r: marginal_rate(r.paid, 0.0, path.loc[r.paid, "sdr_rate"]), axis=1) - y10_dec18) / 100 * dec18.eur * dec18.years).sum()
    lines.append(f"  December 2018 final payment: SDR {dec18.sdr.sum():.2f} bn (EUR {dec18.eur.sum() / 1000:.2f} bn), {np.average(dec18.years, weights=dec18.sdr):.1f} years early; "
                 f"interest saving {(dec18.avoided - dec18.coupons).sum():,.0f}; ex ante {dec18.ex_ante.sum():,.0f}; "
                 f"ex ante against the ten-year yield of {y10_dec18:.2f}% instead of the matched one: {alt:,.0f}")

    # the whole-portfolio view: charges paid on each path
    cf = path.loc[:END]
    lines += ["", "Whole-portfolio view, 2015-2024 (EUR million): IMF charges on the original schedule vs on the actual path",
              f"  schedule {cf.charges_a_eur.loc['2015-01':].sum():,.0f}; actual {cf.charges_b_eur.loc['2015-01':].sum():,.0f}; "
              f"difference {cf.charges_a_eur.loc['2015-01':].sum() - cf.charges_b_eur.loc['2015-01':].sum():,.0f} (before replacement coupons)"]

    # duration choice: ten-year replacement
    post = rep10.post_due_cf.sum() - rep10.post_due_b.sum()
    pre = (rep10.avoided - rep10.coupons).sum()
    lines += ["", "Lengthening instead: every early repayment refinanced with ten-year bonds (EUR million)",
              f"  interest saving to the original due dates {pre:,.0f} (the ten-year coupon is higher than the matched one by {np.average(rep10.coupon - rep10.coupon_matched, weights=rep10.eur):.2f} pp on average)",
              f"  after the due dates, to ten years: refinancing at realised yields would have cost {rep10.post_due_cf.sum():,.0f}, the ten-year bonds cost {rep10.post_due_b.sum():,.0f}: gain {post:,.0f}",
              f"  realised refinancing yield on the due dates, average {np.average(rep10.refi_yield.dropna(), weights=rep10.eur[rep10.refi_yield.notna()]):.2f}% against ten-year coupons {np.average(rep10.coupon, weights=rep10.eur):.2f}%"]

    # sensitivity: time-based surcharge conventions
    lines += ["", "Sensitivity: time-based surcharge included (average spread over the SDR rate at end-2016; whole-portfolio charge difference 2015-2024, EUR million)"]
    for conv in ("low", "high"):
        pth, _ = paths(conv)
        r = pth.loc["2016-12"]
        lines.append(f"  {conv}: spread {100 * (r.rate_b - r.sdr_rate):.0f} bp; charge difference {pth.charges_a_eur.loc['2015-01':].sum() - pth.charges_b_eur.loc['2015-01':].sum():,.0f}")
    return write("imf_summary.txt", lines)


if __name__ == "__main__":
    print(run())
