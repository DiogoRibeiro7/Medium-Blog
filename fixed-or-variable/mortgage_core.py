"""Replaying fixed- and variable-rate mortgages over the euro's history.

A loan of 100,000 euros over 30 years, repaid in equal monthly instalments, is started in every month from
January 1999. The variable loan pays 12-month Euribor plus a margin, reset once a year. The fixed loan pays the
10-year par yield of AAA-rated euro-area government bonds plus the same margin, fixed for ten years. Both
reference rates are floored at zero, as most euro-area contracts were when market rates were negative.

Every sum that reaches a printed number is a fixed-order Python sum or math.fsum, so results do not depend on
the processor's vector kernels.
"""

import csv
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

PRINCIPAL = 100_000.0
TERM = 360  # months
HORIZON = 120  # months compared: the ten years of the fixed rate
MARGIN = 1.0  # percentage points over the reference rate, both loans
START_SHARE = 0.30  # illustration: the first payment is 30% of the household's income
SPLICE = "2004-09"  # first month of Eurostat's current yield-curve series
FIRST, LAST = "1999-01", "2026-10"

# Published euro-area averages used to check the replay's rates (ECB bank interest rate statistics, new business,
# loans to households for house purchase; press releases of 3 November 2021, 2 August 2023, 4 December 2023 and
# 1 February 2024).
ECB_CHECKS = [
    ("2021-09", "fixed", 1.29, "initial rate fixation over ten years"),
    ("2023-06", "fixed", 3.41, "initial rate fixation over ten years"),
    ("2023-10", "variable", 4.81, "floating rate and initial rate fixation up to one year"),
    ("2023-12", "variable", 4.86, "floating rate and initial rate fixation up to one year"),
]


# ---------------------------------------------------------------------------------------------- data files
def jsonstat_rows(d, item_dims):
    """Rows (geo, item, period, value) from a Eurostat JSON-stat 2.0 response; the item joins item_dims with |."""
    dims, sizes = d["id"], d["size"]
    keys = [list(d["dimension"][k]["category"]["index"]) for k in dims]
    rows = []
    for flat, value in d["value"].items():
        i, pos = int(flat), []
        for s in reversed(sizes):
            pos.append(i % s)
            i //= s
        pos.reverse()
        label = {k: keys[j][p] for j, (k, p) in enumerate(zip(dims, pos, strict=True))}
        item = "|".join(label[k] if k in label else k for k in item_dims)
        rows.append((label["geo"], item, label["time"], float(value)))
    return rows


def wide_csv(rows):
    """A deterministic wide CSV: one line per (geo, item), one column per period, values as repr(float)."""
    table, periods = {}, set()
    for geo, item, period, value in rows:
        key = (geo, item)
        if period in table.get(key, {}) and table[key][period] != value:
            raise ValueError(f"conflicting values for {geo} {item} {period}")
        table.setdefault(key, {})[period] = value
        periods.add(period)
    periods = sorted(periods)
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["geo", "item"] + periods)
    for key in sorted(table):
        w.writerow(list(key) + [repr(table[key][p]) if p in table[key] else "" for p in periods])
    return out.getvalue()


def read_wide(path):
    """{item: Series(index=period)} from a wide CSV (geo is dropped; items are unique per file)."""
    df = pd.read_csv(path, dtype={"geo": str, "item": str}).drop(columns="geo").set_index("item")
    df.columns = df.columns.astype(str)
    return {item: row.dropna().astype(float) for item, row in df.iterrows()}


class Market:
    """Monthly reference rates, prices and incomes."""

    def __init__(self, euribor, curves, prices):
        self.e = {k.split("|")[0]: v for k, v in euribor.items()}  # IRT_M3, IRT_M6, IRT_M12
        self.curves = curves
        self.hicp = prices["HICP"]
        self.wage = prices["COMP_EMP"]
        self.months = [m for m in self.e["IRT_M12"].index if FIRST <= m <= LAST]

    def curve(self, kind, maturity, bonds="CGB_EA_AAA"):
        """Par or spot yield for one maturity, spliced: Eurostat's historical series before SPLICE."""
        old = self.curves[f"{kind}|HIST|Y{maturity}"]
        new = self.curves[f"{kind}|{bonds}|Y{maturity}"]
        return pd.concat([old[old.index < SPLICE], new[new.index >= SPLICE]])

    def income(self):
        """Monthly income index: compensation per employee, placed at mid-year and interpolated in logs."""
        years = sorted(int(y) for y in self.wage.index)
        out = {}
        for y in range(years[0], years[-1] + 1):
            for m in range(1, 13):
                t = y + (m - 7) / 12
                y0 = min(max(math.floor(t), years[0]), years[-1] - 1)
                a, b = math.log(self.wage[str(y0)]), math.log(self.wage[str(y0 + 1)])
                out[f"{y}-{m:02d}"] = math.exp(a + (b - a) * (t - y0))
        return pd.Series(out)


def load(raw_euribor=None):
    from fetch_data import euribor_csv  # noqa: PLC0415 (Euribor is downloaded, not committed)

    return Market(
        read_wide(raw_euribor or euribor_csv()),
        read_wide(DATA / "yield_curves.csv"),
        read_wide(DATA / "prices_income.csv"),
    )


# ------------------------------------------------------------------------------------------------ the loan
def payment(balance, rate, months):
    """Equal monthly instalment that repays balance over months at an annual nominal rate in per cent."""
    i = rate / 1200
    if abs(i) < 1e-15:
        return balance / months
    return balance * i / (1 - (1 + i) ** -months)


def replay(rate_at, reset, n, term=TERM, principal=PRINCIPAL):
    """Monthly schedule for n months. rate_at(k) is the annual rate set at month k; it is called when k is a
    multiple of reset, and the instalment is recomputed over the remaining term."""
    balance, rows = principal, []
    rate = pay = None
    for k in range(min(n, term)):
        if k % reset == 0:
            rate = rate_at(k)
            pay = payment(balance, rate, term - k)
        interest = balance * rate / 1200
        balance = balance + interest - pay
        rows.append((rate, pay, interest, balance))
    return pd.DataFrame(rows, columns=["rate", "payment", "interest", "balance"])


def floored(x):
    return x if x > 0 else 0.0


class Loans:
    """Variable and fixed loans started in any month, with settings for the checks."""

    def __init__(
        self,
        market,
        margin_var=MARGIN,
        margin_fix=MARGIN,
        index="IRT_M12",
        reset=12,
        fixed_years=10,
        bonds="CGB_EA_AAA",
        floor=True,
    ):
        self.m = market
        self.margin_var, self.margin_fix = margin_var, margin_fix
        self.euribor = market.e[index]
        self.reset, self.fixed_years, self.floor = reset, fixed_years, floor
        self.par = market.curve("PAR", fixed_years, bonds)
        self.months = market.months

    def _ref(self, x):
        return floored(x) if self.floor else x

    def offset(self, start, k):
        return self.months[self.months.index(start) + k]

    def available(self, start):
        return len(self.months) - self.months.index(start)

    def variable(self, start, n=None):
        n = n or self.available(start)
        return replay(lambda k: self._ref(self.euribor[self.offset(start, k)]) + self.margin_var, self.reset, n)

    def fixed(self, start, n=None):
        n = n or self.available(start)
        return replay(lambda k: self._ref(self.par[self.offset(start, k)]) + self.margin_fix, 12 * self.fixed_years, n)


def cohort(loans, start, horizon=HORIZON, income=None, hicp=None):
    """What a loan started in `start` cost and how its payment moved over the first `horizon` months (or up to
    the last month of data)."""
    n = min(horizon, loans.available(start))
    v, f = loans.variable(start, n), loans.fixed(start, n)
    months = loans.months[loans.months.index(start) : loans.months.index(start) + n]
    out = {
        "start": start,
        "months": n,
        "complete": n == horizon,
        "rate_var": v.rate.iloc[0],
        "rate_fix": f.rate.iloc[0],
        "pay_var0": v.payment.iloc[0],
        "pay_fix": f.payment.iloc[0],
        "pay_var_max": max(v.payment),
        "peak_month": months[int(np.argmax(v.payment.to_numpy()))],
        "interest_var": math.fsum(v.interest),
        "interest_fix": math.fsum(f.interest),
    }
    out["extra_fixed"] = out["interest_fix"] - out["interest_var"]  # > 0: fixing cost more
    out["rise"] = out["pay_var_max"] / out["pay_var0"] - 1
    out["extra_peak_year"] = max(
        math.fsum(max(p - out["pay_var0"], 0.0) for p in v.payment.iloc[j : j + 12]) for j in range(max(n - 11, 1))
    )
    if income is not None and start in income.index:
        known = [i for i, m in enumerate(months) if m in income.index]
        inc0 = income[start]
        out["share_var_max"] = max(
            START_SHARE * v.payment.iloc[i] / out["pay_var0"] * inc0 / income[months[i]] for i in known
        )
        out["share_fix_end"] = START_SHARE * inc0 / income[months[known[-1]]]
    if hicp is not None and start in hicp.index:
        out["real_fix_end"] = 1 / (hicp[months[-1]] / hicp[start])
    out["euribor_avg"] = math.fsum(loans.euribor[m] for m in months) / n
    out["par_start"] = loans.par[start]
    return out


def all_cohorts(loans, **kw):
    return pd.DataFrame([cohort(loans, s, **kw) for s in loans.months]).set_index("start")


# ------------------------------------------------------------------------------------------- expectations
def forward_path(market, start, years=10):
    """Euribor the yield curve implied at `start` for each annual reset: the one-year forward rate h years ahead
    (spot rates are continuously compounded) plus the gap between 12-month Euribor and the one-year spot rate."""
    spot = {h: market.curve("SPOT_RT", h)[start] for h in range(1, years + 1)}
    gap = market.e["IRT_M12"][start] - spot[1]
    out = [market.e["IRT_M12"][start]]
    for h in range(1, years):
        out.append(spot[h + 1] * (h + 1) - spot[h] * h + gap)
    return out


def forecast_errors(market, years=10):
    """For each start month with ten years of outcomes: realised 12-month Euribor at each annual reset minus the
    rate the yield curve implied at the start."""
    months = market.months
    rows = {}
    for i, start in enumerate(months):
        if i + 12 * (years - 1) >= len(months):
            break
        fwd = forward_path(market, start, years)
        rows[start] = [market.e["IRT_M12"][months[i + 12 * h]] - fwd[h] for h in range(years)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"h{h}" for h in range(years)])


def scenario_loans(path, fixed_rate, margin=MARGIN, floor=True):
    """A variable loan whose annual Euribor fixings follow `path`, against a fixed loan at fixed_rate."""

    def ref(x):
        return floored(x) if floor else x

    v = replay(lambda k: ref(path[k // 12]) + margin, 12, HORIZON)
    f = replay(lambda k: fixed_rate, 120, HORIZON)
    return {
        "extra_fixed": math.fsum(f.interest) - math.fsum(v.interest),
        "rise": max(v.payment) / v.payment.iloc[0] - 1,
    }


def new_loan_scenarios(market, start=LAST, demean=False, margin=MARGIN):
    """A loan started at `start`, replayed against every past forecast error laid on today's yield curve."""
    errors = forecast_errors(market)
    if demean:
        errors = errors - errors.mean()
    fwd = forward_path(market, start)
    fixed_rate = floored(market.curve("PAR", 10)[start]) + margin
    rows = []
    for origin, e in errors.iterrows():
        path = [fwd[h] + e[f"h{h}"] for h in range(10)]
        rows.append({"origin": origin, **scenario_loans(path, fixed_rate, margin)})
    return pd.DataFrame(rows).set_index("origin"), fwd, fixed_rate


# ----------------------------------------------------------------------------------------------- helpers
def mean(x):
    x = np.asarray(x, float)
    return math.fsum(x) / len(x)


def quantile(x, q):
    return float(np.percentile(np.asarray(x, float), q))
