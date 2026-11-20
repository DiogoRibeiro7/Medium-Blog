"""Investing a windfall at once, or spreading it over several months: the replay behind the article.

A windfall of 1 is either invested in a country's stock index in the first month (lump sum) or in n equal monthly
instalments (spreading). Money waiting to be invested earns that country's short-term interest rate. Both are compared
one month after the last instalment, when both are fully invested; from then on they move together, so the gap in per
cent stays the same for as long as both are held.

Months are integers: year * 12 + month - 1. Sums use math.fsum, so the numbers do not depend on the machine.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

PANEL = tuple("AUS AUT BEL CAN CHE DEU DNK ESP FIN FRA GBR GRC IRL ISL ITA JPN LUX NLD NOR NZL PRT SWE USA".split())
NAMES = {
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "FIN": "Finland",
    "FRA": "France",
    "GBR": "United Kingdom",
    "GRC": "Greece",
    "IRL": "Ireland",
    "ISL": "Iceland",
    "ITA": "Italy",
    "JPN": "Japan",
    "LUX": "Luxembourg",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "NZL": "New Zealand",
    "PRT": "Portugal",
    "SWE": "Sweden",
    "USA": "United States",
}
SPREAD = 12  # months of instalments in the main comparison
HISTORY = 60  # months of prices needed before a start month to call it an all-time high


def month(text):
    """'1957-01' or '1957-01-01' -> integer month."""
    y, m = text.split("-")[:2]
    return int(y) * 12 + int(m) - 1


def label(m):
    return f"{m // 12}-{m % 12 + 1:02d}"


# ---------------------------------------------------------------------------------------------------- reading data
def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def series(rows, key="country", when="month", value="value"):
    out = {}
    for r in rows:
        out.setdefault(r[key], {})[month(r[when])] = float(r[value])
    return out


@dataclass
class Data:
    prices: dict  # {country: {month: share price index}}, OECD, monthly average
    cash: dict  # {country: {month: per cent a year}}, 3-month rate, overnight rate where the 3-month is missing
    cash_source: dict  # {country: {month: "3m" or "overnight"}}


def load():
    prices = series(read_rows(DATA / "share_prices.csv"))
    three = series(read_rows(DATA / "rate_3m.csv"))
    overnight = series(read_rows(DATA / "rate_overnight.csv"))
    cash, source = {}, {}
    for c in PANEL:
        cash[c], source[c] = {}, {}
        for m, v in overnight.get(c, {}).items():
            cash[c][m], source[c][m] = v, "overnight"
        for m, v in three.get(c, {}).items():
            cash[c][m], source[c][m] = v, "3m"
    return Data(prices=prices, cash=cash, cash_source=source)


# ---------------------------------------------------------------------------------------------------- statistics
def quantile(values, q):
    """Linear interpolation between order statistics (type 7)."""
    v = sorted(values)
    if not v:
        return math.nan
    h = (len(v) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (h - lo) * (v[hi] - v[lo])


def mean(values):
    v = list(values)
    return math.fsum(v) / len(v) if v else math.nan


def share(values, test):
    v = list(values)
    return sum(1 for x in v if test(x)) / len(v) if v else math.nan


# ---------------------------------------------------------------------------------------------------- the replay
def total_return(prices, dividend_yield=0.0):
    """Index that adds a constant dividend yield (per year, reinvested monthly) to a price index."""
    months = sorted(prices)
    out = {months[0]: 1.0}
    for prev, m in zip(months, months[1:], strict=False):
        if m != prev + 1:
            raise ValueError(f"gap in prices at {label(m)}")
        out[m] = out[prev] * (prices[m] / prices[prev] + dividend_yield / 12.0)
    return out


def can_start(index, cash, start, n):
    return all(start + k in index for k in range(n + 1)) and all(start + k in cash for k in range(n - 1))


def replay(index, cash, start, n):
    """Value of a windfall of 1 one month after the last of n instalments: (lump sum, spread).

    Instalments of 1/n are bought at the index of months start .. start + n - 1. Cash not yet invested earns the
    month's annual rate / 12. The last instalment invests everything left, interest included.
    """
    end = start + n
    lump = index[end] / index[start]
    units, left = 0.0, 1.0
    for k in range(n):
        buy = left if k == n - 1 else 1.0 / n
        units += buy / index[start + k]
        left -= buy
        if k < n - 1:
            left *= 1.0 + cash[start + k] / 1200.0
    return lump, units * index[end]


@dataclass
class Outcome:
    country: str
    start: int
    lump: float  # value of the lump sum, windfall = 1
    spread: float  # value of the spread purchase
    cash_rate: float  # annual rate in the start month, per cent
    at_high: bool  # price index at its highest level so far, with at least HISTORY months of history
    after_fall: bool  # price index 20% or more below its highest level so far

    @property
    def advantage(self):
        """Lump sum over spreading, in per cent of the spread value: positive means the lump sum did better."""
        return self.lump / self.spread - 1.0


def outcomes(d, n=SPREAD, dividend_yield=0.0, countries=PANEL):
    out = []
    for c in countries:
        prices = d.prices.get(c, {})
        if not prices:
            continue
        index = total_return(prices, dividend_yield)
        months = sorted(prices)
        running = -math.inf
        for s in months:
            running = max(running, prices[s])
            if not can_start(index, d.cash[c], s, n):
                continue
            lump, spread = replay(index, d.cash[c], s, n)
            history = s - months[0]
            out.append(
                Outcome(
                    country=c,
                    start=s,
                    lump=lump,
                    spread=spread,
                    cash_rate=d.cash[c][s],
                    at_high=history >= HISTORY and prices[s] >= running,
                    after_fall=prices[s] <= 0.8 * running,
                )
            )
    return out


# ---------------------------------------------------------------------------------------------- the US with dividends
def shiller(path):
    """Shiller's monthly S&P composite: {month: (price, annual dividend)} for months with a dividend."""
    out = {}
    for r in read_rows(path):
        p, dv = float(r["SP500"]), float(r["Dividend"])
        if p > 0 and dv > 0:
            out[month(r["Date"])] = (p, dv)
    return out


def shiller_indices(rows):
    """Price-only and total-return indices from Shiller's prices and dividends (dividend paid monthly as D / 12)."""
    months = sorted(rows)
    price = {m: rows[m][0] for m in months}
    tr = {months[0]: 1.0}
    for prev, m in zip(months, months[1:], strict=False):
        tr[m] = tr[prev] * (rows[m][0] + rows[m][1] / 12.0) / rows[prev][0]
    return price, tr
