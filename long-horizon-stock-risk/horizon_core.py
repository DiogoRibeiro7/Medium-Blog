"""Real stock and bond wealth over holding periods of 1 to 30 years, and what the outcomes look like.

The rules, in one place:

- Year ends. Every series is read at the end of each calendar year: the December monthly average for
  share prices and bond yields, and December for consumer prices (the fourth-quarter average where a
  market's CPI is quarterly; November where a monthly series has no December, as in Ireland before 1975).
- Stocks. The OECD share price indices leave out dividends. A constant dividend yield d is added, paid
  and reinvested once a year: a year's return is (price at the end / price at the start) times (1 + d),
  minus 1. d = 0 gives price returns only. For the United States, Shiller's monthly S&P composite gives
  actual dividends: each month returns (P_t + D_t / 12) / P_{t-1}.
- Bonds. A ten-year government bond bought at par at the year-end yield, held for a year (one annual
  coupon) and sold as a nine-year bond at the next year-end yield. It is an approximation of a constant
  maturity bond fund; the OECD and Shiller publish yields, not bond returns.
- Real. Every wealth level is divided by the consumer price index.
- Windows. A holding period starts at the end of year s and ends at the end of year s + H. Every start
  year is used, so windows overlap: the 30-year window from 1960 and the one from 1961 share 29 years.
"""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from dataclasses import dataclass

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
MATURITY = 10


# ----------------------------------------------------------------------------------------------- readers


def read_sdmx(text: str) -> dict[tuple[str, str], dict[str, float]]:
    """SDMX-CSV text -> {(REF_AREA, FREQ): {TIME_PERIOD: value}}, skipping empty observations."""
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        if row["OBS_VALUE"]:
            out[(row["REF_AREA"], row["FREQ"])][row["TIME_PERIOD"]] = float(row["OBS_VALUE"])
    return dict(out)


def year_end(monthly: dict[str, float]) -> dict[int, float]:
    """December values of a monthly series."""
    return {int(k[:4]): v for k, v in monthly.items() if k[5:] == "12"}


def cpi_year_end(cpi: dict[tuple[str, str], dict[str, float]], area: str) -> tuple[dict[int, float], str]:
    """Year-end consumer prices for one market, and which frequency they come from.

    The source is the monthly December value if the market has at least as many Decembers as fourth
    quarters, with November where December is missing; otherwise the fourth-quarter average throughout.
    """
    monthly = cpi.get((area, "M"), {})
    quarterly = cpi.get((area, "Q"), {})
    dec = year_end(monthly)
    q4 = {int(k[:4]): v for k, v in quarterly.items() if k.endswith("-Q4")}
    if len(dec) >= len(q4) and dec:
        nov = {int(k[:4]): v for k, v in monthly.items() if k[5:] == "11"}
        return {y: dec.get(y, nov.get(y)) for y in sorted(set(dec) | set(nov))}, "monthly"
    return q4, "quarterly"


def longest_run(years) -> list[int]:
    """The longest run of consecutive years."""
    ys = sorted(years)
    best: list[int] = []
    run: list[int] = []
    for y in ys:
        run = run + [y] if run and y == run[-1] + 1 else [y]
        if len(run) > len(best):
            best = run
    return best


# ----------------------------------------------------------------------------------------------- returns


def par_bond_price(coupon: float, rate: float, years: int) -> float:
    """Price per unit face of a bond with annual coupons, valued at a yield (which may be zero or negative)."""
    disc = 1.0 + rate
    return math.fsum(coupon / disc**k for k in range(1, years + 1)) + 1.0 / disc**years


def bond_return(y0: float, y1: float, maturity: int = MATURITY) -> float:
    """One year's return on a par bond bought at yield y0 and sold a year later at yield y1 (decimals)."""
    return y0 + par_bond_price(y0, y1, maturity - 1) - 1.0


def chain(returns: dict[int, float], first: int) -> dict[int, float]:
    """Wealth levels from yearly returns: level[first] = 1, level[y] = level[y-1] * (1 + returns[y])."""
    out = {first: 1.0}
    y = first + 1
    while y in returns:
        out[y] = out[y - 1] * (1.0 + returns[y])
        y += 1
    return out


def deflate(level: dict[int, float], cpi: dict[int, float]) -> dict[int, float]:
    """Real levels, kept only for years with a price index; stops at the first missing year."""
    out = {}
    for y in sorted(level):
        if y not in cpi or cpi[y] is None:
            break
        out[y] = level[y] / cpi[y]
    return out


@dataclass(frozen=True)
class Market:
    area: str
    stocks: dict[int, float]  # real wealth levels, year-end
    bonds: dict[int, float]  # real wealth levels, year-end (may be empty)
    cpi_source: str


def oecd_market(area, share, irlt, cpi, dividend_yield: float) -> Market | None:
    """One OECD market's real stock and bond levels, from the three SDMX tables."""
    prices = year_end(share.get((area, "M"), {}))
    yields = year_end(irlt.get((area, "M"), {}))
    deflator, source = cpi_year_end(cpi, area)
    years = longest_run(set(prices) & set(deflator))
    if len(years) < 2:
        return None
    stock_ret = {y: prices[y] / prices[y - 1] * (1.0 + dividend_yield) - 1.0 for y in years[1:]}
    stocks = deflate(chain(stock_ret, years[0]), deflator)
    bonds: dict[int, float] = {}
    byears = longest_run(set(yields) & set(deflator))
    if len(byears) >= 2:
        bond_ret = {y: bond_return(yields[y - 1] / 100, yields[y] / 100) for y in byears[1:]}
        bonds = deflate(chain(bond_ret, byears[0]), deflator)
    return Market(area, stocks, bonds, source)


def read_shiller(text: str) -> list[dict[str, float | str]]:
    """Shiller's monthly rows with a price, dividend, CPI and long rate (the mirror writes 0 for missing)."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        p, d, c, y = (float(r[k]) for k in ("SP500", "Dividend", "Consumer Price Index", "Long Interest Rate"))
        if p > 0 and d > 0 and c > 0 and y > 0:
            rows.append({"month": r["Date"][:7], "price": p, "dividend": d, "cpi": c, "long_rate": y})
    return rows


def shiller_market(rows) -> Market:
    """United States from Shiller: stocks with actual dividends, the ten-year bond rule on his long rate."""
    by_month = {r["month"]: r for r in rows}
    months = sorted(by_month)
    tr = {months[0]: 1.0}
    for a, b in zip(months, months[1:], strict=False):
        pa, rb = by_month[a]["price"], by_month[b]
        tr[b] = tr[a] * (rb["price"] + rb["dividend"] / 12.0) / pa
    years = [int(m[:4]) for m in months if m.endswith("-12")]
    nominal = {y: tr[f"{y}-12"] for y in years}
    cpi = {y: by_month[f"{y}-12"]["cpi"] for y in years}
    rate = {y: by_month[f"{y}-12"]["long_rate"] / 100 for y in years}
    stocks = {y: nominal[y] / cpi[y] for y in years}
    bond_ret = {y: bond_return(rate[y - 1], rate[y]) for y in years[1:]}
    bonds = deflate(chain(bond_ret, years[0]), cpi)
    return Market("USA", stocks, bonds, "Shiller monthly")


def shiller_dividend_yield(rows, first_year: int, last_year: int) -> float:
    """Average December dividend yield D/P over a span of years."""
    vals = [r["dividend"] / r["price"] for r in rows if r["month"].endswith("-12")]
    yrs = [int(r["month"][:4]) for r in rows if r["month"].endswith("-12")]
    sel = [v for v, y in zip(vals, yrs, strict=True) if first_year <= y <= last_year]
    return math.fsum(sel) / len(sel)


# ----------------------------------------------------------------------------------------------- windows


@dataclass(frozen=True)
class Window:
    area: str
    start: int
    horizon: int
    multiple: float  # real wealth at the end per unit at the start


def windows(level: dict[int, float], area: str, horizon: int, starts=None) -> list[Window]:
    out = []
    for s in sorted(level):
        if s + horizon in level and (starts is None or s in starts):
            out.append(Window(area, s, horizon, level[s + horizon] / level[s]))
    return out


def relative_windows(m: Market, horizon: int, starts=None) -> list[Window]:
    """Stocks against bonds: real stock multiple divided by real bond multiple, where both exist."""
    out = []
    for s in sorted(m.stocks):
        e = s + horizon
        if e in m.stocks and s in m.bonds and e in m.bonds and (starts is None or s in starts):
            out.append(Window(m.area, s, horizon, (m.stocks[e] / m.stocks[s]) / (m.bonds[e] / m.bonds[s])))
    return out


def episodes(ws: list[Window]) -> list[tuple[str, int, int]]:
    """Runs of consecutive start years with a loss (multiple below 1), per market: (area, first, last)."""
    bad = sorted((w.area, w.start) for w in ws if w.multiple < 1.0)
    out: list[tuple[str, int, int]] = []
    for area, s in bad:
        if out and out[-1][0] == area and out[-1][2] == s - 1:
            out[-1] = (area, out[-1][1], s)
        else:
            out.append((area, s, s))
    return out


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile (the same rule as numpy's default)."""
    v = sorted(values)
    if not v:
        return float("nan")
    pos = q * (len(v) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


@dataclass(frozen=True)
class Summary:
    horizon: int
    n: int
    markets: int
    p_loss: float  # share of windows with a multiple below 1, all windows pooled
    p_loss_equal: float  # mean over markets of each market's share
    median_loss: float  # median of 1 - multiple among losing windows
    mean_loss: float
    expected_shortfall: float  # mean of max(0, 1 - multiple) over all windows
    p05: float
    median: float
    worst: Window | None
    episodes: int
    independent: int  # sum over markets of floor(years spanned by starts and ends / horizon)


def summarise(ws: list[Window], horizon: int) -> Summary:
    by_area: dict[str, list[float]] = defaultdict(list)
    for w in ws:
        by_area[w.area].append(w.multiple)
    mult = [w.multiple for w in ws]
    losses = [1.0 - m for m in mult if m < 1.0]
    n = len(mult)
    indep = 0
    for area in by_area:
        starts = [w.start for w in ws if w.area == area]
        indep += (max(starts) + horizon - min(starts)) // horizon
    return Summary(
        horizon=horizon,
        n=n,
        markets=len(by_area),
        p_loss=len(losses) / n if n else float("nan"),
        p_loss_equal=math.fsum(sum(v < 1.0 for v in vals) / len(vals) for vals in by_area.values()) / len(by_area)
        if by_area
        else float("nan"),
        median_loss=quantile(losses, 0.5),
        mean_loss=math.fsum(losses) / len(losses) if losses else float("nan"),
        expected_shortfall=math.fsum(losses) / n if n else float("nan"),
        p05=quantile(mult, 0.05),
        median=quantile(mult, 0.5),
        worst=min(ws, key=lambda w: w.multiple) if ws else None,
        episodes=len(episodes(ws)),
        independent=indep,
    )


def equal_weight_mix(markets, minimum: int = 5) -> dict[int, float]:
    """Real wealth of an equal-weight mix of markets, rebalanced every year-end, in local real terms.

    Each year's return is the average of the real returns of every market with data for that year; the
    mix starts at the first year-end from which at least `minimum` markets have data. Exchange rates are
    left out: it is the average saver's experience across markets, not a portfolio one saver could hold.
    """
    years = sorted({y for m in markets for y in m})
    rets: dict[int, float] = {}
    for y in years:
        r = [m[y] / m[y - 1] - 1.0 for m in markets if y in m and y - 1 in m]
        if len(r) >= minimum:
            rets[y] = math.fsum(r) / len(r)
    first = min(rets) - 1
    return chain(rets, first)


def annualised(multiple: float, horizon: int) -> float:
    return multiple ** (1.0 / horizon) - 1.0


# ----------------------------------------------------------------------------------------------- shuffles


def log_returns(level: dict[int, float]) -> list[float]:
    ys = sorted(level)
    return [math.log(level[b] / level[a]) for a, b in zip(ys, ys[1:], strict=False)]


def level_from_logs(logs, first: int) -> dict[int, float]:
    out = {first: 1.0}
    acc = 0.0
    for k, r in enumerate(logs, start=1):
        acc += r
        out[first + k] = math.exp(acc)
    return out


def variance_ratio(logs: list[float], horizon: int) -> float:
    """Variance of overlapping H-year log returns over H times the variance of one-year log returns."""
    n = len(logs)
    if n <= horizon + 1:
        return float("nan")
    mean1 = math.fsum(logs) / n
    var1 = math.fsum((r - mean1) ** 2 for r in logs) / (n - 1)
    sums = [math.fsum(logs[i : i + horizon]) for i in range(n - horizon + 1)]
    meanh = math.fsum(sums) / len(sums)
    varh = math.fsum((s - meanh) ** 2 for s in sums) / (len(sums) - 1)
    return varh / (horizon * var1)
