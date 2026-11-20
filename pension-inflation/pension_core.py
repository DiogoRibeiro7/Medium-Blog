"""Fixed and indexed pensions replayed on each country's price history.

A pension that pays the same number of euros, pounds or dollars every year is valued in the money of the year it
starts. This module replays such a pension, and pensions with partial indexation rules, on the annual consumer price
index of 23 countries from 1955, weights each year by the chance of still being alive to receive it, prices level and
indexed annuities, and rebuilds the monthly amount of one Portuguese pension from the published annual updates.

Everything is plain Python with math.fsum, so the numbers do not depend on the machine.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# The 24 countries that were OECD members by 1973, without Turkey, whose inflation in the 1990s ran above 60% a
# year and would set the tails on its own. Turkey is reported separately.
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
    "TUR": "Turkey",
}
EXTRA = ("TUR",)
FIRST_START = 1955
AGE = 65
SPAN = 30  # payment years counted from 65: ages 65 to 94
LIFE_GEO = "EU27_2020"
LIFE_YEAR = 2019  # the last table before the pandemic years
REAL_RATE = 0.02  # real rate used to price an indexed annuity in the replay of annuity purchases


# ---------------------------------------------------------------------------------------------------- reading data
def read_rows(name):
    with open(DATA / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def series(rows, key, year="year", value="value"):
    """{key: {year: value}} from long rows."""
    out = {}
    for r in rows:
        out.setdefault(r[key], {})[int(r[year])] = float(r[value])
    return out


@dataclass
class Data:
    cpi: dict  # {country: {year: index}}, OECD consumer price index, annual average
    rates: dict  # {country: {year: per cent}}, OECD long-term government bond yield, annual average
    wages: dict  # {country: {year: constant-price average annual wage}}
    qx: dict  # {(geo, year, sex): {age: probability of dying within the year}}
    e65: dict  # {(geo, year, sex): life expectancy at 65}
    hicp: dict  # {geo: {year: index}}, Eurostat HICP, annual average, 2015 = 100
    pt_updates: list  # rows of data/portugal_updates.csv
    pt_coefficients: dict  # {year: coefficient}, official table for the lowest bracket


def load():
    life = read_rows("life_tables")
    qx, e65 = {}, {}
    for r in life:
        k = (r["geo"], int(r["year"]), r["sex"])
        if r["indicator"] == "qx":
            qx.setdefault(k, {})[r["age"]] = float(r["value"])
        else:
            e65[k] = float(r["value"])
    coef = {int(r["year"]): float(r["coefficient"]) for r in read_rows("portugal_first_bracket_coefficients")}
    return Data(
        cpi=series(read_rows("cpi"), "country"),
        rates=series(read_rows("long_rates"), "country"),
        wages=series(read_rows("wages"), "country"),
        qx=qx,
        e65=e65,
        hicp=series(read_rows("hicp"), "geo"),
        pt_updates=read_rows("portugal_updates"),
        pt_coefficients=coef,
    )


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


# ---------------------------------------------------------------------------------------------------- survival
def survival(qx, age=AGE, n=SPAN):
    """Probability of being alive at age, age + 1, ..., age + n - 1, given alive at age."""
    out, s = [], 1.0
    for k in range(n):
        out.append(s)
        s *= 1.0 - qx[f"Y{age + k}"] if age + k < 95 else 1.0 - qx["Y_GE95"]
    return out


def curtate_check(qx, age=AGE):
    """Expected years lived from age by the life table, counting to 95 and the open group after it."""
    s, total = 1.0, 0.0
    for a in range(age, 95):
        q = qx[f"Y{a}"]
        total += s * (1.0 - q / 2.0)
        s *= 1.0 - q
    return total, s  # years lived 65-94, and probability of reaching 95


# ---------------------------------------------------------------------------------------------------- price paths
def complete(prices, start, n):
    return all(start + k in prices for k in range(n))


def real_path(prices, start, n):
    """Real value of a fixed nominal payment set in year start: P[start] / P[start + k], k = 0..n-1."""
    p0 = prices[start]
    return [p0 / prices[start + k] for k in range(n)]


def windows(cpi, countries, n, first=FIRST_START, last=None):
    """(country, start) pairs with n consecutive years of prices from start."""
    out = []
    for c in countries:
        years = sorted(y for y in cpi.get(c, {}) if y >= first)
        for s in years:
            if (last is None or s <= last) and complete(cpi[c], s, n):
                out.append((c, s))
    return out


def average_inflation(prices, start, years):
    """Geometric mean annual inflation from start to start + years."""
    return (prices[start + years] / prices[start]) ** (1.0 / years) - 1.0


def weighted_real(path, weights):
    """Survival-weighted average real value of the payments: 1 minus the share of lifetime real income lost."""
    return math.fsum(w * v for w, v in zip(weights, path, strict=True)) / math.fsum(weights)


# ---------------------------------------------------------------------------------------------------- indexation rules
def _clip(lo, hi):
    return lambda p: min(max(p, lo), hi)


RULES = {
    "fixed": lambda p: 0.0,
    "full": lambda p: p,
    "cap5": _clip(0.0, 0.05),
    "cap3": _clip(0.0, 0.03),
    "cap25": _clip(0.0, 0.025),
    "half": lambda p: max(p, 0.0) / 2.0,
    "minus05": lambda p: max(p - 0.005, 0.0),
    "step2": lambda p: 0.02,
    "step3": lambda p: 0.03,
}
RULE_LABELS = {
    "fixed": "No indexation",
    "full": "Full price indexation",
    "cap5": "Prices, capped at 5%",
    "cap3": "Prices, capped at 3%",
    "cap25": "Prices, capped at 2.5%",
    "half": "Half of price inflation",
    "minus05": "Prices minus 0.5 point",
    "step2": "Fixed rise of 2% a year",
    "step3": "Fixed rise of 3% a year",
}


def indexed_path(prices, start, n, rule):
    """Real value of a payment raised each year by rule(inflation of that year), starting at 1."""
    p0, pay, out = prices[start], 1.0, [1.0]
    for k in range(1, n):
        inflation = prices[start + k] / prices[start + k - 1] - 1.0
        pay *= 1.0 + RULES[rule](inflation)
        out.append(pay * p0 / prices[start + k])
    return out


# ---------------------------------------------------------------------------------------------------- annuities
def annuity_factor(weights, rate, growth=0.0):
    """Expected present value of a payment of 1 a year at each age, growing at growth, paid while alive."""
    return math.fsum(w * ((1.0 + growth) / (1.0 + rate)) ** k for k, w in enumerate(weights))


def first_payment(premium, weights, rate, growth=0.0):
    return premium / annuity_factor(weights, rate, growth)


def level_value(prices, start, nominal, weights, real_rate=REAL_RATE):
    """Real value, per unit of premium, of a level annuity bought in year start at nominal yield nominal.

    The indexed annuity bought with the same premium, priced at real_rate, is worth exactly 1 on the same scale, so
    the result minus 1 is what the level buyer gained (positive) or lost (negative) from inflation surprises.
    """
    pay = first_payment(1.0, weights, nominal)
    p0 = prices[start]
    return pay * math.fsum(w * (p0 / prices[start + k]) / (1.0 + real_rate) ** k for k, w in enumerate(weights))


def breakeven(nominal, real_rate=REAL_RATE):
    """Inflation priced into the nominal yield if the real rate is real_rate."""
    return (1.0 + nominal) / (1.0 + real_rate) - 1.0


# ---------------------------------------------------------------------------------------------------- Portugal
def pt_update_rows(rows, year, when):
    return [r for r in rows if int(r["year"]) == year and r["when"] == when]


def pt_bracket(rows, amount):
    for r in rows:
        lo = float(r["lower"]) if r["lower"] else -math.inf
        hi = float(r["upper"]) if r["upper"] else math.inf
        if lo < amount <= hi:
            return r
    raise ValueError(f"no bracket for {amount}")


def pt_pension(rows, start_amount, first_year=2008, last_year=2026):
    """Monthly amount of a Portuguese general-regime pension, year by year, from the published annual updates.

    Returns {year: (january amount, amount from july, average monthly amount over the year)}. The percentage of the
    bracket the pension is in applies, with the minimum increase in euros where the ordinance sets one. The interim
    update of July 2023 is applied to the December 2022 amount, as Decree-Law 28/2023 provides. Fixed-euro
    extraordinary rises (for pensions up to 1.5 IAS), one-off payments and temporary cuts are not included.
    """
    out = {first_year: (start_amount, start_amount, start_amount)}
    amount = start_amount
    for y in range(first_year + 1, last_year + 1):
        december = amount
        jan = pt_update_rows(rows, y, "jan")
        if not jan:
            raise ValueError(f"no January update for {y}")
        b = pt_bracket(jan, amount)
        rise = float(b["rate"]) / 100.0 * amount
        if b["minimum"]:
            rise = max(rise, float(b["minimum"]))
        amount = round(amount + rise, 2)
        january = amount
        jul = pt_update_rows(rows, y, "jul")
        if jul:
            b = pt_bracket(jul, december)
            amount = round(amount + float(b["rate"]) / 100.0 * december, 2)
        out[y] = (january, amount, (6 * january + 6 * amount) / 12.0)
    return out


def _lowest(rows):
    return min(rows, key=lambda r: float(r["upper"]) if r["upper"] else math.inf)


def pt_rate(row, regular=False):
    """Rate of a bracket in per cent; regular=True leaves out the extraordinary 1.25 points of 2025."""
    return float(row["regular_rate"] if regular and row["regular_rate"] else row["rate"])


def pt_first_bracket(rows, first_year=2008, last_year=2026, regular=False):
    """Year-on-year factors of the lowest bracket's percentage updates, {year: factor}.

    The interim rise of July 2023 was a percentage of the December 2022 amount, so in that year the two rates add.
    """
    out = {}
    for y in range(first_year + 1, last_year + 1):
        factor = 1.0 + pt_rate(_lowest(pt_update_rows(rows, y, "jan")), regular) / 100.0
        jul = pt_update_rows(rows, y, "jul")
        if jul:
            factor += pt_rate(_lowest(jul), regular) / 100.0
        out[y] = factor
    return out


def cumulative(factors, first_year):
    out, idx = {first_year: 1.0}, 1.0
    for y in sorted(factors):
        idx *= factors[y]
        out[y] = idx
    return out
