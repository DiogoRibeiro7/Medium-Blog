"""Definitions shared by the numbers, the figures and the tests.

Output per hour is real gross value added divided by hours worked. The numerator is the same in both measures; only
the hours change. Because of that, the difference between the two productivity growth rates is exactly the
difference between the two hours growth rates, with the sign reversed.
"""

import csv
import math

import numpy as np

EU = "EU27_2020"
MEMBERS = (
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU",
    "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK",
)  # fmt: skip
NAMES = {
    EU: "EU-27", "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus", "CZ": "Czechia",
    "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland",
    "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LT": "Lithuania",
    "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta", "NL": "Netherlands", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
}  # fmt: skip
DEMO_GROUPS = tuple(f"Y{a}-{a + 4}" for a in range(15, 75, 5))
NA_FIRST = "1995-Q1"
LFS_FIRST = "2009-Q1"
LAST = "2026-Q2"
BASE_YEAR = 2009
TREND = ("1997-Q1", "2007-Q4")
BREAK = "2021-Q1"  # Integrated European Social Statistics regulation: new LFS questionnaire and definitions
COVID = ("2020-Q1", "2021-Q2")
STAGNATION = 0.5  # per cent a year: the threshold used to call productivity growth "stagnant"


def q(text):
    """'2019-Q3' -> integer quarter count, so that consecutive quarters differ by one."""
    year, quarter = text.split("-Q")
    return 4 * int(year) + int(quarter) - 1


def label(n):
    return f"{n // 4}-Q{n % 4 + 1}"


def year_of(n):
    return n // 4


def annual_mean(s, year):
    """Mean of the four quarters of a year; None unless all four are there."""
    vals = [s.get(4 * year + i) for i in range(4)]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return None
    return math.fsum(vals) / 4


def index_to(s, year=BASE_YEAR):
    base = annual_mean(s, year)
    if base is None:
        return {}
    return {k: 100 * v / base for k, v in s.items()}


def ratio(num, den):
    return {k: num[k] / den[k] for k in num if k in den}


def growth(s, y0, y1):
    """Annualised growth between two annual means, per cent a year."""
    a, b = annual_mean(s, y0), annual_mean(s, y1)
    if a is None or b is None:
        return None
    return 100 * ((b / a) ** (1 / (y1 - y0)) - 1)


def growth_ratio(num, den, y0, y1):
    """Annualised growth of an annual ratio, per cent a year: annual output over annual hours, not a mean of
    quarterly ratios, so that productivity growth is exactly output growth minus hours growth in logs."""
    a0, a1 = annual_mean(num, y0), annual_mean(num, y1)
    b0, b1 = annual_mean(den, y0), annual_mean(den, y1)
    if None in (a0, a1, b0, b1):
        return None
    return 100 * (((a1 / b1) / (a0 / b0)) ** (1 / (y1 - y0)) - 1)


def window_mean(s, end, n=4):
    vals = [s.get(end - i) for i in range(n)]
    if any(v is None for v in vals):
        return None
    return math.fsum(vals) / n


def growth_last_years(s, end, years=2):
    """Annualised growth of the four-quarter mean ending at `end` over the one ending `years` earlier."""
    a, b = window_mean(s, end - 4 * years), window_mean(s, end)
    if a is None or b is None:
        return None
    return 100 * ((b / a) ** (1 / years) - 1)


def log_trend(s, first, last):
    """Least-squares line through log values between two quarters. Returns (growth % a year, fitted function)."""
    keys = sorted(k for k in s if first <= k <= last)
    x = np.array(keys, dtype=float)
    y = np.log([s[k] for k in keys])
    slope, intercept = np.polyfit(x - first, y, 1)
    return 100 * (math.exp(4 * slope) - 1), lambda k: math.exp(intercept + slope * (k - first))


def hac_slope(x, y, lags=4):
    """OLS slope of y on x with a Newey-West (Bartlett kernel) standard error. Returns (slope, se)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        G = xu[lag:].T @ xu[:-lag]
        S += w * (G + G.T)
    bread = np.linalg.inv(X.T @ X)
    V = bread @ S @ bread
    return float(beta[1]), float(math.sqrt(V[1, 1]))


def log_change(s, a, b):
    if s.get(a) is None or s.get(b) is None:
        return None
    return math.log(s[b] / s[a])


def spearman(a, b):
    """Spearman rank correlation (no ties expected in continuous data)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ranks(values):
    """Rank 1 = highest value."""
    order = sorted(values, key=lambda k: -values[k])
    return {k: i + 1 for i, k in enumerate(order)}


def load(path, **keep):
    """{country: {quarter number or year: value}} for the rows matching `keep`, and the flags in a parallel dict."""
    values, flags = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if any(r[k] != v for k, v in keep.items()):
                continue
            p = r["period"]
            key = q(p) if "-Q" in p else int(p)
            values.setdefault(r["country"], {})[key] = float(r["value"])
            if r["flag"]:
                flags.setdefault(r["country"], {})[key] = r["flag"]
    return values, flags
