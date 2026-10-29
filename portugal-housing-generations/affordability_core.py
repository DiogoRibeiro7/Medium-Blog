"""Pure functions for the regional affordability panel (no files, no network).

Prices here are medians per square metre and incomes are medians per tax
household or taxpayer. Medians do not add up across regions, so these functions
only ever combine values that INE published for the same region and year.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

Panel = Mapping[tuple[str, int], float]

_QUARTER = re.compile(r"^\s*(\d)\.?\s*º?\s*Trimestre de (\d{4})\s*$", re.I)
_QUARTER_CODE = re.compile(r"^S5A(\d{4})([1-4])$")
_YEAR_CODE = re.compile(r"^S7A(\d{4})$")


def period_year(label: str) -> tuple[int, int | None]:
    """Parse an INE period label or code: '2021' -> (2021, None); '4.º Trimestre de 2019' or 'S5A20194' -> (2019, 4)."""
    text = label.strip()
    if text.isdigit() and len(text) == 4:
        return int(text), None
    for pattern, year_group, quarter_group in ((_QUARTER, 2, 1), (_QUARTER_CODE, 1, 2)):
        match = pattern.match(text)
        if match:
            quarter = int(match.group(quarter_group))
            if not 1 <= quarter <= 4:
                break
            return int(match.group(year_group)), quarter
    match = _YEAR_CODE.match(text)
    if match:
        return int(match.group(1)), None
    raise ValueError(f"Unrecognised period label: {label!r}")


def m2_per_income_year(price_per_m2: Panel, income: Panel) -> dict[tuple[str, int], float]:
    """Square metres of a median-priced dwelling that one year of median income would pay for.

    A ratio of two medians from different populations (dwellings sold, tax
    households): a descriptive proxy, not the floor area any household could buy.
    """
    result: dict[tuple[str, int], float] = {}
    for key in sorted(price_per_m2.keys() & income.keys()):
        if price_per_m2[key] <= 0 or income[key] <= 0:
            raise ValueError(f"Non-positive price or income for {key}")
        result[key] = income[key] / price_per_m2[key]
    return result


def price_income_index(price: Panel, income: Panel, region: str, base: int, years: Iterable[int]) -> dict[int, float]:
    """(P_t / P_base) / (Y_t / Y_base) * 100 for one region; years missing an input are left out."""
    if (region, base) not in price or (region, base) not in income:
        raise ValueError(f"No base-year {base} values for {region}")
    out: dict[int, float] = {}
    for year in years:
        if (region, year) in price and (region, year) in income:
            out[year] = 100 * (price[(region, year)] / price[(region, base)]) / (income[(region, year)] / income[(region, base)])
    return out


def annual_growth(series: Panel, region: str, start: int, end: int) -> float | None:
    """Compound annual growth in percent between two years, or None if either is missing."""
    if end <= start:
        raise ValueError("end must be after start")
    if (region, start) not in series or (region, end) not in series:
        return None
    return 100 * ((series[(region, end)] / series[(region, start)]) ** (1 / (end - start)) - 1)


def log_points(series: Panel, region: str, start: int, end: int) -> float | None:
    """100 * ln(X_end / X_start), so price minus income log points equals the change in the ratio."""
    if (region, start) not in series or (region, end) not in series:
        return None
    return 100 * math.log(series[(region, end)] / series[(region, start)])


def growth_comparison(price: Panel, income: Panel, region: str, start: int, end: int) -> dict[str, float] | None:
    """Price and income growth of one region over the same years, or None if any of the four values is missing.

    ``gap_log_points`` is price minus income growth in log points; it equals
    100 * the log change of the price/income ratio, so a positive gap means
    prices rose faster than incomes.
    """
    keys = [(region, start), (region, end)]
    if any(k not in price or k not in income for k in keys):
        return None
    p, y = log_points(price, region, start, end), log_points(income, region, start, end)
    pa, ya = annual_growth(price, region, start, end), annual_growth(income, region, start, end)
    assert p is not None and y is not None and pa is not None and ya is not None
    return {"price_annual_pct": pa, "income_annual_pct": ya, "price_log_points": p, "income_log_points": y,
            "gap_log_points": p - y, "ratio_change_pct": 100 * (math.exp((p - y) / 100) - 1)}


def median(values: Iterable[float]) -> float:
    """Median of a non-empty collection."""
    data = sorted(values)
    if not data:
        raise ValueError("Empty input")
    mid = len(data) // 2
    return data[mid] if len(data) % 2 else (data[mid - 1] + data[mid]) / 2


def ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Average ranks (1 = smallest), ties sharing the mean rank."""
    ordered = sorted(values, key=lambda k: values[k])
    result: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        for k in ordered[i:j + 1]:
            result[k] = (i + j) / 2 + 1
        i = j + 1
    return result


def spearman(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Spearman rank correlation over the keys present in both mappings."""
    keys = sorted(a.keys() & b.keys())
    if len(keys) < 3:
        raise ValueError("Need at least three common keys")
    ra, rb = ranks({k: a[k] for k in keys}), ranks({k: b[k] for k in keys})
    return pearson([ra[k] for k in keys], [rb[k] for k in keys])


def pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation of two equal-length lists."""
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("Need two lists of equal length, at least three")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        raise ValueError("Constant input")
    return sxy / math.sqrt(sxx * syy)


def dispersion(values: Iterable[float]) -> dict[str, float]:
    """Coefficient of variation and max/min ratio across regions."""
    data = list(values)
    if len(data) < 2 or min(data) <= 0:
        raise ValueError("Need at least two positive values")
    mean = sum(data) / len(data)
    cv = math.sqrt(sum((v - mean) ** 2 for v in data) / len(data)) / mean
    return {"cv": cv, "max_over_min": max(data) / min(data), "n": float(len(data))}
