"""Pure functions for the rental-market analysis (no files, no network).

Eurostat JSON-stat selection, index rebasing, and a shift-share decomposition
that separates changes in tenure composition from changes within each tenure
group. Nothing here fills a gap: a year missing any input is left out.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def jsonstat_select(document: Mapping[str, Any], fixed: Mapping[str, str], over: str = "time") -> tuple[dict[str, float], dict[str, str]]:
    """Values along one dimension of a JSON-stat dataset with every other dimension fixed.

    ``fixed`` must name a category for every dimension except ``over`` that has
    more than one category; single-category dimensions may be omitted. Returns
    (category -> value, category -> Eurostat status flag) for the cells present.
    """
    ids = document["id"]
    sizes = document["size"]
    if len(ids) != len(sizes) or over not in ids:
        raise ValueError(f"Invalid JSON-stat structure or unknown dimension {over!r}")
    index = {d: document["dimension"][d]["category"]["index"] for d in ids}
    index = {d: (dict(zip(v, range(len(v)), strict=True)) if isinstance(v, list) else v) for d, v in index.items()}
    for d in fixed:
        if d not in ids:
            raise ValueError(f"Unknown dimension {d!r}")
    coords: dict[str, int] = {}
    for d in ids:
        if d == over:
            continue
        if d in fixed:
            if fixed[d] not in index[d]:
                raise ValueError(f"Dimension {d!r} has no category {fixed[d]!r}")
            coords[d] = index[d][fixed[d]]
        elif len(index[d]) == 1:
            coords[d] = 0
        else:
            raise ValueError(f"Dimension {d!r} has {len(index[d])} categories; choose one")
    values = document.get("value", {})
    status = document.get("status", {})
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}
    out: dict[str, float] = {}
    flags: dict[str, str] = {}
    for category, pos in index[over].items():
        flat = 0
        for d, size in zip(ids, sizes, strict=True):
            flat = flat * size + (pos if d == over else coords[d])
        value = values.get(str(flat))
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid value at {category}")
        out[category] = float(value)
        if isinstance(status, Mapping) and str(flat) in status:
            flags[category] = str(status[str(flat)])
    return out, flags


def years(series: Mapping[str, float]) -> dict[int, float]:
    """Re-key an annual JSON-stat series by integer year (rejects non-annual labels)."""
    out = {}
    for label, value in series.items():
        if not (label.isdigit() and len(label) == 4):
            raise ValueError(f"Not an annual period: {label!r}")
        out[int(label)] = value
    return dict(sorted(out.items()))


def rebase(series: Mapping[int, float], base: int) -> dict[int, float]:
    """Series divided by its base-year value, times 100."""
    if base not in series or series[base] <= 0:
        raise ValueError(f"No positive base-year value for {base}")
    return {y: 100 * v / series[base] for y, v in sorted(series.items())}


def relative_index(numerator: Mapping[int, float], denominator: Mapping[int, float], base: int) -> dict[int, float]:
    """(N_t / N_base) / (D_t / D_base) * 100 over the years both series cover."""
    a, b = rebase(numerator, base), rebase(denominator, base)
    return {y: 100 * a[y] / b[y] for y in sorted(a.keys() & b.keys())}


def shift_share(shares_a: Mapping[str, float], rates_a: Mapping[str, float],
                shares_b: Mapping[str, float], rates_b: Mapping[str, float]) -> dict[str, float]:
    """Split the change in a population rate into composition and within-group parts.

    The aggregate is sum_g share_g * rate_g / 100 (shares in percent of the
    population). With mean shares and mean rates as weights the two parts add up
    exactly to the change (Kitagawa decomposition).
    """
    groups = sorted(shares_a)
    if set(groups) != set(rates_a) or set(groups) != set(shares_b) or set(groups) != set(rates_b):
        raise ValueError("Groups differ between inputs")
    total_a = sum(shares_a[g] * rates_a[g] for g in groups) / 100
    total_b = sum(shares_b[g] * rates_b[g] for g in groups) / 100
    within = sum((shares_a[g] + shares_b[g]) / 2 * (rates_b[g] - rates_a[g]) for g in groups) / 100
    composition = sum((rates_a[g] + rates_b[g]) / 2 * (shares_b[g] - shares_a[g]) for g in groups) / 100
    return {"implied_a": total_a, "implied_b": total_b, "change": total_b - total_a,
            "within_groups": within, "composition": composition}


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean of a non-empty collection."""
    data = list(values)
    if not data:
        raise ValueError("Empty input")
    return sum(data) / len(data)
