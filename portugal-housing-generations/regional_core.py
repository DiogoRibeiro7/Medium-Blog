"""Pure functions for the regional panel: regrouping, rates, growth and coverage.

Nothing here reads files or knows about INE's response format; ``regional.py``
does that and passes plain dictionaries in. Every function refuses to fill a gap:
a region-year with any member municipality missing is missing, and a rate or a
growth figure with a missing input is left out rather than estimated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

Series = dict[tuple[str, int], float]


def regroup(values: Mapping[tuple[str, int], float], members: Mapping[str, Iterable[str]],
            years: Iterable[int]) -> tuple[Series, dict[tuple[str, int], list[str]]]:
    """Sum municipal values into regions for each year.

    ``members`` maps a region code to the municipality codes it contains. A
    region-year is produced only when every member has a value; otherwise the
    missing members are reported and the region-year is left out.
    """
    totals: Series = {}
    gaps: dict[tuple[str, int], list[str]] = {}
    for region, municipalities in members.items():
        codes = sorted(municipalities)
        if not codes:
            raise ValueError(f"Region {region} has no member municipalities")
        for year in years:
            missing = [code for code in codes if (code, year) not in values]
            if missing:
                gaps[(region, year)] = missing
                continue
            totals[(region, year)] = sum(values[(code, year)] for code in codes)
    return totals, gaps


def compare_totals(computed: Mapping[tuple[str, int], float], published: Mapping[tuple[str, int], float],
                   tolerance: float = 0.0) -> list[tuple[str, int, float, float]]:
    """Return (region, year, computed, published) wherever both exist and differ by more than ``tolerance``."""
    return [
        (region, year, computed[(region, year)], published[(region, year)])
        for region, year in sorted(computed.keys() & published.keys())
        if abs(computed[(region, year)] - published[(region, year)]) > tolerance
    ]


def per_thousand(numerator: Mapping[tuple[str, int], float], population: Mapping[tuple[str, int], float]) -> Series:
    """Numerator per 1,000 residents where both values exist and population is positive."""
    result: Series = {}
    for key in sorted(numerator.keys() & population.keys()):
        if population[key] <= 0:
            raise ValueError(f"Non-positive population for {key}")
        result[key] = 1000 * numerator[key] / population[key]
    return result


def growth(series: Mapping[tuple[str, int], float], region: str, start: int, end: int) -> dict[str, float] | None:
    """Total and compound annual growth between two years, in percent, or None if either is missing."""
    if end <= start:
        raise ValueError("end must be after start")
    if (region, start) not in series or (region, end) not in series:
        return None
    first, last = series[(region, start)], series[(region, end)]
    if first <= 0 or last <= 0:
        raise ValueError(f"Growth needs positive values for {region}")
    return {
        "total_pct": 100 * (last / first - 1),
        "annual_pct": 100 * ((last / first) ** (1 / (end - start)) - 1),
    }


def period_mean(series: Mapping[tuple[str, int], float], region: str, start: int, end: int) -> float | None:
    """Mean over every year from ``start`` to ``end`` inclusive, or None if any year is missing."""
    years = range(start, end + 1)
    if any((region, year) not in series for year in years):
        return None
    return sum(series[(region, year)] for year in years) / len(years)


def coefficient_of_variation(values: Iterable[float]) -> float:
    """Population standard deviation divided by the mean (dispersion across regions)."""
    data = list(values)
    if len(data) < 2:
        raise ValueError("Need at least two values")
    mean = sum(data) / len(data)
    if mean <= 0:
        raise ValueError("Mean must be positive")
    return math.sqrt(sum((x - mean) ** 2 for x in data) / len(data)) / mean


def coverage(series: Mapping[tuple[str, int], float], regions: Iterable[str], years: Iterable[int]) -> dict[str, object]:
    """Count observed and missing region-years and list the missing ones."""
    region_list, year_list = sorted(regions), sorted(years)
    missing = [(r, y) for r in region_list for y in year_list if (r, y) not in series]
    expected = len(region_list) * len(year_list)
    return {
        "expected": expected,
        "observed": expected - len(missing),
        "missing": len(missing),
        "missing_cells": [f"{r}:{y}" for r, y in missing],
        "complete_years": [y for y in year_list if all((r, y) in series for r in region_list)],
    }


def longest_complete_span(years: Iterable[int]) -> tuple[int, int]:
    """Longest run of consecutive years in ``years``."""
    ordered = sorted(set(years))
    if not ordered:
        raise ValueError("No years")
    best = (ordered[0], ordered[0])
    start = prev = ordered[0]
    for year in ordered[1:]:
        if year != prev + 1:
            start = year
        prev = year
        if prev - start > best[1] - best[0]:
            best = (start, prev)
    return best
