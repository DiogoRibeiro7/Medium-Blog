"""Definitions for reproducible, comparable EU-27 quarterly series."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlencode

API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
EU27: tuple[str, ...] = (
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR",
    "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK",
)
QUARTER_RE = re.compile(r"^(?:19|20)\d{2}-Q[1-4]$")


class EurostatDataError(ValueError):
    """An unexpected dataset schema or incompatible time series."""


@dataclass(frozen=True)
class SeriesSpec:
    """One fully specified Eurostat indicator."""

    name: str
    dataset: str
    column: str
    filters: Mapping[str, str]


SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("gdp", "namq_10_gdp", "gdp_growth_qoq_pct", {
        "freq": "Q", "unit": "CLV_PCH_PRE", "s_adj": "SCA", "na_item": "B1GQ",
    }),
    SeriesSpec("unemployment", "une_rt_q", "unemployment_rate_pct", {
        "freq": "Q", "age": "Y15-74", "sex": "T", "unit": "PC_ACT", "s_adj": "SA",
    }),
    SeriesSpec("participation", "lfsi_emp_q", "participation_rate_pct", {
        "freq": "Q", "indic_em": "ACT", "age": "Y15-74", "sex": "T", "unit": "PC_POP",
        "s_adj": "SA",
    }),
)


@dataclass(frozen=True)
class Observation:
    """A numeric observation, or a missing cell, and its published quality flag."""

    value: float | None
    status: str


def quarter_sequence(start: str, end: str) -> list[str]:
    """Generate all quarterly period identifiers, including both endpoints."""
    if not QUARTER_RE.fullmatch(start) or not QUARTER_RE.fullmatch(end):
        raise ValueError("Quarters must have the format YYYY-Q[1-4].")
    first = int(start[:4]) * 4 + int(start[-1]) - 1
    last = int(end[:4]) * 4 + int(end[-1]) - 1
    if last < first:
        raise ValueError("End quarter cannot precede start quarter.")
    return [f"{q // 4:04d}-Q{q % 4 + 1}" for q in range(first, last + 1)]


def request_url(spec: SeriesSpec, start: str, end: str) -> str:
    """Build a filtered Eurostat Statistics API query."""
    quarter_sequence(start, end)
    params = {"lang": "EN", **spec.filters, "sinceTimePeriod": start, "untilTimePeriod": end}
    return f"{API_BASE}/{spec.dataset}?{urlencode(params)}"
