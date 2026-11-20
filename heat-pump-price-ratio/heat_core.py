"""Shared definitions for the heat pump price-ratio analysis.

A heat pump turns one kWh of electricity into SPF kWh of heat, where SPF is its seasonal performance factor.
A gas boiler turns one kWh of gas (gross calorific value) into ETA kWh of heat. Per kWh of heat, the heat pump
costs p_e / SPF and the boiler costs p_g / ETA, so the heat pump is cheaper to run exactly when

    ratio = p_e / p_g  <  SPF / ETA.

The ratio decides the sign of the saving. The gas price decides its size:

    saving per kWh of heat = p_g / ETA - p_e / SPF = p_g * (1 / ETA - ratio / SPF).

Half-years are integers: year * 2 + semester - 1, so 2026-S1 is 4052.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# EU members with household gas prices in Eurostat's band data, and the United Kingdom.
# Cyprus, Finland and Malta are absent: they report no household gas prices.
PANEL = tuple("AT BE BG CZ DE DK EE EL ES FR HR HU IE IT LT LU LV NL PL PT RO SE SI SK UK".split())
NAMES = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "EL": "Greece",
    "ES": "Spain",
    "FR": "France",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "UK": "United Kingdom",
    "EU27_2020": "EU (27)",
}

FIRST = "2007-S2"
LAST = "2026-S1"
ELECTRICITY_BAND = "KWH5000-14999"  # band DD: a home that also heats with electricity
ELECTRICITY_SMALL = "KWH2500-4999"  # band DC: a home that does not
GAS_BAND = "GJ20-199"  # band D2: a home that heats with gas
ETA = 0.90  # seasonal efficiency of a condensing gas boiler, gross calorific value
HEAT = 10_000.0  # kWh of heat a year for the reference home
FIELD_SPF = {"UK field median (EoH)": 2.80, "German field mean (Fraunhofer ISE)": 3.40}


def half(label):
    year, s = label.split("-S")
    return int(year) * 2 + int(s) - 1


def label(h):
    return f"{h // 2}-S{h % 2 + 1}"


def quantile(values, q):
    """Type 7 quantile (linear interpolation), as in R's default and numpy's."""
    v = sorted(values)
    if not v:
        raise ValueError("no values")
    pos = (len(v) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def share(flags):
    flags = list(flags)
    return sum(1 for f in flags if f) / len(flags)


def breakeven_spf(ratio, eta=ETA):
    """The seasonal performance factor at which a heat pump costs the same to run as the boiler."""
    return ratio * eta


def saving(p_e, p_g, spf, eta=ETA, heat=HEAT):
    """Euros a year saved by heating with the heat pump instead of the boiler; negative means it costs more."""
    return heat * (p_g / eta - p_e / spf)


def read_prices(name):
    """{(country, band, tax): {half: (value, flag)}} from data/<name>.csv."""
    out = {}
    with open(DATA / f"{name}.csv", newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault((r["country"], r["band"], r["tax"]), {})[half(r["half"])] = (
                float(r["value"]),
                r["flag"],
            )
    return out


def read_hdd():
    out = {}
    with open(DATA / "hdd.csv", newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["country"], {})[int(r["year"])] = float(r["value"])
    return out


@dataclass(frozen=True)
class Point:
    """One country in one half-year."""

    country: str
    half: int
    p_e: float
    p_g: float

    @property
    def ratio(self):
        return self.p_e / self.p_g

    def cheaper(self, spf, eta=ETA):
        return self.ratio < spf / eta

    def saving(self, spf, eta=ETA, heat=HEAT):
        return saving(self.p_e, self.p_g, spf, eta, heat)


def points(electricity, gas, e_band=ELECTRICITY_BAND, tax="I_TAX", countries=PANEL):
    """Every country and half-year with both prices, in order."""
    out = []
    for c in countries:
        e = electricity.get((c, e_band, tax), {})
        g = gas.get((c, GAS_BAND, tax), {})
        for h in sorted(set(e) & set(g)):
            if half(FIRST) <= h <= half(LAST):
                out.append(Point(c, h, e[h][0], g[h][0]))
    return out


def lifetime(series, start, halves, spf, eta=ETA, heat=HEAT):
    """Saving over `halves` consecutive half-years from `start`, at the prices of each half-year.

    `series` maps half-year to Point. Each half-year counts for half of the year's heat. Returns None when a
    half-year is missing.
    """
    total = []
    for h in range(start, start + halves):
        if h not in series:
            return None
        total.append(series[h].saving(spf, eta, heat) / 2)
    return math.fsum(total)
