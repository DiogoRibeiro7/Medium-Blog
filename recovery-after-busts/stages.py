"""EU integration stages by country and year, used to code bust episodes.

Stage at a given year:

- ``pre-accession``: before the country joined the EEC/EU (for the founding six, before the
  customs union was complete on 1 July 1968);
- ``customs union``: member, before the single market (1993; 1994 for AT, FI, SE, which
  entered it through the EEA a year before accession);
- ``single market``: from 1993 (or accession, if later) until euro adoption;
- ``euro``: from euro adoption;
- ``outside``: non-members in the comparison group, and the UK from 2021.
"""

from __future__ import annotations

OUTSIDE = {"US", "JP", "CA", "AU", "NZ", "CH", "NO", "IS"}
CEE = {"BG", "CZ", "EE", "HR", "HU", "LT", "LV", "PL", "RO", "SI", "SK"}

ACCESSION = {
    "BE": 1958, "DE": 1958, "FR": 1958, "IT": 1958, "LU": 1958, "NL": 1958,
    "DK": 1973, "IE": 1973, "UK": 1973, "EL": 1981, "ES": 1986, "PT": 1986,
    "AT": 1995, "FI": 1995, "SE": 1995,
    "CY": 2004, "CZ": 2004, "EE": 2004, "HU": 2004, "LV": 2004, "LT": 2004, "MT": 2004, "PL": 2004, "SK": 2004, "SI": 2004,
    "BG": 2007, "RO": 2007, "HR": 2013,
}
# first year inside the single market: 1993, or accession if later; AT, FI and SE entered
# through the EEA in 1994, a year before they joined
SINGLE_MARKET = {c: max(1993, a) for c, a in ACCESSION.items()} | {"AT": 1994, "FI": 1994, "SE": 1994}
EURO = {
    "AT": 1999, "BE": 1999, "DE": 1999, "ES": 1999, "FI": 1999, "FR": 1999, "IE": 1999, "IT": 1999, "LU": 1999,
    "NL": 1999, "PT": 1999, "EL": 2001, "SI": 2007, "CY": 2008, "MT": 2008, "SK": 2009, "EE": 2011, "LV": 2014,
    "LT": 2015, "HR": 2023,
}
UK_LEFT = 2021  # end of the transition period


def stage(geo: str, year: int) -> str:
    if geo in OUTSIDE or (geo == "UK" and year >= UK_LEFT):
        return "outside"
    if geo in EURO and year >= EURO[geo]:
        return "euro"
    if year >= SINGLE_MARKET[geo]:
        return "single market"
    customs_union = 1968 if ACCESSION[geo] == 1958 else ACCESSION[geo]  # completed 1 July 1968
    if year >= customs_union:
        return "customs union"
    return "pre-accession"


def transition(geo: str, year: int) -> bool:
    """Central and eastern European episode with a trough before accession."""
    return geo in CEE and year < ACCESSION.get(geo, 0)
