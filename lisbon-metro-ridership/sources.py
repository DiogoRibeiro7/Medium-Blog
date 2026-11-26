"""The downloads behind the analysis.

* INE, Statistics Portugal (contract ``ine-portugal-metro-ridership``):
  - the monthly metro survey (Inquérito ao transporte por metropolitano): passengers, passenger-km and
    place-km offered for the three systems (Metropolitano de Lisboa, Metro do Porto, Metro Sul do Tejo),
    January 2004 to August 2026;
  - passengers embarked and disembarked at Lisbon airport, monthly, January 2015 to August 2026;
  - passengers of regular bus services by NUTS II region of origin, 2023 to 2025;
  - passengers on the river lines, 2019 and 2023 to 2025.
  Every query names its periods, so a new month published by INE does not change the response.
* INE responses already pinned in the data registry (contract ``ine-portugal-municipal-infrastructure``):
  resident population by municipality at the end of 2024 and 2025.
* Eurostat (contract ``eurostat-regional-employment-pt``): employed persons aged 15-74 by NUTS II region.
"""

from __future__ import annotations

EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
INE_API = "https://www.ine.pt/ine/json_indicador/pindica.jsp"

LAST_MONTH = (2026, 8)  # last month of the metro survey published when the data were retrieved


def months(first: tuple[int, int], last: tuple[int, int]) -> str:
    """INE month codes S3AYYYYMM from first to last, comma-separated."""
    out = []
    y, m = first
    while (y, m) <= last:
        out.append(f"S3A{y}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return ",".join(out)


def years(*ys: int) -> str:
    return ",".join(f"S7A{y}" for y in ys)


METRO_MONTHS = months((2004, 1), LAST_MONTH)
AIRPORT_MONTHS = months((2015, 1), LAST_MONTH)

# (label, indicator, dimension filters, what it is)
INE_NEW: list[tuple[str, str, dict[str, str], str]] = [
    ("metro_passengers", "0000898", {"Dim1": METRO_MONTHS}, "Passengers carried by metro system, monthly (thousands)"),
    ("metro_passenger_km", "0000899", {"Dim1": METRO_MONTHS}, "Passenger-km by metro system, monthly (thousands)"),
    ("metro_place_km", "0000900", {"Dim1": METRO_MONTHS}, "Place-km offered by metro system, monthly (thousands)"),
    (
        "airport_lisbon_embarked",
        "0000861",
        {"Dim1": AIRPORT_MONTHS, "Dim2": "LPPT", "Dim3": "T"},
        "Passengers embarked at Lisbon airport (LPPT), all traffic, monthly",
    ),
    (
        "airport_lisbon_disembarked",
        "0000862",
        {"Dim1": AIRPORT_MONTHS, "Dim2": "LPPT", "Dim3": "T"},
        "Passengers disembarked at Lisbon airport (LPPT), all traffic, monthly",
    ),
    (
        "bus_passengers",
        "0013815",
        {"Dim1": years(2023, 2024, 2025)},
        "Passengers of road passenger transport companies by NUTS II region of origin and type of service (thousands)",
    ),
    ("river_passengers", "0000896", {"Dim1": years(2019, 2023, 2024, 2025)}, "Passengers on inland waterway lines, by line"),
]

# Labels of responses pinned in data_sources/ine-portugal-municipal-infrastructure.json
INE_SHARED: list[str] = ["pop_est_2024", "pop_est_2025"]

# (label, dataset, query, what it is)
EUROSTAT: list[tuple[str, str, str, str]] = [
    (
        "employment_nuts2",
        "lfst_r_lfe2emp",
        "geo=PT&geo=PT1A&geo=PT1B&sex=T&age=Y15-74&sinceTimePeriod=2019",
        "Employed persons aged 15-74 by NUTS 2 region, thousands",
    ),
]


def ine_url(code: str, dims: dict[str, str]) -> str:
    params = "&".join(f"{k}={v}" for k, v in dims.items())
    return f"{INE_API}?op=2&varcd={code}&{params}&lang=PT"


def eurostat_url(dataset: str, query: str) -> str:
    return f"{EUROSTAT_API}/{dataset}?format=JSON&lang=EN&{query}"
