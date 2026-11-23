"""The downloads behind the analysis.

Three groups of public responses:

* Eurostat, city statistics (Urban Audit): population, jobs at the place of work and dwellings, for
  each city and for its functional urban area (FUA). Six full-indicator downloads.
* INE, Statistics Portugal, already pinned in the data registry for portugal-infrastructure-saturation
  (contract ``ine-portugal-municipal-infrastructure``). They are downloaded again here and checked
  against that contract.
* INE responses used only here (contract ``ine-portugal-centre-users-homes``): the census share of
  people entering each municipality to work or study, full-time employees by place of work, beds in tourist
  accommodation, and dwellings completed in new residential construction.
"""

from __future__ import annotations

EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
INE_API = "https://www.ine.pt/ine/json_indicador/pindica.jsp"

# (label, dataset, indicator, what it is)
EUROSTAT: list[tuple[str, str, str, str]] = [
    ("eu_pop_city", "urb_cpop1", "DE1001V", "Population on 1 January, cities"),
    ("eu_pop_fua", "urb_lpop1", "DE1001V", "Population on 1 January, functional urban areas"),
    ("eu_jobs_city", "urb_clma", "EC2020V", "Total employment (jobs), workplace based, cities"),
    ("eu_jobs_fua", "urb_llma", "EC2020V", "Total employment (jobs), workplace based, functional urban areas"),
    ("eu_dwellings_city", "urb_clivcon", "SA1001V", "Number of conventional dwellings, cities"),
    ("eu_dwellings_fua", "urb_llivcon", "SA1001V", "Number of conventional dwellings, functional urban areas"),
]

# Labels of responses pinned in data_sources/ine-portugal-municipal-infrastructure.json
INE_SHARED: list[str] = [
    "pop_est_2011",
    "pop_est_2021",
    "pop_est_2024",
    "pop_est_2025",
    "census21_pop_age",
    "census21_dwellings_occupancy",
    "census21_households_tenure",
    "census21_crowding",
    "census_dwellings_change",
    "census_households_change",
    "sale_price_2019",
    "sale_price_2025",
    "new_rent_2021",
    "new_rent_2025",
    "commute_out_share",
    "school_pre_2025",
    "school_basic_2025",
    "hospital_beds_sns_2024",
]

# (label, indicator, dimension filters, what it is)
INE_NEW: list[tuple[str, str, dict[str, str], str]] = [
    ("commute_in_share", "0011654", {"Dim1": "S7A2021"}, "Census 2021 people entering the municipality to work or study, % of residents"),
    *[
        (
            f"employees_{y}",
            "0012650",
            {"Dim1": f"S7A{y}"},
            f"Full-time employees on full pay by place of work and size of establishment, {y} (Quadros de Pessoal)",
        )
        for y in (2021, 2024)
    ],
    *[
        (f"tourist_beds_{y}", "0013366", {"Dim1": f"S7A{y}", "Dim3": "T"}, f"Bed capacity in tourist accommodation establishments, {y}")
        for y in (2023, 2025)
    ],
    *[
        (f"completions_{y}", "0013249", {"Dim1": f"S7A{y}", "Dim3": "T"}, f"Dwellings completed in new construction for family housing, {y}")
        for y in (2022, 2023, 2024, 2025)
    ],
]


def eurostat_url(dataset: str, indicator: str) -> str:
    return f"{EUROSTAT_API}/{dataset}?format=JSON&lang=EN&indic_ur={indicator}"


def ine_url(code: str, dims: dict[str, str]) -> str:
    params = "&".join(f"{k}={v}" for k, v in dims.items())
    return f"{INE_API}?op=2&varcd={code}&{params}&lang=PT"
