"""The INE queries behind the analysis: one entry per downloaded response.

Every query is a call to the public INE API,

    https://www.ine.pt/ine/json_indicador/pindica.jsp?op=2&varcd=<code>&<dims>&lang=PT

with the dimension filters listed below. ``label`` is the file name under ``data/raw/``.
The municipal geography differs between indicators: annual series use seven-character codes
(NUTS III 2024 code followed by the four-digit municipality code), census series use the bare
four-digit code. ``panel.py`` reduces both to the four-digit code.
"""

from __future__ import annotations

API = "https://www.ine.pt/ine/json_indicador/pindica.jsp"

# (label, indicator code, dimension filters, what it is)
QUERIES: list[tuple[str, str, dict[str, str], str]] = [
    # Population
    ("pop_est_2011", "0008273", {"Dim1": "S7A2011", "Dim3": "T", "Dim4": "T"}, "Resident population, 31 Dec 2011 (NUTS 2013)"),
    ("pop_est_2021_n13", "0008273", {"Dim1": "S7A2021", "Dim3": "T", "Dim4": "T"}, "Resident population, 31 Dec 2021 (NUTS 2013)"),
    *[
        (f"pop_est_{y}", "0012918", {"Dim1": f"S7A{y}", "Dim3": "T", "Dim4": "T"}, f"Resident population, 31 Dec {y} (NUTS 2024)")
        for y in range(2021, 2026)
    ],
    *[(f"births_{y}", "0012450", {"Dim1": f"S7A{y}", "Dim3": "T"}, f"Live births by mother's residence, {y}") for y in range(2022, 2026)],
    *[(f"deaths_{y}", "0012538", {"Dim1": f"S7A{y}", "Dim3": "T", "Dim4": "T"}, f"Deaths by residence, {y}") for y in range(2022, 2026)],
    ("census21_pop_age", "0014630", {"Dim1": "S7A2021", "Dim3": "T"}, "Census 2021 residents by life-cycle age group"),
    ("census21_migration", "0011631", {"Dim1": "S7A2021", "Dim3": "T"}, "Census 2021 residents by place of residence one year before"),
    ("census21_density", "0011613", {"Dim1": "S7A2021", "Dim3": "T"}, "Census 2021 population density (per km2)"),
    # Housing stock
    ("census21_dwellings_occupancy", "0014624", {"Dim1": "S7A2021", "Dim4": "T"}, "Census 2021 classic dwellings by form of occupancy"),
    ("census21_households_tenure", "0014622", {"Dim1": "S7A2021"}, "Census 2021 main-residence classic dwellings by tenure"),
    ("census21_crowding", "0012507", {"Dim1": "S7A2021"}, "Census 2021 main-residence classic dwellings by crowding level"),
    ("census_dwellings_change", "0012534", {"Dim1": "S7A2021"}, "Change in family dwellings, 2011-2021 (%)"),
    ("census_households_change", "0012488", {"Dim1": "S7A2021"}, "Change in private households, 2011-2021 (%)"),
    # Prices and incomes
    *[
        (f"sale_price_{y}", "0012255", {"Dim1": f"S7A{y}", "Dim3": "H1"}, f"Median sale price of dwellings, EUR/m2, {y}")
        for y in (2019, 2021, 2024, 2025)
    ],
    *[(f"new_rent_{y}", "0014698", {"Dim1": f"S7A{y}"}, f"Median rent of new leases, EUR/m2/month, {y}") for y in (2021, 2025)],
    *[(f"income_{y}", "0012741", {"Dim1": f"S7A{y}"}, f"Median declared income net of IRS per tax household, EUR, {y}") for y in (2019, 2021, 2024)],
    # Transport
    ("commute_time_collective", "0011794", {"Dim1": "S7A2021"}, "Census 2021 mean commute (min), public transport users"),
    ("commute_time_individual", "0011795", {"Dim1": "S7A2021"}, "Census 2021 mean commute (min), private transport users"),
    ("commute_out_share", "0011655", {"Dim1": "S7A2021"}, "Census 2021 share of residents commuting out of the municipality (%)"),
    ("commute_pairs_lisboa", "0012340", {"Dim1": "S7A2021", "Dim2": "1106"}, "Census 2021 work/study interactions with Lisboa"),
    ("commute_pairs_porto", "0012340", {"Dim1": "S7A2021", "Dim2": "1312"}, "Census 2021 work/study interactions with Porto"),
    # Schools (gross enrolment rates: pupils enrolled in the municipality / resident children of that age)
    ("school_pre_2012", "0009550", {"Dim1": "7A2012_3", "Dim3": "T"}, "Gross pre-school enrolment rate, 2011/12 (%)"),
    ("school_basic_2012", "0009552", {"Dim1": "7A2012_3", "Dim3": "T"}, "Gross basic-education enrolment rate, 2011/12 (%)"),
    ("school_sec_2012", "0009555", {"Dim1": "7A2012_3", "Dim3": "T"}, "Gross secondary enrolment rate, 2011/12 (%)"),
    ("school_pre_2025", "0012616", {"Dim1": "7A2025_3", "Dim3": "T"}, "Gross pre-school enrolment rate, 2024/25 (%)"),
    ("school_basic_2025", "0012614", {"Dim1": "7A2025_3", "Dim3": "T"}, "Gross basic-education enrolment rate, 2024/25 (%)"),
    ("school_sec_2025", "0012615", {"Dim1": "7A2025_3", "Dim3": "T"}, "Gross secondary enrolment rate, 2024/25 (%)"),
    # Hospitals
    ("hospitals_2024", "0012885", {"Dim1": "S7A2024", "Dim3": "T"}, "Hospitals, 2024"),
    ("hospital_beds_sns_2024", "0012893", {"Dim1": "S7A2024", "Dim3": "T"}, "Beds in public universal-access and PPP hospitals, 2024"),
    ("hospital_bed_occupancy_2024", "0012892", {"Dim1": "S7A2024"}, "Hospital bed occupancy rate (%), 2024"),
]


def url(code: str, dims: dict[str, str]) -> str:
    params = "&".join(f"{k}={v}" for k, v in dims.items())
    return f"{API}?op=2&varcd={code}&{params}&lang=PT"
