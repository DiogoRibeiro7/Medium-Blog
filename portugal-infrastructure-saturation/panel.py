"""Read the INE responses in data/raw and build one row per municipality (308 rows).

    python panel.py     # writes data/panel.csv

Municipalities are identified by their four-digit code (DICO). Annual INE series give the
municipality as a seven-character code, the NUTS III 2024 region followed by the DICO; census
series give the bare DICO; parishes have six or nine characters and are dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
PANEL = HERE / "data" / "panel.csv"

LISBOA, PORTO = "1106", "1312"
LISBON_METRO_NUTS3 = {"1A0", "1B0"}  # Grande Lisboa and Peninsula de Setubal (NUTS 2024)
PORTO_METRO_NUTS3 = {"11A"}  # Area Metropolitana do Porto
MEDIUM_CITY_MIN_POP = 50_000  # residents in the 2021 census, outside the two metropolitan areas

GROUPS = ["Lisboa", "Porto", "Lisbon metro ring", "Porto metro ring", "Medium cities", "Rest of mainland", "Islands"]


def read(label: str) -> pd.DataFrame:
    """All rows of one response, with the value as a float (NaN where INE publishes none)."""
    payload = json.loads((RAW / f"{label}.json").read_text(encoding="utf-8"))
    rows = next(iter(payload[0]["Dados"].values()))
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df.get("valor"), errors="coerce")
    return df


def municipal(df: pd.DataFrame) -> pd.DataFrame:
    """Keep municipality rows and add the four-digit code."""
    code = df["geocod"].astype(str)
    keep = code.str.len().isin([4, 7])
    out = df.loc[keep].copy()
    out["dico"] = code[keep].str[-4:]
    return out


def series(label: str, **dims: str) -> pd.Series:
    """One value per municipality, after fixing the remaining dimensions."""
    df = municipal(read(label))
    for k, v in dims.items():
        df = df[df[k] == v]
    if df["dico"].duplicated().any():
        raise ValueError(f"{label}: more than one row per municipality for {dims}")
    return df.set_index("dico")["value"]


def master() -> pd.DataFrame:
    """Names and NUTS III 2024 regions, from the 2025 population estimates."""
    df = read("pop_est_2025")
    df = df[df["geocod"].str.len() == 7].copy()
    out = pd.DataFrame(
        {
            "dico": df["geocod"].str[-4:],
            "name": df["geodsg"],
            "nuts3": df["geocod"].str[:3],
        }
    ).set_index("dico")
    if len(out) != 308 or out.index.duplicated().any():
        raise ValueError("expected 308 municipalities")
    return out


def group_of(row: pd.Series) -> str:
    if row.name == LISBOA:
        return "Lisboa"
    if row.name == PORTO:
        return "Porto"
    if row["nuts3"] in LISBON_METRO_NUTS3:
        return "Lisbon metro ring"
    if row["nuts3"] in PORTO_METRO_NUTS3:
        return "Porto metro ring"
    if not row["nuts3"].startswith("1"):
        return "Islands"
    if row["pop_census_2021"] >= MEDIUM_CITY_MIN_POP:
        return "Medium cities"
    return "Rest of mainland"


def build() -> pd.DataFrame:
    p = master()
    # Population
    p["pop_est_2011"] = series("pop_est_2011", dim_3="T", dim_4="T")
    p["pop_est_2021_n13"] = series("pop_est_2021_n13", dim_3="T", dim_4="T")
    for y in range(2021, 2026):
        p[f"pop_est_{y}"] = series(f"pop_est_{y}", dim_3="T", dim_4="T")
    p["births_2225"] = sum(series(f"births_{y}", dim_3="T") for y in range(2022, 2026))
    p["deaths_2225"] = sum(series(f"deaths_{y}", dim_3="T", dim_4="T") for y in range(2022, 2026))
    p["pop_census_2021"] = series("census21_pop_age", dim_3="T", dim_4="T")
    p["children_0_14_2021"] = series("census21_pop_age", dim_3="T", dim_4="1")
    p["aged_65_2021"] = series("census21_pop_age", dim_3="T", dim_4="4")
    p["movers_other_municipality_2021"] = series("census21_migration", dim_3="T", dim_4="33")
    p["movers_abroad_2021"] = series("census21_migration", dim_3="T", dim_4="34")
    p["density_2021"] = series("census21_density", dim_3="T")
    p["area_km2"] = p["pop_census_2021"] / p["density_2021"]
    # Housing
    for code, col in [("T", "dw_total"), ("1", "dw_main"), ("2", "dw_secondary"), ("3", "dw_vacant_market"), ("4", "dw_vacant_other")]:
        p[f"{col}_2021"] = series("census21_dwellings_occupancy", dim_3=code, dim_4="T")
    p["households_2021"] = series("census21_households_tenure", dim_3="T")
    p["rented_2021"] = series("census21_households_tenure", dim_3="3")
    crowd_total = series("census21_crowding", dim_3="T")
    crowded = sum(series("census21_crowding", dim_3=c) for c in ("5", "6", "7"))
    p["crowded_share_2021"] = 100 * crowded / crowd_total
    p["dwellings_change_pct"] = series("census_dwellings_change")
    p["households_change_pct"] = series("census_households_change")
    for y in (2019, 2021, 2024, 2025):
        p[f"sale_price_{y}"] = series(f"sale_price_{y}", dim_3="H1")
    for y in (2021, 2025):
        p[f"new_rent_{y}"] = series(f"new_rent_{y}")
    for y in (2019, 2021, 2024):
        p[f"income_{y}"] = series(f"income_{y}")
    # Transport
    p["commute_min_collective"] = series("commute_time_collective")
    p["commute_min_individual"] = series("commute_time_individual")
    p["commute_out_share"] = series("commute_out_share")
    for label, col in [("commute_pairs_lisboa", "pairs_lisboa"), ("commute_pairs_porto", "pairs_porto")]:
        df = read(label)
        df = df[df["dim_3"].astype(str).str.len() == 4]
        p[col] = df.set_index("dim_3")["value"].reindex(p.index).fillna(0.0)
    # Schools
    for level in ("pre", "basic", "sec"):
        for y in (2012, 2025):
            p[f"school_{level}_{y}"] = series(f"school_{level}_{y}", dim_3="T")
    # Hospitals
    p["hospitals_2024"] = series("hospitals_2024", dim_3="T").fillna(0.0)
    p["sns_beds_2024"] = series("hospital_beds_sns_2024", dim_3="T").fillna(0.0)
    p["bed_occupancy_2024"] = series("hospital_bed_occupancy_2024")

    p["group"] = p.apply(group_of, axis=1)
    p["mainland"] = p["group"] != "Islands"
    p["metro"] = p["group"].isin(["Lisboa", "Porto", "Lisbon metro ring", "Porto metro ring"])
    return p


def check(p: pd.DataFrame) -> None:
    """Identities that must hold if the responses were read correctly."""
    parts = p[["dw_main_2021", "dw_secondary_2021", "dw_vacant_market_2021", "dw_vacant_other_2021"]].sum(axis=1)
    if not np.allclose(parts, p["dw_total_2021"]):
        raise ValueError("dwelling occupancy categories do not add up to the total")
    if not np.allclose(p["households_2021"], p["dw_main_2021"]):
        raise ValueError("main-residence dwellings differ between the two census tables")
    if p["pop_census_2021"].isna().any() or p["pop_est_2025"].isna().any():
        raise ValueError("missing population")


def main() -> None:
    p = build()
    check(p)
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(PANEL)
    print(f"wrote {PANEL.relative_to(HERE)}: {len(p)} municipalities, {p.shape[1]} columns")
    print(p["group"].value_counts().reindex(GROUPS).to_string())


if __name__ == "__main__":
    main()
