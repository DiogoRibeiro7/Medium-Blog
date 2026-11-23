"""Turn the raw responses into two tables.

    python panel.py   # writes data/europe.csv and data/metro.csv

``europe.csv``: one row per functional urban area (FUA) with population, jobs and dwellings for the
city and for the whole FUA at a start year near 2015 and an end year near 2021. The ring is the FUA
minus the city.

``national.csv``: national benchmarks: employment in the national accounts of each country (Eurostat) and
the full-time employees of Portugal as a whole (Quadros de Pessoal).

``metro.csv``: one row per municipality of the Lisbon and Porto metropolitan areas (35), with the census
2021 commuting shares, employees by place of work in 2021 and 2024, tourist beds, resident population,
the 2021 dwelling stock, dwellings completed since, prices and rents.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
EUROPE = HERE / "data" / "europe.csv"
METRO = HERE / "data" / "metro.csv"
NATIONAL = HERE / "data" / "national.csv"

START = (2014, 2016, 2015)  # window and preferred year
END = (2020, 2022, 2021)
LISBON_METRO_NUTS3 = {"1A0", "1B0"}
PORTO_METRO_NUTS3 = {"11A"}
LISBOA, PORTO = "1106", "1312"


# ---------------------------------------------------------------- Eurostat
def jsonstat(label: str) -> pd.DataFrame:
    """Long table from a Eurostat JSON-stat response: unit code, unit name, year, value."""
    d = json.loads((RAW / f"{label}.json").read_text(encoding="utf-8"))
    ids, size = d["id"], d["size"]
    index = {k: {pos: code for code, pos in d["dimension"][k]["category"]["index"].items()} for k in ids}
    names = d["dimension"]["cities"]["category"]["label"]
    rows = []
    for flat, value in d["value"].items():
        f = int(flat)
        coord = []
        for s in reversed(size):
            coord.append(f % s)
            f //= s
        coord.reverse()
        r = {k: index[k][c] for k, c in zip(ids, coord, strict=True)}
        rows.append((r["cities"], int(r["time"]), float(value)))
    df = pd.DataFrame(rows, columns=["code", "year", "value"])
    df = df[df["code"].str.len() > 2].copy()  # drop country totals
    df["stem"] = df["code"].str[:5]
    df["name"] = df["code"].map(names)
    return df


def flat_jsonstat(label: str) -> pd.DataFrame:
    """Long table from any Eurostat JSON-stat response: one column per dimension, plus value."""
    d = json.loads((RAW / f"{label}.json").read_text(encoding="utf-8"))
    ids, size = d["id"], d["size"]
    index = {k: {pos: code for code, pos in d["dimension"][k]["category"]["index"].items()} for k in ids}
    rows = []
    for flat, value in d["value"].items():
        f = int(flat)
        coord = []
        for s in reversed(size):
            coord.append(f % s)
            f //= s
        coord.reverse()
        rows.append({**{k: index[k][c] for k, c in zip(ids, coord, strict=True)}, "value": float(value)})
    return pd.DataFrame(rows)


def national() -> pd.DataFrame:
    na = flat_jsonstat("eu_employment_national")
    na = pd.DataFrame({"series": "employment_national_accounts", "geo": na["geo"], "year": na["time"].astype(int), "value": na["value"]})
    rows = []
    for y in (2021, 2024):
        df = json.loads((RAW / f"employees_{y}.json").read_text(encoding="utf-8"))
        pt = [r for r in next(iter(df[0]["Dados"].values())) if r["geocod"] == "PT"]
        if any("valor" not in r for r in pt):
            raise ValueError(f"employees_{y}: a size band is withheld for Portugal")
        rows.append({"series": "employees_quadros_de_pessoal", "geo": "PT", "year": y, "value": sum(float(r["valor"]) for r in pt)})
    return pd.concat([na, pd.DataFrame(rows)], ignore_index=True)


def wide(label: str) -> tuple[pd.DataFrame, pd.Series]:
    """Years in columns, one row per city or FUA. Only codes ending in C (city) or F (functional urban
    area) are kept: a few Swiss FUAs also have a cross-border total coded T, which would otherwise be
    mixed with the national part."""
    df = jsonstat(label)
    df = df[df["code"].str[5:] == ("F" if label.endswith("_fua") else "C")]
    if df.duplicated(["stem", "year"]).any():
        raise ValueError(f"{label}: more than one unit per stem and year")
    return df.pivot(index="stem", columns="year", values="value"), df.drop_duplicates("stem").set_index("stem")["name"]


def common_year(stem: str, frames: list[pd.DataFrame], window: tuple[int, int, int]) -> int | None:
    lo, hi, pref = window
    ok = [y for y in range(lo, hi + 1) if all(stem in X.index and y in X.columns and pd.notna(X.at[stem, y]) for X in frames)]
    return min(ok, key=lambda y: (abs(y - pref), -y)) if ok else None


def europe() -> pd.DataFrame:
    frames, names = {}, None
    for key, label in [
        ("P", "eu_pop_city"),
        ("PF", "eu_pop_fua"),
        ("J", "eu_jobs_city"),
        ("JF", "eu_jobs_fua"),
        ("D", "eu_dwellings_city"),
        ("DF", "eu_dwellings_fua"),
    ]:
        frames[key], n = wide(label)
        if key == "P":
            names = n
    stems = sorted(set.intersection(*(set(X.index) for X in frames.values())))
    rows = []
    for s in stems:
        fr = list(frames.values())
        y0, y1 = common_year(s, fr, START), common_year(s, fr, END)
        if y0 is None or y1 is None:
            continue
        r = {"stem": s, "city": names.get(s), "country": s[:2], "y0": y0, "y1": y1}
        for k, X in frames.items():
            r[f"{k}0"], r[f"{k}1"] = X.at[s, y0], X.at[s, y1]
        rows.append(r)
    return pd.DataFrame(rows).set_index("stem")


# ---------------------------------------------------------------- INE
def ine(label: str) -> pd.DataFrame:
    payload = json.loads((RAW / f"{label}.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(next(iter(payload[0]["Dados"].values())))
    df["value"] = pd.to_numeric(df.get("valor"), errors="coerce")
    code = df["geocod"].astype(str)
    df = df[code.str.len().isin([4, 7])].copy()
    df["dico"] = df["geocod"].str[-4:]
    return df


def series(label: str, **dims: str) -> pd.Series:
    df = ine(label)
    for k, v in dims.items():
        df = df[df[k] == v]
    if df["dico"].duplicated().any():
        raise ValueError(f"{label}: duplicate municipalities for {dims}")
    return df.set_index("dico")["value"]


def count(label: str, **dims: str) -> pd.Series:
    """A count where INE's sign '-' (nil) means zero; confidential values stay missing."""
    df = ine(label)
    for k, v in dims.items():
        df = df[df[k] == v]
    nil = df.get("sinal_conv", pd.Series("", index=df.index)) == "-"
    df.loc[nil & df["value"].isna(), "value"] = 0.0
    return df.set_index("dico")["value"]


def employees(label: str) -> tuple[pd.Series, pd.Series]:
    """Employees summed over establishment-size bands, and the number of bands INE withholds."""
    df = ine(label)
    withheld = df["value"].isna() & (df.get("sinal_conv", "") == "…")
    return df.groupby("dico")["value"].sum(min_count=1), withheld.groupby(df["dico"]).sum()


def metro() -> pd.DataFrame:
    base = ine("pop_est_2025")
    base = base[base["geocod"].str.len() == 7]
    m = pd.DataFrame({"name": base["geodsg"].to_numpy(), "nuts3": base["geocod"].str[:3].to_numpy()}, index=base["dico"].to_numpy())
    m.index.name = "dico"
    m = m[m["nuts3"].isin(LISBON_METRO_NUTS3 | PORTO_METRO_NUTS3)].copy()
    m["metro"] = np.where(m["nuts3"].isin(LISBON_METRO_NUTS3), "Lisbon", "Porto")
    m["core"] = m.index.isin([LISBOA, PORTO])
    m["pop_est_2011"] = series("pop_est_2011", dim_3="T", dim_4="T")
    for y in (2021, 2024, 2025):
        m[f"pop_est_{y}"] = series(f"pop_est_{y}", dim_3="T", dim_4="T")
    m["pop_census_2021"] = series("census21_pop_age", dim_3="T", dim_4="T")
    for code, col in [("T", "dw_total"), ("1", "dw_main"), ("2", "dw_secondary"), ("3", "dw_vacant_market"), ("4", "dw_vacant_other")]:
        m[f"{col}_2021"] = series("census21_dwellings_occupancy", dim_3=code, dim_4="T")
    m["households_2021"] = series("census21_households_tenure", dim_3="T")
    m["rented_2021"] = series("census21_households_tenure", dim_3="3")
    crowd_total = series("census21_crowding", dim_3="T")
    m["crowded_share_2021"] = 100 * sum(series("census21_crowding", dim_3=c) for c in ("5", "6", "7")) / crowd_total
    m["dwellings_change_pct"] = series("census_dwellings_change")
    m["households_change_pct"] = series("census_households_change")
    for y in (2019, 2025):
        m[f"sale_price_{y}"] = series(f"sale_price_{y}", dim_3="H1")
    for y in (2021, 2025):
        m[f"new_rent_{y}"] = series(f"new_rent_{y}")
    for y in (2023, 2025):
        m[f"tourist_beds_{y}"] = count(f"tourist_beds_{y}", dim_3="T")
    for y in (2022, 2023, 2024, 2025):
        m[f"completions_{y}"] = count(f"completions_{y}", dim_3="T")
    m["commute_in_share"] = series("commute_in_share")
    m["commute_out_share"] = series("commute_out_share")
    m["school_pre_2025"] = series("school_pre_2025", dim_3="T")
    m["school_basic_2025"] = series("school_basic_2025", dim_3="T")
    m["sns_beds_2024"] = series("hospital_beds_sns_2024", dim_3="T").fillna(0.0)
    for y in (2021, 2024):
        e, w = employees(f"employees_{y}")
        m[f"employees_{y}"] = e
        m[f"employees_withheld_bands_{y}"] = w.reindex(m.index).fillna(0).astype(int)
    if len(m) != 35:
        raise ValueError(f"expected 35 metro municipalities, got {len(m)}")
    gaps = m.drop(columns=["name", "nuts3", "metro", "core"]).isna().sum()
    if gaps.any():
        raise ValueError(f"missing values in the metro panel: {gaps[gaps > 0].to_dict()}")
    return m


def main() -> None:
    eu = europe()
    EUROPE.parent.mkdir(parents=True, exist_ok=True)
    eu.to_csv(EUROPE)
    mt = metro()
    mt.to_csv(METRO)
    national().to_csv(NATIONAL, index=False)
    print(f"wrote {EUROPE.relative_to(HERE)}: {len(eu)} FUAs with all six series at a start and an end year")
    print(f"wrote {METRO.relative_to(HERE)}: {len(mt)} metro municipalities")
    print(f"wrote {NATIONAL.relative_to(HERE)}: national benchmarks")


if __name__ == "__main__":
    main()
