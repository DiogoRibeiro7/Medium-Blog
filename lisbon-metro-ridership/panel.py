"""Turn the raw INE and Eurostat responses into the tables the analysis reads.

    python panel.py   # writes data/monthly.csv, data/bus.csv, data/river.csv, data/population.csv, data/employment.csv

All five files are rebuilt from data/raw/ at each run and ignored by git.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data"

MONTHS_PT = {
    "Janeiro": 1,
    "Fevereiro": 2,
    "Março": 3,
    "Abril": 4,
    "Maio": 5,
    "Junho": 6,
    "Julho": 7,
    "Agosto": 8,
    "Setembro": 9,
    "Outubro": 10,
    "Novembro": 11,
    "Dezembro": 12,
}
SYSTEMS = {"Lisboa": "lisboa", "Porto": "porto", "Metro Sul do Tejo": "mst"}

# Municipalities with a station of the Lisbon metro, and the rest of the metro area (NUTS 2024: 1A and 1B).
METRO_MUNICIPALITIES = ["Lisboa", "Amadora", "Odivelas", "Loures"]
AML = [
    "Lisboa",
    "Amadora",
    "Odivelas",
    "Loures",
    "Cascais",
    "Oeiras",
    "Sintra",
    "Mafra",
    "Vila Franca de Xira",
    "Alcochete",
    "Almada",
    "Barreiro",
    "Moita",
    "Montijo",
    "Palmela",
    "Seixal",
    "Sesimbra",
    "Setúbal",
]


def payload(label: str) -> dict:
    return json.loads((RAW / f"{label}.json").read_text())[0]


def month_of(text: str) -> pd.Timestamp:
    name, _, year = text.split(" ")
    return pd.Timestamp(int(year), MONTHS_PT[name], 1)


def value(row: dict) -> float | None:
    v = row.get("valor")
    return float(v) if v not in (None, "") else None


def metro(label: str, name: str) -> pd.DataFrame:
    rows = []
    for period, items in payload(label)["Dados"].items():
        for r in items:
            if r["geodsg"] in SYSTEMS:
                rows.append({"month": month_of(period), "column": f"{name}_{SYSTEMS[r['geodsg']]}", "value": value(r)})
    return pd.DataFrame(rows).pivot(index="month", columns="column", values="value")


def airport() -> pd.Series:
    total: dict[pd.Timestamp, float] = {}
    for label in ("airport_lisbon_embarked", "airport_lisbon_disembarked"):
        for period, items in payload(label)["Dados"].items():
            for r in items:
                if r.get("geocod") == "LPPT" and r.get("dim_3") == "T" and r.get("dim_4") == "T":
                    m = month_of(period)
                    total[m] = total.get(m, 0.0) + value(r)
    return pd.Series(total, name="airport_lisbon").sort_index()


def bus() -> pd.DataFrame:
    rows = []
    for year, items in payload("bus_passengers")["Dados"].items():
        for r in items:
            if r.get("dim_3") == "111" and r["geocod"] in ("1A", "1B"):
                rows.append({"year": int(year), "region": r["geodsg"], "passengers_thousand": value(r)})
    return pd.DataFrame(rows).pivot(index="year", columns="region", values="passengers_thousand")


def river() -> pd.DataFrame:
    rows = []
    for year, items in payload("river_passengers")["Dados"].items():
        for r in items:
            rows.append({"year": int(year), "line": r["geodsg"], "passengers": value(r)})
    return pd.DataFrame(rows).pivot(index="year", columns="line", values="passengers")


def population() -> pd.DataFrame:
    out = {}
    for label, year in (("pop_est_2024", 2024), ("pop_est_2025", 2025)):
        d = payload(label)["Dados"][str(year)]
        out[year] = {r["geodsg"]: value(r) for r in d if r.get("geodsg") in AML and len(r.get("geocod", "")) == 7}
    frame = pd.DataFrame(out)
    missing = set(AML) - set(frame.index)
    if missing:
        raise ValueError(f"population: municipalities missing {sorted(missing)}")
    return frame


def employment() -> pd.DataFrame:
    d = json.loads((RAW / "employment_nuts2.json").read_text())
    dims, size = d["id"], d["size"]
    cats = {k: d["dimension"][k]["category"]["index"] for k in dims}
    rows = []
    for flat, v in d["value"].items():
        i, coord = int(flat), {}
        for k, s in zip(reversed(dims), reversed(size), strict=True):
            i, pos = divmod(i, s)
            coord[k] = next(c for c, p in cats[k].items() if p == pos)
        rows.append({"geo": coord["geo"], "year": int(coord["time"]), "employed_thousand": v})
    return pd.DataFrame(rows).pivot(index="year", columns="geo", values="employed_thousand")


def main() -> None:
    monthly = pd.concat(
        [metro("metro_passengers", "pax"), metro("metro_passenger_km", "pkm"), metro("metro_place_km", "placekm"), airport()],
        axis=1,
    ).sort_index()
    expected = pd.date_range("2004-01-01", "2026-08-01", freq="MS")
    if not monthly.index.equals(expected):
        raise ValueError("monthly panel: months missing or extra")
    if monthly.loc["2015-01-01":, "airport_lisbon"].isna().any():
        raise ValueError("airport series has gaps")
    monthly.to_csv(OUT / "monthly.csv", index_label="month")
    bus().to_csv(OUT / "bus.csv")
    river().to_csv(OUT / "river.csv")
    population().to_csv(OUT / "population.csv", index_label="municipality")
    employment().to_csv(OUT / "employment.csv")
    print("wrote data/monthly.csv, bus.csv, river.csv, population.csv, employment.csv")


if __name__ == "__main__":
    main()
