"""Synthetic geolocated-DVF rows for the tests: the real column layout, invented sales."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pandas as pd

HEADER = ["id_mutation", "date_mutation", "numero_disposition", "nature_mutation", "valeur_fonciere",
          "adresse_numero", "adresse_suffixe", "adresse_nom_voie", "adresse_code_voie", "code_postal",
          "code_commune", "nom_commune", "code_departement", "ancien_code_commune",
          "ancien_nom_commune", "id_parcelle", "ancien_id_parcelle", "numero_volume", "lot1_numero",
          "lot1_surface_carrez", "lot2_numero", "lot2_surface_carrez", "lot3_numero",
          "lot3_surface_carrez", "lot4_numero", "lot4_surface_carrez", "lot5_numero",
          "lot5_surface_carrez", "nombre_lots", "code_type_local", "type_local",
          "surface_reelle_bati", "nombre_pieces_principales", "code_nature_culture", "nature_culture",
          "code_nature_culture_speciale", "nature_culture_speciale", "surface_terrain", "longitude",
          "latitude"]
TYPE_NAME = {"1": "Maison", "2": "Appartement", "3": "Dépendance", "4": "Local industriel"}
VENTE = "Vente"
VEFA = "Vente en l'état futur d'achèvement"
AUCTION = "Adjudication"


def row(mid, date, price, dep, parc, tl="", sb="", np_="", lots=(), nlots=0, cult="", st="",
        nat=VENTE, disp="000001"):
    r = dict.fromkeys(HEADER, "")
    lots = list(lots) + [""] * (5 - len(lots))
    r.update({"id_mutation": mid, "date_mutation": date, "numero_disposition": disp,
              "nature_mutation": nat, "valeur_fonciere": str(price), "code_departement": dep,
              "code_commune": dep + "001", "id_parcelle": parc, "nombre_lots": str(nlots),
              "code_type_local": tl, "type_local": TYPE_NAME.get(tl, ""), "surface_reelle_bati": sb,
              "nombre_pieces_principales": np_, "code_nature_culture": cult, "surface_terrain": st})
    for i, lot in enumerate(lots[:5], 1):
        r[f"lot{i}_numero"] = lot
    return r


def house(mid, date, price, dep, parc, sb="90", np_="4", land=("S", "400"), nat=VENTE):
    """A house on one parcel: one row per land-use type, the house repeated on each."""
    return [row(mid, date, price, dep, parc, "1", sb, np_, cult=land[0], st=land[1], nat=nat),
            row(mid, date, price, dep, parc, "1", sb, np_, cult="J", st="200", nat=nat)]


def flat(mid, date, price, dep, parc, lots, sb="50", np_="2", nat=VENTE, cellar=True):
    """A flat and its cellar: one row per unit, each listing every lot of the disposition."""
    out = [row(mid, date, price, dep, parc, "2", sb, np_, lots=lots, nlots=len(lots), nat=nat)]
    if cellar:
        out.append(row(mid, date, price, dep, parc, "3", "", "0", lots=lots, nlots=len(lots), nat=nat))
    return out


def sales_by_year() -> dict[int, list[dict]]:
    rows = []
    # 1 Bordeaux house: resold at exactly the old price
    rows += house("2021-1", "2021-03-01", 200000, "33", "33001000AB0001")
    rows += house("2024-1", "2024-06-01", 200000, "33", "33001000AB0001")
    # 2 Paris flat with cellar: loss then gain; lot numbers written with and without zeros
    rows += flat("2021-2", "2021-05-01", 400000, "75", "75101000CD0002", ["12", "0003"])
    rows += flat("2023-2", "2023-09-01", 380000, "75", "75101000CD0002", ["12", "3"])
    rows += flat("2025-2", "2025-06-01", 420000, "75", "75101000CD0002", ["0012", "3"])
    # 3 another flat in the same building: its own pair
    rows += flat("2022-3", "2022-01-10", 300000, "75", "75101000CD0002", ["13"], sb="40", cellar=False)
    rows += flat("2024-3", "2024-02-10", 310000, "75", "75101000CD0002", ["13"], sb="40", cellar=False)
    # 4 new house sold off plan, then resold (first sale kept apart)
    rows += house("2021-4", "2021-07-01", 250000, "44", "44001000EF0004", nat=VEFA)
    rows += house("2023-4", "2023-07-01", 240000, "44", "44001000EF0004")
    # 5 house extended between the two sales (structure changed)
    rows += house("2021-5", "2021-02-01", 150000, "23", "23001000GH0005", sb="90")
    rows += house("2024-5", "2024-02-01", 210000, "23", "23001000GH0005", sb="120")
    # 6 house resold after three months (dropped)
    rows += house("2022-6", "2022-01-01", 100000, "23", "23001000GH0006")
    rows += house("2022-60", "2022-04-01", 120000, "23", "23001000GH0006")
    # 7 two houses in one deed (not a single dwelling), parcel later resold alone
    rows += [row("2021-7", "2021-09-01", 500000, "23", "23001000GH0007", "1", "80", "3", cult="S", st="300"),
             row("2021-7", "2021-09-01", 500000, "23", "23001000GH0007", "1", "70", "3", cult="S", st="300")]
    rows += house("2024-7", "2024-09-01", 300000, "23", "23001000GH0007", sb="80", np_="3", land=("S", "300"))
    # 8 a flat key sold twice on the same day (ambiguous, both dropped)
    rows += flat("2022-8", "2022-03-03", 200000, "92", "92001000IJ0008", ["7"])
    rows += flat("2022-80", "2022-03-03", 205000, "92", "92001000IJ0008", ["7"])
    rows += flat("2025-8", "2025-03-03", 210000, "92", "92001000IJ0008", ["7"])
    # 9 house resold at auction
    rows += house("2021-9", "2021-04-01", 180000, "59", "59001000KL0009")
    rows += house("2023-9", "2023-10-01", 150000, "59", "59001000KL0009", nat=AUCTION)
    # 10 a flat sold with a shop in the same deed (excluded), then alone
    rows += [row("2021-10", "2021-06-01", 600000, "69", "69001000MN0010", "2", "60", "3", lots=["1", "2"], nlots=2),
             row("2021-10", "2021-06-01", 600000, "69", "69001000MN0010", "4", "80", "0", lots=["1", "2"], nlots=2)]
    rows += flat("2024-10", "2024-06-01", 350000, "69", "69001000MN0010", ["1"], sb="60", np_="3", cellar=False)
    by_year: dict[int, list[dict]] = {}
    for r in rows:
        by_year.setdefault(int(r["date_mutation"][:4]), []).append(r)
    return by_year


def write_dvf(directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for year, rows in sorted(sales_by_year().items()):
        p = directory / f"dvf_{year}.csv.gz"
        with gzip.open(p, "wt", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(rows)
        paths.append(str(p))
    return paths


def write_cpi(path: Path) -> str:
    months = pd.date_range("2019-01-01", "2026-12-01", freq="MS")
    pd.DataFrame({"month": months.strftime("%Y-%m-%d"),
                  "cpi": [100 * 1.003 ** k for k in range(len(months))]}).to_csv(path, index=False)
    return str(path)
