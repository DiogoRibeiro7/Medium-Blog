"""Medicare Part D Prescribers by Geography and Drug (CMS), national rows.

One dataset per year. Each national row is a brand and generic name with total
claims, 30-day fills and beneficiaries, overall and for beneficiaries aged 65
and over (the GE65 columns). Counts below 11 are suppressed by CMS.
"""

from __future__ import annotations

import io

import pandas as pd

from . import http
from .paths import DATA, RAW

API = "https://data.cms.gov/data-api/v1/dataset/{uuid}/data"
DATASETS = {  # year: (API dataset id, CSV download), from the data.gov catalogue entry
    2013: ("9aec2a70-4ef2-438a-bc35-1747fe2492c9",
           "https://data.cms.gov/sites/default/files/2021-08/MUP_DPR_RY21_P04_V10_DY13_Geo_0.csv"),
    2014: ("63099eb4-6c9e-4fec-8c9e-b6b132f1b7f4",
           "https://data.cms.gov/sites/default/files/2021-08/MUP_DPR_RY21_P04_V10_DY14_Geo_0.csv"),
    2015: ("73748d0b-2acb-420a-ac66-27d8e238fdc8",
           "https://data.cms.gov/sites/default/files/2021-08/MUP_DPR_RY21_P04_V10_DY15_Geo_0.csv"),
    2016: ("c5b3c840-5eb2-4e4c-ac35-c30995dcb051",
           "https://data.cms.gov/sites/default/files/2021-08/MUP_DPR_RY21_P04_V10_DY16_Geo_0.csv"),
    2017: ("7863e41c-3f6f-4889-93a8-be65e9f3d073",
           "https://data.cms.gov/sites/default/files/2024-02/efe83dea-bdce-457f-b8b3-fe3b8e9e0c33/MUP_DPR_RY21_P04_V10_DY17_Geo.csv"),
    2018: ("b083c9b6-b841-4676-8d0a-fb03ee1431a1",
           "https://data.cms.gov/sites/default/files/2021-08/MUP_DPR_RY21_P04_V10_DY18_Geo.csv"),
    2019: ("73a6335e-f16f-4c81-a84b-6b5a986e2bf8",
           "https://data.cms.gov/sites/default/files/2021-08/MUP_DPR_RY21_P04_V10_DY19_Geo.csv"),
    2020: ("83891e77-99cf-4865-b60a-97703b916e09",
           "https://data.cms.gov/sites/default/files/2022-07/ca71b7df-4d48-4c2d-aded-2ca22285739c/MUP_DPR_RY22_P04_V10_DY20_Geo.csv"),
    2021: ("7dda2a9d-034a-446a-b4b3-e1254e0127b2",
           "https://data.cms.gov/sites/default/files/2023-04/3d3ebd5b-b4bf-45b4-876d-afa7916d1b72/MUP_DPR_RY23_P04_V10_DY21_Geo.csv"),
    2022: ("1fc57194-a51d-4864-aee6-de0889488151",
           "https://data.cms.gov/sites/default/files/2024-05/410ed206-d782-4dba-b442-6bcd45ae2016/MUP_DPR_RY24_P04_V10_DY22_Geo.csv"),
    2023: ("3463648b-1971-478d-84ca-80cadc758153",
           "https://data.cms.gov/sites/default/files/2025-04/9fe6b8a6-0cb9-4b7c-9760-87800da010a8/MUP_DPR_RY25_P04_V10_DY23_Geo.csv"),
}

CLASSES = {
    "methylphenidate class": ["methylphenidate", "dexmethylphenidate"],
    "amphetamine class": ["amphetamine", "lisdexamfetamine"],
    "atomoxetine (non-stimulant)": ["atomoxetine"],
    "modafinil and armodafinil": ["modafinil"],
}
NUMERIC = ["Tot_Prscrbrs", "Tot_Clms", "Tot_30day_Fills", "Tot_Drug_Cst", "Tot_Benes",
           "GE65_Tot_Clms", "GE65_Tot_30day_Fills", "GE65_Tot_Drug_Cst", "GE65_Tot_Benes"]


def classify(generic: str) -> str | None:
    g = str(generic).lower()
    for cls, terms in CLASSES.items():
        if any(t in g for t in terms):
            return cls
    return None


def _national_via_api(uuid: str) -> pd.DataFrame | None:
    rows, offset, size = [], 0, 5000
    while True:
        r = http.get(API.format(uuid=uuid), params={"filter[Prscrbr_Geo_Lvl]": "National",
                                                     "size": size, "offset": offset}, ok_404=True)
        if r is None:
            return None
        page = r.json()
        if not isinstance(page, list):
            return None
        rows.extend(page)
        if len(page) < size:
            break
        offset += size
    return pd.DataFrame(rows) if rows else None


def _national_via_csv(url: str, year: int) -> pd.DataFrame | None:
    path = http.download(url, RAW / "medicare" / f"geo_{year}.csv", ok_404=True)
    if path is None:
        return None
    df = pd.read_csv(path, low_memory=False)
    return df[df["Prscrbr_Geo_Lvl"].astype(str).str.lower() == "national"]


def build() -> pd.DataFrame:
    frames = []
    for year, (uuid, url) in DATASETS.items():
        df = None
        try:
            df = _national_via_api(uuid)
        except Exception as exc:  # fall back to the CSV
            print(f"  Medicare {year}: API failed ({exc}); trying CSV")
        if df is None or df.empty:
            df = _national_via_csv(url, year)
        if df is None or df.empty:
            print(f"  Medicare {year}: no data")
            continue
        df["drug_class"] = df["Gnrc_Name"].map(classify)
        df = df[df["drug_class"].notna()].copy()
        for c in NUMERIC:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["year"] = year
        keep = ["year", "Brnd_Name", "Gnrc_Name", "drug_class"] + [c for c in NUMERIC if c in df.columns] + \
               [c for c in df.columns if "Sprsn" in c]
        frames.append(df[keep])
        print(f"  Medicare {year}: {len(df)} national rows for the drug classes")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA / "medicare_national.csv.gz", index=False, compression="gzip")
    return out


def load() -> pd.DataFrame:
    return pd.read_csv(DATA / "medicare_national.csv.gz")
