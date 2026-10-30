"""Inventory of every pinned data file: provider, source, retrieval, reuse terms and redistribution status.

Writes DATA_SOURCES.csv from the retrieval records kept with the snapshots (INE containers, Eurostat and ECB
.meta.json files) and from the source constants in analysis.py for the early Eurostat snapshots, which have no
per-request record. Provider terms were checked on 30 October 2026; see DATA_SOURCES.md for the wording.

Run:  python data_sources.py                          # rewrite DATA_SOURCES.csv
      python data_sources.py --check-redistribution   # exit 1 if any file may not be redistributed in an archive
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import analysis as A

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "DATA_SOURCES.csv"
FIELDS = ["file", "provider", "dataset", "title", "source_url", "doi", "retrieved_utc", "provider_last_update",
          "terms", "attribution", "redistribution", "notes"]

TERMS = {
    "INE": ("CC BY 4.0 (INE Portal terms of use)",
            "Fonte: Instituto Nacional de Estatística, IP – Portugal; name the statistic and reference period; "
            "state any modification"),
    "Eurostat": ("Eurostat copyright notice: reuse authorised, commercial or not, provided the source is acknowledged "
                 "(Commission Decision 2011/833/EU)",
                 "Source: Eurostat, dataset DOI, access date; state any change"),
    "ECB": ("ECB statistics usage policy: free reuse provided the source is quoted; statistics and metadata not "
            "modified",
            "Source: ECB statistics (ECB Data Portal), series key"),
    "PORDATA": ("PORDATA Termos Legais (updated 19.07.2024): content protected by copyright and related rights; no "
                "reuse licence granted",
                "Fonte: PORDATA (data originally from INE)"),
}
PERMITTED = "permitted"
NOT_ESTABLISHED = "not established"

# Early Eurostat snapshots carry no .meta.json: their request URLs are the constants used to download them.
EARLY_EUROSTAT = {
    "eurostat_prc_hpi_a_PT_2026-10-29.json": A.EUROSTAT_URL,
    "eurostat_ilc_di03_PT_2026-10-29.json": A.INCOME_URL,
    "eurostat_ilc_di03_PT_Y18-24_2026-10-29.json": A.INCOME_URL.replace("age=TOTAL", "age=Y18-24"),
    "eurostat_ilc_di03_PT_Y25-49_2026-10-29.json": A.INCOME_URL.replace("age=TOTAL", "age=Y25-49"),
    "eurostat_ilc_lvho07a_PT_2026-10-29.json": A.OVERBURDEN_URL,
    "eurostat_ilc_lvps08_PT_2026-10-29.json": A.PARENTS_URL,
    "eurostat_yth_demo_030_PT_2026-10-29.json": A.LEAVING_AGE_URL,
    "eurostat_ilc_lvho02_PT_2026-10-29.json": A.TENURE_URL,
    "eurostat_sts_cobp_a_PT_dwellings_2026-10-29.json": A.PERMITS_URL,
    "eurostat_lfst_hhnhtych_PT_2026-10-29.json": A.HOUSEHOLDS_URL,
    "eurostat_demo_pjan_PT_2026-10-29.json": A.POPULATION_URL,
}


def _row(file: str, provider: str, **fields: str) -> dict[str, str]:
    terms, attribution = TERMS[provider]
    row = {k: "" for k in FIELDS}
    row.update(file=file, provider=provider, terms=terms, attribution=attribution, redistribution=PERMITTED)
    row.update(fields)
    return row


def _eurostat_dataset(url: str) -> str:
    return url.split("/data/")[1].split("?")[0]


def eurostat_row(path: Path) -> dict[str, str]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    meta_path = path.with_name(path.name.replace(".json", ".meta.json"))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        url, retrieved, note = meta["url"], meta["retrieved_at_utc"], "retrieval record in .meta.json"
    else:
        url, retrieved = EARLY_EUROSTAT[path.name], path.name[-15:-5]
        note = "no per-request record; retrieval date from the file name; covered by SHA256SUMS"
    dataset = _eurostat_dataset(url)
    return _row(path.name, "Eurostat", dataset=dataset, title=str(doc.get("label", "")), source_url=url,
                doi=f"https://doi.org/10.2908/{dataset.upper()}", retrieved_utc=retrieved,
                provider_last_update=str(doc.get("updated", "")), notes=note)


def ine_row(path: Path) -> dict[str, str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    header = next(r for r in records if r.get("kind") == "header")
    meta = next(r for r in records if r.get("kind") == "metadata")
    body = json.loads(meta["body"])[0]
    times = sorted(r["retrieved_at_utc"] for r in records if "retrieved_at_utc" in r)
    retrieved = times[0] if times[0] == times[-1] else f"{times[0]} to {times[-1]}"
    code = header["indicator"]
    return _row(path.name, "INE", dataset=code, title=body["IndicadorNome"], source_url=meta["url"],
                retrieved_utc=retrieved, provider_last_update=body.get("DataUltimaAtualizacao", ""),
                notes="raw API responses, gzip-compressed, with per-request SHA-256")


def ecb_row(path: Path) -> dict[str, str]:
    meta = json.loads(path.with_name(path.name.replace(".csv", ".meta.json")).read_text(encoding="utf-8"))
    return _row(path.name, "ECB", dataset=meta["series_key"],
                title="Cost of borrowing for households for house purchase, new business, Portugal (MIR)",
                source_url=meta["url"], retrieved_utc=meta["retrieved_at_utc"],
                provider_last_update=meta["response_headers"].get("Last-Modified", ""),
                notes="compiled by Banco de Portugal within the ESCB; published by the ECB")


def pordata_row(path: Path) -> dict[str, str]:
    return _row(path.name, "PORDATA", dataset="Fogos concluídos em construções novas para habitação familiar",
                title="Completed dwellings in new construction for family housing, Portugal, 1995-2025",
                source_url=A.COMPLETIONS_URL, retrieved_utc=path.name[-14:-4], redistribution=NOT_ESTABLISHED,
                notes="manual transcription; 2011-2025 equal the sum of INE indicator 0013248 over municipalities "
                      "(see results_regional.json); 1995-2010 have no pinned INE equivalent")


def inventory() -> list[dict[str, str]]:
    """One row per pinned data file (retrieval records are listed with the file they describe)."""
    rows = []
    for path in sorted(DATA.iterdir()):
        name = path.name
        if name == "SHA256SUMS" or name.endswith(".meta.json"):
            continue
        if name.startswith("eurostat_"):
            rows.append(eurostat_row(path))
        elif name.startswith("ine_pordata_"):
            rows.append(pordata_row(path))
        elif name.startswith("ine_"):
            rows.append(ine_row(path))
        elif name.startswith("ecb_"):
            rows.append(ecb_row(path))
        else:
            raise ValueError(f"Unknown provider for {name}")
    return rows


def write(rows: list[dict[str, str]], path: Path = OUT) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def blocked(rows: list[dict[str, str]]) -> list[str]:
    """Files whose redistribution in a public archive has not been established."""
    return [r["file"] for r in rows if r["redistribution"] != PERMITTED]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-redistribution", action="store_true",
                        help="exit 1 if any pinned file may not be redistributed in a public archive")
    args = parser.parse_args()
    rows = inventory()
    if args.check_redistribution:
        stuck = blocked(rows)
        for name in stuck:
            print(f"BLOCKED: redistribution of data/{name} has not been established (see DATA_SOURCES.md)")
        print("archive check:", "OK" if not stuck else f"{len(stuck)} file(s) block a public archive")
        return 1 if stuck else 0
    write(rows)
    print(f"wrote {OUT.name} ({len(rows)} data files; {len(blocked(rows))} not cleared for redistribution)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
