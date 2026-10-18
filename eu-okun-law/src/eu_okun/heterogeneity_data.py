"""Data of the country-heterogeneity case study (``HETEROGENEITY_PROTOCOL.md``, section 2).

Eurostat national-accounts employment (domestic and resident concepts) and hours worked,
LFS temporary and total employees, temporary and part-time shares, and the OECD
employment-protection indices. Raw responses are frozen with SHA-256 hashes; the processed
CSVs are re-derived from them by the audit. Countries a dataset does not publish stay
missing: nothing is interpolated or imputed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from eu_okun.acquisition import fetch_dataset
from eu_okun.contract import EU27, EurostatDataError, Observation, SeriesSpec, quarter_sequence
from eu_okun.jsonstat import decode_series

START, END = "2008-Q1", "2026-Q2"
SERIES: dict[str, SeriesSpec] = {
    "emp_dc": SeriesSpec("emp_dc", "namq_10_pe", "emp_dc", {
        "freq": "Q", "unit": "THS_PER", "s_adj": "SCA", "na_item": "EMP_DC"}),
    "emp_nc": SeriesSpec("emp_nc", "namq_10_pe", "emp_nc", {
        "freq": "Q", "unit": "THS_PER", "s_adj": "SCA", "na_item": "EMP_NC"}),
    "emp_a10": SeriesSpec("emp_a10", "namq_10_a10_e", "emp_a10", {
        "freq": "Q", "unit": "THS_PER", "s_adj": "SCA", "na_item": "EMP_DC", "nace_r2": "TOTAL"}),
    "hours": SeriesSpec("hours", "namq_10_a10_e", "hours", {
        "freq": "Q", "unit": "THS_HW", "s_adj": "SCA", "na_item": "EMP_DC", "nace_r2": "TOTAL"}),
    "temp": SeriesSpec("temp", "lfsq_etgaed", "temp", {
        "freq": "Q", "unit": "THS_PER", "sex": "T", "age": "Y15-74", "isced11": "TOTAL"}),
    "employees": SeriesSpec("employees", "lfsq_eegaed", "employees", {
        "freq": "Q", "unit": "THS_PER", "sex": "T", "age": "Y15-74", "isced11": "TOTAL"}),
    "temp_share": SeriesSpec("temp_share", "lfsq_etpga", "temp_share", {
        "freq": "Q", "unit": "PC", "sex": "T", "age": "Y15-74"}),
    "pt_share": SeriesSpec("pt_share", "lfsq_eppga", "pt_share", {
        "freq": "Q", "unit": "PC", "sex": "T", "age": "Y15-74"}),
}
EPL_URL = ("https://sdmx.oecd.org/public/rest/data/OECD.ELS.JAI,DSD_EPL@DF_EPL,1.0/all"
           "?startPeriod=2008&endPeriod=2008&dimensionAtObservation=AllDimensions")
EPL_MEASURES = ("EPL_R", "EPL_T")
EPL_VERSION = "VERSION3"
ISO3 = {"AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU",
        "DK": "DNK", "EE": "EST", "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA",
        "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA", "LT": "LTU", "LU": "LUX",
        "LV": "LVA", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU",
        "SE": "SWE", "SI": "SVN", "SK": "SVK"}
QUARTERLY_FILE, EPL_FILE = "quarterly.csv", "epl.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_available(payload: Mapping[str, Any], spec: SeriesSpec,
                     ) -> dict[tuple[str, str], Observation]:
    """Decode the EU-27 countries the dataset publishes; the others stay missing."""
    geo = payload.get("dimension", {}).get("geo", {}).get("category", {}).get("index", {})
    present = [g for g in EU27 if g in geo]
    if not present:
        raise EurostatDataError(f"{spec.dataset} publishes no EU-27 country.")
    return decode_series(payload, spec, countries=present)


def _quarterly_rows(cells: Mapping[str, Mapping[tuple[str, str], Observation]],
                    ) -> list[dict[str, str]]:
    rows = []
    for geo in EU27:
        for q in quarter_sequence(START, END):
            row = {"geo": geo, "quarter": q}
            for key in SERIES:
                cell = cells[key].get((geo, q), Observation(None, ""))
                row[key] = "" if cell.value is None else repr(cell.value)
                row[f"{key}_flag"] = cell.status
            rows.append(row)
    return rows


def parse_epl(raw: bytes) -> list[dict[str, str]]:
    """EPL_R and EPL_T, version 3, 2008, for the EU countries the OECD covers."""
    values: dict[tuple[str, str], float] = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        if row["VERSION"] == EPL_VERSION and row["MEASURE"] in EPL_MEASURES \
                and row["TIME_PERIOD"] == "2008":
            values[(row["REF_AREA"], row["MEASURE"])] = float(row["OBS_VALUE"])
    out = []
    for geo in EU27:
        r, t = values.get((ISO3[geo], "EPL_R")), values.get((ISO3[geo], "EPL_T"))
        if r is not None and t is not None:
            out.append({"geo": geo, "epl_r": repr(r), "epl_t": repr(t)})
    if not out:
        raise EurostatDataError("No EU country in the OECD EPL response.")
    return out


def _write(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Medium-Blog/eu-okun-law",
                                    "Accept": "application/vnd.sdmx.data+csv; charset=utf-8"})
    try:
        with urlopen(request, timeout=120) as response:
            return bytes(response.read())
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"OECD download failed: {exc}") from exc


def fetch_heterogeneity(output_dir: Path) -> dict[str, Any]:
    """Download every series, freeze the raw responses, write the processed CSVs."""
    raw_dir, processed = output_dir / "raw", output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    files, cells = [], {}
    for key, spec in SERIES.items():
        raw, payload, url = fetch_dataset(spec, START, END)
        path = raw_dir / f"{key}.json"
        path.write_bytes(raw)
        cells[key] = decode_available(payload, spec)
        files.append({"key": key, "dataset": spec.dataset, "url": url,
                      "updated": str(payload.get("updated", "")), "sha256": _sha256(path),
                      "countries": sorted({g for g, _ in cells[key]})})
    epl_path = raw_dir / "oecd_epl.csv"
    epl_path.write_bytes(_download(EPL_URL))
    files.append({"key": "epl", "dataset": "OECD DSD_EPL@DF_EPL", "url": EPL_URL,
                  "updated": "", "sha256": _sha256(epl_path), "licence": "CC BY 4.0"})
    _write(processed / QUARTERLY_FILE, _quarterly_rows(cells))
    _write(processed / EPL_FILE, parse_epl(epl_path.read_bytes()))
    manifest = {
        "description": "Inputs of HETEROGENEITY_PROTOCOL.md",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "period": {"start": START, "end": END},
        "files": files,
        "processed": {name: _sha256(processed / name) for name in (QUARTERLY_FILE, EPL_FILE)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    return manifest


def audit_heterogeneity(directory: Path) -> dict[str, Any]:
    """Check raw hashes and re-derive both processed CSVs from the raw responses."""
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text(
        encoding="utf-8"))
    for entry in manifest["files"]:
        name = "oecd_epl.csv" if entry["key"] == "epl" else f"{entry['key']}.json"
        if _sha256(directory / "raw" / name) != entry["sha256"]:
            raise EurostatDataError(f"{name}: SHA-256 mismatch.")
    for name in (QUARTERLY_FILE, EPL_FILE):
        if _sha256(directory / "processed" / name) != manifest["processed"][name]:
            raise EurostatDataError(f"{name} does not match its manifest hash.")
    cells = {key: decode_available(json.loads((directory / "raw" / f"{key}.json").read_bytes()),
                                   spec) for key, spec in SERIES.items()}
    if _read(directory / "processed" / QUARTERLY_FILE) != _quarterly_rows(cells):
        raise EurostatDataError("quarterly.csv does not re-derive from the raw responses.")
    if _read(directory / "processed" / EPL_FILE) != parse_epl(
            (directory / "raw" / "oecd_epl.csv").read_bytes()):
        raise EurostatDataError("epl.csv does not re-derive from the raw response.")
    return manifest


def load_heterogeneity(directory: Path) -> tuple[dict[tuple[str, str], dict[str, Any]],
                                                 dict[str, dict[str, float]]]:
    """Quarterly values and flags by (country, quarter); EPL by country."""
    panel: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read(directory / "processed" / QUARTERLY_FILE):
        panel[(row["geo"], row["quarter"])] = {
            **{key: float(row[key]) if row[key] else None for key in SERIES},
            **{f"{key}_flag": row[f"{key}_flag"] for key in SERIES}}
    epl = {row["geo"]: {"epl_r": float(row["epl_r"]), "epl_t": float(row["epl_t"])}
           for row in _read(directory / "processed" / EPL_FILE)}
    return panel, epl
