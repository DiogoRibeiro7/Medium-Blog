"""Detailed European value-added composition using pinned Eurostat inputs.

Current-price values are additive; chain-linked volumes are not. Observations
and flags are preserved, and absent cells never become zero.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline import HERE, ROOT, REPO, download, verify_bytes, ratio, growth, write_csv

GEO = tuple(sorted("""EU27_2020 AT BE BG HR CY CZ DK EE FI FR DE EL HU
IE IT LV LT LU MT NL PL PT RO SK SI ES SE""".split()))
YEARS = range(1995, 2026)
A10 = ("A", "B-E", "F", "G-I", "J", "K", "L", "M_N", "O-Q", "R-U")
A64 = tuple("ABCDEFGHIJKLMNOPQRST")
NESTED = ("J61", "Q86", "Q87_Q88")
LABELS = {
    "A": "Agriculture, forestry and fishing", "B": "Mining and quarrying",
    "C": "Manufacturing", "D": "Electricity and gas", "E": "Water and waste",
    "F": "Construction", "G": "Wholesale and retail", "H": "Transport and storage",
    "I": "Accommodation and food", "J": "Information and communication",
    "K": "Finance and insurance", "L": "Real estate", "M": "Professional, scientific and technical",
    "N": "Administrative and support", "O": "Public administration and defence",
    "P": "Education", "Q": "Health and social work", "R": "Arts and recreation",
    "S": "Other services", "T": "Households as employers", "J61": "Telecommunications",
    "Q86": "Human health", "Q87_Q88": "Residential care and social work",
    "B-E": "Industry excluding construction", "G-I": "Trade, transport and hospitality",
    "M_N": "Professional and administrative services", "O-Q": "Administration, education and health",
    "R-U": "Arts, other services and households", "TOTAL": "Total gross value added",
}
# Reuse existing consumer contracts, rather than inventing new snapshots.
SOURCES = {
    "a10": ("eurostat-services-energy-time", "6911646f783bad0475e5942b1e043169e4e5cb02",
            "nama_10_a10.json", "nama_10_a10_e.json"),
    "a64": ("eurostat-services-relative-prices", "61f630990532bd6bcf6793621d6646d50be23d5c",
            "nama_10_a64.json", "nama_10_a64_e.json"),
}
GROUPS = {
    "agriculture": ("A",),
    "manufacturing": ("C",),
    "other_industry": ("B", "D", "E"),
    "construction": ("F",),
    "trade_transport_hospitality": ("G", "H", "I"),
    "information_communication": ("J",),
    "finance_insurance": ("K",),
    "real_estate": ("L",),
    "professional_scientific_technical": ("M",),
    "administrative_support": ("N",),
    "administration_education_health": ("O", "P", "Q"),
    "arts_other_household_services": ("R", "S", "T"),
}
CellKey = tuple[str, int, str, str]
Record = dict[str, Any]


@dataclass(frozen=True)
class Cell:
    """One observed or explicitly flagged/missing JSON-stat position."""

    value: float | None
    flag: str
    presence: str


def ordered_codes(category: Record, size: int) -> list[str]:
    """Decode category order and reject malformed or duplicate positions."""
    index = category["index"]
    if isinstance(index, list):
        codes = index
    elif isinstance(index, dict):
        if any(type(v) is not int for v in index.values()):
            raise ValueError("Category positions must be integers")
        if sorted(index.values()) != list(range(size)):
            raise ValueError("Noncontiguous category positions")
        codes = sorted(index, key=index.__getitem__)
    else:
        raise ValueError("Unexpected category index")
    if len(codes) != size or len(set(codes)) != size or any(not isinstance(c, str) for c in codes):
        raise ValueError("Malformed category codes")
    return codes


def sparse_map(raw: Any, size: int) -> dict[int, Any]:
    """Accept the JSON-stat sparse object or dense array representation."""
    if isinstance(raw, list):
        if len(raw) != size:
            raise ValueError("Invalid dense array length")
        return dict(enumerate(raw))
    if not isinstance(raw, dict):
        raise ValueError("Expected JSON-stat value/status array or object")
    result = {}
    for k, value in raw.items():
        pos = int(k)
        if str(pos) != k or not 0 <= pos < size or pos in result:
            raise ValueError("Invalid JSON-stat position")
        result[pos] = value
    return result


def decode(raw: bytes, item: str, units: tuple[str, ...]) -> dict[CellKey, Cell]:
    """Select the exact annual national-accounts measures and retain flags."""
    obj = json.loads(raw)
    dims, sizes = obj["id"], obj["size"]
    required = {"freq", "unit", "na_item", "nace_r2", "geo", "time"}
    if set(dims) != required or len(dims) != len(required) or len(sizes) != len(dims):
        raise ValueError("Unexpected JSON-stat dimensions")
    if any(type(s) is not int or s <= 0 for s in sizes):
        raise ValueError("Invalid JSON-stat dimension sizes")
    categories = [
        ordered_codes(obj["dimension"][dim]["category"], size)
        for dim, size in zip(dims, sizes, strict=True)
    ]
    by_dim = dict(zip(dims, categories, strict=True))
    if ("A" not in by_dim["freq"] or item not in by_dim["na_item"]
            or not set(units).issubset(by_dim["unit"]) or "TOTAL" not in by_dim["nace_r2"]):
        raise ValueError("Requested item, unit, frequency or total missing")
    size = math.prod(sizes)
    values = sparse_map(obj["value"], size)
    statuses = sparse_map(obj.get("status", {}), size)
    result: dict[CellKey, Cell] = {}
    for pos in sorted(values.keys() | statuses.keys()):
        n = pos
        indices = []
        for dimension_size in reversed(sizes):
            n, index = divmod(n, dimension_size)
            indices.append(index)
        row = dict(zip(dims, (codes[i] for codes, i in zip(categories, reversed(indices), strict=True)), strict=True))
        if (row["freq"] != "A" or row["na_item"] != item or row["unit"] not in units
                or row["geo"] not in GEO or int(row["time"]) not in YEARS):
            continue
        key = (row["geo"], int(row["time"]), row["nace_r2"], row["unit"])
        value = values.get(pos)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"Invalid national-accounts observation: {key}")
        flag = statuses.get(pos)
        if flag is not None and not isinstance(flag, str):
            raise ValueError(f"Invalid status flag: {key}")
        result[key] = Cell(None if value is None else float(value), flag or "",
                           "observed" if value is not None else
                           ("explicit_null" if pos in values else "absent"))
    return result


def load_input(level: str, index: int, data_dir: Path, fetch: bool) -> bytes:
    """Read the existing registry contract and verify the selected original file."""
    slug, commit, *names = SOURCES[level]
    name = names[index]
    contract = json.loads((ROOT / "data_sources" / f"{slug}.json").read_text(encoding="utf-8"))
    if (contract["repository"] != REPO or contract["commit"] != commit
            or contract["dataset"] != slug or contract["layer"] != "canonical"):
        raise ValueError(f"Unexpected registry contract: {slug}")
    path = f"raw/{name}"
    matches = [entry for entry in contract["files"] if entry["path"] == path]
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicate contract entry: {path}")
    entry = matches[0]
    expected_url = f"https://raw.githubusercontent.com/{REPO}/{commit}/datasets/{slug}/{path}"
    if entry["url"] != expected_url:
        raise ValueError("Unexpected pinned source URL")
    local = data_dir / slug / path
    if local.is_symlink():
        raise ValueError(f"Refusing symlink: {local}")
    if local.exists():
        raw = local.read_bytes()
        verify_bytes(raw, entry)
    elif fetch:
        raw = download(expected_url)
        verify_bytes(raw, entry)
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_suffix(".part")
        temporary.write_bytes(raw)
        temporary.replace(local)
    else:
        raise FileNotFoundError(f"{local}; run with --fetch first")
    return raw


ABSENT = Cell(None, "", "absent")


def get(cells: dict[CellKey, Cell], geo: str, year: int, sector: str, unit: str) -> Cell:
    """Return an explicitly missing cell if the source has no matching position."""
    return cells.get((geo, year, sector, unit), ABSENT)


def section_rows(level: str, gva: dict[CellKey, Cell], hours: dict[CellKey, Cell]) -> list[Record]:
    """Retain nominal/real GVA and hours at the original classification level."""
    sectors = ("TOTAL", *A10, "C") if level == "a10" else ("TOTAL", *A64, *NESTED)
    result = []
    for geo in GEO:
        for year in YEARS:
            total = get(gva, geo, year, "TOTAL", "CP_MEUR").value
            for sector in sectors:
                nominal = get(gva, geo, year, sector, "CP_MEUR")
                real = get(gva, geo, year, sector, "CLV20_MEUR")
                work = get(hours, geo, year, sector, "THS_HW")
                share = ratio(nominal.value, total)
                productivity = ratio(real.value, work.value)
                result.append({
                    "level": level, "geo": geo, "year": year, "sector": sector,
                    "label": LABELS[sector],
                    "partition_member": sector in (A10 if level == "a10" else A64),
                    "nominal_gva_million_eur": nominal.value,
                    "real_gva_clv2020_million_eur": real.value,
                    "hours_thousand": work.value,
                    "gva_share_pct": None if share is None else 100 * share,
                    "real_gva_per_hour_eur": None if productivity is None else 1000 * productivity,
                    "nominal_presence": nominal.presence, "nominal_flag": nominal.flag,
                    "real_presence": real.presence, "real_flag": real.flag,
                    "hours_presence": work.presence, "hours_flag": work.flag,
                })
    return result


def partition_metrics(values: list[float | None], total: float | None) -> Record:
    """Compute concentration only for a complete, reconciled nonnegative partition.

    A 0.5 percentage point reconciliation tolerance accommodates minor rounding.
    A64 omits section U; never hide its residual or infer that it is zero.
    """
    observed = [v for v in values if v is not None]
    result: Record = {
        "expected_sectors": len(values), "observed_sectors": len(observed),
        "reported_total_gva": total, "observed_sector_sum": math.fsum(observed) if observed else None,
        "residual_pct_total": None, "complete_partition": False, "hhi": None,
        "effective_sectors": None,
    }
    if total is None or total <= 0 or len(observed) != len(values) or not observed:
        return result
    sector_sum = math.fsum(observed)
    residual = 100 * (total - sector_sum) / total
    result["residual_pct_total"] = residual
    if abs(residual) > 0.5 or sector_sum <= 0 or any(v < 0 for v in observed):
        return result
    hhi = math.fsum((v / sector_sum) ** 2 for v in observed)
    result.update(complete_partition=True, hhi=hhi, effective_sectors=1 / hhi)
    return result


def concentration_rows(level: str, gva: dict[CellKey, Cell]) -> list[Record]:
    """Use mutually exclusive A10 or A64 groups; exclude manufacturing overlays."""
    sectors = A10 if level == "a10" else A64
    return [
        {"level": level, "geo": geo, "year": year, **partition_metrics(
            [get(gva, geo, year, code, "CP_MEUR").value for code in sectors],
            get(gva, geo, year, "TOTAL", "CP_MEUR").value)}
        for geo in GEO for year in YEARS
    ]


def group_rows(gva: dict[CellKey, Cell]) -> list[Record]:
    """Sum current-price A64 values only for complete, disjoint service groups."""
    result = []
    for geo in GEO:
        for year in YEARS:
            total = get(gva, geo, year, "TOTAL", "CP_MEUR").value
            for name, sectors in GROUPS.items():
                values = [get(gva, geo, year, s, "CP_MEUR") for s in sectors]
                observed = [c.value for c in values if c.value is not None]
                nominal = math.fsum(observed) if len(observed) == len(sectors) else None
                share = ratio(nominal, total)
                result.append({
                    "geo": geo, "year": year, "group": name, "sectors": ";".join(sectors),
                    "nominal_gva_million_eur": nominal,
                    "gva_share_pct": None if share is None else 100 * share,
                    "observed_sectors": len(observed), "expected_sectors": len(sectors),
                    "flags": ";".join(sorted({c.flag for c in values if c.flag})),
                })
    return result


def log_change(start: float | None, end: float | None) -> float | None:
    """Log points require two strictly positive observed endpoints."""
    if start is None or end is None or start <= 0 or end <= 0:
        return None
    return 100 * (math.log(end) - math.log(start))


def share_decomposition(n0: float | None, n1: float | None, q0: float | None,
                        q1: float | None, tn0: float | None, tn1: float | None,
                        tq0: float | None, tq1: float | None) -> Record:
    """Separate changes in nominal shares into relative volume and deflator terms.

    Uses each section's own chain-linked series, never sums those series.
    All changes are log points, not percentage points or causal effects.
    """
    nominal, real = log_change(n0, n1), log_change(q0, q1)
    total_nominal, total_real = log_change(tn0, tn1), log_change(tq0, tq1)
    result: Record = dict(nominal_share_change_log_points=None,
                          relative_volume_change_log_points=None,
                          relative_deflator_change_log_points=None,
                          identity_residual_log_points=None)
    if any(x is None for x in (nominal, real, total_nominal, total_real)):
        return result
    share_change = nominal - total_nominal
    volume = real - total_real
    deflator = (nominal - real) - (total_nominal - total_real)
    result.update(nominal_share_change_log_points=share_change,
                  relative_volume_change_log_points=volume,
                  relative_deflator_change_log_points=deflator,
                  identity_residual_log_points=share_change - volume - deflator)
    return result


def change_rows(level: str, gva: dict[CellKey, Cell], hours: dict[CellKey, Cell]) -> list[Record]:
    """Compare fixed historical endpoints; keep unavailable estimates blank."""
    windows = ((1995, 2024), (2000, 2024), (2010, 2024), (2019, 2020), (2019, 2024))
    sectors = (*A10, "C") if level == "a10" else (*A64, *NESTED)
    result = []
    for geo in GEO:
        for start, end in windows:
            for sector in sectors:
                n0, n1 = [get(gva, geo, y, sector, "CP_MEUR").value for y in (start, end)]
                q0, q1 = [get(gva, geo, y, sector, "CLV20_MEUR").value for y in (start, end)]
                h0, h1 = [get(hours, geo, y, sector, "THS_HW").value for y in (start, end)]
                tn0, tn1 = [get(gva, geo, y, "TOTAL", "CP_MEUR").value for y in (start, end)]
                tq0, tq1 = [get(gva, geo, y, "TOTAL", "CLV20_MEUR").value for y in (start, end)]
                s0, s1 = ratio(n0, tn0), ratio(n1, tn1)
                result.append({
                    "level": level, "geo": geo, "sector": sector, "start": start, "end": end,
                    "nominal_share_change_pp": None if s0 is None or s1 is None else 100 * (s1 - s0),
                    "real_gva_growth_pct": growth(q0, q1), "hours_growth_pct": growth(h0, h1),
                    "real_gva_per_hour_growth_pct": growth(ratio(q0, h0), ratio(q1, h1)),
                    **share_decomposition(n0, n1, q0, q1, tn0, tn1, tq0, tq1),
                })
    return result


def run(data_dir: Path, output: Path, fetch: bool = False) -> None:
    """Generate deterministic section, group, concentration and change tables."""
    sections, concentrations, changes = [], [], []
    groups = []
    for level in SOURCES:
        gva = decode(load_input(level, 0, data_dir, fetch), "B1G", ("CP_MEUR", "CLV20_MEUR"))
        hours = decode(load_input(level, 1, data_dir, fetch), "EMP_DC", ("THS_HW",))
        sections.extend(section_rows(level, gva, hours))
        concentrations.extend(concentration_rows(level, gva))
        changes.extend(change_rows(level, gva, hours))
        if level == "a64":
            groups = group_rows(gva)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"eurostat_sections.csv": sections, "eurostat_concentration.csv": concentrations,
              "eurostat_groups.csv": groups, "eurostat_endpoint_changes.csv": changes}
    for name, rows in tables.items():
        write_csv(output / name, rows)
    balanced = {
        level: [year for year in YEARS if all(
            row["complete_partition"] for row in concentrations
            if row["level"] == level and row["year"] == year)]
        for level in SOURCES
    }
    summary = {
        "source_commits": {level: source[1] for level, source in SOURCES.items()},
        "geographies": len(GEO), "years": [min(YEARS), max(YEARS)],
        "complete_partition_years_all_geographies": balanced,
        "complete_partitions": {level: sum(
            row["complete_partition"] for row in concentrations if row["level"] == level)
            for level in SOURCES},
        "tables": {name: len(rows) for name, rows in tables.items()},
        "scope": "Descriptive NACE composition; not occupational AI exposure or sector essentiality.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    """Run the analysis from any working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "eurostat")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.output, args.fetch)


if __name__ == "__main__":
    main()
