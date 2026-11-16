"""Capture and validate 2021 INE 0011631 residence categories for 17 AMP towns.

The source measures where current residents lived *by category* at the
previous reference date. It does not disclose the named municipality of
origin, directed OD flows, reasons for moving or a housing-price effect.
Original bytes and an explicit per-source receipt are retained unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError

from extract_commuting import municipality_codes_from_panel
from ine_commuting_capture import _parse_json
from ine_residential_categories_probe import (
    INDICATOR,
    MAX_BYTES,
    CaptureResponse,
    download_source,
    municipality_candidate_url,
)

CAPTURE_PROTOCOL = "ine_0011631_2021_amp_municipal_categories_v1"
PERIOD = "2021"
UNIT = "Número (N.º)"
SEX_NAMES = {"1": "H", "2": "M", "T": "HM"}
# The official source contains this spelling of "munícipio" in category 33.
CATEGORY_NAMES = {
    "1": "Ainda não tinha nascido",
    "2": "Sem alteração de residência",
    "3": "Com alteração de residência",
    "31": "Dentro da mesma freguesia",
    "32": "Proveniente de outra freguesia do mesmo município",
    "33": "Proveniente de outro munícipio",
    "34": "Proveniente do estrangeiro",
    "T": "Total",
}
MAX_MUNICIPALITIES_PER_RUN = 17


def _single_indicator(raw: bytes) -> dict[str, Any]:
    """Require exactly one successful indicator from a JSON response."""
    root = _parse_json(raw)
    if not isinstance(root, list) or len(root) != 1 or not isinstance(root[0], dict):
        raise ValueError("Expected a single INE indicator JSON array")
    indicator = root[0]
    if indicator.get("IndicadorCod") != INDICATOR:
        raise ValueError("Unexpected INE indicator ID")
    success = indicator.get("Sucesso")
    if not isinstance(success, dict) or success.get("Verdadeiro") != [{"Msg": "OK"}]:
        raise ValueError("INE indicator did not report success")
    return indicator


def validate_metadata(raw: bytes, names: Mapping[str, str]) -> dict[str, Any]:
    """Validate actual indicator dimensions, period and all AMP identities."""
    indicator = _single_indicator(raw)
    if (
        indicator.get("PrimeiroPeriodo") != "2011"
        or indicator.get("UltimoPeriodo") != PERIOD
        or indicator.get("UnidadeMedida") != UNIT
    ):
        raise ValueError("INE metadata changed its period or measurement unit")
    dimensions = indicator.get("Dimensoes")
    if not isinstance(dimensions, dict):
        raise ValueError("Metadata lacks dimension descriptions")
    descriptions = dimensions.get("Descricao_Dim")
    if not isinstance(descriptions, list) or {
        item.get("dim_num") for item in descriptions if isinstance(item, dict)
    } != {"1", "2", "3", "4"}:
        raise ValueError("Expected period, residence, sex and migration-category dimensions")
    categories = dimensions.get("Categoria_Dim")
    if (
        not isinstance(categories, list)
        or len(categories) != 1
        or not isinstance(categories[0], dict)
    ):
        raise ValueError("Unexpected INE category metadata structure")
    index = categories[0]

    def require_category(dim: int, code: str, label: str) -> None:
        """Require an unambiguous code-label pair in the official dimensions."""
        items = index.get(f"Dim_Num{dim}_{code}")
        if (
            not isinstance(items, list)
            or len(items) != 1
            or not isinstance(items[0], dict)
            or items[0].get("dim_num") != str(dim)
            or items[0].get("categ_cod") != code
            or items[0].get("categ_dsg") != label
        ):
            raise ValueError(f"Changed or missing official category {dim}:{code}")

    require_category(1, "S7A2021", PERIOD)
    for code, name in names.items():
        require_category(2, code, name)
    for code, label in SEX_NAMES.items():
        require_category(3, code, label)
    for code, label in CATEGORY_NAMES.items():
        require_category(4, code, label)
    return {
        "indicator": INDICATOR,
        "year": PERIOD,
        "unit": UNIT,
        "geographic_dimension": 2,
        "sex_dimension": 3,
        "residential_category_dimension": 4,
        "validated_amp_municipalities": len(names),
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
    }


def parse_municipality(raw: bytes, code: str, names: Mapping[str, str]) -> dict[str, int]:
    """Require all 24 sex/category cells and exact published accounting sums."""
    if code not in names:
        raise ValueError("Unknown AMP destination municipality")
    indicator = _single_indicator(raw)
    data = indicator.get("Dados")
    if not isinstance(data, dict) or set(data) != {PERIOD}:
        raise ValueError("INE data must contain exactly the 2021 reference period")
    rows = data[PERIOD]
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("Each municipality must contain 24 sex/category cells")

    cells: dict[tuple[str, str], int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Unexpected INE municipal count row")
        sex, category = row.get("dim_3"), row.get("dim_4")
        raw_count = row.get("valor")
        if (
            row.get("geocod") != code
            or row.get("geodsg") != names[code]
            or sex not in SEX_NAMES
            or row.get("dim_3_t") != SEX_NAMES[sex]
            or category not in CATEGORY_NAMES
            or row.get("dim_4_t") != CATEGORY_NAMES[category]
            or not isinstance(raw_count, str)
            or re.fullmatch(r"0|[1-9][0-9]*", raw_count) is None
            or (sex, category) in cells
        ):
            raise ValueError("INE municipal identity, category or count is inconsistent")
        cells[sex, category] = int(raw_count)

    if set(cells) != {(sex, category) for sex in SEX_NAMES for category in CATEGORY_NAMES}:
        raise ValueError("INE municipal sex/category grid is incomplete")
    for category in CATEGORY_NAMES:
        if cells["1", category] + cells["2", category] != cells["T", category]:
            raise ValueError("INE male/female categories do not sum to both-sex totals")
    for sex in SEX_NAMES:
        if cells[sex, "T"] != cells[sex, "1"] + cells[sex, "2"] + cells[sex, "3"]:
            raise ValueError("INE birth/residence category totals do not reconcile")
        if cells[sex, "3"] != sum(cells[sex, sub] for sub in ("31", "32", "33", "34")):
            raise ValueError("INE mover subgroup totals do not reconcile")
    return {category: cells["T", category] for category in CATEGORY_NAMES}


def capture_amp(
    output: Path,
    *,
    metadata_bytes: bytes,
    start_index: int = 0,
    count: int = MAX_MUNICIPALITIES_PER_RUN,
    fetcher: Callable[[str], CaptureResponse] = download_source,
) -> dict[str, Any]:
    """Capture a bounded region, checking each source without invented values."""
    names = municipality_codes_from_panel()
    codes = sorted(names)
    if (
        type(start_index) is not int
        or type(count) is not int
        or start_index < 0
        or count < 1
        or count > MAX_MUNICIPALITIES_PER_RUN
        or start_index + count > len(codes)
    ):
        raise ValueError("Select 1–17 contiguous AMP destination municipalities")
    if not isinstance(output, Path):
        raise TypeError("Output directory must be a Path")
    metadata = validate_metadata(metadata_bytes, names)
    output.mkdir(parents=True, exist_ok=True)
    selected = codes[start_index:start_index + count]
    items: list[dict[str, Any]] = []
    for code in selected:
        url = municipality_candidate_url(code)
        name = f"ine_0011631_dim2_{code}_2021.json"
        path = output / name
        path.unlink(missing_ok=True)
        try:
            response = fetcher(url)
            if not isinstance(response, CaptureResponse):
                raise TypeError("Expected a CaptureResponse")
            if response.final_url != url or not isinstance(response.payload, bytes):
                raise ValueError("INE source URL or response bytes changed")
            if not 0 < len(response.payload) <= MAX_BYTES:
                raise ValueError("Empty or oversized INE source response")
            # Preserve original valid JSON even if its category shape fails.
            _parse_json(response.payload)
            path.write_bytes(response.payload)
            digest = hashlib.sha256(response.payload).hexdigest()
            counts = parse_municipality(response.payload, code, names)
            items.append({
                "status": "validated_2021_migration_categories",
                "code": code,
                "municipality": names[code],
                "source_url": url,
                "response_url": response.final_url,
                "filename": name,
                "source_sha256": digest,
                "source_bytes": len(response.payload),
                "reference_period": PERIOD,
                "municipal_origin_identified": False,
                "both_sexes_counts": counts,
            })
        except (ValueError, TypeError, OSError, HTTPError, URLError, TimeoutError) as exc:
            items.append({
                "status": "failed",
                "code": code,
                "municipality": names[code],
                "source_url": url,
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
                "captured_raw_unverified": path.exists(),
            })

    report: dict[str, Any] = {
        "protocol": CAPTURE_PROTOCOL,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": metadata,
        "metadata_source_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "requested_codes": selected,
        "requested_count": len(selected),
        "validated_count": sum(item["status"] == "validated_2021_migration_categories" for item in items),
        "data_semantics": "2021 Census residence/migration categories, not named origin-to-destination flows",
        "full_residential_origin_destination_matrix_observed": False,
        "housing_displacement_causality_established": False,
        "items": items,
    }
    (output / "municipal_capture_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Capture 2021 residential destination-category counts in bounded slices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=MAX_MUNICIPALITIES_PER_RUN)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not 0 < args.timeout_seconds <= 120:
        parser.error("Timeout must be in (0,120] seconds")

    def bounded_fetch(url: str) -> CaptureResponse:
        """Use the explicit per-municipality read timeout."""
        return download_source(url, timeout_seconds=args.timeout_seconds)

    result = capture_amp(
        args.output,
        metadata_bytes=args.metadata_json.read_bytes(),
        start_index=args.start_index,
        count=args.count,
        fetcher=bounded_fetch,
    )
    print(f"INE 0011631: {result['validated_count']}/{result['requested_count']} "
          "AMP municipality category responses validated")
    if result["validated_count"] != result["requested_count"]:
        parser.exit(1, "Some source slices were not validated; inspect archived evidence.\n")


if __name__ == "__main__":
    main()
