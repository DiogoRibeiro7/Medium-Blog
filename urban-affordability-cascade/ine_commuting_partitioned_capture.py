"""Capture INE 0012340 in small geographic slices without asserting source validity.

The full 2021 JSON request timed out on 12 November 2026. Official indicator
metadata was captured successfully and identifies the 17 AMP DICO codes in
both municipal dimensions. This script tries an explicitly *candidate* Dim2
filter for a bounded range of municipalities. INE might ignore or reject this
filter. No returned values are admitted into the research panel until the
slice semantics, coverage, and totals are independently reconciled.

No network access is used by the accompanying offline tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.parse import parse_qs, urlencode, urlsplit

from extract_commuting import municipality_codes_from_panel
from ine_commuting_capture import (
    CapturedJSON,
    MAX_BYTES,
    META_URL,
    _parse_json,
    _schema_outline,
    download_json,
)

ROOT = Path(__file__).resolve().parent
INDICATOR = "0012340"
EXPECTED_METADATA_SHA256 = (
    "4a86847af3baae2b7d6beeeb6c4832a4becca9dc77b3d688181430912e2d48a0"
)
MAX_SLICES_PER_RUN = 3
DEFAULT_TIMEOUT_SECONDS = 35.0


def candidate_slice_url(dico: str, *, dimension: str = "Dim2") -> str:
    """Allow only one of the declared official municipality dimensions."""
    if not isinstance(dico, str) or not re.fullmatch(r"\d{4}", dico):
        raise ValueError("DICO must be a four-digit municipality code")
    if dimension not in {"Dim2", "Dim3"}:
        raise ValueError("Municipality slice must filter Dim2 or Dim3")
    query = urlencode([
        ("op", "2"), ("varcd", INDICATOR),
        ("Dim1", "S7A2021"), (dimension, dico), ("lang", "PT"),
    ])
    return "https://www.ine.pt/ine/json_indicador/pindica.jsp?" + query


def require_slice_url(url: str, *, dico: str, dimension: str) -> None:
    """Ensure the request or redirect cannot change the selected indicator."""
    if not isinstance(url, str):
        raise ValueError("Source URL must be a string")
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc != "www.ine.pt"
            or parts.path != "/ine/json_indicador/pindica.jsp" or parts.fragment
            or parts.username or parts.password):
        raise ValueError("Unexpected INE municipality slice URL")
    actual = parse_qs(parts.query, strict_parsing=True)
    expected = parse_qs(urlsplit(candidate_slice_url(dico, dimension=dimension)).query)
    if actual != expected or any(len(values) != 1 for values in actual.values()):
        raise ValueError("INE slice URL changed the candidate filter")



class SliceRedirect(HTTPRedirectHandler):
    """Reject redirects changing requested municipality, period or publisher."""

    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str,
        headers: Any, newurl: str,
    ) -> Request | None:
        query = parse_qs(urlsplit(req.full_url).query)
        dimension = "Dim2" if "Dim2" in query else "Dim3"
        dico = query[dimension][0]
        require_slice_url(newurl, dico=dico, dimension=dimension)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_slice(
    url: str, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_BYTES,
) -> CapturedJSON:
    """GET a single bounded municipality slice from an exact official HTTPS URL."""
    parsed = parse_qs(urlsplit(url).query)
    dimension = "Dim2" if "Dim2" in parsed else "Dim3"
    if dimension not in parsed or len(parsed[dimension]) != 1:
        raise ValueError("Missing exactly one municipality filter")
    code = parsed[dimension][0]
    require_slice_url(url, dico=code, dimension=dimension)
    if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120):
        raise ValueError("Invalid finite timeout")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError("Invalid maximum response bytes")
    opener = build_opener(SliceRedirect())
    request = Request(url, method="GET", headers={
        "Accept": "application/json", "User-Agent": "PortoHousingINEPartitionCapture/1.0",
    })
    with opener.open(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise ValueError(f"INE slice HTTP {response.status}")
        final_url = response.geturl()
        require_slice_url(final_url, dico=code, dimension=dimension)
        claimed = response.headers.get("Content-Length")
        if claimed is not None:
            if not str(claimed).isdigit() or int(claimed) > max_bytes:
                raise ValueError("INE slice Content-Length invalid or exceeds limit")
        raw = response.read(max_bytes + 1)
        content_type = str(response.headers.get("Content-Type", ""))
    if not raw or len(raw) > max_bytes:
        raise ValueError("INE slice is empty or exceeds byte limit")
    _parse_json(raw)
    return CapturedJSON(raw, final_url, content_type)


def download_pinned_metadata() -> bytes:
    """Retrieve candidate official metadata; validate its substantive definitions."""
    response = download_json(META_URL)
    validate_metadata(response.payload, municipality_codes_from_panel())
    return response.payload

def validate_metadata(raw: bytes, codes: dict[str, str]) -> dict[str, Any]:
    """Check declared indicator definitions and all 17 municipality codes.

    INE records a volatile DataExtracao timestamp; raw byte hashes are retained
    for provenance but are not stable identifiers of indicator definitions.
    """
    digest = hashlib.sha256(raw).hexdigest()
    parsed = _parse_json(raw)
    if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], dict):
        raise ValueError("INE metadata must contain exactly one indicator object")
    indicator = parsed[0]
    if (indicator.get("IndicadorCod") != INDICATOR
            or indicator.get("PrimeiroPeriodo") != "2021"
            or indicator.get("UltimoPeriodo") != "2021"
            or indicator.get("UnidadeMedida") != "Número (N.º)"):
        raise ValueError("Official indicator period, code or unit changed")
    dimensions = indicator.get("Dimensoes")
    if not isinstance(dimensions, dict):
        raise ValueError("INE indicator dimensions are missing")
    descriptions = dimensions.get("Descricao_Dim")
    if not isinstance(descriptions, list) or {
        str(item.get("dim_num")) for item in descriptions if isinstance(item, dict)
    } != {"1", "2", "3"}:
        raise ValueError("Expected three official indicator dimensions")
    categories = dimensions.get("Categoria_Dim")
    if not isinstance(categories, list) or len(categories) != 1 or not isinstance(categories[0], dict):
        raise ValueError("INE dimension categories have an unfamiliar structure")
    category_index = categories[0]
    for dimension in (2, 3):
        for code, name in codes.items():
            rows = category_index.get(f"Dim_Num{dimension}_{code}")
            if (not isinstance(rows, list) or len(rows) != 1
                    or not isinstance(rows[0], dict)
                    or rows[0].get("dim_num") != str(dimension)
                    or rows[0].get("categ_cod") != code
                    or rows[0].get("categ_dsg") != name
                    or rows[0].get("categ_nivel") != "5"):
                raise ValueError(f"INE metadata missing/mismatched AMP municipality {code} {name}")
    return {
        "indicator": INDICATOR,
        "period": "2021",
        "metadata_sha256": digest,
        "metadata_unit": indicator["UnidadeMedida"],
        "geographic_dimensions": [2, 3],
        "amp_municipalities_validated_in_both_dimensions": len(codes),
    }


def _status_candidate(raw: bytes, code: str, dimension: str) -> dict[str, Any]:
    """Provide descriptive shape checks, never assume a complete filtered grid."""
    value = _parse_json(raw)
    outline = _schema_outline(value)
    selected: list[str] = []
    if isinstance(value, list) and value and isinstance(value[0], dict):
        data = value[0].get("Dados")
        if isinstance(data, dict):
            selected = sorted(str(x) for x in data)
    return {
        "requested_dico": code,
        "requested_dimension": dimension,
        "schema_outline": outline,
        "observed_period_keys": selected,
        "filter_effect_verified": False,
        "municipal_pair_counts_reconciled": False,
    }


def capture_partition(
    output: Path,
    *,
    metadata_bytes: bytes,
    start_index: int = 0,
    count: int = 2,
    dimension: str = "Dim2",
    fetcher: Callable[[str], CapturedJSON] = download_slice,
) -> dict[str, Any]:
    """Fetch a bounded DICO slice range and preserve all successes and failures."""
    codes = municipality_codes_from_panel()
    ordered = sorted(codes)
    if (type(start_index) is not int or type(count) is not int
            or start_index < 0 or start_index >= len(ordered)
            or count < 1 or count > MAX_SLICES_PER_RUN
            or start_index + count > len(ordered)):
        raise ValueError("Select 1–3 valid, contiguous AMP codes within the 17-code grid")
    if dimension not in ("Dim2", "Dim3"):
        raise ValueError("Unknown official geographic dimension")
    source = validate_metadata(metadata_bytes, codes)
    if not isinstance(output, Path):
        raise TypeError("Output must be a Path")
    output.mkdir(parents=True, exist_ok=True)
    target_codes = ordered[start_index:start_index + count]
    rows: list[dict[str, Any]] = []
    for code in target_codes:
        url = candidate_slice_url(code, dimension=dimension)
        filename = f"ine_0012340_{dimension.lower()}_{code}_candidate.json"
        path = output / filename
        path.unlink(missing_ok=True)
        try:
            response = fetcher(url)
            if not isinstance(response, CapturedJSON):
                raise TypeError("Expected a CapturedJSON source response")
            require_slice_url(response.final_url, dico=code, dimension=dimension)
            if not response.payload or len(response.payload) > MAX_BYTES:
                raise ValueError("Unbounded or empty source response")
            extra = _status_candidate(response.payload, code, dimension)
            path.write_bytes(response.payload)
            rows.append({
                "status": "captured_unverified",
                "code": code, "municipality": codes[code],
                "candidate_url": url, "response_url": response.final_url,
                "filename": filename,
                "sha256": hashlib.sha256(response.payload).hexdigest(),
                "size_bytes": len(response.payload),
                **extra,
            })
        except (ValueError, TypeError, OSError, HTTPError, URLError, TimeoutError) as exc:
            rows.append({
                "status": "failed",
                "code": code, "municipality": codes[code],
                "candidate_url": url, "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            })
    report: dict[str, Any] = {
        "capture_protocol": "ine_0012340_2021_dico_partition_candidate_v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_codes": target_codes,
        "capture_count": count,
        "completed_json_captures": sum(r["status"] == "captured_unverified" for r in rows),
        "dimension": dimension,
        "source_metadata": source,
        "source_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "metadata_dimensions_checked_against_panel": True,
        "metadata_matches_archived_2026_11_12_sha256": (
            hashlib.sha256(metadata_bytes).hexdigest() == EXPECTED_METADATA_SHA256
        ),
        "candidate_slice_semantics_verified": False,
        "all_136_pairs_reconciled": False,
        "source_data_is_residential_migration": False,
        "items": rows,
    }
    (output / "partition_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run one manually selected small geographic slice batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-json", type=Path, default=None,
                        help="Optional downloaded metadata JSON; otherwise fetch and validate official metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--dimension", choices=["Dim2", "Dim3"], default="Dim2")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                        help="Bounded per-slice INE read timeout, >0 and <=120 seconds")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 120:
        parser.error("--timeout-seconds must be finite and in (0, 120]")

    def bounded_fetch(url: str) -> CapturedJSON:
        """Apply the validated per-slice timeout without changing capture semantics."""
        return download_slice(url, timeout_seconds=args.timeout_seconds)

    metadata_bytes = args.metadata_json.read_bytes() if args.metadata_json is not None else download_pinned_metadata()
    result = capture_partition(
        args.output, metadata_bytes=metadata_bytes,
        start_index=args.start_index, count=args.count, dimension=args.dimension,
        fetcher=bounded_fetch,
    )
    print(f"INE 0012340: {result['completed_json_captures']}/{result['capture_count']} "
          "candidate geographic slices saved, all unverified.")
    if result["completed_json_captures"] != result["capture_count"]:
        parser.exit(1, "Some candidate source slices failed; inspect partition_manifest.json.\n")


if __name__ == "__main__":
    main()
