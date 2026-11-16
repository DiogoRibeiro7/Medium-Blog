"""Capture bounded official INE 0011631 Census 2021 category evidence.

The candidate Porto data query is deliberately *not* pre-certified. This
probe saves exact official response bytes, or a failure receipt, so the
indicator's actual dimensions and categories can be reviewed offline.

Indicator 0011631 reports residence/migration categories, not a named
previous-municipality by current-municipality residential flow matrix.
This module never converts category totals into origin-destination moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from extract_commuting import municipality_codes_from_panel
from ine_commuting_capture import _parse_json, _schema_outline

INDICATOR = "0011631"
PORTO_DICO = "1312"
METADATA_URL = (
    "https://www.ine.pt/ine/json_indicador/pindicaMeta.jsp?"
    "varcd=0011631&lang=PT"
)
PORTO_DATA_CANDIDATE_URL = (
    "https://www.ine.pt/ine/json_indicador/pindica.jsp?"
    "op=2&varcd=0011631&Dim1=S7A2021&Dim2=1312&lang=PT"
)
SOURCE_URLS = {
    "metadata": METADATA_URL,
    "porto_dim2_candidate": PORTO_DATA_CANDIDATE_URL,
}


def municipality_candidate_url(dico: str) -> str:
    """Build an allowlisted 2021 residence-category query for one AMP DICO."""
    if not isinstance(dico, str) or dico not in municipality_codes_from_panel():
        raise ValueError("Expected an exact four-digit AMP municipality DICO")
    return PORTO_DATA_CANDIDATE_URL.replace("Dim2=1312", f"Dim2={dico}")


# The separate full-panel collector is restricted to exactly these 17 codes.
MUNICIPALITY_URLS = {
    f"municipality_{code}": municipality_candidate_url(code)
    for code in sorted(municipality_codes_from_panel()) if code != PORTO_DICO
}
OUTPUT_FILENAMES = {
    "metadata": "ine_0011631_metadata.json",
    "porto_dim2_candidate": "ine_0011631_porto_candidate.json",
}
PROTOCOL = "ine_0011631_2021_porto_category_source_probe_v1"
MAX_BYTES = 12 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0


class HTTPResponse(Protocol):
    """Small response interface needed by the bounded HTTP reader."""

    status: int
    headers: Any

    def geturl(self) -> str: ...

    def read(self, count: int) -> bytes: ...

    def __enter__(self) -> HTTPResponse: ...

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class CaptureResponse:
    """The exact candidate source bytes and effective HTTPS URL."""

    payload: bytes
    final_url: str
    content_type: str


def require_source_url(url: str) -> str:
    """Reject non-INE hosts, changed indicators, and altered query filters."""
    if not isinstance(url, str):
        raise ValueError("INE source URL must be a string")
    source = urlsplit(url)
    if (
        source.scheme != "https"
        or source.netloc != "www.ine.pt"
        or source.username is not None
        or source.password is not None
        or source.fragment
    ):
        raise ValueError("Only exact official INE HTTPS endpoints are permitted")
    try:
        fields = parse_qsl(source.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise ValueError("Invalid INE query parameters") from exc
    if len(fields) != len({name for name, _ in fields}):
        raise ValueError("Duplicate INE query parameter")

    for kind, allowed_url in {**SOURCE_URLS, **MUNICIPALITY_URLS}.items():
        allowed = urlsplit(allowed_url)
        if source.path == allowed.path and dict(fields) == dict(
            parse_qsl(allowed.query, strict_parsing=True)
        ):
            return kind
    raise ValueError("Unrecognised indicator, geography or query parameters")


class SameSourceRedirect(HTTPRedirectHandler):
    """Allow no redirect to a different indicator or municipality scope."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        if require_source_url(newurl) != require_source_url(req.full_url):
            raise ValueError("INE redirect changed source selection")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_source(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_BYTES,
) -> CaptureResponse:
    """Download one read-only, byte-bounded official JSON candidate."""
    require_source_url(url)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 120
    ):
        raise ValueError("Expected a finite timeout in (0, 120] seconds")
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError("Invalid maximum downloaded size")

    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "AMPResidentialCategoryResearchCapture/1.0",
        },
    )
    opener = build_opener(SameSourceRedirect())
    with opener.open(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise ValueError(f"INE source HTTP {response.status}")
        final_url = response.geturl()
        if require_source_url(final_url) != require_source_url(url):
            raise ValueError("INE source response changed requested endpoint")
        claimed = response.headers.get("Content-Length")
        if claimed is not None and (
            not str(claimed).isdigit() or int(claimed) > max_bytes
        ):
            raise ValueError("Invalid or oversized declared content length")
        raw = response.read(max_bytes + 1)
        content_type = str(response.headers.get("Content-Type", ""))
    if not raw or len(raw) > max_bytes:
        raise ValueError("INE source was empty or exceeded the response limit")
    _parse_json(raw)
    return CaptureResponse(raw, final_url, content_type)


def _shape_hints(payload: bytes) -> dict[str, Any]:
    """Describe JSON structure without classifying unreviewed categories."""
    parsed = _parse_json(payload)
    indicator = parsed[0] if isinstance(parsed, list) and len(parsed) == 1 else parsed
    observed_code: str | None = None
    periods: list[str] = []
    dimension_ids: list[str] = []
    if isinstance(indicator, dict):
        candidate = indicator.get("IndicadorCod")
        observed_code = candidate if isinstance(candidate, str) else None
        data = indicator.get("Dados")
        if isinstance(data, dict):
            periods = sorted(str(period) for period in data)
        dimensions = indicator.get("Dimensoes")
        if isinstance(dimensions, dict):
            descriptions = dimensions.get("Descricao_Dim")
            if isinstance(descriptions, list):
                dimension_ids = sorted(
                    str(row["dim_num"])
                    for row in descriptions
                    if isinstance(row, dict) and "dim_num" in row
                )
    return {
        "schema_outline": _schema_outline(parsed),
        "observed_indicator_code": observed_code,
        "observed_period_keys": periods,
        "observed_dimension_numbers": dimension_ids,
        "indicator_code_matches_request": observed_code == INDICATOR,
        "geographic_filter_effect_verified": False,
        "migration_categories_verified": False,
    }


def capture(
    output: Path,
    *,
    fetcher: Callable[[str], CaptureResponse] = download_source,
) -> dict[str, Any]:
    """Capture metadata and one Porto query, preserving any failures in JSON."""
    if not isinstance(output, Path):
        raise TypeError("Output directory must be a Path")
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for kind, url in SOURCE_URLS.items():
        filename = OUTPUT_FILENAMES[kind]
        path = output / filename
        path.unlink(missing_ok=True)
        try:
            response = fetcher(url)
            if not isinstance(response, CaptureResponse):
                raise TypeError("Fetcher must return a CaptureResponse")
            if require_source_url(response.final_url) != kind:
                raise ValueError("Source URL changed the selected query")
            if not isinstance(response.payload, bytes) or not 0 < len(response.payload) <= MAX_BYTES:
                raise ValueError("Source payload must be nonempty bounded bytes")
            hints = _shape_hints(response.payload)
            path.write_bytes(response.payload)
            entries.append({
                "kind": kind,
                "status": "captured_unverified",
                "requested_url": url,
                "response_url": response.final_url,
                "filename": filename,
                "content_type": response.content_type,
                "source_sha256": hashlib.sha256(response.payload).hexdigest(),
                "source_bytes": len(response.payload),
                **hints,
            })
        except (ValueError, TypeError, OSError, HTTPError, URLError, TimeoutError) as exc:
            entries.append({
                "kind": kind,
                "status": "failed",
                "requested_url": url,
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            })

    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "indicator_id": INDICATOR,
        "candidate_period": "2021",
        "candidate_destination_dico": PORTO_DICO,
        "requested_kinds": list(SOURCE_URLS),
        "completed_json_captures": sum(row["status"] == "captured_unverified" for row in entries),
        "source_semantics_verified": False,
        "official_previous_municipality_observed": False,
        "directed_residential_origin_destination_counts_reconciled": False,
        "housing_causality_established": False,
        "items": entries,
    }
    (output / "probe_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Save both candidate responses and their manifest for later review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 120:
        parser.error("--timeout-seconds must be finite and between 0 and 120")

    def bounded_fetch(url: str) -> CaptureResponse:
        """Apply one validated timeout to both requests."""
        return download_source(url, timeout_seconds=args.timeout_seconds)

    result = capture(args.output, fetcher=bounded_fetch)
    print(f"INE {INDICATOR}: {result['completed_json_captures']}/2 "
          "unverified JSON source candidates saved")
    if result["completed_json_captures"] != 2:
        parser.exit(1, "Missing source candidate(s); inspect probe_manifest.json.\n")


if __name__ == "__main__":
    main()
