"""Capture unmodified INE indicator 0012340 JSON and inspection metadata.

The data URL is a *candidate* Census 2021 API slice; its dimension codes and
payload structure have not yet been independently verified. Capturing JSON
never establishes source-semantic correctness or reproduces the frozen network.

Only a manual, explicitly invoked run should make external requests. CI tests
exercise this module with synthetic, in-memory responses and make no requests.
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

INDICATOR = "0012340"
META_URL = (
    "https://www.ine.pt/ine/json_indicador/pindicaMeta.jsp?"
    "varcd=0012340&lang=PT"
)
DATA_URL = (
    "https://www.ine.pt/ine/json_indicador/pindica.jsp?"
    "op=2&varcd=0012340&Dim1=S7A2021&lang=PT"
)
SOURCES = {"metadata": META_URL, "data_2021_candidate": DATA_URL}
MAX_BYTES = 64 * 1024 * 1024  # Fail rather than retrieve unbounded data.
DEFAULT_TIMEOUT_SECONDS = 35.0


class HTTPResponse(Protocol):
    """Minimal streaming response interface used by the capture client."""

    status: int
    headers: Any

    def read(self, amount: int) -> bytes: ...

    def geturl(self) -> str: ...

    def __enter__(self) -> HTTPResponse: ...

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class CapturedJSON:
    """The exact JSON response bytes and auditable transport metadata."""

    payload: bytes
    final_url: str
    content_type: str


def _endpoint_kind(url: str) -> str:
    """Reject altered indicators, extra parameters and non-INE redirect URLs."""
    if not isinstance(url, str):
        raise ValueError("Source URL must be a string")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "www.ine.pt"
            or parsed.fragment or parsed.username or parsed.password):
        raise ValueError("Expected HTTPS official INE endpoint")
    try:
        params = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise ValueError("Invalid official query string") from exc
    if len(params) != len({k for k, _ in params}):
        raise ValueError("Duplicate source query parameters")
    options = dict(params)
    if (parsed.path == "/ine/json_indicador/pindicaMeta.jsp"
            and options == {"varcd": INDICATOR, "lang": "PT"}):
        return "metadata"
    if (parsed.path == "/ine/json_indicador/pindica.jsp"
            and options == {"op": "2", "varcd": INDICATOR,
                            "Dim1": "S7A2021", "lang": "PT"}):
        return "data_2021_candidate"
    raise ValueError("Unrecognised or altered INE capture endpoint")


class RestrictedRedirect(HTTPRedirectHandler):
    """Permit redirects only to a semantically identical official INE URL."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        if _endpoint_kind(newurl) != _endpoint_kind(req.full_url):
            raise ValueError("INE redirect changes endpoint, indicator or period")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _parse_json(data: bytes) -> dict[str, Any] | list[Any]:
    """Read UTF-8 JSON while rejecting duplicate keys and non-JSON constants."""
    if not data:
        raise ValueError("Official response is empty")

    def check_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate JSON key: {key!r}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")

    try:
        decoded = json.loads(
            data.decode("utf-8-sig"),
            object_pairs_hook=check_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Response was not valid UTF-8 JSON") from exc
    if not isinstance(decoded, (dict, list)) or len(decoded) == 0:
        raise ValueError("Expected a nonempty official JSON object or array")
    return decoded


def download_json(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_BYTES,
) -> CapturedJSON:
    """Download one allowlisted INE URL using a bounded, read-only GET."""
    requested_kind = _endpoint_kind(url)
    if (not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 120):
        raise ValueError("timeout_seconds must be finite and in (0, 120]")
    if (isinstance(max_bytes, bool) or not isinstance(max_bytes, int)
            or not 0 < max_bytes <= MAX_BYTES):
        raise ValueError("max_bytes must be a positive bounded integer")
    opener = build_opener(RestrictedRedirect())
    request = Request(
        url,
        method="GET",
        headers={"Accept": "application/json", "User-Agent": "PortoHousingINECapture/1.0"},
    )
    with opener.open(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise ValueError(f"INE response status {response.status} is not 200")
        final_url = response.geturl()
        if _endpoint_kind(final_url) != requested_kind:
            raise ValueError("Redirected URL did not preserve the official data slice")
        declared_length = response.headers.get("Content-Length")
        if declared_length is not None:
            try:
                parsed_length = int(declared_length)
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header") from exc
            if parsed_length < 0 or parsed_length > max_bytes:
                raise ValueError("Response Content-Length exceeds limit")
        payload: bytes = response.read(max_bytes + 1)
        content_type = str(response.headers.get("Content-Type", ""))
    if len(payload) > max_bytes:
        raise ValueError("Response exceeds byte limit")
    _parse_json(payload)
    return CapturedJSON(payload=payload, final_url=final_url,
                        content_type=content_type)


def _schema_outline(root: dict[str, Any] | list[Any]) -> dict[str, Any]:
    """Show only structural hints; make no assumption about INE field names."""
    record = root[0] if isinstance(root, list) else root
    outline: dict[str, Any] = {
        "root_type": "array" if isinstance(root, list) else "object",
        "root_items": len(root),
        "first_item_type": type(record).__name__,
    }
    if isinstance(record, dict):
        outline["first_item_keys"] = sorted(record)[:50]
        nested = record.get("Dados")
        if isinstance(nested, dict):
            outline["dados_period_keys"] = sorted(nested)[:30]
            outline["dados_period_sizes"] = {
                str(year): len(items) for year, items in sorted(nested.items())[:30]
                if isinstance(items, (list, dict))
            }
    return outline


def capture_sources(
    directory: Path,
    *,
    fetcher: Callable[[str], CapturedJSON] = download_json,
) -> dict[str, Any]:
    """Save each valid raw response and an honest manifest even on failure.

    A JSON capture is an unreviewed *transport observation*, never an official
    semantic verification. Unexpected responses remain identifiable as such.
    """
    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path")
    directory.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict[str, Any]] = {}
    for kind, url in SOURCES.items():
        filename = f"ine_0012340_{kind}.json"
        # An output directory may be reused: never treat stale bytes as current.
        (directory / filename).unlink(missing_ok=True)
        try:
            response = fetcher(url)
            if not isinstance(response, CapturedJSON):
                raise TypeError("Fetcher must return CapturedJSON")
            if _endpoint_kind(response.final_url) != kind:
                raise ValueError("Captured source URL does not match the requested data slice")
            parsed = _parse_json(response.payload)
            (directory / filename).write_bytes(response.payload)
            sources[kind] = {
                "status": "json_saved_schema_unreviewed",
                "requested_url": url,
                "response_url": response.final_url,
                "filename": filename,
                "sha256": hashlib.sha256(response.payload).hexdigest(),
                "bytes": len(response.payload),
                "http_content_type": response.content_type,
                "schema_outline": _schema_outline(parsed),
            }
        except (ValueError, TypeError, HTTPError, URLError, TimeoutError, OSError) as exc:
            sources[kind] = {
                "status": "capture_failed",
                "requested_url": url,
                "error_type": type(exc).__name__,
                "error": str(exc)[:400],
            }
    success = all(s["status"] == "json_saved_schema_unreviewed"
                  for s in sources.values())
    manifest: dict[str, Any] = {
        "capture_version": 1,
        "indicator_id": INDICATOR,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "captured_unreconciled" if success else "capture_incomplete",
        "semantics_verified": False,
        "frozen_pairs_reconciled": False,
        "candidate_period": "2021",
        "candidate_data_endpoint_not_validated": True,
        "source_semantics": "municipal work/study interactions, NOT residential relocation",
        "sources": sources,
        "required_next_gate": (
            "Independently review official period, dimensions, geography and unit; "
            "only then adapt/re-run the pairwise reconciliation"
        ),
    }
    (directory / "capture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    """Run a manually requested official capture, not a reconciliation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = capture_sources(args.output)
    print(f"INE indicator {INDICATOR}: {manifest['status']}")
    print("Saved capture_manifest.json; source meaning and pair counts not verified.")
    if manifest["status"] != "captured_unreconciled":
        parser.exit(1, "At least one INE response was inaccessible or not JSON. Review manifest.\n")


if __name__ == "__main__":
    main()