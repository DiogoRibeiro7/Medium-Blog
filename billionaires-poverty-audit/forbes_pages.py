"""Retrieve a Forbes annual list with an explicitly verified page fallback.

A large one-shot request may receive upstream 503 responses. This module
first requests that feed and, on a transient failure only, retries through
bounded, explicitly checked pages. No individual profiles are persisted.
A valid response is not evidence that its content is a frozen March vintage.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from resilient_fetch import fetch_bytes


class ForbesSourceError(ValueError):
    """Reject inconsistent Forbes pages or unsupported response shapes."""


def get_people(data: bytes) -> list[dict[str, Any]]:
    """Extract people from documented Forbes JSON layouts without guessing."""
    if not isinstance(data, bytes):
        raise TypeError("Forbes response must be bytes")
    try:
        document: Any = json.loads(data)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ForbesSourceError("Forbes response is not JSON") from exc
    if not isinstance(document, dict):
        raise ForbesSourceError("Forbes response is not a JSON object")
    section = document.get("personList")
    people = section.get("persons") if isinstance(section, dict) else document.get("persons")
    if not isinstance(people, list):
        raise ForbesSourceError("Forbes response contains no persons array")
    if any(not isinstance(record, dict) for record in people):
        raise ForbesSourceError("Forbes persons array contains invalid entries")
    return people


def page_url(url: str, *, start: int, limit: int) -> str:
    """Build a deterministic query while preserving the annual list path."""
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("Forbes source URL must be HTTPS")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a nonnegative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params.update({"start": str(start), "limit": str(limit)})
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment)
    )


def retrieve_annual_list(
    url: str,
    *,
    expected_count: int,
    page_limit: int = 250,
    fetcher: Callable[..., bytes] = fetch_bytes,
) -> tuple[bytes, dict[str, Any]]:
    """Retrieve an exact-size annual list, with audit metadata per response.

    The fallback is attempted only after a transient one-shot download error.
    Neither route bypasses the caller's record-count and identity validation.

    Returns:
        A JSON payload in the original personList layout (page fallback is
        assembled in memory), and a provenance record with per-page hashes.

    Raises:
        ForbesSourceError: If the primary feed has inconsistent page sizes,
            duplicate URI identifiers across pages, or invalid JSON.
        RuntimeError: If remote access fails after bounded retries.
    """
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")
    if isinstance(page_limit, bool) or not isinstance(page_limit, int) or page_limit < 1:
        raise ValueError("page_limit must be a positive integer")

    try:
        raw: bytes = fetcher(
            url, name="forbes_annual_full.json",
            timeout_seconds=75.0, max_attempts=2,
        )
    except RuntimeError as error:
        first_error = f"{type(error).__name__}: {error}"
    else:
        people = get_people(raw)
        if len(people) != expected_count:
            raise ForbesSourceError(
                f"One-shot source has {len(people)} people, expected "
                f"{expected_count}. Do not silently accept a different vintage."
            )
        return raw, {
            "mode": "single_response",
            "url": url,
            "body_sha256": hashlib.sha256(raw).hexdigest(),
            "body_bytes": len(raw),
            "n_people": len(people),
        }

    print(
        "Forbes one-shot request failed due to a transient upstream error; "
        "trying bounded list pages without weakening completeness validation.",
        flush=True,
    )
    pages: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    for start in range(0, expected_count, page_limit):
        count = min(page_limit, expected_count - start)
        request_url = page_url(url, start=start, limit=count)
        raw_page = fetcher(
            request_url,
            name=f"forbes_annual_page_{start}.json",
            timeout_seconds=60.0,
            max_attempts=2,
        )
        subset = get_people(raw_page)
        if len(subset) != count:
            raise ForbesSourceError(
                f"Page start={start} returned {len(subset)}, expected {count}"
            )
        pages.append({
            "start": start,
            "requested_limit": count,
            "url": request_url,
            "sha256": hashlib.sha256(raw_page).hexdigest(),
            "bytes": len(raw_page),
            "n_people": len(subset),
        })
        combined.extend(subset)
    if len(combined) != expected_count:
        raise ForbesSourceError("Paged retrieval did not return complete annual list")
    uris = [r.get("uri") for r in combined]
    if len([u for u in uris if isinstance(u, str) and u.strip()]) < 0.9 * expected_count:
        raise ForbesSourceError("Paged feed has too few identifiable records")
    identifiers = [u for u in uris if isinstance(u, str) and u.strip()]
    if len(identifiers) != len(set(identifiers)):
        raise ForbesSourceError("Paged feed contains duplicate identifiers")
    # This is a DERIVED in-memory representation, NOT a raw Forbes response.
    derived_payload = json.dumps(
        {"personList": {"persons": combined}},
        ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return derived_payload, {
        "mode": "derived_from_paginated_responses",
        "first_attempt_error": first_error,
        "source_annual_url": url,
        "page_count": len(pages),
        "pages": pages,
        "derived_sha256": hashlib.sha256(derived_payload).hexdigest(),
        "n_people": len(combined),
        "note": (
            "Derived JSON bytes are not a raw Forbes response. "
            "Each original page SHA-256 is recorded above. "
            "The list may still differ from the March 2026 publication."
        ),
    }
