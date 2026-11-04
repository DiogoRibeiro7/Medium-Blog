"""Retrieve public research data with bounded retries and explicit diagnostics.

The downloader never replaces verified snapshots, silently changes URLs, or
accepts truncated HTTP responses. It returns exact bytes for SHA-256 capture.
"""

from __future__ import annotations

import socket
import time
from http.client import IncompleteRead
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fetch_bytes(
    url: str,
    *,
    name: str,
    timeout_seconds: float = 120.0,
    max_attempts: int = 2,
    wait_seconds: float = 3.0,
    opener: Callable[..., object] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> bytes:
    """Download one public URL with transparent retries on transient failures.

    Args:
        url: Fully-qualified HTTP(S) URL to the public source.
        name: Human-readable source filename for diagnostic output.
        timeout_seconds: Socket timeout for each attempt in seconds.
        max_attempts: Upper bound on network attempts, including the first.
        wait_seconds: Base backoff between attempts.
        opener: Injectable URL opening function for deterministic unit tests.
        sleeper: Injectable waiting function for deterministic unit tests.

    Returns:
        Complete, unmodified response bytes, for checksum archiving.

    Raises:
        TypeError: For invalid argument types.
        ValueError: For invalid argument values, HTTP client errors, or
            empty/truncated/unexpected response content.
        RuntimeError: After transient network failures exhaust all attempts.
    """
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ValueError("url must be an HTTP(S) URL")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise ValueError("max_attempts must be a positive integer")
    if not 0 < timeout_seconds <= 300:
        raise ValueError("timeout_seconds must be between zero and 300")
    if wait_seconds < 0:
        raise ValueError("wait_seconds must be non-negative")

    request = Request(
        url, headers={"User-Agent": "Our World In Data data fetch/1.0"}
    )
    for attempt in range(1, max_attempts + 1):
        print(
            f"Fetching public source {name}: attempt {attempt}/{max_attempts}"
            f" (timeout={timeout_seconds:g}s), URL={url}",
            flush=True,
        )
        try:
            with opener(request, timeout=timeout_seconds) as response:
                expected_header = response.headers.get("Content-Length")
                media_type = response.headers.get("Content-Type", "").lower()
                payload: bytes = response.read()

            # Strict verification happens after the network response completes.
            if len(payload) < 100:
                raise ValueError(
                    f"Unexpectedly short public source {name}: {len(payload)} bytes"
                )
            if (
                expected_header is not None
                and int(expected_header) != len(payload)
            ):
                raise IncompleteRead(
                    payload,
                    int(expected_header) - len(payload),
                )
            if "text/html" in media_type or payload[:32].lstrip().lower().startswith(
                (b"<!doctype html", b"<html")
            ):
                raise ValueError(f"Source {name} returned HTML, not research data")
            print(
                f"Fetched {name}: {len(payload)} complete bytes"
                f" (Content-Type: {media_type or 'unspecified'})",
                flush=True,
            )
            return payload
        except HTTPError as exc:
            # A genuine 404/403 is not made more credible by retries.
            if exc.code not in (408, 429) and exc.code < 500:
                raise ValueError(
                    f"HTTP {exc.code} for public source {name}: {url}"
                ) from exc
            transient: Exception = exc
        except (TimeoutError, socket.timeout, URLError, ConnectionError,
                IncompleteRead) as exc:
            transient = exc

        print(
            f"Transient download failure for {name}: "
            f"{type(transient).__name__}: {transient}",
            flush=True,
        )
        if attempt == max_attempts:
            raise RuntimeError(
                f"Could not fetch {name} after {max_attempts} attempts "
                f"from {url}: {type(transient).__name__}: {transient}"
            ) from transient
        sleeper(wait_seconds * attempt)

    raise AssertionError("Unreachable: every attempt returns or raises")
