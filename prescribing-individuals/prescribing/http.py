"""A small HTTP client with retries, used by every fetch step."""

from __future__ import annotations

import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "prescribing-individuals research script (github.com/DiogoRibeiro7/Medium-Blog)"}


def get(url: str, params: dict | None = None, tries: int = 5, timeout: int = 120,
        ok_404: bool = False) -> requests.Response | None:
    """GET with exponential back-off on errors and on 429/5xx responses."""
    delay = 2.0
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        except requests.RequestException as exc:  # network error
            last = exc
        else:
            if r.status_code == 404 and ok_404:
                return None
            if r.status_code < 400:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
            last = RuntimeError(f"HTTP {r.status_code} for {r.url}")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"giving up on {url}: {last}")


def download(url: str, dest: Path, ok_404: bool = False) -> Path | None:
    """Download once; reuse the file if it is already there."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = get(url, ok_404=ok_404)
    if r is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest
