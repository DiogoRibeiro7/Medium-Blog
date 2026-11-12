"""Polite HTTP: one identifying User-Agent, a pause between requests, retries with backoff, no retry on 403/404."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

UA = {"User-Agent": "second-count-prediction/0.1 (research; github.com/DiogoRibeiro7/Medium-Blog)"}
PAUSE = 1.0  # seconds between requests to the same kind of source


def get(url: str, tries: int = 5, timeout: int = 300, backoff: int = 30) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                data = r.read()
            time.sleep(PAUSE)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                raise
            time.sleep(min(backoff * 2**i, 300))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(min(backoff * 2**i, 300))
    raise RuntimeError(f"failed after {tries} tries: {url}")


def status(url: str) -> dict:
    """HTTP status and size of a URL, without keeping the body (metadata only)."""
    try:
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            rng = r.headers.get("Content-Range", "")
            size = int(rng.rsplit("/", 1)[-1]) if "/" in rng else r.headers.get("Content-Length")
            out = {"url": url, "status": r.status, "size": size, "type": r.headers.get("Content-Type")}
    except urllib.error.HTTPError as e:
        out = {"url": url, "status": e.code}
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        out = {"url": url, "status": None, "error": str(e)[:200]}
    time.sleep(PAUSE)
    return out


def download(url: str, dest: Path, tries: int = 6) -> Path:
    """Stream a large file to disk; check the size against Content-Length; back off and retry otherwise."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
                expected = r.headers.get("Content-Length")
                with open(dest, "wb") as fh:
                    while chunk := r.read(1 << 22):
                        fh.write(chunk)
            if expected is None or dest.stat().st_size == int(expected):
                time.sleep(PAUSE)
                return dest
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(min(30 * 2**i, 300))
    raise RuntimeError(f"incomplete download after {tries} tries: {url}")
