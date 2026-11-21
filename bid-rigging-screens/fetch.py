"""Download the bid-rigging data of Imhof and Wallimann (2021) from Mendeley Data.

    python fetch.py --out data

Dataset: Wallimann, H. (2021). Detecting bid-rigging coalitions in different countries and
auction formats, Version 1. Mendeley Data, doi:10.17632/x8jwmbcpgn.1, licence CC BY 4.0.
It holds the Swiss, Okinawa (Japan) and Italian tender data prepared for Imhof and
Wallimann (2021), International Review of Law and Economics 68, 106016.

The script asks the Mendeley public API for the file list and downloads every file; if
the API does not answer, it falls back to the archive of the whole dataset. Files go to
``data/raw/`` unchanged (archives are also unpacked there), and ``data/fetch_log.txt``
records the URL, HTTP status, size and SHA-256 of every response.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATASET = "x8jwmbcpgn"
VERSION = 1
UA = "Mozilla/5.0 (X11; Linux x86_64) research-download (dfr@esmad.ipp.pt)"
API_LISTS = [
    f"https://data.mendeley.com/public-api/datasets/{DATASET}/files?folder_id=root&version={VERSION}",
    f"https://data.mendeley.com/public-api/datasets/{DATASET}/files?version={VERSION}",
]
ZIP_URLS = [
    f"https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/{DATASET}-{VERSION}.zip",
    f"https://data.mendeley.com/public-api/zip/{DATASET}/download/{VERSION}",
]

LOG: list[str] = []


def get(url: str, accept: str = "*/*", tries: int = 3) -> tuple[int, bytes]:
    """GET a URL with retries; return (status, body). Status 0 means no response."""
    last = (0, b"")
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                last = (r.status, body)
                break
        except urllib.error.HTTPError as e:
            last = (e.code, e.read() or b"")
            if e.code in (403, 404):
                break
        except Exception as e:  # network error: retry
            last = (0, str(e).encode())
        time.sleep(3 * (attempt + 1))
    status, body = last
    LOG.append(f"{url}\t{status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}")
    return last


def save(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def unpack(body: bytes, dest: Path) -> int:
    n = 0
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name or name.startswith("."):
                continue
            save(dest / name, z.read(info))
            LOG.append(f"  unpacked {name}\t{info.file_size}")
            n += 1
    return n


def from_api(raw: Path) -> int:
    for url in API_LISTS:
        status, body = get(url, accept="application/json")
        if status != 200:
            continue
        try:
            files = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(files, dict):
            files = files.get("files") or files.get("data") or []
        save(raw.parent / "mendeley_file_list.json", body)
        n = 0
        for f in files:
            name = f.get("filename") or f.get("name")
            details = f.get("content_details") or {}
            url_f = details.get("download_url") or f.get("download_url")
            if not name or not url_f:
                continue
            st, data = get(url_f)
            if st == 200 and data:
                save(raw / name, data)
                n += 1
                if name.lower().endswith(".zip"):
                    n += unpack(data, raw)
        if n:
            return n
    return 0


def from_zip(raw: Path) -> int:
    for url in ZIP_URLS:
        status, body = get(url, accept="application/zip")
        if status == 200 and body[:2] == b"PK":
            save(raw.parent / f"{DATASET}-{VERSION}.zip", body)
            return unpack(body, raw)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    raw = out / "raw"
    LOG.append(f"# fetch.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    LOG.append("# url\tstatus\tbytes\tsha256")
    n = from_api(raw)
    if n == 0:
        n = from_zip(raw)
    LOG.append(f"# files saved: {n}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fetch_log.txt").write_text("\n".join(LOG) + "\n")
    print("\n".join(LOG))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
