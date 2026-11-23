"""Download the Eurostat snapshot of eu-material-use and check it against the registry.

    python fetch.py

The ten Eurostat JSON-stat responses are a canonical dataset in the data registry
(DiogoRibeiro7/data, ``datasets/eurostat-material-use``). Their URLs, sizes and SHA-256 are
copied in ``../data_sources/eurostat-material-use.json``, pinned to a registry commit. Each file
is saved under ``data/`` (not tracked by git) and every table is rebuilt from it at each run. A file
that is already present with the right checksum is not downloaded again; a file whose size or checksum
differs stops the run.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent / "data_sources" / "eurostat-material-use.json"
DATA = HERE / "data"
UA = "Mozilla/5.0 (X11; Linux x86_64) research-download (dfr@esmad.ipp.pt)"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download(url: str, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(url)


def main() -> int:
    contract = json.loads(CONTRACT.read_text())
    bad = 0
    for f in contract["files"]:
        dest = DATA / f["path"]
        if dest.exists() and sha256(dest.read_bytes()) == f["sha256"]:
            print(f"ok      {f['path']}")
            continue
        body = download(f["url"])
        if len(body) != f["bytes"] or sha256(body) != f["sha256"]:
            print(
                f"BAD     {f['path']}: got {len(body)} bytes, sha256 {sha256(body)}; expected {f['bytes']} bytes, sha256 {f['sha256']}",
                file=sys.stderr,
            )
            bad += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(dest)
        print(f"fetched {f['path']} ({len(body):,} bytes)")
        time.sleep(0.5)
    print(f"{len(contract['files']) - bad} of {len(contract['files'])} files match registry commit {contract['commit'][:12]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
