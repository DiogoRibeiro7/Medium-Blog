"""Download every input of long-horizon-stock-risk and check it against the data registry.

    python fetch.py

Two contracts in ``../data_sources/`` list the files, their URLs, sizes and SHA-256, pinned to a
registry commit of DiogoRibeiro7/data:

- ``oecd-share-prices-bond-yields-cpi``: canonical dataset, downloaded from the registry itself (OECD
  share price indices, ten-year government bond yields and consumer prices, 23 markets);
- ``shiller-sp-composite-monthly``: external reference, downloaded from a pinned commit of the Datahub
  mirror of Robert Shiller's monthly S&P composite, because its bytes are not stored in the registry.

Each file is saved under ``data/<dataset>/`` (not tracked by git). A file already present with the
right checksum is not downloaded again; a file whose size or checksum differs stops the run.
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
SOURCES = HERE.parent / "data_sources"
CONTRACTS = ("oecd-share-prices-bond-yields-cpi", "shiller-sp-composite-monthly")
DATA = HERE / "data"
UA = "long-horizon-stock-risk/0.1 (+https://github.com/DiogoRibeiro7/Medium-Blog; dfr@esmad.ipp.pt)"


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
    bad = total = 0
    for name in CONTRACTS:
        contract = json.loads((SOURCES / f"{name}.json").read_text())
        for f in contract["files"]:
            total += 1
            dest = DATA / name / f["path"]
            if dest.exists() and sha256(dest.read_bytes()) == f["sha256"]:
                print(f"ok      {name}/{f['path']}")
                continue
            body = download(f["url"])
            if len(body) != f["bytes"] or sha256(body) != f["sha256"]:
                print(
                    f"BAD     {name}/{f['path']}: got {len(body)} bytes, sha256 {sha256(body)}; "
                    f"expected {f['bytes']} bytes, sha256 {f['sha256']}",
                    file=sys.stderr,
                )
                bad += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(body)
            tmp.replace(dest)
            print(f"fetched {name}/{f['path']} ({len(body):,} bytes)")
    print(f"{total - bad} of {total} files match their contracts")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
