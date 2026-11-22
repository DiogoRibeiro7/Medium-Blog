"""Download every input of rent-or-buy-us and check it against the data registry.

    python fetch.py

Five contracts in ``../data_sources/`` list the files, their URLs, sizes and SHA-256, pinned to a
registry commit of DiogoRibeiro7/data:

- ``bls-cpi-rent-us-metros`` and ``census-acs-housing-costs-msa``: canonical datasets, downloaded
  from the registry itself;
- ``sp-corelogic-case-shiller-datahub``, ``freddie-mac-pmms-fred`` and ``damodaran-historical-returns``:
  external references, downloaded from their upstream URLs (a pinned mirror commit, a FRED query
  with fixed dates, an Internet Archive capture), because their bytes may not be stored.

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
CONTRACTS = (
    "bls-cpi-rent-us-metros",
    "census-acs-housing-costs-msa",
    "sp-corelogic-case-shiller-datahub",
    "freddie-mac-pmms-fred",
    "damodaran-historical-returns",
)
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
                    f"BAD     {name}/{f['path']}: got {len(body)} bytes, sha256 {sha256(body)}; expected {f['bytes']} bytes, sha256 {f['sha256']}",
                    file=sys.stderr,
                )
                bad += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(body)
            tmp.replace(dest)
            print(f"fetched {name}/{f['path']} ({len(body):,} bytes)")
            time.sleep(0.3)
    print(f"{total - bad} of {total} files match their contracts")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
