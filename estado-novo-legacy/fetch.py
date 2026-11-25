"""Download every input file, pinned and checked, into data/.

    python fetch.py

Four contracts in ``../data_sources/`` list the files, each with its URL, size and SHA-256:

* ``eurostat-eu-convergence-education.json``: four Eurostat JSON-stat responses kept as a canonical
  dataset in the data registry (DiogoRibeiro7/data), read from one registry commit, saved under data/raw/.
* ``barro-lee-educational-attainment.json``: two CSV files from the Barro-Lee repository at one commit.
* ``penn-world-table-10-01.json``: the PWT 10.01 data file of the CRAN package pwt10, from the GitHub
  mirror of CRAN at one commit.
* ``owid-co2-data.json``: Our World in Data's file with Maddison Project GDP and population, at one commit.

The last three are external references in the registry; their bytes are read from the upstream commit,
saved under data/external/ and never committed here. A file already present with the right checksum is
not downloaded again; a file whose size or checksum differs stops the run.
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
CONTRACTS = [
    "eurostat-eu-convergence-education.json",
    "barro-lee-educational-attainment.json",
    "penn-world-table-10-01.json",
    "owid-co2-data.json",
]
DATA = HERE / "data"
UA = "Mozilla/5.0 (X11; Linux x86_64) research-download (dfr@esmad.ipp.pt)"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download(url: str, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(url)


def main() -> int:
    total = bad = 0
    for name in CONTRACTS:
        contract = json.loads((SOURCES / name).read_text())
        for f in contract["files"]:
            total += 1
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
        print(f"   {contract['dataset']}: registry commit {contract['commit'][:12]}")
    print(f"{total - bad} of {total} files match their contracts")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
