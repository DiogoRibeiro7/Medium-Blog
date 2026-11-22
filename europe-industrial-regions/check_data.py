"""Check that the two Rosés-Wolf files are in data/raw/ and match the data registry.

    python check_data.py

CEPR, which hosts the Rosés-Wolf database on regional GDP, refuses automated downloads,
so the files cannot be fetched by a script or by CI. Download them in a browser from the
URLs in ``../data_sources/roses-wolf-regional-gdp-v7.json`` (also printed below when a file
is missing) and save them under ``data/raw/`` with the names given there. This script checks
size and SHA-256 against the registry record (DiogoRibeiro7/data,
``external/roses-wolf-regional-gdp-v7``) and stops if anything differs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent / "data_sources" / "roses-wolf-regional-gdp-v7.json"
DATA = HERE / "data"


def main() -> int:
    contract = json.loads(CONTRACT.read_text())
    bad = 0
    for f in contract["files"]:
        p = DATA / f["path"]
        if not p.exists():
            print(f"MISSING {f['path']}: download {f['url']} in a browser and save it as {p}")
            bad += 1
            continue
        b = p.read_bytes()
        h = hashlib.sha256(b).hexdigest()
        if len(b) != f["bytes"] or h != f["sha256"]:
            print(f"BAD     {f['path']}: {len(b)} bytes, sha256 {h}; expected {f['bytes']} bytes, "
                  f"sha256 {f['sha256']}")
            bad += 1
            continue
        print(f"ok      {f['path']}")
    print(f"{len(contract['files']) - bad} of {len(contract['files'])} files match registry commit "
          f"{contract['commit'][:12]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
