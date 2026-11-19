"""Download the reanalysis sample used in the analogue test and check it against a pinned digest.

python fetch_data.py           # download into data/ if missing, then check
python fetch_data.py --check   # check only; fails if the file is missing or differs

The file is the near-surface air temperature sample of the NCEP/NCAR reanalysis (2013-2014, 6-hourly,
North America, 2.5 degrees) that xarray distributes as tutorial data, taken from a fixed commit.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
COMMIT = "a35297e9da2cc99c811014f0c8a4297345a5c28d"
BASE = f"https://raw.githubusercontent.com/pydata/xarray-data/{COMMIT}/"
FILES = {"air_temperature.nc": "c606b89c35970a2983b914b76df4adbb409003ef34aa7cfd7f582e41f307482b"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(check_only):
    DATA.mkdir(exist_ok=True)
    bad = []
    for name, digest in FILES.items():
        target = DATA / name
        if not target.exists() and not check_only:
            with urllib.request.urlopen(BASE + name, timeout=120) as r:
                target.write_bytes(r.read())
        if not target.exists():
            bad.append(f"{name}: missing")
        elif sha256(target) != digest:
            bad.append(f"{name}: digest differs")
    for line in bad:
        print(line)
    if bad:
        sys.exit(1)
    print(f"{len(FILES)} file(s) match xarray-data at {COMMIT[:7]}")


if __name__ == "__main__":
    main("--check" in sys.argv)
