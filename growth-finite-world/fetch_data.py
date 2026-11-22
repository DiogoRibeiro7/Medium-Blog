"""Download the three source files, pinned and checked, into .cache/.

Three public sources, each read from one fixed commit:

* datasets/s-and-p-500 (PDDL), a mirror of Robert Shiller's monthly S&P Composite series:
  price, dividends, earnings, CPI and CAPE from 1871. Earnings and dividends end in June 2023.
* owid/co2-data (CC BY 4.0): GDP (Maddison Project Database), population, territorial and
  consumption-based CO2 by country and year.
* owid/energy-data (CC BY 4.0): primary energy consumption (Energy Institute Statistical Review)
  and GDP by country and year.

Every file is checked against a SHA-256 digest, so the inputs cannot drift. The same pins are recorded in
the data registry (DiogoRibeiro7/data, external/shiller-sp-composite-monthly, external/owid-co2-data and
external/owid-energy-data). build_data.py turns the files into the compact tables in data/; neither the
cache nor the tables are committed, so no CSV is stored in this repository.

    python fetch_data.py            # download what is missing, then verify everything
    python fetch_data.py --check    # verify only; exit 1 if a file is missing or altered
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent / ".cache"

SP_COMMIT = "87a2284276f79d75041c71c8f6d9f2aa1c965e84"
CO2_COMMIT = "22d9a01321ddec951b36f2bfc66e05dc6ad50ad8"
ENERGY_COMMIT = "f8de88a4f6f9cab44f1cf606047cbd6c4414042c"

RAW = "https://raw.githubusercontent.com"
FILES = {
    "sp500.csv": (
        f"{RAW}/datasets/s-and-p-500/{SP_COMMIT}/data/data.csv",
        "b797a3c8de4586af837b86a156b05f49be11ffab99e13171dae763b2a11bdf79",
    ),
    "owid-co2-data.csv": (
        f"{RAW}/owid/co2-data/{CO2_COMMIT}/owid-co2-data.csv",
        "300fb68fe6b0d514ed7f5cf773df77d7e24a495438d84b76e2f39eba1584a0f7",
    ),
    "owid-energy-data.csv": (
        f"{RAW}/owid/energy-data/{ENERGY_COMMIT}/owid-energy-data.csv",
        "266f2e2baad7975351bc9bb4aa061d22b1da9fe4c47d51d2ac6071e01e171f76",
    ),
}


def sha256(path):
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, target, attempts=4):
    """Fetch one file, retrying a cut transfer a few times."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
                target.write_bytes(r.read())
            return
        except OSError as e:
            if attempt == attempts:
                raise
            print(f"  retry {attempt} for {target.name}: {type(e).__name__}", file=sys.stderr)


def main(check_only=False):
    CACHE.mkdir(exist_ok=True)
    bad = []
    for name, (url, digest) in FILES.items():
        target = CACHE / name
        if not target.exists() and not check_only:
            print(f"downloading {name}")
            download(url, target)
        if not target.exists():
            bad.append(f"{name}: missing")
        elif sha256(target) != digest:
            bad.append(f"{name}: digest mismatch")
    for line in bad:
        print(line, file=sys.stderr)
    if bad:
        sys.exit(1)
    print(f"all {len(FILES)} source files present and verified")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv[1:])
