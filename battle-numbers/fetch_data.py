"""Download the CDB90 tables used here, from one fixed commit, and check them.

CDB90 is the US Army Concepts Analysis Agency's database of 660 land battles fought between 1600
and 1973, compiled from the Historical Evaluation and Research Organization's battle studies. The
tidy open-data version by Jeffrey B. Arnold is CC0. The five tables used here are small, so they
are committed in data/; this script shows where they come from and verifies them.

    python fetch_data.py            # download what is missing, then verify everything
    python fetch_data.py --check    # verify only; exit 1 if a file is missing or altered
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
COMMIT = "8d984f1a1ede0d8aed857b330621773ce6dade20"
BASE = f"https://raw.githubusercontent.com/jrnold/CDB90/{COMMIT}/data/"
FILES = {
    "battles.csv": "flat/battles.csv",
    "belligerents.csv": "relational/belligerents.csv",
    "commander_links.csv": "relational/commander_links.csv",
    "terrain.csv": "relational/terrain.csv",
}


def sha256(path):
    """Hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned():
    """File name -> pinned digest, from digests.txt."""
    rows = (line.split() for line in (Path(__file__).resolve().parent / "digests.txt").read_text().splitlines())
    return dict(rows)


def main(check_only=False):
    DATA.mkdir(exist_ok=True)
    bad = []
    for name, digest in pinned().items():
        target = DATA / name
        if not target.exists():
            if check_only:
                bad.append(f"missing {name}")
                continue
            with urllib.request.urlopen(BASE + FILES[name], timeout=120) as r:
                target.write_bytes(r.read())
        if sha256(target) != digest:
            bad.append(f"digest mismatch {name}")
    if bad:
        print("\n".join(bad), file=sys.stderr)
        sys.exit(1)
    print(f"{len(FILES)} files verified in {DATA}")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
