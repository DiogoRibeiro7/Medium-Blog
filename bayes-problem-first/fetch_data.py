"""Download the O-ring data at a pinned commit and check it against digests.txt.

python fetch_data.py           # download what is missing into data/, then check
python fetch_data.py --check   # check only; fails if the file is missing or differs

Source (nothing here is committed to this repository):
- rdatasets: ``vcd::SpaceShuttle``, the O-ring thermal-distress counts and launch temperatures of the 24
  shuttle flights before Challenger, from Dalal, Fowlkes and Hoadley (1989), as distributed in
  Vincent Arel-Bundock's Rdatasets (github.com/vincentarelbundock/Rdatasets) at a fixed commit.

The vaccine-trial numbers (8 of 170 cases, the protocol's prior and analysis plan) are published counts
and constants, entered in ``bayes_core.py`` with their sources.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCES = {
    "rdatasets": "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def entries():
    for line in (HERE / "digests.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, source, path = line.split()
            yield digest, source, path


def main(check_only):
    bad = []
    for digest, source, path in entries():
        target = DATA / source / path
        if not target.exists() and not check_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(SOURCES[source] + path, timeout=120) as r:
                target.write_bytes(r.read())
        if not target.exists():
            bad.append(f"{source}/{path}: missing")
        elif sha256(target.read_bytes()) != digest:
            bad.append(f"{source}/{path}: digest differs")
    for line in bad:
        print(line)
    print("all files match digests.txt" if not bad else f"{len(bad)} file(s) missing or different")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv[1:]))
