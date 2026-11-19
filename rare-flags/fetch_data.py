"""Download the two data sources at pinned commits and check every file against digests.txt.

python fetch_data.py           # download what is missing into data/, then check
python fetch_data.py --check   # check only; fails if a file is missing or differs

Sources (nothing here is committed to this repository):
- detectors: scores of seven AI-text detectors on human and AI-written essays, from Liang et al. (2023),
  github.com/Weixin-Liang/ChatGPT-Detector-Bias (MIT licence).
- cards: two days of card transactions by European cardholders in September 2013, released by the Machine
  Learning Group of ULB and Worldline (Dal Pozzolo et al. 2015), from a GitHub copy whose CSV matches the
  published SHA-256 of the original.
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCES = {
    "detectors": "https://raw.githubusercontent.com/Weixin-Liang/ChatGPT-Detector-Bias/"
    "1d05ed8242d4694a0ed7b73a79c52be8864470f9/",
    "cards": "https://raw.githubusercontent.com/stat432/credit-analysis/125090b731479d1566bdd8b7ad1cf674fdb2bb1e/",
}
# SHA-256 of creditcard.csv as published for the original Kaggle file.
CARDS_CSV = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"


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
            with urllib.request.urlopen(SOURCES[source] + path, timeout=300) as r:
                target.write_bytes(r.read())
        if not target.exists():
            bad.append(f"{source}/{path}: missing")
        elif sha256(target.read_bytes()) != digest:
            bad.append(f"{source}/{path}: digest differs")
    cards = DATA / "cards" / "data-raw" / "creditcard.csv.zip"
    if cards.exists():
        with zipfile.ZipFile(cards) as z:
            if sha256(z.read("creditcard.csv")) != CARDS_CSV:
                bad.append("creditcard.csv: differs from the published original")
    for line in bad:
        print(line)
    if bad:
        sys.exit(1)
    print(f"{sum(1 for _ in entries())} files match their pinned digests; creditcard.csv matches the original")


if __name__ == "__main__":
    main("--check" in sys.argv)
