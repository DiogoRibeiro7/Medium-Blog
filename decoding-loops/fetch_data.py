"""Download the 18 Gutenberg books (NLTK's copy) at a pinned commit and check them byte for byte.

    python fetch_data.py            # download if missing, check the digest, extract to data/gutenberg/
    python fetch_data.py --check    # check only; exit 1 if missing or altered

The models are not handled here: generate_runs.py loads them from the Hugging Face Hub at the
revisions pinned in its MODELS table and records the commit each revision resolved to.
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
COMMIT = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"  # nltk/nltk_data, branch gh-pages
URL = f"https://raw.githubusercontent.com/nltk/nltk_data/{COMMIT}/packages/corpora/gutenberg.zip"
NAME = "gutenberg.zip"


def pinned():
    rows = (line.split() for line in (HERE / "digests.txt").read_text().splitlines() if line.strip())
    return dict(rows)


def main(check_only=False):
    target = DATA / NAME
    if not target.exists():
        if check_only:
            sys.exit(f"missing {target}")
        DATA.mkdir(exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as r:
            target.write_bytes(r.read())
    blob = target.read_bytes()
    if hashlib.sha256(blob).hexdigest() != pinned()[NAME]:
        sys.exit(f"digest mismatch {NAME}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(DATA)
    print(f"{NAME} verified (nltk_data {COMMIT[:7]}); books in {DATA / 'gutenberg'}")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
