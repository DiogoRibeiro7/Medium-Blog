"""Download the Anonymous Bank archive into data/ and check its SHA-256.

    python fetch.py

The data are free to use; the documentation asks users to notify Avi
Mandelbaum (avim@ie.technion.ac.il) and to acknowledge the source.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from anonbank.data import ARCHIVE_SHA256, ARCHIVE_URL  # noqa: E402

TARGET = HERE / "data" / "AnonymousBank.rar"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    TARGET.parent.mkdir(exist_ok=True)
    if not TARGET.exists():
        print(f"downloading {ARCHIVE_URL}")
        urllib.request.urlretrieve(ARCHIVE_URL, TARGET)
    digest = sha256(TARGET)
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"checksum mismatch: {digest}")
    print(f"{TARGET} ok ({digest})")


if __name__ == "__main__":
    main()
