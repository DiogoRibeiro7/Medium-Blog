"""Digest of everything the registered test depends on: the evaluation code and the sources.

python inputs_digest.py            prints the digest
python inputs_digest.py --check    exits 0 if results/INPUTS holds the current digest, 1 otherwise
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["evaluate.py", "staff/__init__.py", "staff/comb.py", "staff/data.py", "staff/schedule.py", "sources/FILES"]


def digest() -> str:
    h = hashlib.sha256()
    for name in FILES:
        h.update(name.encode())
        h.update((HERE / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


if __name__ == "__main__":
    d = digest()
    if "--check" in sys.argv:
        f = HERE / "results" / "INPUTS"
        sys.exit(0 if f.exists() and f.read_text().strip() == d else 1)
    print(d)
