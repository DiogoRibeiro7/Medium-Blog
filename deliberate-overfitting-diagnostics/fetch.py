"""Download the two public data sets into data/raw/ and verify their checksums.

  * UCI Adult (Census Income; Becker and Kohavi, 1996): the 32,561 training and
    16,281 test rows of the UCI release combined into one file, labels already
    stripped of the trailing period the UCI test file carries. Mirror:
    jbrownlee/Datasets on GitHub.
  * California housing (Pace and Barry, 1997), 1990 census block groups, the
    version with ocean proximity used in Geron's Hands-On Machine Learning. Mirror:
    ageron/handson-ml2 on GitHub.

Run once: python fetch.py
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"
FILES = {
    "adult-all.csv": (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
        "21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2",
    ),
    "housing.csv": (
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
        "8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in FILES.items():
        target = RAW / name
        if not target.exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, target)
        got = sha256(target)
        if got != digest:
            raise SystemExit(f"{name}: checksum {got} does not match {digest}")
        print("ok", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
