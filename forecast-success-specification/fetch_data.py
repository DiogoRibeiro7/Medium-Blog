"""Download the M5 archive, check its SHA-256 and extract the three files the test uses."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
FILES = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()
    raw = args.data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = args.data / "m5.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    digest = sha256(archive)
    if digest != SHA256:
        raise SystemExit(f"SHA-256 mismatch: {digest}")
    with zipfile.ZipFile(archive) as z:
        for name in FILES:
            z.extract(name, raw)
    print(f"m5.zip {digest}; extracted {', '.join(FILES)} to {raw}")


if __name__ == "__main__":
    main()
