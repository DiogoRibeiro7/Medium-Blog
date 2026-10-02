#!/usr/bin/env python3
"""Fetch the pinned canonical Online Retail II workbook with SHA-256 verification."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

DATA_REPOSITORY = "DiogoRibeiro7/data"
DATA_COMMIT = "e90eed21a474c0a54ab9bf658c01dbd778bf58f1"
DATA_PATH = "datasets/online-retail-ii/raw/online_retail_II.xlsx"
EXPECTED_SHA256 = "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"
RAW_URL = (
    f"https://raw.githubusercontent.com/{DATA_REPOSITORY}/"
    f"{DATA_COMMIT}/{DATA_PATH}"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.is_file() and sha256_file(output) == EXPECTED_SHA256:
        return output

    fd, tmp_name = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(RAW_URL, timeout=120) as response, tmp.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)

        actual = sha256_file(tmp)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(
                f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual}"
            )
        tmp.replace(output)
    finally:
        if tmp.exists():
            tmp.unlink()

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/online-retail-ii/online_retail_II.xlsx"),
    )
    return parser.parse_args()


def main() -> int:
    try:
        path = fetch(parse_args().output)
    except Exception as exc:  # pragma: no cover - surfaced to command-line users
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
