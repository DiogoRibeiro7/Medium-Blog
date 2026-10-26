"""python -m cancelreplay [run|fetch|check]"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import urllib.request

from . import data
from .paths import DATA_FILE


def fetch() -> None:
    """Download the CSV again and store it compressed, after checking its hash."""
    with urllib.request.urlopen(data.SOURCE_URL, timeout=120) as response:
        raw = response.read()
    found = hashlib.sha256(raw).hexdigest()
    if found != data.SOURCE_SHA256:
        raise SystemExit(f"downloaded file has sha256 {found}, expected {data.SOURCE_SHA256}")
    DATA_FILE.parent.mkdir(exist_ok=True)
    with gzip.open(DATA_FILE, "wb", compresslevel=9) as handle:
        handle.write(raw)
    print(f"wrote {DATA_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="cancelreplay")
    parser.add_argument("command", choices=["run", "fetch", "check"], nargs="?", default="run")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch()
    elif args.command == "check":
        _, report = data.load()
        print("\n".join(report.lines()))
    else:
        from .report import run
        run(make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
