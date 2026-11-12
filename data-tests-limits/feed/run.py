"""Profile a range of months of one feed, one file at a time.

    python -m feed.run --feed fhvhv --months 2019-02:2019-12 --out profiles --cache /tmp/tlc

Writes ``<out>/<feed>/<YYYY-MM>.json`` and deletes each file after profiling it. A month whose file does not
exist is recorded in ``<out>/<feed>/missing.txt``.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
from pathlib import Path

from feed.contracts import CONTRACTS
from feed.profile import download, profile_file
from feed.sources import months


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", required=True, choices=sorted(CONTRACTS))
    ap.add_argument("--months", required=True, help="YYYY-MM:YYYY-MM")
    ap.add_argument("--out", default="profiles")
    ap.add_argument("--cache", default="/tmp/tlc")
    a = ap.parse_args()
    first, last = a.months.split(":")
    first = max(first, CONTRACTS[a.feed].first_month)
    out = Path(a.out) / a.feed
    out.mkdir(parents=True, exist_ok=True)
    for m in months(first, last):
        try:
            path = download(a.feed, m, Path(a.cache))
        except (urllib.error.HTTPError, RuntimeError) as e:
            print(f"{a.feed} {m}: not available ({e})", flush=True)
            with (out / "missing.txt").open("a", encoding="utf-8") as fh:
                fh.write(f"{m}\n")
            continue
        p = profile_file(path, m, a.feed)
        (out / f"{m}.json").write_text(json.dumps(p), encoding="utf-8")
        path.unlink()
        print(f"{a.feed} {m}: {p['n_rows']:,} rows", flush=True)


if __name__ == "__main__":
    main()
