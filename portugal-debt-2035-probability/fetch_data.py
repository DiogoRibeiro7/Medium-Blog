"""Download the Eurostat series for Portugal and check them against the pinned contract.

The contract (data_sources/eurostat-portugal-government-finance.json at the repository root,
mirroring external/eurostat-portugal-government-finance in DiogoRibeiro7/data) records, for each
API response, the query URL, the byte count, the SHA-256 and Eurostat's own "updated"
timestamp at retrieval. Eurostat revises these datasets twice a year (April and October
EDP notifications) and the API serves only the latest vintage, so a hash mismatch means
the provider has published a revision, not that the download is corrupt.

    python fetch_data.py                   # fail on any revision
    python fetch_data.py --accept-revision # continue, and print the new hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent / "data_sources" / "eurostat-portugal-government-finance.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--accept-revision", action="store_true", help="continue when Eurostat has published a newer vintage"
    )
    args = ap.parse_args()

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    revised = []
    for item in contract["files"]:
        out = HERE / item["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(item["url"], headers={"User-Agent": "Medium-Blog/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out.write_bytes(r.read())
        digest = sha256(out)
        updated = json.loads(out.read_text(encoding="utf-8")).get("updated")
        ok = digest == item["sha256"]
        print(f"{item['path']:<22} {'ok' if ok else 'REVISED'}  updated={updated}")
        if not ok:
            revised.append(
                {"path": item["path"], "sha256": digest, "bytes": out.stat().st_size, "provider_updated": updated}
            )

    if revised:
        print("\nEurostat has published a newer vintage than the pinned one:")
        print(json.dumps(revised, indent=2))
        if not args.accept_revision:
            print("Re-run with --accept-revision to continue on the new vintage.")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
