"""Download the AMECO bulk file and check it against the pinned contract.

The contract (data_sources/ec-ameco-autumn-2026.json at the repository root, mirroring
external/ec-ameco-autumn-2026 in DiogoRibeiro7/data) records the URL, bytes and SHA-256 of
ameco0.zip as retrieved on 22 November 2026 (the autumn 2026 forecast release). AMECO is
replaced at every Commission forecast, so a hash mismatch means a newer release.

    python fetch_data.py                   # fail on a newer release
    python fetch_data.py --accept-revision # continue, and print the new hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent / "data_sources" / "ec-ameco-autumn-2026.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-revision", action="store_true", help="continue on a newer AMECO release")
    args = ap.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    revised = []
    for item in contract["files"]:
        out = HERE / item["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(item["url"], headers={"User-Agent": "Medium-Blog/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out.write_bytes(r.read())
        digest = sha256(out)
        ok = digest == item["sha256"]
        print(f"{item['path']:<18} {'ok' if ok else 'REVISED'}  {out.stat().st_size} bytes")
        if not ok:
            revised.append({"path": item["path"], "sha256": digest, "bytes": out.stat().st_size})
    if revised:
        print("\nAMECO has published a newer release than the pinned one:")
        print(json.dumps(revised, indent=2))
        if not args.accept_revision:
            print("Re-run with --accept-revision to continue on the new release.")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
