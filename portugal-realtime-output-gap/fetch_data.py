"""Download the current AMECO release and seventeen archived releases, and check them against the pinned contracts.

Contracts (repository root, mirroring DiogoRibeiro7/data):

- data_sources/ec-ameco-autumn-2026.json: the current release, retrieved on 22 November 2026. AMECO is
  replaced at every Commission forecast, so a hash mismatch there means a newer release.
- data_sources/ec-ameco-vintages-2011-2021.json: archived releases, spring 2011 to autumn 2018 and autumn
  2021. Archived files do not change, so a mismatch there is an error.

    python fetch_data.py                   # fail on a newer current release
    python fetch_data.py --accept-revision # continue on a newer current release, and print its hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CURRENT = HERE.parent / "data_sources" / "ec-ameco-autumn-2026.json"
ARCHIVE = HERE.parent / "data_sources" / "ec-ameco-vintages-2011-2021.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(item: dict) -> tuple[Path, str]:
    out = HERE / item["path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (out.exists() and sha256(out) == item["sha256"]):
        req = urllib.request.Request(item["url"], headers={"User-Agent": "Medium-Blog/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out.write_bytes(r.read())
    return out, sha256(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-revision", action="store_true", help="continue on a newer current AMECO release")
    args = ap.parse_args()
    failed, revised = [], []
    for contract, archived in ((ARCHIVE, True), (CURRENT, False)):
        for item in json.loads(contract.read_text(encoding="utf-8"))["files"]:
            out, digest = fetch(item)
            ok = digest == item["sha256"]
            print(f"{item['path']:<38} {'ok' if ok else 'MISMATCH'}  {out.stat().st_size} bytes")
            if not ok:
                (failed if archived else revised).append({"path": item["path"], "sha256": digest})
    if failed:
        print("\nArchived releases do not change; these files do not match their pins:")
        print(json.dumps(failed, indent=2))
        return 1
    if revised:
        print("\nAMECO has published a newer release than the pinned current one:")
        print(json.dumps(revised, indent=2))
        if not args.accept_revision:
            print("Re-run with --accept-revision to continue on the new release.")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
