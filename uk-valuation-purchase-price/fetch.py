"""Download HM Land Registry Price Paid Data for the uk-valuation-purchase-price analysis.

    python fetch.py --ppd /scratch/ppd --log data/fetch_log.txt

The complete file holds every residential sale in England and Wales lodged for registration
since January 1995 (about 5.3 GB). It carries addresses, so it is downloaded to a scratch
directory outside the repository and never committed. ``build.py`` reads it there.

Contains HM Land Registry data (c) Crown copyright and database right 2026. This data is
licensed under the Open Government Licence v3.0.

The log records each URL tried, the HTTP status, the size and the SHA-256 of the file.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = {"User-Agent": "uk-valuation-purchase-price/1.0 (research companion)"}
PPD_URLS = [
    "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv",
    "https://price-paid-data.publicdata.landregistry.gov.uk/pp-complete.csv",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_ppd(dest: Path, log: list[str]) -> Path | None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "pp-complete.csv"
    for url in PPD_URLS:
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=600) as r, target.open("wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
                status = r.status
        except Exception as e:  # noqa: BLE001 - every failure is logged and the next URL tried
            log.append(f"{url}\terror {type(e).__name__}: {e}\t0\t-")
            continue
        size = target.stat().st_size
        log.append(f"{url}\t{status}\t{size}\t{_sha256_file(target)}")
        if status == 200 and size > 1_000_000_000:
            return target
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppd", required=True, help="scratch directory for the Price Paid Data file")
    ap.add_argument("--log", default="data/fetch_log.txt")
    args = ap.parse_args()
    log = [f"# fetched {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}",
           "source\tstatus\tbytes\tsha256"]
    ok = fetch_ppd(Path(args.ppd), log) is not None
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
