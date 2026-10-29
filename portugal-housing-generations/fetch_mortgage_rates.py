"""Download and pin the ECB cost-of-borrowing series for Portuguese house-purchase loans.

Series ``MIR.M.PT.B.A2C.AM.R.A.2250.EUR.N`` (ECB MFI interest rate statistics,
reported for Portugal by Banco de Portugal): monthly composite cost of
borrowing for new loans to households for house purchase, annualised agreed
rate, percent per annum. The raw SDMX-CSV response is saved byte for byte,
next to a JSON file recording the URL, retrieval time, HTTP status, headers
and SHA-256 digest. Nothing is transformed here; parsing and validation of the
values live in ``analysis.py``.

Run:  python fetch_mortgage_rates.py --out data
Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

SERIES_KEY = "M.PT.B.A2C.AM.R.A.2250.EUR.N"
MIR_URL = f"https://data-api.ecb.europa.eu/service/data/MIR/{SERIES_KEY}?format=csvdata"
FILE_STEM = "ecb_mir_PT_house_purchase_cost_of_borrowing"
REQUIRED_COLUMNS = {"KEY", "TIME_PERIOD", "OBS_VALUE"}


def download(url: str, timeout: float = 60.0) -> tuple[bytes, int, dict[str, str]]:
    """Return the body, HTTP status and selected response headers for ``url``."""
    request = urllib.request.Request(url, headers={"Accept": "text/csv", "User-Agent": "Medium-Blog reproducibility fetch"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https URL)
        body = response.read()
        headers = {name: response.headers[name] for name in ("Content-Type", "Last-Modified", "ETag", "Date") if response.headers[name]}
        return body, response.status, headers


def check_body(body: bytes, series_key: str = SERIES_KEY) -> int:
    """Check that the response is SDMX-CSV for exactly one series; return its row count.

    Raises:
        ValueError: If the body is empty, lacks required columns, or contains another series.
    """
    if not body:
        raise ValueError("Empty response")
    reader = csv.DictReader(io.StringIO(body.decode("utf-8")))
    if reader.fieldnames is None or not REQUIRED_COLUMNS <= set(reader.fieldnames):
        raise ValueError(f"Missing SDMX-CSV columns: {REQUIRED_COLUMNS - set(reader.fieldnames or [])}")
    rows = list(reader)
    keys = {row["KEY"] for row in rows}
    if keys != {f"MIR.{series_key}"}:
        raise ValueError(f"Unexpected series in response: {sorted(keys)}")
    return len(rows)


def main() -> int:
    """Download the series and write the raw body and its retrieval record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    retrieved = dt.datetime.now(dt.timezone.utc)
    body, status, headers = download(MIR_URL)
    if status != 200:
        print(f"HTTP {status} for {MIR_URL}", file=sys.stderr)
        return 1
    rows = check_body(body)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{FILE_STEM}_{retrieved:%Y-%m-%d}"
    (args.out / f"{stem}.csv").write_bytes(body)
    record = {
        "url": MIR_URL,
        "series_key": f"MIR.{SERIES_KEY}",
        "retrieved_at_utc": retrieved.isoformat(timespec="seconds"),
        "http_status": status,
        "response_headers": headers,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "rows": rows,
    }
    (args.out / f"{stem}.meta.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
