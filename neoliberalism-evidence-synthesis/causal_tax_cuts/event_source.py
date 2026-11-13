"""Probe the authors' publicly linked CSV and record its actual schema.

The original file is publicly linked by Hope and Limberg on taxrich.uk/data.
This command writes it only to a temporary research artifact. It never guesses
CSV columns, bypasses a sign-in requirement, or commits author data to GitHub.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

FILENAME = "hope_limberg_reforms_11_2020.csv"
SOURCE_PAGE = "https://www.taxrich.uk/data"
SOURCE_URLS = (
    "https://drive.google.com/uc?export=download&id=1FgV9wfFl7FG5cl7PZ7UMzQQ24Wye0VT3",
    "https://drive.usercontent.google.com/download?id=1FgV9wfFl7FG5cl7PZ7UMzQQ24Wye0VT3&export=download&confirm=t",
)
MAX_DOWNLOAD_BYTES = 5_000_000


def inspect_csv(content: bytes) -> dict[str, object]:
    """Accept a real parseable text CSV, rejecting login pages and HTML."""
    if len(content) < 20 or len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Unreasonable author CSV size")
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Author payload is not UTF-8 CSV data") from exc
    opening = decoded.lstrip().lower()[:512]
    if opening.startswith(("<!doctype", "<html")) or "<title>sign in" in opening:
        raise ValueError("Author URL returned an HTML/sign-in page, not public CSV")
    try:
        dialect = csv.Sniffer().sniff(decoded[:4096], delimiters=",;\t")
    except csv.Error as exc:
        raise ValueError("Cannot identify a consistent CSV delimiter") from exc
    rows = csv.DictReader(io.StringIO(decoded), dialect=dialect)
    if not rows.fieldnames or len(rows.fieldnames) < 2:
        raise ValueError("Author CSV has no usable headings")
    first_row = next(rows, None)
    if first_row is None or all(
        value is None or not value.strip() for value in first_row.values()
    ):
        raise ValueError("Author CSV has no meaningful records")
    # Schema probe intentionally displays column names and row count. A couple
    # of real examples make it possible to configure the next importer safely.
    examples = [first_row, *list(_first_n(rows, 2))]
    return {
        "delimiter": dialect.delimiter,
        "columns": list(rows.fieldnames),
        "row_count": 1 + len(examples) - 1 + sum(1 for _ in rows),
        "examples": examples,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _first_n(rows: csv.DictReader[str], limit: int) -> list[dict[str, str | None]]:
    """Read a bounded prefix without changing the original CSV's content."""
    sample: list[dict[str, str | None]] = []
    for _ in range(limit):
        record = next(rows, None)
        if record is None:
            break
        sample.append(record)
    return sample


def download_author_file(
    timeout_seconds: int = 30,
) -> tuple[bytes, str, dict[str, object]]:
    """Try only the two documented public download endpoints of this file."""
    failures: list[str] = []
    for url in SOURCE_URLS:
        try:
            request = Request(url, headers={"User-Agent": "research-audit/1.0"})
            with urlopen(request, timeout=timeout_seconds) as response:
                content = response.read(MAX_DOWNLOAD_BYTES + 1)
            summary = inspect_csv(content)
            return content, url, summary
        except (OSError, ValueError) as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Public author CSV unavailable: " + " | ".join(failures))


def probe(destination: Path) -> dict[str, object]:
    """Write the exact downloaded bytes only after checking their CSV format."""
    destination.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "source_page": SOURCE_PAGE,
        "original_filename": FILENAME,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
    }
    try:
        content, url, metadata = download_author_file()
    except RuntimeError as exc:
        report.update({"status": "unavailable", "reason": str(exc)})
    else:
        (destination / FILENAME).write_bytes(content)
        report.update({"status": "downloaded", "actual_download_url": url, **metadata})
    (destination / "author_event_probe.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Download and inspect the original author CSV on a networked runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = probe(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "downloaded":
        print("AUTHOR SOURCE BLOCKED: no reform events have been fabricated")


if __name__ == "__main__":
    main()
