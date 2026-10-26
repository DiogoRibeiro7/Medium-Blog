"""Inventory all references in the 20-part causal inference series.

Run: python causal_inference_statistics/audit_references.py
Produces bibliographic_inventory.csv; extraction only, not DOI validation.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
REFERENCE: re.Pattern[str] = re.compile(r"^\s*([^\n]+?)\s*$")
DOI_URL: re.Pattern[str] = re.compile(r"https?://(?:dx\.)?doi\.org/(\S+)", re.I)


def references(text: str) -> list[str]:
    """Extract standalone bibliography entries after the References heading."""
    marker = re.search(r"^## References\s*$", text, flags=re.M)
    if marker is None:
        return []
    body = text[marker.end():]
    next_heading = re.search(r"^## ", body, flags=re.M)
    if next_heading:
        body = body[:next_heading.start()]
    return [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.startswith(("#", "-", "|"))
    ]


def extract_doi(reference: str) -> str:
    """Preserve DOI-internal parentheses; trim only terminal punctuation."""
    match = DOI_URL.search(reference)
    return match.group(1).rstrip(".,;") if match else ""


def main() -> None:
    """Write an auditable reference inventory without external network calls."""
    rows: list[dict[str, str | int]] = []
    for part in range(1, 21):
        path = ROOT / f"part{part}" / "article.md"
        if not path.is_file():
            raise FileNotFoundError(path)
        for index, ref in enumerate(references(path.read_text(encoding="utf-8")), 1):
            rows.append({
                "part": part,
                "index": index,
                "doi": extract_doi(ref),
                "reference": ref,
                "verification_status": "not_verified",
            })
    destination = ROOT / "bibliographic_inventory.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Inventoried {len(rows)} entries across 20 articles; metadata verification remains required.")


if __name__ == "__main__":
    main()
