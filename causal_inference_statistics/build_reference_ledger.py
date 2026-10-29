"""Build a deterministic bibliography verification ledger for the 20-part series.

Run: python causal_inference_statistics/build_reference_ledger.py

Only entries with explicit evidence recorded in a separate verification process
may be marked verified. This script defaults to 'pending', never guesses.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOI = re.compile(r"https?://(?:dx\.)?doi\.org/(\S+)", re.IGNORECASE)
YEAR = re.compile(r"\((?:19|20)\d{2}\)\.")
HEADING = re.compile(r"^## References\s*$", re.MULTILINE)


def bibliography(text: str) -> list[str]:
    """Extract standalone citations from the final references section."""
    matches = list(HEADING.finditer(text))
    if not matches:
        raise ValueError("Missing '## References' section")
    tail = text[matches[-1].end():]
    tail = re.split(r"^## ", tail, maxsplit=1, flags=re.MULTILINE)[0]
    return [line.strip() for line in tail.splitlines() if YEAR.search(line) and not line.lstrip().startswith("#")]


def canonical_id(reference: str) -> str:
    """Create a stable identifier based on the citation's bibliographic text."""
    normalized = re.sub(r"https?://\S+", "", reference)
    normalized = re.sub(r"\s+", " ", normalized).casefold().strip().rstrip(".")
    # FNV-1a 64-bit provides deterministic IDs compatible with archived ledger.
    value = 14695981039346656037
    for byte in normalized.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def doi(reference: str) -> str:
    """Extract an explicit DOI, retaining internal punctuation and parentheses."""
    match = DOI.search(reference)
    return match.group(1).rstrip(".,;") if match else ""


def main() -> None:
    """Write occurrence and unique-reference CSVs without changing source files."""
    unique: dict[str, dict[str, str]] = {}
    occurrences: list[dict[str, str | int]] = []
    for part in range(1, 21):
        path = ROOT / f"part{part}" / "article.md"
        for index, citation in enumerate(bibliography(path.read_text(encoding="utf-8")), start=1):
            ref_id = canonical_id(citation)
            if ref_id in unique and unique[ref_id]["reference"] != citation:
                # Differences limited to identifier links are allowed; other
                # collisions are handled explicitly rather than silently merged.
                base = lambda s: re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", s)).casefold().strip().rstrip(".")
                if base(unique[ref_id]["reference"]) != base(citation):
                    raise ValueError(f"Unexpected identifier collision: {ref_id}")
            unique.setdefault(ref_id, {
                "reference_id": ref_id,
                "reference": citation,
                "doi": doi(citation),
                "status": "pending",
                "verified_source": "",
                "verification_note": "",
            })
            occurrences.append({"part": part, "position": index, "reference_id": ref_id})

    # Retain evidence only when both the ID and citation text still match.
    # Editing a citation therefore invalidates its previous verification.
    previous: dict[str, dict[str, str]] = {}
    ledger_path = ROOT / "reference_ledger.csv"
    if ledger_path.is_file():
        with ledger_path.open(newline="", encoding="utf-8") as handle:
            previous = {row["reference_id"]: row for row in csv.DictReader(handle)}
    for ref_id, row in unique.items():
        old = previous.get(ref_id)
        if old and old["reference"] == row["reference"] and old["status"] == "verified" and old["verified_source"]:
            row["status"] = "verified"
            row["verified_source"] = old["verified_source"]
            row["verification_note"] = old["verification_note"]

    output_files = (
        (ROOT / "reference_ledger.csv", list(unique.values()),
         ["reference_id", "reference", "doi", "status", "verified_source", "verification_note"]),
        (ROOT / "reference_occurrences.csv", occurrences,
         ["part", "position", "reference_id"]),
    )
    for destination, data, fields in output_files:
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    print(f"{len(occurrences)} citation occurrences, {len(unique)} distinct citation strings.")
    print("All statuses are pending; publisher verification must be recorded separately.")


if __name__ == "__main__":
    main()
