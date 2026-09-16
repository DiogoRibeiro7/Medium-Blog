"""Validate internal integrity of the case-study publication snapshot."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

from blood_pressure_missingness.publication_manifest import build_publication_manifest

SCOPED_URL = (
    "https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/"
    "Blood_Pressure_Missingness"
)


def _project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as handle:
        payload = tomllib.load(handle)
    return str(payload["project"]["version"])


def _citation_field(text: str, field: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(field)}:\s*\"?([^\"\n]+)\"?\s*$", text)
    return match.group(1).strip() if match else None


def _unpinned_requirements(text: str) -> list[str]:
    errors: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line or any(token in line for token in (">=", "<=", "~=", "!=", ">", "<")):
            errors.append(line)
    return errors


def validate_publication_integrity(root: Path) -> list[str]:
    """Return integrity errors; an empty list means the snapshot is coherent."""
    root = root.resolve()
    errors: list[str] = []
    required = {
        "CITATION.cff",
        "PUBLICATION_SNAPSHOT.md",
        "PUBLICATION_MANIFEST.json",
        "requirements-publication-lock.txt",
        "pyproject.toml",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        return ["Missing publication files: " + ", ".join(missing)]

    version = _project_version(root)
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    citation_version = _citation_field(citation, "version")
    if citation_version != version:
        errors.append(
            f"CITATION.cff version {citation_version!r} does not match package version {version!r}."
        )
    if _citation_field(citation, "license") != "Apache-2.0":
        errors.append("CITATION.cff must declare Apache-2.0.")
    if SCOPED_URL not in citation:
        errors.append("CITATION.cff must point to the scoped case-study directory.")

    lock_text = (root / "requirements-publication-lock.txt").read_text(encoding="utf-8")
    unpinned = _unpinned_requirements(lock_text)
    if unpinned:
        errors.append("Publication lock contains non-exact requirements: " + ", ".join(unpinned))

    try:
        committed = json.loads((root / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return errors + [f"PUBLICATION_MANIFEST.json is invalid JSON: {exc}"]

    expected = build_publication_manifest(root)
    if committed != expected:
        errors.append("PUBLICATION_MANIFEST.json does not match the current public publication artifacts.")
    if committed.get("version") != version:
        errors.append("Publication manifest version does not match pyproject.toml.")
    if committed.get("snapshot_kind") != "publication":
        errors.append("Publication manifest must declare snapshot_kind='publication'.")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_publication_integrity(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    manifest = build_publication_manifest(args.root.resolve())
    print(
        "Publication snapshot integrity passed: "
        f"version {manifest['version']}, {manifest['artifact_count']} integrity-tracked artifacts."
    )


if __name__ == "__main__":
    main()
