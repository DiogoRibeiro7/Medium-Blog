"""Validate whether the case-study tree is internally ready to tag."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

from blood_pressure_missingness.release_manifest import TAG_PREFIX, build_release_manifest

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


def validate_release_readiness(
    root: Path,
    *,
    allow_archival_metadata: bool = False,
) -> list[str]:
    """Return release-readiness errors; an empty list means the tree is ready."""

    root = root.resolve()
    errors: list[str] = []
    required = {
        "CITATION.cff",
        "PUBLICATION_RELEASE.md",
        "RELEASE_MANIFEST.json",
        "requirements-publication-lock.txt",
        "pyproject.toml",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        errors.append("Missing release files: " + ", ".join(missing))
        return errors

    version = _project_version(root)
    expected_tag = f"{TAG_PREFIX}{version}"

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

    if not allow_archival_metadata:
        if re.search(r"(?m)^doi:\s*", citation):
            errors.append("CITATION.cff contains a DOI before an archival DOI is allowed.")
        if re.search(r"(?m)^date-released:\s*", citation):
            errors.append("CITATION.cff contains date-released before a real release is allowed.")

    release_note = (root / "PUBLICATION_RELEASE.md").read_text(encoding="utf-8")
    if expected_tag not in release_note:
        errors.append(f"PUBLICATION_RELEASE.md does not document expected tag {expected_tag!r}.")

    lock_text = (root / "requirements-publication-lock.txt").read_text(encoding="utf-8")
    unpinned = _unpinned_requirements(lock_text)
    if unpinned:
        errors.append(
            "Publication lock contains non-exact requirements: " + ", ".join(unpinned)
        )

    try:
        committed_manifest = json.loads(
            (root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        errors.append(f"RELEASE_MANIFEST.json is invalid JSON: {exc}")
        return errors

    expected_manifest = build_release_manifest(root)
    if committed_manifest != expected_manifest:
        errors.append("RELEASE_MANIFEST.json does not match the current public release artifacts.")
    if committed_manifest.get("version") != version:
        errors.append("Release manifest version does not match pyproject.toml.")
    if committed_manifest.get("release_tag") != expected_tag:
        errors.append("Release manifest tag does not match the namespaced project version.")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Blood_Pressure_Missingness project directory.",
    )
    parser.add_argument(
        "--allow-archival-metadata",
        action="store_true",
        help="Allow a real DOI/date-released after an archival release exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_release_readiness(
        args.root,
        allow_archival_metadata=args.allow_archival_metadata,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    version = _project_version(args.root.resolve())
    manifest = build_release_manifest(args.root.resolve())
    print(
        "Release readiness gate passed: "
        f"version {version}, tag {TAG_PREFIX}{version}, "
        f"{manifest['artifact_count']} integrity-tracked artifacts."
    )


if __name__ == "__main__":
    main()
