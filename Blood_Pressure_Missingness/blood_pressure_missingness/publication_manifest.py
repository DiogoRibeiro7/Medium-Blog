"""Build a deterministic integrity manifest for the public publication snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
PROJECT_NAME = "Blood_Pressure_Missingness"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as handle:
        payload = tomllib.load(handle)
    return str(payload["project"]["version"])


def publication_artifact_paths(root: Path) -> list[Path]:
    """Return canonical privacy-safe publication artifacts in stable order."""
    paths: list[Path] = []
    paths.extend(sorted((root / "data").glob("*.csv")))
    paths.extend(sorted((root / "data").glob("*.json")))
    paths.extend(sorted((root / "figures").glob("*.json")))
    paths.extend(
        root / name
        for name in (
            "blood-pressure-missingness.ipynb",
            "CITATION.cff",
            "requirements-publication-lock.txt",
        )
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Publication artifact set contains missing files: "
            + ", ".join(str(path.relative_to(root)) for path in missing)
        )
    return paths


def build_publication_manifest(root: Path) -> dict[str, Any]:
    """Return the publication snapshot manifest without mutating the project."""
    root = root.resolve()
    version = _project_version(root)
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in publication_artifact_paths(root)
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "project": PROJECT_NAME,
        "version": version,
        "snapshot_kind": "publication",
        "hash_algorithm": "sha256",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def write_publication_manifest(root: Path, output: Path) -> None:
    payload = build_publication_manifest(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("PUBLICATION_MANIFEST.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_publication_manifest(args.root, args.output)
    print(f"Wrote publication manifest to {args.output}")


if __name__ == "__main__":
    main()
