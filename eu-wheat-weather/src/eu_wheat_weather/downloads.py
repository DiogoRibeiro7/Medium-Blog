"""Large inputs that are not committed: E-OBS grids and GISCO boundaries.

Each download records its URL, size and SHA-256 in a manifest that is committed, so a
local copy can be checked against the files the study used.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eu_wheat_weather.contract import (
    CALENDAR_BASE,
    CALENDAR_FILE,
    CALENDAR_VERSION,
    EOBS_BASE,
    EOBS_FILES,
    EOBS_VERSION,
    GISCO_BASE,
    NUTS_VERSIONS,
    DataError,
    gisco_file,
)

CHUNK = 1 << 20


def download(url: str, target: Path, timeout: float = 600.0) -> tuple[str, int]:
    """Stream ``url`` to ``target`` (via a partial file); return its SHA-256 and size."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=timeout) as response, \
            partial.open("wb") as out:  # noqa: S310
        while chunk := response.read(CHUNK):
            digest.update(chunk)
            size += len(chunk)
            out.write(chunk)
    partial.replace(target)
    return digest.hexdigest(), size


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_all(sources: dict[str, str], directory: Path, manifest: Path,
              description: str) -> dict[str, Any]:
    """Download every ``name -> url`` into ``directory`` and write the manifest."""
    files = {}
    for name, url in sources.items():
        sha, size = download(url, directory / name)
        files[name] = {"url": url, "bytes": size, "sha256": sha}
    document = {"description": description,
                "retrieved_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "files": files}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8",
                        newline="\n")
    return document


def weather_sources() -> dict[str, str]:
    return {name: EOBS_BASE + name for name in EOBS_FILES.values()}


def boundary_sources() -> dict[str, str]:
    return {gisco_file(v): GISCO_BASE + gisco_file(v) for v in NUTS_VERSIONS}


def fetch_weather(directory: Path, manifest: Path) -> dict[str, Any]:
    return fetch_all(weather_sources(), directory, manifest,
                     f"E-OBS {EOBS_VERSION}, ensemble mean, 0.25 degree regular grid")


def fetch_boundaries(directory: Path, manifest: Path) -> dict[str, Any]:
    return fetch_all(boundary_sources(), directory, manifest,
                     "GISCO NUTS-2 boundaries, 1:3 million, EPSG:4326")


def fetch_calendar(directory: Path) -> dict[str, Any]:
    """The crop calendar of Amendment A1, small enough to commit: the file and its
    manifest both go in ``directory``."""
    return fetch_all({CALENDAR_FILE: CALENDAR_BASE + CALENDAR_FILE + "?download=1"},
                     directory, directory / "manifest.json",
                     f"GGCMI Phase 3 crop calendar {CALENDAR_VERSION}, rainfed winter wheat, "
                     "0.5 degree grid. CC BY 4.0. Jägermeyr, J., et al. (2021), Climate impacts "
                     "on global agriculture emerge earlier in new generation of climate and "
                     "crop models, Nature Food 2, 873-885; data doi:10.5281/zenodo.5062513")


def verify(directory: Path, manifest: Path) -> list[str]:
    """Names of the manifest's files that are present locally and match their hashes.

    A file that is present but differs raises; a missing file is skipped (the large
    inputs are not committed, so a fresh clone has none).
    """
    document = json.loads(manifest.read_text(encoding="utf-8"))
    checked = []
    for name, entry in document["files"].items():
        path = directory / name
        if not path.exists():
            continue
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise DataError(f"{path} does not match {manifest}.")
        checked.append(name)
    return checked
