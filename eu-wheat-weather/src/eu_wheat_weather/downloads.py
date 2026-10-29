"""Large inputs that are not committed: E-OBS grids, GISCO boundaries and the MIRCA-OS
map of maize.

Each download records its URL, size and SHA-256 in a manifest that is committed, so a
local copy can be checked against the files the study used.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
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
    IRRIGATION_FILTERS,
    MAIZE_CALENDAR_FILE,
    MIRCA_ARCHIVE,
    MIRCA_LAYERS,
    MIRCA_MEMBER_DIR,
    MIRCA_URL,
    MIRCA_VERSION,
    NUTS_VERSIONS,
    DataError,
    gisco_file,
    irrigation_url,
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
    """The crop calendars of Amendment A1 (wheat) and section 9 (maize), small enough to
    commit: the files and their manifest all go in ``directory``."""
    return fetch_all({name: CALENDAR_BASE + name + "?download=1"
                      for name in (CALENDAR_FILE, MAIZE_CALENDAR_FILE)},
                     directory, directory / "manifest.json",
                     f"GGCMI Phase 3 crop calendar {CALENDAR_VERSION}, rainfed winter wheat "
                     "(Amendment A1) and rainfed maize (section 9), 0.5 degree grid. "
                     "CC BY 4.0. Jägermeyr, J., et al. (2021), Climate impacts on global "
                     "agriculture emerge earlier in new generation of climate and crop "
                     "models, Nature Food 2, 873-885; data doi:10.5281/zenodo.5062513")


def _bsdtar() -> str:
    """A tar that reads RAR archives: bsdtar, which Windows ships as its own tar.exe."""
    candidates = [shutil.which("bsdtar"),
                  str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"),
                  shutil.which("tar")]
    for tar in candidates:
        if tar and Path(tar).exists():
            version = subprocess.run([tar, "--version"], capture_output=True, text=True,
                                     check=False).stdout
            if "bsdtar" in version:
                return tar
    raise DataError("Extracting the MIRCA-OS archive (RAR) needs bsdtar.")


def fetch_mirca(directory: Path, manifest: Path) -> dict[str, Any]:
    """Amendment A2's map of maize (section 10.1): the archive of annual grids, and its two
    2015 maize layers, extracted. The manifest records all three files."""
    document = fetch_all({MIRCA_ARCHIVE: MIRCA_URL}, directory, manifest,
                         f"MIRCA-OS {MIRCA_VERSION}, annual harvested area grids; the 2015 "
                         "5-arcminute layers of irrigated and rainfed maize are extracted. "
                         "CC BY 4.0. Kebede, E. A., et al. (2025), A global open-source "
                         "dataset of monthly irrigated and rainfed cropped areas (MIRCA-OS) "
                         "for the 21st century, Scientific Data 12, 208; data "
                         "doi:10.4211/hs.60a890eb841c460192c03bb590687145")
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        subprocess.run([_bsdtar(), "-xf", str(directory / MIRCA_ARCHIVE), "-C", tmp,
                        *(MIRCA_MEMBER_DIR + name for name in MIRCA_LAYERS.values())],
                       check=True)
        for name in MIRCA_LAYERS.values():
            (Path(tmp) / MIRCA_MEMBER_DIR / name).replace(directory / name)
    for name in MIRCA_LAYERS.values():
        path = directory / name
        document["files"][name] = {"url": f"{MIRCA_ARCHIVE}: {MIRCA_MEMBER_DIR}{name}",
                                   "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    return document


def fetch_irrigation(directory: Path) -> dict[str, Any]:
    """Section 10.3's check: Eurostat's irrigated and total grain maize area in 2023 for
    Spain, Portugal, Italy and Greece, small enough to commit with its manifest."""
    return fetch_all({f"{name}.json": irrigation_url(name) for name in IRRIGATION_FILTERS},
                     directory, directory / "manifest.json",
                     "Eurostat integrated farm statistics 2023: irrigated area by crop "
                     "(ef_lus_irri, C1500T) and area by crop (ef_lus_allcrops, C1500), "
                     "holdings above the survey threshold, ES, PT, IT and EL")


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
