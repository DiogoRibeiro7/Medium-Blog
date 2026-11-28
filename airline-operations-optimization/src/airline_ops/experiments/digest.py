"""Digest of the inputs that determine the experiment's results.

The workflow reruns the full experiment only when this digest differs from the one stored
in ``results/INPUTS``: the model code, the experiment configuration, the data contract and
the pinned dependency versions. Figure and summary code are left out, so redrawing a figure
does not trigger a two-hour run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

EXCLUDED = (
    "viz/",
    "experiments/summary.py",
    "experiments/details.py",
    "experiments/digest.py",
    "cli.py",
    "__main__.py",
)


def input_files(folder: Path) -> list[Path]:
    """Files whose content changes the results, in a fixed order."""
    package = folder / "src" / "airline_ops"
    code = [
        p
        for p in sorted(package.rglob("*.py"))
        if not any(p.relative_to(package).as_posix().startswith(x) for x in EXCLUDED)
    ]
    others = [
        folder / "configs" / "experiment.toml",
        folder / "requirements.txt",
        folder.parent / "data_sources" / "openflights-airports.json",
    ]
    return code + [p for p in others if p.is_file()]


def compute_digest(folder: Path) -> str:
    """SHA-256 over the relative path and the bytes of every input file."""
    h = hashlib.sha256()
    for path in input_files(folder):
        h.update(path.relative_to(folder.parent).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def digest_matches(folder: Path, stored: Path) -> bool:
    """Whether the stored digest equals the digest of the current inputs."""
    return stored.is_file() and stored.read_text(encoding="utf-8").strip() == compute_digest(folder)
