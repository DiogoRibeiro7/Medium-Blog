"""File locations, relative to the project folder."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"          # tidy extracts, committed
RAW = DATA / "raw"            # downloads, not committed (uploaded as a workflow artifact)
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def ensure() -> None:
    for p in (DATA, RAW, RESULTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
