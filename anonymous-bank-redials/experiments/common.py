"""Shared paths, seeds and helpers for the experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "mechanistic-vs-data-driven-generators"
                       / "src"))

from anonbank.sim import Centre  # noqa: E402
from anonbank.stats import Statistics  # noqa: E402
from anonbank.table import day_matrix  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
TABLE = DATA / "halfhour.csv.gz"
MASTER_SEED = 19990101


def parse(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true",
                    help="small sizes, to check the code runs")
    return ap.parse_args()


def raw_source() -> Path | None:
    """The archive or a folder of extracted monthly files, if present."""
    env = os.environ.get("ANONBANK_DATA")
    for cand in ([Path(env)] if env else []) + [DATA / "AnonymousBank.rar",
                                                 DATA / "raw"]:
        if cand.exists():
            return cand
    return None


def seed(tag: str, *path: int) -> int:
    ss = np.random.SeedSequence([MASTER_SEED, sum(ord(c) * 31 ** i
                                                  for i, c in enumerate(tag)),
                                 *path])
    return int(ss.generate_state(1)[0])


def rng(tag: str, *path: int) -> np.random.Generator:
    return np.random.default_rng(seed(tag, *path))


def load_table() -> pd.DataFrame:
    return pd.read_csv(TABLE, parse_dates=["date"])


def matrices(tab: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {c: day_matrix(tab, c) for c in ("offered", "abandoned", "agents",
                                           "id_redials", "answered_20s")}
    out["dow"] = tab.groupby("day")["dow"].first().to_numpy()
    return out


def real_statistics(tab: pd.DataFrame, column: str = "offered"
                    ) -> Statistics:
    m = matrices(tab)
    offered = m["offered"] if column == "offered" else \
        m["offered"] - m["id_redials"] if column == "non_redials" else \
        m["id_redials"]
    return Statistics(offered, m["abandoned"], m["dow"])


def calibrated_centre(tab: pd.DataFrame, quick: bool = False) -> tuple[
        Centre, float]:
    m = matrices(tab)
    service = tab["service_total"].sum() / tab["answered"].sum()
    centre = Centre.from_table(m["offered"], m["abandoned"], m["agents"],
                               m["dow"], tab["wait_total"].sum(), service)
    k = centre.calibrate_service(m["abandoned"].sum() / m["offered"].sum(),
                                 seed("calibrate"),
                                 n_days=60 if quick else 247)
    return centre, k


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_default))
    return path


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    raise TypeError(type(o))
