"""Shared paths and the cached network."""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # also works without pip install -e
DATA = Path(os.environ.get("CABLE_DATA", ROOT / "data"))
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
CACHE = RESULTS / "network.pkl"

ZONE_KM = 100.0            # incidents are centred within this distance of an island's landing stations
RADII_KM = (1.0, 10.0, 50.0)


def network():
    from cableresilience.data import load
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    net = load(DATA)
    CACHE.write_bytes(pickle.dumps(net))
    return net


def save_json(name, obj):
    import numpy as np

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    (RESULTS / name).write_text(json.dumps(obj, indent=1, default=default))


def load_json(name):
    return json.loads((RESULTS / name).read_text())
