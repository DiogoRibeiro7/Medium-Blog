"""Shared set-up for the experiments: sizes, seeds and fitted generators."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gentradeoffs.data_driven import (CartSynthesizer, GaussianCopula,  # noqa: E402
                                      GaussianMixture, Smote)
from gentradeoffs.mechanistic import (DemandModel, MechanisticGenerator,  # noqa: E402
                                      erlang_models)
from gentradeoffs.world import COLUMNS, StaffingPolicy, World  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
WORLD = World()
POLICY = StaffingPolicy()
MASTER_SEED = 20261024
TRAIN_DAYS = 90
EVAL_DAYS = 40
GMM_CANDIDATES = (4, 8, 12, 16, 24)


def parse(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true",
                    help="tiny sizes, to check the code runs")
    return ap.parse_args()


def seeds(tag: str, n: int) -> list[np.random.SeedSequence]:
    """Independent seed sequences for ``n`` replicates of experiment ``tag``."""
    entropy = [MASTER_SEED, sum(ord(c) * 31 ** i for i, c in enumerate(tag))]
    return np.random.SeedSequence(entropy).spawn(n)


def rng(seq: np.random.SeedSequence, *path: int) -> np.random.Generator:
    """A generator for one branch of a replicate, by position."""
    child = seq
    for p in path:
        child = np.random.SeedSequence(child.entropy,
                                       spawn_key=child.spawn_key + (p,))
    return np.random.default_rng(child)


NAMES = ("Erlang C", "Erlang A", "Gaussian copula", "CART", "SMOTE",
         "Gaussian mixture")


def make_generator(name: str, train: pd.DataFrame, seed: int = 0,
                   gmm_components: int | None = None):
    """Fit one generator of the comparison on a training table."""
    if name in ("Erlang C", "Erlang A"):
        queue = erlang_models(train)[name]
        return MechanisticGenerator(queue, DemandModel.fit(train), POLICY,
                                    WORLD)
    data = train[COLUMNS]
    if name == "Gaussian copula":
        return GaussianCopula().fit(data)
    if name == "CART":
        return CartSynthesizer(seed=seed).fit(data)
    if name == "SMOTE":
        return Smote().fit(data)
    if name == "Gaussian mixture":
        return GaussianMixture(gmm_components, candidates=GMM_CANDIDATES,
                               seed=seed).fit(data)
    raise ValueError(name)


def fit_generators(train: pd.DataFrame, seed: int = 0,
                   gmm_components: int | None = None,
                   which: tuple[str, ...] | None = None) -> dict:
    """Generators of the comparison, fitted on one training table."""
    return {name: make_generator(name, train, seed, gmm_components)
            for name in (which or NAMES)}


def cell_means(train: pd.DataFrame) -> pd.Series:
    return train.groupby(["dow", "interval", "campaign"])["offered"].mean()


def expected_offered(train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Training mean of offered calls for each row's calendar cell, falling
    back to the interval-and-weekday mean when the cell is empty."""
    full = cell_means(train)
    fallback = train.groupby(["dow", "interval"])["offered"].mean()
    keys = list(zip(frame["dow"], frame["interval"], frame["campaign"]))
    out = np.array([full.get(k, np.nan) for k in keys], dtype=float)
    miss = np.isnan(out)
    if miss.any():
        out[miss] = [fallback[(d, i)] for d, i, _ in np.array(keys)[miss]]
    return out


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t
