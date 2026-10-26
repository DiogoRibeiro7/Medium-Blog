"""Shared set-up: paths, seeds, sizes and the fitted hybrids."""

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
sys.path.insert(0, str(ROOT.parent / "mechanistic-vs-data-driven-generators"
                       / "src"))

from gentradeoffs.data_driven import CartSynthesizer  # noqa: E402
from gentradeoffs.mechanistic import (DemandModel, MechanisticGenerator,  # noqa: E402
                                      erlang_models, service_mean)
from gentradeoffs.world import COLUMNS, StaffingPolicy, World, rows  # noqa: E402
from hybridgen.discrepancy import Discrepancy  # noqa: E402
from hybridgen.eta import eta_table  # noqa: E402
from hybridgen.hybrid import (GreyBox, OutputHybrid,  # noqa: E402
                              day_factor_generator)

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
WORLD = World()
POLICY = StaffingPolicy()
MASTER_SEED = 20261026
TRAIN_DAYS = 90
EVAL_DAYS = 40
LEARNERS = ("boosting", "Gaussian process", "linear",
            "boosting (link scale)")
DELAY = 600.0          # the analyst's assumed mean delay before a redial


def parse(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true",
                    help="tiny sizes, to check the code runs")
    return ap.parse_args()


def seeds(tag: str, n: int) -> list[np.random.SeedSequence]:
    entropy = [MASTER_SEED, sum(ord(c) * 31 ** i for i, c in enumerate(tag))]
    return np.random.SeedSequence(entropy).spawn(n)


def rng(seq: np.random.SeedSequence, *path: int) -> np.random.Generator:
    child = seq
    for p in path:
        child = np.random.SeedSequence(child.entropy,
                                       spawn_key=child.spawn_key + (p,))
    return np.random.default_rng(child)


def int_seed(seq: np.random.SeedSequence, *path: int) -> int:
    return int(rng(seq, *path).integers(2 ** 31))


def erlang_a(train: pd.DataFrame) -> MechanisticGenerator:
    return MechanisticGenerator(erlang_models(train)["Erlang A"],
                                DemandModel.fit(train), POLICY, WORLD)


def fit_hybrids(train: pd.DataFrame, seq: np.random.SeedSequence,
                learners: tuple[str, ...] = LEARNERS,
                greybox: bool = True, sim_days: int = 600) -> dict:
    """Erlang A, the output hybrids and the grey box, fitted on ``train``."""
    base = erlang_a(train)
    gens: dict = {"Erlang A": base}
    eta = eta_table(base.queue, rng(seq, 90), train)
    for i, learner in enumerate(learners):
        scale = "link" if learner.endswith("(link scale)") else "additive"
        kind = learner.replace(" (link scale)", "")
        d = Discrepancy(kind, service_mean(train), scale).fit(
            train, eta, rng(seq, 91, i))
        gens[f"Erlang A + {learner}"] = OutputHybrid(base, d)
    gens["Erlang A + resampled day levels"] = day_factor_generator(
        train, POLICY, WORLD)
    if greybox:
        gb = GreyBox.fit(train, int_seed(seq, 92), DELAY, POLICY, WORLD,
                         sim_days=sim_days)
        gens["Erlang A + estimated redials"] = gb
        gens["Erlang A + both"] = day_factor_generator(
            train, POLICY, WORLD, p=gb.estimate.p_hat, delay=DELAY)
    return gens


def sample(gen, g: np.random.Generator, n_days: int,
           multiplier: float = 1.0,
           calendar: list[tuple[int, bool]] | None = None,
           extra: int = 0) -> pd.DataFrame:
    """Days from any generator, as a table, with ``extra`` agents added to
    every half hour on top of the staffing rule."""
    if isinstance(gen, MechanisticGenerator):
        frames = []
        for i in range(n_days):
            if calendar is None:
                dow = i % 5
                campaign = bool(g.random() < gen.demand.campaign_prob)
            else:
                dow, campaign = calendar[i]
            out, staff = gen.day(g, dow, campaign, multiplier, extra)
            frame = rows(out, staff, dow, campaign)
            frame.insert(0, "day", i)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
    out = gen.sample_days(g, n_days, multiplier=multiplier,
                          calendar=calendar, extra=extra)
    return out[0] if isinstance(out, tuple) else out


def calendar_of(table: pd.DataFrame) -> list[tuple[int, bool]]:
    first = table.groupby("day")[["dow", "campaign"]].first()
    return [(int(d), bool(c)) for d, c in first.to_numpy()]


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


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


__all__ = ["CartSynthesizer", "COLUMNS"]
