"""Settings shared by the experiment scripts."""

from pathlib import Path

import numpy as np

from learnbranch.instances import set_cover

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

WORKERS = 2

# set covering sizes: (rows, columns); training uses SMALL only
SMALL = (500, 1000)
MEDIUM = (750, 1000)

# instance seeds: each split has its own disjoint range
SPLITS = {
    "train": (SMALL, 1000, 80),
    "valid": (SMALL, 2000, 20),
    "test_small": (SMALL, 3000, 20),
    "test_medium": (MEDIUM, 4000, 20),
}
QUICK_SPLITS = {
    "train": (SMALL, 1000, 2),
    "valid": (SMALL, 2000, 1),
    "test_small": (SMALL, 3000, 2),
    "test_medium": (MEDIUM, 4000, 1),
}

COLLECT_PROB = 0.5        # chance of recording a node
COLLECT_MAX_NODES = 40    # recorded nodes per instance
QUICK_COLLECT_MAX_NODES = 8

SOLVER_SEEDS = [0, 1, 2, 3, 4]
QUICK_SOLVER_SEEDS = [0]
POLICIES = {               # policy -> solver seeds it is run with
    "test_small": {"relpscost": SOLVER_SEEDS, "pscost": SOLVER_SEEDS,
                   "learned": SOLVER_SEEDS, "fullstrong": SOLVER_SEEDS[:2]},
    "test_medium": {"relpscost": SOLVER_SEEDS, "pscost": SOLVER_SEEDS,
                    "learned": SOLVER_SEEDS},
}
TIME_LIMIT = {"test_small": 300.0, "test_medium": 600.0}
QUICK_TIME_LIMIT = 60.0


def instance(split: str, k: int, quick: bool = False):
    (rows, cols), base, n = (QUICK_SPLITS if quick else SPLITS)[split]
    if k >= n:
        raise IndexError(k)
    return set_cover(rows, cols, np.random.default_rng(base + k))


def n_instances(split: str, quick: bool = False) -> int:
    return (QUICK_SPLITS if quick else SPLITS)[split][2]
