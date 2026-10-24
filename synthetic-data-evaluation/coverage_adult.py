"""Coverage on real data: the Adult file treated as the population.

The 48,842 Adult records play the population, and the analyst's logistic
regression fitted to all of them defines the true coefficients (the
population value of that working model). Each replication draws n = 2,000
records without replacement, builds m = 5 synthetic data sets with each
generator, simple and proper, and records whether each rule's 95% interval
contains the population coefficient.

Writes results/coverage_adult.json.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from data import load_adult
from generators import Encoder, factories, synthesize
from inference import combine, covers, logit
from metrics import ANALYSIS_TERMS, analysis_fit

N = 2000
M_SETS = 5
R = int(os.environ.get("SDE_ADULT_REPS", 1000))
SEED = 20261025
OUT = Path(__file__).resolve().parent / "results" / "coverage_adult.json"

POP = load_adult()
ENC = Encoder.from_frame(POP)
CODES = ENC.to_codes(POP)
COL = {c: i for i, c in enumerate(ENC.columns)}
MALE = list(ENC.levels["sex"]).index("Male")
MARRIED = list(ENC.levels["marital-status"]).index("Married-civ-spouse")
RICH = list(ENC.levels["income"]).index(">50K")
TRUTH = analysis_fit(POP)[0]
GENS = factories(ENC, class_col=COL["income"])


def fit_codes(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The analyst's model (see metrics.analysis_design) on a code matrix."""
    X = np.column_stack(
        [
            np.ones(len(M)),
            M[:, COL["age"]],
            M[:, COL["education-num"]],
            M[:, COL["hours-per-week"]],
            (M[:, COL["sex"]] == MALE).astype(float),
            (M[:, COL["marital-status"]] == MARRIED).astype(float),
        ]
    )
    y = (M[:, COL["income"]] == RICH).astype(float)
    return logit(X, y)


def _record(store, key, iv):
    store[key] = np.vstack([covers(iv, TRUTH), iv.estimate, iv.upper - iv.lower, iv.fallback])


def one_replication(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    real = CODES[rng.choice(len(CODES), size=N, replace=False)]
    store: dict = {}
    b, v = fit_codes(real)
    _record(store, ("real",), combine(b[None], v[None], 1.0, "naive"))
    for name, make in GENS.items():
        for proper in (False, True):
            mode = "proper" if proper else "simple"
            sets = synthesize(make, real, N, M_SETS, rng, proper)
            fits = [fit_codes(S) for S in sets]
            q = np.array([f[0] for f in fits])
            u = np.array([f[1] for f in fits])
            _record(store, (name, mode, "naive", 1), combine(q[:1], u[:1], 1.0, "naive"))
            for rule in ("naive", "simple", "proper", "raghunathan"):
                _record(store, (name, mode, rule, M_SETS), combine(q, u, 1.0, rule))
    return store


def _single_thread() -> None:
    threadpool_limits(1)


def main() -> None:
    seeds = np.random.SeedSequence(SEED).generate_state(R)
    # one BLAS/OpenMP thread per worker: the work is parallel across replications
    workers = int(os.environ.get("SDE_WORKERS", min(2, os.cpu_count() or 1)))
    with Pool(processes=workers, initializer=_single_thread) as pool:
        reps = pool.map(one_replication, [int(s) for s in seeds], chunksize=10)
    rows = []
    real_sd = np.stack([r[("real",)][1] for r in reps]).std(axis=0, ddof=1)
    for key in reps[0]:
        stack = np.stack([r[key] for r in reps])
        cov = stack[:, 0, :].mean(axis=0)
        est = stack[:, 1, :]
        rows.append(
            {
                "key": list(key),
                "coverage": cov.tolist(),
                "mc_se": np.sqrt(cov * (1 - cov) / R).tolist(),
                "bias": (est.mean(axis=0) - TRUTH).tolist(),
                "bias_in_real_sd": ((est.mean(axis=0) - TRUTH) / real_sd).tolist(),
                "sd": est.std(axis=0, ddof=1).tolist(),
                "mean_length": stack[:, 2, :].mean(axis=0).tolist(),
                "fallback_rate": stack[:, 3, :].mean(axis=0).tolist(),
            }
        )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"n": N, "m": M_SETS, "reps": R, "population": len(POP), "terms": ANALYSIS_TERMS,
             "truth": TRUTH.tolist(), "rows": rows},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
