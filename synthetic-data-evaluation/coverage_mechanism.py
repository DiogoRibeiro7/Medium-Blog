"""Coverage of nominal 95% intervals computed from synthetic data, known truth.

For each of R replications: draw n = 1,000 records from the mechanism in
`mechanism.py`, build synthetic data with each generator (simple and proper
synthesis), fit the analyst's regression to every synthetic data set and
combine the results with each rule in `inference.py`. Two experiments share
the replications:

* rules: m = 1, 2, 5, 10, 20 synthetic data sets of n rows each (k = 1);
* size:  one synthetic data set of k n rows, k = 1, 2, 5, 10, 20, analysed
  as if it were real.

Writes results/coverage_mechanism.json.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from generators import factories, synthesize
from inference import combine, covers
from mechanism import TERMS, TRUE_BETA, Parametric, draw, encoder, fit_analysis

N = 1000
R = int(os.environ.get("SDE_MECH_REPS", 2000))
MS = [1, 2, 5, 10, 20]
KS = [1, 2, 5, 10, 20]
RULES = ["naive", "simple", "proper", "raghunathan"]
SEED = 20261024
OUT = Path(__file__).resolve().parent / "results" / "coverage_mechanism.json"

ENC = encoder()
GENS = {"parametric": lambda: Parametric(ENC)}
GENS.update({k: v for k, v in factories(ENC, class_col=None).items()})


def _record(store, key, iv, truth):
    store[key].append(
        np.vstack([covers(iv, truth), iv.estimate, iv.upper - iv.lower, iv.fallback])
    )


def one_replication(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    real = ENC.to_codes(draw(N, rng))
    store: dict = defaultdict(list)
    b, v = fit_analysis(real)
    _record(store, ("real",), combine(b[None], v[None], 1.0, "naive"), TRUE_BETA)
    for name, make in GENS.items():
        for proper in (False, True):
            mode = "proper" if proper else "simple"
            sets = synthesize(make, real, N, max(MS), rng, proper)
            fits = [fit_analysis(S) for S in sets]
            q = np.array([f[0] for f in fits])
            u = np.array([f[1] for f in fits])
            for m in MS:
                for rule in RULES:
                    if rule == "raghunathan" and m < 2:
                        continue
                    _record(store, ("rules", name, mode, rule, m), combine(q[:m], u[:m], 1.0, rule), TRUE_BETA)
            big = synthesize(make, real, N * max(KS), 1, rng, proper)[0]
            for k in KS:
                bk, vk = fit_analysis(big[: N * k])
                _record(store, ("size", name, mode, k), combine(bk[None], vk[None], 1.0, "naive"), TRUE_BETA)
    return {k: np.array(v[0]) for k, v in store.items()}


def _single_thread() -> None:
    threadpool_limits(1)


def main() -> None:
    seeds = np.random.SeedSequence(SEED).generate_state(R)
    # one BLAS/OpenMP thread per worker: the work is parallel across replications
    workers = int(os.environ.get("SDE_WORKERS", min(2, os.cpu_count() or 1)))
    with Pool(processes=workers, initializer=_single_thread) as pool:
        reps = pool.map(one_replication, [int(s) for s in seeds], chunksize=10)
    keys = reps[0].keys()
    summary = []
    for key in keys:
        stack = np.stack([r[key] for r in reps])  # R x 4 x 5
        cov = stack[:, 0, :].mean(axis=0)
        est = stack[:, 1, :]
        summary.append(
            {
                "key": list(key),
                "coverage": cov.tolist(),
                "mc_se": np.sqrt(cov * (1 - cov) / R).tolist(),
                "bias": (est.mean(axis=0) - TRUE_BETA).tolist(),
                "sd": est.std(axis=0, ddof=1).tolist(),
                "mean_length": stack[:, 2, :].mean(axis=0).tolist(),
                "fallback_rate": stack[:, 3, :].mean(axis=0).tolist(),
            }
        )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"n": N, "reps": R, "terms": TERMS, "truth": TRUE_BETA.tolist(), "rows": summary}, indent=1))


if __name__ == "__main__":
    main()
