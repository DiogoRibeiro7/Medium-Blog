"""Experiment 3: solve every test instance under every policy and solver seed.

Policies: SCIP's default reliability pseudocost branching (relpscost), plain
pseudocost branching (pscost), the learned scorer (learned) and, on the small
test set only, full strong branching (fullstrong), the rule being imitated.
Each (instance, policy, seed) run is independent; the seed changes SCIP's
random seed shift and permutes variables and constraints.

Outputs
  results/e3_runs.csv   one row per run (appended, so an interrupted run resumes)
"""

from __future__ import annotations

import os

# one thread per solve: OpenMP threads inside the regressor's predict otherwise
# spin against the other worker and make every branching call far slower
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import time
from multiprocessing import get_context

import joblib
from threadpoolctl import threadpool_limits

import config
from learnbranch.solve import solve

FIELDS = ["split", "instance", "policy", "seed", "time", "cpu", "nodes", "status",
          "gap", "py_seconds", "py_calls"]
_model = None


def _init(model_path):
    global _model
    threadpool_limits(1)
    _model = joblib.load(model_path)


def _task(args):
    split, k, policy, seed, quick = args
    inst = config.instance(split, k, quick)
    tl = config.QUICK_TIME_LIMIT if quick else config.TIME_LIMIT[split]
    r = solve(inst, policy, seed=seed, time_limit=tl,
              predictor=_model if policy == "learned" else None)
    return dict(split=split, instance=k, **r)


def main(quick: bool = False):
    sfx = "_quick" if quick else ""
    path = config.RESULTS / f"e3_runs{sfx}.csv"
    done = set()
    if path.exists():
        with open(path) as fh:
            for row in csv.DictReader(fh):
                done.add((row["split"], int(row["instance"]), row["policy"], int(row["seed"])))
    seeds_for = (lambda pol, split: config.QUICK_SOLVER_SEEDS) if quick else \
        (lambda pol, split: config.POLICIES[split][pol])
    tasks = []
    for split in ["test_small", "test_medium"]:
        for k in range(config.n_instances(split, quick)):
            for policy in config.POLICIES[split]:
                for seed in seeds_for(policy, split):
                    if (split, k, policy, seed) not in done:
                        tasks.append((split, k, policy, seed, quick))
    # longest jobs first keeps both workers busy at the end
    weight = {"fullstrong": 4, "learned": 2, "relpscost": 2, "pscost": 1}
    tasks.sort(key=lambda t: (t[0] != "test_medium", -weight[t[2]]))
    print(f"{len(tasks)} runs to do, {len(done)} already in {path.name}", flush=True)
    t0 = time.perf_counter()
    new = not path.exists()
    with open(path, "a", newline="") as fh, \
            get_context("fork").Pool(config.WORKERS, initializer=_init,
                                     initargs=(config.RESULTS / f"model{sfx}.joblib",)) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        for i, r in enumerate(pool.imap_unordered(_task, tasks), 1):
            w.writerow({k: r[k] for k in FIELDS})
            fh.flush()
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} runs, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
