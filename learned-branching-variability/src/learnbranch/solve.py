"""Solve one instance under one branching policy and one random seed.

All policies share the settings used by Gasse et al. (2019): cutting planes
only at the root, no restarts, everything else at SCIP defaults. The seed
changes SCIP's random seed shift and permutes the order of variables and
constraints, which is the usual way of exposing performance variability
(Lodi and Tramontani, 2013).
"""

from __future__ import annotations

import time

import numpy as np

from .branching import Collector, Featuriser, LearnedBranching
from .instances import to_model

SCIP_RULES = {"relpscost", "pscost", "fullstrong", "random", "mostinf"}


def configure(model, seed: int, time_limit: float) -> None:
    model.setParam("limits/time", time_limit)
    model.setParam("presolving/maxrestarts", 0)
    model.setParam("separating/maxrounds", 0)
    model.setParam("randomization/randomseedshift", seed)
    model.setParam("randomization/permutationseed", seed)
    model.setParam("randomization/permuteconss", True)
    model.setParam("randomization/permutevars", True)
    model.setParam("parallel/maxnthreads", 1)
    model.setParam("lp/threads", 1)


def solve(inst, policy: str, seed: int = 0, time_limit: float = 600.0,
          predictor=None) -> dict:
    """Return wall and CPU time, nodes, status and the Python branching overhead."""
    m = to_model(inst)
    configure(m, seed, time_limit)
    rule = None
    if policy in SCIP_RULES:
        m.setParam(f"branching/{policy}/priority", 666666)
    elif policy == "learned":
        rule = LearnedBranching(Featuriser.from_model(m), predictor)
        m.includeBranchrule(rule, "learned", "regressor imitating strong branching",
                            priority=999999, maxdepth=-1, maxbounddist=1.0)
    else:
        raise ValueError(policy)
    t0, c0 = time.perf_counter(), time.process_time()
    m.optimize()
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    out = dict(policy=policy, seed=seed, time=wall, cpu=cpu, nodes=m.getNNodes(),
               status=m.getStatus(), gap=m.getGap(),
               py_seconds=rule.seconds if rule else 0.0,
               py_calls=rule.calls if rule else 0)
    m.freeProb()
    return out


def collect(inst, seed: int, rng: np.random.Generator, prob: float = 0.5,
            max_nodes: int = 60, time_limit: float = 600.0):
    """Run the default solver and record strong-branching samples at sampled nodes."""
    m = to_model(inst)
    configure(m, seed, time_limit)
    col = Collector(Featuriser.from_model(m), rng, prob=prob, max_nodes=max_nodes)
    m.includeBranchrule(col, "collector", "records strong-branching scores",
                        priority=999999, maxdepth=-1, maxbounddist=1.0)
    m.optimize()
    m.freeProb()
    return col.samples
