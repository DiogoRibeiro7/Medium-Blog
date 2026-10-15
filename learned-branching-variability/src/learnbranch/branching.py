"""Candidate features, strong-branching labels, and the two Python branching rules.

`Collector` runs at selected nodes, computes full strong-branching scores for
every candidate (without changing SCIP's state), stores (features, scores) and
then returns DIDNOTRUN so that SCIP's own rule still makes the decision: the
training states come from default-solver trajectories, as in most of the
imitation-learning literature.

`LearnedBranching` computes the same features, asks a fitted regressor for a
score per candidate, and branches on the highest.

Features follow the spirit of Khalil et al. (2016): cheap per-variable
quantities, several of them normalised within the node so that a model trained
on one instance size can be applied to another.
"""

from __future__ import annotations

import time

import numpy as np
from pyscipopt import SCIP_BRANCHDIR, SCIP_RESULT, Branchrule

FEATURES = [
    "obj_rel",          # objective coefficient / largest among candidates
    "frac",             # LP value's fractional part
    "frac_dist",        # min(frac, 1 - frac): distance to the nearest integer
    "ps_score_rel",     # pseudocost score / largest among candidates
    "pc_down_rel",      # downward pseudocost / largest among candidates
    "pc_up_rel",        # upward pseudocost / largest among candidates
    "locks_rel",        # up + down locks / largest among candidates
    "nbranch_rel",      # times branched on so far / largest among candidates
    "avg_sol",          # average value in solutions found so far
    "colsize_rel",      # rows the column appears in / largest among candidates
    "cost_per_row_rel", # objective / column size, relative to the node minimum
    "depth",            # node depth
    "log_ncands",       # log number of candidates at the node
    "gap",              # relative gap between LP bound and incumbent (1 if none)
]
SB_ITLIM = 100_000
EPS = 1e-6


def _rel(v: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(v))
    return v / m if m > 0 else np.zeros_like(v)


class Featuriser:
    """Static column statistics keyed by transformed-variable name."""

    def __init__(self, colsize: dict[str, float]):
        self.colsize = colsize

    @classmethod
    def from_model(cls, model) -> "Featuriser":
        size: dict[str, float] = {}
        for cons in model.getConss():
            for v in model.getValsLinear(cons):
                size[v] = size.get(v, 0.0) + 1.0
        return cls({f"t_{k}": c for k, c in size.items()})

    def __call__(self, model, cands, sols, fracs) -> np.ndarray:
        n = len(cands)
        obj = np.array([v.getObj() for v in cands])
        frac = np.asarray(fracs, float)
        ps = np.array([model.getVarPseudocostScore(v, s) for v, s in zip(cands, sols)])
        pcd = np.array([model.getVarPseudocost(v, SCIP_BRANCHDIR.DOWNWARDS) for v in cands])
        pcu = np.array([model.getVarPseudocost(v, SCIP_BRANCHDIR.UPWARDS) for v in cands])
        locks = np.array([v.getNLocksDown() + v.getNLocksUp() for v in cands], float)
        nbr = np.array([v.getNBranchings(SCIP_BRANCHDIR.DOWNWARDS)
                        + v.getNBranchings(SCIP_BRANCHDIR.UPWARDS) for v in cands], float)
        avg = np.array([v.getAvgSol() for v in cands])
        col = np.array([self.colsize.get(v.name, 1.0) for v in cands])
        cpr = obj / np.maximum(col, 1.0)
        mn = np.min(np.abs(cpr))
        cpr_rel = np.abs(cpr) / mn if mn > 0 else np.ones(n)
        lp = model.getLPObjVal()
        pb = model.getPrimalbound()
        gap = abs(pb - lp) / max(abs(pb), 1e-9) if model.getNSols() > 0 else 1.0
        node = np.array([model.getDepth(), np.log(n), min(gap, 1.0)])
        X = np.column_stack([
            _rel(obj), frac, np.minimum(frac, 1 - frac), _rel(ps), _rel(pcd), _rel(pcu),
            _rel(locks), _rel(nbr), avg, _rel(col), cpr_rel,
            np.tile(node, (n, 1)),
        ])
        return X


def strong_branching(model, cands) -> dict:
    """Full strong branching on every candidate, leaving SCIP's state unchanged.

    Gains are measured from the node's LP bound and capped at the distance to
    the cutoff bound: a child whose bound reaches the cutoff (reported by SCIP
    as infeasible) would be pruned, and any larger gain is worth no more. The
    score is SCIP's product rule, max(down, eps) * max(up, eps), so among
    candidates that prune one child the other child's gain still ranks them.
    """
    lp = model.getLPObjVal()
    gap = max(model.getCutoffbound() - lp, EPS)
    n = len(cands)
    dg, ug = np.empty(n), np.empty(n)
    dinf, uinf, err = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    model.startStrongbranch()
    try:
        for k, v in enumerate(cands):
            down, up, _, _, di, ui, _, _, lperr = \
                model.getVarStrongbranch(v, SB_ITLIM, idempotent=True)
            err[k], dinf[k], uinf[k] = lperr, di, ui
            dg[k] = gap if di else min(max(down - lp, 0.0), gap)
            ug[k] = gap if ui else min(max(up - lp, 0.0), gap)
    finally:
        model.endStrongbranch()
    score = np.maximum(dg, EPS) * np.maximum(ug, EPS)
    score[err] = np.nan
    return dict(score=score, down_gain=dg, up_gain=ug, down_inf=dinf, up_inf=uinf,
                gap=np.full(n, gap))


def strong_branching_scores(model, cands) -> np.ndarray:
    return strong_branching(model, cands)["score"]


class Collector(Branchrule):
    """Record features and strong-branching scores at sampled nodes."""

    def __init__(self, featuriser: Featuriser, rng: np.random.Generator,
                 prob: float = 0.5, max_nodes: int = 60):
        self.featuriser, self.rng, self.prob, self.max_nodes = featuriser, rng, prob, max_nodes
        self.samples: list[tuple[np.ndarray, dict]] = []

    def branchexeclp(self, allowaddcons):
        if len(self.samples) < self.max_nodes and self.rng.random() < self.prob:
            cands, sols, fracs, _, nprio, _ = self.model.getLPBranchCands()
            cands, sols, fracs = cands[:nprio], sols[:nprio], fracs[:nprio]
            if len(cands) > 1:
                X = self.featuriser(self.model, cands, sols, fracs)
                sb = strong_branching(self.model, cands)
                ok = np.isfinite(sb["score"])
                if ok.sum() > 1:
                    self.samples.append((X[ok], {k: v[ok] for k, v in sb.items()}))
        return {"result": SCIP_RESULT.DIDNOTRUN}


class LearnedBranching(Branchrule):
    """Branch on the candidate with the highest predicted score."""

    def __init__(self, featuriser: Featuriser, predictor):
        self.featuriser, self.predictor = featuriser, predictor
        self.calls, self.seconds = 0, 0.0

    def branchexeclp(self, allowaddcons):
        t0 = time.perf_counter()
        cands, sols, fracs, _, nprio, _ = self.model.getLPBranchCands()
        cands, sols, fracs = cands[:nprio], sols[:nprio], fracs[:nprio]
        if len(cands) == 1:
            best = 0
        else:
            X = self.featuriser(self.model, cands, sols, fracs)
            best = int(np.argmax(self.predictor.predict(X)))
        self.model.branchVar(cands[best])
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return {"result": SCIP_RESULT.BRANCHED}
