"""Checks on the generators, the features, strong-branching labels and the rules."""

import numpy as np
import pytest
from pyscipopt import SCIP_RESULT, Branchrule

from learnbranch.branching import (FEATURES, Collector, Featuriser, LearnedBranching,
                                   strong_branching_scores)
from learnbranch.instances import facility, load, save, set_cover, to_model
from learnbranch.solve import collect, configure, solve


@pytest.fixture(scope="module")
def small_cover():
    return set_cover(250, 500, np.random.default_rng(3))   # needs a few nodes


def test_set_cover_structure(small_cover):
    assert all(len(r) >= 1 for r in small_cover.rows)
    counts = np.bincount(np.concatenate(small_cover.rows), minlength=500)
    assert counts.min() >= 1
    assert small_cover.cost.min() >= 1 and small_cover.cost.max() <= 100


def test_facility_capacity_ratio():
    f = facility(30, 20, np.random.default_rng(0), ratio=5.0)
    assert f.capacity.sum() == pytest.approx(5.0 * f.demand.sum(), rel=0.02)
    assert f.trans.shape == (30, 20)


def test_save_load_roundtrip(tmp_path, small_cover):
    save(small_cover, tmp_path / "sc.npz")
    back = load(tmp_path / "sc.npz")
    assert np.array_equal(back.cost, small_cover.cost)
    assert all(np.array_equal(a, b) for a, b in zip(back.rows, small_cover.rows))


class _Probe(Branchrule):
    """Check features and strong branching at the first node, then defer."""

    def __init__(self, featuriser):
        self.featuriser, self.result = featuriser, None

    def branchexeclp(self, allowaddcons):
        if self.result is None:
            cands, sols, fracs, _, nprio, _ = self.model.getLPBranchCands()
            cands, sols, fracs = cands[:nprio], sols[:nprio], fracs[:nprio]
            lp_before = self.model.getLPObjVal()
            X = self.featuriser(self.model, cands, sols, fracs)
            y = strong_branching_scores(self.model, cands)
            self.result = (X, y, lp_before, self.model.getLPObjVal())
        return {"result": SCIP_RESULT.DIDNOTRUN}


def test_features_and_strong_branching_leave_state_unchanged(small_cover):
    m = to_model(small_cover)
    configure(m, 0, 60)
    probe = _Probe(Featuriser.from_model(m))
    m.includeBranchrule(probe, "probe", "", priority=999999, maxdepth=-1, maxbounddist=1.0)
    m.optimize()
    assert probe.result is not None, "instance solved at the root; pick another seed"
    X, y, lp_before, lp_after = probe.result
    assert X.shape == (len(y), len(FEATURES))
    assert np.all(np.isfinite(X))
    assert np.all(X[:, FEATURES.index("frac_dist")] <= 0.5 + 1e-9)
    assert np.all(y > 0)
    assert lp_after == pytest.approx(lp_before)


class _ByPseudocost:
    """A stand-in predictor: rank candidates by their pseudocost score."""

    def predict(self, X):
        return X[:, FEATURES.index("ps_score_rel")]


def test_learned_rule_reaches_the_same_optimum(small_cover):
    ref = to_model(small_cover)
    configure(ref, 0, 60)
    ref.optimize()
    out = solve(small_cover, "learned", seed=0, time_limit=60, predictor=_ByPseudocost())
    assert out["status"] == "optimal"
    m = to_model(small_cover)
    configure(m, 0, 60)
    rule = LearnedBranching(Featuriser.from_model(m), _ByPseudocost())
    m.includeBranchrule(rule, "learned", "", priority=999999, maxdepth=-1, maxbounddist=1.0)
    m.optimize()
    assert m.getObjVal() == pytest.approx(ref.getObjVal())
    assert rule.calls == out["py_calls"]


def test_collector_records_bounded_samples(small_cover):
    samples = collect(small_cover, seed=0, rng=np.random.default_rng(0), prob=1.0, max_nodes=3)
    assert 1 <= len(samples) <= 3
    for X, info in samples:
        assert X.shape[0] == info["score"].shape[0] > 1
        assert np.all(info["down_gain"] <= info["gap"] + 1e-9)
        assert np.all(info["up_gain"] <= info["gap"] + 1e-9)


def test_collector_does_not_change_the_answer(small_cover):
    m = to_model(small_cover)
    configure(m, 1, 60)
    m.optimize()
    best = m.getObjVal()
    m2 = to_model(small_cover)
    configure(m2, 1, 60)
    col = Collector(Featuriser.from_model(m2), np.random.default_rng(0), prob=1.0, max_nodes=5)
    m2.includeBranchrule(col, "collector", "", priority=999999, maxdepth=-1, maxbounddist=1.0)
    m2.optimize()
    assert m2.getObjVal() == pytest.approx(best)
