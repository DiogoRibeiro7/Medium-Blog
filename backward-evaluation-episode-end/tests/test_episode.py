import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402,F401
from backward_eval import ARGaussian, ARStudentT, embed_ar  # noqa: E402
from episode import Selection, criteria, end_check, run_fast, run_generic  # noqa: E402


def ar1(sd: np.ndarray, rng, phi=0.5):
    y = np.zeros(len(sd))
    for t in range(1, len(sd)):
        y[t] = phi * y[t - 1] + sd[t] * rng.normal()
    return y


def test_end_check_detects_an_ended_episode():
    rng = np.random.default_rng(0)
    sd = np.ones(361)
    sd[240:264] = 3.0  # an episode early in the backward window, over by the last 12 rows
    X, z, _ = embed_ar(ar1(sd, rng), 1)
    o = 300  # backward window [240, 300) = block j = 1, rejected block [180, 240)
    z_end, m = end_check(X, z, o, 1, 60, 12, ARGaussian)
    assert z_end > 1.645 and m > 0


def test_end_check_keeps_a_lasting_change():
    zs = []
    for rep in range(40):
        sd = np.ones(361)
        sd[240:] = 2.0  # a lasting rise in volatility
        X, z, _ = embed_ar(ar1(sd, np.random.default_rng(rep)), 1)
        zs.append(end_check(X, z, 300, 1, 60, 12, ARGaussian)[0])
    zs = np.array(zs)
    assert zs.mean() < 0 and np.mean(zs > 1.645) < 0.15


def test_end_check_is_neutral_when_the_change_is_only_in_the_recent_rows():
    rng = np.random.default_rng(2)
    sds = []
    for rep in range(40):
        sd = np.ones(361)
        sd[288:300] = 2.0  # only the last 12 rows changed: neither model has seen it
        X, z, _ = embed_ar(ar1(sd, np.random.default_rng(100 + rep)), 1)
        sds.append(end_check(X, z, 300, 1, 60, 12, ARGaussian)[0])
    assert np.mean(np.array(sds) > 1.645) < 0.25


def test_start_C_switches_only_where_the_change_ended():
    sel = Selection(start_B=np.array([0, 100, 100, 50]), stopped=np.array([False, True, True, True]),
                    z_end={12: np.array([np.nan, 2.0, 1.0, -3.0])}, decisions=[])
    assert sel.start_C(12, 1.645).tolist() == [0, 0, 100, 50]
    assert sel.start_C(12, 0.0).tolist() == [0, 0, 0, 50]


def test_fast_and_generic_paths_agree_for_gaussian_models():
    rng = np.random.default_rng(3)
    sd = np.ones(481)
    sd[300:324] = 3.0
    X, z, _ = embed_ar(ar1(sd, rng), 1)
    origins = np.arange(330, 480)
    kw = dict(reselect=12, folds=5, seed=4, rs=(12,), cs=(1.645,), primary=(12, 1.645))
    fast, sf = run_fast(X, z, origins, 60, 4, 2.498, 6, **kw)
    gen, sg = run_generic(X, z, origins, ARGaussian, 60, 4, 2.498, 6, **kw)
    assert np.array_equal(sf.start_B, sg.start_B)
    for k in ("E", "B", "C", "P0.25", "PC0.25", "AveW"):
        np.testing.assert_allclose(fast[k].nlpd, gen[k].nlpd, rtol=1e-6, atol=1e-8)


def test_pooled_checked_rule_reduces_to_E_where_the_check_switches():
    rng = np.random.default_rng(5)
    sd = np.ones(481)
    sd[300:324] = 4.0
    X, z, _ = embed_ar(ar1(sd, rng), 1)
    origins = np.arange(330, 480)
    out, sel = run_generic(X, z, origins, ARStudentT, 60, 4, 2.498, 6, avew=False, reselect=12, folds=5, seed=6)
    useC = sel.stopped & (sel.start_C(12, 1.645) > 0)
    np.testing.assert_allclose(out["PC0.25"].nlpd[~useC], out["E"].nlpd[~useC])
    np.testing.assert_allclose(out["C"].nlpd[~useC], out["E"].nlpd[~useC])


def test_criteria_verdicts():
    def g(B, C, AveW, P):
        return {"E": 0.0, "B": B, "C": C, "AveW": AveW, "P0.25": P}
    good = {"a": g(-0.2, -0.05, 0.0, 0.0), "b": g(0.1, 0.09, 0.0, 0.0)}
    assert criteria(good, ["a", "b"])["verdict"] == "supported"
    repair = {"a": g(-0.2, -0.05, 0.3, 0.0), "b": g(0.1, 0.09, 0.3, 0.0)}
    assert criteria(repair, ["a", "b"])["verdict"] == "repairs the rule but does not beat the defaults"
    bad = {"a": g(-0.2, -0.19, 0.0, 0.0), "b": g(0.1, 0.09, 0.0, 0.0)}
    assert criteria(bad, ["a", "b"])["verdict"] == "not supported"
    no_losses = {"a": g(0.2, 0.15, 0.0, 0.1), "b": g(0.1, 0.09, 0.0, 0.0)}
    cr = criteria(no_losses, ["a", "b"])
    assert cr["H1"] is None and cr["verdict"] == "supported"
