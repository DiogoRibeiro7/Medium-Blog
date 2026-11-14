import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import build as B  # noqa: E402
import model as M  # noqa: E402
import synth  # noqa: E402


def test_area_index_exact_without_smoothing():
    # three months, true log index 0, 0.1, 0.3; pairs measured without noise
    agg = pd.DataFrame({"area": "X", "t0": [0, 0, 1], "t1": [1, 2, 2], "n": [5, 3, 4],
                        "s": [5 * 0.1, 3 * 0.3, 4 * 0.2]})
    idx = B.area_index(agg, 3, lam=0.0)
    assert np.allclose(idx.log_index, [0.0, 0.1, 0.3])
    assert idx.month.tolist() == ["1995-01-01", "1995-02-01", "1995-03-01"]


def test_conformal_quantile_and_coverage():
    rng = np.random.default_rng(0)
    s = rng.normal(size=10000)
    q = M.conformal_q(s, 0.2)
    assert abs((s <= q).mean() - 0.8) < 0.002


def test_local_linear_recovers_kink_and_jump():
    rng = np.random.default_rng(1)
    m = rng.uniform(-0.3, 0.3, 200000)
    r = 0.01 + 0.2 * m - 0.3 * np.minimum(m, 0) + 0.02 * (m >= 0) + rng.normal(0, 0.05, m.size)
    cl = rng.integers(0, 400, m.size).astype(str)
    est = M.local_linear(m, r, cl, 0.0, 0.08)
    assert abs(est["kink"] - 0.3) < 0.03
    assert abs(est["jump"] - 0.02) < 0.004
    placebo = M.local_linear(m, r, cl, 0.15, 0.08)
    assert abs(placebo["kink"]) < 0.03 and abs(placebo["jump"]) < 0.004


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synth")
    ppd = tmp / "pp.csv"
    paths = synth.make_ppd(ppd, homes_per_area=4000)
    data = tmp / "data"
    pairs = tmp / "pairs.parquet"
    stats = B.build(str(ppd), str(pairs), data, memory_limit="2GB", threads=2, min_area_pairs=100)
    out = tmp / "results"
    argv = sys.argv
    sys.argv = ["model.py", "--pairs", str(pairs), "--data", str(data), "--out", str(out),
                "--cap", "60000", "--qcap", "40000", "--rounds", "150", "--qrounds", "100",
                "--threads", "2"]
    try:
        assert M.main() == 0
    finally:
        sys.argv = argv
    return stats, data, out, paths


def test_build_counts_and_privacy(run):
    stats, data, out, _ = run
    assert stats["areas_in_index"] == 4
    assert stats["clean_pairs"] > 20000
    # nothing committed carries an address or a single transaction
    for f in list(data.glob("*.csv")) + list(out.glob("*.csv")):
        cols = set(pd.read_csv(f, nrows=1).columns)
        assert not cols & {"paon", "saon", "street", "postcode", "pc", "sector", "outward", "p", "p0"}


def test_index_tracks_the_planted_market(run):
    _, data, _, paths = run
    idx = pd.read_csv(data / "area_index.csv")
    for a, true in paths.items():
        est = idx[idx.area == a].sort_values("t").log_index.to_numpy()
        sl = slice(36, 360)
        d_est = est[sl] - est[sl].mean()
        d_true = true[sl] - true[sl].mean()
        assert np.corrcoef(d_est, d_true)[0, 1] > 0.99
        assert np.abs(d_est - d_true).max() < 0.06


def test_floor_shows_in_event_probabilities_not_in_extrapolation(run):
    _, _, out, _ = run
    ev = pd.read_csv(out / "events.csv")
    a = ev[ev.group_type == "all"].set_index(["split", "model"])
    # trained on years with losses, the classifier beats the implied distribution ...
    assert a.loc[("2022-2025", "C2"), "brier_loss"] < a.loc[("2022-2025", "M0-dist"), "brier_loss"]
    # ... the implied distribution overstates losses, because the floor removes some
    assert a.loc[("2022-2025", "M0-dist"), "pred_loss"] > a.loc[("2022-2025", "M0-dist"), "obs_loss"]
    # trained only on rising years, the classifier cannot foresee a fall; the index can
    assert a.loc[("2008-2012", "C2"), "brier_loss"] > a.loc[("2008-2012", "M0-dist"), "brier_loss"]
    met = pd.read_csv(out / "metrics.csv")
    alls = met[(met.group_type == "all")].set_index(["split", "model"])
    assert alls.loc[("2022-2025", "M2"), "mae"] < alls.loc[("2022-2025", "M0"), "mae"]


def test_partial_dependence_on_the_gain_is_higher_below_zero(run):
    # holding the indexed value fixed, a higher purchase price (lower m) means a planted markup
    _, _, out, _ = run
    pdp = pd.read_csv(out / "pdp_m.csv")
    for split in ["2017-2019", "2022-2025"]:   # 2008-2012 was trained on rising years only
        s = pdp[pdp.split == split].set_index("m").mean_pred
        assert s.loc[-0.2] - s.loc[0.1] > 0.01


def test_event_classes_and_implied_probabilities():
    lr = np.array([-0.05, -0.01, 0.0, 0.01, 0.05])
    exact = np.array([False, False, True, False, False])
    assert M.event_class(lr, exact).tolist() == [0, 1, 2, 3, 4]
    rng = np.random.default_rng(2)
    u = rng.normal(0, 0.1, 100000)
    p = M.dist_events(u, np.array([0.0, 0.3]))
    assert np.allclose(p.sum(1), 1)
    assert abs(p[0, :2].sum() - 0.5) < 0.01 and p[1, :2].sum() < 0.01
    assert (p[:, 2] == 0).all()


def test_outputs_present(run):
    _, _, out, _ = run
    for f in ["metrics.csv", "coverage.csv", "curve_observed.csv", "curve_predicted.csv",
              "importance.csv", "pdp_m.csv", "discontinuity.csv", "events.csv",
              "model_run.json"]:
        assert (out / f).stat().st_size > 0
    ev = pd.read_csv(out / "events.csv")
    a = ev[ev.group_type == "all"].set_index(["split", "model"])
    # the learned classifier knows about exact resales; the implied distributions cannot
    assert (a.xs("C2", level="model").brier_exact < a.xs("M0-dist", level="model").brier_exact).all()
    cov = pd.read_csv(out / "coverage.csv")
    g = cov[(cov.group_type == "all") & (cov.method == "global")]
    assert g.coverage.between(0.70, 0.90).all()
