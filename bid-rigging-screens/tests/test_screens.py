"""Checks of the methods on cases with a known answer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import screens as S  # noqa: E402


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    firms = [tuple(map(str, rng.choice(12, 3, replace=False))) for _ in range(n)]
    y = rng.integers(0, 2, n)
    d = pd.DataFrame({c: rng.normal(size=n) for c in S.ALL})
    d["median3cvbid"] = np.where(y == 1, 2.0, 5.0) + rng.normal(0, 0.5, n)
    d["y"] = y
    d["n_tenders"] = rng.integers(3, 30, n)
    d["coalition"] = [str(f) for f in firms]
    d["firms"] = firms
    return d


def test_balanced_has_equal_classes():
    d = toy()
    b = S.balanced(d, np.random.default_rng(1))
    assert (b.y == 1).sum() == (b.y == 0).sum() == min((d.y == 1).sum(), (d.y == 0).sum())


def test_split_by_firms_keeps_held_firms_out_of_training():
    d = toy()
    held = {"0", "1", "2", "3"}
    tr, te, k = S.split_by_firms(d, held)
    assert not any(f in held for t in tr.firms for f in t)
    assert all(any(f in held for f in t) for t in te.firms)
    assert len(tr) + len(te) == len(d)
    assert set(k.unique()) <= {0, 1, 2, 3}


def test_signed_screen_orients_cv_towards_collusion():
    d = toy()
    assert S.auc(d.y, S.signed_screen(d)) > 0.95


def test_within_strata_auc_equals_pooled_auc_with_one_stratum():
    d = toy()
    s = S.signed_screen(d)
    one = pd.Series(0, index=d.index)
    assert abs(S.within_strata_auc(d, s, one) - S.auc(d.y, s)) < 1e-12


def test_ranks_are_percentiles_within_the_data_set():
    d = toy()
    r = S.ranks(d)
    assert r[S.SCALE_FREE].min().min() > 0 and r[S.SCALE_FREE].max().max() == 1


def test_shares_counts_correct_classification():
    acc, c1, c0 = S.shares([1, 1, 0, 0], [1, 0, 0, 0])
    assert (acc, c1, c0) == (75.0, 50.0, 100.0)
