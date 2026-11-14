import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import build as B  # noqa: E402
import synth  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synth")
    ppd = tmp / "pp.csv"
    paths = synth.make_ppd(ppd)
    stats = B.build(str(ppd), tmp / "data", memory_limit="2GB", threads=2, min_area_pairs=100)
    return stats, tmp / "data", paths


def test_counts_add_up(built):
    stats, data, _ = built
    hc = pd.read_csv(data / "hazard_cells.csv.gz")
    dy = pd.read_csv(data / "district_year.csv.gz")
    assert stats["areas_in_index"] == 4
    assert hc.stock.sum() == dy.stock.sum() == stats["owner_years"]
    assert hc.resales.sum() == dy.resales.sum() == stats["resales"]
    band = [c for c in dy.columns if c.startswith("stock_") and c not in ("stock_flat", "stock_new")]
    m_bands = [c for c in band if not c.startswith("stock_d")]
    d_bands = [c for c in band if c.startswith("stock_d")]
    assert (dy[m_bands].sum(axis=1) == dy.stock).all()
    assert (dy[d_bands].sum(axis=1) == dy.stock).all()
    assert (dy.resales <= dy.stock).all() and (dy.resales <= dy.sales_all).all()


def test_no_addresses_committed(built):
    _, data, _ = built
    for f in data.iterdir():
        if f.suffix in (".csv", ".gz"):
            cols = set(pd.read_csv(f, nrows=1).columns)
            assert not cols & {"paon", "saon", "street", "postcode", "pc", "price", "p", "tuid"}


def test_index_tracks_market(built):
    _, data, paths = built
    idx = pd.read_csv(data / "area_index.csv")
    for a, true in paths.items():
        est = idx[idx.area == a].sort_values("t").log_index.to_numpy()[36:360]
        tr = true[36:360]
        assert np.corrcoef(est - est.mean(), tr - tr.mean())[0, 1] > 0.99


def test_planted_lock_in_visible(built):
    _, data, _ = built
    hc = pd.read_csv(data / "hazard_cells.csv.gz")
    hc = hc[hc.dur_year.between(2, 8)]
    below = hc[hc.m_bin.between(-10, -3)]
    above = hc[hc.m_bin.between(2, 9)]
    h_below = below.resales.sum() / below.stock.sum()
    h_above = above.resales.sum() / above.stock.sum()
    assert h_below < 0.75 * h_above


def test_analysis_recovers_planted_lock_in(built, tmp_path, monkeypatch):
    _, data, _ = built
    import analysis as A
    monkeypatch.setattr(A, "DATA", data)
    monkeypatch.setattr(A, "RES", tmp_path)
    assert A.main() == 0
    curve = pd.read_csv(tmp_path / "lock_in_curve.csv")
    c = curve[curve.period == "all"].set_index("m_bin").effect
    assert c.loc[-3] < -0.005 and c.loc[8] > 0.01          # below water sells less, well above more
    cf = pd.read_csv(tmp_path / "counterfactual.csv").set_index("year")
    fall = cf.rate[2007] - cf.rate[2009]
    narrow = fall - (cf.rate_without_lock_in[2007] - cf.rate_without_lock_in[2009])
    wide = fall - (cf.rate_without_position[2007] - cf.rate_without_position[2009])
    # lock-in is the only cause of the fall in the synthetic market: both measures find part of
    # it, the narrow one (owners below zero only) less than the wide one
    assert 0.1 * fall < narrow < wide


def test_monthly_cohorts_match_annual_stock(built):
    _, data, _ = built
    mp = pd.read_csv(data / "monthly_position.csv")
    hc = pd.read_csv(data / "hazard_cells.csv.gz")
    jan = mp[mp.month == 1].groupby("year").stock.sum()
    annual = hc.groupby("year").stock.sum()
    common = jan.index.intersection(annual.index)
    assert len(common) >= 15
    assert (jan.loc[common] == annual.loc[common]).all()
    yearly = mp.groupby("year").resales.sum()
    assert (yearly.loc[common] >= hc.groupby("year").resales.sum().loc[common]).all()
