import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import regions as R  # noqa: E402

needs_data = pytest.mark.skipif(not (R.WORKBOOK.exists() and R.SHAPES.exists()),
                                reason="download the Rosés-Wolf files first (check_data.py)")


def grid(nx=12, ny=10):
    nodes = [f"{i}-{j}" for i in range(nx) for j in range(ny)]
    nb = {c: [] for c in nodes}
    pairs = []
    for i in range(nx):
        for j in range(ny):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < nx and j + dj < ny:
                    a, b = f"{i}-{j}", f"{i + di}-{j + dj}"
                    nb[a].append(b)
                    nb[b].append(a)
                    pairs.append((a, b))
    return nodes, nb, pairs


def test_between_share_limits_and_weights():
    y = np.array([1.0, 1.0, 3.0, 3.0])
    assert R.between_share(y, np.array([0, 0, 1, 1]), np.ones(4)) == pytest.approx(1.0)
    assert R.between_share(y, np.zeros(4), np.ones(4)) == pytest.approx(0.0)
    y = np.array([0.0, 2.0, 10.0, 12.0])
    g = np.array([0, 0, 1, 1])
    w = np.array([1.0, 3.0, 1.0, 1.0])
    m = np.average(y, weights=w)
    gm = np.array([np.average(y[:2], weights=w[:2])] * 2 + [11.0] * 2)
    expect = 1 - np.average((y - gm) ** 2, weights=w) / np.average((y - m) ** 2, weights=w)
    assert R.between_share(y, g, w) == pytest.approx(expect)


def test_random_partition_is_complete_and_contiguous():
    nodes, nb, _ = grid()
    sizes = [5, 8, 10, 12, 15, 20, 20, 30]
    rng = np.random.default_rng(3)
    parts = R.draw_partitions(nodes, nb, sizes, 20, rng, best_of=10)
    for p in parts:
        assert set(p) == set(nodes)
        assert len(set(p.values())) == len(sizes)
        for k in set(p.values()):
            members = [c for c in nodes if p[c] == k]
            sub = [(a, b) for a in members for b in nb[a] if p[b] == k and a < b]
            assert len(set(R.components(members, sub).values())) == 1
    assert R.size_drift(parts, sizes) < 3


def test_components_and_connect_join_islands():
    nodes, nb, pairs = grid(4, 3)
    nodes = nodes + ["island"]
    xy = pd.DataFrame({"x": [int(c.split("-")[0]) if c != "island" else 9 for c in nodes],
                       "y": [int(c.split("-")[1]) if c != "island" else 0 for c in nodes]}, index=nodes)
    assert len(set(R.components(nodes, pairs).values())) == 2
    links = R.connect(nodes, pairs, xy)
    assert len(set(R.components(nodes, links).values())) == 1
    assert [p for p in links if p not in pairs] == [("3-0", "island")]


def test_border_premium_recovers_a_planted_border():
    nodes, nb, pairs = grid(20, 10)
    rng = np.random.default_rng(0)
    side = {c: int(c.split("-")[0]) >= 10 for c in nodes}
    y = {c: rng.normal(0, 1) + (3.0 if side[c] else 0.0) for c in nodes}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    diff = np.abs(np.array([y[i] for i in a]) - np.array([y[j] for j in b]))
    cross = np.array([side[i] != side[j] for i, j in pairs])
    logd = np.zeros(len(pairs))
    logd[::2] = 0.1  # constant distance apart from a small wiggle so the design is full rank
    prem, se, base = R.border_premium(diff, cross, logd, a, b)
    assert prem > 4 * se
    assert prem == pytest.approx(R.border_premium(diff, cross, logd, a, b, se=False))
    assert 1.5 < prem < 3.5


def test_dyadic_se_matches_ols_scale_without_shared_nodes():
    rng = np.random.default_rng(1)
    n = 400
    a = np.array([f"a{i}" for i in range(n)])
    b = np.array([f"b{i}" for i in range(n)])  # no two pairs share a node
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = X @ np.array([1.0, 2.0]) + rng.normal(size=n)
    beta, se = R.dyadic_ols(y, X, a, b)
    e = y - X @ beta
    XtX = np.linalg.inv(X.T @ X)
    hc0 = np.sqrt(np.diag(XtX @ (X.T * e ** 2) @ X @ XtX))
    assert np.allclose(se, hc0)


@needs_data
def test_workbook_regions_all_have_polygons_and_neighbours():
    df = R.load_panel()
    codes = pd.Series(sorted(df.code.unique()))
    assert len(codes) == 173
    g = R.load_map(codes)
    assert len(g) == 173
    pairs = R.land_pairs(g)
    assert len(pairs) == 386
    assert df[df.year == 1900].ind.notna().sum() == 171
    assert R.benchmark_years(df) == [1900, 1910, 1925, 1938, 1950, 1960, 1970, 1980, 1990, 2000,
                                     2010, 2015, 2021]


@needs_data
def test_price_bases_differ_by_one_national_factor():
    """The 1990-dollar and 2011-PPP GDP series give the same regional ratios inside a country and
    year, so the choice of base can move only national levels (and therefore border gaps)."""
    df = R.load_panel()
    r = (df.lgdppc - df.lgdppc90).dropna()
    spread = r.groupby([df.country, df.year]).agg(lambda x: x.max() - x.min())
    assert spread.max() < 1e-9
    assert r.groupby(df.country).mean().std() > 0.05
