"""Checks on the real map data, run only when fetch_data.sh has been run (as it is in CI)."""
import os
from pathlib import Path

import pytest

DATA = Path(os.environ.get("CABLE_DATA", Path(__file__).resolve().parents[1] / "data"))
pytestmark = pytest.mark.skipif(not (DATA / "ixpavi_subsea-siting").exists(),
                                reason="map data not fetched; run ./fetch_data.sh")


@pytest.fixture(scope="module")
def net():
    from cableresilience.data import load
    return load(DATA)


def test_snapshot_size(net):
    assert len(net.cables) == 606                  # systems in service in the 2026-09-24 snapshot
    assert len(net.landing_points) > 1000


def test_madeira_and_sao_miguel(net):
    from cableresilience.cuts import build_graph, edge_cut_lambda, external_cables, min_incidents
    G = build_graph(net)
    lp = {v["name"]: k for k, v in net.landing_points.items()}
    expected = {"Funchal, Portugal": (3, 3, 1), "Ponta Delgada, Portugal": (3, 2, 1)}
    for name, (k, lam, n10) in expected.items():
        u = net.landing_points[lp[name]]["unit"]
        assert len(external_cables(net, u)) == k
        assert edge_cut_lambda(net, u, G) == lam
        assert min_incidents(net, u, 10.0, 100.0, G)[0] == n10
