import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import Params, State, _step_bargaining, _step_transparent, draw_population, simulate, with_  # noqa: E402

SMALL = Params(n_workers=3000, n_firms=60, burn_in=120, n_measure=3, measure_every=6)


def run_state(prm: Params, months: int = 150) -> tuple[State, np.ndarray, np.ndarray, np.ndarray]:
    e, p, band = draw_population(prm)
    b = prm.b0 * e
    rng = np.random.default_rng(prm.seed)
    st = State(firm=-np.ones(prm.n_workers, dtype=int), wage=np.zeros(prm.n_workers), threat=b.copy())
    if prm.transparency:
        st.cell_wage = np.full(prm.n_firms * prm.n_bands, np.nan)
        st.cell_count = np.zeros(prm.n_firms * prm.n_bands, dtype=int)
    for _ in range(months):
        if prm.transparency:
            _step_transparent(prm, st, e, p, b, band, rng)
        else:
            _step_bargaining(prm, st, e, p, b, rng)
    return st, e, p, band


def test_wages_respect_floor_and_output():
    prm = with_(SMALL, m=0.6)
    st, e, p, _ = run_state(prm)
    emp = st.firm >= 0
    y = e[emp] * p[st.firm[emp]]
    assert np.all(st.wage[emp] >= 0.6 - 1e-12)
    assert np.all(st.wage[emp] <= y + 1e-12)


def test_no_rival_offers_gives_nash_wage():
    prm = with_(SMALL, lambda1=0.0, m=0.0, beta=0.3)
    st, e, p, _ = run_state(prm)
    emp = st.firm >= 0
    y = e[emp] * p[st.firm[emp]]
    b = prm.b0 * e[emp]
    np.testing.assert_allclose(st.wage[emp], b + 0.3 * (y - b))


def test_full_bargaining_power_pays_output():
    r = simulate(with_(SMALL, beta=1.0, m=0.0))
    assert abs(r["labour_share"] - 1.0) < 1e-9


def test_equal_pay_within_cells():
    prm = with_(SMALL, transparency=True)
    st, _, _, band = run_state(prm)
    emp = st.firm >= 0
    cells = st.firm[emp] * prm.n_bands + band[emp]
    w = st.wage[emp]
    for c in np.unique(cells):
        assert np.ptp(w[cells == c]) == 0.0


def test_higher_floor_more_workers_at_floor():
    lo = simulate(with_(SMALL, m=0.4))
    hi = simulate(with_(SMALL, m=0.8))
    assert hi["share_at_floor"] > lo["share_at_floor"]
    assert hi["p50_p10"] < lo["p50_p10"]
