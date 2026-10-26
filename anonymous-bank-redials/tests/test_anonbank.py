"""Checks of the pipeline on small cases with a known answer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from anonbank.data import load_calls
from anonbank.ids import callback_curves, detection_weights, \
    flag_identified_redials, queue_share, with_neighbours
from anonbank.indirect import Binding
from anonbank.sim import Centre, Simulator
from anonbank.stats import Statistics, calendar_expectation, surplus
from anonbank.table import N_INTERVALS, halfhour_table

HEADER = ("vru+line\tcall_id\tcustomer_id\tpriority\ttype\tdate\tvru_entry\t"
          "vru_exit\tvru_time\tq_start\tq_exit\tq_time\toutcome\tser_start\t"
          "ser_exit\tser_time\tserver")


def write_month(tmp_path, rows):
    text = "\r\n".join([HEADER] + ["\t".join(r) for r in rows]) + "\r\n"
    (tmp_path / "January1999.txt").write_text(text)
    return tmp_path


def row(cid, date, entry, vru, q, outcome, ser=0, server="A"):
    def clock(s):
        s = int(s)
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
    h, m, s = (int(x) for x in entry.split(":"))
    t0 = h * 3600 + m * 60 + s
    t1 = t0 + vru
    qs, qe = (t1, t1 + q) if q > 0 else (0, 0)
    ss = t1 + q if outcome == "AGENT" else 0
    return ["AA0101", "1", cid, "2", "PS", date, clock(t0), clock(t1),
            str(vru), clock(qs), clock(qe), str(q), outcome, clock(ss),
            clock(ss + ser if ss else 0), str(ser),
            server if outcome == "AGENT" else "NO_SERVER"]


def test_parsing_flags_and_malformed_rows(tmp_path):
    rows = [row("11", "990103", "9:00:00", 5, 30, "HANG"),
            row("11", "990103", "9:05:00", 5, 10, "AGENT", 120),
            row("0", "990103", "9:10:00", 5, 0, "AGENT", 60)]
    bad = row("12", "990103", "9:20:00", 5, 0, "AGENT", 60)
    bad[8] = "0:00:05"          # a clock where a number belongs
    calls, info = load_calls(write_month(tmp_path, rows + [bad]))
    assert info["malformed_rows"] == 1 and len(calls) == 3
    assert calls["abandoned"].tolist() == [True, False, False]
    assert calls["queued"].tolist() == [True, True, False]
    assert calls["has_id"].tolist() == [True, True, False]


def test_callback_after_abandonment_is_found(tmp_path):
    rows = [row("11", "990103", "9:00:00", 5, 30, "HANG"),
            row("11", "990103", "9:05:00", 5, 10, "AGENT", 120),
            row("22", "990103", "9:00:00", 5, 10, "AGENT", 100),
            row("22", "990103", "11:00:00", 5, 10, "AGENT", 100)]
    calls, _ = load_calls(write_month(tmp_path, rows))
    ids = with_neighbours(calls)
    c = callback_curves(ids, np.array([10, 180]), max_volume=None)
    get = {(r.window_min, r.group): r.again for r in c.itertuples()}
    # caller 11 hung up at 9:00:35 and called at 9:05:00
    assert get[(10, "abandoned")] == 1.0
    # caller 22's served call ended 9:01:55; the next came after 2 hours
    assert get[(10, "served")] == 0.0
    flag = flag_identified_redials(calls, 60.0)
    assert flag.tolist() == [False, True, False, False]


def test_halfhour_table_counts(tmp_path):
    rows = [row("11", "990103", "9:00:00", 5, 30, "HANG"),
            row("11", "990103", "9:05:00", 5, 10, "AGENT", 120, "A"),
            row("0", "990103", "9:10:00", 5, 0, "AGENT", 60, "B"),
            row("0", "990103", "9:40:00", 5, 40, "AGENT", 60, "A")]
    calls, _ = load_calls(write_month(tmp_path, rows))
    tab = halfhour_table(calls, flag_identified_redials(calls))
    nine = tab[tab["half"] == 18].iloc[0]
    assert nine["offered"] == 3 and nine["abandoned"] == 1
    assert nine["answered_20s"] == 2 and nine["agents"] == 2
    assert nine["id_redials"] == 1
    assert tab[tab["half"] == 19].iloc[0]["offered"] == 1
    assert len(tab) == N_INTERVALS


def _planted(rho, sigma, effect, seed, days=400, n=34):
    rng = np.random.default_rng(seed)
    z = np.zeros((days, n))
    z[:, 0] = rng.normal(0, sigma, days)
    for t in range(1, n):
        z[:, t] = rho * z[:, t - 1] + rng.normal(
            0, sigma * np.sqrt(1 - rho ** 2), days)
    base = 50 * np.exp(z)
    ab = rng.poisson(0.15 * base).astype(float)
    offered = rng.poisson(base).astype(float)
    offered[:, 1:] += rng.poisson(effect * ab[:, :-1])
    return Statistics(offered, ab, np.zeros(days, int))


def test_slopes_recover_a_planted_effect_without_clustering():
    st = _planted(0.0, 0.0, 0.3, seed=0)
    assert abs(st.lag(1) - 0.3) < 0.08
    assert abs(st.controlled() - 0.3) < 0.08
    assert abs(st.lag(-1)) < 0.08


def test_clustering_alone_raises_lag_and_lead_alike():
    st = _planted(0.7, 0.3, 0.0, seed=1)
    assert st.lag(1) > 0.3 and st.lag(-1) > 0.3
    assert abs(st.lag_minus_lead()) < 0.08
    assert st.acf1() > 0.3 and st.dispersion() > 2


def test_surplus_sums_to_zero_each_day():
    rng = np.random.default_rng(1)
    off = rng.poisson(30, (5, 10)).astype(float)
    r = surplus(off, calendar_expectation(off, np.zeros(5, int)))
    assert np.allclose(r.sum(axis=1), 0)


def test_simulator_adds_redials():
    days, n = 30, N_INTERVALS
    off = np.full((days, n), 40.0)
    ab = np.full((days, n), 8.0)
    ag = np.full((days, n), 6)
    dow = np.zeros(days, int)
    c = Centre.from_table(off, ab, ag, dow, wait_total=8.0 * days * n * 200,
                          service_mean=250.0)
    o0, a0, _ = Simulator(c, p=0.0).days(np.random.default_rng(2), days)
    o1, a1, _ = Simulator(c, p=0.8).days(np.random.default_rng(2), days)
    assert abs(o0.mean() - 40) < 3
    fresh = 40 * (1 - 0.8 * 0.2)
    assert o1.mean() > fresh + 2      # redials are counted as offered


def test_binding_inversion():
    b = Binding("lag", np.array([0.0, 0.2, 0.4, 0.6]),
                np.array([0.0, 0.1, 0.25, 0.45]))
    assert abs(float(b.invert(0.1)) - 0.2) < 0.03
    assert float(b.invert(-1)) == 0.0 and float(b.invert(9)) == 0.6


def test_detection_weights_follow_the_queue_share(tmp_path):
    # 9:00-9:30: two offered calls, one queued; 10:00-10:30: one, queued
    rows = [row("11", "990103", "9:00:00", 5, 30, "HANG"),
            row("0", "990103", "9:10:00", 5, 0, "AGENT", 60),
            row("11", "990103", "10:05:00", 5, 10, "AGENT", 120)]
    calls, _ = load_calls(write_month(tmp_path, rows))
    share = queue_share(calls)
    assert share.loc[(pd.Timestamp("1999-01-03"), 18)] == 0.5
    ids = with_neighbours(calls)
    assert np.allclose(detection_weights(ids, share), [2.0, 1.0])
    c = callback_curves(ids, np.array([90]), None, share)
    ab = c[c.group == "abandoned"].iloc[0]
    assert ab.again == 1.0 and ab.again_weighted == 1.0
