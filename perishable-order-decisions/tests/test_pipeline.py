from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evaluate import CAL_END, FIT_END, comparisons, decomposition, h4, run, shares, verdicts, windows
from feed import decide as K
from feed import forecast as F
from feed import recover as R
from feed.data import build, read


def synthetic(n: int = 40, T: int = 97, test_days: int = 7, seed: int = 0, out_value: int = 1) -> list[dict]:
    """Two splits in the dataset's column layout, with out-of-stock afternoons on some days."""
    rng = np.random.default_rng(seed)
    dates = np.array([str(np.datetime64("2024-03-28") + np.timedelta64(t, "D")) for t in range(T)])
    profile = np.zeros(24)
    profile[7:21] = np.hanning(16)[1:-1]
    profile /= profile.sum()
    parts = []
    for split, days in (("train", range(T - test_days)), ("eval", range(T - test_days, T))):
        rows = {
            k: []
            for k in (
                "city_id",
                "store_id",
                "management_group_id",
                "first_category_id",
                "second_category_id",
                "third_category_id",
                "product_id",
                "dt",
                "sale_amount",
                "stock_hour6_22_cnt",
                "discount",
                "holiday_flag",
                "activity_flag",
                "hours_sale",
                "hours_stock_status",
            )
        }
        for i in range(n):
            level = 1.0 + (i % 5)
            for t in days:
                mean = level * (1.3 if t % 7 in (5, 6) else 1.0)
                hourly = rng.poisson(mean * 10 * profile) / 10.0
                out = np.zeros(24, dtype=int)
                if rng.random() < 0.2:
                    out[15:] = 1
                    hourly[15:] = 0.0
                status = out if out_value == 1 else 1 - out
                rows["city_id"].append(i % 3)
                rows["store_id"].append(i // 4)
                rows["management_group_id"].append(0)
                rows["first_category_id"].append(i % 2)
                rows["second_category_id"].append(i % 2)
                rows["third_category_id"].append(i % 4)
                rows["product_id"].append(i % 4)
                rows["dt"].append(dates[t])
                rows["sale_amount"].append(float(hourly.sum()))
                rows["stock_hour6_22_cnt"].append(int(out[6:23].sum()))
                rows["discount"].append(1.0)
                rows["holiday_flag"].append(0)
                rows["activity_flag"].append(0)
                rows["hours_sale"].append(hourly.tolist())
                rows["hours_stock_status"].append(status.tolist())
        parts.append((split, rows))
    return parts


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    parts = []
    for split, rows in synthetic():
        pq.write_table(pa.table(rows), d / f"{split}.parquet")
        parts.append(read(d / f"{split}.parquet", split))
    p, lines = build(parts)
    return p


def test_build_shapes_and_order(panel):
    assert panel.sales.shape == (40, 97)
    assert panel.hours_sale.shape == (40, 97, 24)
    assert list(panel.dates) == sorted(panel.dates)
    assert panel.test.sum() == 7 and panel.test[-7:].all()
    assert np.allclose(panel.hours_sale.sum(axis=2), panel.sales, atol=1e-5)


def test_stock_coding_found(panel):
    c = R.stock_coding(panel)
    assert (c.out_value, c.first, c.last) == (1, 6, 22) and c.agreement == 1.0


def test_recovery_restores_cut_afternoon(panel):
    fit, _, _ = windows(panel.days)
    c = R.stock_coding(panel)
    out = R.out_hours(panel, c)
    prof = R.profiles(panel, fit)
    rec = R.recover(panel, out, prof)
    clean = ~rec.stockout
    assert np.array_equal(rec.demand[clean], panel.sales[clean])
    assert (rec.demand >= panel.sales).all()
    cut = rec.stockout & ~rec.fallback
    # the cut afternoon holds about 40% of the profile, so recovered demand is well above sales on those days
    assert rec.demand[cut].sum() > 1.4 * panel.sales[cut].sum()


def test_forecasts_align_and_never_leak(panel):
    y = panel.sales.copy()
    f = F.forecasts(panel, y, list(range(28, FIT_END)), list(range(FIT_END, panel.days)), {"n_estimators": 5})
    t = 90
    assert np.allclose(f["naive7"][:, t], y[:, t - 7])
    assert np.allclose(f["ma4w"][:, t], y[:, [t - 7, t - 14, t - 21, t - 28]].mean(axis=1))
    y2 = y.copy()
    y2[:, t:] = 1e6  # changing day t and later must not change the forecast for day t
    f2 = F.forecasts(panel, y2, list(range(28, FIT_END)), [t], {"n_estimators": 5})
    for a in F.ALGOS:
        assert np.allclose(f[a][:, t], f2[a][:, t])


def test_order_rules_and_cost():
    f = np.array([[10.0, 10.0]])
    cal = K.Calibration(
        level=np.array([10.0]),
        scale=np.array([2.0]),
        by_level=np.linspace(-0.5, 0.5, 101),
        by_scale=np.linspace(-2, 2, 101),
    )
    assert np.array_equal(K.order(f, "point", 0.9, cal), f)
    assert np.allclose(K.order(f, "pooled", 0.9, cal), 10 + 10 * 0.4)
    assert np.allclose(K.order(f, "series", 0.5, cal), 10.0)
    assert np.allclose(K.cost(np.array([5.0]), np.array([8.0]), 0.9), 0.9 * 3)
    assert np.allclose(K.cost(np.array([8.0]), np.array([5.0]), 0.9), 0.1 * 3)


def test_shares_add_up_and_separate_effects():
    algo = np.array([0.0, 1.0, 2.0, 3.0])[:, None]
    layer = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[None, :]
    s = shares(algo + layer)
    assert np.allclose(s, [0.0, 1.0, 0.0])
    s = shares(np.zeros((4, 1)) + np.arange(6.0)[None, :])
    assert np.allclose(s, [1.0, 0.0, 0.0])


def test_end_to_end(panel):
    res = run(panel, {"n_estimators": 10}, boot=50)
    assert len(res["rows"]) == 72
    comp = comparisons(res)
    assert {x["hypothesis"] for x in comp} == {"H1", "H2", "H3"}
    assert len([x for x in comp if x["convention"] == "all"]) == 4 + 2 + 24
    v = verdicts(comp)
    assert v["H3"]["n"] == 24
    dec = decomposition(res, 0.9)
    assert abs(dec["algorithm"] + dec["layer"] + dec["remainder"] - 1) < 1e-9
    assert set(h4(res["rows"])["by_cost"]) == set(F.ALGOS)
    assert CAL_END - FIT_END == 14
