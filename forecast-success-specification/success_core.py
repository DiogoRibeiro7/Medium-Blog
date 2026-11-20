"""Core of the success-specification test on M5 (FOODS, 10 stores).

Data loading, the nine candidate forecasters, the LightGBM feature builders and the
metrics of the five specifications. Day numbers follow M5: day 1 is 2011-01-29 (a
Saturday); index d - 1 in the arrays holds day d. An origin is a Friday; data up to and
including the origin are known.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_DAYS = 1941
FIRST_ORIGIN = 1470
WARMUP_ORIGINS = list(range(1470, 1562, 7))  # 14 Fridays, d_1470 .. d_1561
TEST_ORIGINS = list(range(1568, 1926, 7))  # 52 Fridays, d_1568 .. d_1925
ALL_ORIGINS = WARMUP_ORIGINS + TEST_ORIGINS
REFITS = [1470, 1568, 1659, 1750, 1841]  # first origin of the warm-up block and of each test quarter
S0_ORIGINS = list(range(1575, 1912, 28))  # 13 origins four weeks apart, d_1575 .. d_1911
DAILY_MODELS = ["snaive", "ma28", "dow4", "ses", "lgb_l2", "lgb_l1", "lgb_tweedie"]
WEEKLY_MODELS = ["lgbw_tweedie", "lgbw_q80"]
ALL_MODELS = DAILY_MODELS + WEEKLY_MODELS
INCUMBENT = "ma28"
SES_GRID = (0.02, 0.05, 0.1, 0.2, 0.3)
EVENT_TYPES = ["Sporting", "Cultural", "National", "Religious"]

LGB_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 7,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 2,
    "verbose": -1,
}
LGB_ROUNDS = 400
OBJECTIVES = {
    "lgb_l2": {"objective": "l2"},
    "lgb_l1": {"objective": "l1"},
    "lgb_tweedie": {"objective": "tweedie", "tweedie_variance_power": 1.1},
    "lgbw_tweedie": {"objective": "tweedie", "tweedie_variance_power": 1.1},
    "lgbw_q80": {"objective": "quantile", "alpha": 0.8},
}


def refit_for(origin: int) -> int:
    """The refit origin whose model serves this origin."""
    return max(r for r in REFITS if r <= origin)


# --------------------------------------------------------------------------- data


@dataclass
class Panel:
    sales: np.ndarray  # (n, N_DAYS) float32
    price: np.ndarray  # (n, N_DAYS) float32, NaN when not on sale
    item: np.ndarray  # (n,) int item code
    store: np.ndarray  # (n,) int store code
    dept: np.ndarray  # (n,) int dept code
    state: np.ndarray  # (n,) int state code (CA, TX, WI)
    ids: np.ndarray  # (n,) series id strings
    wday: np.ndarray  # (N_DAYS,) 1..7, Saturday = 1
    month: np.ndarray  # (N_DAYS,)
    event: np.ndarray  # (N_DAYS,) 0 none, 1..4 event type of event_name_1
    snap: np.ndarray  # (3, N_DAYS) SNAP flag by state
    dates: np.ndarray  # (N_DAYS,) datetime64[D]

    @property
    def n(self) -> int:
        return self.sales.shape[0]


def load_panel(raw: Path, category: str = "FOODS") -> Panel:
    """Read the M5 files, keep one category and the series sold in the year up to the first origin."""
    sales = pd.read_csv(raw / "sales_train_evaluation.csv")
    sales = sales[sales["cat_id"] == category].reset_index(drop=True)
    day_cols = [f"d_{d}" for d in range(1, N_DAYS + 1)]
    y = sales[day_cols].to_numpy(dtype=np.float32)
    keep = y[:, FIRST_ORIGIN - 364 : FIRST_ORIGIN].sum(axis=1) > 0
    sales, y = sales[keep].reset_index(drop=True), y[keep]

    cal = pd.read_csv(raw / "calendar.csv")
    if len(cal) < N_DAYS:
        raise ValueError("calendar shorter than the sales file")
    cal = cal.iloc[:N_DAYS]
    week_to_days: dict[int, list[int]] = {}
    for idx, wk in enumerate(cal["wm_yr_wk"].to_numpy()):
        week_to_days.setdefault(int(wk), []).append(idx)

    prices = pd.read_csv(raw / "sell_prices.csv")
    prices = prices[prices["item_id"].str.startswith(category + "_")]
    key = sales["item_id"] + "|" + sales["store_id"]
    row_of = pd.Series(np.arange(len(sales)), index=key)
    pkey = prices["item_id"] + "|" + prices["store_id"]
    prices = prices[pkey.isin(row_of.index)]
    pkey = prices["item_id"] + "|" + prices["store_id"]
    rows = row_of.loc[pkey].to_numpy()
    price = np.full(y.shape, np.nan, dtype=np.float32)
    for wk, grp_rows, vals in _group_by_week(prices["wm_yr_wk"].to_numpy(), rows, prices["sell_price"].to_numpy()):
        days = week_to_days.get(int(wk))
        if days:
            price[np.ix_(grp_rows, days)] = vals[:, None]

    ev = cal["event_type_1"].fillna("").to_numpy()
    event = np.array([EVENT_TYPES.index(e) + 1 if e in EVENT_TYPES else 0 for e in ev], dtype=np.int8)
    snap = cal[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(dtype=np.int8).T
    states = ["CA", "TX", "WI"]
    return Panel(
        sales=y,
        price=price,
        item=pd.factorize(sales["item_id"])[0],
        store=pd.factorize(sales["store_id"], sort=True)[0],
        dept=pd.factorize(sales["dept_id"], sort=True)[0],
        state=sales["state_id"].map({s: i for i, s in enumerate(states)}).to_numpy(),
        ids=(sales["item_id"] + "_" + sales["store_id"]).to_numpy(),
        wday=cal["wday"].to_numpy(dtype=np.int8),
        month=cal["month"].to_numpy(dtype=np.int8),
        event=event,
        snap=snap,
        dates=pd.to_datetime(cal["date"]).to_numpy().astype("datetime64[D]"),
    )


def _group_by_week(weeks, rows, vals):
    order = np.argsort(weeks, kind="stable")
    weeks, rows, vals = weeks[order], rows[order], vals[order]
    cuts = np.flatnonzero(np.diff(weeks)) + 1
    for w, r, v in zip(np.split(weeks, cuts), np.split(rows, cuts), np.split(vals, cuts)):
        yield w[0], r, v.astype(np.float32)


# --------------------------------------------------------------------------- helpers


def col(day: int) -> int:
    """Array column of M5 day `day`."""
    return day - 1


def window_mean(cs: np.ndarray, end_day: int, length: int) -> np.ndarray:
    """Mean over days end_day-length+1 .. end_day from a cumulative sum with a leading zero column."""
    return (cs[:, end_day] - cs[:, end_day - length]) / length


def cumsum0(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], a.shape[1] + 1), dtype=np.float64)
    np.cumsum(a, axis=1, out=out[:, 1:])
    return out


def week_totals(y: np.ndarray, origin: int) -> tuple[np.ndarray, np.ndarray]:
    """Sales in week A (o+1..o+7) and week B (o+8..o+14)."""
    a = y[:, col(origin + 1) : col(origin + 7) + 1].sum(axis=1)
    b = y[:, col(origin + 8) : col(origin + 14) + 1].sum(axis=1)
    return a, b


# --------------------------------------------------------------------------- simple forecasters


def snaive(y: np.ndarray, origin: int, horizons: int = 28) -> np.ndarray:
    h = np.arange(1, horizons + 1)
    src = origin - 6 + (h - 1) % 7
    return y[:, src - 1]


def ma28(y: np.ndarray, origin: int, horizons: int = 28) -> np.ndarray:
    m = y[:, col(origin - 27) : col(origin) + 1].mean(axis=1)
    return np.repeat(m[:, None], horizons, axis=1)


def dow4(y: np.ndarray, origin: int, horizons: int = 28) -> np.ndarray:
    h = np.arange(1, horizons + 1)
    src = origin - 6 + (h - 1) % 7
    return np.mean([y[:, src - 1 - 7 * j] for j in range(4)], axis=0)


def ses_run(y: np.ndarray, origins: list[int], refits: list[int], alphas=SES_GRID,
            start: int = 700) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Simple exponential smoothing for every alpha of the grid, from day `start` on.

    Returns the levels after observing each origin, levels[o] of shape (len(alphas), n), and
    for each refit the one-step squared error summed over the 364 days up to it,
    sse[r] of shape (len(alphas), n). Initial level: mean of the 28 days before `start`.
    """
    n = y.shape[0]
    init = y[:, col(start - 28) : col(start)].mean(axis=1).astype(np.float64)
    lev = np.repeat(init[None, :], len(alphas), axis=0)
    a = np.asarray(alphas, dtype=np.float64)[:, None]
    run_sq = np.zeros((len(alphas), n))
    cum_sq = {}
    want_cum = {r - 364 for r in refits} | set(refits)
    levels = {}
    origin_set = set(origins)
    for d in range(start, N_DAYS + 1):
        obs = y[:, col(d)].astype(np.float64)[None, :]
        run_sq += (obs - lev) ** 2
        lev = lev + a * (obs - lev)
        if d in want_cum:
            cum_sq[d] = run_sq.copy()
        if d in origin_set:
            levels[d] = lev.astype(np.float32)
    sse = {r: cum_sq[r] - cum_sq[r - 364] for r in refits}
    return levels, sse


def ses_choose(sse: np.ndarray) -> np.ndarray:
    """Index of the alpha with the smallest one-step squared error, per series."""
    return np.argmin(sse, axis=0)


# --------------------------------------------------------------------------- features


@dataclass
class HistoryStats:
    """Per-series statistics at one origin, shared by the daily and weekly feature builders."""

    m7: np.ndarray
    m14: np.ndarray
    m28: np.ndarray
    m56: np.ndarray
    m112: np.ndarray
    sd28: np.ndarray
    zero28: np.ndarray
    since: np.ndarray
    item28: np.ndarray
    price_o: np.ndarray
    price_52w: np.ndarray


def history_stats(p: Panel, cs: np.ndarray, origin: int) -> HistoryStats:
    y = p.sales
    last28 = y[:, col(origin - 27) : col(origin) + 1]
    sold = y[:, col(origin - 111) : col(origin) + 1] > 0
    any_sold = sold.any(axis=1)
    last_idx = 111 - np.argmax(sold[:, ::-1], axis=1)
    since = np.where(any_sold, 111 - last_idx, 112).astype(np.float32)
    m28 = window_mean(cs, origin, 28)
    item28 = pd.Series(m28).groupby(p.item).transform("mean").to_numpy()
    with np.errstate(all="ignore"):
        price_52w = np.nanmean(p.price[:, col(origin - 363) : col(origin) + 1], axis=1)
    return HistoryStats(
        m7=window_mean(cs, origin, 7),
        m14=window_mean(cs, origin, 14),
        m28=m28,
        m56=window_mean(cs, origin, 56),
        m112=window_mean(cs, origin, 112),
        sd28=last28.std(axis=1),
        zero28=(last28 == 0).mean(axis=1),
        since=since,
        item28=item28,
        price_o=p.price[:, col(origin)],
        price_52w=price_52w,
    )


DAILY_FEATURES = [
    "h", "wday", "month", "event", "snap", "price", "price_rel52", "price_rel_o",
    "m7", "m14", "m28", "m56", "m112", "sd28", "zero28", "since",
    "dow4", "dow8", "ly", "item28", "store", "dept",
]
WEEKLY_FEATURES = [
    "k", "month", "events", "snaps", "price", "price_rel52", "price_rel_o",
    "m7", "m14", "m28", "m56", "m112", "sd28", "zero28", "since",
    "ly", "item28", "store", "dept",
]
CATEGORICAL = ["store", "dept", "event"]


def daily_features(p: Panel, cs: np.ndarray, origin: int, rows: np.ndarray | None = None,
                   horizons: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix for days o+1..o+horizons of the given series; rows ordered series-major.

    Returns (X, target_day) with target_day the M5 day of each row.
    """
    if rows is None:
        rows = np.arange(p.n)
    hs = history_stats(p, cs, origin)
    y = p.sales
    nr = len(rows)
    h = np.tile(np.arange(1, horizons + 1), nr)
    r = np.repeat(rows, horizons)
    t = origin + h
    src = origin - 6 + (h - 1) % 7  # last observed same weekday
    dow4_ = np.mean([y[r, src - 1 - 7 * j] for j in range(4)], axis=0)
    dow8_ = np.mean([y[r, src - 1 - 7 * j] for j in range(8)], axis=0)
    ly = (cs[r, t - 364 + 3] - cs[r, t - 364 - 4]) / 7.0
    price_t = p.price[r, t - 1]
    with np.errstate(all="ignore"):
        rel52 = price_t / hs.price_52w[r]
        rel_o = price_t / hs.price_o[r]
    snap_t = p.snap[p.state[r], t - 1]
    X = np.column_stack([
        h, p.wday[t - 1], p.month[t - 1], p.event[t - 1], snap_t, price_t, rel52, rel_o,
        hs.m7[r], hs.m14[r], hs.m28[r], hs.m56[r], hs.m112[r], hs.sd28[r], hs.zero28[r], hs.since[r],
        dow4_, dow8_, ly, hs.item28[r], p.store[r], p.dept[r],
    ]).astype(np.float32)
    return X, t


def weekly_features(p: Panel, cs: np.ndarray, origin: int, rows: np.ndarray | None = None) -> np.ndarray:
    """Feature matrix for weeks A (k=1) and B (k=2) of the given series; rows ordered series-major."""
    if rows is None:
        rows = np.arange(p.n)
    hs = history_stats(p, cs, origin)
    nr = len(rows)
    k = np.tile(np.array([1, 2]), nr)
    r = np.repeat(rows, 2)
    first = origin + 7 * (k - 1) + 1
    last = first + 6
    days = first[:, None] + np.arange(7)[None, :]
    events = (p.event[days - 1] > 0).sum(axis=1)
    snaps = p.snap[p.state[r][:, None], days - 1].sum(axis=1)
    with np.errstate(all="ignore"):
        price_w = np.nanmean(p.price[r[:, None], days - 1], axis=1)
        rel52 = price_w / hs.price_52w[r]
        rel_o = price_w / hs.price_o[r]
    ly = cs[r, last - 364] - cs[r, first - 1 - 364]
    X = np.column_stack([
        k, p.month[first - 1], events, snaps, price_w, rel52, rel_o,
        hs.m7[r], hs.m14[r], hs.m28[r], hs.m56[r], hs.m112[r], hs.sd28[r], hs.zero28[r], hs.since[r],
        ly, hs.item28[r], p.store[r], p.dept[r],
    ]).astype(np.float32)
    return X


def daily_training_set(p: Panel, cs: np.ndarray, refit: int, frac: float, rng: np.random.Generator):
    """Rows for the daily models: 52 weekly origins with targets ending by the refit, sampled pairs."""
    Xs, ys = [], []
    for j in range(52):
        o = refit - 28 - 7 * j
        rows = np.flatnonzero(rng.random(p.n) < frac)
        X, t = daily_features(p, cs, o, rows)
        Xs.append(X)
        ys.append(p.sales[np.repeat(rows, 28), t - 1])
    return np.vstack(Xs), np.concatenate(ys)


def weekly_training_set(p: Panel, cs: np.ndarray, refit: int, frac: float, rng: np.random.Generator):
    Xs, ys = [], []
    for j in range(52):
        o = refit - 14 - 7 * j
        rows = np.flatnonzero(rng.random(p.n) < frac)
        Xs.append(weekly_features(p, cs, o, rows))
        a, b = week_totals(p.sales[rows], o)
        ys.append(np.column_stack([a, b]).ravel())
    return np.vstack(Xs), np.concatenate(ys)


# --------------------------------------------------------------------------- metrics


def wape(err_abs: np.ndarray, actual: np.ndarray) -> float:
    return float(err_abs.sum() / actual.sum())


def pinball(order: np.ndarray, actual: np.ndarray, tau: float) -> np.ndarray:
    return tau * np.maximum(actual - order, 0) + (1 - tau) * np.maximum(order - actual, 0)


def bad_weeks(forecast: np.ndarray, actual: np.ndarray, units: float = 5, share: float = 0.5) -> np.ndarray:
    """A week whose absolute error exceeds the larger of `units` and `share` of sales.

    Forecasts are rounded to 1e-3 units first: float32 sums put errors that equal the
    threshold exactly (common, since sales are integers) on either side of it at random.
    """
    err = np.abs(np.round(forecast.astype(np.float64), 3) - actual)
    return err > np.maximum(units, share * actual)


def residual_quantile_orders(fc_b: np.ndarray, act_b: np.ndarray, origins: list[int], tau: float,
                             lookback: int = 13) -> dict[int, np.ndarray]:
    """Order = forecast + tau-quantile of the series' own week B errors at the `lookback` latest
    origins whose week B had ended by the origin (o-14 back to o-14-7*(lookback-1))."""
    pos = {o: i for i, o in enumerate(origins)}
    err = act_b - fc_b
    out = {}
    for o in origins:
        prev = [o - 14 - 7 * j for j in range(lookback)]
        if not all(q in pos for q in prev):
            continue
        e = err[:, [pos[q] for q in prev]]
        out[o] = np.maximum(fc_b[:, pos[o]] + np.quantile(e, tau, axis=1), 0)
    return out


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return 2 * s / (n * (n - 1))
