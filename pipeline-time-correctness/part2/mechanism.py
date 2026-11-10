"""Functions behind Part 2: five mechanisms, each simulated where the right answer is known.

M1. Right truncation: what the latest periods are missing, for a given delay distribution.
M2. Waiting against estimating: watermark slack, allowed lateness, and a chain-ladder nowcast,
    with delays that stay the same and delays that change.
M3. Loads: a high-water mark on event time against one on arrival time; corrections under an
    insert-only load, an upsert and an append-only log of versions (SQL, DuckDB).
M4. Features: a model trained on final values and served on values as known at the time; a
    dimension joined in its current version instead of the version valid at the time.
M5. Backfills: rerunning a job that appends, rerunning one that replaces, and what the
    restatement does to values already sent and to comparisons across days.

Every random number comes from numpy's default generator with a fixed seed, and money is kept
in integer cents, so the printed numbers are the same on every machine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

D_MAX = 30          # delays are counted in whole days; anything later than 30 days counts at 30
DAY = 86_400        # seconds
HOUR = 3_600

_erf = np.vectorize(math.erf, otypes=[float])


# ---------------------------------------------------------------------------
# Delay distributions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Delay:
    """A reporting delay X (in days) with a log-normal distribution.

    An event on day t with delay X is reported on day t + floor(X). "Known at the end of day T"
    therefore holds the events of day t whose floor(X) <= T - t.
    """

    name: str
    median: float  # median of X, in days
    sigma: float   # standard deviation of log X

    def cdf(self, x) -> np.ndarray:
        """P(X <= x) for the continuous delay, x in days."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros_like(x)
        pos = x > 0
        z = (np.log(x[pos]) - math.log(self.median)) / (self.sigma * math.sqrt(2.0))
        out[pos] = 0.5 * (1.0 + _erf(z))
        return out

    def F(self, dmax: int = D_MAX) -> np.ndarray:
        """P(reported within d whole days), d = 0..dmax; F[dmax] = 1 (later delays count at dmax)."""
        f = self.cdf(np.arange(dmax + 1) + 1.0)
        f[-1] = 1.0
        return f

    def pmf(self, dmax: int = D_MAX) -> np.ndarray:
        return np.diff(np.concatenate([[0.0], self.F(dmax)]))

    def quantile_days(self, q: float) -> int:
        """Smallest whole number of days d with F(d) >= q."""
        return int(np.argmax(self.F() >= q))


FAST = Delay("fast", 0.6, 1.0)
MEDIUM = Delay("medium", 2.0, 1.0)
SLOW = Delay("slow", 6.0, 0.8)
SCENARIOS = (FAST, MEDIUM, SLOW)
BACKLOG = Delay("backlog", 3.0, 1.0)  # M2: the medium delays, half as long again, after a backlog


# ---------------------------------------------------------------------------
# M1. Right truncation
# ---------------------------------------------------------------------------
def window_shortfall(delay: Delay, days: int) -> float:
    """Share of the last `days` days still missing at the end of the last day (constant incidence)."""
    return float(1.0 - delay.F()[:days].mean())


def apparent_weekly_change(delay: Delay, growth: float) -> float:
    """Week-over-week change of the last two weeks as known at the end of the last day.

    Incidence grows as exp(growth * t); the true change is exp(7 * growth) - 1.
    """
    age = np.arange(14)
    w = np.exp(-growth * age) * delay.F()[age]
    return float(w[:7].sum() / w[7:].sum() - 1.0)


def growth_that_looks_flat(delay: Delay) -> float:
    """Daily growth rate at which the last week, as known, equals the week before it."""
    lo, hi = 0.0, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if apparent_weekly_change(delay, mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate_counts(delay: Delay, n_days: int = 56, base: float = 1000.0, growth: float = 0.0,
                    seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Daily counts with Poisson noise, split by whole-day delay.

    Returns (final count per day, count known at the end of the last day).
    """
    rng = np.random.default_rng(seed)
    lam = base * np.exp(growth * (np.arange(n_days) - (n_days - 1)))
    y = rng.poisson(lam)
    n = np.vstack([rng.multinomial(k, delay.pmf()) for k in y])
    age = (n_days - 1) - np.arange(n_days)
    C = n.cumsum(axis=1)
    known = C[np.arange(n_days), np.minimum(age, D_MAX)]
    return C[:, -1], known


def truncation_bias_against_noise(delay: Delay, days: int = 7, base: float = 1000.0,
                                  reps: int = 2000, seed: int = 2) -> dict[str, float]:
    """Monte Carlo of the last-`days` total: mean shortfall against the Poisson noise of the truth."""
    rng = np.random.default_rng(seed)
    p = delay.F()[:days]
    final = rng.poisson(base, size=(reps, days))
    known = rng.binomial(final, p)
    tot_f, tot_k = final.sum(axis=1), known.sum(axis=1)
    return {
        "mean_final": float(tot_f.mean()),
        "mean_known": float(tot_k.mean()),
        "mean_shortfall": float((tot_f - tot_k).mean()),
        "sd_final": float(tot_f.std(ddof=1)),
    }


# ---------------------------------------------------------------------------
# M2. Waiting against estimating
# ---------------------------------------------------------------------------
def on_time_completeness(delay: Delay, slack: float, window: float = 1.0, grid: int = 4000) -> float:
    """Share of a window's events in its on-time pane under a watermark with fixed slack.

    Events are spread evenly over a window of `window` days. The watermark passes the end of the
    window `slack` days after it in processing time, so an event `v` days before the window's end
    is in the pane if its delay is at most v + slack.
    """
    v = (np.arange(grid) + 0.5) / grid * window
    return float(delay.cdf(slack + v).mean())


def slack_for(delay: Delay, share: float, window: float = 1.0) -> float:
    """Smallest watermark slack (days) that puts `share` of a window's events in its on-time pane."""
    lo, hi = 0.0, 60.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if on_time_completeness(delay, mid, window) < share:
            lo = mid
        else:
            hi = mid
    return hi


def dropped_share(delay: Delay, slack: float, lateness: float, window: float = 1.0) -> float:
    """Share of events dropped for good: arriving after the watermark passed window end + lateness."""
    return 1.0 - on_time_completeness(delay, slack + lateness, window)


def simulate_reporting(n_days: int = 420, base: float = 1000.0, drift_sd: float = 0.03,
                       before: Delay = MEDIUM, after: Delay | None = None, shift_day: int | None = None,
                       seed: int = 3) -> np.ndarray:
    """Reporting triangle n[t, d]: events of day t reported with a delay of d whole days.

    Incidence follows a random walk on the log scale. With `after` and `shift_day`, events from
    `shift_day` on use the `after` delay distribution.
    """
    rng = np.random.default_rng(seed)
    z = np.cumsum(rng.normal(0.0, drift_sd, n_days))
    y = rng.poisson(base * np.exp(z - z.mean()))
    pb = before.pmf()
    pa = after.pmf() if after is not None else pb
    rows = [rng.multinomial(k, pa if (shift_day is not None and t >= shift_day) else pb)
            for t, k in enumerate(y)]
    return np.vstack(rows)


def development_factors(C: np.ndarray, T: int, window: int = 28) -> np.ndarray:
    """Chain-ladder factors f[k] = C[., k+1] / C[., k] from the `window` latest days with k+1 known."""
    dmax = C.shape[1] - 1
    f = np.ones(dmax)
    for k in range(dmax):
        last = T - k - 1
        rows = np.arange(max(0, last - window + 1), last + 1)
        den = C[rows, k].sum()
        if den > 0:
            f[k] = C[rows, k + 1].sum() / den
    return f


def nowcast_by_age(C: np.ndarray, T: int, ages: np.ndarray, window: int = 28) -> np.ndarray:
    """Chain-ladder nowcast, at the end of day T, of the days whose age T - t is in `ages`."""
    f = development_factors(C, T, window)
    tail = np.concatenate([np.cumprod(f[::-1])[::-1], [1.0]])  # tail[a] = prod_{k>=a} f[k]
    t = T - ages
    return C[t, ages] * tail[ages]


def compare_policies(n: np.ndarray, T_range: range, waits: np.ndarray, window: int = 28) -> pd.DataFrame:
    """Mean absolute percentage error of two ways to publish day t after waiting w days.

    raw: the count known at the end of day t + w, published as it is.
    nowcast: the same count scaled up by the chain-ladder factors estimated at that moment.
    """
    C = n.cumsum(axis=1)
    final = C[:, -1].astype(float)
    raw_err, now_err = [], []
    for T in T_range:
        t = T - waits
        raw = C[t, waits]
        now = nowcast_by_age(C, T, waits, window)
        raw_err.append(np.abs(raw - final[t]) / final[t])
        now_err.append(np.abs(now - final[t]) / final[t])
    return pd.DataFrame({"wait": waits, "raw": np.mean(raw_err, axis=0), "nowcast": np.mean(now_err, axis=0)})


def first_wait_below(errors: pd.Series, waits: np.ndarray, level: float) -> int | None:
    below = np.flatnonzero(errors.to_numpy() < level)
    return int(waits[below[0]]) if below.size else None


# ---------------------------------------------------------------------------
# A simulated shop, used by M3 and M5
# ---------------------------------------------------------------------------
SHOP_DAYS = 90                 # orders are placed on days 0..89
ORDERS_PER_DAY = 2000
WEEKEND_FACTOR = 0.8
OUTAGES = {25: 0.65, 47: 0.65, 63: 0.65}  # days with a real fall in demand (factor)
OFFLINE_SHARE = 0.10           # orders taken offline (app without signal, partner feed)
ONLINE_LAG = (60.0, 1.0)       # seconds: median and log-sd of the lag of online orders
OFFLINE_LAG = (8 * HOUR, 1.2)  # seconds: median and log-sd of the lag of offline orders
REFUND_RATE = 0.06             # orders later partly refunded (amount cut to 30-90%)
CANCEL_RATE = 0.02             # orders later cancelled (amount set to 0)
CORRECTION_LAG = (4 * DAY, 0.8)  # seconds after the order: median and log-sd
RUN_HOUR = 6                   # the daily job runs at 06:00
LAST_RUN_DAY = 150             # the shop's pipeline is followed until day 150


@dataclass
class Shop:
    orders: pd.DataFrame    # one row per order: order_id, event_ts, amount, offline, arrival_ts
    versions: pd.DataFrame  # one row per version: order_id, event_ts, version, amount, visible_at


def simulate_shop(seed: int = 4) -> Shop:
    rng = np.random.default_rng(seed)
    days = np.arange(SHOP_DAYS)
    factor = np.where(days % 7 >= 5, WEEKEND_FACTOR, 1.0) * np.array([OUTAGES.get(d, 1.0) for d in days])
    counts = rng.poisson(ORDERS_PER_DAY * factor)
    day = np.repeat(days, counts)
    n = day.size
    event_ts = day * DAY + rng.integers(0, DAY, n)
    order = np.argsort(event_ts, kind="stable")
    event_ts = event_ts[order]
    amount = np.round(np.exp(rng.normal(math.log(4000.0), 0.8, n))).astype(np.int64)  # cents
    offline = rng.random(n) < OFFLINE_SHARE
    lag = np.where(offline,
                   rng.lognormal(math.log(OFFLINE_LAG[0]), OFFLINE_LAG[1], n),
                   rng.lognormal(math.log(ONLINE_LAG[0]), ONLINE_LAG[1], n))
    arrival_ts = event_ts + np.ceil(lag).astype(np.int64)
    order_id = np.arange(1, n + 1)
    orders = pd.DataFrame({"order_id": order_id, "event_ts": event_ts, "amount": amount,
                           "offline": offline, "arrival_ts": arrival_ts})

    u = rng.random(n)
    corrected = u < REFUND_RATE + CANCEL_RATE
    cancelled = u < CANCEL_RATE
    new_amount = np.where(cancelled, 0, np.round(amount * rng.uniform(0.3, 0.9, n))).astype(np.int64)
    corr_lag = np.ceil(rng.lognormal(math.log(CORRECTION_LAG[0]), CORRECTION_LAG[1], n)).astype(np.int64)
    corr_ts = np.maximum(arrival_ts + 1, event_ts + corr_lag)
    v1 = pd.DataFrame({"order_id": order_id, "event_ts": event_ts, "version": 1,
                       "amount": amount, "visible_at": arrival_ts})
    v2 = pd.DataFrame({"order_id": order_id[corrected], "event_ts": event_ts[corrected], "version": 2,
                       "amount": new_amount[corrected], "visible_at": corr_ts[corrected]})
    versions = pd.concat([v1, v2], ignore_index=True).sort_values(["visible_at", "order_id"], kind="stable")
    return Shop(orders, versions.reset_index(drop=True))


def run_ts(day: int) -> int:
    return day * DAY + RUN_HOUR * HOUR


# ---------------------------------------------------------------------------
# M3a. High-water marks
# ---------------------------------------------------------------------------
def hwm_loaded(stamp: np.ndarray, visible_at: np.ndarray, interval: int, start: int = 0) -> np.ndarray:
    """Which rows an incremental load with a high-water mark on `stamp` ever picks up.

    The load runs every `interval` seconds from `start`. Each run reads the rows visible by then
    whose stamp is greater than the largest stamp loaded so far. A row is therefore loaded if, at
    the first run after it becomes visible, its stamp exceeds the largest stamp among all rows
    visible at the run before. (Rows skipped earlier never raise the mark, because they were
    skipped for being below it.)
    """
    order = np.argsort(visible_at, kind="stable")
    vis_sorted = visible_at[order]
    cummax = np.maximum.accumulate(stamp[order])
    first_run = start + interval * np.ceil((visible_at - start) / interval).astype(np.int64)
    prev_run = first_run - interval
    idx = np.searchsorted(vis_sorted, prev_run, side="right") - 1
    mark = np.where(idx >= 0, cummax[np.maximum(idx, 0)], np.iinfo(np.int64).min)
    return stamp > mark


HWM_EVENT_SQL = """
insert into target
select order_id, event_ts, amount from source
where arrival_ts <= $run
  and event_ts > (select coalesce(max(event_ts), -1) from target)
"""


def hwm_daily_sql(orders: pd.DataFrame, first_day: int = 0, last_day: int = LAST_RUN_DAY) -> int:
    """The same event-time high-water mark as SQL, run every day at 06:00. Returns rows loaded."""
    con = duckdb.connect()
    con.execute("set threads = 1")
    con.register("source", orders[["order_id", "event_ts", "amount", "arrival_ts"]])
    con.execute("create table target (order_id bigint, event_ts bigint, amount bigint)")
    for d in range(first_day, last_day + 1):
        con.execute(HWM_EVENT_SQL, {"run": run_ts(d)})
    return con.execute("select count(*) from target").fetchone()[0]


# ---------------------------------------------------------------------------
# M3b. Corrections under three load patterns, and M5 backfills (SQL, DuckDB)
# ---------------------------------------------------------------------------
LOAD_SQL = {
    # Only orders not yet in the table; a later version of a loaded order is ignored.
    "insert_only": """
        insert into insert_only
        select order_id, event_ts, amount from new_versions
        where order_id not in (select order_id from insert_only)
        qualify row_number() over (partition by order_id order by version desc) = 1
    """,
    # Each order's latest version replaces the stored one.
    "upsert": """
        insert or replace into upsert
        select order_id, event_ts, amount from new_versions
        qualify row_number() over (partition by order_id order by version desc) = 1
    """,
    # Every version is kept, stamped with the run that loaded it.
    "version_log": """
        insert into version_log
        select order_id, event_ts, version, amount, $run as loaded_at from new_versions
    """,
}

REPORT_SQL = {
    "insert_only": "select count(*), coalesce(sum(amount), 0) from insert_only where event_ts // 86400 = $day",
    "upsert": "select count(*), coalesce(sum(amount), 0) from upsert where event_ts // 86400 = $day",
    # As of a run: each order's latest version loaded by that run.
    "version_log": """
        select count(*), coalesce(sum(amount), 0) from (
            select amount from version_log
            where event_ts // 86400 = $day and loaded_at <= $as_of
            qualify row_number() over (partition by order_id order by version desc) = 1)
    """,
}

DAILY_JOB_SQL = """
insert into daily (event_day, orders, revenue, computed_at)
select $day, count(*), coalesce(sum(amount), 0), $run from upsert where event_ts // 86400 = $day
"""

BACKFILL_APPEND_SQL = """
insert into daily (event_day, orders, revenue, computed_at)
select event_ts // 86400, count(*), sum(amount), $run from upsert
where event_ts // 86400 between $first and $last group by 1
"""

BACKFILL_REPLACE_SQL = (  # run in one transaction
    "delete from daily where event_day between $first and $last",
    """insert into daily (event_day, orders, revenue, computed_at)
       select event_ts // 86400, count(*), sum(amount), $run from upsert
       where event_ts // 86400 between $first and $last group by 1""",
)

BACKFILL_DAY = 60  # M5: on day 60, after the daily job, days 0..58 are recomputed


def shop_connection(shop: Shop) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("set threads = 1")
    con.register("versions_df", shop.versions)
    con.execute("create table versions as select * from versions_df")
    con.execute("create table insert_only (order_id bigint primary key, event_ts bigint, amount bigint)")
    con.execute("create table upsert (order_id bigint primary key, event_ts bigint, amount bigint)")
    con.execute("create table version_log (order_id bigint, event_ts bigint, version integer, "
                "amount bigint, loaded_at bigint)")
    con.execute("create table daily (event_day bigint, orders bigint, revenue bigint, computed_at bigint)")
    return con


@dataclass
class ShopRun:
    sent: pd.DataFrame        # per pattern and report day: what the 06:00 report said
    daily: pd.DataFrame       # the daily aggregate as it stands at the end (replace backfill)
    daily_appended: pd.DataFrame  # the same aggregate if the backfill had appended
    frozen: pd.DataFrame      # the daily aggregate as it stood just before the backfill
    live: pd.DataFrame        # per run: yesterday and a week before, read from the upsert table
    regenerated: pd.DataFrame  # per pattern and report day: the report rerun at the end
    truth: pd.DataFrame       # per day: final orders and revenue from every version
    totals: pd.DataFrame      # per pattern: orders and revenue of days 0..89 at the end
    con: duckdb.DuckDBPyConnection


def run_shop(shop: Shop, backfill_day: int = BACKFILL_DAY) -> ShopRun:
    """Run the daily pipeline from day 1 to LAST_RUN_DAY at 06:00 and keep what it reported."""
    con = shop_connection(shop)
    sent, live = [], []
    frozen = None
    prev = -1
    for d in range(1, LAST_RUN_DAY + 1):
        run = run_ts(d)
        con.execute("create or replace temp table new_versions as select * from versions "
                    "where visible_at > $prev and visible_at <= $run", {"prev": prev, "run": run})
        for name, sql in LOAD_SQL.items():
            params = {"run": run} if "$run" in sql else {}
            con.execute(sql, params)
        report_day = d - 1
        if report_day < SHOP_DAYS:
            for name, sql in REPORT_SQL.items():
                params = {"day": report_day} | ({"as_of": run} if "$as_of" in sql else {})
                o, r = con.execute(sql, params).fetchone()
                sent.append((name, report_day, o, r))
            con.execute(DAILY_JOB_SQL, {"day": report_day, "run": run})
            if report_day >= 7:
                o1, r1 = con.execute(REPORT_SQL["upsert"], {"day": report_day}).fetchone()
                o7, r7 = con.execute(REPORT_SQL["upsert"], {"day": report_day - 7}).fetchone()
                live.append((report_day, o1, r1, o7, r7))
        if d == backfill_day:
            frozen = con.execute("select * from daily order by event_day").df()
            con.execute("create table daily_appended as select * from daily")
            con.execute(BACKFILL_APPEND_SQL.replace("into daily", "into daily_appended"),
                        {"first": 0, "last": backfill_day - 2, "run": run})
            con.execute("begin transaction")
            con.execute(BACKFILL_REPLACE_SQL[0], {"first": 0, "last": backfill_day - 2})
            con.execute(BACKFILL_REPLACE_SQL[1], {"first": 0, "last": backfill_day - 2, "run": run})
            con.execute("commit")
        elif d > backfill_day and report_day < SHOP_DAYS:
            con.execute("insert into daily_appended select * from daily where event_day = $day",
                        {"day": report_day})
        prev = run

    sent_df = pd.DataFrame(sent, columns=["pattern", "day", "orders", "revenue"])
    regen = []
    for name, sql in REPORT_SQL.items():
        for day in range(SHOP_DAYS):
            params = {"day": day} | ({"as_of": run_ts(day + 1)} if "$as_of" in sql else {})
            o, r = con.execute(sql, params).fetchone()
            regen.append((name, day, o, r))
    truth = con.execute("""
        select event_ts // 86400 as day, count(*) as orders, sum(amount) as revenue from (
            select * from versions
            qualify row_number() over (partition by order_id order by version desc) = 1)
        group by 1 order by 1""").df()
    totals = con.execute("""
        select 'insert_only' as pattern, count(*) as orders, sum(amount) as revenue from insert_only
        union all select 'upsert', count(*), sum(amount) from upsert
        union all select 'version_log', count(*), sum(amount) from (
            select amount from version_log
            qualify row_number() over (partition by order_id order by version desc) = 1)
        union all select 'truth', count(*), sum(amount) from (
            select amount from versions
            qualify row_number() over (partition by order_id order by version desc) = 1)
    """).df()
    daily = con.execute("select * from daily order by event_day").df()
    appended = con.execute("""select event_day, sum(orders) as orders, sum(revenue) as revenue
                              from daily_appended group by 1 order by 1""").df()
    live_df = pd.DataFrame(live, columns=["day", "orders", "revenue", "orders_week_before", "revenue_week_before"])
    return ShopRun(sent_df, daily, appended, frozen, live_df,
                   pd.DataFrame(regen, columns=["pattern", "day", "orders", "revenue"]),
                   truth, totals, con)


ALERT_DROP = 0.10  # alert when a day is more than 10% below the same weekday a week earlier


def comparisons(run: ShopRun, col: str, backfill_day: int = BACKFILL_DAY) -> pd.DataFrame:
    """Each report day D >= 7 against D - 7, read four ways, with the alert each reading raises.

    truth:  both days final.
    frozen: both days as the daily job computed them (same maturity, never backfilled).
    stored: what the daily table held when D was reported; after the backfill, D - 7 may be
            a restated (complete) value while D is not.
    live:   both days read from the upsert table at the run that reported D.
    """
    t = run.truth.set_index("day")[col]
    sent = run.sent[run.sent.pattern == "upsert"].set_index("day")[col]
    restated = run.daily.set_index("event_day")[col]
    live = run.live.set_index("day")
    rows = []
    for D in range(7, SHOP_DAYS):
        restated_before = (D - 7 <= backfill_day - 2) and (D + 1 > backfill_day)
        prev_stored = restated.loc[D - 7] if restated_before else sent.loc[D - 7]
        rows.append({
            "day": D, "mixed": restated_before,
            "truth": t.loc[D] / t.loc[D - 7],
            "frozen": sent.loc[D] / sent.loc[D - 7],
            "stored": sent.loc[D] / prev_stored,
            "live": live.loc[D, col] / live.loc[D, col + "_week_before"],
        })
    out = pd.DataFrame(rows).set_index("day")
    for k in ("truth", "frozen", "stored", "live"):
        out[k + "_alert"] = out[k] < 1.0 - ALERT_DROP
    return out


def false_alert_rate(mean: float, completeness: float, threshold: float = ALERT_DROP,
                     reps: int = 1_000_000, seed: int = 8) -> float:
    """Chance that a week-over-week rule fires on Poisson noise alone.

    Both days have the same expected count; the later one is seen with `completeness` of its
    rows (each row present independently), the earlier one complete.
    """
    rng = np.random.default_rng(seed)
    before = rng.poisson(mean, reps)
    now = rng.binomial(rng.poisson(mean, reps), completeness)
    return float((now < (1.0 - threshold) * before).mean())


# ---------------------------------------------------------------------------
# M4. Features
# ---------------------------------------------------------------------------
def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(x, dtype=float), x.astype(float)])
    a, b = np.linalg.lstsq(X, y.astype(float), rcond=None)[0]
    return float(a), float(b)


@dataclass
class CountFeatures:
    final: np.ndarray   # last-7-day count once every event has arrived
    as_of: np.ndarray   # the same count as known at the end of the snapshot day
    target: np.ndarray  # count of the next 7 days (final)


def count_features(delay: Delay = MEDIUM, n: int = 10_000, seed: int = 5) -> CountFeatures:
    """Per customer: a 7-day event count as final and as known on the day, and the next week's count."""
    rng = np.random.default_rng(seed)
    rate = rng.gamma(2.0, 5.0, n)  # events per day
    past = rng.poisson(rate[:, None], size=(n, 7))
    F = delay.F()
    age = np.arange(6, -1, -1)  # the oldest day of the window is 6 days old at the snapshot
    known = rng.binomial(past, F[age][None, :])
    future = rng.poisson(rate[:, None], size=(n, 7)).sum(axis=1)
    return CountFeatures(past.sum(axis=1), known.sum(axis=1), future)


def calibration(train: CountFeatures, serve: CountFeatures) -> dict[str, float]:
    """Mean prediction over mean outcome at serving time, for the two ways of building training rows."""
    out = {}
    for name, x_train in (("trained_on_final", train.final), ("trained_as_of", train.as_of)):
        a, b = fit_linear(x_train, train.target)
        pred = a + b * serve.as_of
        out[name] = float(pred.mean() / serve.target.mean())
        out[name + "_mae"] = float(np.abs(pred - serve.target).mean())
        out[name + "_slope"] = b
    return out


def fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Logistic regression by Newton's method (iteratively reweighted least squares)."""
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        W = p * (1 - p)
        step = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def predict_logistic(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(beta[0] + X @ beta[1:])))


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve from ranks (Mann-Whitney), ties given their average rank."""
    r = pd.Series(score).rank(method="average").to_numpy()
    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


LABEL_DAYS = 90
DOWNGRADE_IF_CHURN = 0.5   # churners who move to the basic plan before they leave
DOWNGRADE_IF_STAY = 0.05   # customers who stay but still move to the basic plan
COHORT_DAYS = {"training": 180, "production": 540}  # snapshot days
TODAY = 720                # the day the training table is built


def simulate_plan_history(n: int = 20_000, seed: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two cohorts of customers with a churn label and a plan whose changes are dated.

    Returns (customers, plan_history). plan_history is a Type 2 table: one row per customer and
    plan, with valid_from and valid_to (exclusive; 999999 for the current row).
    """
    rng = np.random.default_rng(seed)
    cust, hist = [], []
    cid = 0
    for cohort, snap in COHORT_DAYS.items():
        risk = rng.normal(0.0, 1.0, n)
        usage = -0.8 * risk + rng.normal(0.0, 1.0, n)
        basic0 = rng.random(n) < 1.0 / (1.0 + np.exp(-(-0.8 + 0.3 * risk)))
        churn = rng.random(n) < 1.0 / (1.0 + np.exp(-(-2.0 + 1.0 * risk)))
        churn_day = snap + rng.integers(10, LABEL_DAYS, n)
        down = ~basic0 & np.where(churn, rng.random(n) < DOWNGRADE_IF_CHURN, rng.random(n) < DOWNGRADE_IF_STAY)
        down_day = np.where(churn, snap + 1 + (rng.random(n) * (churn_day - snap - 1)).astype(int),
                            snap + rng.integers(1, LABEL_DAYS, n))
        start = snap - rng.integers(30, 150, n)
        for i in range(n):
            cid += 1
            plan0 = "basic" if basic0[i] else "pro"
            cust.append((cid, cohort, snap, usage[i], int(churn[i])))
            if down[i]:
                hist.append((cid, plan0, int(start[i]), int(down_day[i])))
                hist.append((cid, "basic", int(down_day[i]), 999_999))
            else:
                hist.append((cid, plan0, int(start[i]), 999_999))
    customers = pd.DataFrame(cust, columns=["customer_id", "cohort", "snapshot", "usage", "churned"])
    plans = pd.DataFrame(hist, columns=["customer_id", "plan", "valid_from", "valid_to"])
    return customers, plans


FEATURES_TYPE1_SQL = """
-- the plan as it is in the warehouse today (a Type 1 view of the dimension)
select c.customer_id, c.cohort, c.usage, c.churned, (p.plan = 'basic')::int as basic
from customers c join plan_history p on p.customer_id = c.customer_id and p.valid_to = 999999
order by c.customer_id
"""

FEATURES_AS_OF_SQL = """
-- the plan as it was on the snapshot day (point in time)
select c.customer_id, c.cohort, c.usage, c.churned, (p.plan = 'basic')::int as basic
from customers c asof join plan_history p
  on p.customer_id = c.customer_id and c.snapshot >= p.valid_from
order by c.customer_id
"""


def churn_features(customers: pd.DataFrame, plans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    con.execute("set threads = 1")
    con.register("customers", customers)
    con.register("plan_history", plans)
    return con.execute(FEATURES_TYPE1_SQL).df(), con.execute(FEATURES_AS_OF_SQL).df()


def leakage_experiment(seed: int = 6, test_share: float = 0.3) -> dict[str, float]:
    """Offline and production AUC of a churn model with the plan joined in two ways."""
    customers, plans = simulate_plan_history(seed=seed)
    type1, as_of = churn_features(customers, plans)
    rng = np.random.default_rng(seed + 100)
    train_mask = (type1["cohort"] == "training").to_numpy()
    holdout = rng.random(len(type1)) < test_share
    fit_rows = train_mask & ~holdout
    test_rows = train_mask & holdout
    prod_rows = ~train_mask
    cols = ["usage", "basic"]
    y = type1["churned"].to_numpy()
    out = {"churn_rate_training": float(y[train_mask].mean()), "churn_rate_production": float(y[prod_rows].mean())}
    serve = as_of[cols].to_numpy()  # in production the plan can only be the one known on the day
    for name, table in (("type1", type1), ("as_of", as_of)):
        X = table[cols].to_numpy()
        beta = fit_logistic(X[fit_rows], y[fit_rows])
        out[f"{name}_offline_auc"] = auc(y[test_rows], predict_logistic(beta, X[test_rows]))
        p_prod = predict_logistic(beta, serve[prod_rows])
        out[f"{name}_production_auc"] = auc(y[prod_rows], p_prod)
        out[f"{name}_production_mean_pred"] = float(p_prod.mean())
        out[f"{name}_coef_basic"] = float(beta[2])
        out[f"{name}_coef_usage"] = float(beta[1])
    basic_t1 = type1.loc[train_mask, "basic"].to_numpy()
    basic_ao = as_of.loc[train_mask, "basic"].to_numpy()
    yt = y[train_mask]
    out["basic_share_type1"] = float(basic_t1.mean())
    out["basic_share_as_of"] = float(basic_ao.mean())
    out["churn_if_basic_type1"] = float(yt[basic_t1 == 1].mean())
    out["churn_if_pro_type1"] = float(yt[basic_t1 == 0].mean())
    out["churn_if_basic_as_of"] = float(yt[basic_ao == 1].mean())
    out["churn_if_pro_as_of"] = float(yt[basic_ao == 0].mean())
    out["rows_changed"] = int((basic_t1 != basic_ao).sum())
    out["rows_training"] = int(train_mask.sum())
    return out
