"""Functions behind Part 1: one historical idea per section, each rebuilt in code.

A. Insurance: the run-off triangle and the chain ladder (Taylor and Ashe, 1983; Mack, 1993).
B. Epidemiology: reporting delays and nowcasting (STEC O104:H4 outbreak, Germany, 2011).
C. Temporal databases: valid time and transaction time in one bitemporal table.
D. Data warehouses: slowly changing dimensions, Type 1 against Type 2.
E. Stream processing: event time, processing time, watermarks, triggers and accumulation.

Everything is deterministic: the only random numbers (section D) use a fixed seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# A. Insurance: run-off triangle, chain ladder, Mack's standard error
# ---------------------------------------------------------------------------
CHAIN_LADDER_SQL = """
-- tri(origin, dev, cum): cumulative amount for each origin period at each development step.
with pairs as (
    select a.dev, a.cum as c_now, b.cum as c_next
    from tri a join tri b on b.origin = a.origin and b.dev = a.dev + 1
),
factors as (
    select dev, sum(c_next) / sum(c_now) as f from pairs group by dev
),
latest as (
    select origin, arg_max(dev, dev) as dev, arg_max(cum, dev) as cum from tri group by origin
)
select l.origin, l.dev as latest_dev, l.cum as latest,
       l.cum * coalesce((select exp(sum(ln(f))) from factors where factors.dev >= l.dev), 1) as ultimate
from latest l order by l.origin
"""


def taylor_ashe() -> pd.DataFrame:
    """The Taylor-Ashe triangle in long form: origin (accident year), dev (development year), cum."""
    df = pd.read_csv(DATA / "taylor_ashe.csv")
    return df.rename(columns={"accident_year": "origin", "development_year": "dev", "cumulative_claims": "cum"})


def chain_ladder_sql(tri: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chain-ladder development factors and ultimates, computed in SQL by DuckDB."""
    con = duckdb.connect()
    con.register("tri", tri[["origin", "dev", "cum"]].astype({"cum": "float64"}))
    factors = con.execute("""
        with pairs as (select a.dev, a.cum c_now, b.cum c_next from tri a
                       join tri b on b.origin = a.origin and b.dev = a.dev + 1)
        select dev, sum(c_next) / sum(c_now) as f from pairs group by dev order by dev""").df()
    ult = con.execute(CHAIN_LADDER_SQL).df()
    ult["reserve"] = ult["ultimate"] - ult["latest"]
    return factors, ult


def triangle_matrix(tri: pd.DataFrame) -> np.ndarray:
    n = int(tri["origin"].max())
    c = np.full((n, n), np.nan)
    for r in tri.itertuples():
        c[r.origin - 1, r.dev - 1] = r.cum
    return c


def mack_standard_errors(c: np.ndarray) -> tuple[np.ndarray, float]:
    """Mack (1993) standard errors of the chain-ladder reserves, per origin and for the total.

    The last sigma is extrapolated as in Mack (1993): min(s_{n-2}^4 / s_{n-3}^2, s_{n-3}^2, s_{n-2}^2).
    """
    n = c.shape[0]
    f = np.array([np.nansum(c[: n - 1 - k, k + 1]) / np.nansum(c[: n - 1 - k, k]) for k in range(n - 1)])
    full = c.copy()
    for i in range(n):
        for k in range(n - 1):
            if np.isnan(full[i, k + 1]):
                full[i, k + 1] = full[i, k] * f[k]
    s2 = np.zeros(n - 1)
    for k in range(n - 2):
        m = n - 1 - k
        s2[k] = np.sum(c[:m, k] * (c[:m, k + 1] / c[:m, k] - f[k]) ** 2) / (m - 1)
    s2[n - 2] = min(s2[n - 3] ** 2 / s2[n - 4], s2[n - 4], s2[n - 3])
    col_sums = np.array([np.sum(c[: n - 1 - k, k]) for k in range(n - 1)])
    mse = np.zeros(n)
    for i in range(1, n):
        ks = range(n - 1 - i, n - 1)
        mse[i] = full[i, -1] ** 2 * sum(s2[k] / f[k] ** 2 * (1 / full[i, k] + 1 / col_sums[k]) for k in ks)
    total = mse.sum()
    for i in range(1, n):
        for j in range(i + 1, n):
            ks = range(n - 1 - i, n - 1)
            total += full[i, -1] * full[j, -1] * sum(2 * s2[k] / f[k] ** 2 / col_sums[k] for k in ks)
    return np.sqrt(mse), float(np.sqrt(total))


# ---------------------------------------------------------------------------
# B. Epidemiology: reporting delays and a chain-ladder nowcast
# ---------------------------------------------------------------------------
MAX_DELAY = 15  # the surveillance package sets delays above 15 days to 15


def hus_cases() -> pd.DataFrame:
    df = pd.read_csv(DATA / "hus_o104_2011.csv", parse_dates=["hospitalized", "reported"])
    df["delay"] = (df["reported"] - df["hospitalized"]).dt.days
    return df


def known_by(cases: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Cases per hospitalization date, counting only reports received by the end of ``as_of``."""
    seen = cases[cases["reported"] <= as_of]
    days = pd.date_range(cases["hospitalized"].min(), as_of, freq="D")
    return seen.groupby("hospitalized").size().reindex(days, fill_value=0)


def final_counts(cases: pd.DataFrame) -> pd.Series:
    days = pd.date_range(cases["hospitalized"].min(), cases["hospitalized"].max(), freq="D")
    return cases.groupby("hospitalized").size().reindex(days, fill_value=0)


def nowcast_chain_ladder(cases: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Chain-ladder nowcast of cases per hospitalization date, using only reports received by ``as_of``.

    The daily reporting triangle is the actuarial triangle with days in place of years: origin =
    hospitalization date, development = reporting delay in days. Factors pool all origins, so an
    origin with no report yet keeps a nowcast of zero (the method's known weakness).
    """
    seen = cases[cases["reported"] <= as_of]
    days = pd.date_range(cases["hospitalized"].min(), as_of, freq="D")
    counts = seen.groupby(["hospitalized", "delay"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=days, columns=range(MAX_DELAY + 1), fill_value=0)
    cum = counts.cumsum(axis=1).to_numpy(dtype=float)
    observable = np.array([[(as_of - t).days >= d for d in range(MAX_DELAY + 1)] for t in days])
    factors = []
    for d in range(MAX_DELAY):
        rows = observable[:, d + 1]
        num, den = cum[rows, d + 1].sum(), cum[rows, d].sum()
        factors.append(num / den if den > 0 else 1.0)
    factors = np.array(factors)
    out = []
    for i, t in enumerate(days):
        latest_dev = min((as_of - t).days, MAX_DELAY)
        known = cum[i, latest_dev]
        completion = float(np.prod(factors[latest_dev:]))
        out.append((t, known, known * completion, 1 / completion))
    return pd.DataFrame(out, columns=["hospitalized", "known", "nowcast", "completeness"]).set_index("hospitalized")


# ---------------------------------------------------------------------------
# C. Temporal databases: one bitemporal table
# ---------------------------------------------------------------------------
FAR = date(9999, 12, 31)
BITEMPORAL_ROWS = [
    # price, valid_from, valid_to, recorded_from, recorded_to
    (10.00, date(2026, 1, 1), FAR, date(2025, 12, 20), date(2026, 2, 25)),
    (10.00, date(2026, 1, 1), date(2026, 3, 1), date(2026, 2, 25), date(2026, 3, 20)),
    (12.00, date(2026, 3, 1), FAR, date(2026, 2, 25), date(2026, 3, 20)),
    (10.00, date(2026, 1, 1), date(2026, 2, 15), date(2026, 3, 20), FAR),
    (11.50, date(2026, 2, 15), FAR, date(2026, 3, 20), FAR),
]
ORDERS = [  # order_id, order_date, quantity; each order is invoiced on its order date
    (1, date(2026, 1, 20), 100),
    (2, date(2026, 2, 10), 80),
    (3, date(2026, 2, 20), 120),
    (4, date(2026, 3, 5), 90),
    (5, date(2026, 3, 18), 60),
    (6, date(2026, 3, 25), 110),
]


def bitemporal_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("create table price (price decimal(8,2), valid_from date, valid_to date, "
                "recorded_from date, recorded_to date)")
    con.executemany("insert into price values (?, ?, ?, ?, ?)", BITEMPORAL_ROWS)
    con.execute("create table orders (order_id int, order_date date, quantity int)")
    con.executemany("insert into orders values (?, ?, ?)", ORDERS)
    return con


PRICE_AS_OF_SQL = """
select price from price
where valid_from <= $valid and $valid < valid_to
  and recorded_from <= $known and $known < recorded_to
"""

INVOICE_SQL = """
-- What was invoiced (price as believed on the order date) against what is due on current knowledge.
select o.order_id, o.order_date, o.quantity,
       billed.price as billed_price, due.price as due_price,
       o.quantity * (due.price - billed.price) as restatement
from orders o
join price billed on billed.valid_from <= o.order_date and o.order_date < billed.valid_to
                 and billed.recorded_from <= o.order_date and o.order_date < billed.recorded_to
join price due    on due.valid_from <= o.order_date and o.order_date < due.valid_to
                 and due.recorded_from <= $today and $today < due.recorded_to
order by o.order_id
"""


def price_as_of(con, valid: date, known: date) -> float:
    return float(con.execute(PRICE_AS_OF_SQL, {"valid": valid, "known": known}).fetchone()[0])


def invoices(con, today: date) -> pd.DataFrame:
    return con.execute(INVOICE_SQL, {"today": today}).df()


# ---------------------------------------------------------------------------
# D. Data warehouses: slowly changing dimensions
# ---------------------------------------------------------------------------
REGIONS = ["North", "South", "East", "West"]


@dataclass(frozen=True)
class Warehouse:
    customers: pd.DataFrame  # Type 2 history: customer_id, region, valid_from, valid_to
    sales: pd.DataFrame      # customer_id, sale_date, amount


def simulate_warehouse(n_customers: int = 2000, move_rate: float = 0.08, seed: int = 20261110) -> Warehouse:
    """Customers who sometimes move region, and their sales from 2023 to 2025."""
    rng = np.random.default_rng(seed)
    start, end = date(2023, 1, 1), date(2026, 1, 1)
    span = (end - start).days
    history, sales = [], []
    for cid in range(n_customers):
        region = rng.choice(REGIONS)
        t = start
        while True:
            gap = int(rng.exponential(365 / move_rate))
            nxt = t + timedelta(days=max(gap, 1))
            if nxt >= end:
                history.append((cid, region, t, FAR))
                break
            history.append((cid, region, t, nxt))
            region = rng.choice([r for r in REGIONS if r != region])
            t = nxt
        n_sales = rng.poisson(12)
        for _ in range(n_sales):
            sales.append((cid, start + timedelta(days=int(rng.integers(span))), round(float(rng.gamma(2.0, 40.0)), 2)))
    customers = pd.DataFrame(history, columns=["customer_id", "region", "valid_from", "valid_to"])
    sales_df = pd.DataFrame(sales, columns=["customer_id", "sale_date", "amount"]).sort_values(["customer_id", "sale_date"])
    for col in ("valid_from", "valid_to"):
        customers[col] = pd.to_datetime(customers[col])
    sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"])
    return Warehouse(customers, sales_df.reset_index(drop=True))


TYPE2_SQL = """
-- Point-in-time join: each sale takes the region the customer had on the sale date.
select d.region, sum(s.amount) as revenue
from sales s asof join customers d
  on s.customer_id = d.customer_id and s.sale_date >= d.valid_from
where s.sale_date >= date '2024-01-01' and s.sale_date < date '2025-01-01'
group by d.region order by d.region
"""

TYPE1_SQL = """
-- Overwrite-in-place dimension: one row per customer, holding the region as of the run date.
with current_region as (
    select customer_id, region from customers
    where valid_from <= $run_date and $run_date < valid_to
)
select c.region, sum(s.amount) as revenue
from sales s join current_region c using (customer_id)
where s.sale_date >= date '2024-01-01' and s.sale_date < date '2025-01-01'
group by c.region order by c.region
"""


def regional_revenue_2024(wh: Warehouse) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("customers", wh.customers)
    con.register("sales", wh.sales)
    t2 = con.execute(TYPE2_SQL).df().set_index("region")["revenue"].rename("type2")
    runs = {}
    for label, run in (("type1_run_2025_01_05", date(2025, 1, 5)), ("type1_run_2026_01_05", date(2026, 1, 5))):
        runs[label] = con.execute(TYPE1_SQL, {"run_date": run}).df().set_index("region")["revenue"]
    return pd.concat([t2, pd.Series(runs["type1_run_2025_01_05"], name="type1_run_2025_01_05"),
                      pd.Series(runs["type1_run_2026_01_05"], name="type1_run_2026_01_05")], axis=1)


# ---------------------------------------------------------------------------
# E. Stream processing: event time, processing time, watermarks, triggers
# ---------------------------------------------------------------------------
WINDOW = 120          # seconds: fixed event-time windows of two minutes
WATERMARK_SLACK = 30  # heuristic watermark: largest event time seen, minus 30 seconds
ALLOWED_LATENESS = 120  # late data accepted up to two minutes after the window's on-time pane

# (value, event time, arrival time), in seconds after 12:00:00
EVENTS = [
    (5, 20, 75), (7, 135, 150), (3, 160, 170), (4, 150, 190), (8, 230, 245),
    (6, 70, 250),   # arrives after its window has fired: a late firing within the allowed lateness
    (2, 290, 300), (3, 335, 350), (9, 100, 380),  # the 9 arrives far behind: dropped
    (1, 390, 400), (8, 430, 445), (4, 470, 485),
]


def window_of(t: int) -> int:
    return t // WINDOW


def stream(events=EVENTS, slack=WATERMARK_SLACK, lateness=ALLOWED_LATENESS):
    """Run fixed event-time windows with a watermark trigger, late firings and allowed lateness.

    Returns the panes emitted, in arrival order, with what each accumulation mode would send:
    discarding (the new elements only), accumulating (the running total), and accumulating with
    retraction (a retraction of the previous total, then the new total).
    """
    order = sorted(events, key=lambda e: e[2])
    sums, fired, max_event, panes, dropped = {}, {}, -1, [], []
    pending = {}
    for value, et, at in order:
        w = window_of(et)
        end = (w + 1) * WINDOW
        watermark_before = max_event - slack
        if w in fired and watermark_before >= end + lateness:
            dropped.append((value, et, at))
        else:
            sums[w] = sums.get(w, 0) + value
            pending[w] = pending.get(w, 0) + value
            if w in fired:  # late element in an already-fired window: late firing now
                panes.append(dict(window=w, time=at, kind="late", discarding=pending[w],
                                  accumulating=sums[w], retraction=-fired[w]))
                fired[w], pending[w] = sums[w], 0
        max_event = max(max_event, et)
        watermark = max_event - slack
        for win in sorted(sums):
            if win not in fired and watermark >= (win + 1) * WINDOW:
                panes.append(dict(window=win, time=at, kind="on time", discarding=pending[win],
                                  accumulating=sums[win], retraction=None))
                fired[win], pending[win] = sums[win], 0
    # end of input: the watermark advances to infinity and fires what is left
    for win in sorted(sums):
        if win not in fired:
            panes.append(dict(window=win, time=None, kind="final", discarding=pending[win],
                              accumulating=sums[win], retraction=None))
            fired[win] = sums[win]
    return pd.DataFrame(panes), dropped


def true_window_sums(events=EVENTS) -> dict[int, int]:
    out: dict[int, int] = {}
    for value, et, _ in events:
        out[window_of(et)] = out.get(window_of(et), 0) + value
    return dict(sorted(out.items()))


def processing_time_sums(events=EVENTS) -> dict[int, int]:
    out: dict[int, int] = {}
    for value, _, at in events:
        out[window_of(at)] = out.get(window_of(at), 0) + value
    return dict(sorted(out.items()))


def watermark_path(events=EVENTS, slack=WATERMARK_SLACK):
    """(arrival time, watermark) after each element, for the figure."""
    order = sorted(events, key=lambda e: e[2])
    max_event, path = -1, []
    for _, et, at in order:
        max_event = max(max_event, et)
        path.append((at, max_event - slack))
    return path
