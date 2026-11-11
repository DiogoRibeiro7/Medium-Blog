"""Functions behind Part 3: reporting delays, nowcast errors and revisions in three public sources
that kept every version.

R. RKI daily COVID-19 case files (one vintage per publication day, April 2021 to December 2023,
   then monthly to November 2026): counts by reporting date (Meldedatum) and by onset date.
H. The German COVID-19 Hospitalization Nowcast Hub: a reporting triangle of hospitalised cases by
   Meldedatum, and the nowcasts that ten models published every day from 22 November 2021 to
   29 April 2022.
G. Quarterly real-GDP vintages for the US, the euro area, Japan and Switzerland, 2002 Q4 to 2024 Q4.

The fetch_*.py scripts write the inputs to data/; nothing here reads the network.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# R. RKI case files
# ---------------------------------------------------------------------------
RKI_FIRST_DAILY = pd.Timestamp("2021-04-02")
RKI_LAST_DAILY = pd.Timestamp("2023-12-31")
RKI_LAST = pd.Timestamp("2026-11-10")
MAX_AGE = 120
STATES = {1: "Schleswig-Holstein", 2: "Hamburg", 3: "Lower Saxony", 4: "Bremen", 5: "North Rhine-Westphalia",
          6: "Hesse", 7: "Rhineland-Palatinate", 8: "Baden-Wurttemberg", 9: "Bavaria", 10: "Saarland",
          11: "Berlin", 12: "Brandenburg", 13: "Mecklenburg-Vorpommern", 14: "Saxony", 15: "Saxony-Anhalt",
          16: "Thuringia"}


@dataclass
class Vintages:
    """V[p, d]: count for date d as published on day p (rows: publication, columns: date)."""

    V: pd.DataFrame
    new: pd.DataFrame
    removed: pd.DataFrame

    def at_age(self, age: int) -> pd.Series:
        """For each date d, the count published on day d + age (NaN if no publication that day).

        A date missing from a publication that exists means no case was known for it: 0.
        """
        out = {}
        pubs = set(self.V.index)
        for d in self.V.columns:
            p = d + pd.Timedelta(days=age)
            if p in pubs and age <= MAX_AGE:
                v = self.V.at[p, d]
                out[d] = 0.0 if np.isnan(v) else v
        return pd.Series(out, dtype=float).sort_index()


def load_vintages(kind: str) -> Vintages:
    df = pd.read_csv(DATA / f"rki_{kind}.csv.gz", parse_dates=["publication", "date"])
    V = df.pivot(index="publication", columns="date", values="cases").sort_index().sort_index(axis=1)
    new = df.pivot(index="publication", columns="date", values="new").sort_index().sort_index(axis=1)
    removed = df.pivot(index="publication", columns="date", values="removed").sort_index().sort_index(axis=1)
    return Vintages(V, new, removed)


def load_snapshots() -> pd.DataFrame:
    """Counts by Meldedatum as published on the first day of each month and on the last day."""
    s = pd.read_csv(DATA / "rki_snapshots.csv.gz", parse_dates=["publication", "date"])
    return s.pivot(index="date", columns="publication", values="cases").sort_index()


def final_melde() -> pd.Series:
    return load_snapshots()[RKI_LAST].fillna(0)


def completeness_table(vin: Vintages, final: pd.Series, ages, first: str, last: str) -> pd.DataFrame:
    """Share of the final count published by each age, for every date in [first, last]."""
    rows = {}
    for a in ages:
        s = vin.at_age(a)
        s = s[(s.index >= first) & (s.index <= last)]
        f = final.reindex(s.index)
        ok = f > 0
        rows[a] = (s[ok] / f[ok])
    return pd.DataFrame(rows)


def flag_reconciliation(vin: Vintages, window: int = 100) -> pd.DataFrame:
    """For each pair of consecutive daily files: does the change between them equal the flagged change?

    Under the published definitions, count(p) - count(p - 1) for a date equals the cases flagged
    new (NeuerFall = 1) plus those flagged removed (NeuerFall = -1) in file p. Compared for the
    `window` dates before p.
    """
    rows = []
    pubs = vin.V.index
    for i in range(1, len(pubs)):
        p, q = pubs[i], pubs[i - 1]
        if (p - q).days != 1 or p > RKI_LAST_DAILY:
            continue
        d = pd.date_range(p - pd.Timedelta(days=window), p - pd.Timedelta(days=1))
        diff = vin.V.loc[p, d].fillna(0) - vin.V.loc[q, d].fillna(0)
        flagged = vin.new.loc[p, d].fillna(0) + vin.removed.loc[p, d].fillna(0)
        rows.append((p, float((diff - flagged).abs().sum()), float(diff.abs().sum()), int((diff != flagged).sum())))
    return pd.DataFrame(rows, columns=["publication", "abs_mismatch", "abs_change", "dates_off"]).set_index("publication")


def seven_day_sums(vin: Vintages, final: pd.Series) -> pd.DataFrame:
    """The 7-day total for publication day p (dates p-7 .. p-1), three ways.

    published: each day's own file, as the RKI published it that day.
    final: the same seven dates in the latest file.
    """
    out = []
    for p in vin.V.index[(vin.V.index >= RKI_FIRST_DAILY) & (vin.V.index <= RKI_LAST_DAILY)]:
        dates = pd.date_range(p - pd.Timedelta(days=7), p - pd.Timedelta(days=1))
        if dates[0] not in vin.V.columns:
            continue
        pub = vin.V.loc[p, dates].fillna(0).sum()
        out.append((p, pub, final.reindex(dates).sum()))
    return pd.DataFrame(out, columns=["publication", "published", "final"]).set_index("publication")


SEVEN_DAY_SQL = """
-- the 7-day count as published on each day, and the same seven dates in the last file;
-- a date absent from a file had no case yet, so the seven dates come from a calendar, not from the file
with win as (
    select publication,
           unnest(generate_series(publication - interval 7 day, publication - interval 1 day,
                                  interval 1 day))::date as date
    from (select distinct publication from melde)
)
select w.publication,
       coalesce(sum(v.cases), 0) as published,
       coalesce(sum(f.cases), 0) as final
from win w
left join melde v on v.publication = w.publication and v.date = w.date
left join snapshots f on f.publication = $last and f.date = w.date
group by w.publication
order by w.publication
"""


def rki_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("set threads = 1")
    for name, path in (("melde", "rki_melde.csv.gz"), ("snapshots", "rki_snapshots.csv.gz")):
        con.execute(f"create table {name} as select * from read_csv('{DATA / path}', header = true)")
    return con


def chain_ladder_daily(C: np.ndarray, T: int, window: int, dmax: int) -> np.ndarray:
    """Nowcast of the value at delay dmax for every row t <= T of a cumulative triangle C[t, d].

    C[t, d] is the count for date t known d days later; at the end of day T, row t has
    delays 0 .. T - t observed. Factors from the `window` latest rows with each delay observed.
    """
    f = np.ones(dmax)
    for k in range(dmax):
        last = T - k - 1
        rows = np.arange(max(0, last - window + 1), last + 1)
        den = np.nansum(C[rows, k])
        if den > 0:
            f[k] = np.nansum(C[rows, k + 1]) / den
    tail = np.concatenate([np.cumprod(f[::-1])[::-1], [1.0]])
    est = np.full(T + 1, np.nan)
    for t in range(max(0, T - dmax), T + 1):
        a = T - t
        est[t] = C[t, a] * tail[a]
    return est


def chain_ladder_weekday(C: np.ndarray, T: int, weeks: int, dmax: int, min_share: float = 0.2) -> np.ndarray:
    """Nowcast of the value at delay dmax, with factors from rows of the same weekday.

    For row t with a = T - t delays observed, the expected share known at delay a is estimated
    from the `weeks` latest rows of the same weekday that are complete (T - j >= dmax). When
    that share is below `min_share` (almost nothing is known yet, as for weekend dates on a
    Monday), the estimate is the mean final count of those rows instead.
    """
    est = np.full(T + 1, np.nan)
    for t in range(max(0, T - dmax), T + 1):
        a = T - t
        rows = np.array([j for j in range(t - 7 * weeks, t, 7) if j >= 0 and T - j >= dmax])
        if rows.size == 0:
            continue
        full = np.nansum(C[rows, dmax])
        share = np.nansum(C[rows, a]) / full if full > 0 else np.nan
        if np.isnan(share) or share < min_share:
            est[t] = full / rows.size
        else:
            est[t] = C[t, a] / share
    return est


def nowcast_seven_day(vin: Vintages, final: pd.Series, first: str, last: str, dmax: int = 28,
                      window: int = 28, weeks: int = 4) -> pd.DataFrame:
    """For each publication p, the 7-day count (dates p-7 .. p-1) three ways, as a share of final - 1.

    raw: as published. chain: chain ladder with factors from the latest `window` days.
    weekday: factors from the same weekday's latest `weeks` complete rows.
    """
    dates = pd.date_range(pd.Timestamp(first) - pd.Timedelta(days=70), last)
    C = cumulative_triangle(vin, dates, dmax)
    fin = final.reindex(dates).to_numpy()
    rows = []
    for T in range(70, len(dates)):
        Cm = C[:T + 1].copy()
        for j in range(max(0, T - dmax), T + 1):
            Cm[j, T - j + 1:] = np.nan
        cl = chain_ladder_daily(Cm, T, window, dmax)
        wd = chain_ladder_weekday(Cm, T, weeks, dmax)
        f = fin[T - 7:T].sum()
        raw = sum(Cm[t, T - t] for t in range(T - 7, T))
        rows.append((dates[T], raw / f - 1, np.nansum(cl[T - 7:T]) / f - 1, np.nansum(wd[T - 7:T]) / f - 1))
    return pd.DataFrame(rows, columns=["publication", "raw", "chain", "weekday"]).set_index("publication")


def cumulative_triangle(vin: Vintages, dates: pd.DatetimeIndex, dmax: int) -> np.ndarray:
    """C[i, a] = count for dates[i] published a days later (carried forward over missing days)."""
    C = np.full((len(dates), dmax + 1), np.nan)
    V = vin.V
    for i, d in enumerate(dates):
        last = np.nan
        for a in range(dmax + 1):
            p = d + pd.Timedelta(days=a)
            if p in V.index and d in V.columns:
                v = V.at[p, d]
                last = 0.0 if np.isnan(v) else v
            C[i, a] = last
    return C


# ---------------------------------------------------------------------------
# H. Hospitalisation nowcast hub
# ---------------------------------------------------------------------------
HUB_FIRST = pd.Timestamp("2021-11-22")
HUB_LAST = pd.Timestamp("2022-04-29")
HUB_DMAX = 80


@dataclass
class HubTriangle:
    dates: pd.DatetimeIndex
    inc: np.ndarray    # inc[i, k]: change at delay k (k = 0..80), and column 81: everything later
    cum: np.ndarray    # cumulative over k = 0..80
    final: np.ndarray  # latest count (all columns)
    at80: np.ndarray   # count 80 days after the Meldedatum

    def index(self, d) -> int:
        return int(self.dates.get_loc(pd.Timestamp(d)))

    def known(self, i: int, p: int) -> float:
        k = p - i
        if k < 0:
            return 0.0
        return float(self.cum[i, min(k, HUB_DMAX)])

    def known7(self, t: int, p: int) -> float:
        return sum(self.known(i, p) for i in range(t - 6, t + 1))

    def final7(self, t: int, which: str = "final") -> float:
        a = self.final if which == "final" else self.at80
        return float(a[t - 6:t + 1].sum())


def load_hub_triangle(location: str = "DE") -> HubTriangle:
    tri = pd.read_csv(DATA / "hub_triangle.csv.gz", parse_dates=["date"])
    tri = tri[tri.location == location].set_index("date").drop(columns="location").sort_index()
    inc = tri.fillna(0).to_numpy(dtype=float)
    cum = inc[:, :HUB_DMAX + 1].cumsum(axis=1)
    return HubTriangle(tri.index, inc, cum, inc.sum(axis=1), cum[:, -1])


def load_hub_nowcasts() -> pd.DataFrame:
    now = pd.read_csv(DATA / "hub_nowcasts.csv.gz", parse_dates=["forecast_date", "target_end_date"])
    now["h"] = (now.target_end_date - now.forecast_date).dt.days
    q = now[now.type == "quantile"].pivot_table(index=["model", "forecast_date", "target_end_date", "h"],
                                                columns="quantile", values="value")
    m = now[now.type == "mean"].set_index(["model", "forecast_date", "target_end_date", "h"])["value"]
    q = q.join(m.rename("mean"), how="outer").reset_index()
    q.columns = [str(c) for c in q.columns]
    # the point nowcast: the median where a model gave one, else its mean
    q["point"] = q["0.5"].fillna(q["mean"])
    return q


def evaluate_hub(tri: HubTriangle, q: pd.DataFrame) -> pd.DataFrame:
    """Every nowcast with the truth two ways (latest, and 80 days after) and the value known then."""
    q = q.copy()
    t_i = q.target_end_date.map(lambda d: tri.index(d))
    p_i = q.forecast_date.map(lambda d: tri.index(d))
    q["truth"] = [tri.final7(t) for t in t_i]
    q["truth80"] = [tri.final7(t, "80") for t in t_i]
    q["known"] = [tri.known7(t, p) for t, p in zip(t_i, p_i)]
    return q


def chain_ladder_hub(tri: HubTriangle, window: int = 28) -> pd.DataFrame:
    """The Part 2 chain ladder run on the hospitalisation triangle each day of the hub's period.

    It can only estimate up to delay 80, so its target is the count 80 days after.
    """
    rows = []
    C = tri.cum
    for p_day in pd.date_range(HUB_FIRST, HUB_LAST):
        T = tri.index(p_day)
        # rows with delays beyond T - t unknown at T: mask them
        Cm = C[:T + 1].copy()
        for t in range(max(0, T - HUB_DMAX), T + 1):
            Cm[t, T - t + 1:] = np.nan
        est = chain_ladder_daily(Cm, T, window, HUB_DMAX)
        for h in range(0, -29, -1):
            t = T + h
            rows.append((p_day, tri.dates[t], h, float(np.nansum(est[t - 6:t + 1]))))
    out = pd.DataFrame(rows, columns=["forecast_date", "target_end_date", "h", "point"])
    out.insert(0, "model", f"chain ladder ({window}-day window)")
    return out


def score(df: pd.DataFrame, truth: str = "truth") -> pd.DataFrame:
    e = (df["point"] - df[truth]) / df[truth]
    out = {"n": len(df), "bias": e.mean(), "mape": e.abs().mean()}
    if "0.025" in df and df["0.025"].notna().any():
        out["cov95"] = ((df["0.025"] <= df[truth]) & (df[truth] <= df["0.975"])).mean()
    if "0.25" in df and df["0.25"].notna().any():
        out["cov50"] = ((df["0.25"] <= df[truth]) & (df[truth] <= df["0.75"])).mean()
    return pd.Series(out)


# ---------------------------------------------------------------------------
# G. GDP vintages
# ---------------------------------------------------------------------------
GDP_IDS = ("US", "EA", "JP", "CHE")
GDP_NAMES = {"US": "United States", "EA": "euro area", "JP": "Japan", "CHE": "Switzerland"}
GDP_FIRST_Q = pd.Timestamp("2002-07-01")
GDP_LAST_Q = pd.Timestamp("2021-10-01")  # at least three years of later vintages


def gdp_levels(cid: str) -> pd.DataFrame:
    g = pd.read_csv(DATA / "gdp_vintages.csv", parse_dates=["time", "pub_date"])
    return g[g.id == cid].pivot(index="time", columns="pub_date", values="value").sort_index()


def gdp_growth(cid: str) -> pd.DataFrame:
    """Quarter-on-quarter growth, 100 x log difference, in every vintage."""
    return np.log(gdp_levels(cid)).diff() * 100


def releases(growth: pd.DataFrame) -> dict[pd.Timestamp, np.ndarray]:
    """For each quarter, its growth in the first vintage that contained it, the next, and so on."""
    out = {}
    for t in growth.index:
        s = growth.loc[t].dropna()
        if len(s):
            out[t] = s.to_numpy()
    return out


def revision_stats(cid: str) -> dict:
    rel = releases(gdp_growth(cid))
    T = [t for t in rel if GDP_FIRST_Q <= t <= GDP_LAST_Q]
    first = np.array([rel[t][0] for t in T])
    latest = np.array([rel[t][-1] for t in T])

    def at(k):
        return np.array([rel[t][min(k, len(rel[t]) - 1)] for t in T])

    lv = gdp_levels(cid).loc["2000-01-01"]
    jumps = [(lv.index[i], lv.iloc[i] / lv.iloc[i - 1] - 1) for i in range(1, len(lv))
             if abs(lv.iloc[i] / lv.iloc[i - 1] - 1) > 0.02]
    return {
        "quarters": len(T), "first": pd.Series(first, index=T), "latest": pd.Series(latest, index=T),
        "mean_abs_revision": float(np.abs(latest - first).mean()),
        "mean_revision": float((latest - first).mean()),
        "after": {k: float(np.abs(at(k) - first).mean()) for k in (1, 4, 8, 12)},
        "remaining": {k: float(np.abs(latest - at(k)).mean()) for k in (1, 4, 8, 12)},
        "sign_changes": int((np.sign(first) != np.sign(latest)).sum()),
        "negative_first": int((first < 0).sum()), "negative_latest": int((latest < 0).sum()),
        "largest": (T[int(np.argmax(np.abs(latest - first)))], float(np.max(np.abs(latest - first)))),
        "level_jumps": jumps,
        "level_2000q1_first_last": (float(lv.iloc[0]), float(lv.iloc[-1])),
    }


def recession_quarters(series: pd.Series) -> set:
    """Quarters in a run of two or more consecutive negative growth quarters."""
    neg = series < 0
    out = set()
    idx = list(series.index)
    for i in range(1, len(idx)):
        if neg.iloc[i] and neg.iloc[i - 1]:
            out.update({idx[i], idx[i - 1]})
    return out


AS_OF_SQL = """
-- the series as it was known on a given date: for each quarter, the latest vintage published by then
select time, value
from gdp
where id = $id and pub_date <= $as_of
qualify row_number() over (partition by time order by pub_date desc) = 1
order by time
"""

FIRST_RELEASE_SQL = """
-- first-release values: for each quarter, the earliest vintage that contains it
select time, value as first_value, pub_date as first_pub
from gdp
where id = $id
qualify row_number() over (partition by time order by pub_date) = 1
order by time
"""


def gdp_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("set threads = 1")
    con.execute(f"create table gdp as select * from read_csv('{DATA / 'gdp_vintages.csv'}', header = true, "
                "columns = {'id': 'VARCHAR', 'time': 'DATE', 'pub_date': 'DATE', 'value': 'DOUBLE'})")
    return con
