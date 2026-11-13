"""Part 7: the two conjectures of Parts 4 and 6 checked against data the series had not used.

A. The RKI files of 2022 to 2026 on the days that were holidays only for the institute (Berlin) or days
   many people do not work (Christmas Eve, New Year's Eve): did the file of the next day bring anything?
B. US states in the Johns Hopkins daily reports: is a state's file unchanged more often on the state's own
   legal holidays than on federal holidays the state does not observe?
C. The aggregator: did Johns Hopkins' own calendar (US holidays, the Christmas break) leave unchanged files
   in the series of other countries?
D. Timestamps: did a state row whose count had not changed still carry a new Last_Update?
E. Unchanged files and the largest errors: the 7-day count as first published against the final version
   (US states), and against the RKI's count by reporting date (Germany).

A file is "unchanged" for a state or country when its cumulative count equals the count of the day before
as the pipeline had it when the file landed. Only weekdays (Monday to Friday) in periods when the state or
country updated its count on ordinary weekdays are compared ("daily regime"): the share of ordinary weekdays
with an unchanged file in the 57 days centred on the date is at most 10%.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PART3 = HERE.parent / "part3" / "data"
PART6 = HERE.parent / "part6" / "data"

THRESHOLD = 0.10      # at most this share of ordinary weekdays unchanged: the daily regime
HALF_WINDOW = 28      # days on each side of the date for that share
MIN_DAYS = 15         # ordinary weekdays needed in the window
MIN_WEEKLY = 200      # smallest final 7-day count for which a relative error is computed


# ---------------------------------------------------------------------------- loading

def holidays() -> dict[str, dict[dt.date, str]]:
    h = pd.read_csv(DATA / "holidays.csv.gz", parse_dates=["date"])
    out: dict[str, dict] = {}
    for cal, g in h.groupby("calendar"):
        out[cal] = dict(zip(g.date.dt.date, g.name))
    return out


def unchanged(new: pd.Series, before: pd.Series) -> np.ndarray:
    """1.0 if the new count equals the count of the day before, 0.0 if not, NaN if either is missing."""
    a, b = new.astype("float64").to_numpy(), before.astype("float64").to_numpy()
    return np.where(np.isnan(a) | np.isnan(b), np.nan, (a == b).astype(float))


def us_states() -> pd.DataFrame:
    df = pd.read_csv(DATA / "jhu_us_states.csv.gz", parse_dates=["date"])
    df["flat"] = unchanged(df["first"], df.prev_at_landing)
    df["wd"] = df.date.dt.dayofweek
    return df


def countries() -> pd.DataFrame:
    df = pd.read_csv(DATA / "jhu_countries.csv.gz", parse_dates=["date"])
    df["flat"] = unchanged(df["first"], df.prev_in_first)
    df["wd"] = df.date.dt.dayofweek
    return df


STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


# ---------------------------------------------------------------------------- the daily regime

def regime_base(df: pd.DataFrame, group: str, ordinary: pd.Series, half_window: int = HALF_WINDOW,
                min_days: int = MIN_DAYS) -> pd.Series:
    """Share of ordinary weekdays with an unchanged file in the window centred on each row's date."""
    x = df.assign(_o=np.where(ordinary & (df.wd < 5), df.flat.astype(float), np.nan))
    out = pd.Series(np.nan, index=df.index)
    for _, g in x.groupby(group, sort=False):
        g = g.sort_values("date")
        r = g.set_index("date")._o.rolling(f"{2 * half_window + 1}D", center=True, min_periods=min_days).mean()
        out.loc[g.index] = r.values
    return out


# ---------------------------------------------------------------------------- B. US states

def us_day_types(us: pd.DataFrame, hol: dict) -> pd.DataFrame:
    fed = hol["US"]
    fed_dates = set(fed)
    df = us.copy()
    d = df.date.dt.date
    state_cal = {s: hol[f"US-{c}"] for s, c in STATE_CODES.items()}
    df["fed"] = [x in fed_dates for x in d]
    df["state_hol"] = [x in state_cal[s] for s, x in zip(df.state, d)]
    yday = d - dt.timedelta(days=1)
    df["fed_prev"] = [x in fed_dates for x in yday]
    df["state_prev"] = [x in state_cal[s] for s, x in zip(df.state, yday)]
    df["eve"] = pd.Series([(x.month, x.day) in ((12, 24), (12, 31)) for x in d], index=df.index) & ~df.state_hol
    df["hol_name"] = [state_cal[s].get(x) or fed.get(x) for s, x in zip(df.state, d)]
    df["day"] = np.select(
        [df.fed & df.state_hol, df.state_hol & ~df.fed, df.fed & ~df.state_hol, df.eve,
         (df.fed_prev | df.state_prev)],
        ["federal and state holiday", "state holiday only", "federal holiday, state open",
         "Christmas Eve or New Year's Eve", "day after a holiday"], "ordinary")
    return df


def us_daily_regime(df: pd.DataFrame, threshold: float = THRESHOLD, half_window: int = HALF_WINDOW) -> pd.DataFrame:
    df = df.copy()
    df["base"] = regime_base(df, "state", df.day.eq("ordinary"), half_window)
    return df[(df.wd < 5) & (df.base <= threshold) & df.flat.notna()]


def rates(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by).flat.agg(["sum", "size"])
    g["rate"] = g["sum"] / g["size"]
    return g


def within_date(dr: pd.DataFrame) -> pd.DataFrame:
    """Dates on which some states in the daily regime had a legal holiday and others did not."""
    rows = []
    for d, g in dr.groupby("date"):
        obs, non = g[g.state_hol], g[~g.state_hol]
        if len(obs) and len(non):
            label = obs.hol_name.mode().iloc[0]
            rows.append((d, label, int(obs.flat.sum()), len(obs), int(non.flat.sum()), len(non)))
    return pd.DataFrame(rows, columns=["date", "holiday", "obs_flat", "obs_n", "non_flat", "non_n"])


# ---------------------------------------------------------------------------- C. countries

def country_day_types(c: pd.DataFrame, hol: dict) -> pd.DataFrame:
    df = c[c.country != "US"].copy()
    df = df[df.country.isin(hol)]
    fed = set(hol["US"])
    d = df.date.dt.date
    df["own"] = [x in hol[k] for k, x in zip(df.country, d)]
    df["own_prev"] = [(x - dt.timedelta(days=1)) in hol[k] for k, x in zip(df.country, d)]
    df["us"] = [x in fed for x in d]
    df["us_name"] = [hol["US"].get(x) for x in d]
    df["break"] = [(x.month == 12 and x.day >= 24) or (x.month == 1 and x.day == 2) for x in d]
    df["day"] = np.select(
        [df.own, df.own_prev, df.us, df["break"]],
        ["own public holiday", "day after own holiday", "US federal holiday only", "24-31 Dec or 2 Jan"],
        "ordinary")
    df["base"] = regime_base(df, "country", df.day.eq("ordinary"))
    return df


# ---------------------------------------------------------------------------- E. errors

def us_errors(us: pd.DataFrame) -> pd.DataFrame:
    df = us.sort_values(["state", "date"]).copy()
    final = df.set_index(["state", "date"]).final
    week_before = pd.MultiIndex.from_arrays([df.state, df.date - pd.Timedelta(days=7)])
    day_before = pd.MultiIndex.from_arrays([df.state, df.date - pd.Timedelta(days=1)])
    df["final_week"] = final.reindex(week_before).values
    df["final_prev"] = final.reindex(day_before).values
    df["pub7"] = (df["first"] - df.week_at_landing).astype(float)
    df["fin7"] = (df.final - df.final_week).astype(float)
    df["still_flat"] = df.final.eq(df.final_prev)
    ok = df.pub7.notna() & df.fin7.notna() & (df.fin7 >= MIN_WEEKLY) & df.flat.notna()
    df = df[ok].copy()
    df["rel"] = (df.pub7 - df.fin7) / df.fin7
    return df


def worst(df: pd.DataFrame, q: float) -> tuple[float, pd.DataFrame]:
    thr = df.rel.abs().quantile(q)
    return thr, df[df.rel.abs() >= thr]


def germany(c: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Germany as Johns Hopkins first published it, against the RKI's final count by reporting date.

    The count Johns Hopkins published for date d matches best the RKI file published on d+1 (see the
    alignment check in the numbers script). That file's new cases were mostly reported to health offices a
    day or more before, so the 7-day count of d is compared with the RKI's final count for the seven
    reporting dates ending on d - lag; lag 1 is the main comparison and lag 0 is shown beside it.
    """
    de = c[c.country == "Germany"].set_index("date").sort_index().copy()
    snap = pd.read_csv(PART3 / "rki_snapshots.csv.gz", parse_dates=["publication", "date"])
    final = snap[snap.publication == snap.publication.max()].set_index("date").cases.sort_index()
    cum = final.reindex(pd.date_range("2020-01-01", "2023-12-31")).fillna(0).cumsum()
    end = de.index - pd.Timedelta(days=lag)
    de["pub7"] = (de["first"] - de.week_in_first).astype(float)
    de["truth7"] = cum.reindex(end).values - cum.reindex(end - pd.Timedelta(days=7)).values
    de["rel"] = (de.pub7 - de.truth7) / de.truth7
    de["flat"] = de["flat"].fillna(0).astype(bool)
    de["final_flat"] = de.final.eq(de.final.shift(1))
    de["shifted"] = de.final.eq(de["first"].shift(1)) & de.final.ne(de["first"])
    return de


def rki_increments() -> pd.Series:
    """Cases new in each RKI file (the archive's NeuerFall flag), 2021-04-02 to 2026-11-10."""
    a = pd.read_csv(PART3 / "rki_melde.csv.gz", parse_dates=["publication"])
    b = pd.read_csv(PART6 / "rki_melde_2024_2026.csv.gz", parse_dates=["publication"])
    a = a[a.publication < "2024-01-01"]
    inc = pd.concat([a, b]).groupby("publication").new.sum()
    daily = inc.index.to_series().diff().dt.days.eq(1) | inc.index.to_series().diff(-1).dt.days.eq(-1)
    return inc[daily]
