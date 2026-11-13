"""Part 8: does scoring against a count dated by publication hide the errors around holidays?

Three datings of the same German cases are compared:
  publication  the day the RKI published the case (the forecast hub's truth, and the dating of most aggregators)
  reporting    the day the local health office was notified (Meldedatum)
  reference    the onset of symptoms where known, otherwise the reporting date (Refdatum)
and two kinds of number are scored against them: the 7-day count Johns Hopkins published for Germany each
morning (from Part 7), and the weekly case forecasts of the German and Polish COVID-19 Forecast Hub.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PART7 = HERE.parent / "part7" / "data"

# hub location codes, RKI state ids and python-holidays subdivisions
STATES = {
    "GM01": (8, "BW", "Baden-Württemberg"), "GM02": (9, "BY", "Bavaria"), "GM03": (4, "HB", "Bremen"),
    "GM04": (2, "HH", "Hamburg"), "GM05": (6, "HE", "Hesse"), "GM06": (3, "NI", "Lower Saxony"),
    "GM07": (5, "NW", "North Rhine-Westphalia"), "GM08": (7, "RP", "Rhineland-Palatinate"),
    "GM09": (10, "SL", "Saarland"), "GM10": (1, "SH", "Schleswig-Holstein"), "GM11": (12, "BB", "Brandenburg"),
    "GM12": (13, "MV", "Mecklenburg-Vorpommern"), "GM13": (14, "SN", "Saxony"), "GM14": (15, "ST", "Saxony-Anhalt"),
    "GM15": (16, "TH", "Thuringia"), "GM16": (11, "BE", "Berlin"),
}
TRUTHS = ("publication", "reporting", "reference")
FIRST_WEEK, LAST_WEEK = pd.Timestamp("2020-05-09"), pd.Timestamp("2022-11-12")
BOOT = 2000
SEED = 20261113


# ---------------------------------------------------------------------------- loading

def holidays() -> dict[str, dict[dt.date, str]]:
    h = pd.read_csv(DATA / "holidays_de.csv.gz", parse_dates=["date"])
    return {cal: dict(zip(g.date.dt.date, g.name)) for cal, g in h.groupby("calendar")}


def calendar(hol: dict, location: str) -> dict[dt.date, str]:
    return hol["DE"] if location == "GM" else hol[f"DE-{STATES[location][1]}"]


def daily() -> pd.DataFrame:
    """Daily cases by location (GM and the states) under the three datings, plus cases with a known onset."""
    pub = pd.read_csv(DATA / "hub_truth.csv.gz", parse_dates=["date"])
    pub = pub.pivot_table(index="date", columns="location", values="value", aggfunc="sum")
    r = pd.read_csv(DATA / "rki_dates.csv.gz", parse_dates=["reporting_date", "reference_date"])
    loc = {v[0]: k for k, v in STATES.items()}
    r["location"] = r.state.map(loc)
    days = pd.date_range("2020-01-01", "2023-02-28")
    out = []
    for name, col, sub in (("reporting", "reporting_date", r), ("reference", "reference_date", r),
                           ("onset", "reference_date", r[r.onset_known == 1])):
        by = sub.pivot_table(index=col, columns="location", values="cases", aggfunc="sum").reindex(days).fillna(0)
        by["GM"] = by.sum(axis=1)
        out.append(by.stack().rename(name))
    pubs = pub.reindex(days).stack().rename("publication")
    d = pd.concat([pubs, *out], axis=1).reset_index()
    d.columns = ["date", "location", "publication", "reporting", "reference", "onset"]
    return d


def weekly(d: pd.DataFrame) -> pd.DataFrame:
    """Weekly sums, Sunday to Saturday, labelled by the Saturday that ends the week (the hub's convention)."""
    w = (d.set_index("date").groupby("location")[["publication", "reporting", "reference", "onset"]]
         .resample("W-SAT").sum(min_count=7).reset_index().rename(columns={"date": "week"}))
    return w[(w.week >= FIRST_WEEK) & (w.week <= LAST_WEEK)]


def forecasts() -> pd.DataFrame:
    f = pd.read_csv(DATA / "hub_forecasts.csv.gz", parse_dates=["forecast_date", "target_end_date"])
    return f.rename(columns={"target_end_date": "week"})


# ---------------------------------------------------------------------------- holidays at the weekly level

def weekday_holidays(cal: dict, week: pd.Timestamp) -> list[str]:
    days = [week - pd.Timedelta(days=k) for k in range(7)]
    return [cal[x.date()] for x in days if x.date() in cal and x.dayofweek < 5]


def week_types(w: pd.DataFrame, hol: dict) -> pd.DataFrame:
    """National: a weekday national holiday in the week. Regional: a weekday holiday of the state that is
    not national. Previous: the week before had one of either."""
    nat = hol["DE"]
    out = w.copy()
    out["national"] = [bool(weekday_holidays(nat, x)) for x in out.week]
    out["regional"] = [loc != "GM" and any(
        n not in nat.values() for n in weekday_holidays(calendar(hol, loc), x)) for loc, x in zip(out.location, out.week)]
    out["regional_name"] = [", ".join(sorted({n for n in weekday_holidays(calendar(hol, loc), x)
                                              if n not in nat.values()})) if loc != "GM" else ""
                            for loc, x in zip(out.location, out.week)]
    return out


def score(f: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    s = f.merge(w, on=["location", "week"], how="inner")
    ok = (s["median"] > 0) & (s[list(TRUTHS)] > 0).all(axis=1)
    s = s[ok].copy()
    for t in TRUTHS:
        s[f"e_{t}"] = np.abs(np.log(s["median"] / s[t]))
        if "q0100" in s:
            s[f"c80_{t}"] = np.where(s.q0100.notna(), ((s.q0100 <= s[t]) & (s[t] <= s.q0900)).astype(float), np.nan)
    return s


# ---------------------------------------------------------------------------- the daily view of holidays

def holiday_ratios(d: pd.DataFrame, location: str, days: list[pd.Timestamp]) -> pd.DataFrame:
    """For each holiday, the count on the day (the next day for publication, which lags) against the mean of
    the same weekday one week before and after."""
    x = d[d.location == location].set_index("date")
    rows = []
    for h in days:
        r = {"holiday": h}
        for col, shift in (("publication", 1), ("reporting", 0), ("reference", 0), ("onset", 0)):
            t = h + pd.Timedelta(days=shift)
            ref = (x[col].get(t - pd.Timedelta(days=7), np.nan) + x[col].get(t + pd.Timedelta(days=7), np.nan)) / 2
            r[col] = x[col].get(t, np.nan) / ref if ref > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- the daily pipeline (Johns Hopkins)

def jhu_mornings(d: pd.DataFrame, hol: dict, lag: int) -> pd.DataFrame:
    c = pd.read_csv(PART7 / "jhu_countries.csv.gz", parse_dates=["date"])
    de = c[c.country == "Germany"].set_index("date").sort_index()
    g = d[d.location == "GM"].set_index("date")
    out = pd.DataFrame(index=de.index)
    out["first"] = (de["first"] - de.week_in_first).astype(float)
    out["final"] = (de.final - de.final.shift(7)).astype(float)
    end = de.index - pd.Timedelta(days=lag)
    for col in ("reporting", "reference"):
        cum = g[col].cumsum()
        out[col] = cum.reindex(end).values - cum.reindex(end - pd.Timedelta(days=7)).values
    days = [pd.Timestamp(x) for x in hol["DE"]]
    out["holiday"] = [any(h in (x, x - pd.Timedelta(days=1)) for h in days) for x in out.index]
    out["which"] = [next((pd.Timestamp(h) for h in days if pd.Timestamp(h) in (x, x - pd.Timedelta(days=1))), pd.NaT)
                    for x in out.index]
    out = out[(out.index >= "2020-10-01") & (out.index <= "2023-02-28")].dropna(subset=["first", "final", "reporting"])
    out = out[(out.final > 0) & (out.reporting > 0) & (out["first"] > 0)]
    for t in ("final", "reporting", "reference"):
        out[f"e_{t}"] = np.abs(np.log(out["first"] / out[t]))
    return out


# ---------------------------------------------------------------------------- bootstrap by cluster

def cluster_boot(values: pd.Series, groups: pd.Series, stat, n: int = BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% interval of stat(sample) when whole clusters (weeks, holidays) are resampled with replacement."""
    rng = np.random.default_rng(seed)
    keys = groups.unique()
    idx = {k: np.flatnonzero(groups.to_numpy() == k) for k in keys}
    vals = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([idx[k] for k in pick])
        vals.append(stat(values.iloc[rows]))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def neighbours(week: pd.Timestamp, holiday_weeks: set, reach: int = 3) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """The nearest week before and after that has no weekday national holiday, at most `reach` weeks away."""
    before = next((week - pd.Timedelta(days=7 * k) for k in range(1, reach + 1)
                   if week - pd.Timedelta(days=7 * k) not in holiday_weeks), None)
    after = next((week + pd.Timedelta(days=7 * k) for k in range(1, reach + 1)
                  if week + pd.Timedelta(days=7 * k) not in holiday_weeks), None)
    return before, after
