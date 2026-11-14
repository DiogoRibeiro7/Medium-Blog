"""Part 9: is a producer's calendar, learned from the history of its unchanged files, a better guide to its
next gaps than a published holiday calendar? And when that calendar changes, does it change at a point or
drift?

Producers are the rows of two Johns Hopkins files, read as Part 7 read them (each file as first published):
  US states   the 50 states and the District of Columbia in the daily US reports (Part 7's extract);
  provinces   the provinces and territories of Canada, the states and territories of Australia, and Hong
              Kong, which are separate rows of the global time series (this part's extract);
  countries   every other country with a row of its own in the global time series and a calendar in
              python-holidays (Part 7's extract, plus the country rows of Denmark, France, the Netherlands,
              New Zealand and the United Kingdom from this part's extract).
A producer's file is unchanged on a date when its cumulative count equals the count of the day before, as
the pipeline had it when the file landed (Part 7's definition). A producer-day counts only if the producer
was still reporting (on or before the last date its count changed) and was not tiny: the 7-day count in the
28 days before was at least 800 (about 200 a week), so that an unchanged file is unlikely to be a day
without cases. The 28 days are read from the files as first published: the producer's count in the previous
day's file minus its count in the file of 29 days before. A window of 28 days, rather than 7, keeps producers
that report every week or every other week in the count on the days between their reports.

Every prediction for a date uses only producer-days before it. The models:
  published   the published calendar: one rate for each kind of day (working day, legal weekend, legal
              holiday on a working day, working day after such a holiday), pooled over the producers of a
              group in a trailing window;
  weekdays    the producer's own rate on that day of the week in a trailing window, holidays ignored;
  weekdays + published holidays
              the weekday rate with the group's holiday effect added on the log-odds scale;
  learned     the weekday rate with the producer's own holiday effect, shrunk towards the group's.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PART7 = HERE.parent / "part7" / "data"

GROUPS = ("US states", "provinces", "countries")
MIN_COUNT = 800                      # smallest count over the 28 days before a producer-day (about 200 a week)
FIRST_TEST = pd.Timestamp("2021-04-01")
YEARS = {"2020-21": ("2020-04-01", "2021-03-31"), "2021-22": ("2021-04-01", "2022-03-31"),
         "2022-23": ("2022-04-01", "2023-03-09")}
WINDOW = 56                          # days in the trailing window of the weekday rates
HOLIDAY_WINDOW = 365                 # days in the window of the holiday effects
PRIOR = 2.0                          # pseudo-observations pulling a producer's weekday rate to its group's
TAU = 1.0                            # prior standard deviation of a producer's holiday effect around its group's
CLIP = 0.005                         # probabilities are kept in [CLIP, 1 - CLIP] for the log loss
BOOT = 2000
SEED = 20261114

SPLIT_COUNTRY_ROWS = ("Denmark", "France", "Netherlands", "New Zealand", "United Kingdom")
CANADA = {"Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB", "New Brunswick": "NB",
          "Newfoundland and Labrador": "NL", "Northwest Territories": "NT", "Nova Scotia": "NS", "Nunavut": "NU",
          "Ontario": "ON", "Prince Edward Island": "PE", "Quebec": "QC", "Saskatchewan": "SK", "Yukon": "YT"}
AUSTRALIA = {"Australian Capital Territory": "ACT", "New South Wales": "NSW", "Northern Territory": "NT",
             "Queensland": "QLD", "South Australia": "SA", "Tasmania": "TAS", "Victoria": "VIC",
             "Western Australia": "WA"}
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
KINDS = ("working", "weekend", "holiday", "after")


# ---------------------------------------------------------------------------- loading

def calendars() -> dict[str, dict[dt.date, str]]:
    h = pd.concat([pd.read_csv(PART7 / "holidays.csv.gz"), pd.read_csv(DATA / "calendars.csv.gz")])
    h["date"] = pd.to_datetime(h.date)
    return {cal: dict(zip(g.date.dt.date, g.name)) for cal, g in h.groupby("calendar")}


def weekends() -> dict[tuple[str, int], frozenset]:
    w = pd.read_csv(DATA / "weekends.csv", dtype={"weekend": str})
    return {(c, int(y)): frozenset(int(x) for x in s) for c, y, s in zip(w.calendar, w.year, w.weekend)}


def unchanged(new: pd.Series, before: pd.Series) -> np.ndarray:
    a, b = new.astype("float64").to_numpy(), before.astype("float64").to_numpy()
    return np.where(np.isnan(a) | np.isnan(b), np.nan, (a == b).astype(float))


def producers() -> pd.DataFrame:
    """One row per producer and date: group, producer, calendar, counts and the unchanged flag."""
    cols = ["group", "producer", "cal", "date", "first", "prev", "week"]
    us = pd.read_csv(PART7 / "jhu_us_states.csv.gz")
    us = us.rename(columns={"state": "producer", "prev_at_landing": "prev", "week_at_landing": "week"})
    us["group"], us["cal"] = "US states", "US-" + us.producer.map(STATE_CODES)

    hol = calendars()
    split = {"Australia", "Canada", "China", *SPLIT_COUNTRY_ROWS}
    c = pd.read_csv(PART7 / "jhu_countries.csv.gz", keep_default_na=False, na_values=[""])
    c = c[~c.country.isin(split | {"US"}) & c.country.isin(hol)]
    c = c.rename(columns={"country": "producer", "prev_in_first": "prev", "week_in_first": "week"})
    c["group"], c["cal"] = "countries", c.producer
    c["producer"] = c.producer.replace({"Georgia": "Georgia (country)"})  # the US state keeps the plain name

    g = pd.read_csv(DATA / "jhu_global_rows.csv.gz", keep_default_na=False, na_values=[""])
    g = g.rename(columns={"prev_in_first": "prev", "week_in_first": "week"})
    whole = g[g.country.isin(SPLIT_COUNTRY_ROWS) & g.province.isna()].copy()
    whole["producer"], whole["group"], whole["cal"] = whole.country, "countries", whole.country
    sub = g[g.province.isin(set(CANADA) | set(AUSTRALIA) | {"Hong Kong"})].copy()
    sub = sub[~((sub.country == "China") & (sub.province != "Hong Kong"))]
    sub["producer"] = sub.province
    sub["group"] = "provinces"
    sub["cal"] = np.where(sub.country == "Canada", "CA-" + sub.province.map(CANADA),
                          np.where(sub.country == "Australia", "AU-" + sub.province.map(AUSTRALIA), "HK"))
    df = pd.concat([us[cols], c[cols], whole[cols], sub[cols]], ignore_index=True)
    df["date"] = pd.to_datetime(df.date)
    if df.duplicated(["producer", "date"]).any():
        raise ValueError("producer names must be unique")
    df["flat"] = unchanged(df["first"], df.prev)
    return df.sort_values(["group", "producer", "date"]).reset_index(drop=True)


def frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adds eligibility and the kind of day under the published calendar."""
    df = producers() if df is None else df.copy()
    hol, wk = calendars(), weekends()
    first = df.set_index(["producer", "date"])["first"].astype("float64")
    before = lambda k: first.reindex(pd.MultiIndex.from_arrays([df.producer, df.date - pd.Timedelta(days=k)])).to_numpy()  # noqa: E731
    df["count28"] = before(1) - before(29)
    last = df[df.flat == 0].groupby("producer").date.max()
    df["active"] = df.date <= df.producer.map(last)
    df["eligible"] = df.flat.notna() & df.active & (df.count28 >= MIN_COUNT)
    df["wd"] = df.date.dt.dayofweek
    d = df.date.dt.date
    y = df.date.dt.year
    df["weekend"] = [w in wk[(c, yy)] for c, yy, w in zip(df.cal, y, df.wd)]
    df["hol_name"] = [hol[c].get(x) for c, x in zip(df.cal, d)]
    yday = d - dt.timedelta(days=1)
    yday_working = [((x.weekday()) not in wk[(c, x.year)]) and (x in hol[c]) for c, x in zip(df.cal, yday)]
    df["kind"] = np.select([df.weekend, df.hol_name.notna(), yday_working],
                           ["weekend", "holiday", "after"], "working")
    df["year"] = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, (a, b) in YEARS.items():
        df.loc[(df.date >= a) & (df.date <= b), "year"] = name
    return df


# ---------------------------------------------------------------------------- windows

class Grid:
    """Producer x key x day arrays of counts, for windowed sums by cumulative differences."""

    def __init__(self, df: pd.DataFrame, key: str, keys: list, rows: pd.Series):
        self.df = df
        self.dates = pd.date_range(df.date.min(), df.date.max())
        self.t = ((df.date - self.dates[0]).dt.days).to_numpy()
        self.prods = df.producer.unique()
        self.pi = pd.Series(np.arange(len(self.prods)), index=self.prods)[df.producer].to_numpy()
        self.ki = pd.Series(np.arange(len(keys)), index=keys).reindex(df[key]).to_numpy()
        self.groups = df.drop_duplicates("producer").set_index("producer").group.reindex(self.prods).to_numpy()
        P, K, T = len(self.prods), len(keys), len(self.dates)
        Y, N = np.zeros((P, K, T)), np.zeros((P, K, T))
        m = rows.to_numpy() & ~np.isnan(self.ki.astype(float))
        np.add.at(Y, (self.pi[m], self.ki[m].astype(int), self.t[m]), df.flat.to_numpy()[m])
        np.add.at(N, (self.pi[m], self.ki[m].astype(int), self.t[m]), 1)
        z = np.zeros((P, K, 1))
        self.cY, self.cN = np.concatenate([z, Y.cumsum(2)], 2), np.concatenate([z, N.cumsum(2)], 2)
        self.gY = {g: self.cY[self.groups == g].sum(0) for g in GROUPS}
        self.gN = {g: self.cN[self.groups == g].sum(0) for g in GROUPS}

    def sums(self, sel: np.ndarray, kidx: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        """Producer and group sums over days [lo, hi) for the rows sel, keys kidx."""
        p = self.pi[sel]
        s = self.cY[p, kidx, hi] - self.cY[p, kidx, lo]
        n = self.cN[p, kidx, hi] - self.cN[p, kidx, lo]
        S, N = np.zeros(len(p)), np.zeros(len(p))
        g = self.groups[p]
        for name in GROUPS:
            m = g == name
            S[m] = self.gY[name][kidx[m], hi[m]] - self.gY[name][kidx[m], lo[m]]
            N[m] = self.gN[name][kidx[m], hi[m]] - self.gN[name][kidx[m], lo[m]]
        return s, n, S, N


def shrunk(s, n, S, N, prior: float = PRIOR):
    """Producer rate pulled towards the group rate by `prior` pseudo-observations."""
    p0 = (S + 0.5) / (N + 1.0)
    return (s + prior * p0) / (n + prior), p0


def weekday_rates(df: pd.DataFrame, window: int = WINDOW, prior: float = PRIOR, fixed: tuple | None = None):
    """The producer's rate on the same day of the week, from working and weekend days only, in the trailing
    window (or, with fixed=(start, end), in a fixed training period). Returns (rate, group rate)."""
    train = df.eligible & df.kind.isin(["working", "weekend"])
    grid = Grid(df, "wd", list(range(7)), train)
    sel = np.ones(len(df), bool)
    if fixed is None:
        hi = grid.t
        lo = np.maximum(hi - window, 0)
    else:
        a, b = ((pd.Timestamp(x) - grid.dates[0]).days for x in fixed)
        lo, hi = np.full(len(df), a), np.full(len(df), b + 1)
    s, n, S, N = grid.sums(sel, grid.ki.astype(int), lo, hi)
    r, p0 = shrunk(s, n, S, N, prior)
    return pd.Series(r, index=df.index), pd.Series(p0, index=df.index), pd.Series(n, index=df.index)


def published(df: pd.DataFrame, window: int = WINDOW, fixed: tuple | None = None) -> pd.Series:
    """Group rate for the kind of day under the published calendar: working and weekend days over the
    trailing window, holidays and days after holidays over HOLIDAY_WINDOW days."""
    grid = Grid(df, "kind", list(KINDS), df.eligible)
    k = grid.ki.astype(int)
    if fixed is None:
        hi = grid.t
        w = np.where(np.isin(df.kind.to_numpy(), ["holiday", "after"]), HOLIDAY_WINDOW, window)
        lo = np.maximum(hi - w, 0)
    else:
        a, b = ((pd.Timestamp(x) - grid.dates[0]).days for x in fixed)
        lo, hi = np.full(len(df), a), np.full(len(df), b + 1)
    _, _, S, N = grid.sums(np.ones(len(df), bool), k, lo, hi)
    return pd.Series((S + 0.5) / (N + 1.0), index=df.index)


# ---------------------------------------------------------------------------- holiday effects

def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def offset_fit(base: np.ndarray, y: np.ndarray, centre: float, sd: float, steps: int = 30) -> float:
    """MAP estimate of d in logit P(y) = base + d, with a normal prior N(centre, sd^2) on d."""
    d = centre
    for _ in range(steps):
        p = expit(base + d)
        g = np.sum(y - p) - (d - centre) / sd ** 2
        h = -np.sum(p * (1 - p)) - 1 / sd ** 2
        step = np.clip(g / h, -2, 2)
        d -= step
        if abs(step) < 1e-8:
            break
    return d


def holiday_offsets(df: pd.DataFrame, base: pd.Series, tau: float = TAU, window: int = HOLIDAY_WINDOW,
                    fixed: tuple | None = None) -> pd.DataFrame:
    """For each eligible holiday and day after a holiday: the group's effect (d0) and the producer's (dp),
    on the log-odds scale relative to `base`, from that kind of day in the past `window` days of the group
    (or in a fixed training period)."""
    h = df[df.eligible & df.kind.isin(["holiday", "after"])].copy()
    h["base"] = logit(base[h.index].to_numpy())
    out = []
    for (grp, kind), g in h.groupby(["group", "kind"]):
        g = g.sort_values("date")
        dates, prods = g.date.to_numpy(), g.producer.to_numpy()
        b, y = g.base.to_numpy(), g.flat.to_numpy()
        cache = {}
        for i, (idx, d, p) in enumerate(zip(g.index, dates, prods)):
            if fixed is None:
                lo, hi = d - np.timedelta64(window, "D"), d
            else:
                lo, hi = np.datetime64(fixed[0]), np.datetime64(fixed[1]) + np.timedelta64(1, "D")
            past = (dates >= lo) & (dates < hi)
            key = (lo, hi)
            if key not in cache:
                cache[key] = offset_fit(b[past], y[past], 0.0, 3.0) if past.any() else 0.0
            d0 = cache[key]
            mine = past & (prods == p)
            dp = offset_fit(b[mine], y[mine], d0, tau) if mine.any() else d0
            out.append((idx, d0, dp, int(mine.sum()), int(past.sum())))
    return pd.DataFrame(out, columns=["idx", "d0", "dp", "n_own", "n_group"]).set_index("idx")


# ---------------------------------------------------------------------------- the four models

def predictions(df: pd.DataFrame, window: int = WINDOW, tau: float = TAU, fixed: tuple | None = None,
                base: pd.Series | None = None, prior: float = PRIOR) -> pd.DataFrame:
    """P(unchanged) under each model, for every eligible producer-day. `base` replaces the windowed weekday
    rate (for instance by the change-point estimate) when given."""
    wkd, _, _ = weekday_rates(df, window, prior, fixed=fixed)
    if base is not None:
        wkd = base.reindex(df.index).fillna(wkd)
    pub = published(df, window, fixed=fixed)
    off = holiday_offsets(df, wkd, tau, fixed=fixed)
    e = df[df.eligible].copy()
    e["published"] = pub[e.index]
    e["weekdays"] = wkd[e.index]
    lo = logit(e.weekdays.to_numpy())
    d0 = off.d0.reindex(e.index).fillna(0).to_numpy()
    dp = off.dp.reindex(e.index).fillna(0).to_numpy()
    hk = e.kind.isin(["holiday", "after"]).to_numpy()
    e["weekdays + published holidays"] = np.where(hk, expit(lo + d0), e.weekdays)
    e["learned"] = np.where(hk, expit(lo + dp), e.weekdays)
    e["d0"], e["dp"] = d0, dp
    return e


MODELS = ("published", "weekdays", "weekdays + published holidays", "learned")


def logloss(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(y, p) -> np.ndarray:
    return (np.asarray(y, float) - np.asarray(p, float)) ** 2


def cluster_boot(values: pd.Series, groups: pd.Series, stat=np.mean, n: int = BOOT, seed: int = SEED):
    """95% interval of stat when whole producers are resampled with replacement."""
    rng = np.random.default_rng(seed)
    v = values.to_numpy()
    codes, keys = pd.factorize(groups)
    sums = np.bincount(codes, weights=v, minlength=len(keys))
    cnts = np.bincount(codes, minlength=len(keys))
    out = []
    if stat is np.mean:
        for _ in range(n):
            pick = rng.integers(0, len(keys), len(keys))
            out.append(sums[pick].sum() / cnts[pick].sum())
    else:
        idx = [np.flatnonzero(codes == k) for k in range(len(keys))]
        for _ in range(n):
            pick = rng.integers(0, len(keys), len(keys))
            out.append(stat(v[np.concatenate([idx[k] for k in pick])]))
    return tuple(np.percentile(out, [2.5, 97.5]))


# ---------------------------------------------------------------------------- other estimators of the weekday rate

def weekday_series(df: pd.DataFrame):
    """For each producer and day of the week, the dated outcomes of its training days (working and weekend
    days), oldest first, with the index of each row."""
    train = df[df.eligible & df.kind.isin(["working", "weekend"])]
    for (p, w), g in train.groupby(["producer", "wd"], sort=False):
        yield p, w, g.index.to_numpy(), g.date.to_numpy(), g.flat.to_numpy()


def ewma_rates(df: pd.DataFrame, half_life: float, pooled: pd.Series, prior: float = PRIOR) -> pd.Series:
    """Exponentially weighted rate of the same weekday, weights halving every `half_life` days, pulled
    towards the group rate `pooled` (taken at the prediction date) by `prior` pseudo-observations. The
    prediction for a row uses only earlier rows."""
    out = {}
    for _, _, idx, dates, y in weekday_series(df):
        t = (dates - dates[0]) / np.timedelta64(1, "D")
        s = n = 0.0
        last = None
        for i, ti, yi in zip(idx, t, y):
            if last is not None:
                f = 0.5 ** ((ti - last) / half_life)
                s, n = s * f, n * f
            p0 = pooled[i]
            out[i] = (s + prior * p0) / (n + prior)
            s, n, last = s + yi, n + 1, ti
    return pd.Series(out)


def bocpd_rates(df: pd.DataFrame, hazard: float, pooled: pd.Series, prior: float = PRIOR,
                max_run: int = 300) -> pd.Series:
    """Online Bayesian change-point detection (Adams and MacKay, 2007) on each producer's series of one day
    of the week: the rate is constant between change points, a change point occurs before each observation
    with probability `hazard`, and each new run starts from a Beta prior centred on the group rate with
    `prior` pseudo-observations. The training days are working and weekend days; holidays and the days
    after them get the prediction of the state reached before them but do not update it. A prediction uses
    only earlier days."""
    out = {}
    e = df[df.eligible]
    for (p, w), g in e.groupby(["producer", "wd"], sort=False):
        g = g.sort_values("date")
        r, s, n = np.array([1.0]), np.array([0.0]), np.array([0.0])
        for i, yi, k in zip(g.index, g.flat.to_numpy(), g.kind.to_numpy()):
            p0 = pooled[i]
            pred = (s + prior * p0) / (n + prior)
            out[i] = float(r @ pred)
            if k in ("holiday", "after"):
                continue
            lik = pred if yi == 1 else 1 - pred
            grow = r * lik * (1 - hazard)
            cp = float((r * lik).sum() * hazard)
            r = np.concatenate([[cp], grow])
            s = np.concatenate([[0.0], s + yi])
            n = np.concatenate([[0.0], n + 1])
            if len(r) > max_run:
                r[max_run - 1] += r[max_run:].sum()
                r, s, n = r[:max_run], s[:max_run], n[:max_run]
            r /= r.sum()
    return pd.Series(out)


# ---------------------------------------------------------------------------- change points after the fact

def _cost(k, m):
    """-2 log-likelihood of a Bernoulli segment with k ones in m trials, at its own rate."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(k > 0, k * np.log(np.where(k > 0, k / m, 1.0)), 0.0)
        b = np.where(m - k > 0, (m - k) * np.log(np.where(m - k > 0, 1 - k / m, 1.0)), 0.0)
    return -2 * (a + b)


def segment(days: np.ndarray, y: np.ndarray, min_days: int = 28, penalty: float | None = None):
    """Optimal partition of a dated 0/1 series into segments of constant rate (penalised likelihood, PELT
    pruning). A segment must span at least `min_days` calendar days. The penalty per change point is
    2 log n unless given. Returns a list of (first, last) observation positions, inclusive."""
    n = len(y)
    beta = 2 * np.log(n) if penalty is None else penalty
    cy = np.concatenate([[0], np.cumsum(y)])
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    last = np.zeros(n + 1, int)
    cand = np.array([0])
    for j in range(1, n + 1):
        span_ok = (days[j - 1] - days[np.minimum(cand, n - 1)]) >= min_days - 1
        seg = _cost(cy[j] - cy[cand], j - cand)
        cost = np.where(span_ok, F[cand] + seg + beta, np.inf)
        if np.isfinite(cost).any():
            i = int(np.argmin(cost))
            F[j], last[j] = cost[i], cand[i]
        keep = ~span_ok | (F[cand] + seg <= F[j])
        cand = np.append(cand[keep], j)
    out, j = [], n
    while j > 0:
        i = last[j]
        out.append((i, j - 1))
        j = i
    return out[::-1]


def segments(df: pd.DataFrame, min_days: int = 28) -> pd.DataFrame:
    """Segments of each producer's series of one day of the week (working and weekend days; holidays and
    the days after them left out), over all eligible days."""
    rows = []
    e = df[df.eligible & df.kind.isin(["working", "weekend"])]
    for (grp, p, wd), g in e.groupby(["group", "producer", "wd"], sort=False):
        g = g.sort_values("date")
        if len(g) < 8:
            continue
        days = ((g.date - g.date.iloc[0]).dt.days).to_numpy()
        y = g.flat.to_numpy()
        for a, b in segment(days, y, min_days):
            rows.append((grp, p, wd, g.date.iloc[a], g.date.iloc[b], b - a + 1, y[a:b + 1].mean()))
    return pd.DataFrame(rows, columns=["group", "producer", "wd", "start", "end", "n", "rate"])


def transitions(seg: pd.DataFrame, low: float = 0.2, high: float = 0.8) -> pd.DataFrame:
    """Moves from a segment with rate <= low to a later one with rate >= high (or the reverse), through any
    segments in between. Duration: days from the last day of the first segment to the first day of the
    second."""
    rows = []
    for (grp, p, kind), g in seg.groupby(["group", "producer", "wd"], sort=False):
        g = g.reset_index(drop=True)
        i = 0
        while i < len(g) - 1:
            a = g.rate[i]
            if a <= low or a >= high:
                j = i + 1
                while j < len(g) and not ((a <= low and g.rate[j] >= high) or (a >= high and g.rate[j] <= low)):
                    if (a <= low and g.rate[j] <= low) or (a >= high and g.rate[j] >= high):
                        break
                    j += 1
                if j < len(g) and ((a <= low and g.rate[j] >= high) or (a >= high and g.rate[j] <= low)):
                    rows.append((grp, p, kind, "up" if a <= low else "down", g.end[i], g.start[j], j - i - 1,
                                 (g.start[j] - g.end[i]).days, a, g.rate[j]))
                    i = j
                    continue
            i += 1
    return pd.DataFrame(rows, columns=["group", "producer", "wd", "direction", "last_old", "first_new",
                                       "n_between", "days", "rate_old", "rate_new"])


def refine(tr: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """The first day of the new segment on which the producer's file already behaved as in the new regime
    (unchanged after an upward move, changed after a downward one). A segment must span at least four weeks,
    so its first day can still belong to the old regime when the new one is short."""
    e = df[df.eligible & df.kind.isin(["working", "weekend"])]
    by = {k: g.sort_values("date") for k, g in e.groupby(["producer", "wd"])}
    out = []
    for r in tr.itertuples():
        g = by[(r.producer, r.wd)]
        g = g[g.date >= r.first_new]
        m = g[g.flat == (1.0 if r.direction == "up" else 0.0)]
        out.append(m.date.iloc[0] if len(m) else r.first_new)
    return pd.Series(out, index=tr.index)


def process_changes(tr: pd.DataFrame, reach: int = 6) -> pd.DataFrame:
    """Transitions of one producer in the same direction whose refined dates lie within `reach` days of
    the first of them are taken as one change of the producer's process. Returns one row per change: the
    first refined date and the days of the week that changed."""
    rows = []
    for (grp, p, d), g in tr.sort_values("first_ref").groupby(["group", "producer", "direction"], sort=False):
        cur = []
        for r in g.itertuples():
            if cur and (r.first_ref - cur[0].first_ref).days > reach:
                rows.append(cur)
                cur = []
            cur.append(r)
        if cur:
            rows.append(cur)
    out = pd.DataFrame([(c[0].group, c[0].producer, c[0].direction, min(x.first_ref for x in c),
                         tuple(sorted(x.wd for x in c))) for c in rows],
                       columns=["group", "producer", "direction", "date", "wds"])
    out["n_days"] = out.wds.map(len)
    return out.sort_values(["group", "producer", "date"]).reset_index(drop=True)


def event_profile(df: pd.DataFrame, changes: pd.DataFrame, weeks: int = 12) -> np.ndarray:
    """Share of unchanged files on the days of the week that changed, in each week from `weeks` before to
    `weeks` after each change (week 0 starts on the change's date). One row per change."""
    e = df[df.eligible & df.kind.isin(["working", "weekend"])]
    by = {p: g for p, g in e.groupby("producer")}
    out = np.full((len(changes), 2 * weeks + 1), np.nan)
    for i, r in enumerate(changes.itertuples()):
        g = by[r.producer]
        g = g[g.wd.isin(r.wds)]
        k = np.floor((g.date - r.date).dt.days / 7).astype(int)
        m = (k >= -weeks) & (k <= weeks)
        s = g.flat[m].groupby(k[m]).mean()
        out[i, (s.index + weeks).to_numpy()] = s.to_numpy()
    return out


def simulated_profile(ramp_weeks: int, n_series: int = 400, weeks: int = 12, seed: int = SEED,
                      low: float = 0.04, high: float = 0.95, length: int = 150) -> np.ndarray:
    """What the same procedure (segmentation, transition, refinement, profile) shows for series of one
    day of the week whose rate moves from `low` to `high` along a straight line lasting `ramp_weeks`
    (0 for a step), each series with one move at a random week."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    start = pd.Timestamp("2020-06-01")
    prof = []
    for _ in range(n_series):
        c = rng.integers(40, length - 40)
        p = low + (high - low) * (np.clip((t - c) / ramp_weeks, 0, 1) if ramp_weeks else (t >= c))
        y = (rng.random(length) < p).astype(float)
        days = t * 7
        seg = pd.DataFrame([("s", "x", 0, start + pd.Timedelta(days=int(days[a])),
                             start + pd.Timedelta(days=int(days[b])), b - a + 1, y[a:b + 1].mean())
                            for a, b in segment(days, y)],
                           columns=["group", "producer", "wd", "start", "end", "n", "rate"])
        tr = transitions(seg)
        tr = tr[tr.direction == "up"]
        if not len(tr):
            continue
        r = tr.iloc[0]
        pos = int((r.first_new - start).days // 7)
        later = np.flatnonzero(y[pos:] == 1)
        pos = pos + int(later[0]) if len(later) else pos
        row = np.full(2 * weeks + 1, np.nan)
        for k in range(-weeks, weeks + 1):
            if 0 <= pos + k < length:
                row[k + weeks] = y[pos + k]
        prof.append(row)
    return np.array(prof)


def documented(changes: pd.DataFrame, reach: int = 42) -> pd.DataFrame:
    """The announced changes of the sampling frame (data/documented_changes.csv), each matched to the
    detected change of its producer, involving two or more days of the week, nearest to the announced first
    day (within `reach` days)."""
    doc = pd.read_csv(DATA / "documented_changes.csv", parse_dates=["first_day"])
    multi = changes[changes.n_days >= 2]
    det, days, n = [], [], []
    for r in doc.itertuples():
        m = multi[multi.producer == r.producer]
        n.append(len(m))
        gap = (m.date - r.first_day).dt.days.abs()
        m = m[gap <= reach]
        if len(m):
            best = m.loc[(m.date - r.first_day).dt.days.abs().idxmin()]
            det.append(best.date)
            days.append(best.wds)
        else:
            det.append(pd.NaT)
            days.append(())
    doc["detected"], doc["detected_days"], doc["n_multi"] = det, days, n
    doc["offset"] = (doc.detected - doc.first_day).dt.days
    return doc


def max_gap(tr: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """Longest run of days without an eligible observation of that day of the week between the last day of
    the old segment and the first of the new one. Seven means consecutive weeks; anything longer means the
    producer was not counted for a while (too few cases, or not reporting at all)."""
    e = df[df.eligible & df.kind.isin(["working", "weekend"])]
    by = {k: g.date.sort_values().to_numpy() for k, g in e.groupby(["producer", "wd"])}
    out = []
    for r in tr.itertuples():
        d = by[(r.producer, r.wd)]
        d = d[(d >= np.datetime64(r.last_old)) & (d <= np.datetime64(r.first_new))]
        out.append(int(np.diff(d).max() / np.timedelta64(1, "D")) if len(d) > 1 else 0)
    return pd.Series(out, index=tr.index)


def changes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Segments, transitions (with refined dates and gaps) and the process changes built from the
    transitions that were observed week by week (no gap longer than two weeks)."""
    seg = segments(df)
    tr = transitions(seg)
    tr["first_ref"] = refine(tr, df)
    tr["gap"] = max_gap(tr, df)
    ch = process_changes(tr[tr.gap <= 14])
    return seg, tr, ch


def bocpd_joint(df: pd.DataFrame, hazard: float, pooled: pd.Series, prior: float = PRIOR,
                max_run: int = 400) -> pd.Series:
    """Change-point detection with one change-point process per producer, shared by its seven days of the
    week: between change points each day of the week keeps its own rate, and a change point (probability
    `hazard` before each counted day) resets all seven. Holidays and the days after them are predicted but
    do not update the state. A prediction uses only earlier days."""
    out = {}
    e = df[df.eligible]
    for p, g in e.groupby("producer", sort=False):
        g = g.sort_values("date")
        r = np.array([1.0])
        S = np.zeros((1, 7))
        N = np.zeros((1, 7))
        for i, yi, k, kind in zip(g.index, g.flat.to_numpy(), g.wd.to_numpy(), g.kind.to_numpy()):
            p0 = pooled[i]
            pred = (S[:, k] + prior * p0) / (N[:, k] + prior)
            out[i] = float(r @ pred)
            if kind in ("holiday", "after"):
                continue
            lik = pred if yi == 1 else 1 - pred
            rl = r * lik
            cp = float(rl.sum() * hazard)
            r = np.concatenate([[cp], rl * (1 - hazard)])
            S = np.vstack([np.zeros((1, 7)), S])
            N = np.vstack([np.zeros((1, 7)), N])
            S[1:, k] += yi
            N[1:, k] += 1
            if len(r) > max_run:
                r[max_run - 1] += r[max_run:].sum()
                r, S, N = r[:max_run], S[:max_run], N[:max_run]
            r /= r.sum()
    return pd.Series(out)
