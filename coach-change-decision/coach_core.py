"""Core of the analysis: decision points, coach changes, the same-spot comparison and its checks.

A *decision point* is a club right after one of its league matches, with the incumbent manager in
post. At each one the board can change the manager before the next match or keep him. A *change*
is a decision point after which a different manager leads the club in the next league match.

The comparison for a change is the set of clubs that were in the same spot at the same stage of a
season and kept their manager that week: same side of the relegation line, similar season form,
the result of the latest match, recent points, goal difference, shots-on-target share, squad value,
and the strength of the next ten opponents, with a similar estimated chance of a change. Keeping the
manager this week does not mean keeping him for good; that is the choice a board actually faces.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

HORIZON = 10  # matches over which the outcome is measured
FIRST_DECISION = 5  # first match after which a decision point counts
MIN_TENURE = 5  # outgoing manager must have led at least this many league matches in a row
NEXT_JOB_DAYS = 30  # leading another club within this many days: flagged (a sensitivity check)
CONTROLS = 5  # comparison clubs per change
SEED = 20261118

LEAGUE_NAMES = {
    "GB1": "Premier League",
    "ES1": "LaLiga",
    "IT1": "Serie A",
    "L1": "Bundesliga",
    "FR1": "Ligue 1",
}
EXACT = ("in_drop", "last5", "pts")  # matched exactly
MATCH_ON = [
    "pts",
    "last5",
    "form10",
    "ppg",
    "gd_pg",
    "pos_norm",
    "value_rel",
    "sot_share",
    "opp_next",
]


def prepare(m):
    """Points and the club's match number within its season, in club-match order."""
    m = m.copy()
    m["pts"] = np.select([m.gf > m.ga, m.gf == m.ga], [3, 1], 0)
    m = m.sort_values(["league", "season", "club_id", "date"]).reset_index(drop=True)
    m["k"] = m.groupby(["league", "season", "club_id"]).cumcount() + 1
    return m


def load():
    """matches, squad values and next jobs from data/."""
    m = pd.read_csv(DATA / "matches.csv", parse_dates=["date"])
    m = m[~((m.league == "FR1") & (m.season == 2019))]  # Ligue 1 2019/20 stopped after 28 rounds
    sv = pd.read_csv(DATA / "squad_values.csv")
    nj = pd.read_csv(DATA / "next_jobs.csv", parse_dates=["first", "last", "next_date"])
    return prepare(m), sv, nj


def standings(m):
    """League position and points of the safety line after each club match.

    The table is the one on the evening of the match date: every match played up to that date
    counts, so clubs with a game in hand are ranked as the table showed them. Ties are broken by
    goal difference, then goals scored. The safety line is the highest place outside the bottom
    three (17th of 20, 15th of 18).
    """
    out = []
    for _, d in m.groupby(["league", "season"]):
        clubs = np.sort(d.club_id.unique())
        dates = np.sort(d.date.unique())
        ci = np.searchsorted(clubs, d.club_id.to_numpy())
        di = np.searchsorted(dates, d.date.to_numpy())
        grid = {}
        for col in ("pts", "gf", "ga"):
            a = np.zeros((len(clubs), len(dates)))
            np.add.at(a, (ci, di), d[col].to_numpy())
            grid[col] = a.cumsum(axis=1)
        score = grid["pts"] * 1e6 + (grid["gf"] - grid["ga"] + 500) * 1e3 + grid["gf"]
        order = (-score).argsort(axis=0, kind="stable")
        rank = np.empty_like(order)
        rank[order, np.arange(len(dates))[None, :]] = np.arange(1, len(clubs) + 1)[:, None]
        n = len(clubs)
        safe_pts = -np.sort(-grid["pts"], axis=0)[n - 4]  # points of the club in place n-3
        out.append(
            pd.DataFrame(
                {
                    "idx": d.index,
                    "pos": rank[ci, di],
                    "n_clubs": n,
                    "safe_pts": safe_pts[di],
                    "cum_pts": grid["pts"][ci, di],
                }
            )
        )
    t = pd.concat(out).set_index("idx").sort_index()
    return t


def tenure(m):
    """Consecutive league matches the club's manager has led, counting earlier seasons."""
    d = m.sort_values(["club_id", "date"])
    new_spell = (d.manager != d.groupby("club_id").manager.shift()).astype(int)
    spell = new_spell.groupby(d.club_id).cumsum()
    t = d.groupby([d.club_id, spell]).cumcount() + 1
    return t.reindex(m.index)


def decision_points(m, sv, nj):
    """One row per club and league match, with the state a board sees and what happened next."""
    st = standings(m)
    m = m.join(st)
    m["tenure"] = tenure(m)
    g = m.groupby(["league", "season", "club_id"])
    m["n_matches"] = g.k.transform("max")
    m["ppg"] = g.pts.cumsum() / m.k
    m["gd_pg"] = (g.gf.cumsum() - g.ga.cumsum()) / m.k

    def roll(col, w, mp=None):
        return g[col].transform(lambda x: x.rolling(w, min_periods=mp or w).sum())

    m["last5"] = roll("pts", 5)
    m["last10"] = roll("pts", 10)
    m["prev5"] = m.last10 - m.last5
    m["form10"] = g.pts.transform(lambda x: x.rolling(10, min_periods=5).mean())  # last ten, or all if fewer
    sf, sa = (
        roll("sot_for", 10, 5),
        roll("sot_against", 10, 5),
    )  # last ten, or all if fewer
    m["sot_share"] = sf / (sf + sa)
    m["pos_norm"] = (m.pos - 1) / (m.n_clubs - 1)
    m["in_drop"] = (m.pos > m.n_clubs - 3).astype(int)
    m["gap_safety"] = m.cum_pts - m.safe_pts
    m["share"] = m.k / m.n_matches

    v = sv.copy()
    v["log_value"] = np.log(v.squad_value.clip(lower=1e5))
    m = m.merge(v[["club_id", "season", "log_value"]], on=["club_id", "season"], how="left")
    m["value_rel"] = m.log_value - m.groupby(["league", "season"]).log_value.transform("median")

    # what comes next: manager in the next match, points and opponents over the horizon
    g = m.groupby(["league", "season", "club_id"])
    m["next_manager"] = g.manager.shift(-1)

    def fwd(col, h):
        return g[col].transform(lambda x: x[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1))

    m["next10"] = fwd("pts", HORIZON)
    opp_value = m[["club_id", "season", "value_rel"]].drop_duplicates(["club_id", "season"])
    opp_value = opp_value.rename(columns={"club_id": "opp_id", "value_rel": "opp_value"})
    m = m.merge(opp_value, on=["opp_id", "season"], how="left")
    g = m.groupby(["league", "season", "club_id"])
    m["opp_next"] = g.opp_value.transform(
        lambda x: x[::-1].rolling(HORIZON, min_periods=HORIZON).mean()[::-1].shift(-1)
    )
    m["final_pos"] = g.pos.transform("last")
    m["bottom3"] = (m.final_pos > m.n_clubs - 3).astype(int)
    m["rest_ppg"] = (g.pts.transform("sum") - g.pts.cumsum()) / (m.n_matches - m.k)

    # did the manager change before the next match, and was it a board decision?
    m["change"] = (m.next_manager.notna() & (m.next_manager != m.manager)).astype(int)
    later = m.groupby(["league", "season", "club_id"]).manager.transform(
        lambda x: [set(x.iloc[i + 1 :]) for i in range(len(x))]
    )
    m["returns"] = [mgr in s for mgr, s in zip(m.manager, later, strict=True)]
    jobs = nj.set_index(["club_id", "season", "manager"])
    key = pd.MultiIndex.from_frame(m[["club_id", "season", "manager"]])
    gap = (jobs.next_date - jobs["last"]).dt.days.reindex(key).to_numpy()
    m["left_for_job"] = (gap <= NEXT_JOB_DAYS) & m.change.astype(bool)
    m = m.reset_index(drop=True)
    m["row"] = m.index  # survives the filters in eligible(), for event_paths()
    return m


def eligible(dp):
    """Decision points that enter the comparison: in the window, incumbent established, clean data."""
    e = dp[(dp.k >= FIRST_DECISION) & (dp.k <= dp.n_matches - HORIZON) & (dp.tenure >= MIN_TENURE)]
    e = e[e.next10.notna() & e[MATCH_ON].notna().all(axis=1)]
    # a manager who leads the club again later in the season was absent, not replaced: drop it.
    # Departures to another club stay in: the records cannot tell a resignation from a dismissal
    # (several managers flagged left_for_job had been sacked and were hired again within days).
    e = e[~(e.change.astype(bool) & e.returns)]
    return e.reset_index(drop=True)


def features(e):
    """Design matrix for the chance of a change."""
    x = pd.DataFrame(
        {
            "last1": e.pts,
            "last5": e.last5,
            "form10": e.form10,
            "ppg": e.ppg,
            "gd_pg": e.gd_pg,
            "pos_norm": e.pos_norm,
            "in_drop": e.in_drop,
            "gap_safety": e.gap_safety.clip(-15, 30),
            "value_rel": e.value_rel,
            "sot_share": e.sot_share,
            "share": e.share,
            "share2": e.share**2,
            "log_tenure": np.log(e.tenure),
            "last5_x_value": e.last5 * e.value_rel,
        }
    )
    for lg in ["ES1", "IT1", "L1", "FR1"]:
        x[f"lg_{lg}"] = (e.league == lg).astype(int)
    return x


def fit_hazard(e):
    """Logistic model of a change before the next match; returns the fitted model and the logit."""
    import statsmodels.api as sm

    x = features(e)
    x = sm.add_constant(x.loc[:, x.std() > 0])  # a league absent from the data has no dummy
    fit = sm.Logit(e.change.to_numpy(), x).fit(disp=False, maxiter=200)
    return fit, x @ fit.params


def match(e, logit, controls=CONTROLS, caliper_sd=0.2, max_share_gap=0.06, exact=EXACT):
    """Nearest comparison clubs for every change, with replacement.

    Candidates kept the manager at a decision point and share with the change the variables in
    `exact` (side of the relegation line, points from the last five, result of the latest match).
    They are within `max_share_gap` of the same stage of the season (about two matches), within
    `caliper_sd` standard deviations of the change's estimated logit, and not the same club in the
    same season. Among them, the nearest `controls` by Mahalanobis distance on MATCH_ON.
    Returns a frame of (treated row, control row) pairs; changes with no candidate are dropped.

    Matching exactly on what boards react to most matters: on simulated leagues where a change does
    nothing, distance matching alone drifts upwards by about 0.2 points over ten, because the nearest
    clubs by distance can sit a point or two higher on recent form; exact matching removes the drift.
    """
    z = e[MATCH_ON].to_numpy(float)
    vi = np.linalg.inv(np.cov(z, rowvar=False))
    lg = np.asarray(logit)
    cal = caliper_sd * lg.std()
    treated = np.flatnonzero(e.change.to_numpy() == 1)
    ctrl = np.flatnonzero(e.change.to_numpy() == 0)
    cs = club_season(e)
    share = e.share.to_numpy()
    same = {col: e[col].to_numpy() for col in exact}
    rows = []
    for t in treated:
        ok = (np.abs(share[ctrl] - share[t]) <= max_share_gap) & (np.abs(lg[ctrl] - lg[t]) <= cal) & (cs[ctrl] != cs[t])
        for col in exact:
            ok &= same[col][ctrl] == same[col][t]
        c = ctrl[ok]
        if len(c) == 0:
            continue
        diff = z[c] - z[t]
        dist = np.einsum("ij,jk,ik->i", diff, vi, diff)
        for j in c[np.argsort(dist, kind="stable")[:controls]]:
            rows.append((t, j))
    return pd.DataFrame(rows, columns=["t", "c"])


def matched_effect(e, pairs, outcome="next10"):
    """Per change: its outcome, the mean of its comparison clubs, and the difference."""
    y = e[outcome].to_numpy(float)
    p = pairs.assign(yt=y[pairs.t], yc=y[pairs.c])
    per = p.groupby("t").agg(yt=("yt", "first"), yc=("yc", "mean"))
    per["diff"] = per.yt - per.yc
    return per


def bootstrap(e, pairs, stat, draws=2000, seed=SEED):
    """Percentile interval of `stat(per_change_frame)` resampling club-seasons of the changes.

    `stat` receives the frame from matched_effect restricted to the resampled changes (with
    repeats). The comparison clubs stay attached to their change.
    """
    rng = np.random.default_rng(seed)
    per_t = pairs.t.unique()
    cluster = (e.league + e.season.astype(str) + "_" + e.club_id.astype(str)).to_numpy()[per_t]
    groups = pd.Series(per_t).groupby(cluster).apply(list)
    keys = groups.index.to_numpy()
    vals = []
    for _ in range(draws):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        ts = [t for i in pick for t in groups.iat[i]]
        vals.append(stat(ts))
    return np.percentile(vals, [2.5, 97.5])


def event_paths(m, rows, before=9, after=10):
    """Points per match around each decision point: columns -before..after, 0 = the decision match.

    `rows` are row labels of `m` (a decision_points frame in club-match order).
    """
    pts = m.pts.to_numpy(float)
    key = (m.league + m.season.astype(str) + "_" + m.club_id.astype(str)).to_numpy()
    out = np.full((len(rows), before + after + 1), np.nan)
    n = len(m)
    for r, i in enumerate(rows):
        for o in range(-before, after + 1):
            j = i + o
            if 0 <= j < n and key[j] == key[i]:
                out[r, o + before] = pts[j]
    return pd.DataFrame(out, columns=range(-before, after + 1))


def auc(y, score):
    """Area under the ROC curve: the chance a random change scores above a random non-change."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    r = rankdata(np.asarray(score))
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def club_season(frame):
    """Key of the club-season a row belongs to."""
    return (frame.league + frame.season.astype(str) + "_" + frame.club_id.astype(str)).to_numpy()


def kept_through(dp, rows, h=HORIZON):
    """True where the manager after the decision match still led the club in each of the next h."""
    mgr, key = dp.manager.to_numpy(), club_season(dp)
    out = np.zeros(len(rows), bool)
    for n, r in enumerate(rows):
        stop = min(r + h, len(dp) - 1)
        span = slice(r + 1, stop + 1)
        out[n] = stop - r == h and (key[span] == key[r]).all() and (mgr[span] == mgr[r]).all()
    return out


def regression_estimate(e):
    """Change coefficient from a linear model of next-ten points on the same state, with season
    fixed effects; standard errors clustered by club-season. A check on the matching, not a rival."""
    import statsmodels.api as sm

    x = features(e).assign(opp_next=e.opp_next, change=e.change)
    x = x.loc[:, x.std() > 0]
    x = x.join(pd.get_dummies(e.season, prefix="s", drop_first=True, dtype=float))
    groups = pd.factorize(club_season(e))[0]
    fit = sm.OLS(e.next10.to_numpy(float), sm.add_constant(x.astype(float))).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    lo, hi = fit.conf_int().loc["change"]
    return fit.params["change"], lo, hi


def placebo(e, logit, pairs, **kw):
    """Pretend the nearest comparison club of every change had changed manager, drop the real
    changes, and rerun the matching. A sound comparison should find nothing."""
    first = pairs.drop_duplicates("t").c.unique()
    keep = e.change.to_numpy() == 0
    f = e.copy()
    f["change"] = 0
    f.loc[first, "change"] = 1
    f, lg = f[keep].reset_index(drop=True), np.asarray(logit)[keep]
    p = match(f, lg, **kw)
    return f, p, matched_effect(f, p)
