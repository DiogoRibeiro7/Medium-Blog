"""Build the compact tables in data/ from the pinned sources in .cache/.

    python fetch_data.py && python build_panel.py

data/matches.csv     one row per club and league match, big five, 2012/13 to 2025/26: date, opponent,
                     goals, shots and shots on target for and against, the club's manager
data/squad_values.csv  squad market value per club and season, on 15 September (after the window)
data/next_jobs.csv   for every manager spell in those leagues, the first later match in which the same
                     manager led a different club (any competition in the source), if any

The committed files are what the analysis reads; this script exists to show how they were made.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
DATA = HERE / "data"
LEAGUES = {
    "GB1": "England",
    "ES1": "Spain",
    "IT1": "Italy",
    "L1": "Germany",
    "FR1": "France",
}
SEASONS = range(2012, 2026)


def league_games():
    """Big-five league matches from transfermarkt, one row per match."""
    g = pd.read_csv(CACHE / "games.csv.gz", parse_dates=["date"])
    g = g[g.competition_id.isin(LEAGUES) & g.season.isin(SEASONS)].copy()
    return g, pd.read_csv(
        CACHE / "games.csv.gz",
        parse_dates=["date"],
        usecols=[
            "date",
            "home_club_id",
            "away_club_id",
            "home_club_manager_name",
            "away_club_manager_name",
        ],
    )


def shots():
    """football-data.co.uk shots per match, one row per match, keyed by league, season, date, score."""
    frames = []
    for code in LEAGUES:
        for s in SEASONS:
            d = pd.read_csv(CACHE / f"fd_{code}_{s}.csv")
            d["competition_id"], d["season"] = code, s
            frames.append(d)
    fd = pd.concat(frames, ignore_index=True)
    fd["date"] = pd.to_datetime(fd["Date"], dayfirst=True, format="mixed")
    return fd[
        [
            "competition_id",
            "season",
            "date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "HS",
            "AS",
            "HST",
            "AST",
        ]
    ]


def join_shots(g, fd):
    """Attach shots by learning the club-name map from matches that agree on league, date and score.

    Each transfermarkt club id is mapped to the football-data name it shares the most matches with;
    the map is then applied to every match, so a postponed or re-dated fixture still joins by teams.
    """
    key = ["competition_id", "season", "date"]
    m = g.merge(
        fd,
        left_on=key + ["home_club_goals", "away_club_goals"],
        right_on=key + ["FTHG", "FTAG"],
    )
    pairs = pd.concat(
        [
            m[["home_club_id", "HomeTeam"]].set_axis(["club_id", "fd"], axis=1),
            m[["away_club_id", "AwayTeam"]].set_axis(["club_id", "fd"], axis=1),
        ]
    )
    counts = pairs.value_counts().reset_index(name="n").sort_values("n", ascending=False)
    name = counts.drop_duplicates("club_id").set_index("club_id")["fd"]
    g = g.assign(HomeTeam=g.home_club_id.map(name), AwayTeam=g.away_club_id.map(name))
    out = g.merge(
        fd.drop(columns=["date", "FTHG", "FTAG"]),
        on=["competition_id", "season", "HomeTeam", "AwayTeam"],
        how="left",
    )
    return out


def long_form(g):
    """One row per club and match."""
    cols = {
        "home": (
            "home_club_id",
            "away_club_id",
            "home_club_name",
            "home_club_goals",
            "away_club_goals",
            "HS",
            "AS",
            "HST",
            "AST",
            "home_club_manager_name",
        ),
        "away": (
            "away_club_id",
            "home_club_id",
            "away_club_name",
            "away_club_goals",
            "home_club_goals",
            "AS",
            "HS",
            "AST",
            "HST",
            "away_club_manager_name",
        ),
    }
    names = [
        "club_id",
        "opp_id",
        "club",
        "gf",
        "ga",
        "shots_for",
        "shots_against",
        "sot_for",
        "sot_against",
        "manager",
    ]
    rows = []
    for side, c in cols.items():
        d = g[["competition_id", "season", "date", "game_id", *c]].copy()
        d.columns = ["league", "season", "date", "game_id", *names]
        d["home"] = int(side == "home")
        rows.append(d)
    out = pd.concat(rows, ignore_index=True).sort_values(["league", "season", "club_id", "date"])
    return out.reset_index(drop=True)


def squad_values():
    """Sum of each player's latest valuation on or before 15 September, by the club listed with it.

    A valuation older than a year is ignored. A player who moved in the summer and has not been
    revalued since is still counted at the old club: the source records the club at valuation time.
    """
    pv = pd.read_csv(CACHE / "player_valuations.csv.gz", parse_dates=["date"])
    out = []
    for s in SEASONS:
        cut = pd.Timestamp(f"{s}-09-15")
        v = pv[(pv.date <= cut) & (pv.date > cut - pd.Timedelta(days=365))]
        v = v.sort_values("date").drop_duplicates("player_id", keep="last")
        t = v.groupby("current_club_id").market_value_in_eur.sum().rename("squad_value").reset_index()
        t["season"] = s
        out.append(t.rename(columns={"current_club_id": "club_id"}))
    return pd.concat(out, ignore_index=True)


def next_jobs(long, all_games):
    """For each manager spell at a big-five club, the first later match leading a different club."""
    led = (
        pd.concat(
            [
                all_games[["date", "home_club_id", "home_club_manager_name"]].set_axis(
                    ["date", "club_id", "manager"], axis=1
                ),
                all_games[["date", "away_club_id", "away_club_manager_name"]].set_axis(
                    ["date", "club_id", "manager"], axis=1
                ),
            ]
        )
        .dropna()
        .sort_values("date")
    )
    spells = (
        long.dropna(subset=["manager"])
        .groupby(["club_id", "season", "manager"])
        .date.agg(first="min", last="max")
        .reset_index()
    )
    by_manager = {m: d for m, d in led.groupby("manager")}
    nxt_date, nxt_club = [], []
    for r in spells.itertuples():
        d = by_manager.get(r.manager)
        later = d[(d.date > r.last) & (d.club_id != r.club_id)] if d is not None else d
        if later is None or later.empty:
            nxt_date.append(pd.NaT)
            nxt_club.append(np.nan)
        else:
            nxt_date.append(later.date.iloc[0])
            nxt_club.append(later.club_id.iloc[0])
    return spells.assign(next_date=nxt_date, next_club_id=nxt_club)


def main():
    DATA.mkdir(exist_ok=True)
    g, all_games = league_games()
    g = join_shots(g, shots())
    long = long_form(g)
    long.to_csv(DATA / "matches.csv", index=False)
    clubs = long[["club_id", "season"]].drop_duplicates()
    clubs.merge(squad_values(), how="left").to_csv(DATA / "squad_values.csv", index=False)
    next_jobs(long, all_games).to_csv(DATA / "next_jobs.csv", index=False)
    joined = long.shots_for.notna().mean()
    print(f"{len(long)} club-match rows, {long.game_id.nunique()} matches, shots joined for {joined:.4f}")


if __name__ == "__main__":
    main()
