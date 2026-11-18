"""Synthetic leagues where the truth is known: used by the tests and by the null-drift check.

Boards change the manager after bad runs at random; a new manager adds a chosen amount to the club's
strength (zero by default), so every gain after a change in a zero-effect league is regression to the mean.
"""

import numpy as np
import pandas as pd

import coach_core as cc


def round_robin(n):
    """Double round robin: a list of rounds, each a list of (home, away) pairs (circle method)."""
    clubs = list(range(n))
    first = []
    for r in range(n - 1):
        pairs = [(clubs[i], clubs[n - 1 - i]) for i in range(n // 2)]
        first.append([(a, b) if r % 2 == 0 else (b, a) for a, b in pairs])
        clubs = [clubs[0]] + [clubs[-1]] + clubs[1:-1]
    return first + [[(b, a) for a, b in rd] for rd in first]


def synthetic(seasons=30, clubs=20, effect=0.0, seed=1):
    """A league whose boards change the manager after bad runs.

    Club strength is fixed within a season. After each match from the 5th to the 28th, a club with
    at most 3 points from its last five changes manager with probability 0.3; the new manager adds
    `effect` to the club's strength for the rest of the season. With effect=0 a change does nothing,
    so any gain after it is regression to the mean.
    """
    rng = np.random.default_rng(seed)
    schedule = round_robin(clubs)
    rows, sv = [], []
    game = 0
    for s in range(seasons):
        strength = rng.normal(0, 0.45, clubs)
        boost = np.zeros(clubs)
        manager = [f"m{s}_{c}_0" for c in range(clubs)]
        changes = [0] * clubs
        history = [[] for _ in range(clubs)]
        for c in range(clubs):
            sv.append((c, s, float(np.exp(17 + 2 * strength[c] + rng.normal(0, 0.3)))))
        for k, rd in enumerate(schedule, start=1):
            date = pd.Timestamp(f"{2000 + s}-08-01") + pd.Timedelta(days=7 * k)
            for h, a in rd:
                game += 1
                d = (strength[h] + boost[h]) - (strength[a] + boost[a])
                gh, ga = (
                    rng.poisson(1.45 * np.exp(0.5 * d)),
                    rng.poisson(1.15 * np.exp(-0.5 * d)),
                )
                sh, sa = (
                    rng.poisson(4.5 * np.exp(0.4 * d)),
                    rng.poisson(3.8 * np.exp(-0.4 * d)),
                )
                for club, opp, gf, gag, sf, sag, home in (
                    (h, a, gh, ga, sh, sa, 1),
                    (a, h, ga, gh, sa, sh, 0),
                ):
                    rows.append(
                        (
                            "SY1",
                            s,
                            date,
                            game,
                            club,
                            opp,
                            f"club{club}",
                            gf,
                            gag,
                            10,
                            10,
                            sf,
                            sag,
                            manager[club],
                            home,
                        )
                    )
                    history[club].append(3 if gf > gag else (1 if gf == gag else 0))
            for c in range(clubs):
                if 5 <= k <= 28 and sum(history[c][-5:]) <= 3 and rng.random() < 0.3:
                    changes[c] += 1
                    manager[c] = f"m{s}_{c}_{changes[c]}"
                    boost[c] += effect
    m = pd.DataFrame(
        rows,
        columns=[
            "league",
            "season",
            "date",
            "game_id",
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
            "home",
        ],
    )
    sv = pd.DataFrame(sv, columns=["club_id", "season", "squad_value"])
    nj = pd.DataFrame(
        columns=[
            "club_id",
            "season",
            "manager",
            "first",
            "last",
            "next_date",
            "next_club_id",
        ]
    )
    nj = nj.astype(
        {
            "first": "datetime64[ns]",
            "last": "datetime64[ns]",
            "next_date": "datetime64[ns]",
        }
    )
    return cc.prepare(m), sv, nj


def estimate(m, sv, nj):
    """Same-spot estimate, before/after gain and number of changes on a synthetic league."""
    dp = cc.decision_points(m, sv, nj)
    e = cc.eligible(dp)
    _, logit = cc.fit_hazard(e)
    pairs = cc.match(e, logit)
    per = cc.matched_effect(e, pairs)
    t = pairs.t.unique()
    return (
        per["diff"].mean(),
        (per.yt - 2 * e.last5.to_numpy(float)[t]).mean(),
        len(per),
    )
