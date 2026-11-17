"""State-by-election panel, 1972-2026.

Inputs
  data/derived/house_districts_raw_1972_2024.csv   (build_returns.py)
  data/state_results_2026_unofficial.csv           (statewide totals, 2026)
  data/map_drawers.csv                              (who drew each plan)

Output
  data/derived/state_panel.csv, one row per state and election, with

  seats, dem_seats, rep_seats       winners (independents excluded from shares)
  v_raw     Democratic share of the two-party statewide House vote, as cast
  v_imp     the same with uncontested districts imputed (1972-2024 only):
            an unopposed winner's district is given a 75/25 split and the
            mean contested turnout of that state and year
  s         Democratic share of two-party seats
  bias_us   uniform-swing partisan bias (1972-2024): shift every district by
            the same amount until the statewide vote is 50/50; bias is the
            Democratic seat share at that point minus 0.5
  mean_median  median minus mean of district shares (positive = pro-D)
  eg        efficiency gap from district votes (positive = pro-D)
  drawer, confidence, plan_id, plan_start
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "derived"

IMPUTE_SHARE = 0.75
DECENNIAL = [1972, 1982, 1992, 2002, 2012, 2022]

# New plans that took effect between decennial redistrictings.
MID_DECADE = {
    "AL": [2024, 2026], "CA": [1974, 1984, 2026], "FL": [1996, 2016, 2026],
    "GA": [1996, 2006, 2024], "LA": [1984, 1994, 1996, 2024, 2026],
    "NJ": [1984], "NY": [2024], "NC": [1998, 2016, 2020, 2024, 2026],
    "PA": [2018], "TX": [1984, 1996, 2004, 2026], "VA": [2016],
    "OH": [2026], "TN": [2026], "UT": [2026],
}


def impute_districts(d: pd.DataFrame, share: float = IMPUTE_SHARE) -> pd.DataFrame:
    d = d.copy()
    d["two_party"] = d["dem_votes"].fillna(0) + d["rep_votes"].fillna(0)
    contested = d["contested"].astype(bool)
    mean_turn = (d[contested].groupby(["year", "state_po"])["two_party"].mean()
                 .rename("mean_turn"))
    nat_turn = d[contested].groupby("year")["two_party"].mean().rename("nat_turn")
    d = d.join(mean_turn, on=["year", "state_po"]).join(nat_turn, on="year")
    d["mean_turn"] = d["mean_turn"].fillna(d["nat_turn"])
    v = d["dem_share"].copy()
    v[~contested & (d["winner"] == "D")] = share
    v[~contested & (d["winner"] == "R")] = 1 - share
    v[~contested & (d["winner"] == "I")] = np.nan
    d["v_imp_d"] = v
    d["t_imp"] = np.where(contested, d["two_party"], d["mean_turn"])
    return d


def _bias_uniform_swing(v: np.ndarray, w: np.ndarray) -> float:
    v, w = v[np.isfinite(v)], w[np.isfinite(v)]
    if len(v) < 2:
        return np.nan
    vbar = np.sum(v * w) / np.sum(w)
    shifted = v + (0.5 - vbar)
    return float(np.mean(shifted > 0.5) - 0.5)


def _efficiency_gap(dem: np.ndarray, rep: np.ndarray) -> float:
    """Wasted-vote efficiency gap; positive = Democrats waste fewer votes."""
    tot = dem + rep
    need = tot / 2
    d_won = dem > rep
    wasted_d = np.where(d_won, dem - need, dem)
    wasted_r = np.where(d_won, rep, rep - need)
    return float((wasted_r.sum() - wasted_d.sum()) / tot.sum())


def state_rows_1972_2024(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, st), g in d.groupby(["year", "state_po"]):
        seats = len(g)
        dem_s = int((g["winner"] == "D").sum())
        rep_s = int((g["winner"] == "R").sum())
        ind_s = int((g["winner"] == "I").sum())
        dem_v = g["dem_votes"].fillna(0).sum()
        rep_v = g["rep_votes"].fillna(0).sum()
        ok = g["v_imp_d"].notna()
        v_imp = (np.sum(g.loc[ok, "v_imp_d"] * g.loc[ok, "t_imp"])
                 / np.sum(g.loc[ok, "t_imp"]))
        vd = g.loc[ok, "v_imp_d"].to_numpy()
        w = g.loc[ok, "t_imp"].to_numpy()
        dem_imp = vd * w
        rep_imp = (1 - vd) * w
        rows.append(dict(
            year=int(year), state_po=st, seats=seats, dem_seats=dem_s,
            rep_seats=rep_s, ind_seats=ind_s, dem_votes=dem_v, rep_votes=rep_v,
            v_raw=dem_v / (dem_v + rep_v) if dem_v + rep_v > 0 else np.nan,
            v_imp=v_imp,
            n_uncontested=int((~g["contested"].astype(bool)).sum()),
            bias_us=_bias_uniform_swing(vd, np.ones_like(vd)) if seats > 1 else np.nan,
            mean_median=(float(np.median(vd) - np.mean(vd)) if seats > 1 else np.nan),
            eg=_efficiency_gap(dem_imp, rep_imp) if seats > 1 else np.nan,
            source="district returns",
        ))
    return pd.DataFrame(rows)


def state_rows_2026() -> pd.DataFrame:
    s = pd.read_csv(DATA / "state_results_2026_unofficial.csv")
    out = pd.DataFrame(dict(
        year=2026, state_po=s["state_po"],
        seats=s["dem_seats"] + s["rep_seats"] + s["other_seats"] + s["pending_seats"],
        dem_seats=s["dem_seats"],
        # Pending Louisiana runoffs are counted with the Republicans, as in the
        # post being checked (1 of 6 seats Democratic); flagged in the note.
        rep_seats=s["rep_seats"] + s["pending_seats"],
        ind_seats=s["other_seats"], dem_votes=s["dem_votes"], rep_votes=s["rep_votes"],
    ))
    out["v_raw"] = out["dem_votes"] / (out["dem_votes"] + out["rep_votes"])
    out["v_imp"] = np.nan
    out["source"] = "statewide totals, unofficial (Wikipedia, retrieved 17 Nov 2026)"
    return out


def plan_table() -> pd.DataFrame:
    m = pd.read_csv(DATA / "map_drawers.csv")
    rows = []
    for r in m.itertuples(index=False):
        for y in range(r.first_year, r.last_year + 1, 2):
            rows.append(dict(state_po=r.state_po, year=y, drawer=r.drawer,
                             confidence=r.confidence))
    p = pd.DataFrame(rows)
    # plan start = most recent decennial year or mid-decade break
    def start(st, y):
        cands = [c for c in DECENNIAL if c <= y] + [c for c in MID_DECADE.get(st, []) if c <= y]
        return max(cands)
    p["plan_start"] = [start(s, y) for s, y in zip(p["state_po"], p["year"])]
    p["plan_id"] = p["state_po"] + "-" + p["plan_start"].astype(str)
    return p


def main() -> pd.DataFrame:
    d = pd.read_csv(OUT / "house_districts_raw_1972_2024.csv")
    d = impute_districts(d)
    d.to_csv(OUT / "house_districts_imputed_1972_2024.csv", index=False)
    panel = pd.concat([state_rows_1972_2024(d), state_rows_2026()], ignore_index=True)
    panel["s"] = panel["dem_seats"] / (panel["dem_seats"] + panel["rep_seats"])
    panel = panel.merge(plan_table(), on=["state_po", "year"], how="left")
    # plan ids for uncoded states still need to change at each redistricting
    miss = panel["plan_start"].isna()
    panel.loc[miss, "plan_start"] = [
        max([c for c in DECENNIAL if c <= y] + [c for c in MID_DECADE.get(s, []) if c <= y])
        for s, y in zip(panel.loc[miss, "state_po"], panel.loc[miss, "year"])]
    panel["plan_start"] = panel["plan_start"].astype(int)
    panel["plan_id"] = panel["state_po"] + "-" + panel["plan_start"].astype(str)
    panel = panel.sort_values(["state_po", "year"]).reset_index(drop=True)
    panel.to_csv(OUT / "state_panel.csv", index=False)
    return panel


if __name__ == "__main__":
    p = main()
    print(p.groupby("year").agg(states=("state_po", "size"), seats=("seats", "sum"),
                                coded=("drawer", lambda s: s.notna().sum())).to_string())
    print(p[p.year.isin([2024, 2026]) & p.state_po.isin(["TX", "FL", "GA", "TN", "OK", "SC", "LA", "UT"])]
          [["year", "state_po", "seats", "dem_seats", "v_raw", "v_imp", "s", "drawer"]].to_string())
