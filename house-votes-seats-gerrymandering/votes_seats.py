"""Votes-seats curves for every House election, 1972-2026.

For each election year the states with two or more seats are pooled and
the King-Browning model is fitted by binomial GLM, with each state's seats
as trials:

    logit(seat share) = lambda_t + rho_t * logit(vote share)

lambda_t is the partisan bias of the year's curve (lambda = 0 is the
symmetric curve through 50/50); expit(lambda_t) - 0.5 is the Democratic
seat share above one half that the curve gives at a 50/50 vote.
rho_t is responsiveness. Intervals come from resampling states.

Writes results/curves_by_year.csv, results/national_bias.csv,
results/seat_bonus_state_year.csv, results/post_check_2026.csv and
results/summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
DER = HERE / "data" / "derived"
RES = HERE / "results"
RNG = np.random.default_rng(20261117)
N_BOOT = 1000


def vote_col(panel: pd.DataFrame) -> pd.Series:
    """Imputed share where district returns exist (1972-2024), raw in 2026."""
    return panel["v_imp"].where(panel["v_imp"].notna(), panel["v_raw"])


def fit_curve(df: pd.DataFrame) -> tuple[float, float]:
    x = logit(df["v"].clip(1e-4, 1 - 1e-4))
    X = sm.add_constant(x.to_numpy())
    n = (df["dem_seats"] + df["rep_seats"]).to_numpy()
    y = df["s"].to_numpy()
    res = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=n).fit()
    return float(res.params[0]), float(res.params[1])


def curves_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in panel.groupby("year"):
        g = g[(g["seats"] >= 2) & g["v"].notna() & g["s"].notna()]
        lam, rho = fit_curve(g)
        boots = []
        idx = np.arange(len(g))
        for _ in range(N_BOOT):
            b = g.iloc[RNG.choice(idx, size=len(idx), replace=True)]
            try:
                boots.append(fit_curve(b))
            except Exception:  # separation in a resample
                continue
        boots = np.array(boots)
        bias = expit(lam) - 0.5
        bb = expit(boots[:, 0]) - 0.5
        rows.append(dict(
            year=int(year), n_states=len(g), seats=int((g.dem_seats + g.rep_seats).sum()),
            lam=lam, rho=rho, bias_at_50=bias,
            bias_lo=float(np.quantile(bb, 0.025)), bias_hi=float(np.quantile(bb, 0.975)),
            rho_lo=float(np.quantile(boots[:, 1], 0.025)),
            rho_hi=float(np.quantile(boots[:, 1], 0.975)),
            bias_seats_435=bias * 435,
        ))
    return pd.DataFrame(rows)


def pooled_rho(panel: pd.DataFrame) -> float:
    g = panel[(panel["seats"] >= 2) & panel["v"].notna() & panel["s"].notna()].copy()
    x = logit(g["v"].clip(1e-4, 1 - 1e-4)).to_numpy()
    yd = pd.get_dummies(g["year"].astype(int), drop_first=False).astype(float).to_numpy()
    X = np.column_stack([x, yd])
    n = (g["dem_seats"] + g["rep_seats"]).to_numpy()
    res = sm.GLM(g["s"].to_numpy(), X, family=sm.families.Binomial(), var_weights=n).fit()
    return float(res.params[0])


def national_uniform_swing(districts: pd.DataFrame) -> pd.DataFrame:
    """National partisan bias from districts, 1972-2024 (uniform national swing)."""
    rows = []
    for year, g in districts.groupby("year"):
        g = g[g["v_imp_d"].notna()]
        v, w = g["v_imp_d"].to_numpy(), g["t_imp"].to_numpy()
        vnat = np.sum(v * w) / np.sum(w)
        seats_at_50 = np.sum(v + (0.5 - vnat) > 0.5)
        # vote share Democrats need for 218 of the districts with a two-party race
        need = 0.5 - np.sort(v - 0.5)[::-1][217] + 0.0 if len(v) >= 218 else np.nan
        rows.append(dict(year=int(year), v_nat=vnat, districts=len(v),
                         dem_seats_at_50=int(seats_at_50),
                         bias_seats=float(seats_at_50 - len(v) / 2),
                         median_district_minus_mean=float(np.median(v) - vnat),
                         dem_vote_needed_for_majority=float(vnat + (0.5 - np.sort(v)[::-1][217]))))
    return pd.DataFrame(rows)


def main() -> dict:
    RES.mkdir(exist_ok=True)
    panel = pd.read_csv(DER / "state_panel.csv")
    panel["v"] = vote_col(panel)
    curves = curves_by_year(panel)
    curves.to_csv(RES / "curves_by_year.csv", index=False)

    rho_star = pooled_rho(panel)
    panel["f_sym"] = expit(rho_star * logit(panel["v"].clip(1e-4, 1 - 1e-4)))
    n2 = panel["dem_seats"] + panel["rep_seats"]
    panel["bonus_share"] = panel["s"] - panel["f_sym"]
    panel["bonus_seats"] = panel["bonus_share"] * n2
    # simplified efficiency gap from state totals (positive = pro-D)
    panel["eg_simple"] = (panel["s"] - 0.5) - 2 * (panel["v"] - 0.5)
    panel["eg_simple_raw"] = (panel["s"] - 0.5) - 2 * (panel["v_raw"] - 0.5)
    panel["f_sym_raw"] = expit(rho_star * logit(panel["v_raw"].clip(1e-4, 1 - 1e-4)))
    panel["bonus_share_raw"] = panel["s"] - panel["f_sym_raw"]
    panel.to_csv(RES / "seat_bonus_state_year.csv", index=False)

    districts = pd.read_csv(DER / "house_districts_imputed_1972_2024.csv")
    nat = national_uniform_swing(districts)
    nat.to_csv(RES / "national_bias.csv", index=False)

    post = {"TX": (53, 37), "FL": (50, 29), "GA": (56, 36), "TN": (45, 11),
            "OK": (41, 0), "SC": (49, 29), "LA": (45, 17), "UT": (50, 25)}
    p26 = panel[(panel["year"] == 2026) & panel["state_po"].isin(post)].copy()
    p26["post_vote"] = p26["state_po"].map(lambda s: post[s][0])
    p26["post_seat"] = p26["state_po"].map(lambda s: post[s][1])
    p26["vote_pct"] = (100 * p26["v_raw"]).round(1)
    p26["seat_pct"] = (100 * p26["s"]).round(1)
    p26["expected_dem_seats_symmetric"] = (p26["f_sym"] * (p26.dem_seats + p26.rep_seats)).round(1)
    p26["proportional_dem_seats"] = (p26["v_raw"] * (p26.dem_seats + p26.rep_seats)).round(1)
    cols = ["state_po", "seats", "dem_seats", "post_vote", "vote_pct", "post_seat", "seat_pct",
            "expected_dem_seats_symmetric", "proportional_dem_seats", "bonus_seats", "drawer"]
    p26[cols].to_csv(RES / "post_check_2026.csv", index=False)

    summary = dict(
        rho_pooled=rho_star,
        curves=curves.round(4).to_dict(orient="records"),
        post_check=p26[cols].round(3).to_dict(orient="records"),
        post_states_bonus_seats=float(p26["bonus_seats"].sum()),
        national=nat.round(4).to_dict(orient="records"),
    )
    (RES / "summary_votes_seats.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    s = main()
    print("pooled rho", round(s["rho_pooled"], 3))
    print(pd.DataFrame(s["curves"])[["year", "n_states", "lam", "rho", "bias_at_50", "bias_lo", "bias_hi", "bias_seats_435"]].to_string())
    print(pd.DataFrame(s["post_check"]).to_string())
    print("bonus seats in the eight states", round(s["post_states_bonus_seats"], 1))
    print(pd.DataFrame(s["national"]).to_string())


def national_symmetry(year: int = 2024, grid=np.round(np.arange(0.40, 0.6001, 0.005), 3)) -> pd.DataFrame:
    """Seats-votes curve of one year's district map under uniform national swing,
    and its mirror image: S_D(v) against 435 - S_D(1 - v)."""
    d = pd.read_csv(DER / "house_districts_imputed_1972_2024.csv")
    g = d[(d.year == year) & d["v_imp_d"].notna()]
    v, w = g["v_imp_d"].to_numpy(), g["t_imp"].to_numpy()
    vnat = np.sum(v * w) / np.sum(w)
    rows = []
    for x in grid:
        dem = int(np.sum(v + (x - vnat) > 0.5))
        mirror = len(v) - int(np.sum(v + ((1 - x) - vnat) > 0.5))
        rows.append(dict(year=year, dem_vote=float(x), dem_seats=dem, rep_seats_if_rep_had_this_vote=mirror,
                         asymmetry=dem - mirror, districts=len(v)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sym = pd.concat([national_symmetry(y) for y in [2016, 2024]])
    sym.to_csv(RES / "national_symmetry.csv", index=False)
    print(sym[sym.dem_vote.isin([0.44, 0.46, 0.5, 0.52, 0.54, 0.56, 0.58])].to_string())
