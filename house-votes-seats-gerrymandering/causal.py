"""Does the party that draws the map cause the seat bonus?

Treatment: legal control of the plan in force, coded in data/map_drawers.csv
as R (Republicans could enact it alone), D (Democrats could), or neutral
(split government, commission or court). K = +1 for D, -1 for R, 0 neutral.

Outcome: the Democratic seat bonus, the Democratic share of a state's seats
minus the share the symmetric cube-law curve gives at the state's vote
(results/seat_bonus_state_year.csv, rho estimated at about 3). Positive =
the plan gives Democrats more seats than a symmetric plan would.

Designs
  1. Two-way fixed effects, state and year, plans in states with 5+ seats.
  2. Stacked event study around every new plan: change in the outcome from
     the last election under the old plan, regressed on the change in K,
     with cycle-by-event-time fixed effects (so national waves cancel).
  3. The 2025-26 mid-decade redraws: change from 2024 to 2026 in the nine
     states whose plan changed, against the states whose plan did not;
     placebo on 2022-2024; permutation p-values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RNG = np.random.default_rng(1972)
MIN_SEATS = 5
K = {"D": 1, "R": -1, "Court": 0, "Commission": 0, "Split": 0}


def load() -> pd.DataFrame:
    p = pd.read_csv(RES / "seat_bonus_state_year.csv")
    p["n2"] = p["dem_seats"] + p["rep_seats"]
    p["k"] = p["drawer"].map(K)
    p["R"] = (p["drawer"] == "R").astype(int)
    p["D"] = (p["drawer"] == "D").astype(int)
    p["y"] = 100 * p["bonus_share"]          # percentage points of the delegation
    p["y_eg"] = 100 * p["eg_simple"]
    p["y_us"] = 100 * p["bias_us"]
    p["y_raw"] = 100 * p["bonus_share_raw"]
    return p


def twfe(p: pd.DataFrame, outcome: str = "y", label: str = "") -> dict:
    d = p[(p["seats"] >= MIN_SEATS) & p["drawer"].notna() & p[outcome].notna()].copy()
    m = smf.wls(f"{outcome} ~ R + D + C(state_po) + C(year)", data=d, weights=d["n2"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["state_po"]})
    out = dict(label=label, outcome=outcome, n=int(m.nobs), states=int(d.state_po.nunique()),
               years=f"{d.year.min()}-{d.year.max()}")
    for v in ["R", "D"]:
        out[v] = float(m.params[v])
        out[v + "_se"] = float(m.bse[v])
        out[v + "_lo"], out[v + "_hi"] = map(float, m.conf_int().loc[v])
        out[v + "_p"] = float(m.pvalues[v])
    out["mean_seats_R"] = float(d.loc[d.R == 1, "n2"].mean())
    out["mean_seats_D"] = float(d.loc[d.D == 1, "n2"].mean())
    return out


def stacked_events(p: pd.DataFrame, outcome: str = "y", kmin: int = -2, kmax: int = 4):
    d = p[(p["seats"] >= MIN_SEATS) & p["drawer"].notna()].sort_values(["state_po", "year"])
    rows = []
    for st, g in d.groupby("state_po"):
        g = g.reset_index(drop=True)
        starts = g.loc[g["plan_id"] != g["plan_id"].shift(), "year"].tolist()
        for i, t0 in enumerate(starts):
            if i == 0:
                continue  # need the previous plan for the baseline
            prev_start, next_start = starts[i - 1], (starts[i + 1] if i + 1 < len(starts) else 9999)
            base = g[g["year"] == t0 - 2]
            if base.empty or base["plan_start"].iloc[0] != prev_start:
                continue
            k_old, k_new = base["k"].iloc[0], g.loc[g.year == t0, "k"].iloc[0]
            y_base = base[outcome].iloc[0]
            for kk in range(kmin, kmax + 1):
                if kk == -1:
                    continue
                yr = t0 + 2 * kk
                row = g[g["year"] == yr]
                if row.empty:
                    continue
                if kk < 0 and row["plan_start"].iloc[0] != prev_start:
                    continue
                if kk >= 0 and yr >= next_start:
                    continue
                rows.append(dict(state_po=st, t0=t0, k_rel=kk, year=yr,
                                 dk=k_new - k_old, k_old=k_old, k_new=k_new,
                                 d_y=row[outcome].iloc[0] - y_base,
                                 w=row["n2"].iloc[0],
                                 low=("low" in set(g.loc[g.year.isin([t0 - 2, yr]), "confidence"]))))
    ev = pd.DataFrame(rows)
    ev = ev[np.isfinite(ev["d_y"])].reset_index(drop=True)
    ev["dk_pos"] = ev["dk"].clip(lower=0)
    ev["dk_neg"] = ev["dk"].clip(upper=0)
    ev["cell"] = ev["t0"].astype(str) + "_" + ev["k_rel"].astype(str)
    return ev


def event_coefs(ev: pd.DataFrame, split: bool = False) -> pd.DataFrame:
    out = []
    for kk, g in ev.groupby("k_rel"):
        if g["dk"].abs().sum() == 0:
            continue
        f = "d_y ~ dk_pos + dk_neg + C(t0)" if split else "d_y ~ dk + C(t0)"
        m = smf.wls(f, data=g, weights=g["w"]).fit(cov_type="cluster",
                                                   cov_kwds={"groups": g["state_po"]})
        terms = ["dk_pos", "dk_neg"] if split else ["dk"]
        for t in terms:
            lo, hi = m.conf_int().loc[t]
            out.append(dict(k_rel=int(kk), term=t, coef=float(m.params[t]), se=float(m.bse[t]),
                            lo=float(lo), hi=float(hi), n=int(m.nobs),
                            events_changing=int((g["dk"] != 0).sum())))
    return pd.DataFrame(out)


def pooled_post(ev: pd.DataFrame, low_ok: bool = True) -> dict:
    g = ev[ev["k_rel"] >= 0] if low_ok else ev[(ev["k_rel"] >= 0) & ~ev["low"]]
    m = smf.wls("d_y ~ dk_pos + dk_neg + C(cell)", data=g, weights=g["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": g["state_po"]})
    pre = ev[ev["k_rel"] < 0] if low_ok else ev[(ev["k_rel"] < 0) & ~ev["low"]]
    mp = smf.wls("d_y ~ dk + C(cell)", data=pre, weights=pre["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": pre["state_po"]})
    res = {}
    for t in ["dk_pos", "dk_neg"]:
        lo, hi = m.conf_int().loc[t]
        res[t] = dict(coef=float(m.params[t]), lo=float(lo), hi=float(hi), p=float(m.pvalues[t]))
    lo, hi = mp.conf_int().loc["dk"]
    res["pre_trend_dk"] = dict(coef=float(mp.params["dk"]), lo=float(lo), hi=float(hi),
                               p=float(mp.pvalues["dk"]), n=int(mp.nobs))
    res["n_post"] = int(m.nobs)
    res["n_events_toward_R"] = int(g.loc[g.dk < 0, ["state_po", "t0"]].drop_duplicates().shape[0])
    res["n_events_toward_D"] = int(g.loc[g.dk > 0, ["state_po", "t0"]].drop_duplicates().shape[0])
    return res


TREATED_2026 = {"TX": "R", "FL": "R", "NC": "R", "OH": "R", "TN": "R", "LA": "R", "AL": "R",
                "CA": "D", "UT": "Court"}


def redraw_2026(p: pd.DataFrame, outcome: str = "y_raw", n_perm: int = 20000) -> dict:
    a = p[p.year == 2024].set_index("state_po")
    b = p[p.year == 2026].set_index("state_po")
    common = [s for s in b.index if s in a.index and b.loc[s, "seats"] >= 2
              and np.isfinite(b.loc[s, outcome]) and np.isfinite(a.loc[s, outcome])]
    d = pd.DataFrame(dict(d_y=b.loc[common, outcome] - a.loc[common, outcome],
                          seats=b.loc[common, "n2"], y24=a.loc[common, outcome],
                          y26=b.loc[common, outcome], v24=a.loc[common, "v_raw"],
                          v26=b.loc[common, "v_raw"]))
    d["group"] = [TREATED_2026.get(s, "control") for s in d.index]
    d["R"] = (d.group == "R").astype(int)
    d["D"] = (d.group == "D").astype(int)
    d["C"] = (d.group == "Court").astype(int)
    m = smf.wls("d_y ~ R + D + C", data=d, weights=d["seats"]).fit(cov_type="HC1")

    def wmean(x, w):
        return float(np.sum(x * w) / np.sum(w))
    ctrl = d[d.group == "control"]
    obs = wmean(d.loc[d.R == 1, "d_y"], d.loc[d.R == 1, "seats"]) - wmean(ctrl.d_y, ctrl.seats)
    # permutation: which 7 of the R-redraw-plus-control states are "treated"
    pool = d[d.group.isin(["R", "control"])]
    nR = int(pool.R.sum())
    perm = []
    for _ in range(n_perm):
        idx = RNG.choice(len(pool), size=nR, replace=False)
        mask = np.zeros(len(pool), bool)
        mask[idx] = True
        perm.append(wmean(pool.d_y[mask], pool.seats[mask]) - wmean(pool.d_y[~mask], pool.seats[~mask]))
    perm = np.array(perm)
    p_two = float(np.mean(np.abs(perm) >= abs(obs)))
    out = dict(
        n_states=len(d), n_control=int((d.group == "control").sum()),
        R=float(m.params["R"]), R_lo=float(m.conf_int().loc["R", 0]),
        R_hi=float(m.conf_int().loc["R", 1]), R_p_hc1=float(m.pvalues["R"]),
        D_CA=float(m.params["D"]), Court_UT=float(m.params["C"]),
        R_diff_weighted=obs, R_perm_p=p_two,
        control_mean_change=wmean(ctrl.d_y, ctrl.seats),
        by_state=d.round(3).reset_index().rename(columns={"index": "state_po"}).to_dict(orient="records"),
    )
    # seats: effect in seats for the seven R-redraw states
    out["R_seats_total"] = float(m.params["R"] / 100 * d.loc[d.R == 1, "seats"].sum())
    return out


def placebo_2024(p: pd.DataFrame, outcome: str = "y_raw") -> dict:
    """Same states, one cycle earlier, when their plans did not change (2022 -> 2024)."""
    unchanged_2024 = ["TX", "FL", "OH", "TN", "CA", "UT"]
    a = p[p.year == 2022].set_index("state_po")
    b = p[p.year == 2024].set_index("state_po")
    changed_2024 = ["AL", "LA", "NC", "NY", "GA"]
    common = [s for s in b.index if s in a.index and b.loc[s, "seats"] >= 2 and s not in changed_2024
              and np.isfinite(b.loc[s, outcome]) and np.isfinite(a.loc[s, outcome])]
    d = pd.DataFrame(dict(d_y=b.loc[common, outcome] - a.loc[common, outcome],
                          seats=b.loc[common, "n2"]))
    d["R"] = d.index.isin([s for s in unchanged_2024 if TREATED_2026[s] == "R"]).astype(int)
    d = d[~d.index.isin(["CA", "UT"])]
    m = smf.wls("d_y ~ R", data=d, weights=d["seats"]).fit(cov_type="HC1")
    return dict(R=float(m.params["R"]), R_lo=float(m.conf_int().loc["R", 0]),
                R_hi=float(m.conf_int().loc["R", 1]), p=float(m.pvalues["R"]), n=len(d))


def main() -> dict:
    p = load()
    res = {}
    upto24 = p[p.year <= 2024]
    # Primary outcome: uniform-swing partisan bias from district returns (1972-2024)
    res["twfe_us_all"] = twfe(upto24, "y_us", "uniform-swing bias, 1972-2024")
    res["twfe_us_pre2000"] = twfe(upto24[upto24.year < 2000], "y_us", "uniform-swing bias, 1972-1998")
    res["twfe_us_post2000"] = twfe(upto24[upto24.year >= 2002], "y_us", "uniform-swing bias, 2002-2024")
    res["twfe_us_no_low"] = twfe(upto24[upto24.confidence != "low"], "y_us",
                                 "uniform-swing bias, low-confidence codes dropped")
    res["twfe_mm"] = twfe(upto24.assign(y_mm=100 * upto24["mean_median"]), "y_mm",
                          "median-minus-mean, 1972-2024")
    res["twfe_eg_full"] = twfe(upto24.assign(y_egf=100 * upto24["eg"]), "y_egf",
                               "efficiency gap from district votes, 1972-2024")
    # State-level seat bonus (the only measure available for 2026)
    res["twfe_bonus_all"] = twfe(p, "y", "seat bonus vs cube-law curve, 1972-2026")
    res["twfe_bonus_post2000"] = twfe(p[p.year >= 2002], "y", "seat bonus, 2002-2026")
    res["twfe_bonus_raw"] = twfe(p, "y_raw", "seat bonus with unimputed vote, 1972-2026")
    res["twfe_eg_simple"] = twfe(p, "y_eg", "simplified efficiency gap, 1972-2026")

    ev = stacked_events(upto24, "y_us")
    ev.to_csv(RES / "event_study_rows.csv", index=False)
    event_coefs(ev).to_csv(RES / "event_study_coefs.csv", index=False)
    event_coefs(ev, split=True).to_csv(RES / "event_study_coefs_split.csv", index=False)
    event_coefs(ev[ev.t0 >= 1992]).to_csv(RES / "event_study_coefs_post1992.csv", index=False)
    res["event_us_all"] = pooled_post(ev)
    res["event_us_post1992"] = pooled_post(ev[ev.t0 >= 1992])
    res["event_us_pre1992"] = pooled_post(ev[ev.t0 < 1992])
    res["event_us_no_low"] = pooled_post(ev, low_ok=False)
    ev_b = stacked_events(p, "y")
    res["event_bonus_all"] = pooled_post(ev_b)

    res["redraw_2026_bonus"] = redraw_2026(p)
    res["placebo_2022_2024_bonus"] = placebo_2024(p)
    sizes = {}
    for yr in [2012, 2016, 2022, 2024]:
        g = p[(p.year == yr) & (p.seats >= MIN_SEATS)]
        sizes[yr] = dict(
            seats_R_drawn=int(g.loc[g.R == 1, "n2"].sum()),
            seats_D_drawn=int(g.loc[g.D == 1, "n2"].sum()),
            implied_R_seats=float(res["twfe_us_post2000"]["R"] / 100 * g.loc[g.R == 1, "n2"].sum()),
            implied_D_seats=float(res["twfe_us_post2000"]["D"] / 100 * g.loc[g.D == 1, "n2"].sum()))
    res["national_sizes_post2000"] = sizes
    (RES / "causal_summary.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    r = main()
    for k, v in r.items():
        if k.startswith("twfe"):
            print(f"{v['label']:<52} R {v['R']:+6.2f} [{v['R_lo']:+.2f},{v['R_hi']:+.2f}] p={v['R_p']:.3f}"
                  f" | D {v['D']:+6.2f} [{v['D_lo']:+.2f},{v['D_hi']:+.2f}] p={v['D_p']:.3f}  n={v['n']} states={v['states']}")
    for k in ["event_us_all", "event_us_post1992", "event_us_pre1992", "event_us_no_low", "event_bonus_all"]:
        e = r[k]
        print(k, "toward D", round(e["dk_pos"]["coef"], 2), [round(e["dk_pos"]["lo"], 2), round(e["dk_pos"]["hi"], 2)],
              "toward R", round(e["dk_neg"]["coef"], 2), [round(e["dk_neg"]["lo"], 2), round(e["dk_neg"]["hi"], 2)],
              "pre", round(e["pre_trend_dk"]["coef"], 2), [round(e["pre_trend_dk"]["lo"], 2), round(e["pre_trend_dk"]["hi"], 2)],
              "events R/D", e["n_events_toward_R"], e["n_events_toward_D"])
    print(pd.read_csv(RES / "event_study_coefs.csv").round(2).to_string())
    print(pd.read_csv(RES / "event_study_coefs_post1992.csv").round(2).to_string())
    rr = r["redraw_2026_bonus"]
    print({k: v for k, v in rr.items() if k != "by_state"})
    print("placebo", r["placebo_2022_2024_bonus"])
    print(json.dumps(r["national_sizes_post2000"]))
