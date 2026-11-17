"""Write every number quoted in the article to results/summary.txt."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DER = HERE / "data" / "derived"


def f(x, nd=1, sign=False):
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def main() -> str:
    vs = json.loads((RES / "summary_votes_seats.json").read_text())
    ca = json.loads((RES / "causal_summary.json").read_text())
    cf = json.loads((RES / "counterfactual_2026.json").read_text())
    se = json.loads((RES / "sensitivity.json").read_text())
    dist = pd.read_csv(DER / "house_districts_raw_1972_2024.csv")
    seats = dist.groupby("year")["winner"].value_counts().unstack(fill_value=0)
    L = []
    L.append("DATA")
    L.append(f"district rows 1972-2024: {len(dist)}; elections: {dist.year.nunique()}")
    L.append("seats by year (D/R/I): " + "; ".join(
        f"{y} {r.get('D', 0)}/{r.get('R', 0)}/{r.get('I', 0)}" for y, r in seats.iterrows()))
    L.append(f"uncontested districts by year: " + "; ".join(
        f"{y} {int((~g.contested).sum())}" for y, g in dist.groupby("year")))
    L.append("")
    L.append("THE POST'S EIGHT STATES, 2026 (unofficial)")
    for r in vs["post_check"]:
        L.append(f"{r['state_po']}: vote {r['vote_pct']}% (post {r['post_vote']}%), seats {r['dem_seats']}/{r['seats']} = "
                 f"{r['seat_pct']}% (post {r['post_seat']}%), symmetric curve {r['expected_dem_seats_symmetric']} seats, "
                 f"proportional {r['proportional_dem_seats']}, gap {f(r['bonus_seats'], 1, True)}, drawer {r['drawer']}")
    L.append(f"sum of gaps, eight states: {f(vs['post_states_bonus_seats'], 1, True)} seats")
    L.append(f"pooled responsiveness rho (year intercepts): {vs['rho_pooled']:.3f}")
    L.append("")
    L.append("CROSS-STATE CURVE BY YEAR (Democratic seat share at a 50/50 vote; responsiveness)")
    for r in vs["curves"]:
        L.append(f"{r['year']}: {100 * (0.5 + r['bias_at_50']):.1f}% [{100 * (0.5 + r['bias_lo']):.1f}, "
                 f"{100 * (0.5 + r['bias_hi']):.1f}], rho {r['rho']:.2f}, states {r['n_states']}")
    L.append("")
    L.append("NATIONAL UNIFORM-SWING BIAS (seats above 217.5 at 50/50; D vote needed for 218)")
    for r in vs["national"]:
        L.append(f"{r['year']}: {f(r['bias_seats'], 1, True)}; national D share {100 * r['v_nat']:.1f}%; "
                 f"needed {100 * r['dem_vote_needed_for_majority']:.1f}%")
    L.append("")
    p = pd.read_csv(RES / "seat_bonus_state_year.csv")
    p = p[p.seats >= 2]
    p["grp"] = p.drawer.map({"R": "R", "D": "D", "Court": "N", "Commission": "N", "Split": "N"})
    p["us_seats"] = p.bias_us * p.seats
    t = p[p.year <= 2024].pivot_table(index="year", columns="grp", values="us_seats", aggfunc="sum")
    L.append("STATE-LEVEL UNIFORM-SWING BIAS SUMMED BY DRAWER (seats)")
    for y, r in t.iterrows():
        L.append(f"{y}: D {r.get('D', float('nan')):+.1f}, neutral {r.get('N', float('nan')):+.1f}, R {r.get('R', float('nan')):+.1f}")
    b = p.pivot_table(index="year", columns="grp", values="bonus_seats", aggfunc="sum")
    L.append("SEAT BONUS VS CUBE-LAW CURVE SUMMED BY DRAWER (seats; state totals)")
    for y in [2012, 2016, 2018, 2022, 2024, 2026]:
        r = b.loc[y]
        L.append(f"{y}: D {r.get('D'):+.1f}, neutral {r.get('N'):+.1f}, R {r.get('R'):+.1f}, all {p[p.year == y].bonus_seats.sum():+.1f}")
    L.append("")
    L.append("CAUSAL: TWO-WAY FIXED EFFECTS (points of the delegation; 95% CI; states clustered)")
    for k, v in ca.items():
        if k.startswith("twfe"):
            L.append(f"{v['label']}: R {v['R']:+.2f} [{v['R_lo']:+.2f}, {v['R_hi']:+.2f}] p={v['R_p']:.3f}; "
                     f"D {v['D']:+.2f} [{v['D_lo']:+.2f}, {v['D_hi']:+.2f}] p={v['D_p']:.3f}; n={v['n']}, states={v['states']}")
    L.append("CAUSAL: STACKED EVENT STUDY (uniform-swing bias)")
    for k in ["event_us_all", "event_us_post1992", "event_us_pre1992", "event_us_no_low", "event_bonus_all"]:
        e = ca[k]
        L.append(f"{k}: toward D {e['dk_pos']['coef']:+.2f} [{e['dk_pos']['lo']:+.2f}, {e['dk_pos']['hi']:+.2f}], "
                 f"toward R {-e['dk_neg']['coef']:+.2f} [{-e['dk_neg']['hi']:+.2f}, {-e['dk_neg']['lo']:+.2f}] "
                 f"(signed as change in D bias), pre-trend {e['pre_trend_dk']['coef']:+.2f} "
                 f"[{e['pre_trend_dk']['lo']:+.2f}, {e['pre_trend_dk']['hi']:+.2f}]; events toward R {e['n_events_toward_R']}, toward D {e['n_events_toward_D']}")
    ev = pd.read_csv(RES / "event_study_coefs.csv")
    ev92 = pd.read_csv(RES / "event_study_coefs_post1992.csv")
    for name, df in [("all events", ev), ("from 1992", ev92)]:
        L.append(f"event-time coefficients, {name}: " + "; ".join(
            f"k={int(r.k_rel)} {r.coef:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] n={int(r.n)}" for r in df.itertuples()))
    rows = pd.read_csv(RES / "event_study_rows.csv")
    L.append(f"events: {rows[['state_po', 't0']].drop_duplicates().shape[0]} new plans, "
             f"{rows[rows.dk != 0][['state_po', 't0']].drop_duplicates().shape[0]} with a change in control")
    L.append("national implied seats (post-2000 TWFE x seats drawn): " + json.dumps(ca["national_sizes_post2000"]))
    L.append("sensitivity: " + json.dumps(se))
    L.append("")
    L.append("2026: OLD MAP PROJECTED AT THE 2026 SWING")
    for k, v in cf.items():
        L.append(k + ": " + json.dumps({a: b for a, b in v.items() if a != "by_state"}))
    for r in cf["redraw_2026"]["by_state"]:
        if r["status"] == "redrawn":
            L.append(f"  {r['state_po']}: old map {r['projected_old_map']} D seats at swing {100 * r['swing']:+.1f} pts; actual {r['actual']}; gap {r['gap']:+d}")
    rb = ca["redraw_2026_bonus"]
    L.append(f"2024->2026 seat-bonus DiD, seven R redraws vs kept maps: {rb['R']:+.2f} pts "
             f"[{rb['R_lo']:+.2f}, {rb['R_hi']:+.2f}] HC1 p={rb['R_p_hc1']:.3f}, permutation p={rb['R_perm_p']:.3f}; "
             f"kept maps mean change {rb['control_mean_change']:+.2f}")
    pl = ca["placebo_2022_2024_bonus"]
    L.append(f"placebo 2022->2024 (TX, FL, OH, TN): {pl['R']:+.2f} [{pl['R_lo']:+.2f}, {pl['R_hi']:+.2f}] p={pl['p']:.2f}")
    txt = "\n".join(L) + "\n"
    (RES / "summary.txt").write_text(txt)
    return txt


if __name__ == "__main__":
    print(main())
