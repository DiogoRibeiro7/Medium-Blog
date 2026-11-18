"""Every number in the article, printed in order, plus the tables in results/.

python coach_numbers.py > output.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

import coach_core as cc
import simulate

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def ci(lo_hi, scale=1.0, fmt="{:+.2f}"):
    lo, hi = lo_hi
    return f"[{fmt.format(lo * scale)}, {fmt.format(hi * scale)}]"


def main():
    RESULTS.mkdir(exist_ok=True)
    m, sv, nj = cc.load()
    dp = cc.decision_points(m, sv, nj)
    e = cc.eligible(dp)

    print("== Data")
    print(
        f"league matches: {m.game_id.nunique()} in {m.groupby(['league', 'season']).ngroups} league-seasons "
        f"({m.season.min()}/{(m.season.min() + 1) % 100:02d} to {m.season.max()}/{(m.season.max() + 1) % 100:02d}, "
        "Ligue 1 2019/20 left out)"
    )
    board = dp[(dp.change == 1) & ~dp.returns]
    print(f"in-season manager changes (a new name in the next league match, not a temporary absence): {len(board)}")
    print(
        f"  of which the outgoing manager had led at least {cc.MIN_TENURE} league matches in a row: "
        f"{(board.tenure >= cc.MIN_TENURE).sum()}"
    )
    print(
        "temporary absences dropped (the manager led the club again later that season): "
        f"{int(dp[(dp.change == 1)].returns.sum())}"
    )
    print(
        f"decision points in the window (after match {cc.FIRST_DECISION} to {cc.HORIZON} before the end), "
        f"established manager: {len(e)}"
    )
    print(f"changes among them: {int(e.change.sum())}; weekly rate {e.change.mean():.4f}")
    print(
        f"changes where the outgoing manager led another club within {cc.NEXT_JOB_DAYS} days: "
        f"{int((e.change.astype(bool) & e.left_for_job).sum())}"
    )
    by_lg = e[e.change == 1].groupby("league").size().rename(cc.LEAGUE_NAMES)
    print("changes by league: " + ", ".join(f"{k} {v}" for k, v in by_lg.items()))
    print(f"league-average points per match (all clubs): {m.pts.mean():.3f}")
    print(f"in-season changes per season across the five leagues: {len(board) / m.season.nunique():.1f}")
    final = dp.groupby(["league", "season", "club_id"]).agg(final_ppg=("ppg", "last"), worst5=("last5", "min"))
    avg = final[final.final_ppg.between(1.30, 1.45)]
    print(
        f"clubs that finished a season on 1.30 to 1.45 points per match (about average): {len(avg)}; "
        f"share with at least one run of five league matches bringing 3 points or fewer: {(avg.worst5 <= 3).mean():.3f}"
    )

    print("\n== Who gets a change (logistic model, one row per decision point)")
    fit, logit = cc.fit_hazard(e)
    print(f"AUC {cc.auc(e.change, logit):.3f}")
    print(
        f"odds of a change after a defeat relative to after a win in the latest match, all else equal: "
        f"{np.exp(-3 * fit.params['last1']):.1f}"
    )
    or_last5 = np.exp(fit.params["last5"] + fit.params["last5_x_value"] * 0)
    print(f"odds ratio per extra point in the last five (the latest match included), median-value club: {or_last5:.3f}")
    lost = e[e.change == 1].pts.eq(0).mean()
    print(
        f"share of changes that came straight after a defeat: {lost:.3f}; share of all decision points after "
        f"a defeat: {e.pts.eq(0).mean():.3f}"
    )
    print(
        f"odds ratio for one unit of log squad value above the league median, at zero recent points: "
        f"{np.exp(fit.params['value_rel']):.2f}"
    )
    rate = e.groupby(pd.cut(e.last5, [-1, 2, 4, 6, 8, 10, 15])).change.agg(["mean", "size"])
    print("weekly change rate by points from the last five:")
    for idx, r in rate.iterrows():
        print(f"  {idx}: {r['mean']:.4f} (n={int(r['size'])})")
    drop = e.groupby("in_drop").change.mean()
    print(f"weekly rate in the bottom three {drop[1]:.4f}, elsewhere {drop[0]:.4f}")
    bins = pd.cut(
        e.last5,
        [-1, 1, 3, 5, 7, 9, 11, 15],
        labels=["0-1", "2-3", "4-5", "6-7", "8-9", "10-11", "12-15"],
    )
    by = e.groupby([bins, e.in_drop], observed=True).change.agg(["mean", "size"]).reset_index()
    by.columns = ["last5", "in_drop", "rate", "n"]
    by.to_csv(RESULTS / "change_rate_by_form.csv", index=False)

    print("\n== The same-spot comparison")
    pairs = cc.match(e, logit)
    t = pairs.t.unique()
    per = cc.matched_effect(e, pairs)
    print(
        f"changes matched: {len(t)} of {int(e.change.sum())}; comparison clubs per change {cc.CONTROLS}; "
        f"distinct comparison decision points {pairs.c.nunique()}"
    )
    bal = []
    for col in cc.MATCH_ON + ["k", "in_drop", "gap_safety", "tenure"]:
        a = e[col].to_numpy(float)[t].mean()
        b = pairs.assign(v=e[col].to_numpy(float)[pairs.c]).groupby("t").v.mean().mean()
        bal.append((col, a, b, (a - b) / e[col].std()))
    pd.DataFrame(bal, columns=["variable", "changed", "kept", "std_diff"]).to_csv(RESULTS / "balance.csv", index=False)
    print("balance (changed / kept / standardised difference):")
    for col, a, b, d in bal:
        print(f"  {col:11s} {a:8.3f} {b:8.3f} {d:+.3f}")

    yt, yc = per.yt.mean(), per.yc.mean()
    last5_t = e.last5.to_numpy(float)[t].mean()
    last5_c = pairs.assign(v=e.last5.to_numpy(float)[pairs.c]).groupby("t").v.mean().mean()
    print(f"points per match, last five before the decision: changed {last5_t / 5:.3f}, kept {last5_c / 5:.3f}")
    print(f"points per match, next ten: changed {yt / 10:.3f}, kept {yc / 10:.3f}")
    print(
        f"before/after for the changed clubs (next ten minus twice the last five, points over ten matches): "
        f"{yt - 2 * last5_t:+.2f}"
    )
    print(f"before/after for the kept clubs: {yc - 2 * last5_c:+.2f}")

    def diff(ts):
        return per.loc[ts, "diff"].mean()

    est = per["diff"].mean()
    print(
        f"same-spot estimate, points over the next ten matches: {est:+.2f} "
        f"95% interval {ci(cc.bootstrap(e, pairs, diff))}"
    )
    print(f"share of changes whose club did better than its comparison clubs: {(per['diff'] > 0).mean():.3f}")

    print("\n== Yardsticks")
    rows_t, rows_c = e.row.to_numpy()[t], e.row.to_numpy()[pairs.c]
    kept = cc.kept_through(dp, rows_c)
    print(f"comparison clubs that changed manager anyway within the next ten matches: {1 - kept.mean():.3f}")
    y = e.next10.to_numpy(float)
    pk = pairs.assign(yc=y[pairs.c], kept=kept)
    surv = pk[pk.kept].groupby("t").yc.mean()
    ts_s = surv.index.to_numpy()
    surv_diff = y[ts_s] - surv.to_numpy()
    print(f"changes with at least one comparison club that kept its manager all ten matches: {len(ts_s)}")
    print(
        f"yardstick 'kept him and stuck with him': changed minus kept-for-good {surv_diff.mean():+.2f} points over ten"
    )
    s_per = pd.Series(surv_diff, index=ts_s)
    ci_surv = cc.bootstrap(e, pairs[pairs.t.isin(ts_s)], lambda ts: s_per.loc[ts].mean())
    print(f"  95% interval {ci(ci_surv)}")
    own = y[t] - 2 * e.last5.to_numpy(float)[t]
    own_s = pd.Series(own, index=t)
    ci_own = cc.bootstrap(e, pairs, lambda ts: own_s.loc[ts].mean())
    print(f"yardstick 'own last five': {own.mean():+.2f} points over ten, 95% interval {ci(ci_own)}")
    ci_est = cc.bootstrap(e, pairs, diff)
    pd.DataFrame(
        {
            "yardstick": ["own", "stuck", "same_spot"],
            "estimate": [own.mean(), surv_diff.mean(), est],
            "lo": [ci_own[0], ci_surv[0], ci_est[0]],
            "hi": [ci_own[1], ci_surv[1], ci_est[1]],
        }
    ).to_csv(RESULTS / "yardsticks.csv", index=False)

    print("\n== Timing of any gain")
    pt = cc.event_paths(dp, rows_t)
    pc = cc.event_paths(dp, rows_c).assign(t=pairs.t.to_numpy()).groupby("t").mean()
    paths = pd.DataFrame({"changed": pt.mean(), "kept": pc.mean()})
    paths.index.name = "match"
    paths.to_csv(RESULTS / "event_paths.csv")
    for a, b in [(1, 3), (4, 10)]:
        cols = list(range(a, b + 1))
        d = pd.Series(
            pt[cols].sum(axis=1).to_numpy() - pc.loc[t, cols].sum(axis=1).to_numpy(),
            index=t,
        )
        print(
            f"matches {a}-{b} after: changed minus kept {d.mean():+.2f} points, "
            f"95% interval {ci(cc.bootstrap(e, pairs, lambda ts, d=d: d.loc[ts].mean()))}"
        )
    pre = list(range(-14, -9))
    ptp = cc.event_paths(dp, rows_t, before=14, after=0)[pre]
    pcp = cc.event_paths(dp, rows_c, before=14, after=0)[pre].assign(t=pairs.t.to_numpy()).groupby("t").mean()
    full = ptp.notna().all(axis=1).to_numpy() & pcp.loc[t].notna().all(axis=1).to_numpy()
    tp = t[full]
    d = pd.Series(ptp[full].sum(axis=1).to_numpy() - pcp.loc[tp].sum(axis=1).to_numpy(), index=tp)
    pre_diff, pre_ci = (
        d.mean(),
        cc.bootstrap(e, pairs[pairs.t.isin(tp)], lambda ts: d.loc[ts].mean()),
    )
    print(
        f"matches 14-10 before, outside every matching window ({len(tp)} changes made after match 14): "
        f"changed minus kept {pre_diff:+.2f} points, 95% interval {ci(pre_ci)}"
    )
    rest = cc.matched_effect(e, pairs, "rest_ppg")
    print(
        f"rest of the season, points per match: changed {rest.yt.mean():.3f}, kept {rest.yc.mean():.3f}, "
        f"difference {rest['diff'].mean():+.3f}, 95% interval "
        f"{ci(cc.bootstrap(e, pairs, lambda ts: rest.loc[ts, 'diff'].mean()), fmt='{:+.3f}')}"
    )
    remaining = (e.n_matches - e.k).to_numpy(float)[t]
    print(
        f"matches left in the season after a change: mean {remaining.mean():.1f}; the difference over them: "
        f"{(rest['diff'].to_numpy() * remaining).mean():+.2f} points"
    )

    print("\n== Relegation")
    near = np.flatnonzero(e.gap_safety.to_numpy() <= 3)
    tn = np.intersect1d(t, near)
    rel = cc.matched_effect(e, pairs[pairs.t.isin(tn)], "bottom3")
    print(f"changes made within 3 points of the bottom three or inside it: {len(tn)}")
    print(
        f"finished in the bottom three: changed {rel.yt.mean():.3f}, kept {rel.yc.mean():.3f}, "
        f"difference {rel['diff'].mean():+.3f}, 95% interval "
        f"{ci(cc.bootstrap(e, pairs[pairs.t.isin(tn)], lambda ts: rel.loc[ts, 'diff'].mean()), fmt='{:+.3f}')}"
    )

    print("\n== Where the answer might differ (95% intervals; eight groups, so one could look notable by chance)")
    E = e.iloc[per.index]
    slope = np.polyfit(e.sot_share, e.ppg, 1)
    luck = E.ppg.to_numpy() - np.polyval(slope, E.sot_share.to_numpy())
    splits = {
        "all changes": np.ones(len(E), bool),
        "inside the bottom three": E.in_drop.to_numpy() == 1,
        "outside the bottom three": E.in_drop.to_numpy() == 0,
        "first half of the season": E.share.to_numpy() <= 0.5,
        "second half of the season": E.share.to_numpy() > 0.5,
        "squad worth more than the league median": E.value_rel.to_numpy() > 0,
        "squad worth less than the league median": E.value_rel.to_numpy() <= 0,
        "results below what shots on target predict": luck < -0.2,
        "results in line with shots on target": luck >= -0.2,
    }
    sub = []
    for name, mask in splits.items():
        ts = per.index.to_numpy()[mask]
        lo, hi = cc.bootstrap(e, pairs[pairs.t.isin(ts)], lambda x: per.loc[x, "diff"].mean())
        val = per.loc[ts, "diff"].mean()
        sub.append((name, len(ts), val, lo, hi))
        print(f"  {name:45s} n={len(ts):3d} {val:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    pd.DataFrame(sub, columns=["group", "changes", "estimate", "lo", "hi"]).to_csv(
        RESULTS / "subgroups.csv", index=False
    )
    rich = set(per.index.to_numpy()[splits["squad worth more than the league median"]])

    def gap(ts):
        richer = per.loc[[x for x in ts if x in rich], "diff"].mean()
        return richer - per.loc[[x for x in ts if x not in rich], "diff"].mean()

    print(f"richer minus poorer: {gap(list(per.index)):+.2f}, 95% interval {ci(cc.bootstrap(e, pairs, gap))}")
    richer = splits["squad worth more than the league median"]
    in3 = E.in_drop.to_numpy() == 1
    print(
        f"changes made inside the bottom three: {in3[richer].mean():.3f} of the richer clubs' changes, "
        f"{in3[~richer].mean():.3f} of the poorer clubs'"
    )
    for lg_code, name in cc.LEAGUE_NAMES.items():
        mask = E.league.to_numpy() == lg_code
        print(f"  {name:15s} n={mask.sum():3d} {per['diff'].to_numpy()[mask].mean():+.2f}")

    print("\n== Checks")
    f, p2, per2 = cc.placebo(e, logit, pairs)
    ci_pl = cc.bootstrap(f, p2, lambda ts: per2.loc[ts, "diff"].mean())
    print(
        f"placebo (the nearest kept club of each change treated as if it had changed): "
        f"{per2['diff'].mean():+.2f}, 95% interval {ci(ci_pl)}, n={len(per2)}"
    )
    b, lo, hi = cc.regression_estimate(e)
    print(f"linear model on all decision points, same state plus season effects: {b:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    checks = [
        ("main", est, *ci_est),
        ("placebo", per2["diff"].mean(), *ci_pl),
        ("regression", b, lo, hi),
        ("pre_period", pre_diff, *pre_ci),
    ]
    pd.DataFrame(checks, columns=["check", "estimate", "lo", "hi"]).to_csv(RESULTS / "checks.csv", index=False)
    sens = []
    variants = {
        "1 comparison club": {"controls": 1},
        "3 comparison clubs": {"controls": 3},
        "10 comparison clubs": {"controls": 10},
        "caliper 0.1 sd": {"caliper_sd": 0.1},
        "caliper 0.5 sd": {"caliper_sd": 0.5},
        "stage within one match": {"max_share_gap": 0.03},
        "distance matching only": {"exact": ("in_drop",)},
    }
    for name, kw in variants.items():
        pv = cc.match(e, logit, **kw)
        val = cc.matched_effect(e, pv)["diff"].mean()
        sens.append((name, pv.t.nunique(), val))
    no_job = e[~(e.change.astype(bool) & e.left_for_job)].reset_index(drop=True)
    _, lg_nj = cc.fit_hazard(no_job)
    pv = cc.match(no_job, lg_nj)
    sens.append(
        (
            "without quick moves to another club",
            pv.t.nunique(),
            cc.matched_effect(no_job, pv)["diff"].mean(),
        )
    )
    for name, n, val in sens:
        print(f"  {name:40s} n={n:3d} {val:+.2f}")
    pd.DataFrame(sens, columns=["variant", "changes", "estimate"]).to_csv(RESULTS / "sensitivity.csv", index=False)

    nulls = np.array([simulate.estimate(*simulate.synthetic(effect=0.0, seed=seed))[0] for seed in range(1, 11)])
    print(
        f"ten simulated 30-season leagues in which a change does nothing: same-spot estimate mean {nulls.mean():+.2f}, "
        f"range {nulls.min():+.2f} to {nulls.max():+.2f}, "
        f"standard error of the mean {nulls.std(ddof=1) / np.sqrt(10):.2f}"
    )
    drift = []
    for seed in range(1, 11):
        sm_, sv_, nj_ = simulate.synthetic(effect=0.0, seed=seed)
        e_ = cc.eligible(cc.decision_points(sm_, sv_, nj_))
        _, lg_ = cc.fit_hazard(e_)
        drift.append(cc.matched_effect(e_, cc.match(e_, lg_, exact=("in_drop",)))["diff"].mean())
    print(f"the same leagues with distance matching only (no exact match on recent form): mean {np.mean(drift):+.2f}")

    print("\n== Hidden information")
    bounce = (y[t] - 2 * e.last5.to_numpy(float)[t]).mean()
    explained = bounce - est
    print(
        f"before/after gain {bounce:+.2f}; same-spot estimate {est:+.2f}; "
        f"the observed state accounts for {explained:.2f} points"
    )
    for target in (1.0, 2.0, 3.0):
        need = target - est
        print(
            f"  for a true gain of {target:.0f} points over ten, what the board knew and the data do not would have "
            f"to cost the changed clubs {need:.2f} points: "
            f"{need / explained:.0%} of what the observed state accounts for"
        )

    out = e.iloc[per.index][
        [
            "league",
            "season",
            "club",
            "k",
            "date",
            "manager",
            "next_manager",
            "pos",
            "n_clubs",
            "last5",
            "ppg",
            "value_rel",
        ]
    ].copy()
    out["league"] = out.league.map(cc.LEAGUE_NAMES)
    out["next10_changed"], out["next10_kept_mean"], out["difference"] = (
        per.yt.values,
        per.yc.values,
        per["diff"].values,
    )
    out.to_csv(RESULTS / "changes.csv", index=False, float_format="%.3f")

    print("\n== Two remembered changes")
    print(f"changes that did 10 or more points worse than their comparison clubs: {(out.difference <= -10).sum()}")
    for club, season in [("Bayern Munich", 2019), ("Manchester United", 2018)]:
        r = out[(out.club == club) & (out.season == season)].iloc[0]
        print(
            f"{club} {season}/{(season + 1) % 100:02d}: {r.manager} -> {r.next_manager} after match {r.k} "
            f"({r.date:%d %b %Y}), place {r.pos}, {r.last5:.0f} points from the last five; next ten: "
            f"{r.next10_changed:.0f} points, comparison clubs {r.next10_kept_mean:.1f}, difference {r.difference:+.1f}"
        )
        print(
            "  share of all changes with a difference at least this large: "
            f"{(out.difference >= r.difference).mean():.3f}"
        )


if __name__ == "__main__":
    main()
