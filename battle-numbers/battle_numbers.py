"""Every number in the article, printed in order, plus the tables in results/.

python battle_numbers.py > output.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import battle_core as bc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BINS = [-5, -0.69, -0.29, 0.0, 0.29, 0.69, 1.1, 5]
BIN_LABELS = ["under 1:2", "1:2 to 3:4", "3:4 to 1:1", "1:1 to 4:3", "4:3 to 2:1", "2:1 to 3:1", "over 3:1"]


def ci(lo_hi, fmt="{:.2f}"):
    lo, hi = lo_hi
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def slope(d):
    return bc.logit(d.win, d.lr).params[1]


def main():
    # bootstrap resamples of small eras sometimes separate perfectly; their slopes are kept as they come
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

    warnings.simplefilter("ignore", PerfectSeparationWarning)
    warnings.simplefilter("ignore", ConvergenceWarning)
    RESULTS.mkdir(exist_ok=True)
    b = bc.load()
    links = pd.read_csv(bc.DATA / "commander_links.csv")
    d = bc.decisive(b)

    print("== Data")
    print(
        f"battles: {len(b)}, fought {b.year.min()} to {b.year.max()}, in {b.war_key.nunique()} wars and "
        f"{b.campgn.nunique()} campaigns; after 1973: {', '.join(b[b.year > 1973].name.str.title())} ({b.year.max()})"
    )
    print(
        f"from the HERO studies (1984): {int(b.is_hero.sum())}; "
        f"added in CDB90 (1990): {int((~b.is_hero.astype(bool)).sum())}"
    )
    print(
        f"won by the attacker: {int((b.wina == 1).sum())}; by the defender: {int((b.wina == -1).sum())}; "
        f"draws: {int((b.wina == 0).sum())}; outcome missing: {int(b.wina.isna().sum())}"
    )
    print(f"decisive battles with both strengths known: {len(d)}")
    print(
        f"median strength: attacker {b.attacker_str.median():,.0f}, defender {b.defender_str.median():,.0f}; "
        f"median force ratio {np.exp(b.lr.median()):.2f}"
    )
    print("battles by era: " + ", ".join(f"{k} {v}" for k, v in b.era.value_counts().sort_index().items()))
    for _, _, name in bc.ERAS:
        top_e = b[b.era == name].war_key.value_counts().head(3)
        print(f"  {name}, largest wars: " + "; ".join(f"{k} {v}" for k, v in top_e.items()))
    top = b.war_key.value_counts().head(4)
    print("largest wars: " + "; ".join(f"{k} {v}" for k, v in top.items()))

    print("\n== How often numbers win")
    share, n = bc.larger_side_share(d)
    lo_hi = bc.cluster_bootstrap(d, lambda x: bc.larger_side_share(x)[0])
    print(
        f"decisive battles with unequal numbers won by the larger side: {share:.3f} of {n}, "
        f"95% interval {ci(lo_hi, '{:.3f}')}"
    )
    print(f"won by the attacker: {d.win.mean():.3f}")
    rows = []
    for (lo, hi), lab in zip(zip(BINS[:-1], BINS[1:], strict=True), BIN_LABELS, strict=True):
        x = d[(d.lr > lo) & (d.lr <= hi)]
        w0, w1 = wilson(x.win.sum(), len(x))
        rows.append((lab, lo, hi, len(x), x.win.mean(), w0, w1, x.lr.median()))
        print(f"  attacker:defender {lab:11s} n={len(x):3d} attacker won {x.win.mean():.3f} [{w0:.3f}, {w1:.3f}]")
    pd.DataFrame(rows, columns=["bin", "lo", "hi", "n", "win", "w_lo", "w_hi", "lr_median"]).to_csv(
        RESULTS / "win_by_ratio.csv", index=False
    )
    fit = bc.logit(d.win, d.lr)
    a, s = fit.params
    print(f"logistic slope on the log force ratio: {s:.3f}, 95% interval {ci(bc.cluster_bootstrap(d, slope))}")
    print(f"odds ratio for doubling the force ratio: {np.exp(s * np.log(2)):.2f}")
    ws, wse, wn = bc.within_war_slope(d)
    print(
        f"comparing battles within the same war (conditional logit, {wn} battles): slope {ws:.3f}, "
        f"standard error {wse:.3f}; odds ratio for doubling {np.exp(ws * np.log(2)):.2f}"
    )
    p = lambda r: 1 / (1 + np.exp(-(a + s * np.log(r))))  # noqa: E731
    print(f"modelled chance the attacker wins at 1:2, 1:1, 2:1, 3:1: {p(0.5):.3f}, {p(1):.3f}, {p(2):.3f}, {p(3):.3f}")
    grid = np.linspace(np.log(0.25), np.log(17), 200)
    pd.DataFrame({"lr": grid, "p": 1 / (1 + np.exp(-(a + s * grid)))}).to_csv(RESULTS / "win_curve.csv", index=False)

    print("\n== Examples")
    print("attackers with under half the defender's numbers who won, and attackers with over three times who lost")
    for lab, x in (
        ("won with under 1:2", d[(d.lr < BINS[1]) & (d.win == 1)].sort_values("lr")),
        ("lost with over 3:1", d[(d.lr > BINS[-2]) & (d.win == 0)].sort_values("lr", ascending=False)),
    ):
        print(f"  {lab}: {len(x)}, attacker surprise in {int(x.att_surprise.sum())}")
        for r in x.itertuples():
            print(
                f"    {r.name.title()} ({r.year}), {r.attacker_actors} {r.attacker_str:,.0f} attacking "
                f"{r.defender_actors} {r.defender_str:,.0f}{', surprise' if r.att_surprise else ''}"
            )
    x = d[(d.lr > BINS[-2]) & (d.win == 0)]
    print(
        f"  defences in those {len(x)} lost attacks: fortified {int((x.post1 == 'FD').sum())}, prepared "
        f"{int((x.post1 == 'PD').sum())}, hasty {int((x.post1 == 'HD').sum())}; fortified in all decisive "
        f"battles {(d.post1 == 'FD').mean():.3f}"
    )
    x = d[d.war_key.str.startswith("Mexican-American")]
    print(
        f"  Mexican-American War: {len(x)} decisive battles, {int((x.attacker_actors == 'USA').sum())} American "
        f"attacks, all won: {bool(x[x.attacker_actors == 'USA'].win.all())}; attacker outnumbered in "
        f"{int((x.lr < 0).sum())}, of which the attacker won {int(x[x.lr < 0].win.sum())}"
    )

    print("\n== Against Lanchester")
    big, small = 1000, 500
    print(
        f"illustration, {big:,} against {small:,} of equal quality: under the square law the larger side wins with "
        f"{np.sqrt(big**2 - small**2):,.0f} left and the smaller side must be {(big / small) ** 2:.0f} times as "
        f"effective per soldier to draw; under the linear law {big - small:,} left and {big / small:.0f} times"
    )
    for name, e in (("square law", 2), ("linear law", 1)):
        for lab, sl_ in (("all battles", s), ("within wars", ws)):
            sd = bc.implied_spread(sl_, e)
            print(
                f"  {name}, {lab}: the log effectiveness ratio would need a standard deviation of {sd:.2f}; "
                f"one standard deviation is a factor of {np.exp(sd):,.0f}"
            )
    c = bc.casualty_frame(b)
    cd = c[c.wina.isin([1, -1])]
    sl = bc.casualty_slope(c)
    print(f"battles with both strengths and casualties: {len(c)}")
    print(
        f"slope of log(attacker losses / defender losses) on the log force ratio: {sl:+.3f}, "
        f"95% interval {ci(bc.cluster_bootstrap(c, bc.casualty_slope), '{:+.3f}')}"
    )
    slw = bc.casualty_slope(cd, control_outcome=True)
    print(
        f"the same, decisive battles, holding the result fixed: {slw:+.3f}, 95% interval "
        f"{ci(bc.cluster_bootstrap(cd, lambda x: bc.casualty_slope(x, True)), '{:+.3f}')}"
    )
    print("benchmarks: square law -1, linear law 0")
    fa = np.polyfit(c.lr, np.log(c.attacker_cas / c.attacker_str), 1)[0]
    fd = np.polyfit(c.lr, np.log(c.defender_cas / c.defender_str), 1)[0]
    print(f"slope of log loss fraction on the log force ratio: attacker {fa:+.3f}, defender {fd:+.3f}")
    print(
        f"median losses as a share of initial strength: attacker {(c.attacker_cas / c.attacker_str).median():.3f}, "
        f"defender {(c.defender_cas / c.defender_str).median():.3f}"
    )
    c[["isqno", "lr", "llr", "wina"]].to_csv(RESULTS / "casualties.csv", index=False)

    print("\n== How battles end")
    fa_, fd_ = cd.attacker_cas / cd.attacker_str, cd.defender_cas / cd.defender_str
    loser = pd.Series(np.where(cd.wina == 1, fd_, fa_), index=cd.index)
    winner = pd.Series(np.where(cd.wina == 1, fa_, fd_), index=cd.index)
    print(f"decisive battles with strengths and losses: {len(cd)}")
    print(
        f"median losses as a share of initial strength: losing side {loser.median():.3f}, "
        f"winning side {winner.median():.3f}"
    )
    for t in (0.1, 0.2, 0.5):
        print(f"  losing side lost under {t:.0%} of its strength: {(loser < t).mean():.3f}")
    print(f"  losing side lost over half: {(loser > 0.5).mean():.3f} ({int((loser > 0.5).sum())} battles)")
    print(f"losing side lost the larger share: {(loser > winner).mean():.3f}")
    for name, g in loser.groupby(cd.era):
        print(f"  {name}: median share lost, losing side {g.median():.3f}, winning side {winner[g.index].median():.3f}")

    print("\n== Errors in the strengths")
    rel, n_err = bc.attenuation(d)
    print(
        f"decisive battles with stated strength ranges: {n_err}; median half-range attacker {d.att_err.median():.3f}, "
        f"defender {d.def_err.median():.3f}"
    )
    print(f"reliability of the log force ratio, reading half-ranges as one standard deviation: {rel:.3f}")
    print(
        f"slopes corrected for that error: all battles {s / rel:.3f}, within wars {ws / rel:.3f}; square-law "
        f"factor {np.exp(bc.implied_spread(s / rel, 2)):,.0f} and {np.exp(bc.implied_spread(ws / rel, 2)):,.0f}"
    )

    print("\n== What else goes with winning (decisive battles)")
    for flag, name in (("att_surprise", "attacker achieved surprise"), ("def_surprise", "defender achieved surprise")):
        x = d[d[flag] == 1]
        print(f"  {name}: n={len(x)}, attacker won {x.win.mean():.3f}")
    x = d[(d.att_surprise == 0) & (d.def_surprise == 0)]
    print(f"  no surprise either way: n={len(x)}, attacker won {x.win.mean():.3f}")
    post = {"FD": "fortified defence", "PD": "prepared defence", "HD": "hasty defence", "DL": "delaying action"}
    for k, v in post.items():
        x = d[d.post1 == k]
        print(
            f"  defender in {v}: n={len(x)}, attacker won {x.win.mean():.3f}, "
            f"median force ratio {np.exp(x.lr.median()):.2f}"
        )
    for k, v in {"F": "flat", "R": "rolling", "G": "rugged"}.items():
        x = d[d.terra1 == k]
        print(f"  {v} terrain: n={len(x)}, attacker won {x.win.mean():.3f}")
    rec = bc.prior_records(b, links)
    d = d.merge(rec, on="isqno", how="left")
    d["record"] = bc.record_edge(d)
    print(
        f"battles in which either commander had fought an earlier battle in CDB90: "
        f"{int(((d.a_prior_n > 0) | (d.d_prior_n > 0)).sum())} of {len(d)}"
    )
    for k, v in {1: "better", 0: "equal (mostly no earlier battle)", -1: "worse"}.items():
        x = d[d.record == k]
        print(f"  attacker's commander had the {v} prior record: n={len(x)}, attacker won {x.win.mean():.3f}")

    common = d.dropna(subset=[*bc.CODED, "post1", "terra1"]).reset_index(drop=True)
    print(f"battles with every variable below known: {len(common)}")
    models = {
        "numbers": ["lr"],
        "numbers + surprise": ["lr", "att_surprise", "def_surprise"],
        "numbers + posture": ["lr", "fortified", "prepared", "delaying"],
        "numbers + terrain": ["lr", "rugged", "flat"],
        "numbers + prior record": ["lr", "record"],
        "numbers + surprise + posture + terrain + record": [
            "lr",
            "att_surprise",
            "def_surprise",
            "fortified",
            "prepared",
            "delaying",
            "rugged",
            "flat",
            "record",
        ],
        "numbers + coded leadership": ["lr", "leada"],
        "numbers + all ten coded advantages": ["lr", *bc.CODED],
    }
    common = common.assign(
        fortified=(common.post1 == "FD").astype(int),
        prepared=(common.post1 == "PD").astype(int),
        delaying=common.post1.isin(["DL", "WD"]).astype(int),
        rugged=(common.terra1 == "G").astype(int),
        flat=(common.terra1 == "F").astype(int),
    )
    out = []
    for name, cols in models.items():
        f = bc.logit(common.win, common[cols])
        out.append((name, bc.auc(common.win, f.fittedvalues), bc.mz_r2(f)))
        print(f"  {name:48s} AUC {out[-1][1]:.3f}  share of variation explained {out[-1][2]:.3f}")
    pd.DataFrame(out, columns=["model", "auc", "mz_r2"]).to_csv(RESULTS / "predictors.csv", index=False)
    f = bc.logit(common.win, common[models["numbers + surprise + posture + terrain + record"]])
    names = ["const", *models["numbers + surprise + posture + terrain + record"]]
    print(
        "  odds ratios in the model without coded advantages: "
        + ", ".join(f"{n} {np.exp(v):.2f}" for n, v in zip(names, f.params, strict=True) if n != "const")
    )
    print(f"  odds ratio for doubling the force ratio in that model: {np.exp(f.params[1] * np.log(2)):.2f}")

    print("\n== Hindsight in the codes")
    sides = bc.side_rows(b, links)
    hs = []
    for col, name in [("lr", "force ratio (counted, not coded)"), ("surpa", "surprise")] + [
        (c_, bc.CODED_NAMES[c_]) for c_ in bc.CODED
    ]:
        r, k, n = bc.within_commander(sides, col)
        hs.append((name, r, k, n))
        print(f"  within-commander correlation with winning, {name:33s} {r:+.3f} ({k} commanders, {n} battles)")
    pd.DataFrame(hs, columns=["variable", "r", "commanders", "battles"]).to_csv(RESULTS / "hindsight.csv", index=False)
    labels = links.drop_duplicates("wikidata_id").set_index("wikidata_id").wikidata_label
    sides["name"] = sides.commander.map(labels)
    g = sides.dropna(subset=["leada"]).groupby("name")
    tab = pd.DataFrame(
        {
            "battles": g.size(),
            "wins": g.won.sum(),
            "lead_win": g.apply(lambda x: x.loc[x.won == 1, "leada"].mean()),
            "lead_loss": g.apply(lambda x: x.loc[x.won == 0, "leada"].mean()),
        }
    )
    tab = tab[(tab.wins >= 2) & (tab.battles - tab.wins >= 2)].sort_values("battles", ascending=False)
    tab.to_csv(RESULTS / "commanders.csv")
    pr = bc.pair_rows(b, links)
    print(
        f"pairs of opposing commanders who met more than once with different results: "
        f"{pr.groupby(['c1', 'c2']).ngroups}, {len(pr)} battles"
    )
    for col, name in (("lead", "coded leadership"), ("lr1", "force ratio"), ("surp", "surprise")):
        r, pv = bc.within_pair(pr, col, permutations=5000)
        print(f"  within-pair correlation of {name} with the result: {r:+.3f}, permutation p-value {pv:.4f}")
    lab = links.drop_duplicates("wikidata_id").set_index("wikidata_id").wikidata_label
    ex = pr[(pr.c1.map(lab) == "Napoleon") & (pr.c2.map(lab).str.contains("Archduke Charles"))].sort_values("year")
    for r in ex.itertuples():
        print(
            f"  Napoleon v Archduke Charles, {r.name.title()} {r.year}: Napoleon {'won' if r.res == 1 else 'lost'}, "
            f"his coded leadership advantage {r.lead:+.0f}"
        )
    print("commanders with at least two wins and two defeats: coded leadership advantage in wins / in defeats")
    for nm, r in tab.iterrows():
        print(f"  {nm:30s} {int(r.battles):2d} battles, {int(r.wins):2d} won: {r.lead_win:+.2f} / {r.lead_loss:+.2f}")
    print(f"  mean over these {len(tab)} commanders: {tab.lead_win.mean():+.2f} / {tab.lead_loss.mean():+.2f}")

    print("\n== Which battles get fought")
    for lab, mask in (
        ("attacker outnumbered (under 3:4)", d.lr < -0.29),
        ("roughly equal (3:4 to 4:3)", d.lr.abs() <= 0.29),
        ("attacker ahead (over 4:3)", d.lr > 0.29),
    ):
        x = d[mask]
        print(
            f"  {lab:33s} n={len(x):3d} attacker surprise {x.att_surprise.mean():.3f}, attacker won {x.win.mean():.3f}"
        )

    print("\n== By era")
    er = []
    for _, _, name in bc.ERAS:
        x = d[d.era == name]
        sh, nn = bc.larger_side_share(x)
        sl_ = slope(x)
        lo, hi = bc.cluster_bootstrap(x, slope)
        slo, shi = bc.cluster_bootstrap(x, lambda z: bc.larger_side_share(z)[0])
        er.append((name, len(x), x.win.mean(), sh, slo, shi, sl_, lo, hi))
        print(
            f"  {name}: n={len(x)}, attacker won {x.win.mean():.3f}, larger side won {sh:.3f} [{slo:.3f}, {shi:.3f}], "
            f"slope {sl_:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        )
    pd.DataFrame(
        er, columns=["era", "n", "attacker_won", "larger_won", "larger_lo", "larger_hi", "slope", "lo", "hi"]
    ).to_csv(RESULTS / "eras.csv", index=False)


if __name__ == "__main__":
    main()
