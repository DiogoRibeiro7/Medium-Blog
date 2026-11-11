"""Part 1 of the taxation-efficiency-drug-prices series: the premises of "taxation is theft".

Every number quoted in the article is printed by this script. Run:

    python premises_numbers.py > output.txt

Of the four premises of the argument, only the last contains an empirical claim: that what a tax
pays for could be had without it, or not at a price worth paying. This script measures that claim
in the one setting where it has been measured carefully, the linear public-goods game of
experimental economics, and then shows how far that setting is from a national public good.

Nothing is downloaded. The behavioural model is calibrated to published summary figures, which are
quoted with their sources in the comments and in the output.

Sections
1. The economy: a linear public good, welfare per person, and the break-even rule for compulsion.
2. Voluntary contributions: calibration to two published figures, the path over 50 rounds, and its
   sensitivity to the parts of the model that the calibration does not pin down.
3. Peer punishment: calibration to two published figures from Fehr and Gaechter (2000), the
   50-round horizon as an out-of-sample check against Gaechter, Renner and Sefton (2008), and
   antisocial punishment (Herrmann, Thoeni and Gaechter 2008).
4. Compulsion: a tax with a marginal cost of public funds and a share of revenue that never reaches
   the good, against the voluntary regimes, at the laboratory's marginal group return.
5. The break-even frontier: the marginal group return above which compulsion yields more welfare.
6. Where the evidence lives: the group sizes and marginal returns that have been studied in the
   laboratory, against a national public good.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publicgoods import (  # noqa: E402
    ENDOWMENT,
    Behaviour,
    Sanctions,
    equivalent_share,
    g_star,
    make_draws,
    replace,
    s_star,
    simulate,
    welfare_compulsory,
    welfare_voluntary,
)

HERE = Path(__file__).resolve().parent
SEED = 20261111
GROUPS = 10_000
N = 4
MPCR = 0.4
G_LAB = N * MPCR
ROUNDS = 50

# Published figures used as calibration targets.
TARGET_ROUND1 = 0.50   # Ledyard (1995): first-round contributions "between 40% and 60%" of the optimum
TARGET_ROUND10 = 0.20  # Andreoni (1995), citing Isaac and Walker (1988): "about 15-25 percent" by round 10
TARGET_FULL10 = 0.825  # Fehr and Gaechter (2000): 82.5 % give the whole endowment in the final period
TARGET_RATIO10 = 1.00  # Fehr and Gaechter (2000): punishment costs about as large as the gains

# Marginal cost of public funds: Browning (1976), 9-16 cents per marginal dollar of labour-tax
# revenue; Ballard, Shoven and Whalley (1985), 17-56 cents, centring around 33 cents.
LAMBDAS = {"Browning low": 1.09, "Browning high": 1.16, "BSW low": 1.17, "BSW centre": 1.33, "BSW high": 1.56}
WASTES = [0.0, 0.1, 0.3]

np.set_printoptions(linewidth=110, precision=3, suppress=True)


def fmt_loss(x: float) -> str:
    """Squared calibration error; values at rounding level are printed as a bound, not as noise."""
    return "below 1e-8" if x < 1e-8 else f"{x:.1e}"


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def sig(v):
    return 1.0 / (1.0 + np.exp(-np.asarray(v, dtype=float)))


def calibrate_voluntary(base: Behaviour, draws) -> tuple[Behaviour, float]:
    def obj(x):
        q = simulate(replace(base, beta_mean=float(sig(x[0])), belief0=float(sig(x[1]))), draws, MPCR, 10)
        return (q.contribution[0] - TARGET_ROUND1) ** 2 + (q.contribution[9] - TARGET_ROUND10) ** 2

    res = optimize.minimize(obj, [1.5, 0.4], method="Nelder-Mead",
                            options=dict(xatol=1e-6, fatol=1e-14, maxiter=800))
    return replace(base, beta_mean=float(sig(res.x[0])), belief0=float(sig(res.x[1]))), float(res.fun)


def calibrate_sanctions(behaviour: Behaviour, base: Sanctions, draws, vcm10_welfare: float,
                        starts=((12.0, 0.08), (4.0, 0.3), (20.0, 0.05))) -> tuple[Sanctions, float]:
    # eta is kept in (0, 1): a point cannot raise a floor by more than the whole endowment.
    def obj(x):
        s = replace(base, gamma=float(np.exp(x[0])), eta=float(sig(x[1])))
        p = simulate(behaviour, draws, MPCR, 10, s)
        return (p.full_share[9] - TARGET_FULL10) ** 2 + (p.welfare.sum() / vcm10_welfare - TARGET_RATIO10) ** 2

    best = None
    for g0, e0 in starts:
        r = optimize.minimize(obj, [np.log(g0), np.log(e0 / (1 - e0))], method="Nelder-Mead",
                              options=dict(xatol=1e-5, fatol=1e-14, maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    return replace(base, gamma=float(np.exp(best.x[0])), eta=float(sig(best.x[1]))), float(best.fun)


def main() -> None:
    print("Part 1: the premises of 'taxation is theft', and the one that data can test.")
    print(f"NumPy {np.__version__}")
    draws = make_draws(SEED, GROUPS, N, ROUNDS)
    figure = {}

    # ------------------------------------------------------------------------------------------
    section("1. The economy")
    print(f"Endowment E = {ENDOWMENT:.0f} tokens per person and round; groups of n = {N}; MPCR = {MPCR}.")
    print(f"Marginal group return G = n x MPCR = {G_LAB:.1f}: a token given creates {G_LAB:.1f} tokens of value,")
    print(f"of which the giver gets back {MPCR:.1f}. Keeping the token is privately better whatever the others do;")
    print("giving everything is collectively best.")
    print(f"Welfare per person and round, in units of E: none given = 1.00; everything given = {G_LAB:.2f}.")
    print("Voluntary provision share s gives 1 + (G - 1) s; a tax raising the whole endowment, with marginal")
    print("cost of public funds lam and a share w of revenue lost before it reaches the good, gives")
    print("1 - lam + G (1 - w). Compulsion is better exactly when s < s* = (G (1 - w) - lam) / (G - 1).")

    # ------------------------------------------------------------------------------------------
    section("2. Voluntary contributions: calibration and path")
    print("Model: person i gives beta_i x (belief about the others' average) + error, clipped to [0, E];")
    print("beliefs move towards what the others gave: b <- theta b + (1 - theta) observed.")
    print("beta_i ~ Beta with mean beta_mean and concentration 4; first-round beliefs ~ N(belief0, 0.2);")
    print("decision error sd 0.1 of E; theta = 0.3. Calibrated: beta_mean and belief0.")
    print(f"Targets: round 1 = {TARGET_ROUND1:.2f} (Ledyard 1995: 40-60 %); round 10 = {TARGET_ROUND10:.2f}"
          " (Andreoni 1995 / Isaac and Walker 1988: 15-25 %).")
    base = Behaviour(theta=0.3, beta_conc=4.0, belief0_sd=0.2, noise_sd=0.1)
    behaviour, loss = calibrate_voluntary(base, draws)
    print(f"Calibrated: beta_mean = {behaviour.beta_mean:.3f}, belief0 = {behaviour.belief0:.3f}"
          f" (squared error {fmt_loss(loss)}).")
    vcm = simulate(behaviour, draws, MPCR, ROUNDS)
    print("Mean contribution share by round (1-10):", np.round(vcm.contribution[:10], 3))
    print("Rounds 20, 30, 40, 50:", np.round(vcm.contribution[[19, 29, 39, 49]], 3))
    s10 = float(vcm.contribution[:10].mean())
    s50 = float(vcm.contribution.mean())
    s_last = float(vcm.contribution[-1])
    print(f"Average share over 10 rounds: {s10:.3f}; over 50 rounds: {s50:.3f}; in round 50: {s_last:.3f}.")
    w10 = float(vcm.welfare[:10].sum())
    w50 = float(vcm.welfare.sum())
    print(f"Welfare per person, sum over rounds (units of E): 10 rounds {w10:.3f}; 50 rounds {w50:.3f}.")
    print(f"Efficiency (welfare / welfare at full provision): 10 rounds {w10 / (10 * G_LAB):.3f};"
          f" 50 rounds {w50 / (50 * G_LAB):.3f}.")
    noise_floor = float(np.mean(np.clip(0.1 * np.random.default_rng(1).standard_normal(1_000_000), 0, None)))
    print(f"Mean share given by someone who plans to give nothing, from decision error alone: {noise_floor:.3f}.")
    figure["vcm"] = {"contribution": vcm.contribution.tolist(), "welfare": vcm.welfare.tolist()}

    print()
    print("Sensitivity: recalibrated under other values of the fixed parts (theta, concentration).")
    sens = []
    for theta in (0.0, 0.3, 0.6):
        for conc in (4.0, 10.0):
            b_alt, l_alt = calibrate_voluntary(replace(base, theta=theta, beta_conc=conc), draws)
            p_alt = simulate(b_alt, draws, MPCR, ROUNDS)
            row = (theta, conc, b_alt.beta_mean, b_alt.belief0, float(p_alt.contribution[:10].mean()),
                   float(p_alt.contribution.mean()), float(p_alt.contribution[-1]))
            sens.append(row)
            print(f"  theta {theta:.1f}, conc {conc:4.1f}: beta_mean {row[2]:.3f}, belief0 {row[3]:.3f};"
                  f" mean share 10 rounds {row[4]:.3f}, 50 rounds {row[5]:.3f}, round 50 {row[6]:.3f}")
    s50_range = (min(r[5] for r in sens), max(r[5] for r in sens))
    last_range = (min(r[6] for r in sens), max(r[6] for r in sens))
    beta_range = (min(r[2] for r in sens), max(r[2] for r in sens))
    print(f"  Range of beta_mean: {beta_range[0]:.3f}-{beta_range[1]:.3f}; of the 50-round mean share:"
          f" {s50_range[0]:.3f}-{s50_range[1]:.3f}; of the round-50 share: {last_range[0]:.3f}-{last_range[1]:.3f}.")

    # ------------------------------------------------------------------------------------------
    section("3. Peer punishment")
    print("After contributions are seen, people with slope beta_i >= 0.5 punish each member in proportion")
    print("to that member's shortfall below the rest of the group. A point costs the punisher 1 token and the")
    print("target 3. Points received raise the target's contribution floor; floors do not relax (decay 1.0).")
    print(f"Targets: in round 10, {TARGET_FULL10:.1%} give the whole endowment (Fehr and Gaechter 2000, final")
    print("period of the partner treatment with punishment); welfare over 10 rounds equal to the voluntary")
    print("regime's (Fehr and Gaechter 2000: punishment cost about as large as the gain from cooperation).")
    sbase = Sanctions(floor_decay=1.0)
    sanctions, sloss = calibrate_sanctions(behaviour, sbase, draws, w10)
    print(f"Calibrated: gamma = {sanctions.gamma:.2f} points per unit of shortfall, eta = {sanctions.eta:.4f}"
          f" (squared error {fmt_loss(sloss)}).")
    pun = simulate(behaviour, draws, MPCR, ROUNDS, sanctions)
    print("Mean contribution share by round (1-10):", np.round(pun.contribution[:10], 3))
    print("Share giving the whole endowment by round (1-10):", np.round(pun.full_share[:10], 3))
    print("Punishment cost per person by round (1-10), share of E:", np.round(pun.sanction_cost[:10], 3))
    print("Rounds 20, 30, 40, 50: contribution", np.round(pun.contribution[[19, 29, 39, 49]], 3),
          "cost", np.round(pun.sanction_cost[[19, 29, 39, 49]], 4))
    p10 = float(pun.welfare[:10].sum())
    p50 = float(pun.welfare.sum())
    print(f"Welfare, sum over rounds: 10 rounds {p10:.3f} (ratio to voluntary {p10 / w10:.3f});"
          f" 50 rounds {p50:.3f} (ratio {p50 / w50:.3f}).")
    cum_gap = np.cumsum(pun.welfare) - np.cumsum(vcm.welfare)
    lowest = int(np.argmin(cum_gap)) + 1
    after = np.where(cum_gap[10:] > 0)[0]
    overtake = int(after[0]) + 11 if after.size else None
    print(f"Cumulative welfare gap (punishment minus voluntary): lowest after round {lowest}"
          f" ({cum_gap[lowest - 1]:.3f} E per person); positive for good from round {overtake}.")
    print("Per-round welfare with punishment, rounds 1-10:", np.round(pun.welfare[:10], 3))
    print(f"Out-of-sample check: Gaechter, Renner and Sefton (2008) found that over 50 rounds punishment")
    print(f"raised both cooperation and earnings. The model calibrated on 10 rounds gives a 50-round welfare")
    print(f"ratio of {p50 / w50:.3f}. (Their 50-round groups also show an end-game drop, which this model lacks.)")
    p_eq50 = float(equivalent_share(pun.welfare.mean(), G_LAB))
    p_eq10 = float(equivalent_share(pun.welfare[:10].mean(), G_LAB))
    print(f"Voluntary share with the same welfare: 10 rounds {p_eq10:.3f}; 50 rounds {p_eq50:.3f}.")
    figure["punishment"] = {"contribution": pun.contribution.tolist(), "welfare": pun.welfare.tolist(),
                            "full_share": pun.full_share.tolist(), "cost": pun.sanction_cost.tolist()}

    print()
    print("With floors that relax by 10 % a round (decay 0.9), the two targets cannot both be met:")
    s09, l09 = calibrate_sanctions(behaviour, Sanctions(floor_decay=0.9), draws, w10)
    q09 = simulate(behaviour, draws, MPCR, 10, s09)
    print(f"  best fit gamma {s09.gamma:.2f}, eta {s09.eta:.3f}: full givers in round 10 {q09.full_share[9]:.3f},"
          f" 10-round welfare ratio {q09.welfare.sum() / w10:.3f} (squared error {fmt_loss(l09)}).")

    print()
    print("Antisocial punishment (Herrmann, Thoeni and Gaechter 2008): a share of punishers also punishes")
    print("those who gave more than they did; antisocial points lower the target's slope and floor.")
    anti_rows = []
    for a in (0.0, 0.1, 0.2, 0.3, 0.5):
        q = simulate(behaviour, draws, MPCR, ROUNDS, replace(sanctions, antisocial=a))
        r10 = float(q.welfare[:10].sum() / w10)
        r50 = float(q.welfare.sum() / w50)
        anti_rows.append((a, r10, r50, float(q.contribution[9]), float(q.contribution[-1])))
        print(f"  antisocial share {a:.1f}: welfare ratio 10 rounds {r10:.3f}, 50 rounds {r50:.3f};"
              f" contribution share round 10 {q.contribution[9]:.3f}, round 50 {q.contribution[-1]:.3f}")
    figure["antisocial"] = anti_rows

    # ------------------------------------------------------------------------------------------
    section("4. Compulsion at the laboratory's marginal group return (G = 1.6)")
    print("A tax raises the whole endowment and pays for the good. Welfare per round = 1 - lam + G (1 - w).")
    print("Marginal cost of public funds lam from Browning (1976) and Ballard, Shoven and Whalley (1985).")
    for w in WASTES:
        parts = []
        for name, lam in LAMBDAS.items():
            parts.append(f"{name} {lam:.2f}: {float(welfare_compulsory(G_LAB, lam, w)):.3f}")
        print(f"  w = {w:.1f}: " + "; ".join(parts))
    print(f"Voluntary, mean welfare per round: 10 rounds {w10 / 10:.3f}; 50 rounds {w50 / 50:.3f};"
          f" round 50 {vcm.welfare[-1]:.3f}.")
    print(f"Peer punishment, mean welfare per round: 10 rounds {p10 / 10:.3f}; 50 rounds {p50 / 50:.3f};"
          f" round 50 {pun.welfare[-1]:.3f}.")
    lam0, wa0 = LAMBDAS["BSW centre"], 0.1
    comp = float(welfare_compulsory(G_LAB, lam0, wa0))
    print(f"Reference case lam = {lam0}, w = {wa0}: compulsion {comp:.3f} per round, which is"
          f" {'above' if comp > w10 / 10 else 'below'} the voluntary 10-round mean ({w10 / 10:.3f}),"
          f" {'above' if comp > w50 / 50 else 'below'} the voluntary 50-round mean ({w50 / 50:.3f})")
    print(f"and {'above' if comp > p50 / 50 else 'below'} peer punishment over 50 rounds ({p50 / 50:.3f}).")
    print(f"Break-even voluntary share at G = 1.6: s* = {float(s_star(G_LAB, lam0, wa0)):.3f}"
          f" (lam {lam0}, w {wa0}); with w = 0: {float(s_star(G_LAB, lam0, 0.0)):.3f};"
          f" with lam 1.09, w 0: {float(s_star(G_LAB, 1.09, 0.0)):.3f}.")
    figure["compulsion_lab"] = {"lam": lam0, "w": wa0, "welfare": comp}

    # ------------------------------------------------------------------------------------------
    section("5. The break-even frontier")
    print("Compulsion yields more welfare than voluntary provision with share s when G > G*(s, lam, w),")
    print("G* = (lam - s) / ((1 - w) - s). Shares s from the simulations above:")
    shares = {
        "voluntary, 10-round mean": s10,
        "voluntary, 50-round mean": s50,
        "voluntary, round 50": s_last,
        "peer punishment, 50-round equivalent": p_eq50,
    }
    for name, s in shares.items():
        print(f"  {name}: s = {s:.3f}")
    for name, s in shares.items():
        for w in WASTES:
            gs = [float(g_star(s, lam, w)) for lam in LAMBDAS.values()]
            txt = ", ".join("never" if not np.isfinite(g) else f"{g:.2f}" for g in gs)
            print(f"  G* for {name} (s {s:.3f}), w {w:.1f}, lam {list(LAMBDAS.values())}: {txt}")
    grid_g = np.geomspace(1.02, 20, 400)
    figure["frontier"] = {
        "G": grid_g.tolist(),
        "curves": {f"{lam:.2f}|{w:.1f}": s_star(grid_g, lam, w).tolist()
                   for lam in (1.09, 1.33, 1.56) for w in (0.1, 0.3)},
        "shares": shares,
        "G_lab": G_LAB,
    }
    print()
    print("Reading the frontier the other way: the voluntary share needed to beat compulsion, s*(G).")
    print("A negative value means that compulsion loses to voluntary provision even if nobody gives anything.")
    for G in (1.2, 1.6, 2.0, 3.0, 5.0, 10.0):
        vals = [float(s_star(G, lam, 0.1)) for lam in (1.09, 1.33, 1.56)]
        print(f"  G {G:4.1f}, w 0.1, lam 1.09 / 1.33 / 1.56: " + " / ".join(f"{v:.3f}" for v in vals))

    # ------------------------------------------------------------------------------------------
    section("6. Where the evidence lives")
    cells = [
        ("Isaac, Walker and Williams (1994)", 4, 0.30), ("Isaac, Walker and Williams (1994)", 4, 0.75),
        ("Isaac, Walker and Williams (1994)", 10, 0.30), ("Isaac, Walker and Williams (1994)", 10, 0.75),
        ("Isaac, Walker and Williams (1994)", 40, 0.30), ("Isaac, Walker and Williams (1994)", 40, 0.75),
        ("Isaac, Walker and Williams (1994)", 40, 0.03), ("Isaac, Walker and Williams (1994)", 100, 0.30),
        ("Isaac, Walker and Williams (1994)", 100, 0.75),
        ("Fehr and Gaechter (2000)", 4, 0.40),
        ("Weimann et al. (2019)", 60, 0.02), ("Weimann et al. (2019)", 60, 0.04),
        ("Weimann et al. (2019)", 100, 0.02), ("Weimann et al. (2019)", 100, 0.04),
    ]
    print("Laboratory designs (group size n, MPCR, marginal group return G = n MPCR):")
    for src, n, m in cells:
        print(f"  {src:36s} n {n:4d}  MPCR {m:4.2f}  G {n * m:5.2f}")
    n_max = max(c[1] for c in cells)
    m_min = min(c[2] for c in cells)
    n_nat = 1e7
    print(f"Largest group: {n_max}; smallest MPCR: {m_min}. A good shared by n = 1e7 people (the order of")
    print("magnitude of Portugal's population) with G between 1.5 and 10 has MPCR between "
          f"{1.5 / n_nat:.1e} and {10 / n_nat:.1e}.")
    print(f"Distance from the largest design: n is {np.log10(n_nat / n_max):.0f} orders of magnitude larger;"
          f" MPCR is {np.log10(m_min / (10 / n_nat)):.1f} to {np.log10(m_min / (1.5 / n_nat)):.1f}"
          " orders of magnitude smaller.")
    figure["support"] = {"cells": cells, "n_nat": n_nat}

    with open(HERE / "figure_data.json", "w") as fh:
        json.dump(figure, fh)


if __name__ == "__main__":
    main()
