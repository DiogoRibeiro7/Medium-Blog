"""Part 4 of the Monty Hall series: why people do not switch, and what it takes to learn to.

Every number quoted in the article is printed by this script. Run:

    python learning_numbers.py > output.txt

Sections
1. What experience can teach: the outcome of every game reveals whether switching would have won.
   A learner who ignores the reasoning and learns only from outcomes, with exact binomial calculations.
2. Belief and decision: a prior concentrated on "fifty-fifty" slows conviction but not the decision.
3. Matching and maximising: the cost of switching two times in three.
4. Reinforcement learners: learning from one's own outcome only (the hot stove) versus learning from
   the outcome of the door not chosen, with a bias towards staying.
5. Regret: how much worse a loss after switching must feel before staying becomes the better choice.
6. Published learning curves and the learners fitted to them.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np
from scipy import optimize, special, stats

SEED = 2010
P_SWITCH = 2 / 3  # probability that switching wins in the standard game


def header(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------------------
# 1. What experience can teach
# ---------------------------------------------------------------------------

def p_counts_favour_switch(n: int, p: float = P_SWITCH, tie: float = 0.5) -> float:
    """P(switching has won more of n games than staying); ties count with weight `tie`."""
    k = np.arange(n + 1)
    pmf = stats.binom.pmf(k, n, p)
    more = pmf[2 * k > n].sum()
    eq = pmf[2 * k == n].sum()
    return float(more + tie * eq)


def counting_learner_wins(n_games: int, p: float = P_SWITCH) -> float:
    """Expected wins in n_games for a learner who switches in game t+1 iff switching has won more
    of the first t games than staying, and tosses a coin on ties (including game 1)."""
    total = 0.0
    for t in range(n_games):
        s = p_counts_favour_switch(t, p) if t > 0 else 0.5
        total += s * p + (1 - s) * (1 - p)
    return total


# ---------------------------------------------------------------------------
# 2. Belief and decision
# ---------------------------------------------------------------------------

def posterior_not_half(n: int, k: np.ndarray, pi0: float) -> np.ndarray:
    """Posterior probability that theta != 1/2 under the prior pi0 * delta(1/2) + (1 - pi0) * U(0, 1),
    after k switch-wins in n games."""
    log_slab = special.betaln(k + 1, n - k + 1)
    log_spike = n * math.log(0.5)
    log_odds = math.log((1 - pi0) / pi0) + log_slab - log_spike
    return special.expit(log_odds)


def conviction_after(n: int, pi0: float, threshold: float = 0.95, p: float = P_SWITCH) -> float:
    """Share of learners (over the binomial distribution of outcomes) whose posterior probability that
    the game is not fifty-fifty exceeds the threshold after n games."""
    k = np.arange(n + 1)
    pmf = stats.binom.pmf(k, n, p)
    return float(pmf[posterior_not_half(n, k, pi0) > threshold].sum())


def games_until(share: float, pi0: float, threshold: float = 0.95, n_max: int = 2000) -> int:
    for n in range(1, n_max + 1):
        if conviction_after(n, pi0, threshold) >= share:
            return n
    return -1


# ---------------------------------------------------------------------------
# 4. Reinforcement learners
# ---------------------------------------------------------------------------

def simulate_learners(n_agents: int, n_trials: int, alpha: float, beta: float, bias: float,
                      kappa: float = 0.0, feedback: str = "full", q0: float = 0.5,
                      rng: np.random.Generator | None = None, p: float = P_SWITCH):
    """Delta-rule learners choosing between switching and staying by a logistic (softmax) rule.

    P(switch) = expit(beta * (Q_switch - Q_stay) - bias). After each game the car is revealed, so
    w = 1 if switching would have won. A win pays 1, a loss after staying pays 0 and a loss after
    switching pays -kappa (regret for an act of commission).
    feedback "own": only the value of the option chosen is updated (learning from one's own outcome).
    feedback "full": both values are updated, each with the payoff it would have earned.
    Returns the expected switch probability per trial, averaged over agents, and the realised choices.
    The final values of both options are kept in simulate_learners.last_values.
    """
    rng = rng if rng is not None else np.random.default_rng(SEED)
    q_sw = np.full(n_agents, q0)
    q_st = np.full(n_agents, q0)
    probs = np.empty(n_trials)
    choices = np.empty((n_trials, n_agents), dtype=bool)
    for t in range(n_trials):
        ps = special.expit(beta * (q_sw - q_st) - bias)
        probs[t] = ps.mean()
        c = rng.random(n_agents) < ps
        w = rng.random(n_agents) < p
        pay_sw = np.where(w, 1.0, -kappa)
        pay_st = np.where(w, 0.0, 1.0)
        if feedback == "full":
            q_sw += alpha * (pay_sw - q_sw)
            q_st += alpha * (pay_st - q_st)
        elif feedback == "own":
            q_sw = np.where(c, q_sw + alpha * (pay_sw - q_sw), q_sw)
            q_st = np.where(~c, q_st + alpha * (pay_st - q_st), q_st)
        else:
            raise ValueError(feedback)
        choices[t] = c
    simulate_learners.last_values = (q_sw.copy(), q_st.copy())
    return probs, choices


def asymptote(beta: float, bias: float, kappa: float = 0.0, p: float = P_SWITCH) -> float:
    """Switch probability once both values have settled at their expected payoffs."""
    v_sw = p - kappa * (1 - p)
    v_st = 1 - p
    return float(special.expit(beta * (v_sw - v_st) - bias))


def first_trial_reaching(curve: np.ndarray, level: float) -> int | None:
    idx = np.flatnonzero(curve >= level)
    return int(idx[0]) + 1 if idx.size else None


# ---------------------------------------------------------------------------
# 6. Published learning curves and a matching learner fitted to them
# ---------------------------------------------------------------------------

# Block means of the switch rate from repeated-play experiments with the standard game
# (switching wins 2/3). Numbers as printed in the sources; SDs are across participants.
PETROCELLI_HARRIS = {
    # Petrocelli & Harris (2011), PSPB 37(10), 1297-1311, Study 1: N = 47, 60 trials.
    "Study 1": {"n": 47, "first": (0.21, 0.22), "last": (0.38, 0.26),
                "overall": (19.96 / 60, 11.25 / 60)},
    # Study 2: N = 65, 60 trials; switches in the first and last 10 trials 2.57 (1.78) and 4.88 (2.90) of 10;
    # 34.91 (12.66) stick and 25.08 switch decisions overall (the two add up to 60, so their SDs are equal).
    "Study 2": {"n": 65, "first": (0.257, 0.178), "last": (0.488, 0.290),
                "overall": (25.08 / 60, 12.66 / 60)},
}
# Herbranson & Schroeder (2010), J. Comp. Psych. 124(1), 1-13, Experiment 3: 6 people per condition,
# 200 trials in four blocks of 50, no cover story. Switch rates with the 95% intervals they report.
HS_HUMANS = {
    "switching pays 2/3": {"p_win": 2 / 3, "first": (0.5667, 0.3451, 0.7883), "last": (0.6567, 0.4504, 0.8629)},
    "staying pays 2/3": {"p_win": 1 / 3, "first": (0.3000, 0.1321, 0.4679), "last": (0.2767, 0.1112, 0.4422)},
}
HS_PIGEONS = {"day 1": (0.3633, 0.2512, 0.4755), "day 30": (0.9633, 0.9397, 0.9870)}


def matching_curve(p0: float, m: float, p_win: float, games: np.ndarray) -> np.ndarray:
    """Expected switch probability in each game for a learner who switches with probability equal to
    its estimate of P(switching wins): a Beta prior with mean p0 worth m games, updated by the outcome of
    every game (the car is always revealed). Game g has g - 1 games of experience behind it."""
    t = games - 1.0
    return (p0 * m + p_win * t) / (m + t)


def matching_block(p0: float, m: float, p_win: float, g0: int, g1: int) -> float:
    return float(matching_curve(p0, m, p_win, np.arange(g0, g1 + 1)).mean())


def fit_matching_two_blocks(b1: float, bl: float, block1: tuple, blockl: tuple, p_win: float = P_SWITCH):
    """Solve for (p0, m) so that the matching learner reproduces two block means exactly."""
    g1 = np.arange(block1[0], block1[1] + 1) - 1.0
    gl = np.arange(blockl[0], blockl[1] + 1) - 1.0

    def p0_of(m):
        a1 = np.mean(m / (m + g1))
        return (b1 - p_win * (1 - a1)) / a1

    def gap(m):
        al = np.mean(m / (m + gl))
        return p0_of(m) * al + p_win * (1 - al) - bl

    ms = np.logspace(-2, 6, 400)
    vals = np.array([gap(m) if 0 <= p0_of(m) <= 1 else np.nan for m in ms])
    ok = np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (np.sign(vals[:-1]) != np.sign(vals[1:])))
    if ok.size == 0:
        return None
    i = ok[0]
    m = optimize.brentq(gap, ms[i], ms[i + 1])
    return p0_of(m), m


def games_to_reach(level: float, p0: float, m: float, p_win: float = P_SWITCH) -> float:
    """Games of experience after which the matching learner's switch probability reaches `level`."""
    if not p0 < level < p_win:
        return float("nan")
    return m * (level - p0) / (p_win - level)


def memory_distortion_rate(w: float) -> float:
    """Long-run switch rate of a matching learner who remembers each loss after switching as w losses:
    s solves s = (2/3) / (2/3 + (1/3) * (s * w + 1 - s)), i.e. (w - 1) s^2 + 3 s - 2 = 0."""
    if abs(w - 1) < 1e-12:
        return 2 / 3
    return (-3 + math.sqrt(9 + 8 * (w - 1))) / (2 * (w - 1))


def main() -> None:
    np.set_printoptions(linewidth=120)

    header("1. WHAT EXPERIENCE CAN TEACH")
    print("  When the car is revealed at the end of a game, the player learns whether switching would have")
    print("  won, whatever was chosen. Each game is a Bernoulli trial with P(switching wins) = 2/3.")
    print("  A learner who ignores the reasoning and switches whenever switching has won more of the past")
    print("  games than staying, and tosses a coin when they are tied (including the first game):")
    for n in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200):
        print(f"    after {n:>3} games: P(switch next game) = {p_counts_favour_switch(n):.3f}")
    for n in (10, 50, 200):
        w = counting_learner_wins(n)
        print(f"    expected wins in {n} games: counting learner {w:.1f}, always switch {n * P_SWITCH:.1f},"
              f" always stay {n / 3:.1f}")
    p10 = stats.binom.cdf(5, 10, P_SWITCH)
    print(f"  P(switching wins at most 5 of 10 games) = {p10:.3f};"
          f" at most 10 of 20: {stats.binom.cdf(10, 20, P_SWITCH):.3f};"
          f" at most 25 of 50: {stats.binom.cdf(25, 50, P_SWITCH):.4f}")
    kl_half = P_SWITCH * math.log(P_SWITCH / 0.5) + (1 - P_SWITCH) * math.log((1 - P_SWITCH) / 0.5)
    kl_third = P_SWITCH * math.log(P_SWITCH / (1 / 3)) + (1 - P_SWITCH) * math.log((1 - P_SWITCH) / (2 / 3))
    print(f"  expected log-likelihood ratio per game: 2/3 against 1/2 {kl_half:.4f} nats;"
          f" 2/3 against 1/3 {kl_third:.4f} nats")
    for bf in (10, 20, 100):
        print(f"    games for an expected Bayes factor of {bf} against fifty-fifty: {math.log(bf) / kl_half:.0f};"
              f" against 'staying is better': {math.log(bf) / kl_third:.0f}")

    header("2. BELIEF AND DECISION")
    print("  Prior: probability pi0 that the game is exactly fifty-fifty, otherwise P(switching wins) uniform.")
    print("  Predictive P(switching wins next game) = pi * 1/2 + (1 - pi) * (k + 1)/(n + 2), which exceeds 1/2")
    print("  exactly when k > n/2: the decision does not depend on pi0, the conviction does.")
    for pi0 in (0.5, 0.9, 0.99):
        row = []
        for n in (10, 20, 50, 100):
            row.append(f"n={n}: {conviction_after(n, pi0):.2f}")
        print(f"    pi0 = {pi0:<4}: share with P(not fifty-fifty) > 0.95 after  " + "; ".join(row))
    for pi0 in (0.5, 0.9, 0.99):
        print(f"    pi0 = {pi0:<4}: games until half of learners are convinced {games_until(0.5, pi0)};"
              f" until 90 per cent are {games_until(0.9, pi0)}")
    # The decision is invariant to pi0; check numerically on a grid.
    ok = True
    for pi0 in (0.5, 0.9, 0.99):
        for n in range(1, 60):
            k = np.arange(n + 1)
            post = posterior_not_half(n, k, pi0)
            pred = (1 - post) * 0.5 + post * (k + 1) / (n + 2)
            ok &= bool(np.all((pred > 0.5) == (2 * k > n)))
    print(f"  decision 'switch iff k > n/2' identical for all pi0 checked (n = 1..59): {ok}")

    header("3. MATCHING AND MAXIMISING")
    for s in (0.0, 0.1, F(1, 2), F(2, 3), 0.9, 1.0):
        s_f = float(s)
        print(f"    switch with probability {str(s):>4}: P(win) = {1/3 + s_f/3:.4f}")
    print(f"  over 200 games, matching (2/3) wins {200 * (1/3 + 2/9):.1f} cars on average against"
          f" {200 * 2/3:.1f} for always switching and {200/3:.1f} for always staying")
    print("  A matching learner who switches with probability (k + 1)/(n + 2), k = past switch wins")
    print("  (expected value (1 + 2n/3)/(n + 2) after n games, exact):")
    match = matching_curve(0.5, 2.0, P_SWITCH, np.arange(1, 201))
    print("    mean switch probability, games 1, 10, 50, 100, 200: "
          + ", ".join(f"{match[i]:.3f}" for i in (0, 9, 49, 99, 199)))

    header("4. REINFORCEMENT LEARNERS")
    print("  Delta-rule learners, P(switch) = expit(beta * (Q_switch - Q_stay) - bias), Q starting at 0.5.")
    print("  A bias b gives a first-game switch rate of expit(-b): b = ln 9 gives 10 per cent.")
    b = math.log(9)
    for beta in (3, 6, 10):
        print(f"    beta = {beta:>2}, b = ln 9: asymptotic switch rate {asymptote(beta, b):.3f}")
    print(f"    switch rate 2/3 at the asymptote with b = 0 requires beta = 3 ln 2 = {3 * math.log(2):.3f}")
    print(f"    with b = ln 9, beta for an asymptote of 2/3: {3 * (math.log(2) + b):.3f}; of 0.9: {3 * (math.log(9) + b):.3f}")
    print()
    print("  Learning from the outcome of the door not chosen (full) against from one's own outcome only (own):")
    for alpha, beta in ((0.03, 10.0), (0.1, 10.0), (0.1, 15.0), (0.3, 10.0)):
        for fb in ("full", "own"):
            curve, _ = simulate_learners(20_000, 200, alpha, beta, b, feedback=fb,
                                         rng=np.random.default_rng(SEED + 1))
            reach = first_trial_reaching(curve, 0.5)
            print(f"    alpha {alpha}, beta {beta:>4}, {fb:<4}: switch rate in games 1, 10, 25, 50, 100, 200 ="
                  f" {curve[0]:.3f}, {curve[9]:.3f}, {curve[24]:.3f}, {curve[49]:.3f}, {curve[99]:.3f},"
                  f" {curve[199]:.3f}; first game above one half: {reach}")
    print()
    print("  Long run (4,000 learners, 3,000 games, alpha 0.1, beta 10, b = ln 9): mean values at the end")
    for fb in ("full", "own"):
        curve, _ = simulate_learners(4_000, 3_000, 0.1, 10.0, b, feedback=fb, rng=np.random.default_rng(SEED + 3))
        q_sw, q_st = simulate_learners.last_values
        print(f"    {fb:<4}: switch rate over games 2001-3000 {curve[2000:].mean():.3f};"
              f" mean value of switching {q_sw.mean():.3f}, of staying {q_st.mean():.3f}")
    print(f"    switch rate if the values sat at 2/3 and 1/3: {asymptote(10.0, b):.3f}")
    print("  Effect of games in which switching would have lost, starting from settled values (beta 10, b = ln 9):")
    for alpha, runs in ((0.3, 1), (0.1, 1), (0.1, 3)):
        q_sw, q_st = 2 / 3, 1 / 3
        for _ in range(runs):
            q_sw += alpha * (0.0 - q_sw)
            q_st += alpha * (1.0 - q_st)
        print(f"    alpha {alpha}, {runs} such game(s): P(switch) from {asymptote(10.0, b):.3f} to"
              f" {special.expit(10.0 * (q_sw - q_st) - b):.3f}")
    print()
    print("  Learners left switching rarely late in play (the hot-stove pattern of Denrell and March, 2001):")
    alpha, beta = 0.3, 10.0
    _, ch = simulate_learners(20_000, 200, alpha, beta, b, feedback="own", rng=np.random.default_rng(SEED + 2))
    print(f"    (alpha {alpha}, beta {beta}) share who never switch in 200 games: {np.mean(~ch.any(axis=0)):.3f};"
          f" share still switching less than a third of the time in games 151-200:"
          f" {np.mean(ch[150:].mean(axis=0) < 1/3):.3f}")
    _, chf = simulate_learners(20_000, 200, alpha, beta, b, feedback="full", rng=np.random.default_rng(SEED + 2))
    print(f"    same parameters, full feedback: never switch {np.mean(~chf.any(axis=0)):.3f};"
          f" switching less than a third of the time in games 151-200: {np.mean(chf[150:].mean(axis=0) < 1/3):.3f}")

    header("5. REGRET FOR AN ACT OF COMMISSION")
    print("  Payoff of a loss after switching = -kappa (a loss after staying = 0).")
    print("  Expected payoff: switching 2/3 - kappa/3, staying 1/3; staying is better when kappa > 1.")
    for kappa in (0.0, 0.5, 1.0, 1.5, 2.0):
        print(f"    kappa {kappa}: switch {2/3 - kappa/3:+.3f}, stay {1/3:+.3f};"
              f" asymptotic switch rate at beta 10, b = ln 9: {asymptote(10, b, kappa):.3f};"
              f" at beta 10, b = 0: {asymptote(10, 0.0, kappa):.3f}")


    header("6. PUBLISHED LEARNING CURVES AND THE LEARNERS FITTED TO THEM")
    print("  The counting learner of section 1, given the same number of games as the participants:")
    print(f"    games 51-60 (Petrocelli & Harris): expected switch rate"
          f" {np.mean([p_counts_favour_switch(n) for n in range(50, 60)]):.3f}")
    print(f"    games 41-50 (Granberg & Brown, final block): {np.mean([p_counts_favour_switch(n) for n in range(40, 50)]):.3f}")
    print(f"    games 151-200 (Herbranson & Schroeder, final block): "
          f"{np.mean([p_counts_favour_switch(n) for n in range(150, 200)]):.3f}")
    print()
    print("  A matching learner: switches with probability equal to its current estimate of P(switching wins),")
    print("  starting from a prior with mean p0 worth m games. Fitted exactly to the first and last blocks of 10,")
    print("  then used to predict the overall switch rate over all 60 games, which was not used in the fit.")
    print("  Uncertainty: block means drawn from normal distributions with the reported SD / sqrt(N),")
    print("  independently (the within-person correlation of the blocks is not reported); 20,000 draws.")
    rng = np.random.default_rng(SEED + 6)
    for name, d in PETROCELLI_HARRIS.items():
        n = d["n"]
        p0, m = fit_matching_two_blocks(d["first"][0], d["last"][0], (1, 10), (51, 60))
        overall = matching_block(p0, m, P_SWITCH, 1, 60)
        draws_m, draws_p0, draws_ov = [], [], []
        b1s = rng.normal(d["first"][0], d["first"][1] / math.sqrt(n), 20_000)
        bls = rng.normal(d["last"][0], d["last"][1] / math.sqrt(n), 20_000)
        for b1, bl in zip(b1s, bls):
            fit = fit_matching_two_blocks(b1, bl, (1, 10), (51, 60))
            if fit is None:
                continue
            draws_p0.append(fit[0])
            draws_m.append(fit[1])
            draws_ov.append(matching_block(fit[0], fit[1], P_SWITCH, 1, 60))
        dm = np.array(draws_m)
        lo, hi = np.percentile(dm, [2.5, 97.5])
        ov_lo, ov_hi = np.percentile(draws_ov, [2.5, 97.5])
        obs_ov, sd_ov = d["overall"]
        print(f"    Petrocelli & Harris {name} (N = {n}): first block {d['first'][0]:.3f}, last block {d['last'][0]:.3f}")
        print(f"      fitted p0 = {p0:.3f}, prior worth m = {m:.0f} games (95% interval {lo:.0f} to {hi:.0f};"
              f" {len(dm)} of 20000 draws solvable)")
        print(f"      predicted overall switch rate {overall:.3f} (95% interval {ov_lo:.3f} to {ov_hi:.3f});"
              f" observed {obs_ov:.3f} (SE {sd_ov / math.sqrt(n):.3f})")
        for g in (100, 200, 500, 1000):
            print(f"      predicted switch rate in game {g}: {matching_curve(p0, m, P_SWITCH, np.array([g]))[0]:.3f}")
        print(f"      games of experience before the switch rate reaches 0.5: {games_to_reach(0.5, p0, m):.0f};"
              f" 0.6: {games_to_reach(0.6, p0, m):.0f}")
        print(f"      expected cars in 60 games at the observed overall switch rate: {60 * (1/3 + obs_ov/3):.1f}"
              f" (always switching {60 * P_SWITCH:.0f}, always staying {60 / 3:.0f})")
    print()
    print("  Granberg & Brown (1995), approximate: 12% switch in the first game (abstract), about 50% in the final")
    print("  block of 10 of 50 games (as reported by Petrocelli & Harris); N not available, so no interval:")
    for final in (0.50, 0.55):
        fit = fit_matching_two_blocks(0.12, final, (1, 1), (41, 50))
        print(f"    final block {final:.2f}: p0 = {fit[0]:.3f}, m = {fit[1]:.0f} games;"
              f" predicted switch rate in game 200: {matching_curve(fit[0], fit[1], P_SWITCH, np.array([200]))[0]:.3f};"
              f" games of experience before the switch rate reaches 0.5: {games_to_reach(0.5, fit[0], fit[1]):.0f}")
    print()
    print("  Herbranson & Schroeder (2010), humans without a cover story (6 per condition, blocks of 50 games):")

    def hs_loss(x):
        p0, log_m = x
        m = math.exp(log_m)
        err = 0.0
        for d in HS_HUMANS.values():
            err += (matching_block(p0, m, d["p_win"], 1, 50) - d["first"][0]) ** 2
            err += (matching_block(p0, m, d["p_win"], 151, 200) - d["last"][0]) ** 2
        return err

    best = min((optimize.minimize(hs_loss, [p0s, lm], method="Nelder-Mead",
                                  options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 5000})
                for p0s in (0.3, 0.5, 0.7) for lm in (0.0, 2.0, 4.0)), key=lambda r: r.fun)
    p0_hs, m_hs = best.x[0], math.exp(best.x[1])
    print(f"    one matching learner for both conditions (least squares on the four block means):"
          f" p0 = {p0_hs:.3f}, m = {m_hs:.1f} games")
    for name, d in HS_HUMANS.items():
        f1 = matching_block(p0_hs, m_hs, d["p_win"], 1, 50)
        f4 = matching_block(p0_hs, m_hs, d["p_win"], 151, 200)
        mx = np.mean([p_counts_favour_switch(n, d["p_win"]) for n in range(150, 200)])
        print(f"    {name}: observed first block {d['first'][0]:.3f} [{d['first'][1]:.3f}, {d['first'][2]:.3f}],"
              f" matching {f1:.3f}; observed last block {d['last'][0]:.3f} [{d['last'][1]:.3f}, {d['last'][2]:.3f}],"
              f" matching {f4:.3f}, counting learner {mx:.3f}")
    beta_m = 3 * math.log(2)
    print(f"    an unbiased softmax learner with settled values and beta = 3 ln 2 also gives"
          f" {asymptote(beta_m, 0.0):.3f} and {special.expit(beta_m * (1/3 - 2/3)):.3f} in the two conditions;"
          f" when switching wins 90 per cent of the time, matching gives 0.900 and this learner"
          f" {special.expit(beta_m * (0.9 - 0.1)):.3f}")
    print(f"    pigeons (6 birds, about 100 trials a day from day 5): day 1 {HS_PIGEONS['day 1'][0]:.3f},"
          f" day 30 {HS_PIGEONS['day 30'][0]:.3f} [{HS_PIGEONS['day 30'][1]:.3f}, {HS_PIGEONS['day 30'][2]:.3f}];"
          f" matching predicts {2/3:.3f}")
    print()
    print("  Memory that inflates losses after switching: a matching learner who remembers each such loss as w")
    print("  losses settles at the switch rate s solving (w - 1) s^2 + 3 s - 2 = 0:")
    for w in (1.0, 1.5, 2.0, 3.0, 4.0):
        s = memory_distortion_rate(w)
        print(f"    w = {w}: long-run switch rate {s:.3f}; P(win) {1/3 + s/3:.3f}")
    # Petrocelli & Harris (2011), Study 2: actual switch losses by condition (Table 2), 9.26 (n = 34) and
    # 6.87 (n = 31); recalled minus actual switch losses, M = 4.28 over all 65 participants.
    actual = (9.26 * 34 + 6.87 * 31) / 65
    w_ph = (actual + 4.28) / actual
    print(f"  Petrocelli & Harris Study 2: mean actual switch losses {actual:.2f}, recalled {actual + 4.28:.2f},"
          f" ratio w = {w_ph:.2f}; matching plateau at that w {memory_distortion_rate(w_ph):.3f},"
          f" P(win) {1/3 + memory_distortion_rate(w_ph)/3:.3f}")


if __name__ == "__main__":
    main()
