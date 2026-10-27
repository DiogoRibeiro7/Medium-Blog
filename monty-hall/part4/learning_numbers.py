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
    return probs, choices


def asymptote(beta: float, bias: float, kappa: float = 0.0, p: float = P_SWITCH) -> float:
    """Switch probability once both values have settled at their expected payoffs."""
    v_sw = p - kappa * (1 - p)
    v_st = 1 - p
    return float(special.expit(beta * (v_sw - v_st) - bias))


def first_trial_reaching(curve: np.ndarray, level: float) -> int | None:
    idx = np.flatnonzero(curve >= level)
    return int(idx[0]) + 1 if idx.size else None


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
    print("  A matching learner who switches with probability (k + 1)/(n + 2), k = past switch wins:")
    rng = np.random.default_rng(SEED)
    w = rng.random((200, 100_000)) < P_SWITCH
    k = np.vstack([np.zeros((1, w.shape[1])), np.cumsum(w, axis=0)[:-1]])
    n = np.arange(200)[:, None]
    match = ((k + 1) / (n + 2)).mean(axis=1)
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
    for alpha, beta in ((0.1, 10.0), (0.1, 15.0), (0.3, 10.0)):
        for fb in ("full", "own"):
            curve, _ = simulate_learners(20_000, 200, alpha, beta, b, feedback=fb,
                                         rng=np.random.default_rng(SEED + 1))
            reach = first_trial_reaching(curve, 0.5)
            print(f"    alpha {alpha}, beta {beta:>4}, {fb:<4}: switch rate in games 1, 10, 25, 50, 100, 200 ="
                  f" {curve[0]:.3f}, {curve[9]:.3f}, {curve[24]:.3f}, {curve[49]:.3f}, {curve[99]:.3f},"
                  f" {curve[199]:.3f}; first game above one half: {reach}")
    print()
    print("  The hot stove: own-outcome learners whose first switch lost, and how long they then go without switching")
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


if __name__ == "__main__":
    main()
