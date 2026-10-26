"""Every number quoted in Part 1 of the Monty Hall series.

Part 1 tells the history of the problem (Bertrand 1889, Gardner 1959,
Selvin 1975, Parade 1990-91, Morgan et al. 1991) and redoes the
calculations behind it:

1. exact enumeration of the standard game, the biased-host variant of
   Morgan et al. (1991) and the ignorant host, plus Bertrand's boxes and
   Gardner's three prisoners, all with rational arithmetic;
2. the n-door game in which the host opens k goat doors, checked by
   brute-force enumeration and by literal simulation;
3. the evidence in the two experiments the story keeps citing: Monty
   Hall's 10 + 10 games at his dining-room table (New York Times, 21 July
   1991) and the 200 + 200 classroom protocol vos Savant asked teachers to
   run (as described by Brandt, C Users Journal, July 1992);
4. how many simulated switching games a sceptic needs before the Bayes
   factor against "switching wins half the time" reaches 100.

Run:  python history_numbers.py > output.txt
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction as F

import numpy as np
from scipy import stats

SEED = 1975
ALPHA = 0.05


# ---------------------------------------------------------------------------
# 1. Exact enumeration
# ---------------------------------------------------------------------------

def standard_game(q: F = F(1, 2)) -> dict[tuple[int, int], F]:
    """Joint law of (car, opened door) when the player has picked door 1.

    The car is uniform on {1, 2, 3}. The host knows where it is, never opens
    door 1 and never opens the car. When the car is behind door 1 he opens
    door 3 with probability q and door 2 with probability 1 - q.
    """
    joint: dict[tuple[int, int], F] = {}
    for car in (1, 2, 3):
        prior = F(1, 3)
        if car == 1:
            joint[(car, 3)] = prior * q
            joint[(car, 2)] = prior * (1 - q)
        else:
            opened = 5 - car  # the other goat door among {2, 3}
            joint[(car, opened)] = prior
    return joint


def conditional_switch_win(joint: dict[tuple[int, int], F], opened: int) -> F:
    """P(switching wins | the host opened `opened`)."""
    other = 5 - opened
    p_opened = sum(p for (c, o), p in joint.items() if o == opened)
    p_car_other = sum(p for (c, o), p in joint.items() if o == opened and c == other)
    return p_car_other / p_opened


def unconditional_switch_win(joint: dict[tuple[int, int], F]) -> F:
    """P(always-switch wins), averaged over the host's choice."""
    return sum(p for (c, o), p in joint.items() if c == 5 - o)


def ignorant_host() -> F:
    """P(switching wins | a goat was revealed) when the host opens 2 or 3 at random."""
    num = F(0)
    den = F(0)
    for car in (1, 2, 3):
        for opened in (2, 3):
            p = F(1, 3) * F(1, 2)
            if opened == car:
                continue  # the car was revealed; this branch is excluded
            den += p
            if car == 5 - opened:
                num += p
    return num / den


def bertrand_boxes() -> F:
    """P(box is gold-gold | a coin drawn at random from a random box is gold)."""
    boxes = {"GG": "GG", "SS": "SS", "GS": "GS"}
    num = F(0)
    den = F(0)
    for name, coins in boxes.items():
        for coin in coins:
            p = F(1, 3) * F(1, 2)
            if coin == "G":
                den += p
                if name == "GG":
                    num += p
    return num / den


def three_prisoners() -> tuple[F, F]:
    """Gardner (1959). A asks the warden to name one of B, C who will die.

    The warden names B. Returns P(A pardoned | 'B'), P(C pardoned | 'B'),
    with the warden choosing at random when A is the one pardoned.
    """
    joint = standard_game()  # same skeleton: A = door 1, 'B named' = door 2 opened
    # relabel: naming B corresponds to opening door 2; C is door 3
    p_b = sum(p for (c, o), p in joint.items() if o == 2)
    p_a = sum(p for (c, o), p in joint.items() if o == 2 and c == 1) / p_b
    p_c = sum(p for (c, o), p in joint.items() if o == 2 and c == 3) / p_b
    return p_a, p_c


# ---------------------------------------------------------------------------
# 2. n doors, host opens k goat doors
# ---------------------------------------------------------------------------

def p_switch_formula(n: int, k: int) -> F:
    """P(win) when the player switches uniformly to one of the n-1-k other closed doors."""
    return F(n - 1, n * (n - k - 1))


def p_switch_bruteforce(n: int, k: int) -> F:
    """Exact enumeration over car position, the host's k-subset and the switch."""
    player = 0
    others = list(range(1, n))
    total = F(0)
    for car in range(n):
        goats = [d for d in others if d != car]
        subsets = list(itertools.combinations(goats, k))
        for opened in subsets:
            remaining = [d for d in others if d not in opened]
            p_branch = F(1, n) * F(1, len(subsets))
            total += p_branch * F(sum(1 for d in remaining if d == car), len(remaining))
    assert player == 0
    return total


def p_switch_simulated(n: int, k: int, trials: int, rng: np.random.Generator) -> float:
    """Literal simulation: place the car, let the host open k goat doors, switch at random."""
    wins = 0
    others = np.arange(1, n)
    for _ in range(trials):
        car = int(rng.integers(n))
        goats = others[others != car]
        opened = rng.choice(goats, size=k, replace=False)
        remaining = np.setdiff1d(others, opened, assume_unique=True)
        wins += int(rng.choice(remaining) == car)
    return wins / trials


def three_door_simulated(trials: int, rng: np.random.Generator) -> tuple[float, float]:
    """Vectorised literal simulation of the standard game, both strategies."""
    car = rng.integers(0, 3, trials)
    choice = rng.integers(0, 3, trials)
    coin = rng.integers(0, 2, trials)
    # host: if car == choice, open one of the two others at random; else the third door
    first_other = (choice + 1 + coin) % 3
    third = 3 - car - choice
    opened = np.where(car == choice, first_other, third)
    switch_to = 3 - choice - opened
    return float(np.mean(choice == car)), float(np.mean(switch_to == car))


# ---------------------------------------------------------------------------
# 3. Evidence in the experiments
# ---------------------------------------------------------------------------

def binom_loglik(successes: int, n: int, p: float) -> float:
    return successes * math.log(p) + (n - successes) * math.log(1 - p)


def kl_bernoulli(p: float, q: float) -> float:
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def outcome_probabilities(n: int) -> tuple[float, float, float]:
    """P(switch > stay), P(tie), P(switch < stay) for n games of each strategy."""
    s = stats.binom.pmf(np.arange(n + 1), n, 2 / 3)
    t = stats.binom.pmf(np.arange(n + 1), n, 1 / 3)
    joint = np.outer(s, t)  # rows: switch wins, cols: stay wins
    greater = float(np.tril(joint, -1).sum())
    tie = float(np.trace(joint))
    less = float(np.triu(joint, 1).sum())
    return greater, tie, less


def power_binomial(n: int, alpha: float = ALPHA) -> float:
    """Power of the exact one-sided test of 'switch wins half the time' on n switch games."""
    x = np.arange(n + 1)
    tail_h0 = stats.binom.sf(x - 1, n, 0.5)  # P(X >= x | 1/2)
    reject = x[tail_h0 <= alpha]
    if reject.size == 0:
        return 0.0
    return float(stats.binom.sf(reject.min() - 1, n, 2 / 3))


def power_fisher(n: int, alpha: float = ALPHA) -> float:
    """Power of the exact one-sided Fisher test (switch arm better than stay arm)."""
    b = np.arange(n + 1)[:, None]  # switch successes
    a = np.arange(n + 1)[None, :]  # stay successes
    tot = a + b
    # P(B >= b | total successes) under the hypergeometric null
    pval = stats.hypergeom.sf(b - 1, 2 * n, tot, n)
    s = stats.binom.pmf(np.arange(n + 1), n, 2 / 3)[:, None]
    t = stats.binom.pmf(np.arange(n + 1), n, 1 / 3)[None, :]
    return float(np.sum((s * t)[pval <= alpha]))


def prob_difference_at_least(n: int, d: int, p_switch: float, p_stay: float) -> float:
    s = stats.binom.pmf(np.arange(n + 1), n, p_switch)
    t = stats.binom.pmf(np.arange(n + 1), n, p_stay)
    joint = np.outer(s, t)
    diff = np.arange(n + 1)[:, None] - np.arange(n + 1)[None, :]
    return float(joint[diff >= d].sum())


def sprt_stopping_times(reps: int, threshold: float, p_true: float,
                        rng: np.random.Generator, max_games: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Switch-only sequential test of p = 2/3 against p = 1/2.

    Each game adds log(4/3) to the log Bayes factor after a win and
    log(2/3) after a loss. Stops when the factor reaches `threshold` or 1/threshold.
    """
    up = math.log(4 / 3)
    down = math.log(2 / 3)
    bound = math.log(threshold)
    wins = rng.random((reps, max_games)) < p_true
    steps = np.where(wins, up, down)
    path = np.cumsum(steps, axis=1)
    hit_up = path >= bound
    hit_down = path <= -bound
    first_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), max_games)
    first_down = np.where(hit_down.any(axis=1), hit_down.argmax(axis=1), max_games)
    stop = np.minimum(first_up, first_down) + 1
    correct = first_up < first_down
    return stop, correct


def bf_paths(reps: int, games: int, rng: np.random.Generator) -> np.ndarray:
    """log10 Bayes factor (2/3 vs 1/2) along simulated switching sequences."""
    wins = rng.random((reps, games)) < 2 / 3
    steps = np.where(wins, math.log10(4 / 3), math.log10(2 / 3))
    return np.cumsum(steps, axis=1)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(x: F) -> str:
    return f"{x} = {float(x):.6f}"


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("=" * 72)
    print("1. EXACT ENUMERATION (player holds door 1)")
    print("=" * 72)
    joint = standard_game()
    for (car, opened), p in sorted(joint.items()):
        print(f"  car behind {car}, host opens {opened}: probability {p}")
    print(f"  P(stay wins)                       {fmt(sum(p for (c, o), p in joint.items() if c == 1))}")
    print(f"  P(always switch wins)              {fmt(unconditional_switch_win(joint))}")
    print(f"  P(car behind 2 | host opened 3)    {fmt(conditional_switch_win(joint, 3))}")
    lr = F(1) / F(1, 2)
    print(f"  likelihood ratio P(open 3|car 2)/P(open 3|car 1) = {lr}")
    print(f"  posterior odds door 2 : door 1 = 1 x {lr} = {lr}:1")
    print()
    print("  Host with a preference (Morgan et al. 1991): opens door 3 w.p. q when free to choose")
    for q in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
        jq = standard_game(q)
        print(f"    q = {str(q):>4}:  P(win switch | opened 3) = {str(conditional_switch_win(jq, 3)):>4}"
              f"   | opened 2) = {str(conditional_switch_win(jq, 2)):>4}"
              f"   unconditional = {unconditional_switch_win(jq)}")
    print("  closed forms: 1/(1+q) after door 3, 1/(2-q) after door 2; never below 1/2")
    print()
    print(f"  Ignorant host (opens 2 or 3 at random, a goat happens to appear): {fmt(ignorant_host())}")
    print(f"  Bertrand's boxes, P(gold-gold | gold seen):                       {fmt(bertrand_boxes())}")
    pa, pc = three_prisoners()
    print(f"  Three prisoners, P(A pardoned | warden names B) = {pa},  P(C pardoned | B named) = {pc}")
    print()

    stay_mc, switch_mc = three_door_simulated(1_000_000, rng)
    print(f"  Literal simulation, 1,000,000 games: stay {stay_mc:.4f}, switch {switch_mc:.4f}")
    print()

    print("=" * 72)
    print("2. n DOORS, HOST OPENS k GOAT DOORS, PLAYER SWITCHES AT RANDOM")
    print("=" * 72)
    print("  formula: P(switch wins) = (n-1) / (n (n-k-1));  P(stay wins) = 1/n")
    print("  brute-force check against the formula:")
    checked = 0
    for n in range(3, 8):
        for k in range(1, n - 1):
            exact = p_switch_bruteforce(n, k)
            assert exact == p_switch_formula(n, k), (n, k)
            checked += 1
    print(f"    {checked} (n, k) pairs with 3 <= n <= 7, 1 <= k <= n-2: all equal")
    for n, k in [(3, 1), (4, 1), (4, 2), (10, 1), (10, 8), (100, 1), (100, 98)]:
        p = p_switch_formula(n, k)
        ratio = p / F(1, n)
        print(f"    n = {n:>3}, k = {k:>2}: switch {float(p):.4f}  stay {1 / n:.4f}  ratio {float(ratio):.4f}")
    n_m = 1_000_000
    p_m = p_switch_formula(n_m, n_m - 2)
    print(f"    vos Savant's million doors (k = n-2): switch {float(p_m):.6f}")
    print(f"    one opened door out of a million:     switch/stay ratio {float(p_switch_formula(n_m, 1) * n_m):.8f}")
    print("  literal simulation, 100,000 games each:")
    for n, k in [(3, 1), (100, 1), (100, 98)]:
        est = p_switch_simulated(n, k, 100_000, rng)
        print(f"    n = {n:>3}, k = {k:>2}: simulated {est:.4f}   exact {float(p_switch_formula(n, k)):.4f}")
    print()

    print("=" * 72)
    print("3a. MONTY HALL'S DINING-ROOM EXPERIMENT (NYT, 21 July 1991)")
    print("=" * 72)
    stay_w, switch_w, n_each = 4, 8, 10
    print(f"  stay: {stay_w}/{n_each} cars    switch: {switch_w}/{n_each} cars")
    ll_h1 = binom_loglik(stay_w, n_each, 1 / 3) + binom_loglik(switch_w, n_each, 2 / 3)
    ll_h0 = binom_loglik(stay_w, n_each, 1 / 2) + binom_loglik(switch_w, n_each, 1 / 2)
    bf = math.exp(ll_h1 - ll_h0)
    bf_switch = math.exp(binom_loglik(switch_w, n_each, 2 / 3) - binom_loglik(switch_w, n_each, 1 / 2))
    bf_stay = math.exp(binom_loglik(stay_w, n_each, 1 / 3) - binom_loglik(stay_w, n_each, 1 / 2))
    exact_bf = F(2 ** 34, 3 ** 20)
    print(f"  Bayes factor, (1/3, 2/3) against (1/2, 1/2): {bf:.3f}  (exactly 2^34/3^20 = {float(exact_bf):.3f})")
    print(f"    from the switch arm alone: {bf_switch:.3f};  from the stay arm alone: {bf_stay:.3f}")
    print(f"    posterior P(vos Savant's model) from 50:50 prior odds: {bf / (1 + bf):.3f}")
    fisher = stats.fisher_exact([[switch_w, n_each - switch_w], [stay_w, n_each - stay_w]],
                                alternative="greater")
    print(f"  Fisher exact test, one-sided (switch better): p = {fisher.pvalue:.4f}")
    binom_sw = stats.binomtest(switch_w, n_each, 0.5, alternative="greater")
    print(f"  Binomial test of 8/10 against 1/2, one-sided: p = {binom_sw.pvalue:.4f}  (exactly 56/1024)")
    for label, w in (("stay", stay_w), ("switch", switch_w)):
        ci = stats.binomtest(w, n_each).proportion_ci(confidence_level=0.95, method="exact")
        print(f"  95% Clopper-Pearson interval, {label:>6}: [{ci.low:.3f}, {ci.high:.3f}]")
    d_obs = switch_w - stay_w
    p_null = prob_difference_at_least(n_each, d_obs, 0.5, 0.5)
    p_alt = prob_difference_at_least(n_each, d_obs, 2 / 3, 1 / 3)
    print(f"  P(switch - stay >= {d_obs}) if both win half the time: {p_null:.4f}")
    print(f"  P(switch - stay >= {d_obs}) under vos Savant's model:  {p_alt:.4f}")
    g, t, l = outcome_probabilities(n_each)
    print(f"  under vos Savant's model, 10 + 10 games give switch ahead {g:.4f}, tie {t:.4f}, stay ahead {l:.4f}")
    med = int(stats.binom.median(n_each, 2 / 3))
    bf_med = math.exp(binom_loglik(med, n_each, 2 / 3) - binom_loglik(med, n_each, 1 / 2))
    print(f"  ten switching games under the 2/3 model: median {med} wins, Bayes factor {bf_med:.2f};"
          f" P(at least {switch_w} wins) = {stats.binom.sf(switch_w - 1, n_each, 2 / 3):.4f}")
    print(f"  expected log10 Bayes factor after ten switching games: "
          f"{n_each * kl_bernoulli(2 / 3, 1 / 2) / math.log(10):.3f}"
          f" (factor {10 ** (n_each * kl_bernoulli(2 / 3, 1 / 2) / math.log(10)):.2f})")
    print(f"  stay rate at which ten stays favour neither model: ln(4/3)/ln(2) = {math.log(4 / 3) / math.log(2):.4f}")
    print()

    print("=" * 72)
    print("3b. THE CLASSROOM PROTOCOL: n GAMES STAYING, n GAMES SWITCHING")
    print("=" * 72)
    kl_pair = kl_bernoulli(1 / 3, 1 / 2) + kl_bernoulli(2 / 3, 1 / 2)
    kl_switch = kl_bernoulli(2 / 3, 1 / 2)
    print(f"  KL(Bern(2/3) || Bern(1/2)) = {kl_switch:.5f} nats per switching game")
    print(f"  KL per stay-plus-switch pair = {kl_pair:.5f} nats = {kl_pair / math.log(10):.5f} decimal digits")
    print("  n per arm | switch ahead | tie   | stay ahead | power Fisher | power binomial | E[log10 BF]")
    for n in (5, 10, 20, 30, 50, 100, 200):
        g, t, l = outcome_probabilities(n)
        pf = power_fisher(n)
        pb = power_binomial(n)
        print(f"  {n:>9} | {g:>12.4f} | {t:.4f} | {l:>10.2e} | {pf:>12.4f} | {pb:>14.4f} | {n * kl_pair / math.log(10):>10.2f}")
    pf_all = {n: power_fisher(n) for n in range(1, 401)}
    pb_all = {n: power_binomial(n) for n in range(1, 401)}
    for target in (0.8, 0.95, 0.99):
        nf = min(n for n, v in pf_all.items() if v >= target)
        nb = min(n for n, v in pb_all.items() if v >= target)
        nf_stay = 1 + max(n for n, v in pf_all.items() if v < target)
        nb_stay = 1 + max(n for n, v in pb_all.items() if v < target)
        print(f"  power >= {target}: Fisher first at n = {nf}, from n = {nf_stay} on;"
              f"  binomial on the switch arm first at n = {nb}, from n = {nb_stay} on")
    g200, t200, l200 = outcome_probabilities(200)
    print(f"  200 + 200: probability the class does not see switching ahead = {t200 + l200:.2e}")
    g20, t20, l20 = outcome_probabilities(20)
    print(f"  1,000 classes of 200 + 200: expected number not seeing switching ahead = {1000 * (t200 + l200):.2e}")
    print(f"  1,000 classes of 20 + 20:   expected number not seeing switching ahead = {1000 * (t20 + l20):.1f}")
    print()

    print("=" * 72)
    print("4. SEQUENTIAL EVIDENCE: SWITCHING GAMES UNTIL BF = 100 EITHER WAY")
    print("=" * 72)
    print("  each win multiplies the Bayes factor by 4/3, each loss by 2/3")
    print(f"  ln(100) / KL = {math.log(100) / kl_switch:.1f} games (drift approximation)")
    stop, correct = sprt_stopping_times(20_000, 100.0, 2 / 3, rng)
    print("  20,000 simulated sceptics, true switch rate 2/3:")
    print(f"    median games {np.median(stop):.0f}, mean {stop.mean():.1f}, "
          f"90th percentile {np.percentile(stop, 90):.0f}, 99th percentile {np.percentile(stop, 99):.0f}")
    print(f"    share stopping at the wrong boundary (BF = 1/100): {1 - correct.mean():.4f}")
    stop0, correct0 = sprt_stopping_times(20_000, 100.0, 1 / 2, rng)
    print("  same test when switching truly wins half the time:")
    print(f"    median games {np.median(stop0):.0f}, share wrongly reaching BF = 100: {correct0.mean():.4f}")


if __name__ == "__main__":
    main()
