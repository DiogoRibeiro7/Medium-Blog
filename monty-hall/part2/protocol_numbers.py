"""Every number quoted in Part 2 of the Monty Hall series.

Part 2 treats the host's rules as a statistical model with four components
and asks, for each one, whether it changes the probability that switching
wins and whether it changes what the contestant should do:

1. the tie-break q: which goat door the host opens when he has a choice;
2. the host's knowledge phi: the probability that he knows where the car is
   (the rest of the time he opens one of the other doors at random, and the
   games in which he reveals the car are void);
3. the placement p of the car behind the doors;
4. the offer policy (a, b): the probability that the host offers the switch
   when the contestant's first pick is right (a) or wrong (b).

The script computes exact posterior probabilities for a general host,
checks the closed forms, computes the expected value of perfect
information (EVPI) of each component under a prior, solves the game in
which the host may decline to offer the switch, and simulates a contestant
who learns the host's offer policy from past episodes.

Run:  python protocol_numbers.py > output.txt
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np
from scipy import integrate, optimize, stats

SEED = 2017
DOORS = (1, 2, 3)  # the contestant holds door 1 unless stated otherwise


# ---------------------------------------------------------------------------
# 1. A general host, by exact enumeration
# ---------------------------------------------------------------------------

def observation_law(p=(F(1, 3), F(1, 3), F(1, 3)), phi=F(1), q=F(1, 2),
                    a=F(1), b=F(1), o=F(1)):
    """Joint law of what the contestant sees, with the contestant holding door 1.

    Returns {observation: (P(obs), P(obs and switching wins), P(obs and staying wins))}.
    An observation is (opened door, offered) after a goat is shown, or
    ('car', door) when an ignorant host reveals the car (a void game).
    The knowing host opens a goat door among {2, 3}; with the car behind 1 he
    opens door 3 with probability q. He offers the switch with probability a
    if the car is behind 1 and b otherwise. The ignorant host opens 2 or 3 at
    random and, when a goat appears, offers the switch with probability o.
    """
    out: dict = {}

    def add(key, prob, switch_wins, stay_wins):
        tot, sw, st = out.get(key, (F(0), F(0), F(0)))
        out[key] = (tot + prob, sw + (prob if switch_wins else 0), st + (prob if stay_wins else 0))

    for car, pc in zip(DOORS, p):
        # knowing host
        if phi > 0:
            opens = [(3, q), (2, 1 - q)] if car == 1 else [(5 - car, F(1))]
            offer = a if car == 1 else b
            for d, pd in opens:
                base = pc * phi * pd
                switch_target = 5 - d
                add((d, True), base * offer, car == switch_target, car == 1)
                add((d, False), base * (1 - offer), False, car == 1)
        # ignorant host
        if phi < 1:
            for d in (2, 3):
                base = pc * (1 - phi) * F(1, 2)
                if d == car:
                    add(("car", d), base, False, False)
                    continue
                switch_target = 5 - d
                add((d, True), base * o, car == switch_target, car == 1)
                add((d, False), base * (1 - o), False, car == 1)
    return out


def p_switch_given(law, opened: int, offered: bool = True) -> F:
    tot, sw, _ = law[(opened, offered)]
    return sw / tot


def p_switch_given_offer(law) -> F:
    tot = sum(v[0] for k, v in law.items() if k[0] != "car" and k[1])
    sw = sum(v[1] for k, v in law.items() if k[0] != "car" and k[1])
    return sw / tot


def best_value(law) -> F:
    """Win probability of the best response to a known host, void games excluded."""
    valid = {k: v for k, v in law.items() if k[0] != "car"}
    tot = sum(v[0] for v in valid.values())
    win = F(0)
    for (d, offered), (pr, sw, st) in valid.items():
        win += max(sw, st) if offered else st
    return win / tot


def policy_value(law, switch_after: dict) -> F:
    """Win probability of a fixed rule: switch_after[d] says whether to switch after door d opens."""
    valid = {k: v for k, v in law.items() if k[0] != "car"}
    tot = sum(v[0] for v in valid.values())
    win = F(0)
    for (d, offered), (pr, sw, st) in valid.items():
        win += (sw if switch_after[d] else st) if offered else st
    return win / tot


def mutual_information_bits(q: float) -> float:
    """I(car location; opened door) for the knowing host, contestant on door 1."""
    def h(x):
        return 0.0 if x in (0.0, 1.0) else -(x * math.log2(x) + (1 - x) * math.log2(1 - x))
    return h((1 + q) / 3) - h(q) / 3


# ---------------------------------------------------------------------------
# 2. Expected value of perfect information, component by component
# ---------------------------------------------------------------------------

POLICIES = [{2: s2, 3: s3} for s2 in (False, True) for s3 in (False, True)]


def evpi_tiebreak(grid: int = 2001) -> tuple[float, float]:
    qs = np.linspace(0, 1, grid)
    best = [float(best_value(observation_law(q=F(qi).limit_denominator(10 ** 6)))) for qi in qs]
    fixed = max(np.mean([float(policy_value(observation_law(q=F(qi).limit_denominator(10 ** 6)), pol))
                         for qi in qs]) for pol in POLICIES)
    return float(np.mean(best)), float(fixed)


def evpi_knowledge(grid: int = 501) -> tuple[float, float]:
    # Void games are excluded, so the prior is over hosts among the games that are played.
    phis = np.linspace(0, 1, grid)
    laws = [observation_law(phi=F(x).limit_denominator(10 ** 6)) for x in phis]
    best = np.mean([float(best_value(l)) for l in laws])
    fixed = max(np.mean([float(policy_value(l, pol)) for l in laws]) for pol in POLICIES)
    return float(best), float(fixed)


def placement_values(p: np.ndarray, q: float = 0.5) -> np.ndarray:
    """Value of every (pick, rule) policy for placements p (rows), knowing host, offer always.

    Returns an array of shape (n, 3 picks, 4 rules). Rule index: (switch after the
    lower-numbered other door, switch after the higher-numbered other door).
    """
    n = p.shape[0]
    vals = np.zeros((n, 3, 4))
    for i in range(3):
        j, k = [d for d in range(3) if d != i]  # other doors, j < k
        pi, pj, pk = p[:, i], p[:, j], p[:, k]
        # host opens k (revealing goat) when car at j, or car at i with prob q; opens j when car at k or at i w.p. 1-q
        open_k_switch, open_k_stay = pj, pi * q
        open_j_switch, open_j_stay = pk, pi * (1 - q)
        for r, (sw_j, sw_k) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            v_k = open_k_switch if sw_k else open_k_stay
            v_j = open_j_switch if sw_j else open_j_stay
            vals[:, i, r] = v_k + v_j
    return vals


def evpi_placement(draws: int, rng: np.random.Generator) -> tuple[float, float, float]:
    p = rng.dirichlet(np.ones(3), size=draws)
    vals = placement_values(p)
    informed = vals.reshape(draws, -1).max(axis=1).mean()
    uninformed = vals.mean(axis=0).max()
    random_pick_switch = (1 - p.mean(axis=1)).mean()  # pick uniformly, always switch: 1 - E[p_pick]
    return float(informed), float(uninformed), float(random_pick_switch)


def evpi_offer(alpha_a=(1, 1), alpha_b=(1, 1)) -> tuple[float, float, float]:
    """EVPI of the offer policy with independent Beta priors on a and b."""
    fa = stats.beta(*alpha_a).pdf
    fb = stats.beta(*alpha_b).pdf
    ea = stats.beta(*alpha_a).mean()
    eb = stats.beta(*alpha_b).mean()
    v_switch_prior = (1 - ea) / 3 + 2 * eb / 3
    uninformed = max(1 / 3, v_switch_prior)
    informed, _ = integrate.dblquad(
        lambda bb, aa: max(1 / 3, (1 - aa) / 3 + 2 * bb / 3) * fa(aa) * fb(bb), 0, 1, 0, 1,
        epsabs=1e-10, epsrel=1e-10)
    return informed, uninformed, v_switch_prior


# ---------------------------------------------------------------------------
# 3. The game in which the host may decline to offer the switch
# ---------------------------------------------------------------------------

def contestant_guarantee(s: float, m: float) -> float:
    """Worst-case win probability when the contestant picks at random and switches
    with probability s when offered, against a host who must offer in at least a
    share m of games but chooses a = P(offer | right) and b = P(offer | wrong)."""
    # win = (1 - a)/3 + a (1 - s)/3 + 2 b s / 3 ; minimise over a, b in [0, 1]
    # subject to a/3 + 2b/3 >= m.
    c = np.array([-s / 3, 2 * s / 3])
    res = optimize.linprog(c, A_ub=np.array([[-1 / 3, -2 / 3]]), b_ub=np.array([-m]),
                           bounds=[(0, 1), (0, 1)], method="highs")
    return 1 / 3 + res.fun


def game_value(m: float, grid: int = 1001) -> tuple[float, float]:
    ss = np.linspace(0, 1, grid)
    g = np.array([contestant_guarantee(s, m) for s in ss])
    i = int(np.argmax(g))
    return float(g[i]), float(ss[i])


# ---------------------------------------------------------------------------
# 4. Learning the offer policy from past episodes
# ---------------------------------------------------------------------------

HOSTS = {
    "always offers (a=1, b=1)": (1.0, 1.0),
    "trap: offers only to right picks (a=1, b=0)": (1.0, 0.0),
    "gift: offers only to wrong picks (a=0, b=1)": (0.0, 1.0),
    "sparing, leans right (a=0.3, b=0.1)": (0.3, 0.1),
    "sparing, leans wrong (a=0.3, b=0.2)": (0.3, 0.2),
}


def win_rate(a: float, b: float, switch: np.ndarray) -> np.ndarray:
    """Expected win probability in a new game for a contestant who switches when offered iff `switch`."""
    return np.where(switch, (1 - a) / 3 + 2 * b / 3, 1 / 3)


def learning_curves(a: float, b: float, ns, reps: int, rng: np.random.Generator):
    """Full learner sees every past episode; broadcast learner sees only episodes with an offer."""
    full, broadcast, right_full, right_broad = [], [], [], []
    oracle_switch = 2 * b > a
    for n in ns:
        nc = rng.binomial(n, 1 / 3, reps)
        nw = n - nc
        oc = rng.binomial(nc, a)
        ow = rng.binomial(nw, b)
        ea = (oc + 1) / (nc + 2)
        eb = (ow + 1) / (nw + 2)
        sw_full = 2 * eb >= ea
        er = (oc + 1) / (oc + ow + 2)  # P(first pick right | offered), Beta(1, 1) prior
        sw_broad = er <= 0.5
        full.append(win_rate(a, b, sw_full).mean())
        broadcast.append(win_rate(a, b, sw_broad).mean())
        right_full.append((sw_full == oracle_switch).mean())
        right_broad.append((sw_broad == oracle_switch).mean())
    return (np.array(full), np.array(broadcast), np.array(right_full), np.array(right_broad))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(x) -> str:
    return f"{x} = {float(x):.4f}" if isinstance(x, F) else f"{x:.4f}"


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("=" * 76)
    print("1. CLOSED FORMS CHECKED AGAINST EXACT ENUMERATION (contestant holds door 1)")
    print("=" * 76)
    checks = 0
    for q in [F(k, 8) for k in range(9)]:
        law = observation_law(q=q)
        assert p_switch_given(law, 3) == 1 / (1 + q)
        assert p_switch_given(law, 2) == 1 / (2 - q)
        assert best_value(law) == F(2, 3)
        checks += 1
    for phi in [F(k, 8) for k in range(9)]:
        for q in [F(k, 4) for k in range(5)]:
            law = observation_law(phi=phi, q=q)
            assert p_switch_given(law, 3) == (1 + phi) / (2 * (1 + phi * q))
            checks += 1
    for p1 in [F(k, 10) for k in range(1, 10)]:
        p = (p1, (1 - p1) / 2, (1 - p1) / 2)
        for q in [F(k, 4) for k in range(5)]:
            law = observation_law(p=p, q=q)
            if law[(3, True)][0] > 0:
                assert p_switch_given(law, 3) == p[1] / (p[1] + q * p[0])
            checks += 1
    for a in [F(k, 4) for k in range(5)]:
        for b in [F(k, 4) for k in range(5)]:
            if a == 0 and b == 0:
                continue
            for q in (F(0), F(1, 2), F(1)):
                law = observation_law(a=a, b=b, q=q)
                assert p_switch_given_offer(law) == 2 * b / (2 * b + a)
                checks += 1
    gen = np.random.default_rng(SEED + 11)
    general = 0
    for _ in range(200):
        w = [F(int(x), 1) for x in gen.integers(1, 10, 3)]
        p = tuple(x / sum(w) for x in w)
        phi, q, a, b, o = (F(int(gen.integers(0, 9)), 8) for _ in range(5))
        law = observation_law(p=p, phi=phi, q=q, a=a, b=b, o=o)
        num = p[1] * (phi * b + (1 - phi) * o / 2)
        den = num + p[0] * (phi * q * a + (1 - phi) * o / 2)
        if den == 0:
            continue
        assert p_switch_given(law, 3) == num / den
        checks += 1
        general += 1
    for phi in [F(k, 8) for k in range(9)]:
        law = observation_law(phi=phi, a=F(1), b=F(0), o=F(1))
        assert p_switch_given(law, 3) == (1 - phi) / (2 - phi)
        checks += 1
    # cash offer made with probability c1 when the contestant's door hides the car, c0 otherwise
    for c1 in [F(k, 4) for k in range(5)]:
        for c0 in [F(k, 4) for k in range(5)]:
            if c1 == 0 and c0 == 0:
                continue
            joint_right = F(1, 3) * c1
            joint_wrong = F(2, 3) * c0
            assert joint_right / (joint_right + joint_wrong) == c1 / (c1 + 2 * c0)
            checks += 1
    print(f"  {checks} parameter settings checked; all closed forms hold exactly")
    print(f"  (including the general formula with all four dials at {general} random rational settings"
          f" with a non-zero denominator, out of 200 drawn)")
    print()

    print("  Tie-break q (knowing host, always offers):")
    for q in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
        law = observation_law(q=q)
        print(f"    q = {str(q):>4}: after door 3 {str(p_switch_given(law, 3)):>4}, after door 2 "
              f"{str(p_switch_given(law, 2)):>4}; best win rate {best_value(law)}; "
              f"I(car; door) = {mutual_information_bits(float(q)):.3f} bits")
    print()
    print("  Host knowledge phi (q = 1/2, always offers; void games excluded):")
    for phi in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
        law = observation_law(phi=phi)
        void = sum(v[0] for k, v in law.items() if k[0] == "car")
        print(f"    phi = {str(phi):>4}: P(switch wins | goat shown) = {fmt(p_switch_given(law, 3))};"
              f" share of games void = {fmt(void)}")
    print("    worst case over q and phi: (1+phi)/(2(1+phi q)) at q = 1 is exactly 1/2")
    print()
    print("  Placement (knowing host, q = 1/2, contestant on door 1):")
    for p in [(F(1, 2), F(3, 10), F(1, 5)), (F(3, 5), F(1, 5), F(1, 5)), (F(1, 5), F(2, 5), F(2, 5))]:
        law = observation_law(p=p)
        v_switch = policy_value(law, {2: True, 3: True})
        print(f"    p = ({', '.join(str(x) for x in p)}): after door 3 {fmt(p_switch_given(law, 3))},"
              f" after door 2 {fmt(p_switch_given(law, 2))}; always switch {fmt(v_switch)};"
              f" best response {fmt(best_value(law))}")
    p = np.array([[0.5, 0.3, 0.2]])
    v = placement_values(p)[0]
    print(f"    p = (0.5, 0.3, 0.2), best pick and rule: {v.max():.4f} (pick the least likely door,"
          f" door 3, and always switch: {v[2, 3]:.4f})")
    print()
    print("  Offer policy (knowing host, uniform placement): P(switch wins | offer) = 2b / (2b + a)")
    for name, (a, b) in HOSTS.items():
        pw = 2 * b / (2 * b + a)
        v_sw = (1 - a) / 3 + 2 * b / 3
        print(f"    {name:<46} P(switch wins | offer) = {pw:.4f};"
              f" always switch {v_sw:.4f}, always stay {1 / 3:.4f}; offer rate {a / 3 + 2 * b / 3:.4f}")
    print("  given the door as well as the offer: P(switch wins | door 3, offer) = b / (b + q a)")
    for q, a, b in ((F(1, 2), F(1), F(1, 4)), (F(0), F(1), F(1, 4)), (F(1), F(1), F(1, 4))):
        law = observation_law(q=q, a=a, b=b)
        assert p_switch_given(law, 3) == b / (b + q * a)
        print(f"    q = {q}, a = {a}, b = {b}: after door 3 {p_switch_given(law, 3)},"
              f" after door 2 {p_switch_given(law, 2)}, offer alone {p_switch_given_offer(law)}")
    print()

    print("  Does any setting make staying strictly better than switching after the door opens?")
    worst_q = min(min(p_switch_given(observation_law(q=q), 3), p_switch_given(observation_law(q=q), 2))
                  for q in [F(k, 100) for k in range(101)])
    worst_phi = min(p_switch_given(observation_law(phi=ph, q=q), 3)
                    for ph in [F(k, 20) for k in range(21)] for q in [F(k, 20) for k in range(21)])
    print(f"    tie-break: lowest P(switch wins | door opened) over q = {worst_q} (never below staying)")
    print(f"    knowledge: lowest over phi and q = {worst_phi} (never below staying)")
    pfix = (F(3, 5), F(1, 5), F(1, 5))
    lawfix = observation_law(p=pfix)
    print(f"    placement, fixed pick: p = (3/5, 1/5, 1/5) gives switch {policy_value(lawfix, {2: True, 3: True})}"
          f" against stay {pfix[0]}; a uniformly random pick restores exactly 2/3 for every p")
    print("    offer policy: the trap host (a=1, b=0) gives switch 0 against stay 1/3, and no rule"
          " guarantees more than 1/3 against every offer policy (section 3)")
    print()

    print("  Knowledge makes the offer dangerous: trap policy (a=1, b=0) when the host knows,")
    print("  offer always (o=1) when he does not; door 3 opened, switch offered, q = 1/2")
    for phi in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
        law = observation_law(phi=phi, a=F(1), b=F(0), o=F(1))
        print(f"    phi = {str(phi):>4}: P(switch wins | door 3, offer) = {fmt(p_switch_given(law, 3))}"
              f"  (closed form (1 - phi)/(2 - phi) = {fmt((1 - phi) / (2 - phi))})")
    print()
    print("  Calibration without consequence: a contestant who assumes q = 1/2 when the true q differs")
    print("  predicts 2/3 after either door; both he and a contestant who knows q always switch.")
    for q in (0.0, 0.25, 0.5, 1.0):
        p3, p2 = (1 + q) / 3, (2 - q) / 3
        r3, r2 = 1 / (1 + q), 1 / (2 - q)
        pi = 2 / 3

        def logloss(pred, r):
            pred = min(max(pred, 1e-12), 1 - 1e-12)
            return -(r * math.log(pred) + (1 - r) * math.log(1 - pred))

        ll_assume = p3 * logloss(pi, r3) + p2 * logloss(pi, r2)
        ll_true = p3 * logloss(r3, r3) + p2 * logloss(r2, r2)
        brier_assume = p3 * (r3 * (1 - pi) ** 2 + (1 - r3) * pi ** 2) + p2 * (r2 * (1 - pi) ** 2 + (1 - r2) * pi ** 2)
        brier_true = p3 * r3 * (1 - r3) + p2 * r2 * (1 - r2)
        ece = p3 * abs(pi - r3) + p2 * abs(pi - r2)
        print(f"    true q = {q:.2f}: log loss {ll_assume:.4f} assumed vs {ll_true:.4f} true;"
              f" Brier {brier_assume:.4f} vs {brier_true:.4f}; mean |error| {ece:.4f}; win rate 2/3 for both")
    print()

    print("=" * 76)
    print("2. EXPECTED VALUE OF PERFECT INFORMATION, ONE COMPONENT AT A TIME")
    print("=" * 76)
    inf_q, unf_q = evpi_tiebreak()
    print(f"  tie-break q ~ U(0,1):        informed {inf_q:.4f}, best fixed rule {unf_q:.4f}, EVPI {inf_q - unf_q:.4f}")
    inf_k, unf_k = evpi_knowledge()
    print(f"  knowledge phi ~ U(0,1):      informed {inf_k:.4f}, best fixed rule {unf_k:.4f}, EVPI {inf_k - unf_k:.4f}")
    inf_p, unf_p, rps = evpi_placement(400_000, rng)
    print(f"  placement p ~ Dirichlet(1,1,1), 400,000 draws: informed {inf_p:.3f}, best fixed rule {unf_p:.3f},"
          f" EVPI {inf_p - unf_p:.3f}; random pick + always switch {rps:.3f}")
    print(f"    exact: informed E[1 - min p] = 8/9 = {8 / 9:.4f}, uninformed 2/3, EVPI 2/9 = {2 / 9:.4f}")
    for label, pa, pb in (("uniform priors a, b ~ U(0,1)", (1, 1), (1, 1)),
                          ("thrifty host a ~ Beta(3,1), b ~ Beta(1,3)", (3, 1), (1, 3)),
                          ("generous host a ~ Beta(1,3), b ~ Beta(3,1)", (1, 3), (3, 1))):
        inf_o, unf_o, vsw = evpi_offer(pa, pb)
        choice = "switch" if vsw > 1 / 3 else "stay"
        print(f"  offer policy, {label:<42}: informed {inf_o:.4f}, best fixed rule {unf_o:.4f} ({choice}),"
              f" EVPI {inf_o - unf_o:.4f}")
    print()

    print("=" * 76)
    print("3. THE GAME WHEN THE HOST MAY DECLINE TO OFFER (contestant picks at random)")
    print("=" * 76)
    print("  value(m) = max(1/3, m - 1/3), m = minimum share of games with an offer")
    for m in (0.0, 1 / 3, 0.5, 2 / 3, 0.75, 0.9, 1.0):
        val, s_star = game_value(m)
        closed = max(1 / 3, m - 1 / 3)
        print(f"    m = {m:.3f}: game value {val:.4f} (closed form {closed:.4f}), "
              f"maximin switching probability {s_star:.3f}")
    print("  forced offer (m = 1): contestant guarantees 1/3 + s/3; host holds him to 2/3 with"
          " uniform placement and a fair tie-break")
    print()

    print("=" * 76)
    print("4. LEARNING THE OFFER POLICY FROM PAST EPISODES (Beta(1,1) priors)")
    print("=" * 76)
    ns = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    reps = 40_000
    print(f"  {reps:,} simulated contestants per host and per number of past episodes")
    for name, (a, b) in HOSTS.items():
        oracle = max(1 / 3, (1 - a) / 3 + 2 * b / 3)
        vos = (1 - a) / 3 + 2 * b / 3
        full, broad, rf, rb = learning_curves(a, b, ns, reps, rng)
        print(f"  {name}: oracle {oracle:.4f}, always switch {vos:.4f}, always stay {1 / 3:.4f}")
        for n, f_, b_, r1, r2 in zip(ns, full, broad, rf, rb):
            print(f"    after {n:>4} episodes: full learner {f_:.4f} (right call {r1:.3f}),"
                  f" offers-only learner {b_:.4f} (right call {r2:.3f})")
    print()
    print("  Past episodes from which the full learner makes the right call at least 95% of the time")
    print("  (every n up to 100, then steps of 5 up to 2,000; 20,000 contestants per point):")
    fine = list(range(0, 101)) + list(range(105, 2001, 5))
    for name, (a, b) in HOSTS.items():
        _, _, rf, _ = learning_curves(a, b, fine, 20_000, np.random.default_rng(SEED + 1))
        below = [n for n, r in zip(fine, rf) if r < 0.95]
        start = 0 if not below else fine[fine.index(below[-1]) + 1]
        worst = min(rf)
        worst_n = fine[int(np.argmin(rf))]
        print(f"    {name:<46} from {start:>4} on; lowest right-call rate {worst:.3f} at {worst_n} episodes")


if __name__ == "__main__":
    main()
