"""Part 2 of the Simpson's paradox series: which table is right.

Every number quoted in the article is printed by this script. Run:

    python causal_numbers.py > output.txt

Sections
1. One table, many causal structures: every directed acyclic graph on three variables, which ones the
   table rejects, and what each compatible one says the effect of the treatment is.
2. The adjustment formula, by standardisation and by inverse probability weighting.
3. A mediator: total, controlled direct, natural direct and natural indirect effects.
4. Berkeley as a question about direct effects, and what an unmeasured qualification does to it.
5. Colliders: Berkson's hospital, M-bias and butterfly bias in a linear Gaussian model.
6. Lord's paradox: change scores against covariance adjustment.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import expit, logit

SEED = 1967
HERE = Path(__file__).parent
DATA = HERE / "data"
RDATASETS = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/"
UCB = ("csv/datasets/UCBAdmissions.csv", "92ae5a362fd3a09847ee388823286381b847b48fd1c9a1eca387c987a1b5eba3")

# The table of Part 1 (Charig et al., 1986, open surgery against PCNL), written as counts n[x, z, y]:
# x = 0 treatment A (open surgery), 1 treatment B (PCNL); z = 0 small stones, 1 large or multiple;
# y = 1 success, 0 failure.
COUNTS = np.zeros((2, 2, 2))
COUNTS[0, 0, 1], COUNTS[0, 0, 0] = 81, 87 - 81
COUNTS[0, 1, 1], COUNTS[0, 1, 0] = 192, 263 - 192
COUNTS[1, 0, 1], COUNTS[1, 0, 0] = 234, 270 - 234
COUNTS[1, 1, 1], COUNTS[1, 1, 0] = 55, 80 - 55
VARS = ("X", "Z", "Y")


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Directed acyclic graphs on three variables
# ---------------------------------------------------------------------------

def all_dags():
    """Every DAG on the three labelled variables, as a frozenset of directed edges (parent, child)."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    dags = set()
    for choice in itertools.product((None, 0, 1), repeat=3):
        edges = []
        for (a, b), c in zip(pairs, choice):
            if c == 0:
                edges.append((a, b))
            elif c == 1:
                edges.append((b, a))
        # acyclic iff some topological order exists
        for order in itertools.permutations(range(3)):
            pos = {v: i for i, v in enumerate(order)}
            if all(pos[a] < pos[b] for a, b in edges):
                dags.add(frozenset(edges))
                break
    return sorted(dags, key=lambda e: (len(e), sorted(e)))


def parents(dag, v):
    return sorted(a for a, b in dag if b == v)


def d_separated(dag, a, b, given):
    """d-separation of single nodes a and b given a set, for a DAG on three nodes: checked through the
    only possible path structures (direct edge, or a path through the third node)."""
    if (a, b) in dag or (b, a) in dag:
        return False
    c = ({0, 1, 2} - {a, b}).pop()
    in_ac = (a, c) in dag or (c, a) in dag
    in_bc = (b, c) in dag or (c, b) in dag
    if not (in_ac and in_bc):
        return True
    collider = (a, c) in dag and (b, c) in dag
    if collider:
        return c not in given
    return c in given


def implied_independences(dag):
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        c = ({0, 1, 2} - {a, b}).pop()
        if d_separated(dag, a, b, set()):
            out.append((a, b, None))
        if d_separated(dag, a, b, {c}):
            out.append((a, b, c))
    return out


def g_test(counts: np.ndarray, a: int, b: int, c: int | None):
    """G-test (likelihood-ratio) of a independent of b (given c) in a 2x2x2 count array."""
    n = counts
    if c is None:
        m = n.sum(axis=tuple(i for i in range(3) if i not in (a, b)))
        if a > b:
            m = m.T
        e = np.outer(m.sum(axis=1), m.sum(axis=0)) / m.sum()
        g = 2 * np.sum(m * np.log(m / e))
        df = 1
    else:
        g, df = 0.0, 0
        for k in range(2):
            idx = [slice(None)] * 3
            idx[c] = k
            m = n[tuple(idx)]
            rest = [i for i in range(3) if i != c]
            if rest.index(a) > rest.index(b):
                m = m.T
            e = np.outer(m.sum(axis=1), m.sum(axis=0)) / m.sum()
            g += 2 * np.sum(m * np.log(m / e))
            df += 1
    return g, df, stats.chi2.sf(g, df)


def has_directed_path(dag, a, b):
    frontier, seen = [a], set()
    while frontier:
        v = frontier.pop()
        for x, y in dag:
            if x == v and y not in seen:
                if y == b:
                    return True
                seen.add(y)
                frontier.append(y)
    return False


def effect_under(dag, q):
    """P(Y = 1 | do(X = x)) for x = 0, 1 under a DAG, computed from a joint distribution q that
    factorises according to it. With no directed path from X to Y the effect is zero by
    construction; otherwise the parents of X block every back-door path, so adjusting for them is
    valid. Returns (values, has_path)."""
    if not has_directed_path(dag, 0, 2):
        base = q[:, :, 1].sum()
        return np.array([base, base]), False
    pa = parents(dag, 0)
    out = []
    for x in range(2):
        if 1 in pa:
            qz = q.sum(axis=(0, 2))
            val = sum(q[x, z, 1] / q[x, z].sum() * qz[z] for z in range(2))
        else:
            val = q[x, :, 1].sum() / q[x].sum()
        out.append(val)
    return np.array(out), True


def dag_name(dag):
    if not dag:
        return "no edges"
    return ", ".join(f"{VARS[a]}->{VARS[b]}" for a, b in sorted(dag))


def factorised_joint(dag, p):
    """The joint distribution rebuilt from the DAG's factorisation using the table's own conditionals."""
    order = next(o for o in itertools.permutations(range(3))
                 if all(o.index(a) < o.index(b) for a, b in dag))
    q = np.ones((2, 2, 2))
    for v in order:
        pa = parents(dag, v)
        keep = sorted(set(pa) | {v})
        marg = p.sum(axis=tuple(i for i in range(3) if i not in keep), keepdims=True)
        if pa:
            denom = p.sum(axis=tuple(i for i in range(3) if i not in pa), keepdims=True)
            q = q * (marg / denom)
        else:
            q = q * marg
    return q


# ---------------------------------------------------------------------------
# 3. Mediation in a binary table
# ---------------------------------------------------------------------------

def mediation(p_y_xz: np.ndarray, p_z_x: np.ndarray, x1: int = 0, x0: int = 1):
    """Total, controlled direct (per z), natural direct and natural indirect effects of switching
    x0 -> x1, from P(y = 1 | x, z) and P(z | x), assuming no unmeasured confounding of x, z, y."""
    te = (p_y_xz[x1] * p_z_x[x1]).sum() - (p_y_xz[x0] * p_z_x[x0]).sum()
    cde = p_y_xz[x1] - p_y_xz[x0]
    nde = ((p_y_xz[x1] - p_y_xz[x0]) * p_z_x[x0]).sum()
    nie = (p_y_xz[x1] * (p_z_x[x1] - p_z_x[x0])).sum()
    return te, cde, nde, nie


def load_berkeley():
    path = DATA / "UCBAdmissions.csv"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RDATASETS + UCB[0], timeout=300) as r:
            path.write_bytes(r.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != UCB[1]:
        raise RuntimeError("UCBAdmissions.csv does not match the pinned digest")
    rows = list(csv.DictReader(open(path)))
    depts = sorted({r["Dept"] for r in rows})
    adm = np.zeros((2, len(depts)))
    app = np.zeros((2, len(depts)))
    for r in rows:
        s = 0 if r["Gender"] == "Male" else 1
        k = depts.index(r["Dept"])
        app[s, k] += int(r["Freq"])
        if r["Admit"] == "Admitted":
            adm[s, k] += int(r["Freq"])
    return depts, adm, app


# ---------------------------------------------------------------------------
# 4. A stylised model of applicants with an unmeasured qualification
# ---------------------------------------------------------------------------

def stylised_admissions(gamma_men: float, gamma_women: float, delta: float = 1.0,
                        hard_rate: float = 0.25, easy_rate: float = 0.65, women_hard: float = 0.6,
                        men_hard: float = 0.25):
    """Two departments, hard and easy. Each applicant has a standardised qualification U ~ N(0, 1).
    The probability of applying to the hard department is logistic in U with slope gamma (by sex) and
    an intercept chosen so that a share women_hard / men_hard of each sex applies there. Admission
    depends only on the department and on U (log-odds slope delta), never on sex: there is no direct
    effect of sex. Returns the within-department admission rates by sex, computed by numerical
    integration over U."""
    u = np.linspace(-8, 8, 16001)
    w = stats.norm.pdf(u)
    w = w / w.sum()

    def intercept(share, gamma):
        lo, hi = -30.0, 30.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (expit(mid + gamma * u) * w).sum() < share:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def admit_intercept(rate):
        lo, hi = -30.0, 30.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (expit(mid + delta * u) * w).sum() < rate:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    a_hard = admit_intercept(hard_rate)
    a_easy = admit_intercept(easy_rate)
    rates = np.zeros((2, 2))  # [sex, dept]: dept 0 hard, 1 easy
    shares = np.zeros((2, 2))
    for s, (share, gamma) in enumerate(((men_hard, gamma_men), (women_hard, gamma_women))):
        c = intercept(share, gamma)
        p_hard = expit(c + gamma * u)
        for d, (pd, a) in enumerate(((p_hard, a_hard), (1 - p_hard, a_easy))):
            mass = pd * w
            shares[s, d] = mass.sum()
            rates[s, d] = (mass * expit(a + delta * u)).sum() / mass.sum()
    return rates, shares


# ---------------------------------------------------------------------------
# 5. Linear Gaussian structural models
# ---------------------------------------------------------------------------

def sem_covariance(B: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
    """Covariance of V = B V + e, with B[i, j] the coefficient of V_j in the equation for V_i."""
    A = np.linalg.inv(np.eye(len(B)) - B)
    return A @ np.diag(noise_var) @ A.T


def ols_coef(S: np.ndarray, y: int, xs: list[int]) -> np.ndarray:
    """Population regression coefficients of V_y on V_xs from a covariance matrix."""
    return np.linalg.solve(S[np.ix_(xs, xs)], S[np.ix_(xs, [y])]).ravel()


def unit_variance_sem(edges: dict[tuple[int, int], float], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build B and noise variances so that every variable has variance one (edges: (parent, child) ->
    coefficient; variables must be numbered in a causal order)."""
    B = np.zeros((n, n))
    for (pa, ch), c in edges.items():
        B[ch, pa] = c
    noise = np.ones(n)
    for v in range(n):
        S = sem_covariance(B, noise)
        explained = S[v, v] - noise[v]
        noise[v] = 1 - explained
        if noise[v] <= 0:
            raise ValueError("coefficients too large for unit variance")
    return B, noise


def main() -> None:
    p = COUNTS / COUNTS.sum()

    header("1. ONE TABLE, MANY CAUSAL STRUCTURES (counts of Part 1: open surgery vs PCNL by stone size)")
    print("  X = treatment (A = open surgery, B = PCNL), Z = stone group, Y = success.")
    pz = p.sum(axis=(0, 2))
    print(f"  P(Z = large or multiple) = {pz[1]:.4f}; P(X = A) = {p[0].sum():.4f}")
    print("  Tests of the conditional independences a structure may imply (G-test):")
    tests = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        c = ({0, 1, 2} - {a, b}).pop()
        for given in (None, c):
            g, df, pv = g_test(COUNTS, a, b, given)
            tests[(a, b, given)] = pv
            lab = f"{VARS[a]} _||_ {VARS[b]}" + (f" | {VARS[given]}" if given is not None else "")
            print(f"    {lab:<14} G = {g:8.3f} on {df} df, p = {pv:.3g}")
    dags = all_dags()
    print(f"  number of DAGs on three labelled variables: {len(dags)}")
    compatible = []
    for dag in dags:
        ind = implied_independences(dag)
        worst = min((tests[i] for i in ind), default=1.0)
        if worst >= 0.05:
            compatible.append((dag, ind, worst))
    print(f"  DAGs whose implied independences are all not rejected at the 5% level: {len(compatible)}")
    for dag, ind, worst in compatible:
        q = factorised_joint(dag, p)
        eff, path = effect_under(dag, q)
        fit = np.max(np.abs(q - p))
        if not path:
            desc = "no directed path from X to Y: effect 0"
        else:
            desc = f"P(Y | do(A)) = {eff[0]:.4f}, P(Y | do(B)) = {eff[1]:.4f}, effect A - B = {eff[0] - eff[1]:+.4f}"
        ind_s = "; ".join(f"{VARS[a]}_||_{VARS[b]}" + (f"|{VARS[c]}" if c is not None else "") for a, b, c in ind) or "none"
        print(f"    {dag_name(dag):<22} implies: {ind_s:<12} max |fitted - observed| {fit:.2e}; {desc}")
    complete = [d for d, _, _ in compatible if len(d) == 3]
    answers = {}
    for d in complete:
        e, path = effect_under(d, factorised_joint(d, p))
        key = "0 (no directed path from X to Y)" if not path else f"{e[0] - e[1]:+.4f}"
        answers.setdefault(key, []).append(dag_name(d))
    print(f"  the {len(complete)} complete DAGs fit the table exactly and give these answers for A - B:")
    for k, v in answers.items():
        print(f"    {k}: {len(v)} DAG(s): {' | '.join(v)}")

    header("2. THE ADJUSTMENT FORMULA WHEN Z IS A CONFOUNDER")
    p_y_xz = p[:, :, 1] / p.sum(axis=2)
    adj = (p_y_xz * pz[None, :]).sum(axis=1)
    naive = p[:, :, 1].sum(axis=1) / p.sum(axis=(1, 2))
    print(f"  P(Y | X) without adjustment: A {naive[0]:.4f}, B {naive[1]:.4f}, difference {naive[0] - naive[1]:+.4f}")
    print(f"  sum_z P(Y | X, z) P(z): A {adj[0]:.4f}, B {adj[1]:.4f}, difference {adj[0] - adj[1]:+.4f}")
    px_z = p.sum(axis=2) / pz[None, :]
    ipw = np.array([(p[x, :, 1] / px_z[x]).sum() for x in range(2)])
    print(f"  inverse probability weighting, sum_z P(x, z, y = 1) / P(x | z): A {ipw[0]:.4f}, B {ipw[1]:.4f}")
    print(f"  P(X = A | Z): small {px_z[0, 0]:.4f}, large or multiple {px_z[0, 1]:.4f}")
    # How large a study would be needed to reject "no effect given stone group" (X independent of Y | Z)?
    g_obs, df_obs, _ = g_test(COUNTS, 0, 2, 1)
    crit = stats.chi2.ppf(0.95, df_obs)
    for k in np.arange(1.0, 20.01, 0.01):
        if stats.ncx2.sf(crit, df_obs, g_obs * k) >= 0.8:
            break
    print(f"  sample size multiple for 80% power to reject X _||_ Y | Z at 5%, if the observed table were the truth"
          f" (noncentrality grows with n): {k:.2f}, about {k * COUNTS.sum():.0f} patients")
    # E-value for the adjusted risk ratio (VanderWeele and Ding, 2017), with a parametric bootstrap interval.
    rr = adj[0] / adj[1]
    rng = np.random.default_rng(SEED + 2)
    n_xz = COUNTS.sum(axis=2)
    sims = rng.binomial(n_xz[None].astype(int), p_y_xz[None], size=(20_000, 2, 2)) / n_xz[None]
    rr_b = (sims * pz[None, None, :]).sum(axis=2)
    rr_b = rr_b[:, 0] / rr_b[:, 1]
    lo, hi = np.percentile(rr_b, [2.5, 97.5])

    def evalue(r):
        r = r if r >= 1 else 1 / r
        return r + math.sqrt(r * (r - 1))

    print(f"  adjusted risk ratio A / B {rr:.4f} (95% bootstrap interval {lo:.4f} to {hi:.4f});"
          f" E-value {evalue(rr):.3f}; E-value for the lower limit {evalue(lo) if lo > 1 else 1.0:.3f}")

    header("3. THE SAME TABLE WHEN Z IS A MEDIATOR")
    p_z_x = p.sum(axis=2) / p.sum(axis=(1, 2))[:, None]
    te, cde, nde, nie = mediation(p_y_xz, p_z_x, x1=0, x0=1)
    print(f"  P(Z = large or multiple | X): A {p_z_x[0, 1]:.4f}, B {p_z_x[1, 1]:.4f}")
    print(f"  total effect of A against B: {te:+.4f}")
    print(f"  controlled direct effect with Z fixed: small {cde[0]:+.4f}, large or multiple {cde[1]:+.4f}")
    print(f"  natural direct effect (Z at its distribution under B): {nde:+.4f}; natural indirect effect: {nie:+.4f};"
          f" sum {nde + nie:+.4f}")

    header("4. BERKELEY AS A QUESTION ABOUT DIRECT EFFECTS")
    depts, adm, app = load_berkeley()
    r = adm / app
    mix = app / app.sum(axis=1, keepdims=True)
    te, cde, nde, nie = mediation(r, mix, x1=0, x0=1)
    print("  sex -> department -> admission, sex -> admission; effects of being a man rather than a woman:")
    print(f"    total effect {te:+.4f}")
    print("    controlled direct effects by department: " + ", ".join(f"{d} {c:+.4f}" for d, c in zip(depts, cde)))
    print(f"    natural direct effect, departments as women chose them: {nde:+.4f}; natural indirect effect {nie:+.4f}")
    _, _, nde_m, nie_m = mediation(r, mix, x1=1, x0=0)
    print(f"    natural direct effect of being a woman, departments as men chose them: {nde_m:+.4f}")
    print()
    print("  A stylised model with no direct effect of sex at all: two departments (hard: 25% admitted on average,")
    print("  easy: 65%), 60% of women and 25% of men applying to the hard one, admission depending on department")
    print("  and on an unmeasured qualification U (log-odds slope 1), and application to the hard department")
    print("  depending on U with a slope that may differ by sex:")
    for gm, gw in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.5, -0.5), (-0.5, 1.5)):
        rates, shares = stylised_admissions(gm, gw)
        overall = (rates * shares).sum(axis=1)
        print(f"    slope men {gm:+.1f}, women {gw:+.1f}: hard dept men {rates[0, 0]:.3f} women {rates[1, 0]:.3f}"
              f" (men - women {rates[0, 0] - rates[1, 0]:+.3f}); easy dept men {rates[0, 1]:.3f} women {rates[1, 1]:.3f}"
              f" ({rates[0, 1] - rates[1, 1]:+.3f}); overall men {overall[0]:.3f} women {overall[1]:.3f}")
    grid = np.linspace(-1.5, 1.5, 7)
    print("  apparent within-department advantage of men in the hard department, (slope men) - (slope women) on a grid:")
    for g in grid:
        rates, _ = stylised_admissions(max(g, 0.0), max(-g, 0.0))
        print(f"    difference in slopes {g:+.2f}: hard {rates[0, 0] - rates[1, 0]:+.4f}, easy {rates[0, 1] - rates[1, 1]:+.4f}")

    header("5. COLLIDERS")
    print("  Berkson: two diseases, independent in the population, P(A) = 0.10, P(B) = 0.20; admitted to hospital")
    print("  with probability 0.05 with neither, 0.40 with A only, 0.40 with B only, 0.70 with both:")
    pa_, pb_ = 0.10, 0.20
    adm_p = {(0, 0): 0.05, (1, 0): 0.40, (0, 1): 0.40, (1, 1): 0.70}
    joint = {(a, b): (pa_ if a else 1 - pa_) * (pb_ if b else 1 - pb_) * adm_p[(a, b)] for a in (0, 1) for b in (0, 1)}
    or_pop = 1.0
    or_hosp = joint[(1, 1)] * joint[(0, 0)] / (joint[(1, 0)] * joint[(0, 1)])
    pb_given_a = joint[(1, 1)] / (joint[(1, 1)] + joint[(1, 0)])
    pb_given_nota = joint[(0, 1)] / (joint[(0, 1)] + joint[(0, 0)])
    print(f"    odds ratio of A and B in the population {or_pop:.3f}; among the admitted {or_hosp:.3f}")
    print(f"    among the admitted, P(B | A) = {pb_given_a:.3f}, P(B | not A) = {pb_given_nota:.3f}")
    print()
    print("  Linear Gaussian models, all variables with unit variance; true effect of X on Y = 0.30.")
    print("  M-structure: U1 -> X, U1 -> Z, U2 -> Z, U2 -> Y, X -> Y (U1, U2 unobserved):")
    beta = 0.30
    for a in (0.3, 0.5, 0.7):
        # order: U1 0, U2 1, Z 2, X 3, Y 4
        B, nv = unit_variance_sem({(0, 3): a, (0, 2): a, (1, 2): a, (1, 4): a, (3, 4): beta}, 5)
        S = sem_covariance(B, nv)
        b_un = ols_coef(S, 4, [3])[0]
        b_adj = ols_coef(S, 4, [3, 2])[0]
        print(f"    all four U-paths {a:.1f}: unadjusted {b_un:.4f} (bias {b_un - beta:+.4f}); adjusted for Z {b_adj:.4f}"
              f" (bias {b_adj - beta:+.4f})")
    print("  Butterfly: as above, plus Z -> X and Z -> Y with coefficient c (Z is both confounder and collider):")
    a = 0.5
    for c in (0.0, 0.1, 0.2, 0.3, 0.4):
        B, nv = unit_variance_sem({(0, 3): a, (0, 2): a, (1, 2): a, (1, 4): a, (2, 3): c, (2, 4): c, (3, 4): beta}, 5)
        S = sem_covariance(B, nv)
        b_un = ols_coef(S, 4, [3])[0]
        b_adj = ols_coef(S, 4, [3, 2])[0]
        print(f"    U-paths {a:.1f}, Z-paths {c:.1f}: unadjusted bias {b_un - beta:+.4f}; adjusted bias {b_adj - beta:+.4f}")
    print("  Adjusting for a mediator: X -> M -> Y (0.5 and 0.5), X -> Y (0.30):")
    B, nv = unit_variance_sem({(0, 1): 0.5, (1, 2): 0.5, (0, 2): beta}, 3)
    S = sem_covariance(B, nv)
    print(f"    total effect {ols_coef(S, 2, [0])[0]:.4f}; coefficient of X adjusted for M {ols_coef(S, 2, [0, 1])[0]:.4f}")

    header("6. LORD'S PARADOX")
    print("  Two groups whose mean weight differs by 14 kg (boys 70, girls 56), within-group SD 8 in both")
    print("  September and June, within-group correlation 0.7 between the two weighings, and no change in either")
    print("  group's distribution between September and June.")
    mu = np.array([70.0, 56.0])
    sd, rho = 8.0, 0.7
    slope_pop = rho
    print(f"    population: mean gain 0 in both groups; covariance-adjusted group difference (1 - rho) * 14 ="
          f" {(1 - slope_pop) * (mu[0] - mu[1]):.3f} kg")
    rng = np.random.default_rng(SEED)
    n = 1000
    w1 = np.concatenate([rng.normal(mu[g], sd, n) for g in (0, 1)])
    grp = np.repeat([1.0, 0.0], n)
    e = rng.normal(0, sd * math.sqrt(1 - rho ** 2), 2 * n)
    mu_i = np.where(grp == 1, mu[0], mu[1])
    w2 = mu_i + rho * (w1 - mu_i) + e
    gain = w2 - w1
    d_gain = gain[grp == 1].mean() - gain[grp == 0].mean()
    se_gain = math.sqrt(gain[grp == 1].var(ddof=1) / n + gain[grp == 0].var(ddof=1) / n)
    X = np.column_stack([np.ones(2 * n), grp, w1])
    coef, res, *_ = np.linalg.lstsq(X, w2, rcond=None)
    resid = w2 - X @ coef
    s2 = resid @ resid / (2 * n - 3)
    cov = s2 * np.linalg.inv(X.T @ X)
    print(f"    simulated, {n} per group: difference in mean gain (boys - girls) {d_gain:+.3f} kg (SE {se_gain:.3f});"
          f" covariance-adjusted difference {coef[1]:+.3f} kg (SE {math.sqrt(cov[1, 1]):.3f}),"
          f" slope on September weight {coef[2]:.3f}")
    print(f"    the same comparison for boys and girls of equal September weight ({mu.mean():.0f} kg): expected June weight"
          f" boys {mu[0] + rho * (mu.mean() - mu[0]):.2f}, girls {mu[1] + rho * (mu.mean() - mu[1]):.2f}")


if __name__ == "__main__":
    main()
