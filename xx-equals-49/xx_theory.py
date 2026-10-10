"""Checks behind the theorems and propositions in the x**x = 49 article.

Each block verifies numerically a statement that the article proves or quotes.
Requires: numpy, scipy, mpmath.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
from scipy.special import lambertw

K = 49.0
C = math.log(K)
mp.mp.dps = 60
R = mp.exp(mp.lambertw(mp.log(49)))          # the real root, 60 digits
r = float(R)


def rule(title):
    print("=" * 76)
    print(title)


# --------------------------------------------------------------------------
def t1_newton_bound():
    rule("T1. Theorem 1: Newton on g(x) = x ln x - c, global bound e_{n+1} <= M e_n^2")
    M = 1.0 / (2.0 * r * (1.0 + math.log(r)))
    print(f"  M = 1/(2 r (1 + ln r)) = {M:.6f},  1/M = {1 / M:.3f}")
    mp.mp.dps = 400
    c = mp.log(49)
    Rm = mp.exp(mp.lambertw(c))
    Mm = 1 / (2 * Rm * (1 + mp.log(Rm)))
    for x0 in ("0.4", "1", "4", "49", "1e6"):
        x = mp.mpf(x0)
        errs = []
        for _ in range(12):
            x = x - (x * mp.log(x) - c) / (mp.log(x) + 1)
            errs.append(x - Rm)
        n16 = next(i + 1 for i, e in enumerate(errs) if abs(e) < mp.mpf("1e-16"))
        errs = [e for e in errs if abs(e) > mp.mpf("1e-300")]       # drop iterates at working precision
        ok_sign = all(e >= 0 for e in errs)
        ok_mono = all(errs[i + 1] <= errs[i] for i in range(len(errs) - 1))
        ok_bound = all(errs[i + 1] <= Mm * errs[i] ** 2 for i in range(len(errs) - 1))
        print(f"  x0 = {x0:>4s}: x1 - r = {mp.nstr(errs[0], 6):>12s} | all iterates >= r: {ok_sign} | "
              f"monotone: {ok_mono} | bound holds: {ok_bound} | error < 1e-16 after {n16} steps")
    # a priori count from x0 = 1
    e1 = float(1 + C - r)
    n = 1
    while (M * e1) ** (2 ** (n - 1)) / M > 1e-16:
        n += 1
    print(f"  from x0 = 1: e1 = {e1:.4f}, M e1 = {M * e1:.4f}; the bound guarantees error < 1e-16 after n = {n} steps")
    x = mp.mpf(1)
    seq = []
    for _ in range(6):
        x = x - (x * mp.log(x) - c) / (mp.log(x) + 1)
        seq.append(x - Rm)
    print("  actual errors:", [mp.nstr(e, 3) for e in seq])
    print("  bound  errors:", [mp.nstr((Mm * seq[0]) ** (2 ** i) / Mm, 3) for i in range(6)])
    mp.mp.dps = 60


# --------------------------------------------------------------------------
def t2_raw_step_count():
    rule("T2. Newton on x^x - k far from the root: steps ~ ln(f(x0)/k) = x0 ln x0 - ln k")

    def count(x0):
        x, n = x0, 0
        while True:
            step = (x ** x - K) / (x ** x * (math.log(x) + 1.0))
            x -= step
            n += 1
            if abs(step) <= 1e-15 * abs(x):
                return n

    for x0 in (6.0, 10.0, 20.0, 49.0, 100.0, 140.0):
        pred = x0 * math.log(x0) - C
        print(f"  x0 = {x0:6.1f}: predicted {pred:7.1f}, observed {count(x0):4d}")
    # drop in g = x ln x per step, far out
    x = 49.0
    drops = []
    for _ in range(5):
        g0 = x * math.log(x)
        x -= (x ** x - K) / (x ** x * (math.log(x) + 1.0))
        drops.append(g0 - x * math.log(x))
    print("  decrease of ln f per step starting at 49:", [round(d, 4) for d in drops])


# --------------------------------------------------------------------------
def t3_orders():
    rule("T3. Convergence orders measured in 4000-digit arithmetic")
    mp.mp.dps = 4000
    c = mp.log(49)
    Rm = mp.exp(mp.lambertw(c))
    Wm = mp.lambertw(c)

    def report(name, errs, p):
        errs = [abs(e) for e in errs if e != 0]
        orders = [mp.log(errs[i + 1] / errs[i]) / mp.log(errs[i] / errs[i - 1]) for i in range(1, len(errs) - 1)]
        consts = [errs[i + 1] / errs[i] ** p for i in range(len(errs) - 1)]
        print(f"  {name}")
        print("     measured order :", [mp.nstr(o, 7) for o in orders[-4:]])
        print(f"     e_(n+1)/e_n^{p}  :", [mp.nstr(k_, 8) for k_ in consts[-3:]])

    x = mp.mpf(4)
    e = []
    for _ in range(9):
        x = x - (x * mp.log(x) - c) / (mp.log(x) + 1)
        e.append(x - Rm)
    report("Newton, log form", e, 2)
    print("     theory 1/(2r(1+ln r)) =", mp.nstr(1 / (2 * Rm * (1 + mp.log(Rm))), 8))

    x = mp.mpf(4)
    e = []
    for _ in range(12):
        fx = x ** x
        x = x - (fx - 49) / (fx * (mp.log(x) + 1))
        e.append(x - Rm)
    report("Newton, raw form", e, 2)
    L = 1 + mp.log(Rm)
    print("     theory ((1+ln r)^2 + 1/r)/(2(1+ln r)) =", mp.nstr((L ** 2 + 1 / Rm) / (2 * L), 8))

    w = mp.log(1 + c)
    e = []
    for _ in range(6):
        ew = mp.exp(w)
        g = w * ew - c
        w = w - g / (ew * (w + 1) - (w + 2) * g / (2 * w + 2))
        e.append(w - Wm)
    report("Halley on w e^w = c (error in w)", e, 3)
    th = ((Wm + 2) / (2 * (Wm + 1))) ** 2 - (Wm + 3) / (6 * (Wm + 1))
    print("     theory (f''/2f')^2 - f'''/6f' =", mp.nstr(th, 8))
    mp.mp.dps = 60


# --------------------------------------------------------------------------
def tower_pair(t):
    """The classical parametrisation of a^b = b^a with a < e < b."""
    a = (1.0 + 1.0 / t) ** t
    b = (1.0 + 1.0 / t) ** (t + 1.0)
    return a, b


def attractor_of_B(k, n_burn=20000, x0=2.5):
    c = math.log(k)
    x = x0
    for _ in range(n_burn):
        x = math.exp(c / x)
    return tuple(sorted((x, math.exp(c / x))))


def bifurcation_data(k_values):
    """Attractor of B (one value below e^e, two above) and the fixed point r(k)."""
    lows, highs, fixed = [], [], []
    for k in k_values:
        a, b = attractor_of_B(float(k), n_burn=60000)
        lows.append(a)
        highs.append(b)
        fixed.append(math.exp(lambertw(math.log(k)).real))
    return np.array(lows), np.array(highs), np.array(fixed)


def t4_power_tower():
    rule("T4. Iteration B is Euler's power tower; Theorem 2 on the two-cycle")
    print(f"  e^-e = {math.exp(-math.e):.6f},  1/49 = {1 / 49:.6f}  ->  base 1/49 is below Euler's interval")
    print(f"  e^e  = {math.e ** math.e:.6f}  (k above this <=> base 1/k below e^-e)")
    # tower iteration in y = 1/x
    y = 1.0 / 3.0
    for _ in range(2000):
        y = (1.0 / K) ** y
    y2 = (1.0 / K) ** y
    print(f"  tower y <- (1/49)^y settles on {min(y, y2):.9f}, {max(y, y2):.9f};"
          f" reciprocals {1 / max(y, y2):.6f}, {1 / min(y, y2):.6f}")
    print(f"  tower limit formula W(c)/c = {lambertw(C).real / C:.9f} = 1/r = {1 / r:.9f} (unstable here)")
    # two-cycle multiplier, three ways
    print("  k        a           b           a^b        b^a      ln a ln b   (ln k)^2/ab  numeric d(phi o phi)")
    for k in (15.2, 16.0, 20.0, 49.0, 1000.0, 1e6):
        a, b = attractor_of_B(k)
        c = math.log(k)
        phi2 = lambda x: math.exp(c / math.exp(c / x))
        h = 1e-6 * a
        num = (phi2(a + h) - phi2(a - h)) / (2 * h)
        print(f"  {k:<8g} {a:<11.8f} {b:<11.6f} {a ** b:<10.6f} {b ** a:<8.4f} {math.log(a) * math.log(b):<11.6f}"
              f" {c * c / (a * b):<12.6f} {num:.6f}")
    # parametrisation: recover t for k = 49 and check
    from scipy.optimize import brentq
    kt = lambda t: tower_pair(t)[0] ** tower_pair(t)[1]
    t49 = brentq(lambda t: math.log(kt(t)) - C, 1e-3, 50.0)
    a, b = tower_pair(t49)
    print(f"  k = 49 corresponds to t = {t49:.8f}: a = {a:.8f}, b = {b:.6f}, t(t+1)ln^2(1+1/t) = "
          f"{t49 * (t49 + 1) * math.log1p(1 / t49) ** 2:.6f}")
    print(f"  t = 1 gives (a, b) = {tower_pair(1.0)}, k = {kt(1.0)}, multiplier 2 ln^2 2 = {2 * math.log(2) ** 2:.6f}")
    ts = np.exp(np.linspace(math.log(1e-6), math.log(1e6), 400001))
    mult = ts * (ts + 1) * np.log1p(1 / ts) ** 2
    print(f"  max of t(t+1)ln^2(1+1/t) over t in [1e-6, 1e6]: {mult.max():.12f} (< 1); min: {mult.min():.3e}")
    u = np.exp(np.linspace(math.log(1e-3), math.log(1e8), 400001))
    print("  inequality ln(1+u) < u/sqrt(1+u) holds on a grid of u in [1e-3, 1e8]:",
          bool(np.all(np.log1p(u) < u / np.sqrt(1 + u))))
    for uu in ("1e-6", "1e-3", "1", "1e3"):
        um = mp.mpf(uu)
        d = um / mp.sqrt(1 + um) - mp.log(1 + um)
        print(f"     u = {uu:>5s}: u/sqrt(1+u) - ln(1+u) = {mp.nstr(d, 6)}   (u^3/24 = {mp.nstr(um ** 3 / 24, 6)})")
    ts2 = np.exp(np.linspace(math.log(0.02), math.log(2000.0), 200001))
    lnk = (1 + 1 / ts2) ** (ts2 + 1) * ts2 * np.log1p(1 / ts2)
    print("  ln k(t) is strictly decreasing on the grid:", bool(np.all(np.diff(lnk) < 0)),
          f"; k(t) -> e^e as t -> inf: k(2000) = {math.exp(lnk[-1]):.5f}")
    # multiplier of the fixed point across k
    for k in (5.0, 15.0, math.e ** math.e, 16.0, 49.0):
        rk = math.exp(lambertw(math.log(k)).real)
        print(f"  k = {k:9.5f}: fixed point r = {rk:.6f}, multiplier of B = -ln r = {-math.log(rk):+.6f}")


# --------------------------------------------------------------------------
def t5_fold():
    rule("T5. The fold at k_min = e^(-1/e) and the branch point of W at -1/e")
    kmin = math.exp(-1 / math.e)
    print("  k_min =", kmin)
    mp.mp.dps = 50
    for eps in ("1e-2", "1e-4", "1e-8", "1e-12"):
        k = mp.exp(-1 / mp.e) * (1 + mp.mpf(eps))
        z = mp.log(k)
        xu = mp.exp(mp.lambertw(z, 0))
        xl = mp.exp(mp.lambertw(z, -1))
        p = mp.sqrt(2 * (mp.e * z + 1))
        approx_u = (1 + p + p ** 2 / 6) / mp.e
        approx_l = (1 - p + p ** 2 / 6) / mp.e
        cond = 1 / (xu * (1 + mp.log(xu)))
        print(f"  k = k_min(1+{eps}): roots {mp.nstr(xl, 12)}, {mp.nstr(xu, 12)} | gap {mp.nstr(xu - xl, 4)} |"
              f" 2p/e = {mp.nstr(2 * p / mp.e, 4)} | series error {mp.nstr(abs(approx_u - xu), 2)},"
              f" {mp.nstr(abs(approx_l - xl), 2)} | condition number {mp.nstr(cond, 4)}")
    print("  condition number at k = 49:", 1 / (r * (1 + math.log(r))))
    # what double precision delivers near the fold
    k = kmin * (1 + 1e-12)
    xd = math.exp(lambertw(math.log(k), 0).real)
    xt = mp.exp(mp.lambertw(mp.log(mp.mpf(k)), 0))
    print(f"  double-precision upper root at k_min(1+1e-12): {xd!r}; relative error {float(abs(xd - xt) / xt):.2e}")
    mp.mp.dps = 60
    # Taylor series of W at 0: coefficients (-n)^(n-1)/n!, radius 1/e
    def taylor(x, n_terms):
        return sum(mp.mpf(-n) ** (n - 1) / mp.factorial(n) * mp.mpf(x) ** n for n in range(1, n_terms + 1))
    for x in (0.3, 1 / math.e, C):
        print(f"  Taylor series of W at x = {x:.4f}: 10 terms {mp.nstr(taylor(x, 10), 8)}, 40 terms"
              f" {mp.nstr(taylor(x, 40), 8)}, 160 terms {mp.nstr(taylor(x, 160), 8)} | W = {mp.nstr(mp.lambertw(x), 8)}")
    print("  n^(n-1) for n = 1..7 (labelled rooted trees):", [n ** (n - 1) for n in range(1, 8)])


# --------------------------------------------------------------------------
def z_solution(m, k=49):
    return mp.exp(mp.lambertw(mp.log(k) + 2j * mp.pi * m, 0))


def count_zeros(x0, x1, y0, y1, k=49):
    """Argument principle for F(z) = z^z - k on a rectangle that avoids the cut."""
    c = mp.log(k)

    def integrand(z):
        zl = z * mp.log(z)
        return (mp.log(z) + 1) / (1 - mp.exp(c - zl))        # F'/F, overflow-free

    corners = [mp.mpc(x0, y0), mp.mpc(x1, y0), mp.mpc(x1, y1), mp.mpc(x0, y1), mp.mpc(x0, y0)]
    total = mp.mpc(0)
    for p, q in zip(corners[:-1], corners[1:]):
        n_seg = max(8, int(abs(q - p) * 4))
        pts = [p + (q - p) * mp.mpf(i) / n_seg for i in range(n_seg + 1)]
        total += mp.quad(integrand, pts)
    return total / (2j * mp.pi)


def t6_complex():
    rule("T6. Proposition 3: every solution of z^z = 49 is exp(W_0(ln 49 + 2 pi i m))")
    mp.mp.dps = 30
    c = mp.log(49)
    print("   m     z_m                                    |z_m^z_m - 49|   Im Log z_m")
    for m in range(0, 7):
        z = z_solution(m)
        res = abs(mp.exp(z * mp.log(z)) - 49)
        print(f"  {m:2d}   {mp.nstr(z.real, 12):>15s} {'+' if z.imag >= 0 else '-'} {mp.nstr(abs(z.imag), 12):<15s}i"
              f"   {mp.nstr(res, 3):>10s}   {mp.nstr(mp.arg(z), 8)}")
    zs = [z_solution(m) for m in range(-400, 401)]
    print("  all 801 solutions with |m| <= 400 lie in Re z > 0:", all(z.real > 0 for z in zs))
    print("  max |Im Log z_m| over |m| <= 400:", mp.nstr(max(abs(mp.arg(z)) for z in zs), 8), " (pi/2 =", mp.nstr(mp.pi / 2, 8), ")")
    print("  conjugate symmetry z_{-m} = conj(z_m):", all(abs(z_solution(-m) - mp.conj(z_solution(m))) < mp.mpf("1e-25") for m in range(1, 30)))
    # other branches never land in the strip
    bad = 0
    for j in (-3, -2, -1, 1, 2, 3):
        for m in range(-400, 401):
            u = mp.lambertw(c + 2j * mp.pi * m, j)
            if -mp.pi < u.imag <= mp.pi:
                bad += 1
    print("  values W_j(c + 2 pi i m), j in {-3..3} minus {0}, |m| <= 400, that fall in the strip -pi < Im <= pi:", bad)
    # the two facts the proof rests on, sampled
    rng = np.random.default_rng(3)
    in_d0 = lambda u: abs(u.imag) < math.pi and (u.real > -1 if u.imag == 0 else u.real > -u.imag / math.tan(u.imag))
    zs_rand = (rng.normal(size=20000) + 1j * rng.normal(size=20000)) * np.exp(rng.uniform(-3, 6, size=20000))
    ws = lambertw(zs_rand, 0)
    print("  W_0 of 20000 random complex numbers lies in D_0 = {|eta| < pi, xi > -eta cot eta}:",
          all(in_d0(complex(w)) for w in ws))
    xi = rng.uniform(-40, 40, size=400000)
    eta = rng.uniform(-math.pi, math.pi, size=400000)
    with np.errstate(divide="ignore", invalid="ignore"):
        outside = xi <= -eta / np.tan(eta)
    re_part = np.exp(xi) * (xi * np.cos(eta) - eta * np.sin(eta))
    print(f"  sup of Re(u e^u) over {int(outside.sum())} sampled points of the strip outside D_0: {re_part[outside].max():.6f}"
          f"  (1/e = {1 / math.e:.6f})")
    # the quantity bounded in the proof
    print("  bound used in the proof: sup of |xi| e^-|xi| = 1/e =", mp.nstr(1 / mp.e, 8), "< ln 49 =", mp.nstr(c, 8))
    # argument principle
    mp.mp.dps = 20
    boxes = [(0.3, 12.3, -18.2, 18.2), (0.3, 30.3, -60.2, 60.2), (-30.3, -0.3, 0.3, 60.2), (-30.3, -0.3, -60.2, -0.3),
             (-0.3, 0.3, 0.3, 60.2)]
    for (x0, x1, y0, y1) in boxes:
        n = count_zeros(x0, x1, y0, y1)
        inside = sum(1 for z in zs if x0 < z.real < x1 and y0 < z.imag < y1)
        print(f"  box [{x0}, {x1}] x [{y0}, {y1}]: argument principle {mp.nstr(n.real, 8)} (imag part {mp.nstr(n.imag, 2)}),"
              f" listed solutions inside: {inside}")
    mp.mp.dps = 30
    # asymptotics
    for m in (1, 10, 100, 1000, 10000):
        z = z_solution(m)
        a = 2j * mp.pi * m
        lead = a / mp.log(a)
        two = a / (mp.log(a) - mp.log(mp.log(a)))
        print(f"  m = {m:5d}: |z_m| = {mp.nstr(abs(z), 8):>10s} | a/ln a: ratio {mp.nstr(abs(z) / abs(lead), 6)}"
              f" | a/(ln a - ln ln a): ratio {mp.nstr(abs(z) / abs(two), 6)}, relative error {mp.nstr(abs(z - two) / abs(z), 3)}")
    # the negative even integers
    for n in (-2, -4):
        print(f"  ({n})^({n}) = {complex(n) ** n}")
    print("  e^(1/e) =", math.exp(1 / math.e))
    mp.mp.dps = 60


# --------------------------------------------------------------------------
def t7_interval_newton():
    rule("T7. A certified enclosure of the real root by the interval Newton method")
    iv = mp.iv
    iv.dps = 50
    lo = lambda I: mp.make_mpf(I._mpi_[0])
    hi = lambda I: mp.make_mpf(I._mpi_[1])
    mp.mp.dps = 60
    c = iv.log(iv.mpf(49))
    X = iv.mpf([3, 4])
    print("  g' = 1 + ln x on [3, 4] is enclosed by", [mp.nstr(lo(iv.log(X) + 1), 8), mp.nstr(hi(iv.log(X) + 1), 8)],
          "so it has no zero there")
    for n in range(1, 12):
        m = iv.mpf(X.mid)
        N = m - (m * iv.log(m) - c) / (iv.log(X) + 1)          # interval Newton operator
        inside = lo(N) > lo(X) and hi(N) < hi(X)
        Xn = iv.mpf([max(lo(N), lo(X)), min(hi(N), hi(X))])
        width = hi(Xn) - lo(Xn)
        print(f"  step {n}: N(X) strictly inside X: {inside} | width of the new enclosure {mp.nstr(width, 3)}")
        done = width >= hi(X) - lo(X) or width < mp.mpf(10) ** (-48)
        X = Xn
        if done:
            break
    print("  enclosure of the root:")
    print("    lower", mp.nstr(lo(X), 52))
    print("    upper", mp.nstr(hi(X), 52))
    mp.mp.dps = 60
    print("  60-digit reference inside the enclosure:", lo(X) <= R <= hi(X))


def run_all():
    t1_newton_bound()
    t2_raw_step_count()
    t3_orders()
    t4_power_tower()
    t5_fold()
    t6_complex()
    t7_interval_newton()


if __name__ == "__main__":
    run_all()
