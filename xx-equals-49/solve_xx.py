"""Solving x**x = 49 numerically, six ways.

Reproduces every number quoted in the article.
Requires: numpy, scipy, mpmath.
"""
from __future__ import annotations

import math
import time

import mpmath as mp
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

K = 49.0
C = math.log(K)                      # the equation in log form is x ln x = C

# ---------------------------------------------------------------- reference
mp.mp.dps = 60
X_REF = mp.exp(mp.lambertw(mp.log(49)))
X_STAR = float(X_REF)


def err(x: float) -> float:
    return abs(float(mp.mpf(x) - X_REF))


# ---------------------------------------------------------------- 1. bisection
def bisection(f, lo, hi, tol=0.0, max_iter=200):
    """Halve [lo, hi] until it cannot be halved any further."""
    flo = f(lo)
    hist = [0.5 * (lo + hi)]
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi or (hi - lo) <= tol:
            break
        fmid = f(mid)
        if (fmid > 0) == (flo > 0):
            lo, flo = mid, fmid
        else:
            hi = mid
        hist.append(0.5 * (lo + hi))
    return hist


# ---------------------------------------------------------------- 2. Newton
def newton(f, df, x0, tol=1e-15, max_iter=500):
    hist = [x0]
    x = x0
    for _ in range(max_iter):
        step = f(x) / df(x)
        x = x - step
        hist.append(x)
        if abs(step) <= tol * abs(x):
            break
    return hist


def f_raw(x):                         # x**x - 49
    return x ** x - K


def df_raw(x):
    return x ** x * (math.log(x) + 1.0)


def f_log(x):                         # x ln x - ln 49
    return x * math.log(x) - C


def df_log(x):
    return math.log(x) + 1.0


# ---------------------------------------------------------------- 3. fixed point
def fixed_point(phi, x0, tol=1e-15, max_iter=400):
    hist = [x0]
    x = x0
    for _ in range(max_iter):
        try:
            x_new = phi(x)
        except (OverflowError, ValueError, ZeroDivisionError):
            break
        hist.append(x_new)
        if not math.isfinite(x_new) or abs(x_new - x) <= tol * abs(x_new):
            break
        x = x_new
    return hist


def phi_a(x):                         # x <- C / ln x
    return C / math.log(x)


def phi_b(x):                         # x <- exp(C / x)
    return math.exp(C / x)


# ---------------------------------------------------------------- 4. Lambert W
def lambert_w0(a, tol=1e-15, max_iter=50, return_hist=False):
    """Principal branch of W for a > 0, by Halley's iteration on w e^w = a."""
    w = math.log1p(a)                 # crude but safe starting point
    hist = [w]
    for _ in range(max_iter):
        ew = math.exp(w)
        g = w * ew - a
        step = g / (ew * (w + 1.0) - (w + 2.0) * g / (2.0 * w + 2.0))
        w -= step
        hist.append(w)
        if abs(step) <= tol * abs(w):
            break
    return (w, hist) if return_hist else w


def newton_w(a, w0, tol=1e-15, max_iter=50):
    """Plain Newton on w e^w = a, for comparison with Newton on x ln x = C."""
    hist = [w0]
    w = w0
    for _ in range(max_iter):
        ew = math.exp(w)
        step = (w * ew - a) / (ew * (w + 1.0))
        w -= step
        hist.append(w)
        if abs(step) <= tol * abs(w):
            break
    return hist


def run():
    out = {}
    print("=" * 72)
    print("reference (mpmath, 60 digits)")
    print("  x*        =", mp.nstr(X_REF, 50))
    print("  ln x*     =", mp.nstr(mp.log(X_REF), 30))
    print("  W(ln 49)  =", mp.nstr(mp.lambertw(mp.log(49)), 30))
    print("  ln 49     =", mp.nstr(mp.log(49), 30))
    print("  x*^x*     =", mp.nstr(X_REF ** X_REF, 30))
    print("  float64 x*=", repr(X_STAR), " x**x =", repr(X_STAR ** X_STAR))
    print("  3.278^3.278 =", 3.278 ** 3.278)

    print("=" * 72)
    print("shape of x^x")
    print("  argmin 1/e =", 1 / math.e, " min e^(-1/e) =", math.exp(-1 / math.e))
    for v in (0.1, 0.2, 1 / math.e, 0.5, 1, 2, 3, 4):
        print(f"  {v:8.5f}^{v:<8.5f} = {v ** v:.6f}")

    print("=" * 72)
    print("1. bisection on x ln x - ln 49, bracket [3, 4]")
    h = bisection(f_log, 3.0, 4.0)
    out["Bisection"] = h
    print("  iterations to float64 limit:", len(h) - 1, " x =", repr(h[-1]), " err =", err(h[-1]))
    for n in (1, 5, 10, 20, 30, 40, 50):
        if n < len(h):
            print(f"   after {n:2d}: {h[n]:.16f}  err {err(h[n]):.2e}")
    # first iteration with error below thresholds
    for thr in (1e-3, 1e-6, 1e-12):
        n = next(i for i, v in enumerate(h) if err(v) < thr)
        print(f"   first err < {thr:g}: iteration {n}")

    print("=" * 72)
    print("2. Newton")
    for name, f, df in (("x^x - 49", f_raw, df_raw), ("x ln x - ln 49", f_log, df_log)):
        for x0 in (3.0, 4.0, 1.0, 10.0):
            h = newton(f, df, x0)
            print(f"  {name:16s} x0={x0:5.1f}: {len(h) - 1:3d} iterations, x = {h[-1]!r}, err = {err(h[-1]):.2e}")
            if x0 == 4.0:
                out["Newton on " + name] = h
            if x0 == 1.0:
                print("      first iterates:", [round(v, 4) for v in h[:6]])
                out["Newton from 1 on " + name] = h
    h = out["Newton on x ln x - ln 49"]
    print("  log form from 4, errors:", [f"{err(v):.2e}" for v in h])
    h = out["Newton on x^x - 49"]
    print("  raw form from 4, errors:", [f"{err(v):.2e}" for v in h])
    lx = math.log(X_STAR)
    c_log = (1 / X_STAR) / (2 * (lx + 1))
    c_raw = ((lx + 1) ** 2 + 1 / X_STAR) / (2 * (lx + 1))
    print(f"  asymptotic error constants |f''/2f'|: log form {c_log:.4f}, raw form {c_raw:.4f}, ratio {c_raw / c_log:.1f}")

    print("=" * 72)
    print("3. fixed-point iterations from x0 = 3")
    ha = fixed_point(phi_a, 3.0)
    hb = fixed_point(phi_b, 3.0, max_iter=40)
    out["Fixed point x = ln49/ln x"] = ha
    out["Fixed point x = exp(ln49/x)"] = hb
    print("  A: x <- C/ln x     :", len(ha) - 1, "iterations, x =", repr(ha[-1]), "err =", err(ha[-1]))
    print("     first:", [round(v, 5) for v in ha[:8]])
    print("     theoretical rate |phi'| = 1/ln x* =", 1 / lx)
    ratios = [err(ha[i + 1]) / err(ha[i]) for i in range(40, 46)]
    print("     observed error ratios:", [round(r, 4) for r in ratios])
    for thr in (1e-3, 1e-6, 1e-12):
        n = next(i for i, v in enumerate(ha) if err(v) < thr)
        print(f"     first err < {thr:g}: iteration {n}")
    print("  B: x <- exp(C/x)   : first:", [round(v, 5) for v in hb[:10]])
    print("     last few:", [f"{v:.6g}" for v in hb[-6:]])
    print("     theoretical rate |phi'| = ln x* =", lx)
    # start B extremely close to the root
    hb2 = fixed_point(phi_b, X_STAR + 1e-9, max_iter=200)
    print("     started 1e-9 from the root, error after 0/50/100/120 steps:",
          [f"{err(hb2[i]):.2e}" for i in (0, 50, 100, 120)])
    print("     two-cycle it settles on:", [repr(v) for v in hb2[-2:]])
    xc = 4.0
    seq = [xc]
    for _ in range(12):
        xc = (xc + C) / (1.0 + math.log(xc))
        seq.append(xc)
    print("  C: x <- (x + C)/(1 + ln x) from 4:", [repr(v) for v in seq[:6]])
    print("     settles on:", sorted({repr(v) for v in seq[6:]}))
    print("  crossover: ln x* = 1 at x* = e, i.e. k = e^e =", math.e ** math.e)
    for k in (5.0, 15.0, 16.0, 49.0, 1000.0):
        c = math.log(k)
        xa = fixed_point(lambda x: c / math.log(x), 2.5, max_iter=5000)
        xb = fixed_point(lambda x: math.exp(c / x), 2.5, max_iter=5000)
        root = math.exp(lambertw(c).real)
        print(f"     k={k:7.1f} root={root:.6f}  A: {len(xa) - 1:4d} its -> {xa[-1]:.6f} | B: {len(xb) - 1:4d} its -> {xb[-1]:.6f}")

    print("=" * 72)
    print("4. Lambert W by hand (Halley) vs SciPy")
    w, hw = lambert_w0(C, return_hist=True)
    out["Halley on w e^w (Lambert W)"] = [math.exp(v) for v in hw]
    print("  start log1p(C) =", hw[0], "-> x0 =", math.exp(hw[0]))
    print("  Halley iterations:", len(hw) - 1, " W =", repr(w), " exp(W) =", repr(math.exp(w)))
    print("  errors in x:", [f"{err(math.exp(v)):.2e}" for v in hw])
    print("  scipy lambertw(C) =", lambertw(C), " exp =", repr(math.exp(lambertw(C).real)))
    print("  err ours:", err(math.exp(w)), " err scipy:", err(math.exp(lambertw(C).real)))
    hn = newton_w(C, math.log(4.0))
    print("  Newton on w e^w from w0 = ln 4:", len(hn) - 1, "iterations; x iterates:",
          [f"{math.exp(v):.12f}" for v in hn])
    hx = out["Newton on x ln x - ln 49"]
    print("  Newton on x ln x from x0 = 4      :", len(hx) - 1, "iterations; x iterates:",
          [f"{v:.12f}" for v in hx])

    print("=" * 72)
    print("5. library root-finder: brentq on x ln x - ln 49 over [3, 4]")
    r, info = brentq(f_log, 3.0, 4.0, xtol=1e-15, rtol=4 * np.finfo(float).eps, full_output=True)
    print("  root =", repr(r), " iterations =", info.iterations, " f evals =", info.function_calls, " err =", err(r))
    r2, info2 = brentq(f_raw, 3.0, 4.0, xtol=1e-15, rtol=4 * np.finfo(float).eps, full_output=True)
    print("  on raw form: root =", repr(r2), " iterations =", info2.iterations, " f evals =", info2.function_calls, " err =", err(r2))

    print("=" * 72)
    print("6. two roots when e^(-1/e) < k < 1, e.g. k = 0.8")
    c8 = math.log(0.8)
    x_hi = math.exp(lambertw(c8, 0).real)
    x_lo = math.exp(lambertw(c8, -1).real)
    print("  ln 0.8 =", c8, " -1/e =", -1 / math.e)
    print("  W0 root :", repr(x_hi), " check x^x =", x_hi ** x_hi)
    print("  W-1 root:", repr(x_lo), " check x^x =", x_lo ** x_lo)
    print("  brentq on [1e-9, 1/e]:", brentq(lambda x: x * math.log(x) - c8, 1e-9, 1 / math.e),
          " on [1/e, 1]:", brentq(lambda x: x * math.log(x) - c8, 1 / math.e, 1.0))

    print("=" * 72)
    print("7. sensitivity")
    dxdk = 1.0 / (K * (lx + 1.0))
    print("  dx/dk at 49 =", dxdk, " relative condition number =", K * dxdk / X_STAR)
    print("  via W'(a) = W/(a(1+W)):", (lx / (C * (1 + lx))) * X_STAR / K)
    h = 1e-4
    fd = (math.exp(lambertw(math.log(K + h)).real) - math.exp(lambertw(math.log(K - h)).real)) / (2 * h)
    print("  finite difference       :", fd)
    print("  x(50) - x(49) =", math.exp(lambertw(math.log(50.0)).real) - X_STAR)

    print("=" * 72)
    print("8. a million right-hand sides")
    rng = np.random.default_rng(7)
    ks = np.exp(rng.uniform(math.log(1.5), math.log(1e12), size=1_000_000))
    cs = np.log(ks)

    def solve_xlnx(c, sweeps=6, start="simple"):
        """Solve x ln x = c elementwise: Newton on the log form, collapsed to one line."""
        c = np.asarray(c, dtype=float)
        if start == "simple":
            x = 1.0 + c
        else:                                             # large-c asymptotics of W
            x = np.where(c > math.e, c / np.log(np.maximum(c, math.e)), 1.0 + c)
        for _ in range(sweeps):
            x = (x + c) / (1.0 + np.log(x))
        return x

    def best_of(fn, reps=7):
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            res = fn()
            ts.append(time.perf_counter() - t)
        return min(ts), res

    t_n, x_n = best_of(lambda: solve_xlnx(cs))
    t_w, x_w = best_of(lambda: np.exp(lambertw(cs).real))
    resid_n = np.abs(x_n * np.log(x_n) - cs) / cs
    resid_w = np.abs(x_w * np.log(x_w) - cs) / cs
    print(f"  vectorised Newton (6 sweeps): {t_n * 1e3:7.1f} ms, max rel residual {resid_n.max():.2e}")
    print(f"  exp(lambertw(log k))        : {t_w * 1e3:7.1f} ms, max rel residual {resid_w.max():.2e}")
    print(f"  ratio lambertw / Newton: {t_w / t_n:.1f}")
    print(f"  max rel disagreement between the two: {np.max(np.abs(x_n - x_w) / x_w):.2e}")
    for start in ("simple", "asymptotic"):
        for n_iter in (3, 4, 5, 6, 7):
            xx = solve_xlnx(cs, n_iter, start)
            print(f"   start={start:10s} sweeps={n_iter}: max rel diff vs lambertw {np.max(np.abs(xx - x_w) / x_w):.2e}")
    x1 = (1.0 + cs + cs) / (1.0 + np.log(1.0 + cs))
    print("  after one sweep every iterate is to the right of its root:", bool(np.all(x1 >= x_w)))
    print("  k range: 1.5 to 1e12; root range:", float(x_w.min()), "to", float(x_w.max()))
    t0 = time.perf_counter()
    sub = [brentq(lambda x, c=c: x * math.log(x) - c, 1.0, 1e12) for c in cs[:20000]]
    t_b = (time.perf_counter() - t0) * 50
    print(f"  brentq in a Python loop (extrapolated from 20k): {t_b * 1e3:7.0f} ms")

    print("=" * 72)
    print("9. floating-point neighbours of the root")
    cands = [X_STAR]
    lo = hi = X_STAR
    for _ in range(3):
        lo, hi = math.nextafter(lo, 0.0), math.nextafter(hi, 10.0)
        cands += [lo, hi]
    print("  ulp at the root:", math.ulp(X_STAR))
    for v in sorted(cands):
        print(f"   {v!r:22s} -> x**x = {v ** v!r}")
    v, hits = X_STAR, 0
    for _ in range(1000):
        v = math.nextafter(v, 0.0)
    for _ in range(2000):
        hits += (v ** v == 49.0)
        v = math.nextafter(v, 10.0)
    print("  doubles within 1000 ulp of the root with x**x == 49.0 exactly:", hits)

    print("=" * 72)
    print("10. the two-cycle of x <- 49**(1/x)")
    hb3 = fixed_point(phi_b, 3.0, max_iter=400)
    a, b = sorted(hb3[-2:])
    print("  a =", repr(a), " b =", repr(b))
    print("  a**b =", a ** b, "  b**a =", b ** a)
    print("  multiplier of the two-cycle (ln k)^2 / (ab) =", C ** 2 / (a * b))

    print("=" * 72)
    print("11. the same equation in a familiar costume: n log2 n = budget")
    for budget in (1e6, 1e9, 1e12):
        c = budget * math.log(2.0)
        n = math.exp(lambertw(c).real)
        print(f"  budget {budget:.0e}: n = {n:,.0f}   check n log2 n = {n * math.log2(n):.6e}")

    print("=" * 72)
    print("12. assorted figures quoted in the text")
    print("  49**49 =", 49.0 ** 49.0)
    print("  Newton step of the raw form at x = 49: 1/(ln 49 + 1) =", 1.0 / (C + 1.0))
    big = brentq(lambda x: x * math.log(x) - math.log(1.7976931348623157e308), 100.0, 200.0)
    print("  x**x overflows a double beyond x =", big)
    r16 = math.exp(lambertw(math.log(16.0)).real)
    print("  k = 16: root", r16, " contraction factor of A = 1/ln r =", 1.0 / math.log(r16))
    r, info = brentq(f_log, 3.0, 4.0, xtol=1e-15, full_output=True)
    print("  brentq default rtol, log form:", repr(r), info.function_calls, "evaluations")
    r, info = brentq(f_raw, 3.0, 4.0, xtol=1e-15, full_output=True)
    print("  brentq default rtol, raw form:", repr(r), info.function_calls, "evaluations,",
          round((r - X_STAR) / math.ulp(X_STAR)), "ulp from the correctly rounded root")
    wd = lambertw(C).real
    print("  double-precision W error in ulps of W:", float(abs(mp.mpf(wd) - mp.lambertw(mp.log(49)))) / math.ulp(wd))
    h = 1.0
    seq = []
    for _ in range(6):
        h = 0.5 * (h + 2.0 / h)
        seq.append(repr(h))
    print("  Heron's iteration for sqrt(2) from 1:", seq, " math.sqrt(2) =", repr(math.sqrt(2.0)))
    return out


if __name__ == "__main__":
    run()
