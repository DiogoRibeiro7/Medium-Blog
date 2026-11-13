"""Checks for het.py against established implementations and published output.

1. Classical, HC0, HC1, HC2 and HC3 standard errors equal statsmodels' on
   two data sets.
2. HC0 and HC4 standard errors on sandwich::PublicSchools equal the values
   printed by R's sandwich package (vignette "sandwich", saved output of
   version 3.1-3: coeftest(fm.ps, df = Inf, vcov = vcovHC(fm.ps, type = ...))),
   and the HC4 z-test p-value of Income^2 is 0.7725 as printed there, and
   0.8923303 without Alaska, as printed in the package's example output
   (tests/Examples/sandwich-Ex.Rout.save, after Cribari-Neto 2004, Table 8).
3. Koenker's Breusch-Pagan and White's test equal statsmodels'
   het_breuschpagan (robust = True) and het_white.
4. WLS coefficients and their classical and HC0-HC3 standard errors equal
   statsmodels WLS.
5. The vectorised wild bootstrap equals a plain loop over the same weights,
   and with n = 10 Rademacher weights enumerated in full it gives the exact
   wild-bootstrap p-value; under the null its rejection rate is near 5%.
6. Linear quantile regression by linear programming equals statsmodels QuantReg.
7. FGLS with an exponential variance function recovers the variance
   parameters in a large sample.

Run:  python check_het.py   (output also written to results/00_checks.txt)
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

from common import RES, fetch

_LINES = []
_print = print


def print(*args):                       # also keep every line for results/00_checks.txt
    _print(*args)
    _LINES.append(" ".join(str(a) for a in args))

from het import bp_test, fgls, ols, quantreg, t_pvalue, white_test, wild_test, wls

warnings.filterwarnings("ignore")
rng = np.random.default_rng(11)

# 1. against statsmodels
ps = fetch("sandwich", "PublicSchools").dropna()
inc = ps.Income.values * 1e-4
Xps = np.column_stack([np.ones(len(ps)), inc, inc ** 2])
yps = ps.Expenditure.values
n = 300
x = rng.lognormal(size=n)
Xr = np.column_stack([np.ones(n), x, rng.normal(size=n)])
yr = Xr @ [1, 2, -1] + x * rng.normal(size=n)
for X, y in ((Xps, yps), (Xr, yr)):
    f = ols(X, y)
    for mine, theirs in (("const", "nonrobust"), ("HC0", "HC0"), ("HC1", "HC1"), ("HC2", "HC2"), ("HC3", "HC3")):
        ref = sm.OLS(y, X).fit() if theirs == "nonrobust" else sm.OLS(y, X).fit(cov_type=theirs)
        assert np.allclose(f.se(mine), ref.bse, rtol=1e-10), (mine, f.se(mine), ref.bse)
print("classical and HC0-HC3 standard errors equal statsmodels: OK")

# 2. against the printed sandwich output
f = ols(Xps, yps)
assert np.allclose(f.coef, [832.91, -1834.20, 1587.04], atol=0.01)
assert np.allclose(f.se("HC0"), [460.89, 1243.04, 829.99], atol=0.01), f.se("HC0")
assert np.allclose(f.se("HC4"), [3008.01, 8183.19, 5488.93], atol=0.01), f.se("HC4")
p4 = t_pvalue(f.coef[2], f.se("HC4")[2])
assert abs(p4 - 0.7725) < 5e-5, p4
noak = ps.rownames.values != "Alaska"
f1 = ols(Xps[noak], yps[noak])
p4n = t_pvalue(f1.coef[2], f1.se("HC4")[2])
assert abs(p4n - 0.8923303) < 1e-6, p4n          # sandwich-Ex.Rout.save, Cribari-Neto (2004) Table 8
print(f"HC0 {np.round(f.se('HC0'), 2)} and HC4 {np.round(f.se('HC4'), 2)} equal sandwich 3.1-3; "
      f"HC4 p-value of Income^2 {p4:.4f} (printed 0.7725), without Alaska {p4n:.7f} (printed 0.8923303): OK")

# 3. variance tests
for X, y in ((Xps, yps), (Xr, yr)):
    e = ols(X, y).resid
    lm, p = bp_test(e, X[:, 1:])
    ref = het_breuschpagan(e, X)
    assert np.isclose(lm, ref[0]) and np.isclose(p, ref[1])
    lm, p = white_test(e, X[:, 1:])
    ref = het_white(e, X)
    assert np.isclose(lm, ref[0]) and np.isclose(p, ref[1]), (lm, ref)
print("Koenker Breusch-Pagan and White tests equal statsmodels: OK")

# 4. WLS
w = 1 / x
fw = wls(Xr, yr, w)
for mine, theirs in (("const", "nonrobust"), ("HC0", "HC0"), ("HC1", "HC1"), ("HC2", "HC2"), ("HC3", "HC3")):
    ref = sm.WLS(yr, Xr, weights=w).fit() if theirs == "nonrobust" else sm.WLS(yr, Xr, weights=w).fit(cov_type=theirs)
    assert np.allclose(fw.coef, ref.params, rtol=1e-10)
    assert np.allclose(fw.se(mine), ref.bse, rtol=1e-10), (mine, fw.se(mine), ref.bse)
print("WLS coefficients and classical/HC0-HC3 standard errors equal statsmodels WLS: OK")

# 5. wild bootstrap
V = rng.choice([-1.0, 1.0], size=(len(yps), 199))
p_vec, t_obs, t_star = wild_test(Xps, yps, 2, 0.0, V=V, hc="HC3", return_draws=True)
keep = [0, 1]
br = np.linalg.lstsq(Xps[:, keep], yps, rcond=None)[0]
er = yps - Xps[:, keep] @ br
loop = []
for b in range(V.shape[1]):
    ys = Xps[:, keep] @ br + er * V[:, b]
    fb = ols(Xps, ys)
    loop.append(fb.coef[2] / fb.se("HC3")[2])
assert np.allclose(t_star, loop, atol=1e-9)
print("vectorised wild bootstrap equals a plain loop: OK")
n10 = 10
x10 = rng.uniform(size=n10)
X10 = np.column_stack([np.ones(n10), x10])
y10 = 1 + 0.5 * x10 + (0.5 + x10) * rng.normal(size=n10)
Vall = np.array(list(itertools.product([-1.0, 1.0], repeat=n10))).T          # all 1024 sign vectors
p_exact, t0, ts = wild_test(X10, y10, 1, 0.5, V=Vall, return_draws=True)
assert np.isclose(p_exact, np.mean(np.abs(ts) >= abs(t0)))
p_mc = wild_test(X10, y10, 1, 0.5, B=20000, rng=1)
assert abs(p_mc - p_exact) < 0.02, (p_mc, p_exact)
print(f"n = 10, exact wild-bootstrap p-value over all 1,024 sign vectors {p_exact:.4f}; "
      f"Monte Carlo with 20,000 draws {p_mc:.4f}: OK")
rej = np.mean([wild_test(X10 := np.column_stack([np.ones(40), xx := rng.lognormal(size=40)]),
                         1 + 0.5 * xx + xx * rng.normal(size=40), 1, 0.5, B=199, rng=r) < 0.05
               for r in range(400)])
print(f"restricted Rademacher wild bootstrap, n = 40, lognormal x, sd proportional to x: rejection rate "
      f"{rej:.3f} at 5% in 400 samples")
assert 0.02 < rej < 0.09

# 6. quantile regression
for tau in (0.1, 0.5, 0.9):
    mine = quantreg(Xr, yr, tau)
    ref = sm.QuantReg(yr, Xr).fit(q=tau, max_iter=5000).params
    obj = lambda b: np.sum(np.maximum(tau * (yr - Xr @ b), (tau - 1) * (yr - Xr @ b)))
    assert obj(mine) <= obj(ref) + 1e-6 and np.allclose(mine, ref, atol=1e-3), (tau, mine, ref)
print("linear quantile regression equals statsmodels QuantReg (and its check loss is no larger): OK")

# 7. FGLS
n = 200_000
z = rng.uniform(0, 2, n)
Xb = np.column_stack([np.ones(n), z])
yb = 1 + z + np.exp(0.5 * (-1 + 1.5 * z)) * rng.normal(size=n)
fw, var, g = fgls(Xb, yb, z)
assert abs(g[1] - 1.5) < 0.02, g
assert np.allclose(fw.coef, [1, 1], atol=0.01)
print(f"FGLS exponential variance slope {g[1]:.3f} (truth 1.5; the intercept absorbs E log chi2_1): OK")
(RES / "00_checks.txt").write_text("\n".join(_LINES) + "\n")
