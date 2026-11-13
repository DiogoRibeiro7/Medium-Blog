"""Checks for sel.py against an established implementation and its printed output.

1. The probit equals statsmodels' Probit.
2. Heckman's two-step estimates, their corrected standard errors, sigma and rho
   equal the output printed by the R package sampleSelection 1.2-16
   (tests/Mroz87SelectionTest.Rout.save) for two specifications on Mroz87:
   Wooldridge's (log wage; nwifeinc, age, kids5 and kids618 excluded from the
   wage equation) and Greene's (wage in levels; age, age^2, family income and a
   dummy for children excluded).
3. Maximum likelihood estimates, standard errors and the log-likelihood equal
   the same output; for Greene's specification, the published maximum is a
   local one, and a second start reaches the higher maximum reported in the
   sampleSelection vignette (log-likelihood -1479.654).
4. With a degree-1 polynomial the series control function equals the two-step
   slopes; on a large simulated sample with an exclusion restriction the
   two-step and ML estimators recover the true parameters.

Run:  python check_sel.py   (output also written to results/00_checks.txt)
"""

import warnings

import numpy as np
import statsmodels.api as sm

from common import RES, fetch
from sel import heckit, heckman_ml, probit, series_cf

warnings.filterwarnings("ignore")
LINES = []


def fmt(a, nd=6):
    return "[" + ", ".join(f"{v:.{nd}f}" for v in np.asarray(a)) + "]"


def say(s):
    print(s)
    LINES.append(s)


m = fetch("sampleSelection", "Mroz87")
d = m.lfp.values.astype(bool)
one = np.ones(len(m))

# Wooldridge (2003), example 17.5
Zw = np.column_stack([one, m.nwifeinc, m.educ, m.exper, m.exper ** 2, m.age, m.kids5, m.kids618])
Xw = np.column_stack([one, m.educ, m.exper, m.exper ** 2])
yw = np.log(np.where(d, m.wage, 1.0))

# 1. probit
p = probit(Zw, d)
ref = sm.Probit(d.astype(float), Zw).fit(disp=0)
assert np.allclose(p.coef, ref.params, atol=1e-8) and np.allclose(np.sqrt(np.diag(p.cov)), ref.bse, rtol=1e-6)
assert abs(p.loglik - ref.llf) < 1e-8
say("probit equals statsmodels Probit (coefficients, standard errors, log-likelihood): OK")

# 2. two-step, Wooldridge
h = heckit(Zw, d, Xw, yw)
assert np.allclose(h.probit.coef, [0.270077, -0.012024, 0.130905, 0.123348, -0.001887, -0.052853, -0.868329, 0.036005],
                   atol=6e-7)
assert np.allclose(h.beta, [-0.578103, 0.109066, 0.043887, -0.000859], atol=6e-7)
assert abs(h.beta_lambda - 0.032262) < 6e-7 and abs(h.sigma - 0.663629) < 6e-7 and abs(h.rho - 0.048614) < 6e-7
se = h.se()
assert np.allclose(se[:4], [0.305006, 0.015523, 0.016261, 0.000439], atol=6e-7) and abs(se[4] - 0.1336) < 6e-5
say(f"two-step, Wooldridge's specification: outcome {fmt(h.beta)}, inverse Mills ratio {h.beta_lambda:.6f} "
    f"(SE {se[4]:.4f}), sigma {h.sigma:.6f}, rho {h.rho:.6f}, outcome SEs {fmt(se[:4])} equal "
    f"sampleSelection 1.2-16: OK")

# 2. two-step, Greene
kids = ((m.kids5 + m.kids618) > 0).astype(float)
Zg = np.column_stack([one, m.age, m.age ** 2, m.faminc, kids, m.educ])
Xg = np.column_stack([one, m.exper, m.exper ** 2, m.educ, m.city])
yg = np.where(d, m.wage, 0.0)
hg = heckit(Zg, d, Xg, yg)
assert np.allclose(hg.beta, [-0.971200, 0.021061, 0.000137, 0.417017, 0.443838], atol=6e-7)
assert np.allclose(hg.se()[:5], [2.059351, 0.062465, 0.001878, 0.100250, 0.315898], atol=6e-7)
assert abs(hg.beta_lambda + 1.098) < 6e-4 and abs(hg.se()[5] - 1.266) < 6e-4
assert abs(hg.sigma - 3.200) < 6e-4 and abs(hg.rho + 0.343) < 6e-4
say(f"two-step, Greene's specification: outcome {fmt(hg.beta)}, SEs {fmt(hg.se()[:5])}, inverse Mills "
    f"ratio {hg.beta_lambda:.3f} (SE {hg.se()[5]:.3f}), sigma {hg.sigma:.3f}, rho {hg.rho:.3f} equal sampleSelection: OK")

# 3. maximum likelihood, Wooldridge
ml = heckman_ml(Zw, d, Xw, yw)
assert ml.converged
assert np.allclose(ml.gamma, [0.266449, -0.012132, 0.131341, 0.123282, -0.001886, -0.052829, -0.867399, 0.035872],
                   atol=2e-5)
assert np.allclose(ml.beta, [-0.552696, 0.108350, 0.042837, -0.000837], atol=2e-5)
assert abs(ml.sigma - 0.663398) < 2e-5 and abs(ml.rho - 0.026607) < 5e-5
s = ml.se()
kz, kx = Zw.shape[1], Xw.shape[1]
assert np.allclose(s[kz:kz + kx], [0.260379, 0.014861, 0.014879, 0.000417], rtol=2e-3)
assert abs(s[-2] - 0.0227) < 6e-5 and abs(s[-1] - 0.1471) < 6e-5 and round(ml.loglik) == -833
say(f"ML, Wooldridge's specification: outcome {fmt(ml.beta)}, sigma {ml.sigma:.6f}, rho {ml.rho:.6f}, "
    f"outcome SEs {fmt(s[kz:kz + kx])}, SE(sigma) {s[-2]:.4f}, SE(rho) {s[-1]:.4f}, log-likelihood "
    f"{ml.loglik:.3f} equal sampleSelection (printed -833): OK")

# 3. maximum likelihood, Greene: two maxima
mg = heckman_ml(Zg, d, Xg, yg)
assert abs(mg.loglik + 1581.258) < 1e-3, mg.loglik
assert np.allclose(mg.beta, [-1.963024, 0.027868, -0.000104, 0.457005, 0.446529], atol=5e-5)
assert abs(mg.sigma - 3.108) < 6e-4 and abs(mg.rho + 0.132) < 6e-4
best = mg
for r0 in (0.5, 0.9, -0.9):
    st = mg.theta.copy()
    st[-1] = np.arctanh(r0)
    alt = heckman_ml(Zg, d, Xg, yg, start=st)
    if alt.loglik > best.loglik + 1e-6:
        best, rbest = alt, r0
assert abs(best.loglik + 1479.654) < 1e-2, best.loglik
say(f"ML, Greene's specification from the two-step start: log-likelihood {mg.loglik:.3f}, outcome {fmt(mg.beta)}, "
    f"sigma {mg.sigma:.3f}, rho {mg.rho:.3f} equal sampleSelection; started at rho = {rbest} it reaches the higher "
    f"maximum {best.loglik:.3f} (vignette: -1479.654), with rho {best.rho:.4f}, educ {best.beta[3]:.4f}: OK")

# 4. series and a large-sample recovery
assert np.allclose(series_cf(Zw, d, Xw, yw, degree=1), h.beta, atol=1e-10)
rng = np.random.default_rng(1)
n = 200_000
x, z = rng.normal(size=n), rng.normal(size=n)
uv = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
dd = (0.2 + 0.5 * x + 1.0 * z + uv[:, 1]) > 0
yy = 1 + 0.5 * x + uv[:, 0]
Z = np.column_stack([np.ones(n), x, z])
X = np.column_stack([np.ones(n), x])
hh = heckit(Z, dd, X, yy)
mm = heckman_ml(Z, dd, X, yy)
assert np.allclose(hh.beta, [1, 0.5], atol=0.02) and abs(hh.rho - 0.6) < 0.03
assert np.allclose(mm.beta, [1, 0.5], atol=0.02) and abs(mm.rho - 0.6) < 0.03 and abs(mm.sigma - 1) < 0.01
say(f"large simulated sample (n = 200,000, rho = 0.6): two-step slope {hh.beta[1]:.4f}, rho {hh.rho:.3f}; ML slope "
    f"{mm.beta[1]:.4f}, rho {mm.rho:.3f}, sigma {mm.sigma:.3f} (truth 0.5, 0.6, 1): OK")
(RES / "00_checks.txt").write_text("\n".join(LINES) + "\n")
