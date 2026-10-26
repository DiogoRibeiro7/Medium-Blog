"""Checks for me.py.

1. Large-sample consistency under classical error: naive slope -> lambda * beta,
   moment correction, regression calibration (replicates and validation),
   2SLS and MI recover beta; SIMEX with the exact extrapolant does too and the
   quadratic one under-corrects by the known amount.
2. In linear regression, regression calibration with replicates equals the
   moment correction with the error variance of the replicate mean (they are
   the same estimator; Carroll et al. 2006, sec. 4.4).
3. 2SLS matches statsmodels' IV formula computed by hand from the projection.
4. Logistic Newton-Raphson matches statsmodels Logit.
5. MI with a 100% validation sample reproduces the fit on the true x.

Run:  python check_me.py
"""

import warnings

import numpy as np
import statsmodels.api as sm

from me import (calibrate_replicates, design, iv_replicate, logit, mime, moment_correct, ols,
                rc_replicates, rc_validation, simex)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
n = 200_000
z = rng.normal(size=n)
x = 0.6 * z + rng.normal(size=n)                    # var(x) = 1.36, x correlated with z
beta = np.array([1.0, 2.0, -1.0])
y = design(x, z) @ beta + rng.normal(size=n)
s_u2 = 0.6
W = x[:, None] + np.sqrt(s_u2) * rng.normal(size=(n, 2))
lam_cond = 1.0 / (1.0 + s_u2)                       # var(x | z) = 1
naive = ols(design(W[:, 0], z), y).coef
assert abs(naive[1] - beta[1] * lam_cond) < 0.02, naive
assert abs(naive[2] - (beta[2] + beta[1] * (1 - lam_cond) * 0.6)) < 0.02, naive   # leakage into z
print(f"naive slope {naive[1]:.3f} (theory {beta[1] * lam_cond:.3f}); z coefficient {naive[2]:.3f} "
      f"(theory {beta[2] + beta[1] * (1 - lam_cond) * 0.6:.3f})")

for name, b in (("moment", moment_correct(W[:, 0], z, y, s_u2).coef),
                ("RC replicates", rc_replicates(W, z, y).coef),
                ("RC validation 20%", rc_validation(W, z, y, x, rng.random(n) < 0.2).coef),
                ("2SLS", iv_replicate(W, z, y).coef),
                ("MI 20%", mime(W, z, y, x, rng.random(n) < 0.2, m=5, rng=1).coef)):
    assert np.allclose(b, beta, atol=0.03), (name, b)
    print(f"{name:18s} {np.round(b, 3)}")

# 2. RC with replicates == moment correction with s_u^2 / k on the replicate mean
xhat, s_hat = calibrate_replicates(W, z)
rc = ols(design(xhat, z), y).coef
mc = moment_correct(W.mean(axis=1), z, y, s_hat / 2).coef
assert np.allclose(rc, mc, atol=1e-8), (rc, mc)
print("RC with replicates equals the moment correction: OK")

# 1b. SIMEX: quadratic extrapolant vs exact rational-linear target
f, lam, path = simex(W[:, 0], z, y, s_u2, B=5, rng=3, return_path=True)
# the exact path for the slope is b(l) = beta * v / (v + (1 + l) s_u2), v = var(x|z) = 1
exact = beta[1] * 1 / (1 + (1 + lam) * s_u2)
assert np.allclose(path[:, 1], exact, atol=0.02), (path[:, 1], exact)
from me import extrapolate
rat = extrapolate(lam, path, "rational")
assert abs(rat[1] - beta[1]) < 0.03, rat
print(f"SIMEX path matches theory; quadratic extrapolation {f.coef[1]:.3f}, rational {rat[1]:.3f} "
      f"(truth {beta[1]})")

# 3. 2SLS against a hand-built just-identified IV estimate
X, Zi = design(W[:2000, 0], z[:2000]), design(W[:2000, 1], z[:2000])
b_iv = np.linalg.solve(Zi.T @ X, Zi.T @ y[:2000])
assert np.allclose(iv_replicate(W[:2000], z[:2000], y[:2000]).coef, b_iv, atol=1e-10)
print("2SLS equals (Z'X)^-1 Z'y when just identified: OK")

# 4. logistic against statsmodels
p = 1 / (1 + np.exp(-(design(x[:5000], z[:5000]) @ np.array([-1.0, 0.8, 0.5]))))
yb = (rng.random(5000) < p).astype(float)
mine = logit(design(x[:5000], z[:5000]), yb)
ref = sm.Logit(yb, design(x[:5000], z[:5000])).fit(disp=0)
assert np.allclose(mine.coef, ref.params, atol=1e-8) and np.allclose(mine.se, ref.bse, rtol=1e-6)
print("logistic Newton-Raphson matches statsmodels Logit: OK")

# 5. MI with everything validated is the fit on the true x
all_val = np.ones(5000, bool)
mi = mime(W[:5000], z[:5000], y[:5000], x[:5000], all_val, m=3, rng=0)
assert np.allclose(mi.coef, ols(design(x[:5000], z[:5000]), y[:5000]).coef)
print("MI with a full validation sample reproduces the true-x fit: OK")

# 6. multivariate moment correction: two error-prone covariates with different error variances
x2 = 0.5 * x + rng.normal(size=n)
y2 = 1 + 2 * x - 1 * x2 + 0.5 * z + rng.normal(size=n)
W2 = np.column_stack([x + np.sqrt(0.6) * rng.normal(size=n), x2 + np.sqrt(0.3) * rng.normal(size=n)])
b2 = moment_correct(W2, z, y2, [0.6, 0.3]).coef
assert np.allclose(b2, [1, 2, -1, 0.5], atol=0.03), b2
print(f"multivariate moment correction recovers [1, 2, -1, 0.5]: {np.round(b2, 3)}")
