"""Checks for rob.py against published and library results.

1. MM on robustbase::starsCYG and datasets::stackloss reproduces the
   coefficients and scale printed by robustbase 0.99-7 (tests/lmrob-data.Rout,
   lmrob with default control: S start, bisquare, 95% efficiency).
2. Huber M-estimation reproduces statsmodels RLM (HuberT, MAD scale) exactly.
3. LTS and S recover the true coefficients on data with 40% bad leverage points.

Run:  python check_rob.py
"""

import warnings

import numpy as np
import statsmodels.api as sm

from common import fetch
from rob import lts, m_estimate, mm_estimate, s_estimate

warnings.filterwarnings("ignore")

st = fetch("robustbase", "starsCYG")
X = np.column_stack([np.ones(len(st)), st["log.Te"]])
f = mm_estimate(X, st["log.light"].values, n_sub=500, seed=0)
assert np.allclose(f.coef, [-4.9694, 2.2532], atol=5e-4), f.coef
assert abs(f.scale - 0.4715) < 5e-4
assert np.allclose(f.init.coef, [-9.5708, 3.2904], atol=2e-3), f.init.coef
print(f"starsCYG MM {np.round(f.coef, 4)} scale {f.scale:.4f} (robustbase: -4.9694, 2.2532; 0.4715)")

sl = fetch("datasets", "stackloss")
X = np.column_stack([np.ones(len(sl)), sl["Air.Flow"], sl["Water.Temp"], sl["Acid.Conc."]])
f = mm_estimate(X, sl["stack.loss"].values, n_sub=500, seed=0)
assert np.allclose(f.coef, [-41.5246, 0.9388, 0.5796, -0.1129], atol=5e-4), f.coef
assert abs(f.scale - 1.91) < 0.01
print(f"stackloss MM {np.round(f.coef, 4)} scale {f.scale:.3f} "
      f"(robustbase: -41.5246, 0.9388, 0.5796, -0.1129; 1.91)")

y = sl["stack.loss"].values
r = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
h = m_estimate(X, y, "huber")
assert np.allclose(h.coef, r.params, atol=1e-6) and np.allclose(h.se, r.bse, rtol=1e-4)
print("Huber M vs statsmodels RLM: identical coefficients and standard errors")

rng = np.random.default_rng(1)
n = 400
X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(0, 0.5, n)
bad = rng.random(n) < 0.4
X[bad, 1] += 8
for name, fit in (("LTS", lts(X, y, seed=2)), ("S", s_estimate(X, y, seed=2)),
                  ("MM", mm_estimate(X, y, seed=2))):
    assert np.allclose(fit.coef, [1, 2, -1], atol=0.15), (name, fit.coef)
print("LTS, S and MM recover the coefficients with 40% bad leverage points: OK")
