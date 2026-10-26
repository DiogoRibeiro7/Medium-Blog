"""Experiment 5 - MM-estimation in Python, three ways, against robustbase.

* rob.py: FAST-S start (500 random subsamples), bisquare MM step, sandwich SE.
* statsmodels RLMDetSMM (new in statsmodels 0.15): S start from deterministic
  starting sets (after Hubert, Rousseeuw & Verdonck 2012), then RLM's
  bisquare MM step with RLM's H1 covariance.
* statsmodels RLM with TukeyBiweight(c = 4.685), started from the rob.py
  S-estimate with its scale held fixed: the "do the MM step yourself" recipe.

Reference values: robustbase 0.99-7, lmrob() with default control
(tests/lmrob-data.Rout and summary(lmrob(log.light ~ log.Te, starsCYG))).

Run:  python 05_software.py
"""

import warnings

import numpy as np
import statsmodels.api as sm
from statsmodels.robust.resistant_linear_model import RLMDetS, RLMDetSMM

from common import fetch, write
from rob import C_MM, m_sandwich, mm_estimate, s_scale

warnings.filterwarnings("ignore")

st = fetch("robustbase", "starsCYG")
sl = fetch("datasets", "stackloss")
cases = {
    "stars": (np.column_stack([np.ones(len(st)), st["log.Te"]]), st["log.light"].values,
              "coef -4.9694, 2.2532; SE 3.4100, 0.7690; scale 0.4715"),
    "stackloss": (np.column_stack([np.ones(len(sl)), sl["Air.Flow"], sl["Water.Temp"], sl["Acid.Conc."]]),
                  sl["stack.loss"].values, "coef -41.5246, 0.9388, 0.5796, -0.1129; scale 1.91"),
}


def fmt(v):
    return ", ".join(f"{x:.4f}" for x in v)


out = []
for name, (X, y, ref) in cases.items():
    out.append(f"== {name}: n = {len(y)}; robustbase lmrob: {ref}")
    mm = mm_estimate(X, y, n_sub=500, seed=0)
    out.append(f"  rob.py MM                        coef {fmt(mm.coef)}; SE {fmt(mm.se)}; "
               f"scale {mm.scale:.4f}")
    d = RLMDetSMM(y, X).fit()
    sand = np.sqrt(np.diag(m_sandwich(X, y - X @ d.params, d.scale, "bisquare", C_MM)))
    out.append(f"  statsmodels RLMDetSMM            coef {fmt(d.params)}; SE (H1) {fmt(d.bse)}; "
               f"sandwich SE at its fit {fmt(sand)}; scale {d.scale:.4f}")
    r = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight(c=C_MM)).fit(
        start_params=mm.init.coef, start_scale=mm.init.scale, update_scale=False)
    out.append(f"  statsmodels RLM from rob.py S    coef {fmt(r.params)}; SE (H1) {fmt(r.bse)}; "
               f"max |coef - rob.py MM| {np.max(np.abs(r.params - mm.coef)):.1e}")
    # in small samples the S objective depends on the convention for the
    # M-scale equation (robustbase averages rho over n - p rather than n), so
    # each S solution is evaluated under both conventions
    n, k = X.shape
    ds = RLMDetS(y, X).fit(max(n // 2 + 1, k + 1))
    rd, rr = y - X @ np.asarray(ds.params), mm.init.resid
    out.append(f"  S starts: statsmodels DetS coef {fmt(ds.params)}; rob.py FAST-S coef {fmt(mm.init.coef)}")
    out.append(f"  M-scale of the S residuals, (n - p) convention: DetS {s_scale(rd, p=k):.4f}, "
               f"rob.py {s_scale(rr, p=k):.4f}; n convention: DetS {s_scale(rd, p=0):.4f}, "
               f"rob.py {s_scale(rr, p=0):.4f}")
write("05_software.txt", "\n".join(out))
