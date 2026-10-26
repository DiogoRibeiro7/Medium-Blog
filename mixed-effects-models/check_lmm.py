"""Cross-check lmm.LMM against statsmodels.MixedLM (REML) on two of the article's models."""

import warnings

import numpy as np
import statsmodels.formula.api as smf

from common import load_hsb, load_radon
from lmm import LMM

warnings.filterwarnings("ignore")

r = load_radon()
own = LMM().fit(np.column_stack([np.ones(len(r)), r.floor]), np.ones(len(r)), r.y, r.county)
ref = smf.mixedlm("y ~ floor", r, groups="county").fit(reml=True)
assert np.allclose(own.beta, ref.fe_params.values, atol=1e-4)
assert abs(own.D[0, 0] - ref.cov_re.values[0, 0]) < 1e-4
print("radon random intercept: OK")

h = load_hsb()
h["mc"], h["cc"] = h.meanses * h.cses, h.catholic * h.cses
X = np.column_stack([np.ones(len(h)), h.meanses, h.catholic, h.cses, h.mc, h.cc])
own = LMM().fit(X, np.column_stack([np.ones(len(h)), h.cses]), h.mAch, h.school)
ref = smf.mixedlm("mAch ~ meanses + catholic + cses + mc + cc", h, groups="school",
                  re_formula="~cses").fit(reml=True)
assert np.allclose(own.beta, ref.fe_params.values, atol=1e-3)
assert np.allclose(own.D, ref.cov_re.values, atol=1e-3)
print("HSB random intercept + slope: OK")
