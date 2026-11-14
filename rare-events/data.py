"""The two rare-event data sets and their design matrices.

  default   10,000 credit-card customers, of whom 333 defaulted (ISLR::Default;
            James, Witten, Hastie and Tibshirani 2021). Regressors: balance,
            income in thousands of dollars, student.
  caravan   5,822 customers of an insurance company, of whom 348 bought a
            caravan insurance policy (ISLR::Caravan, from the CoIL 2000
            challenge; van der Putten and van Someren 2004). Regressors: the 85
            socio-demographic and product-ownership variables.
"""

import numpy as np

from common import fetch


def default():
    d = fetch("ISLR", "Default")
    y = (d.default == "Yes").to_numpy(int)
    X = np.column_stack([np.ones(len(d)), d.balance, d.income / 1000, d.student == "Yes"]).astype(float)
    return y, X, ["(Intercept)", "balance", "income (thousands)", "student[Yes]"]


def caravan():
    d = fetch("ISLR", "Caravan")
    y = (d.Purchase == "Yes").to_numpy(int)
    feats = [c for c in d.columns if c not in ("rownames", "Purchase")]
    X = np.column_stack([np.ones(len(d)), d[feats].to_numpy(float)])
    return y, X, ["(Intercept)"] + feats
