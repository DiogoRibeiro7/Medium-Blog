"""Small, tested estimators shared by the billionaire-poverty follow-up series.

All functions take a data frame, an outcome column and a list of predictor
columns. Outcomes measured in percent are modelled on the 0-100 scale by OLS
and on the unit interval by the fractional logit.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

Z95 = 1.959963984540054


def ols_hc3(frame: pd.DataFrame, y: str, cols: Sequence[str]) -> dict[str, Any]:
    """OLS with HC3 standard errors; returns coefficients, p-values and intervals."""
    X = sm.add_constant(frame[list(cols)], has_constant="add")
    fit = sm.OLS(frame[y].to_numpy(dtype=float), X).fit(cov_type="HC3")
    ci = fit.conf_int()
    return {
        "n": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "coef": {k: float(v) for k, v in fit.params.items()},
        "p": {k: float(v) for k, v in fit.pvalues.items()},
        "ci95": {k: [float(ci.loc[k, 0]), float(ci.loc[k, 1])] for k in fit.params.index},
    }


def fractional_logit(frame: pd.DataFrame, y_pct: str, cols: Sequence[str], focus: str) -> dict[str, Any]:
    """Papke-Wooldridge fractional logit of a percentage with a sandwich covariance.

    Returns the average marginal effect of ``focus`` in percentage points per
    unit change, with a delta-method 95% interval.
    """
    X = sm.add_constant(frame[list(cols)], has_constant="add")
    share = frame[y_pct].to_numpy(dtype=float) / 100.0
    fit = sm.GLM(share, X, family=sm.families.Binomial()).fit(cov_type="HC0")
    beta, cov, Xm = fit.params.to_numpy(), fit.cov_params().to_numpy(), X.to_numpy(dtype=float)
    p = 1.0 / (1.0 + np.exp(-Xm @ beta))
    w = p * (1.0 - p)
    j = list(X.columns).index(focus)
    ame = 100.0 * float(w.mean()) * beta[j]
    grad = 100.0 * beta[j] * (w * (1.0 - 2.0 * p)) @ Xm / len(p)
    grad[j] += 100.0 * float(w.mean())
    se = float(math.sqrt(grad @ cov @ grad))
    return {"ame": float(ame), "se": se, "ci95": [float(ame - Z95 * se), float(ame + Z95 * se)],
            "coef_p": float(fit.pvalues.iloc[j])}


def loo_rmse_ols(frame: pd.DataFrame, y: str, cols: Sequence[str]) -> float:
    """Leave-one-out root-mean-square prediction error of OLS (PRESS)."""
    X = sm.add_constant(frame[list(cols)], has_constant="add").to_numpy(dtype=float) if cols else \
        np.ones((len(frame), 1))
    fit = sm.OLS(frame[y].to_numpy(dtype=float), X).fit()
    h = fit.get_influence().hat_matrix_diag
    return float(np.sqrt(np.mean((fit.resid / (1.0 - h)) ** 2)))


def loo_rmse_fractional(frame: pd.DataFrame, y_pct: str, cols: Sequence[str]) -> float:
    """Leave-one-out prediction error (percentage points) of the fractional logit."""
    errors = np.empty(len(frame))
    for i in range(len(frame)):
        train = frame.drop(index=frame.index[i])
        X = sm.add_constant(train[list(cols)], has_constant="add")
        fit = sm.GLM(train[y_pct] / 100.0, X, family=sm.families.Binomial()).fit()
        x = np.r_[1.0, frame.iloc[i][list(cols)].to_numpy(dtype=float)]
        errors[i] = frame[y_pct].iloc[i] - 100.0 / (1.0 + math.exp(-float(x @ fit.params.to_numpy())))
    return float(np.sqrt(np.mean(errors**2)))
