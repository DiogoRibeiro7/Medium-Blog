"""Checks for mi.py.

1. Rubin's rules: hand-computed example and the B = 0 limit.
2. MICE on simulated MAR data: pooled estimates unbiased, intervals cover.
3. Agreement with statsmodels' MICE on the NHANES regression (within Monte Carlo error).

Run:  python check_mi.py
"""

import warnings

import numpy as np
from scipy.special import expit

from mi import mice, ols, pool

warnings.filterwarnings("ignore")

# 1. Rubin's rules ------------------------------------------------------------
Q = np.array([[1.0], [1.2], [0.8], [1.1], [0.9]])
U = np.full((5, 1), 0.04)
p = pool(Q, U, df_complete=100)
b = Q.var(ddof=1)
assert np.isclose(p.est[0], 1.0) and np.isclose(p.b[0], b)
assert np.isclose(p.se[0] ** 2, 0.04 + 1.2 * b)
lam = 1.2 * b / (0.04 + 1.2 * b)
df_old = 4 / lam ** 2
df_obs = 101 / 103 * 100 * (1 - lam)
assert np.isclose(p.df[0], df_old * df_obs / (df_old + df_obs))
p0 = pool(np.ones((5, 1)), U, df_complete=100)
assert np.isclose(p0.se[0], 0.2) and np.isclose(p0.df[0], 101 / 103 * 100)
print("Rubin's rules: OK")

# 2. coverage on simulated MAR data -------------------------------------------
rng = np.random.default_rng(1)
cov = np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1]])
hits, ests = [], []
for r in range(200):
    Z = rng.multivariate_normal(np.zeros(3), cov, 500)
    y = 1 + 0.5 * Z[:, 0] - 0.3 * Z[:, 1] + 0.2 * Z[:, 2] + rng.normal(0, 1, 500)
    D = np.column_stack([y, Z])
    D[rng.random(500) < expit(-1.5 + 1.2 * y), 1] = np.nan       # MAR on y
    D[rng.random(500) < expit(-1.0 + 1.0 * Z[:, 2]), 2] = np.nan  # MAR on x3
    imps = mice(D, {1: "pmm", 2: "pmm"}, m=10, seed=r)
    E, V = zip(*(ols(np.column_stack([np.ones(500), Zc[:, 1:]]), Zc[:, 0]) for Zc in imps))
    P = pool(E, V, 496)
    lo, hi = P.ci()
    hits.append(lo[1] <= 0.5 <= hi[1]); ests.append(P.est[1])
cover, bias = np.mean(hits), np.mean(ests) - 0.5
print(f"MICE on MAR data: mean estimate {np.mean(ests):.4f} (true 0.5), coverage {cover:.3f}")
assert abs(bias) < 0.02 and 0.91 <= cover <= 0.99

# 3. statsmodels MICE on NHANES -----------------------------------------------
if __name__ == "__main__":
    import statsmodels.api as sm
    from statsmodels.imputation.mice import MICE, MICEData

    from common import load_nhanes

    d = load_nhanes()
    d["age2"] = (d.age - 50) ** 2 / 100
    keep = ["sbp", "age", "age2", "female", "bmi", "poverty", "chol", "black", "hispanic",
            "otherrace", "smoker", "dbp", "pulse", "hdl", "weight", "income"]
    d = d[keep]
    formula = ("sbp ~ age + age2 + female + bmi + poverty + chol + black + hispanic"
               " + otherrace + smoker")
    np.random.seed(5)
    res = MICE(formula, sm.OLS, MICEData(d)).fit(n_burnin=10, n_imputations=20)
    A = d.values.astype(float)
    meth = {i: ("logreg" if c == "smoker" else "pmm") for i, c in enumerate(keep)
            if d[c].isna().any()}
    imps = mice(A, meth, m=20, seed=5)
    xi = [keep.index(c) for c in keep[1:11]]
    E, V = zip(*(ols(np.column_stack([np.ones(len(Z)), Z[:, xi]]), Z[:, 0]) for Z in imps))
    P = pool(E, V, len(d) - 11)
    names = ["const"] + keep[1:11]
    print("NHANES regression, this implementation vs statsmodels MICE (m = 20 each):")
    for i, k in enumerate(names):
        diff = (P.est[i] - res.params[i]) / np.hypot(P.se[i], res.bse[i])
        print(f"  {k:9s} {P.est[i]:8.3f} vs {res.params[i]:8.3f}   SE {P.se[i]:.3f} vs "
              f"{res.bse[i]:.3f}   difference / combined SE {diff:+.2f}")
        assert abs(diff) < 1.0
    print("statsmodels agreement: OK")
