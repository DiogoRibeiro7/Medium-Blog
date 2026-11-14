"""Checks for rare.py.

1. Logistic regression on ISLR::Default equals Tables 4.1 and 4.3 of James et al.
   (2021, An Introduction to Statistical Learning, 2nd edition) and statsmodels.
2. On ISLR::Caravan, logistic regression with the 85 regressors, trained on all
   but the first 1,000 customers, reproduces the two confusion tables of the
   book's Caravan lab (cutoffs 0.5 and 0.25).
3. Weighted logistic regression equals statsmodels' GLM with frequency weights;
   the ridge-penalised fit equals scikit-learn's LogisticRegression (L2, intercept
   not penalised); calibration intercept and slope equal statsmodels' GLM with an
   offset.
4. AUC, Brier score and log loss equal scikit-learn; SMOTE equals
   imbalanced-learn's SMOTE given the same seed.
5. Case-control theory (Prentice and Pyke 1979): on a large simulated sample,
   undersampling the non-events leaves the slopes unchanged and shifts the
   intercept by log(1 / keep rate); prior correction undoes the shift.

Run:  python check_rare.py   (output also written to results/00_checks.txt)
"""

import warnings

import imblearn
import numpy as np
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from common import RES
from data import caravan, default
from rare import auc, balanced_weights, brier, calibration, confusion, logistic, logloss, prior_correct, smote, undersample

warnings.filterwarnings("ignore")
LINES = []


def say(s):
    print(s)
    LINES.append(s)


def fmt(a, nd=6):
    return "[" + ", ".join(f"{v:.{nd}f}" for v in np.asarray(a)) + "]"


# 1. ISLR Tables 4.1 and 4.3, statsmodels
y, X, names = default()
f1 = logistic(X[:, :2], y)
assert np.allclose(np.round(f1.coef, 4), [-10.6513, 0.0055]) and np.allclose(np.round(f1.se(), 4), [0.3612, 0.0002])
f3 = logistic(X, y)
assert np.allclose(np.round(f3.coef, 4), [-10.8690, 0.0057, 0.0030, -0.6468])
assert np.allclose(f3.se(), [0.4923, 0.0002, 0.0082, 0.2362], atol=1e-4)   # the book prints 0.2362 for 0.236257
r = sm.Logit(y, X).fit(disp=0)
assert np.allclose(f3.coef, r.params, atol=1e-8) and np.allclose(f3.se(), r.bse, rtol=1e-6) and abs(f3.loglik - r.llf) < 1e-8
say(f"Default ({len(y)} customers, {y.sum()} defaults): balance only {fmt(f1.coef, 4)} (SEs {fmt(f1.se(), 4)}); balance, "
    f"income, student {fmt(f3.coef, 4)} (SEs {fmt(f3.se(), 6)}) equal Tables 4.1 and 4.3 of ISLR (2nd ed.; coefficients to "
    f"four decimals, standard errors within 1e-4) and statsmodels: OK")

# 2. ISLR Caravan lab
yc, Xc, _ = caravan()
test = np.arange(len(yc)) < 1000
fc = logistic(Xc[~test], yc[~test])
pc = fc.predict(Xc[test])
t5, t25 = confusion(yc[test], pc, 0.5), confusion(yc[test], pc, 0.25)
# book's tables, rows = prediction (No, Yes), columns = truth (No, Yes): 0.5 -> 934 59 / 7 0; 0.25 -> 919 48 / 22 11
assert (t5[3], t5[2], t5[1], t5[0]) == (934, 59, 7, 0) and (t25[3], t25[2], t25[1], t25[0]) == (919, 48, 22, 11)
r = sm.Logit(yc[~test], Xc[~test]).fit(disp=0, maxiter=200, method="newton")
assert np.allclose(fc.predict(Xc[test]), r.predict(Xc[test]), atol=1e-6)
say(f"Caravan ({len(yc)} customers, {yc.sum()} buyers): trained on customers 1,001-5,822, at cutoff 0.5 the test set has "
    f"{t5[0] + t5[1]} predicted buyers ({t5[0]} correct); at cutoff 0.25, {t25[0] + t25[1]} ({t25[0]} correct); equal "
    f"the ISLR Caravan lab (934/59/7/0 and 919/48/22/11) and statsmodels: OK")

# 3. weights, ridge, calibration
w = balanced_weights(y)
fw = logistic(X, y, w=w)
r = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit()
assert np.allclose(fw.coef, r.params, atol=1e-7)
Xs = X.copy()
Xs[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
fr = logistic(Xs, y, ridge=2.0)
sk = LogisticRegression(C=0.5, l1_ratio=0, solver="lbfgs", tol=1e-12, max_iter=10000).fit(Xs[:, 1:], y)
assert np.allclose(fr.coef, np.r_[sk.intercept_, sk.coef_[0]], atol=1e-6), (fr.coef, sk.intercept_, sk.coef_)
p = f3.predict(X)
a, s = calibration(y, p)
lp = np.log(p / (1 - p))
ra = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
rs = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
assert abs(a - ra.params[0]) < 1e-8 and abs(s - rs.params[1]) < 1e-8
say(f"weighted logistic regression (balanced weights) equals statsmodels GLM with frequency weights; ridge logistic "
    f"(penalty 2 on standardised regressors) equals scikit-learn LogisticRegression(C = 0.5); calibration intercept "
    f"{a:+.2e} and slope {s:.6f} of the in-sample fit equal statsmodels: OK")

# 4. metrics and SMOTE
rng = np.random.default_rng(0)
s_ = rng.normal(size=5000)
yy = (rng.random(5000) < 1 / (1 + np.exp(-(s_ - 2)))).astype(int)
pp = 1 / (1 + np.exp(-(s_ - 2)))
assert abs(auc(yy, s_) - roc_auc_score(yy, s_)) < 1e-12 and abs(auc(yy, np.round(s_, 1)) - roc_auc_score(yy, np.round(s_, 1))) < 1e-12
assert abs(brier(yy, pp) - brier_score_loss(yy, pp)) < 1e-12 and abs(logloss(yy, pp) - log_loss(yy, pp)) < 1e-12
Z = X[:, 1:3]
Xa, ya = smote(Z, y, np.random.RandomState(42), k=5)
Xb, yb = SMOTE(random_state=42, k_neighbors=5).fit_resample(Z, y)
assert Xa.shape == Xb.shape and np.allclose(Xa, Xb) and np.array_equal(ya, yb)
say(f"AUC (with and without ties), Brier score and log loss equal scikit-learn; SMOTE on (balance, income) equals "
    f"imbalanced-learn {imblearn.__version__} with the same seed ({len(ya) - len(y)} synthetic defaults): OK")

# 5. case-control theory
n = 1_000_000
x = rng.normal(size=(n, 2))
eta = -5 + x @ [1.0, -0.5]
yt = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
Xt = np.column_stack([np.ones(n), x])
full = logistic(Xt, yt)
Xu, yu = undersample(Xt, yt, rng)
keep = yt.sum() / (n - yt.sum())
und = logistic(Xu, yu)
shift = und.coef[0] - full.coef[0]
pc_ = prior_correct(und.predict(Xt), 0.5, yt.mean())
assert np.allclose(und.coef[1:], full.coef[1:], atol=4 * und.se()[1:].max())
assert abs(shift - np.log(1 / keep)) < 4 * und.se()[0]
assert abs(pc_.mean() - yt.mean()) < 0.1 * yt.mean()
say(f"case-control check (n = 1,000,000, event rate {yt.mean():.4f}): full-data slopes {fmt(full.coef[1:], 4)}, after "
    f"undersampling to 50% {fmt(und.coef[1:], 4)}; intercept shift {shift:.4f} against log(1 / keep rate) "
    f"{np.log(1 / keep):.4f}; prior-corrected mean probability {pc_.mean():.4f} (event rate {yt.mean():.4f}): OK")
(RES / "00_checks.txt").write_text("\n".join(LINES) + "\n")
