"""Checks for counts.py.

1. On the NMES 1988 data (Deb and Trivedi 1997; AER::NMES1988), the Poisson,
   negative binomial, hurdle and zero-inflated fits equal the output printed in
   the vignette of the R package pscl 1.5.9 ("Regression Models for Count Data
   in R", Zeileis, Kleiber and Jackman 2008, rerun by the package with R 4.4.0):
   the Poisson coefficients, model-based and sandwich standard errors to six
   decimals; the full hurdle model's coefficients, standard errors and theta to
   six decimals; Table 2 (coefficients and standard errors to three decimals,
   log-likelihoods, AIC, BIC and the expected number of zeros).
2. Poisson, NB2, zero-inflated Poisson and NB2 fits equal statsmodels on the
   bioChemists data (Long 1990; pscl::bioChemists); the heterogeneous-dispersion
   NB reduces to the NB when the dispersion has only an intercept.
3. The analytic gradients of every likelihood equal numerical derivatives.
4. On a large simulated sample each model recovers its true parameters.

Run:  python check_counts.py   (output also written to results/00_checks.txt)
"""

import warnings

import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP

from common import RES
from counts import hurdle, negbin, poisson, zeroinfl
from data import biochemists, nmes

warnings.filterwarnings("ignore")
LINES = []


def fmt(a, nd=6):
    return "[" + ", ".join(f"{v:.{nd}f}" for v in np.asarray(a)) + "]"


def say(s):
    print(s)
    LINES.append(s)


def close(a, b, tol):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=0)


# ------------------------------------------------------------------ 1. pscl vignette, NMES 1988
y, X, Z, _ = nmes()
fp = poisson(X, y)
assert close(fp.params, [1.028874, 0.164797, 0.248307, -0.361993, 0.146639, -0.112320, 0.026143, 0.201687], 6e-7)
assert close(fp.se(), [0.023785, 0.005997, 0.017845, 0.030304, 0.004580, 0.012945, 0.001843, 0.016860], 6e-7)
rse = np.sqrt(np.diag(fp.extra["cov_robust"]))
assert close(rse, [0.064530, 0.021945, 0.054022, 0.077449, 0.012908, 0.035343, 0.005084, 0.043128], 1.1e-6)
assert round(fp.aic) == 35959
say(f"Poisson on NMES: coefficients {fmt(fp.params)}, SEs {fmt(fp.se())}, sandwich SEs {fmt(rse)}, AIC {fp.aic:.1f} "
    f"equal the pscl vignette: OK")

fh0 = hurdle(X, y)
c_ref = [1.197699, 0.211898, 0.315958, -0.331861, 0.126421, -0.068317, 0.020693, 0.100172]
c_se = [0.058973, 0.021396, 0.048056, 0.066093, 0.012452, 0.032416, 0.004535, 0.042619]
z_ref = [0.043147, 0.312449, -0.008716, -0.289570, 0.535213, -0.415658, 0.058541, 0.747120]
z_se = [0.139852, 0.091437, 0.161024, 0.142682, 0.045378, 0.087608, 0.011989, 0.100880]
assert close(fh0.block("count"), c_ref, 2e-6) and close(fh0.se()[fh0.blocks["count"]], c_se, 2e-6)
assert close(fh0.block("zero"), z_ref, 2e-6) and close(fh0.se()[fh0.blocks["zero"]], z_se, 2e-6)
assert abs(fh0.block("dispersion")[0] - 0.333255) < 2e-6 and abs(fh0.se()[-1] - 0.042754) < 2e-6
assert abs(np.exp(fh0.block("dispersion")[0]) - 1.3955) < 6e-5 and round(fh0.loglik / 1e4, 3) == -1.209
diff = np.max(np.abs(np.r_[fh0.block("count") - c_ref, fh0.se()[fh0.blocks["count"]] - c_se, fh0.block("zero") - z_ref,
                           fh0.se()[fh0.blocks["zero"]] - z_se, fh0.block("dispersion")[0] - 0.333255,
                           fh0.se()[-1] - 0.042754]))
say(f"Hurdle NB, all regressors in both parts: count {fmt(fh0.block('count'))}, zero {fmt(fh0.block('zero'))}, "
    f"log(theta) {fh0.block('dispersion')[0]:.6f} (SE {fh0.se()[-1]:.6f}), theta {np.exp(fh0.block('dispersion')[0]):.4f}; "
    f"equal the pscl vignette (largest difference in coefficients, standard errors and log(theta) {diff:.1e}): OK")

fn = negbin(X, y)
fh = hurdle(X, y, Z)
fz = zeroinfl(X, y, Z)
tab = {
    "NB": (fn, [0.929, 0.218, 0.305, -0.342, 0.175, -0.126, 0.027, 0.224], [0.055, 0.020, 0.049, 0.061, 0.012, 0.031, 0.004, 0.039],
           None, None, -12170.6, 24359.1, 24416.6, 608),
    "Hurdle-NB": (fh, [1.198, 0.212, 0.316, -0.332, 0.126, -0.068, 0.021, 0.100], [0.059, 0.021, 0.048, 0.066, 0.012, 0.032, 0.005, 0.043],
                  [0.016, 0.318, 0.548, 0.746, 0.057, -0.419], [0.138, 0.091, 0.044, 0.100, 0.012, 0.088],
                  -12090.1, 24210.1, 24306.0, 683),
    "ZINB": (fz, [1.194, 0.201, 0.285, -0.319, 0.129, -0.080, 0.021, 0.126], [0.057, 0.020, 0.045, 0.060, 0.012, 0.031, 0.004, 0.042],
             [-0.047, -0.800, -1.248, -1.176, -0.084, 0.648], [0.269, 0.421, 0.178, 0.220, 0.026, 0.200],
             -12090.7, 24211.4, 24307.3, 709),
}
for lab, (f, c, cs, zc, zs, ll, aic, bic, zeros) in tab.items():
    se = f.se()
    if lab == "NB":
        se = np.r_[np.sqrt(np.diag(f.extra["cov_expected"])), se[-1]]
    assert close(f.block("count"), c, 6e-4) and close(se[f.blocks["count"]], cs, 6e-4), lab
    if zc is not None:
        assert close(f.block("zero"), zc, 6e-4) and close(se[f.blocks["zero"]], zs, 6e-4), lab
    assert abs(f.loglik - ll) < 0.06 and abs(f.aic - aic) < 0.06 and abs(f.bic(len(y)) - bic) < 0.06, lab
    assert round(f.pmf(0).sum()) == zeros, lab
    say(f"{lab} on NMES: count {fmt(f.block('count'), 3)}" + (f", zero {fmt(f.block('zero'), 3)}" if zc else "") +
        f", log-likelihood {f.loglik:.1f}, AIC {f.aic:.1f}, BIC {f.bic(len(y)):.1f}, expected zeros {f.pmf(0).sum():.1f} "
        f"equal Table 2 of the pscl vignette: OK")
assert round(fp.loglik, 1) == -17971.6 and round(fp.pmf(0).sum()) == 47 and int((y == 0).sum()) == 683
say(f"Poisson log-likelihood {fp.loglik:.1f}, expected zeros {fp.pmf(0).sum():.1f}; observed zeros {int((y == 0).sum())} of "
    f"{len(y)}: OK")

# ------------------------------------------------------------------ 2. statsmodels, bioChemists
yb, Xb = biochemists()
r = sm.Poisson(yb, Xb).fit(disp=0)
pb = poisson(Xb, yb)
assert close(pb.params, r.params, 1e-8) and abs(pb.loglik - r.llf) < 1e-8
r = sm.NegativeBinomial(yb, Xb, loglike_method="nb2").fit(disp=0, method="newton", maxiter=200)
nb = negbin(Xb, yb)
assert close(nb.block("count"), r.params[:-1], 1e-6) and abs(np.exp(-nb.block("dispersion")[0]) - r.params[-1]) < 1e-6
assert abs(nb.loglik - r.llf) < 1e-7
nbw = negbin(Xb, yb, np.ones((len(yb), 1)))
assert abs(nbw.loglik - nb.loglik) < 1e-8
r = ZeroInflatedPoisson(yb, Xb, exog_infl=Xb, inflation="logit").fit(disp=0, method="bfgs", maxiter=5000, gtol=1e-9)
zp = zeroinfl(Xb, yb, Xb, dist="poisson")
assert abs(zp.loglik - r.llf) < 1e-5, (zp.loglik, r.llf)
k = Xb.shape[1]
assert close(zp.block("count"), r.params[k:], 2e-4) and close(zp.block("zero"), r.params[:k], 2e-4)
r = ZeroInflatedNegativeBinomialP(yb, Xb, exog_infl=Xb, inflation="logit", p=2).fit(disp=0, method="bfgs", maxiter=5000,
                                                                                     gtol=1e-9)
zn = zeroinfl(Xb, yb, Xb)
assert abs(zn.loglik - r.llf) < 1e-5, (zn.loglik, r.llf)
assert close(zn.block("count"), r.params[k:2 * k], 2e-4) and close(zn.block("zero"), r.params[:k], 2e-4)
say(f"bioChemists: Poisson ({pb.loglik:.4f}), NB ({nb.loglik:.4f}), ZIP ({zp.loglik:.4f}) and ZINB ({zn.loglik:.4f}) "
    f"log-likelihoods and coefficients equal statsmodels; NB with an intercept-only dispersion equation equals the NB: OK")
hb = hurdle(Xb, yb)
pos = yb > 0
tr = TruncatedLFNegativeBinomialP(yb[pos], Xb[pos], p=2)
r_def = tr.fit(disp=0)
r_good = tr.fit(disp=0, method="bfgs", maxiter=5000, start_params=np.r_[0.4, np.zeros(k - 1), 1.0])
lg = sm.Logit(pos.astype(float), Xb).fit(disp=0)
ll_trunc = hb.loglik - lg.llf
assert hb.converged and abs(ll_trunc - r_good.llf) < 1e-5 and close(hb.block("count"), r_good.params[:k], 2e-4)
assert abs(np.exp(hb.block("dispersion")[0]) - 1 / r_good.params[-1]) < 1e-3
say(f"bioChemists hurdle NB: log-likelihood {hb.loglik:.4f} = logit {lg.llf:.4f} (statsmodels) + zero-truncated NB "
    f"{ll_trunc:.4f}, which equals statsmodels' zero-truncated NB started from theta = 1 ({r_good.llf:.4f}, theta "
    f"{1 / r_good.params[-1]:.2f}); from statsmodels' default start the same model stops at {r_def.llf:.4f} with theta "
    f"{1 / r_def.params[-1]:.2f} (converged flag {bool(r_def.mle_retvals.get('converged'))}): OK")

# ------------------------------------------------------------------ 3. gradients
rng = np.random.default_rng(3)
n = 400
x1, z1 = rng.standard_normal(n), rng.standard_normal(n)
Xs = np.column_stack([np.ones(n), x1])
Zs = np.column_stack([np.ones(n), z1])
ys = rng.negative_binomial(1.5, 1.5 / (1.5 + np.exp(0.3 + 0.5 * x1))) * (rng.random(n) > 0.2)


def numgrad(f, p, h=1e-6):
    return np.array([(f(p + h * e)[0] - f(p - h * e)[0]) / (2 * h) for e in np.eye(len(p))])


from counts import negll_negbin, negll_truncated, negll_zeroinfl  # noqa: E402

worst = 0.0
pos = ys > 0
cases = [(negll_negbin(Xs, ys, np.ones((n, 1))), 3), (negll_negbin(Xs, ys, Zs), 4),
         (negll_zeroinfl(Xs, ys, Zs, False), 4), (negll_zeroinfl(Xs, ys, Zs, True), 5),
         (negll_truncated(Xs[pos], ys[pos], False), 2), (negll_truncated(Xs[pos], ys[pos], True), 3)]
for f, k in cases:
    p = rng.normal(0, 0.4, k)
    worst = max(worst, float(np.max(np.abs(numgrad(f, p) - f(p)[1]))))
assert worst < 1e-4, worst
say(f"analytic gradients of the NB, heterogeneous-dispersion NB, ZIP, ZINB and zero-truncated Poisson and NB "
    f"likelihoods equal numerical derivatives (largest difference {worst:.1e}): OK")

# ------------------------------------------------------------------ 4. recovery on a large sample
n = 100_000
x1, z1 = rng.standard_normal(n), rng.standard_normal(n)
Xs = np.column_stack([np.ones(n), x1])
Zs = np.column_stack([np.ones(n), z1])
mu = np.exp(0.5 + 0.5 * x1)
pi = 1 / (1 + np.exp(-(-1 + 1.0 * z1)))
yz = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)) * (rng.random(n) > pi)
f = zeroinfl(Xs, yz, Zs)
assert f.converged and np.all(np.abs(f.params - [0.5, 0.5, -1, 1, np.log(2)]) < 4 * f.se()), (f.params, f.se())
th = np.exp(0.5 - 1.0 * z1)
yw = rng.negative_binomial(th, th / (th + mu))
g = negbin(Xs, yw, Zs)
assert g.converged and np.all(np.abs(g.params - [0.5, 0.5, 0.5, -1.0]) < 4 * g.se()), (g.params, g.se())
say(f"large simulated samples (n = 100,000): ZINB {fmt(f.params, 3)} (truth [0.5, 0.5, -1, 1, {np.log(2):.3f}]); NB with "
    f"log(theta) linear in z {fmt(g.params, 3)} (truth [0.5, 0.5, 0.5, -1]); every estimate within four standard "
    f"errors of the truth: OK")
(RES / "00_checks.txt").write_text("\n".join(LINES) + "\n")
