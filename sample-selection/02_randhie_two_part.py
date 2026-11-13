"""Case 2 - a published selection model with no exclusion restriction: medical spending in the RAND experiment.

sampleSelection::RandHIE, the RAND Health Insurance Experiment as used by
Cameron and Trivedi (2005, section 16.6): study year 2 with education known.
The outcome is the log of medical spending (lnmeddol), observed when spending
is positive (binexp); both equations have the same 17 regressors, so the
selection correction is identified by the normal distribution alone. The
coinsurance rate (logc = log(coinsurance + 1)) was randomised across families
by the experiment.

The script compares the two-part model's second part (least squares on the
people with positive spending, which estimates E[log spending | spending > 0])
with Heckman's two-step and ML estimators (which estimate a latent outcome
equation); measures how close the inverse Mills ratio is to a linear function
of the regressors; refits after adding squares of the continuous regressors;
bootstraps all estimators (persons resampled, whole procedure repeated, 300
replicates), with ML started from the two-step estimates and from the best
of four starts; and runs two placebos that keep the regressors and who spends
but replace log spending among spenders by the least-squares fit plus (a) the
least-squares residuals shuffled, which keeps their shape but removes any
link to selection, or (b) normal noise with the same SD (200 replicates each).

Run:  python 02_randhie_two_part.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, FIG, INK, INK2, ORANGE, AQUA, fetch, style, write
from multiprocessing import Pool

from scipy import stats

from sel import heckit, heckman_ml, heckman_ml_multistart, imr_diagnostics, mills

warnings.filterwarnings("ignore")
style()
r = fetch("sampleSelection", "RandHIE")
s = r[(r.year == 2) & r.educdec.notna()].reset_index(drop=True)
REG = ["logc", "idp", "lpi", "fmde", "physlm", "disea", "hlthg", "hlthf", "hlthp", "linc", "lfam", "educdec", "xage",
       "female", "child", "fchild", "black"]
CONT = ["lpi", "fmde", "physlm", "disea", "linc", "lfam", "educdec", "xage"]
n = len(s)
d = s.binexp.values.astype(bool)
y = np.where(d, s.lnmeddol.values, 0.0)
X = np.column_stack([np.ones(n)] + [s[v].values.astype(float) for v in REG])
NAMES = ["const"] + REG
j = NAMES.index("logc")
out = [f"RandHIE, year 2 with education known: {n} persons ({s.zper.nunique()} distinct ids), {d.sum()} with positive "
       f"spending ({d.mean():.3f}); mean log spending among them {y[d].mean():.4f}"]


def ols(Xm, yv):
    b = np.linalg.lstsq(Xm, yv, rcond=None)[0]
    e = yv - Xm @ b
    V = e @ e / (len(yv) - Xm.shape[1]) * np.linalg.inv(Xm.T @ Xm)
    return b, np.sqrt(np.diag(V))


def fit_all(Xm, label):
    b, se = ols(Xm[d], y[d])
    h = heckit(Xm, d, Xm, y)
    ml0 = heckman_ml(Xm, d, Xm, y)
    ml, nmax = heckman_ml_multistart(Xm, d, Xm, y)
    r2, cond, _ = imr_diagnostics(Xm, d, Xm)
    k = Xm.shape[1]
    sm = ml.se()
    lines = [f"-- {label}: {k} regressors in each equation (the intercept and {k - 1} variables)",
             f"   inverse Mills ratio among spenders: R^2 on the outcome regressors {r2:.4f}; condition number {cond:.1f}",
             f"   two-part (OLS on spenders): logc {b[j]:.4f} (SE {se[j]:.4f})",
             f"   two-step: logc {h.beta[j]:.4f} (SE {h.se()[j]:.4f}); lambda {h.beta_lambda:.4f} (SE {h.se()[-1]:.4f}); "
             f"rho {h.rho:.4f}; sigma {h.sigma:.4f}",
             f"   ML (best of four starts; distinct maxima found: {nmax}): logc {ml.beta[j]:.4f} (SE {sm[k + j]:.4f}, "
             f"z {ml.beta[j] / sm[k + j]:+.2f}); rho "
             f"{ml.rho:.4f} (SE {sm[-1]:.4f}); sigma {ml.sigma:.4f}; log-likelihood {ml.loglik:.3f}; converged {ml.converged}",
             f"   ML from the two-step start alone: rho {ml0.rho:.4f}; log-likelihood {ml0.loglik:.3f}"]
    # predicted mean log spending for the non-spenders under the latent model, and for everyone
    zg = h.probit.index
    lat = (Xm @ ml.beta).mean()
    lines.append(f"   ML latent mean log spending for all {n} (x'b) {lat:.4f}; observed mean among spenders "
                 f"{y[d].mean():.4f}")
    return dict(b=b, se=se, h=h, ml=ml, r2=r2, cond=cond), lines




def boot(seed):
    rng = np.random.default_rng([16, seed])
    t = rng.integers(0, n, n)
    Xb, db, yb = X[t], d[t], y[t]
    b2 = np.linalg.lstsq(Xb[db], yb[db], rcond=None)[0][j]
    hb = heckit(Xb, db, Xb, yb)
    m0 = heckman_ml(Xb, db, Xb, yb)
    m1, _ = heckman_ml_multistart(Xb, db, Xb, yb)
    return b2, hb.beta[j], hb.rho, m0.beta[j], m0.rho, m1.beta[j], m1.rho, m1.loglik - m0.loglik


def placebo(task):
    kind, seed = task
    rng = np.random.default_rng([17, kind == "normal", seed])
    yp = y.copy()
    noise = rng.permutation(E_OLS) if kind == "shuffled" else rng.normal(0, E_OLS.std(ddof=X.shape[1]), d.sum())
    yp[d] = X[d] @ B_OLS + noise
    m, _ = heckman_ml_multistart(X, d, X, yp)
    return m.rho, m.beta[j]


B_OLS = np.linalg.lstsq(X[d], y[d], rcond=None)[0]
E_OLS = y[d] - X[d] @ B_OLS

if __name__ == "__main__":
    base, l1 = fit_all(X, "Cameron and Trivedi's specification")
    out += l1
    out.append(f"   probit (two-step first stage): logc {base['h'].probit.coef[j]:.4f} (SE "
               f"{np.sqrt(base['h'].probit.cov[j, j]):.4f})")
    out.append(f"   least-squares residuals among spenders: skewness {stats.skew(E_OLS):.3f}, excess kurtosis "
               f"{stats.kurtosis(E_OLS):.3f}")
    Xf = np.column_stack([X] + [s[v].values.astype(float) ** 2 for v in CONT])
    flex, l2 = fit_all(Xf, "same, plus squares of the eight continuous regressors")
    out += l2
    # profile likelihood in rho
    st = base["ml"].theta[:-1]
    grid = np.round(np.linspace(-0.9, 0.9, 19), 2)
    prof = []
    for r0 in grid:
        f = heckman_ml(X, d, X, y, start=st, rho_fixed=r0)
        prof.append(f.loglik - base["ml"].loglik)
    out.append("Profile log-likelihood in rho (minus the maximum): " + "; ".join(
        f"{r0:+.1f} {v:.2f}" for r0, v in zip(grid, prof)))
    B = 300
    with Pool(2) as pool:
        bt = np.array(pool.map(boot, range(B), chunksize=10))
        pl = pool.map(placebo, [(k, i) for k in ("shuffled", "normal") for i in range(200)], chunksize=10)
    pl = np.array(pl)
    names = ["Two-part (OLS on spenders), logc", "Two-step, logc", "Two-step, rho", "ML from two-step start, logc",
             "ML from two-step start, rho", "ML best of four starts, logc", "ML best of four starts, rho"]
    out.append(f"Bootstrap over persons ({B} replicates):")
    for c, nm in enumerate(names):
        v = bt[:, c]
        out.append(f"   {nm:34s} mean {v.mean():.4f}; SD {v.std(ddof=1):.4f}; 2.5%-97.5% {np.quantile(v, .025):.4f} to "
                   f"{np.quantile(v, .975):.4f}")
    gain = bt[:, 7]
    out.append(f"   ML from the two-step start stopped at a lower maximum in {np.mean(gain > 1e-3):.3f} of replicates "
               f"(log-likelihood gap of the better start: median {np.median(gain[gain > 1e-3]):.2f} where it occurred); "
               f"two-step rho outside [-1, 1] in {np.mean(np.abs(bt[:, 2]) > 1):.3f}")
    low, hi_ = bt[gain > 1e-3, 4], bt[gain <= 1e-3, 4]
    out.append(f"   rho reported by ML from the two-step start alone: where it stopped at a lower maximum ({low.size} "
               f"replicates) median {np.median(low):.4f}, 2.5%-97.5% {np.quantile(low, .025):.4f} to "
               f"{np.quantile(low, .975):.4f}; elsewhere ({hi_.size}) median {np.median(hi_):.4f}, 2.5%-97.5% "
               f"{np.quantile(hi_, .025):.4f} to {np.quantile(hi_, .975):.4f}")
    for k, sl in (("shuffled least-squares residuals", slice(0, 200)), ("normal noise", slice(200, 400))):
        r_, b_ = pl[sl, 0], pl[sl, 1]
        out.append(f"Placebo with {k} (no selection by construction, 200 replicates): ML rho mean {r_.mean():.4f}, "
                   f"SD {r_.std(ddof=1):.4f}, "
                   f"2.5%-97.5% {np.quantile(r_, .025):.4f} to {np.quantile(r_, .975):.4f}; ML logc mean {b_.mean():.4f}, "
                   f"2.5%-97.5% {np.quantile(b_, .025):.4f} to {np.quantile(b_, .975):.4f} (OLS fit used to build the "
                   f"placebo: logc {B_OLS[j]:.4f})")
    write("02_randhie_two_part.txt", "\n".join(out))

    # --------------------------------------------------------------- figure 3
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), gridspec_kw={"wspace": 0.28})
    ml = base["ml"]
    binsr = np.linspace(-1.0, 1.0, 51)
    ax = axes[0]
    ax.hist(pl[200:, 0], bins=binsr, histtype="step", lw=2, color=INK2, density=True,
            label="Placebo: normal residuals")
    ax.hist(pl[:200, 0], bins=binsr, histtype="step", lw=2, color=ORANGE, density=True,
            label="Placebo: the real residuals, shuffled")
    ax.axvline(ml.rho, color=BLUE, lw=2)
    ax.annotate(f"estimate on the real data, {ml.rho:.2f}", (ml.rho, ax.get_ylim()[1] * 0.55), xytext=(0.5, ax.get_ylim()[1] * 0.55),
                ha="right", va="center", fontsize=8.5, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    ax.set_xlabel("ML estimate of ρ in data with no selection effect")
    ax.set_ylabel("Density over 200 placebo samples")
    ax.set_title("Where the estimate of ρ comes from")
    ax.legend(fontsize=8.5, loc="upper left")
    ax = axes[1]
    ax.hist(bt[:, 4], bins=binsr, histtype="step", lw=2, color=AQUA, density=True,
            label="ML started from the two-step estimates")
    ax.hist(bt[:, 6], bins=binsr, histtype="step", lw=2, color=ORANGE, density=True, label="ML, best of four starts")
    ax.hist(bt[:, 2], bins=binsr, histtype="step", lw=2, color=BLUE, density=True, label="Two-step")
    ax.set_xlabel("Estimate of ρ in bootstrap samples of the RAND data")
    ax.set_ylabel("Density over 300 bootstrap samples")
    ax.set_title("Two maxima for ML, little precision for two-step")
    ax.legend(fontsize=8.5, loc="upper left")
    for a in axes:
        a.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    fig.savefig(FIG / "fig3_randhie.png"); plt.close(fig)
