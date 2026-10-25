"""Experiment 4 - Bangladesh contraception: conditional versus marginal effects.

* random-intercept logistic regression (adaptive Gauss-Hermite, 25 nodes)
* GEE with exchangeable working correlation (population-averaged)
* ordinary logistic regression, naive and cluster-robust SEs
* the GLMM's own population-averaged effect, by integrating over districts

Run:  python 04_contraception_glmm.py
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit

from common import BLUE, FIG, INK2, ORANGE, load_contraception, style, write
from glmm import fit

warnings.filterwarnings("ignore")
style()
c = load_contraception()
N, J = len(c), c.district.nunique()
names = ["const", "urban", "age", "ch1", "ch2", "ch3p"]
X = np.column_stack([np.ones(N), c.urban, c.age, c.ch1, c.ch2, c.ch3p])
y = c.use.values.astype(float)
nd = c.groupby("district").size()
out = [f"Contraception: N = {N} women, J = {J} districts (size median {nd.median():.0f}, "
       f"range {nd.min()}-{nd.max()}); overall use rate {y.mean():.3f}"]

g1 = fit(X, y, c.district, n_agq=1)
g = fit(X, y, c.district, n_agq=25)
out.append(f"GLMM Laplace vs AGQ-25: urban {g1.beta[1]:.4f} vs {g.beta[1]:.4f}; "
           f"district SD {g1.sigma:.4f} vs {g.sigma:.4f}; loglik {g1.loglik:.2f} vs {g.loglik:.2f}")

gee = sm.GEE(y, X, groups=c.district, family=sm.families.Binomial(),
             cov_struct=sm.cov_struct.Exchangeable()).fit()
glm = sm.GLM(y, X, family=sm.families.Binomial()).fit()
glm_cr = sm.GLM(y, X, family=sm.families.Binomial()).fit(
    cov_type="cluster", cov_kwds={"groups": c.district})
out.append(f"GEE exchangeable working correlation: {gee.cov_struct.dep_params:.3f}")
out.append("\nCoefficients (log-odds), SE in brackets")
for i, k in enumerate(names):
    out.append(f"  {k:6s} GLMM {g.beta[i]:7.3f} ({g.se[i]:.3f}) | GEE {gee.params[i]:7.3f} "
               f"({gee.bse[i]:.3f}) | logistic {glm.params[i]:7.3f} (naive {glm.bse[i]:.3f}, "
               f"cluster {glm_cr.bse[i]:.3f})")
c2 = 16 * np.sqrt(3) / (15 * np.pi)
att = 1 / np.sqrt(1 + c2 ** 2 * g.sigma ** 2)
out.append(f"\nZeger-Liang-Albert attenuation 1/sqrt(1 + c^2 s^2), c^2 = {c2**2:.3f}: "
           f"{att:.3f}; predicted marginal urban coef {g.beta[1]*att:.3f}; GEE {gee.params[1]:.3f}; "
           f"ratio GEE/GLMM across slopes: "
           + ", ".join(f"{k} {gee.params[i]/g.beta[i]:.3f}" for i, k in enumerate(names) if i))

# GLMM population-averaged quantities: integrate over b ~ N(0, s^2)
z, w = np.polynomial.hermite.hermgauss(40)
bnodes, bw = np.sqrt(2) * g.sigma * z, w / np.sqrt(np.pi)


def pa_prob(Xm):
    return (expit((Xm @ g.beta)[:, None] + bnodes[None, :]) * bw).sum(1)


X1, X0 = X.copy(), X.copy()
X1[:, 1], X0[:, 1] = 1, 0
p1_pa, p0_pa = pa_prob(X1).mean(), pa_prob(X0).mean()
p1_b0, p0_b0 = expit(X1 @ g.beta).mean(), expit(X0 @ g.beta).mean()
g1_, g0_ = expit(X1 @ gee.params).mean(), expit(X0 @ gee.params).mean()
out.append(f"Urban - rural difference in use probability, averaged over the sample's women:")
out.append(f"  GLMM, 'typical district' (b = 0): {p1_b0:.3f} - {p0_b0:.3f} = {p1_b0-p0_b0:.3f}")
out.append(f"  GLMM, averaged over districts:    {p1_pa:.3f} - {p0_pa:.3f} = {p1_pa-p0_pa:.3f}")
out.append(f"  GEE (marginal model):              {g1_:.3f} - {g0_:.3f} = {g1_-g0_:.3f}")
out.append(f"  observed use rate: urban {y[c.urban==1].mean():.3f}, rural {y[c.urban==0].mean():.3f}")
# a profile: rural, 3+ children, mean age (age is centred)
xr = np.array([[1, 0, 0, 0, 0, 1]], float)
eta_r = float((xr @ g.beta)[0])
lo, hi = expit(eta_r - 1.2816 * g.sigma), expit(eta_r + 1.2816 * g.sigma)
out.append(f"Rural woman, 3+ children, mean age: P(use) in a typical district "
           f"{expit(eta_r):.3f}; averaged over districts {pa_prob(xr)[0]:.3f}; "
           f"middle 80% of districts between {lo:.3f} and {hi:.3f}")
mo = dict(zip(g.labels, g.modes))
small = nd[nd <= 10].index
raw_eff = c.groupby("district").use.mean()
out.append(f"Districts with <= 10 women: {len(small)}; their raw use rates span "
           f"{raw_eff[small].min():.2f}-{raw_eff[small].max():.2f}; conditional modes span "
           f"{min(mo[k] for k in small):.2f} to {max(mo[k] for k in small):.2f} on the logit scale")

# why does plain logistic regression give a larger urban coefficient than GEE?
mu = c.groupby("district").urban.transform("mean").values
Xm = np.column_stack([np.ones(N), c.urban, mu, c.age, c.ch1, c.ch2, c.ch3p])
gm = fit(Xm, y, c.district, n_agq=25)
both = int((c.groupby("district").urban.nunique() == 2).sum())
out.append(f"\nMundlak GLMM (urban + district share urban): within-district urban "
           f"{gm.beta[1]:.3f} (SE {gm.se[1]:.3f}); contextual effect of district share "
           f"urban (between minus within) {gm.beta[2]:.3f} (SE {gm.se[2]:.3f}, z = "
           f"{gm.beta[2]/gm.se[2]:.2f}); between-district effect {gm.beta[1]+gm.beta[2]:.3f}; "
           f"{both} of {J} districts contain both urban and rural women")
write("04_contraception.txt", "\n".join(out))

# figure: conditional curves vs the population-averaged curve, two heterogeneity levels
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
eta = np.linspace(-4.5, 4.5, 300)
rng = np.random.default_rng(7)
for ax, s, title in ((axes[0], g.sigma, f"District SD as estimated ({g.sigma:.2f})"),
                     (axes[1], 2.0, "District SD = 2.0 (strong clustering)")):
    for bj in rng.normal(0, s, 25):
        ax.plot(eta, expit(eta + bj), color="#cfcec9", lw=0.8, zorder=1)
    zn, wn = np.polynomial.hermite.hermgauss(60)
    pa = (expit(eta[:, None] + np.sqrt(2) * s * zn[None, :]) * wn / np.sqrt(np.pi)).sum(1)
    ax.plot(eta, expit(eta), color=BLUE, lw=2.2, label="Typical district (b = 0)", zorder=3)
    ax.plot(eta, pa, color=ORANGE, lw=2.2, label="Average over districts", zorder=3)
    i0 = np.argmin(abs(eta))
    slope_m = (logit(pa[i0 + 1]) - logit(pa[i0 - 1])) / (eta[i0 + 1] - eta[i0 - 1])
    ax.text(-4.3, 0.92, f"log-odds slope of the average:\n{slope_m:.2f} x the conditional slope",
            color=INK2, fontsize=9, va="top")
    ax.set_title(title)
    ax.set_xlabel("Fixed-effect linear predictor x'β")
axes[0].set_ylabel("P(use)")
axes[0].legend(loc="lower right", fontsize=9)
fig.savefig(FIG / "fig6_conditional_vs_marginal.png"); plt.close(fig)
