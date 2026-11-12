"""Case 1 - which variables identify the correction: married women's wages (Mroz 1987).

sampleSelection::Mroz87: 753 married women in the Panel Study of Income
Dynamics, 428 of whom worked for pay in 1975. The wage equation is Wooldridge's
textbook one, log wage on education, experience and experience squared. The
selection (participation) equation adds four variables that are excluded from
the wage equation: other household income (nwifeinc), age, the number of
children under six (kids5) and of children aged six to eighteen (kids618).

The script fits the wage equation by least squares on the workers and by
Heckman's two-step and maximum-likelihood estimators under five choices of
exclusion restriction (all four variables, the children only, nwifeinc only,
age only, and none, where the selection equation has exactly the wage-equation
regressors and identification rests on the normal distribution alone). For
each it records the return to education, rho, the strength of the excluded
variables in the participation probit, the collinearity of the inverse Mills
ratio with the wage regressors, and the implied mean log offered wage of the
325 women who did not work. It then tests each excluded variable by putting
it into the wage equation while the others identify the model, profiles the
log-likelihood over rho, and computes Manski's worst-case bounds on the median
offered wage.

Run:  python 01_mroz_exclusions.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import log_ndtr

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, VIOLET, fetch, style, write
from sel import heckit, heckman_ml, imr_diagnostics, manski_quantile_bounds, mills, probit

warnings.filterwarnings("ignore")
style()
m = fetch("sampleSelection", "Mroz87")
d = m.lfp.values.astype(bool)
n, n1 = len(m), int(d.sum())
one = np.ones(n)
X = np.column_stack([one, m.educ, m.exper, m.exper ** 2])
y = np.log(np.where(d, m.wage, 1.0))
CORE = [m.educ, m.exper, m.exper ** 2]
EXCL = {"nwifeinc": m.nwifeinc, "age": m.age, "kids5": m.kids5, "kids618": m.kids618}
SETS = {"All four (Wooldridge)": ["nwifeinc", "age", "kids5", "kids618"],
        "Children only": ["kids5", "kids618"],
        "Other income only": ["nwifeinc"],
        "Age only": ["age"],
        "None (normality only)": []}
out = [f"Mroz87: {n} women, {n1} worked ({n1 / n:.3f}); mean log wage of workers {y[d].mean():.4f}, SD {y[d].std(ddof=1):.4f}"]

# least squares on the workers
b_ols, *_ = np.linalg.lstsq(X[d], y[d], rcond=None)
e = y[d] - X[d] @ b_ols
s2 = e @ e / (n1 - X.shape[1])
se_ols = np.sqrt(np.diag(s2 * np.linalg.inv(X[d].T @ X[d])))
out.append(f"OLS on workers: educ {b_ols[1]:.4f} (SE {se_ols[1]:.4f}); exper {b_ols[2]:.4f}; exper^2 {b_ols[3]:.6f}")
pred_ols_nonworkers = (X[~d] @ b_ols).mean()
out.append(f"OLS prediction of the mean log wage of the {n - n1} non-workers: {pred_ols_nonworkers:.4f} "
           f"(workers' observed mean {y[d].mean():.4f})")

results = {}
for lab, ex in SETS.items():
    Z = np.column_stack([one] + CORE + [EXCL[v] for v in ex])
    h = heckit(Z, d, X, y)
    ml = heckman_ml(Z, d, X, y)
    r2, cond, pr = imr_diagnostics(Z, d, X)
    # strength of the exclusions: likelihood-ratio test of the excluded variables in the probit
    if ex:
        p0 = probit(np.column_stack([one] + CORE), d)
        lr = 2 * (pr.loglik - p0.loglik)
        p_lr = stats.chi2.sf(lr, len(ex))
        strength = f"LR {lr:.2f} on {len(ex)} df (p {p_lr:.2g})"
    else:
        lr, strength = np.nan, "none"
    # implied mean log offered wage of the non-workers: x'b + rho sigma E[v | v < -z'g] = x'b - rho sigma lambda(-z'g)
    zg = ml.gamma @ Z[~d].T
    off_ml = (X[~d] @ ml.beta - ml.rho * ml.sigma * mills(-zg)).mean()
    zg2 = h.probit.index[~d]
    off_2s = (X[~d] @ h.beta - h.beta_lambda * mills(-zg2)).mean()
    kz, kx = Z.shape[1], X.shape[1]
    se_ml = ml.se()
    results[lab] = dict(h=h, ml=ml, r2=r2, cond=cond, lr=lr, off_ml=off_ml, off_2s=off_2s,
                        se_ml_educ=se_ml[kz + 1], se_ml_rho=se_ml[-1])
    out.append(f"-- exclusions: {lab} [{', '.join(ex) if ex else 'none'}]; strength in the probit: {strength}")
    out.append(f"   inverse Mills ratio among workers: R^2 on the wage regressors {r2:.4f}; condition number of the "
               f"standardised design with the ratio {cond:.1f}")
    out.append(f"   two-step: educ {h.beta[1]:.4f} (SE {h.se()[1]:.4f}); lambda {h.beta_lambda:.4f} (SE {h.se()[-1]:.4f}); "
               f"rho {h.rho:.4f}; sigma {h.sigma:.4f}")
    out.append(f"   ML: educ {ml.beta[1]:.4f} (SE {se_ml[kz + 1]:.4f}); rho {ml.rho:.4f} (SE {se_ml[-1]:.4f}); "
               f"sigma {ml.sigma:.4f}; log-likelihood {ml.loglik:.3f}; converged {ml.converged}")
    out.append(f"   implied mean log offered wage of non-workers: two-step {off_2s:.4f}, ML {off_ml:.4f} "
               f"(gap to workers' mean {off_ml - y[d].mean():+.4f})")
_se = {k: v["h"].se()[1] for k, v in results.items()}
out.append(f"Two-step SE of educ with nothing excluded / with all four excluded: {_se['None (normality only)']:.4f} / "
           f"{_se['All four (Wooldridge)']:.4f} = {_se['None (normality only)'] / _se['All four (Wooldridge)']:.2f}")

# testing each exclusion: put it in the wage equation, identify with the others
out.append("Each excluded variable added to the wage equation, the other three identifying the model (ML):")
tests = {}
for v in EXCL:
    others = [w for w in EXCL if w != v]
    Z = np.column_stack([one] + CORE + [EXCL[w] for w in EXCL])
    Xv = np.column_stack([X, EXCL[v]])
    ml = heckman_ml(Z, d, Xv, y)
    kz = Z.shape[1]
    b, s = ml.beta[-1], ml.se()[kz + Xv.shape[1] - 1]
    tests[v] = (b, s)
    out.append(f"   {v:8s} coefficient in the wage equation {b:+.4f} (SE {s:.4f}, z {b / s:+.2f}); educ {ml.beta[1]:.4f}; "
               f"rho {ml.rho:.4f}")

# profile likelihood in rho
grid = np.linspace(-0.95, 0.95, 39)
prof = {}
for lab in ("All four (Wooldridge)", "None (normality only)"):
    Z = np.column_stack([one] + CORE + [EXCL[v] for v in SETS[lab]])
    base = results[lab]["ml"]
    st = base.theta[:-1]
    ll, ed = [], []
    for r in grid:
        f = heckman_ml(Z, d, X, y, start=st, rho_fixed=r)
        ll.append(f.loglik)
        ed.append(f.beta[1])
        st = f.theta
    ll = np.array(ll)
    prof[lab] = (ll - base.loglik, np.array(ed))
    inside = grid[(base.loglik - ll) <= stats.chi2.ppf(0.95, 1) / 2]
    out.append(f"Profile likelihood in rho, {lab}: log-likelihood range over rho in [-0.95, 0.95] "
               f"{ll.max() - ll.min():.3f}; 95% likelihood-ratio interval for rho covers [{inside.min():.2f}, "
               f"{inside.max():.2f}] on the grid; educ ranges {np.min(ed):.4f} to {np.max(ed):.4f}")
    peaks = [i for i in range(1, len(ll) - 1) if ll[i] >= ll[i - 1] and ll[i] >= ll[i + 1]]
    out.append(f"   local maxima of the profile on the grid: " + "; ".join(
        f"rho {grid[i]:.2f} (log-likelihood {ll[i] - base.loglik:+.3f} from the maximum, educ {ed[i]:.4f})" for i in peaks))

# Manski bounds on the median offered (log) wage of all 753 women
lo, hi, lo_p, hi_p = manski_quantile_bounds(y[d], n1 / n, 0.5)
out.append(f"Manski worst-case bounds on the median log offered wage of all women (observed share {n1 / n:.3f}): "
           f"[{lo:.4f}, {hi:.4f}] = workers' quantiles {lo_p:.3f} and {hi_p:.3f}; in dollars [{np.exp(lo):.2f}, "
           f"{np.exp(hi):.2f}]; workers' median {np.median(y[d]):.4f} ({np.exp(np.median(y[d])):.2f} dollars)")
write("01_mroz_exclusions.txt", "\n".join(out))

# --------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.45})
ax = axes[0]
labels = ["OLS on workers"] + list(SETS)
rows = [(b_ols[1], se_ols[1], None, None)] + [(r["h"].beta[1], r["h"].se()[1], r["ml"].beta[1], r["se_ml_educ"])
                                              for r in results.values()]
for i, (b2, s2_, bm, sm_) in enumerate(rows):
    ax.errorbar(b2, i - 0.12, xerr=1.96 * s2_, fmt="o", color=INK2 if i == 0 else BLUE, capsize=2, ms=5,
                label="Two-step" if i == 1 else ("Least squares" if i == 0 else None))
    if bm is not None:
        ax.errorbar(bm, i + 0.12, xerr=1.96 * sm_, fmt="D", color=ORANGE, capsize=2, ms=4,
                    label="Maximum likelihood" if i == 1 else None)
ax.set_yticks(range(len(labels)), ["Least squares,\nworkers only", "Excluded: all four\n(Wooldridge's choice)",
                                   "Excluded: children\nonly", "Excluded: other\nincome only", "Excluded: age only",
                                   "Excluded: none\n(normality only)"], fontsize=9)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.invert_yaxis()
ax.axvline(b_ols[1], color=INK2, lw=0.8, ls=":")
ax.set_xlabel("Return to a year of education (log points), ±1.96 SE")
ax.set_title("Same data, five exclusion restrictions")
ax.legend(fontsize=8.5, loc="center left")
ax = axes[1]
for lab, col, ls in (("All four (Wooldridge)", BLUE, "-"), ("None (normality only)", ORANGE, "--")):
    ax.plot(grid, prof[lab][0], color=col, ls=ls, lw=2,
            label="Four excluded variables" if lab.startswith("All") else "No excluded variable")
ax.axhline(-stats.chi2.ppf(0.95, 1) / 2, color=INK2, lw=0.8, ls=":")
ax.text(0.97, -stats.chi2.ppf(0.95, 1) / 2 - 0.15, "95% likelihood-ratio cut-off", fontsize=8.5, color=INK2,
        ha="right", va="top", transform=ax.get_yaxis_transform())
ax.set_xlabel("ρ, correlation of the participation and wage errors (held fixed)")
ax.set_ylabel("Log-likelihood minus its maximum")
ax.set_title("How much the data say about ρ")
ax.set_ylim(-8, 0.5)
ax.legend(fontsize=8.5, loc="lower center")
fig.savefig(FIG / "fig1_mroz_exclusions.png"); plt.close(fig)
