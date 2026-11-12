"""Decision support - what to report when no exclusion restriction is credible.

Mroz87 again. Two quantities that a selection correction is usually wanted for:
the return to education, and the wage the 325 women who did not work would
have been offered (for example, to correct a gender-gap or a labour-supply
calculation). Three ways of reporting them without leaning on an untested
exclusion restriction or on normality alone:

  1. a sensitivity curve: maximum likelihood with rho held fixed over a grid
     (the participation equation with all its regressors), reporting the
     return to education and the mean log offered wage of the non-workers
     minus the workers' mean, for each assumed rho;
  2. the same two quantities under the five exclusion restrictions of
     01_mroz_exclusions.py and under least squares (selection ignorable given
     education and experience);
  3. Manski's worst-case bounds on the median log offered wage of all 753
     women, which assume nothing about why women do not work, compared with
     the median implied by each model (median of the mixture of the
     individual normal offered-wage distributions).

Run:  python 05_mroz_sensitivity.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, VIOLET, YELLOW, fetch, style, write
from sel import heckman_ml, manski_quantile_bounds, mills

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
SETS = {"All four": ["nwifeinc", "age", "kids5", "kids618"], "Children only": ["kids5", "kids618"],
        "Other income only": ["nwifeinc"], "Age only": ["age"], "None": []}
ZALL = np.column_stack([one] + CORE + list(EXCL.values()))
ybar = y[d].mean()


def mixture_median(mu, sigma):
    f = lambda q: np.mean(stats.norm.cdf((q - mu) / sigma)) - 0.5
    return optimize.brentq(f, mu.min() - 5 * sigma, mu.max() + 5 * sigma)


def gap_nonworkers(ml, Z):
    zg = Z[~d] @ ml.gamma
    return (X[~d] @ ml.beta - ml.rho * ml.sigma * mills(-zg)).mean() - ybar


out = [f"Mroz87: {n} women, {n1} worked; workers' mean log wage {ybar:.4f}"]
# 1. sensitivity curve in rho
base = heckman_ml(ZALL, d, X, y)
grid = np.round(np.arange(-0.8, 0.81, 0.1), 2) + 0.0
st = base.theta[:-1]
sens = []
for r0 in grid:
    f = heckman_ml(ZALL, d, X, y, start=st, rho_fixed=r0)
    kz = ZALL.shape[1]
    se_e = f.se()[kz + 1]
    sens.append((r0, f.beta[1], se_e, gap_nonworkers(f, ZALL), f.loglik - base.loglik,
                 mixture_median(X @ f.beta, f.sigma)))
    st = f.theta
sens = np.array(sens)
out.append("Sensitivity to an assumed rho (ML, participation equation with all regressors): rho; return to education "
           "(SE); non-workers' mean log offered wage minus workers' mean; log-likelihood minus the maximum; implied median "
           "log offered wage of all women")
for r0, b, se, g, ll, med in sens:
    out.append(f"   rho {r0:+.1f}: educ {b:.4f} ({se:.4f}); gap {g:+.4f}; log-likelihood {ll:+.2f}; median {med:.4f}")
inside = sens[sens[:, 4] >= -stats.chi2.ppf(0.95, 1) / 2]
out.append(f"   rho values within the 95% likelihood-ratio region: {inside[:, 0].min():+.1f} to {inside[:, 0].max():+.1f}; "
           f"over them educ {inside[:, 1].min():.4f}-{inside[:, 1].max():.4f} and the gap {inside[:, 3].min():+.4f} to "
           f"{inside[:, 3].max():+.4f}")
out.append(f"   in percent (100 x (1 - exp(gap))): non-workers offered {100 * (1 - np.exp(inside[:, 3].max())):.1f}% to "
           f"{100 * (1 - np.exp(inside[:, 3].min())):.1f}% less than workers earn (geometric means)")
out.append(f"   over rho in [-0.5, 0.5]: educ {sens[(sens[:, 0] >= -0.5) & (sens[:, 0] <= 0.5), 1].min():.4f}-"
           f"{sens[(sens[:, 0] >= -0.5) & (sens[:, 0] <= 0.5), 1].max():.4f}; gap "
           f"{sens[(sens[:, 0] >= -0.5) & (sens[:, 0] <= 0.5), 3].min():+.4f} to "
           f"{sens[(sens[:, 0] >= -0.5) & (sens[:, 0] <= 0.5), 3].max():+.4f}")

# 2. the five exclusion restrictions and least squares
b_ols = np.linalg.lstsq(X[d], y[d], rcond=None)[0]
e = y[d] - X[d] @ b_ols
s_ols = np.sqrt(e @ e / (n1 - 4))
points = {"Least squares": (b_ols[1], (X[~d] @ b_ols).mean() - ybar, mixture_median(X @ b_ols, s_ols), np.nan)}
for lab, ex in SETS.items():
    Z = np.column_stack([one] + CORE + [EXCL[v] for v in ex])
    f = heckman_ml(Z, d, X, y)
    points[f"Heckman ML, excluded: {lab.lower()}"] = (f.beta[1], gap_nonworkers(f, Z), mixture_median(X @ f.beta, f.sigma),
                                                       f.rho)
out.append("Point estimates: return to education; non-workers' gap; implied median log offered wage of all women; rho")
for k, (b, g, med, rho) in points.items():
    out.append(f"   {k:40s} educ {b:.4f}; gap {g:+.4f}; median {med:.4f}; rho {rho:+.4f}")
out.append("Each point against the sensitivity curve at its own rho (ML, participation equation with all regressors, "
           "rho held at the point's estimate)")
for k, (b, g, med, rho) in points.items():
    if np.isnan(rho):
        continue
    f = heckman_ml(ZALL, d, X, y, start=base.theta[:-1], rho_fixed=rho)
    gc = gap_nonworkers(f, ZALL)
    out.append(f"   {k:40s} gap {g:+.4f}; curve {gc:+.4f}; point minus curve {g - gc:+.4f}")

# 3. Manski bounds on the median
lo, hi, lo_p, hi_p = manski_quantile_bounds(y[d], n1 / n, 0.5)
out.append(f"Manski worst-case bounds on the median log offered wage of all women: [{lo:.4f}, {hi:.4f}] (dollars "
           f"{np.exp(lo):.2f} to {np.exp(hi):.2f}); every model's implied median lies "
           f"{'inside' if all(lo <= v[2] <= hi for v in points.values()) else 'NOT all inside'} the bounds; the "
           f"implied medians range {min(v[2] for v in points.values()):.4f} to {max(v[2] for v in points.values()):.4f}")
write("05_mroz_sensitivity.txt", "\n".join(out))

# --------------------------------------------------------------- figure 6
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), gridspec_kw={"wspace": 0.3})
ax = axes[0]
ok = sens[:, 4] >= -stats.chi2.ppf(0.95, 1) / 2
ax.plot(sens[:, 0], sens[:, 3], color=INK2, lw=1.5, ls="--", label="Gap if ρ were known (ML, ρ held fixed)")
ax.plot(sens[ok, 0], sens[ok, 3], color=BLUE, lw=2.5, label="  same, ρ inside the 95% likelihood region")
ax.axhline(0, color=INK2, lw=0.8)
mk = {"Least squares": ("s", INK), "Heckman ML, excluded: all four": ("o", BLUE),
      "Heckman ML, excluded: children only": ("o", AQUA), "Heckman ML, excluded: other income only": ("o", VIOLET),
      "Heckman ML, excluded: age only": ("o", YELLOW), "Heckman ML, excluded: none": ("D", ORANGE)}
for k, (b, g, med, rho) in points.items():
    xr = 0.0 if np.isnan(rho) else rho
    ax.plot(xr, g, mk[k][0], color=mk[k][1], ms=7, mec="white", label=k.replace("Heckman ML, excluded: ", "Excluded: "))
ax.set_xlabel("ρ (assumed, or estimated for the points)")
ax.set_ylabel("Non-workers' mean log offered wage\nminus workers' mean log wage")
ax.set_title("The offered-wage gap is set by ρ")
ax.legend(fontsize=7.8, loc="lower left", ncol=1)
ax = axes[1]
ax.axvspan(lo, hi, color=INK2, alpha=0.15, lw=0, label="Manski worst-case bounds")
names = list(points)
for i, k in enumerate(names):
    ax.plot(points[k][2], i, mk[k][0], color=mk[k][1], ms=7, mec="white")
ax.axvline(np.median(y[d]), color=INK, lw=1, ls=":", label="Median log wage of the workers")
ax.set_yticks(range(len(names)), [k.replace("Heckman ML, excluded: ", "Excluded: ") for k in names], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("Median log offered wage of all 753 women")
ax.set_title("What the data alone allow")
ax.legend(fontsize=8.5, loc="lower left")
fig.savefig(FIG / "fig6_mroz_sensitivity.png"); plt.close(fig)
