"""Case 4 - weighting by a variance model: net financial wealth of single people.

wooldridge::k401ksubs, single-person households (fsize == 1, n = 2,017):
net total financial assets (nettfa, $1,000s) on income, (age - 25)^2, sex and
eligibility for a 401(k) plan. Wooldridge's textbook assumes var(u | x)
proportional to income and fits WLS with weights 1/inc. Here the variance
function is also estimated (FGLS, power and exponential forms), and every
weighted fit is reported with classical and HC3 standard errors; a pairs
bootstrap gives the sampling spread of each estimator without relying on any
standard-error formula.

Run:  python 04_wealth_wls.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, FIG, INK2, ORANGE, SURFACE, fetch, style, write
from het import bp_test, fgls, ols, wls

warnings.filterwarnings("ignore")
style()
k = fetch("wooldridge", "k401ksubs")
s = k[k.fsize == 1].reset_index(drop=True)
NAMES = ["const", "inc", "(age-25)^2", "male", "e401k"]
X = np.column_stack([np.ones(len(s)), s.inc, (s.age - 25) ** 2, s.male, s.e401k]).astype(float)
y = s.nettfa.values.astype(float)
inc = s.inc.values.astype(float)
out = [f"k401ksubs: {len(k)} households, {len(s)} with one person"]
out += [f"k401ksubs, single-person households: n = {len(s)}; nettfa mean {y.mean():.2f}, SD {y.std(ddof=1):.2f}, "
       f"min {y.min():.1f}, max {y.max():.1f}; income mean {inc.mean():.2f}, range {inc.min():.2f}-{inc.max():.2f}"]

f_ols = ols(X, y)
lm, p = bp_test(f_ols.resid, X[:, 1:])
lm2, p2 = bp_test(f_ols.resid, inc)
out.append(f"OLS residuals: Koenker BP on all regressors LM {lm:.2f} (p {p:.2e}); on income alone LM {lm2:.2f} "
           f"(p {p2:.2e})")
# what variance function do the data suggest?
D = np.column_stack([np.ones(len(s)), np.log(inc)])
g_pow = np.linalg.lstsq(D, np.log(np.maximum(f_ols.resid ** 2, 1e-12)), rcond=None)[0]
bins = np.quantile(inc, np.linspace(0, 1, 6))
band = np.clip(np.digitize(inc, bins[1:-1]), 0, 4)
sd_band = [f_ols.resid[band == b].std(ddof=1) for b in range(5)]
mid_band = [np.median(inc[band == b]) for b in range(5)]
out.append(f"Residual SD by income quintile (median income): " + "; ".join(
    f"{m:.1f}: {v:.2f}" for m, v in zip(mid_band, sd_band)))
out.append(f"Power variance function fitted to log e^2: var proportional to inc^{g_pow[1]:.2f}")
out.append(f"OLS residuals outside the plotted range -200 to 330 (figure 5): "
           f"{int(np.sum((f_ols.resid > 330) | (f_ols.resid < -200)))}; largest residual {f_ols.resid.max():.0f}")

fits = {"OLS": f_ols,
        "WLS, var proportional to income": wls(X, y, 1 / inc)}
fp, var_p, gp = fgls(X, y, inc, form="power")
fe, var_e, ge = fgls(X, y, X[:, 1:], form="exponential")
fits["FGLS, power in income"] = fp
fits["FGLS, exponential in all regressors"] = fe
out.append(f"FGLS power: log var = {gp[0]:.3f} + {gp[1]:.3f} log inc; FGLS exponential: log var on regressors, "
           f"coefficients {np.round(ge, 4)}")
res = {}
for lab, f in fits.items():
    c, h = f.se("const"), f.se("HC3")
    res[lab] = (f.coef, c, h)
    out.append(f"-- {lab}")
    for j, nm in enumerate(NAMES):
        out.append(f"   {nm:11s} {f.coef[j]:9.4f}  SE classical {c[j]:.4f}  HC3 {h[j]:.4f}")
    if lab != "OLS":
        z = f.resid                                 # residuals on the weighted scale
        lmw, pw = bp_test(z, inc)
        out.append(f"   weighted residuals: Koenker BP on income LM {lmw:.2f} (p {pw:.4f})")

# pairs bootstrap: the spread of each estimator, with no standard-error formula
rng = np.random.default_rng(6)
boot = {lab: [] for lab in fits}
for _ in range(2000):
    i = rng.integers(0, len(s), len(s))
    Xi, yi, inci = X[i], y[i], inc[i]
    boot["OLS"].append(ols(Xi, yi).coef)
    boot["WLS, var proportional to income"].append(wls(Xi, yi, 1 / inci).coef)
    boot["FGLS, power in income"].append(fgls(Xi, yi, inci, form="power")[0].coef)
    boot["FGLS, exponential in all regressors"].append(fgls(Xi, yi, Xi[:, 1:], form="exponential")[0].coef)
out.append("Pairs bootstrap (2,000 resamples): SD of each coefficient")
for lab in fits:
    sd = np.std(boot[lab], axis=0, ddof=1)
    res[lab] = (*res[lab], sd)
    out.append(f"   {lab:38s} " + "; ".join(f"{nm} {sd[j]:.4f}" for j, nm in enumerate(NAMES)))
for j in (1, 4):
    base = res["OLS"][3][j]
    out.append(f"   {NAMES[j]}: bootstrap SD relative to OLS: " + "; ".join(
        f"{lab} {res[lab][3][j] / base:.3f}" for lab in fits))
out.append("Standard error divided by the pairs-bootstrap SD (1 = agrees with the formula-free spread)")
for lab in fits:
    b, c, h, sd = res[lab]
    out.append(f"   {lab:38s} " + "; ".join(f"{nm} classical {c[j] / sd[j]:.2f}, HC3 {h[j] / sd[j]:.2f}"
                                           for j, nm in enumerate(NAMES)))
for lab in list(fits)[1:]:
    b = res[lab][0]
    out.append(f"   {lab}: shift from OLS in OLS HC3 standard errors: " + "; ".join(
        f"{nm} {(b[j] - res['OLS'][0][j]) / res['OLS'][2][j]:+.2f}" for j, nm in enumerate(NAMES)))
write("04_wealth_wls.txt", "\n".join(out))

# --------------------------------------------------------------- figure 4
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.5})
ax = axes[0]
ax.scatter(inc, f_ols.resid, s=5, color="#9a9993", alpha=0.45, lw=0, rasterized=True)
gx = np.linspace(inc.min(), inc.max(), 200)
s2_lin = np.mean(f_ols.resid ** 2 / inc)                      # sigma^2 * inc, sigma^2 by moments
for curve, col, lab in ((np.sqrt(s2_lin * gx), ORANGE, "SD ∝ √income (Wooldridge's weights)"),
                        (np.sqrt(np.exp(gp[0] + gp[1] * np.log(gx)) * np.exp(1.2704)), BLUE,
                         f"SD from fitted power function (var ∝ inc^{gp[1]:.2f})")):
    ax.plot(gx, 2 * curve, color=col, lw=2, label=lab)
    ax.plot(gx, -2 * curve, color=col, lw=2)
ax.set_ylim(-200, 330)
n_off = int(np.sum((f_ols.resid > 330) | (f_ols.resid < -200)))
ax.text(0.02, 0.03, f"{n_off} residuals beyond the axis (largest {f_ols.resid.max():.0f})", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8, color=INK2)
ax.set_xlabel("Income ($1,000s)")
ax.set_ylabel("OLS residual ($1,000s)")
ax.set_title("Wealth residuals fan out with income")
ax.legend(fontsize=8.5, loc="upper left", title="Lines: ±2 SD", title_fontsize=8.5, alignment="left", frameon=True,
          facecolor=SURFACE, edgecolor="none", framealpha=0.9)
ax = axes[1]
labs = list(fits)
short = ["OLS", "WLS 1/inc", "FGLS power", "FGLS exponential"]
for i, lab in enumerate(labs):
    b, c, h, sd = res[lab]
    for off, se, col, mk, nm in ((-0.15, c, INK2, "o", "classical SE"), (0.0, h, BLUE, "s", "HC3 SE"),
                                 (0.15, sd, ORANGE, "D", "bootstrap SD")):
        ax.errorbar(b[4], i + off, xerr=1.96 * se[4], fmt=mk, color=col, capsize=2, ms=4,
                    label=nm if i == 0 else None)
ax.set_yticks(range(len(labs)))
ax.set_yticklabels(short)
ax.invert_yaxis()
ax.set_xlabel("401(k) eligibility effect on net financial assets ($1,000s), ±1.96 SE")
ax.set_title("Weighting buys precision; HC3 keeps it honest")
ax.legend(fontsize=8.5, loc="lower right")
fig.savefig(FIG / "fig5_wealth_wls.png"); plt.close(fig)
