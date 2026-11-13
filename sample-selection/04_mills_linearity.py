"""Mechanism - how close the inverse Mills ratio is to a straight line where the data are.

Without an exclusion restriction the correction term lambda(z'g) is a function
of the same regressors as the outcome equation, and it is separately
identified only through its curvature. This script measures that curvature
where the data are:

  - for the two real cases without an exclusion restriction (Mroz87 with the
    selection equation restricted to the wage regressors; RandHIE as in
    Cameron and Trivedi), the R^2 of a straight line fitted to lambda over the
    fitted probit indices of the selected observations;
  - for a normally distributed index with SD s and the intercept set to give
    a selected share of 50% or 85%, the same R^2 as a function of s
    (integrated numerically on 200,000 draws).

Run:  python 04_mills_linearity.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import BLUE, FIG, INK, INK2, ORANGE, AQUA, fetch, style, write
from sel import mills, probit

warnings.filterwarnings("ignore")
style()


def line_r2(t):
    lam = mills(t)
    A = np.column_stack([np.ones_like(t), t])
    fit = A @ np.linalg.lstsq(A, lam, rcond=None)[0]
    return 1 - np.sum((lam - fit) ** 2) / np.sum((lam - lam.mean()) ** 2)


out = []
# Mroz87, no exclusion: selection on the wage regressors only
m = fetch("sampleSelection", "Mroz87")
d_m = m.lfp.values.astype(bool)
Zm = np.column_stack([np.ones(len(m)), m.educ, m.exper, m.exper ** 2])
idx_m = probit(Zm, d_m).index
# RandHIE, Cameron and Trivedi's specification
r = fetch("sampleSelection", "RandHIE")
s = r[(r.year == 2) & r.educdec.notna()].reset_index(drop=True)
REG = ["logc", "idp", "lpi", "fmde", "physlm", "disea", "hlthg", "hlthf", "hlthp", "linc", "lfam", "educdec", "xage",
       "female", "child", "fchild", "black"]
d_r = s.binexp.values.astype(bool)
Zr = np.column_stack([np.ones(len(s))] + [s[v].values.astype(float) for v in REG])
idx_r = probit(Zr, d_r).index
cases = {"Mroz87, no exclusion": (idx_m, d_m), "RandHIE, no exclusion": (idx_r, d_r)}
for lab, (idx, d) in cases.items():
    t = idx[d]
    q = np.quantile(t, [0.025, 0.975])
    out.append(f"{lab}: fitted probit index among the selected: mean {t.mean():.3f}, SD {t.std():.3f}, central 95% "
               f"[{q[0]:.3f}, {q[1]:.3f}]; selected share {d.mean():.3f}; R^2 of a straight line through lambda over "
               f"these indices {line_r2(t):.4f}")

# a normal index: R^2 of the straight-line approximation as a function of its SD
rng = np.random.default_rng(4)
w = rng.standard_normal(200_000)
SD = np.round(np.arange(0.1, 3.01, 0.1), 2)
curves = {}
for share in (0.5, 0.85):
    vals, sds = [], []
    for sd in SD:
        # index t = mu + sd * w with latent selection t + v > 0, v ~ N(0, 1): P(select) = Phi(mu / sqrt(1 + sd^2))
        mu = stats.norm.ppf(share) * np.sqrt(1 + sd ** 2)
        t = mu + sd * w
        p_sel = stats.norm.cdf(t)
        keep = rng.random(t.size) < p_sel
        vals.append(line_r2(t[keep]))
        sds.append(t[keep].std())
    curves[share] = (np.array(sds), np.array(vals))
    out.append(f"Normal index, selected share {share}: R^2 of the straight line through lambda over the selected, by SD of "
               f"the latent index (SD among the selected in brackets): " + "; ".join(
                   f"{sd:.1f} ({ss:.2f}): {v:.4f}" for sd, ss, v in zip(SD, sds, vals) if sd in (0.5, 1.0, 1.5, 2.0, 3.0)))
write("04_mills_linearity.txt", "\n".join(out))

# --------------------------------------------------------------- figure 2
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"wspace": 0.3})
ax = axes[0]
tt = np.linspace(-3, 3, 400)
ax.plot(tt, mills(tt), color=INK, lw=2, label="Inverse Mills ratio λ(index)")
for (lab, (idx, d)), col in zip(cases.items(), (ORANGE, BLUE)):
    t = idx[d]
    q = np.quantile(t, [0.025, 0.975])
    ax.axvspan(q[0], q[1], color=col, alpha=0.15, lw=0,
               label=f"{lab.split(',')[0]}: central 95% of the selected,\nR² of a straight line {line_r2(t):.3f}")
ax.set_xlabel("Probit index z′γ of a selected observation")
ax.set_ylabel("λ = φ(z′γ) / Φ(z′γ)")
ax.set_ylim(0, 2.5)
ax.set_title("The correction term is almost a straight line")
ax.legend(fontsize=8.5, loc="upper right")
ax = axes[1]
for share, ls in ((0.5, "-"), (0.85, "--")):
    ax.plot(curves[share][0], curves[share][1], color=INK2, lw=2, ls=ls, label=f"Normal index, selected share {share:.0%}")
for (lab, (idx, d)), col in zip(cases.items(), (ORANGE, BLUE)):
    t = idx[d]
    ax.plot(t.std(), line_r2(t), "o", color=col, ms=7, mec="white")
    if "Rand" in lab:
        ax.annotate(lab.split(",")[0], (t.std(), line_r2(t)), xytext=(-10, -4), textcoords="offset points",
                    fontsize=8.5, color=col, ha="right")
    else:
        ax.annotate(lab.split(",")[0], (t.std(), line_r2(t)), xytext=(0.78, 0.997), textcoords="data", fontsize=8.5,
                    color=col, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
ax.set_xlabel("SD of the probit index among the selected")
ax.set_ylabel("R² of a straight line through λ")
ax.set_ylim(0.6, 1.005)
ax.set_xlim(0, 2.0)
ax.set_title("Only a wide spread of the index bends it")
ax.legend(fontsize=8.5, loc="upper right")
fig.savefig(FIG / "fig2_mills_linearity.png"); plt.close(fig)
