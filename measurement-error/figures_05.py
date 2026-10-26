"""Figures 5-8 from the experiment-5 summaries (and one illustrative SIMEX run)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import AQUA, BLUE, FIG, INK2, MAGENTA, ORANGE, RES, VIOLET, YELLOW, GREEN, style
from me import design, extrapolate, simex

style()
t = pd.read_csv(RES / "05_summary.csv").set_index(["family", "struct", "rel", "method"])
pr = pd.read_csv(RES / "05_prediction.csv")
METHODS = ["Naive, one reading", "Naive, mean of two", "Regression calibration, replicates",
           "SIMEX, quadratic", "SIMEX, rational", "IV, second reading",
           "Regression calibration, validation", "Multiple imputation, validation", "Validation rows only"]
SHORT = ["Naive, one reading", "Naive, mean of two", "RC, replicates", "SIMEX, quadratic", "SIMEX, rational",
         "IV, second reading", "RC, validation", "MI, validation", "Validation rows only"]
COL = [INK2, "#9a9993", BLUE, VIOLET, MAGENTA, YELLOW, AQUA, GREEN, ORANGE]
TITLES = {"classical": "Classical", "correlated": "Correlated replicate errors",
          "heteroscedastic": "Heteroscedastic", "berkson": "Berkson",
          "systematic": "Systematic (flattened reports)", "differential": "Differential (tied to the outcome)"}

# ------------------------------------------------ figure 6: bias by structure
REL = 0.7
fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.4), sharex=True, sharey=True)
for ax, st in zip(axes.flat, TITLES):
    for i, m in enumerate(METHODS):
        r = t.loc[("gaussian", st, REL, m)]
        sd = r.sd / r.truth
        ax.errorbar(100 * r.relbias, i, xerr=100 * 1.96 * sd, fmt="o", color=COL[i], ms=5, capsize=2, lw=1.4)
        if not np.isnan(r.cover):
            ax.text(64, i, f"{100 * r.cover:.0f}%", va="center", ha="right", fontsize=8,
                    color=ORANGE if abs(r.cover - 0.95) > 0.04 else INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(TITLES[st], fontsize=10.5)
    ax.set_xlim(-75, 66)
for ax in axes[:, 0]:
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(SHORT, fontsize=9)
    ax.invert_yaxis()
for ax in axes[1]:
    ax.set_xlabel("Bias of the exposure coefficient (%), ±1.96 SD")
fig.suptitle(f"Nine estimators, six kinds of error, reliability {REL}; numbers at right: coverage of 95% intervals",
             x=0.02, ha="left", fontweight="bold", fontsize=11.5, y=1.0)
fig.savefig(FIG / "fig6_simulation_bias.png"); plt.close(fig)

# ----------------------------------------------- figure 7: SIMEX, one sample
from common import load_nhanes
P = load_nhanes().dropna(subset=["sbp_ave", "age", "bmi"]).reset_index(drop=True)
rng = np.random.default_rng(31)
idx = rng.choice(len(P), 1000, replace=False)
x = P.sbp_ave.values[idx] / 10
Z = P[["age", "female", "bmi"]].values[idx]
beta = np.array([2.0, 0.5, 0.02, -0.3, 0.05])
y = design(x, Z) @ beta + rng.standard_normal(1000)
s2 = (P.sbp_ave.values / 10).var() * (1 - 0.5) / 0.5
w = x + np.sqrt(s2) * rng.standard_normal(1000)
fit, lam, path = simex(w, Z, y, s2, B=200, lambdas=np.linspace(0.25, 2.5, 10), rng=32, return_path=True)
g = path[:, 1]
grid = np.linspace(-1, 2.6, 200)
A = np.column_stack([np.ones_like(lam), lam, lam ** 2])
qc = np.linalg.lstsq(A, g, rcond=None)[0]
best = (np.inf, None)
for c in 1 + np.geomspace(1e-3, 1e3, 4000):
    M = np.column_stack([np.ones_like(lam), 1 / (c + lam)])
    ab = np.linalg.lstsq(M, g, rcond=None)[0]
    sse = np.sum((M @ ab - g) ** 2)
    if sse < best[0]:
        best = (sse, (ab, c))
ab, c = best[1]
fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.plot(lam, g, "o", color=INK2, label="Naive slope with extra noise λ·σ²ᵤ added (mean of 200 draws)")
ax.plot(grid, qc[0] + qc[1] * grid + qc[2] * grid ** 2, color=VIOLET, lw=2, label="Quadratic extrapolant")
ax.plot(grid, ab[0] + ab[1] / (c + grid), color=MAGENTA, lw=2, ls="--", label="Rational extrapolant a + b/(c + λ)")
ax.axhline(beta[1], color=BLUE, lw=1, ls=":")
ax.text(1.6, beta[1] + 0.008, f"true slope {beta[1]}", color=BLUE, fontsize=9)
ax.axvline(-1, color=INK2, lw=0.8)
ax.plot(-1, qc[0] - qc[1] + qc[2], "s", color=VIOLET, ms=7)
ax.plot(-1, ab[0] + ab[1] / (c - 1), "D", color=MAGENTA, ms=7)
ax.annotate("λ = −1: no measurement error", (-1, 0.06), xytext=(6, 0), textcoords="offset points", fontsize=9,
            color=INK2)
ax.set_xlabel("λ, added error variance as a multiple of the measurement-error variance")
ax.set_ylabel("Slope of the outcome on the noisy exposure")
ax.set_title("SIMEX on one sample: reliability 0.5, n = 1,000")
ax.set_ylim(0.04, 0.62)
ax.legend(fontsize=8.5, loc="upper right")
fig.savefig(FIG / "fig7_simex.png"); plt.close(fig)
XP, ZP = P.sbp_ave.values / 10, P[["age", "female", "bmi"]].values
DZ = np.column_stack([np.ones(len(XP)), ZP])
r2 = 1 - np.var(XP - DZ @ np.linalg.lstsq(DZ, XP, rcond=None)[0]) / np.var(XP)
gam = np.linalg.lstsq(DZ, XP, rcond=None)[0][1]
lines = [f"Population: R^2 of x (BPSysAve/10) on age, female and BMI {r2:.3f}; coefficient of age in that "
         f"regression {gam:.4f} per year"]
for lam_u in (0.9, 0.7, 0.5):
    lam_c = (1 - r2) / ((1 - r2) + (1 - lam_u) / lam_u)
    lines.append(f"  unconditional reliability {lam_u}: conditional reliability {lam_c:.3f}; theoretical naive bias "
                 f"(one reading) {lam_c - 1:+.1%}; theoretical age-coefficient bias "
                 f"{(1 - lam_c) * beta[1] * gam / beta[2]:+.1%}")
lines.append(f"SIMEX example (classical error, reliability 0.5, n = 1000, B = 200): naive slope {g[0]:.4f}; "
             f"quadratic extrapolation {qc[0] - qc[1] + qc[2]:.4f}; rational extrapolation "
             f"{ab[0] + ab[1] / (c - 1):.4f}; truth {beta[1]}")
rep = pd.read_csv(RES / "05_replications.csv.gz")
lr = rep[(rep.family == "binomial") & (rep.method == "SIMEX, rational")]
for rel, s_ in lr.groupby("rel"):
    lines.append(f"Logistic, rational SIMEX, reliability {rel}: {int((s_.est > 1).sum())} of {len(s_)} estimates above 1 "
                 f"(truth 0.3), largest {s_.est.max():.2f}; median {s_.est.median():.4f} "
                 f"(bias {s_.est.median() / 0.3 - 1:+.1%})")
(RES / "05_population_and_simex.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ------------------------------------------ figure 5: leakage into age
rels = [0.9, 0.7, 0.5]
show = [("Naive, one reading", INK2, "o-"), ("Naive, mean of two", "#9a9993", "o-"),
        ("Regression calibration, replicates", BLUE, "s-"), ("Multiple imputation, validation", GREEN, "^-")]
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
for m, col, st in show:
    axes[0].plot(rels, [100 * t.loc[("gaussian", "classical", r, m)].relbias for r in rels], st, color=col, label=m)
    axes[1].plot(rels, [100 * t.loc[("gaussian", "classical", r, m)].age_relbias for r in rels], st, color=col)
for ax, ttl in zip(axes, ("Exposure (systolic pressure) coefficient", "Age coefficient (measured without error)")):
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Reliability of one reading")
    ax.set_xticks(rels)
    ax.invert_xaxis()
    ax.set_title(ttl, fontsize=10.5)
axes[0].set_ylabel("Bias (%)")
axes[0].legend(fontsize=8.5, loc="lower left")
fig.suptitle("Classical error in one covariate biases the others", x=0.02, ha="left", fontweight="bold",
             fontsize=11.5, y=1.02)
fig.savefig(FIG / "fig5_leakage.png"); plt.close(fig)

# --------------------------------------- figure 8: prediction and correction
p = pr[pr.struct == "classical"].groupby(["rel", "model", "measured"]).mean(numeric_only=True)
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
combos = [("naive", "one reading", INK2, "o-", "Naive model, deployed on one reading"),
          ("corrected", "one reading", BLUE, "o--", "Corrected model, deployed on one reading"),
          ("naive", "exact", ORANGE, "s--", "Naive model, deployed on exact values"),
          ("corrected", "exact", GREEN, "s-", "Corrected model, deployed on exact values")]
for model, meas, col, st, lab in combos:
    axes[0].plot(rels, [p.loc[(r, model, meas)].slope for r in rels], st, color=col, label=lab)
    axes[1].plot(rels, [p.loc[(r, model, meas)].mse for r in rels], st, color=col)
axes[0].axhline(1, color=INK2, lw=0.8)
axes[0].set_ylabel("Calibration slope (1 = well calibrated)")
axes[1].set_ylabel("Mean squared prediction error")
for ax in axes:
    ax.set_xlabel("Reliability of one reading")
    ax.set_xticks(rels)
    ax.invert_xaxis()
axes[0].set_title("Calibration on new data", fontsize=10.5)
axes[1].set_title("Accuracy on new data", fontsize=10.5)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8.8,
           bbox_to_anchor=(0.5, 0.0), frameon=False)
fig.suptitle("Correct the coefficients for the measurement you will deploy on, not for the truth",
             x=0.02, ha="left", fontweight="bold", fontsize=11.5, y=1.02)
fig.savefig(FIG / "fig8_prediction.png"); plt.close(fig)
print("figures 5, 6, 7, 8 written")
