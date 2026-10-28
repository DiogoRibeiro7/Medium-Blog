"""Dropout on the inputs of a linear regression, solved exactly.

Covariates are rows of the UCI power-plant data (ambient temperature, exhaust
vacuum, ambient pressure, relative humidity), standardised; the outcome is
simulated from the full-data least-squares fit, so the true regression
function is known. Larger samples resample covariate rows.

With inverted dropout (keep probability q) on the inputs, the expected
training loss is least squares plus lambda * sum_j w_j^2 * sum_i x_ij^2 with
lambda = (1 - q) / q, so the trained weights are
    w_q = (X'X + lambda diag(X'X))^{-1} X'y
and the MC dropout predictive distribution of f(x) = b + sum_j x_j r_j w_j / q
has, exactly,
    mean  b + x'w_q
    var   lambda * sum_j x_j^2 w_qj^2.
The Bayesian posterior of f(x) under a flat prior and known noise variance
has variance sigma^2 (1/n + x'(X'X)^{-1} x). Nothing here is sampled except
the data sets themselves.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcd import (AQUA, BLUE, FIGURES, INK, INK2, ORANGE, YELLOW, Report, load_uci,
                 style_axes)

rep = Report("01_linear_floor")
rng = np.random.default_rng(20261028)

D = load_uci("power-plant")
Xall = D["X"]
yall = D["y"]
mu, sd = Xall.mean(0), Xall.std(0)
Z = (Xall - mu) / sd
A = np.c_[np.ones(len(Z)), Z]
coef, *_ = np.linalg.lstsq(A, yall, rcond=None)
b_true, w_true = coef[0], coef[1:]
sigma = float(np.sqrt(np.mean((yall - A @ coef) ** 2)))
names = ["temperature", "vacuum", "pressure", "humidity"]

rep("Covariates: UCI power plant, standardised; outcome simulated from the full-data OLS fit")
rep(f"true intercept {b_true:.3f}; true slopes " + ", ".join(f"{n} {w:.3f}" for n, w in zip(names, w_true)))
rep(f"noise sd sigma {sigma:.3f}")
corr = np.corrcoef(Z.T)
rep("covariate correlations: " + ", ".join(f"{corr[i, j]:.2f}" for i in range(4) for j in range(i + 1, 4)))

test = rng.choice(len(Z), 2000, replace=False)
Zt = Z[test]
f_true = b_true + Zt @ w_true


def fit_dropout(Zs, ys, q):
    lam = (1 - q) / q
    zc = Zs - Zs.mean(0)
    yc = ys - ys.mean()
    G = zc.T @ zc
    w = np.linalg.solve(G + lam * np.diag(np.diag(G)), zc.T @ yc)
    b = ys.mean() - Zs.mean(0) @ w
    return b, w, lam


def bayes_var(Zs, Zt):
    zbar = Zs.mean(0)
    zc = Zs - zbar
    Ginv = np.linalg.inv(zc.T @ zc)
    dz = Zt - zbar
    return sigma ** 2 * (1 / len(Zs) + np.einsum("ij,jk,ik->i", dz, Ginv, dz))


QS = [0.995, 0.95, 0.9]
NS = np.unique(np.round(np.logspace(np.log10(50), 5, 16)).astype(int))
REPS = 200

# Asymptotic weights and floors
Sigma = corr  # standardised covariates: X'X / n -> correlation matrix
floor = {}
for q in QS:
    lam = (1 - q) / q
    w_inf = np.linalg.solve(Sigma + lam * np.eye(4), Sigma @ w_true)
    f_inf_sd = np.sqrt(lam * (Zt ** 2) @ (w_inf ** 2))
    bias_inf = np.abs(Zt @ (w_inf - w_true))
    floor[q] = f_inf_sd.mean()
    rep(f"q={q}: lambda {lam:.4f}; limit slopes " + ", ".join(f"{w:.3f}" for w in w_inf)
        + f"; limit MC sd of f(x), mean over test points {f_inf_sd.mean():.3f};"
        + f" limit |bias| of the MC mean, mean over test points {bias_inf.mean():.3f}")

rows = {q: [] for q in QS}
rows_b = []
cover = {q: [] for q in QS}
cover_b = []
for n in NS:
    sd_q = {q: [] for q in QS}
    cv_q = {q: [] for q in QS}
    sd_b, cv_b = [], []
    for r in range(REPS):
        idx = rng.integers(0, len(Z), n)
        Zs = Z[idx]
        ys = b_true + Zs @ w_true + sigma * rng.normal(size=n)
        vb = bayes_var(Zs, Zt)
        A_s = np.c_[np.ones(n), Zs]
        c_ols, *_ = np.linalg.lstsq(A_s, ys, rcond=None)
        fb = c_ols[0] + Zt @ c_ols[1:]
        sd_b.append(np.sqrt(vb).mean())
        cv_b.append(np.mean(np.abs(fb - f_true) <= 1.96 * np.sqrt(vb)))
        for q in QS:
            b, w, lam = fit_dropout(Zs, ys, q)
            v = lam * (Zt ** 2) @ (w ** 2)
            fq = b + Zt @ w
            sd_q[q].append(np.sqrt(v).mean())
            cv_q[q].append(np.mean(np.abs(fq - f_true) <= 1.96 * np.sqrt(v)))
    rows_b.append(np.mean(sd_b))
    cover_b.append(np.mean(cv_b))
    for q in QS:
        rows[q].append(np.mean(sd_q[q]))
        cover[q].append(np.mean(cv_q[q]))

rep("")
rep(f"Spread of f(x) (mean over 2000 test points, average of {REPS} data sets) and coverage of the 95% interval for the true f(x)")
for i, n in enumerate(NS):
    parts = [f"n={n:6d}", f"posterior sd {rows_b[i]:.3f} (cover {cover_b[i]:.3f})"]
    for q in QS:
        parts.append(f"q={q}: MC sd {rows[q][i]:.3f} (cover {cover[q][i]:.3f})")
    rep("  " + "; ".join(parts))

# Panel B: MC spread against effect size at fixed n
n_fix = 2000
scales = np.linspace(0, 3, 13)
sd_scale = {q: [] for q in QS}
sd_scale_b = []
for c in scales:
    acc = {q: [] for q in QS}
    accb = []
    for r in range(100):
        idx = rng.integers(0, len(Z), n_fix)
        Zs = Z[idx]
        ys = b_true + Zs @ (c * w_true) + sigma * rng.normal(size=n_fix)
        accb.append(np.sqrt(bayes_var(Zs, Zt)).mean())
        for q in QS:
            b, w, lam = fit_dropout(Zs, ys, q)
            acc[q].append(np.sqrt(lam * (Zt ** 2) @ (w ** 2)).mean())
    sd_scale_b.append(np.mean(accb))
    for q in QS:
        sd_scale[q].append(np.mean(acc[q]))
rep("")
rep(f"Spread of f(x) at n={n_fix} when every slope is multiplied by c (noise fixed)")
for i, c in enumerate(scales):
    rep(f"  c={c:.2f}: posterior sd {sd_scale_b[i]:.3f}; " + "; ".join(
        f"q={q}: MC sd {sd_scale[q][i]:.3f}" for q in QS))
rep.save()

# ---------------------------------------------------------------------------
# Figures 1 and 2
# ---------------------------------------------------------------------------
FIGURES.mkdir(exist_ok=True)
cols = {0.995: YELLOW, 0.95: ORANGE, 0.9: "#b4441a"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
ax = axes[0]
style_axes(ax)
ax.plot(NS, rows_b, color=BLUE, lw=2, marker="o", ms=4)
ax.text(NS[-1] * 1.12, rows_b[-1], "Bayesian\nposterior", color=INK, va="center", fontsize=9)
for q in QS:
    ax.plot(NS, rows[q], color=cols[q], lw=2, marker="o", ms=4)
    ax.axhline(floor[q], color=cols[q], lw=0.8, ls=":")
    ax.text(NS[-1] * 1.12, rows[q][-1], f"drop {1 - q:.3g}", color=INK, va="center", fontsize=9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(NS[0] * 0.8, NS[-1] * 6)
ax.set_xlabel("training rows n", color=INK2)
ax.set_ylabel("sd of f(x), MW (mean over test points)", color=INK2)
ax.set_title("A. Spread of the regression function", loc="left", fontsize=10.5, color=INK)

ax = axes[1]
style_axes(ax)
ax.axhline(0.95, color=INK2, lw=0.8, ls="--")
ax.plot(NS, cover_b, color=BLUE, lw=2, marker="o", ms=4)
for q in QS:
    ax.plot(NS, cover[q], color=cols[q], lw=2, marker="o", ms=4)
ax.text(NS[-1] * 1.12, cover_b[-1] - 0.01, "Bayesian\nposterior", color=INK, va="center", fontsize=9)
for q, dy in zip(QS, (0.008, 0.0, -0.006)):
    ax.text(NS[-1] * 1.12, cover[q][-1] + dy, f"drop {1 - q:.3g}", color=INK, va="center", fontsize=9)
ax.set_xscale("log")
ax.set_xlim(NS[0] * 0.8, NS[-1] * 6)
ax.set_ylim(0.68, 1.02)
ax.set_xlabel("training rows n", color=INK2)
ax.set_ylabel("coverage of the 95% interval for f(x)", color=INK2)
ax.set_title("B. Coverage of the true regression function", loc="left", fontsize=10.5, color=INK)
fig.text(0.01, 0.01, "Linear regression with inverted dropout on the inputs, exact formulas, 200 simulated data sets per n. "
         "Covariates from the UCI power-plant data; dotted lines: limits as n grows.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIGURES / "fig1_linear_floor.png", dpi=170)

fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor="white")
style_axes(ax)
ax.plot(scales, sd_scale_b, color=BLUE, lw=2, marker="o", ms=4)
ax.text(scales[-1] + 0.06, sd_scale_b[-1], "Bayesian posterior", color=INK, va="center", fontsize=9)
for q in QS:
    ax.plot(scales, sd_scale[q], color=cols[q], lw=2, marker="o", ms=4)
    ax.text(scales[-1] + 0.06, sd_scale[q][-1], f"MC dropout, drop {1 - q:.3g}", color=INK, va="center", fontsize=9)
ax.set_xlim(-0.1, 4.3)
ax.set_xlabel("every slope multiplied by c (noise and n unchanged)", color=INK2)
ax.set_ylabel(f"sd of f(x), MW, n = {n_fix}", color=INK2)
ax.set_title("The dropout spread is proportional to the size of the effect", loc="left", fontsize=10.5, color=INK)
fig.text(0.01, 0.01, "Same design as Figure 1; c = 1 is the power-plant fit, c = 0 is no relationship at all.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIGURES / "fig2_effect_size.png", dpi=170)
print("figures written")
