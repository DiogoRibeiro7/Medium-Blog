"""Experiment 4 - multi-item scales: personality and educational attainment.

psych::bfi: 2,800 respondents to 25 Big Five personality items (five per
trait, 1-6 agreement scale) from the SAPA project, with age, gender and
education. Survey scale scores are the everyday case of a covariate measured
with error, and the items themselves give the reliability: Cronbach's alpha
or McDonald's omega from a one-factor model per scale.

The analysis is a "driver" regression of education level on the five scale
scores, age and gender. Correcting for unreliability (multivariate moment
correction with error variance (1 - reliability) x variance of each scale)
changes the size and the ranking of the coefficients, because the scales are
not equally reliable.

Run:  python 04_bfi_scales.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

from common import BLUE, FIG, INK2, ORANGE, AQUA, fetch, style, write
from me import moment_correct, ols

warnings.filterwarnings("ignore")
style()
b = fetch("psych", "bfi")
KEYS = {"Agreeableness": ["-A1", "A2", "A3", "A4", "A5"],
        "Conscientiousness": ["C1", "C2", "C3", "-C4", "-C5"],
        "Extraversion": ["-E1", "-E2", "E3", "E4", "E5"],
        "Neuroticism": ["N1", "N2", "N3", "N4", "N5"],
        "Openness": ["O1", "-O2", "O3", "O4", "-O5"]}
items = [k.lstrip("-") for v in KEYS.values() for k in v]
d = b.dropna(subset=items + ["education", "age", "gender"]).reset_index(drop=True)
out = [f"bfi: {len(b)} respondents; complete on the 25 items: {int(b[items].notna().all(axis=1).sum())}; "
       f"complete on the 25 items, education, age and gender: {len(d)}; "
       f"education 1-5 (1 = high school not finished ... 5 = graduate degree), mean {d.education.mean():.2f}"]

def cronbach(M):
    k = M.shape[1]
    return k / (k - 1) * (1 - M.var(axis=0, ddof=1).sum() / M.sum(axis=1).var(ddof=1))


scores, rel, ITEMS = {}, {}, {}
for trait, keys in KEYS.items():
    M = np.column_stack([7 - d[k[1:]].values if k.startswith("-") else d[k].values for k in keys]).astype(float)
    ITEMS[trait] = M
    s = M.mean(axis=1)
    alpha = cronbach(M)
    fa = FactorAnalysis(n_components=1, random_state=0).fit(M)
    lam, psi = fa.components_[0], fa.noise_variance_
    omega = lam.sum() ** 2 / (lam.sum() ** 2 + psi.sum())
    scores[trait], rel[trait] = s, (alpha, omega)
    out.append(f"  {trait:18s} mean {s.mean():.2f}, SD {s.std(ddof=1):.3f}; Cronbach's alpha {alpha:.3f}, "
               f"omega (one factor) {omega:.3f}")
S = np.column_stack(list(scores.values()))
traits = list(KEYS)
Z = np.column_stack([d.age.values / 10, (d.gender == 2).astype(float).values])   # gender 2 = female
y = d.education.values.astype(float)
X = np.column_stack([np.ones(len(d)), S, Z])
naive = ols(X, y)
sd = S.std(axis=0, ddof=1)
corr = {}
for which in ("alpha", "omega"):
    r = np.array([rel[t][0 if which == "alpha" else 1] for t in traits])
    corr[which] = moment_correct(S, Z, y, (1 - r) * S.var(axis=0)).coef

rng = np.random.default_rng(5)
boot = {"naive": [], "alpha": []}
r_a = np.array([rel[t][0] for t in traits])
for _ in range(1000):
    i = rng.integers(0, len(d), len(d))
    boot["naive"].append(ols(X[i], y[i], se=False).coef[1:6])
    r_i = np.array([cronbach(ITEMS[t][i]) for t in traits])          # reliability re-estimated too
    boot["alpha"].append(moment_correct(S[i], Z[i], y[i], (1 - r_i) * S[i].var(axis=0)).coef[1:6])
ci = {k: np.percentile(np.array(v), [2.5, 97.5], axis=0) for k, v in boot.items()}

out.append("Education ~ five scale scores + age (per decade) + female; coefficients per scale SD:")
for j, t in enumerate(traits):
    out.append(f"  {t:18s} naive {naive.coef[1 + j] * sd[j]:+.4f} (95% CI {ci['naive'][0, j] * sd[j]:+.4f} to "
               f"{ci['naive'][1, j] * sd[j]:+.4f}); alpha-corrected {corr['alpha'][1 + j] * sd[j]:+.4f} "
               f"(CI {ci['alpha'][0, j] * sd[j]:+.4f} to {ci['alpha'][1, j] * sd[j]:+.4f}); omega-corrected "
               f"{corr['omega'][1 + j] * sd[j]:+.4f}; ratio alpha-corrected / naive "
               f"{corr['alpha'][1 + j] / naive.coef[1 + j]:.2f}")
rank = lambda c: [traits[j] for j in np.argsort(-np.abs(c[1:6] * sd))]
out.append(f"  ranking by |standardised coefficient|: naive {rank(naive.coef)}; alpha-corrected {rank(corr['alpha'])}")
out.append(f"  age per decade: naive {naive.coef[6]:+.4f}, alpha-corrected {corr['alpha'][6]:+.4f}; female: naive "
           f"{naive.coef[7]:+.4f}, alpha-corrected {corr['alpha'][7]:+.4f}")
R = np.corrcoef(S.T)
out.append("Scale correlations, observed -> disattenuated with alpha (Spearman 1904):")
for i in range(5):
    for j in range(i + 1, 5):
        out.append(f"  {traits[i][:5]}-{traits[j][:5]}: {R[i, j]:+.3f} -> {R[i, j] / np.sqrt(r_a[i] * r_a[j]):+.3f}")
write("04_bfi_scales.txt", "\n".join(out))

# --------------------------------------------------------------- figure 4
fig, ax = plt.subplots(figsize=(8.8, 4.4))
order = np.argsort(naive.coef[1:6] * sd)
for pos, j in enumerate(order):
    a, c_ = naive.coef[1 + j] * sd[j], corr["alpha"][1 + j] * sd[j]
    ax.plot([a, c_], [pos, pos], color=INK2, lw=1, zorder=1)
    ax.errorbar(a, pos + 0.12, xerr=[[a - ci["naive"][0, j] * sd[j]], [ci["naive"][1, j] * sd[j] - a]],
                fmt="o", color=AQUA, capsize=2, label="As measured" if pos == 0 else None)
    ax.errorbar(c_, pos - 0.12, xerr=[[c_ - ci["alpha"][0, j] * sd[j]], [ci["alpha"][1, j] * sd[j] - c_]],
                fmt="o", color=BLUE, capsize=2, label="Corrected for unreliability (alpha)" if pos == 0 else None)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(5))
ax.set_yticklabels([f"{traits[j]} (α = {r_a[j]:.2f})" for j in order])
ax.set_xlabel("Education levels per SD of the observed scale score (95% bootstrap CI)")
ax.set_title("A driver analysis on survey scales, before and after the correction")
ax.legend(fontsize=9, loc="lower right")
fig.savefig(FIG / "fig4_bfi_drivers.png"); plt.close(fig)
