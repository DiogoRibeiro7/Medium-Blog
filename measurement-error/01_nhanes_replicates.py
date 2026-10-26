"""Experiment 1 - replicate readings: blood pressure in NHANES 2009-2012.

Every examined adult had systolic and diastolic pressure read three times in
one session. The three readings let us estimate the within-session error,
and their differences show that the first reading is not a replicate of the
others: it is higher, and more so in older people. Used as an outcome, that
non-classical error biases a regression slope, which classical outcome error
would not.

Run:  python 01_nhanes_replicates.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, FIG, INK2, ORANGE, AQUA, load_nhanes, style, write
from me import design, ols

warnings.filterwarnings("ignore")
style()
d = load_nhanes()
s = d.dropna(subset=["sbp1", "sbp2", "sbp3", "age", "bmi"]).reset_index(drop=True)
out = [f"NHANES 2009-2012 adults aged 20-79: {len(d)}; with three systolic readings, age and BMI: {len(s)}"]


def icc(M):
    """One-way intraclass correlation of a single reading (k readings per person)."""
    k = M.shape[1]
    within = M.var(axis=1, ddof=1).mean()
    between = M.mean(axis=1).var(ddof=1) - within / k
    return between / (between + within), within


R = s[["sbp1", "sbp2", "sbp3"]].values
r_all, w_all = icc(R)
r_23, w_23 = icc(R[:, 1:])
c = np.corrcoef(R.T)
out.append(f"Systolic: pairwise correlations r1-r2 {c[0, 1]:.3f}, r1-r3 {c[0, 2]:.3f}, r2-r3 {c[1, 2]:.3f}; "
           f"ICC of one reading from readings 1-3 {r_all:.3f} (within-person SD {np.sqrt(w_all):.2f} mmHg), "
           f"from readings 2-3 {r_23:.3f} (SD {np.sqrt(w_23):.2f}); between-person SD of the 2-3 mean "
           f"{R[:, 1:].mean(1).std(ddof=1):.2f}")
D = d.dropna(subset=["dbp1", "dbp2", "dbp3"])[["dbp1", "dbp2", "dbp3"]].values
zeros = (D == 0).any(axis=1)
rd, wd = icc(D[~zeros])
out.append(f"Diastolic: {len(D)} with three readings; {zeros.sum()} people with at least one reading of 0 mmHg "
           f"({(D == 0).sum()} zero readings); ICC of one reading without them {rd:.3f} "
           f"(within SD {np.sqrt(wd):.2f})")
out.append(f"All systolic readings even: {bool(np.all(R % 2 == 0))} (recorded to 2 mmHg); share of readings "
           f"ending in 0: {np.mean(R % 10 == 0):.3f} (0.200 if the last even digit were uniform)")

# first-reading effect
m23 = R[:, 1:].mean(axis=1)
diff = R[:, 0] - m23
Z = s[["age", "female", "bmi"]].values
f = ols(design(s.age.values / 10, s[["female", "bmi"]].values), diff)
out.append(f"First reading minus mean of readings 2-3: mean {diff.mean():+.2f} mmHg (SD {diff.std(ddof=1):.2f}); "
           f"regression on age per decade {f.coef[1]:+.3f} (SE {f.se[1]:.3f}), female {f.coef[2]:+.3f} "
           f"(SE {f.se[2]:.3f}), BMI {f.coef[3]:+.3f} (SE {f.se[3]:.3f})")
bins = [20, 30, 40, 50, 60, 70, 80]
band = np.digitize(s.age.values, bins[1:-1])
by_age = [(diff[band == b].mean(), diff[band == b].std(ddof=1) / np.sqrt((band == b).sum()), (band == b).sum())
          for b in range(len(bins) - 1)]
out.append("  by age band: " + "; ".join(f"{bins[b]}-{bins[b + 1] - 1}: {m:+.2f} (SE {e:.2f}, n {k})"
                                         for b, (m, e, k) in enumerate(by_age)))
d32 = R[:, 1] - R[:, 2]
f32 = ols(design(s.age.values / 10, s[["female", "bmi"]].values), d32)
out.append(f"Second minus third reading: mean {d32.mean():+.2f}; on age per decade {f32.coef[1]:+.3f} "
           f"(SE {f32.se[1]:.3f})")

# the reading used as the outcome changes the age slope
outcomes = {"reading 1": R[:, 0], "reading 2": R[:, 1], "reading 3": R[:, 2],
            "mean of 2-3 (NHANES)": m23, "mean of 1-3": R.mean(axis=1)}
res = {}
for lab, yv in outcomes.items():
    f = ols(design(s.age.values, s[["female", "bmi"]].values), yv)
    r2 = 1 - np.sum((yv - design(s.age.values, s[["female", "bmi"]].values) @ f.coef) ** 2) / np.sum((yv - yv.mean()) ** 2)
    res[lab] = (f.coef[1], f.se[1], r2)
    out.append(f"  SBP ({lab:20s}) ~ age + female + BMI: age slope {f.coef[1]:.4f} mmHg/year "
               f"(SE {f.se[1]:.4f}), R^2 {r2:.4f}")
gap = res["reading 1"][0] - res["mean of 2-3 (NHANES)"][0]
out.append(f"  age slope with reading 1 minus with the 2-3 mean: {gap:+.4f} ({gap / res['mean of 2-3 (NHANES)'][0]:+.1%})")

# what the replicates imply for a regression on SBP as an exposure
lam_1 = r_23                                  # a single reading
lam_2 = 1 / (1 + (1 - r_23) / r_23 / 2)       # mean of two readings
out.append(f"Implied attenuation factor for SBP as an exposure (unconditional): one reading {lam_1:.3f}, "
           f"mean of two {lam_2:.3f}")
write("01_nhanes_replicates.txt", "\n".join(out))

# --------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})
ax = axes[0]
mid = [(bins[b] + bins[b + 1] - 1) / 2 for b in range(len(bins) - 1)]
ax.errorbar(mid, [m for m, _, _ in by_age], yerr=[1.96 * e for _, e, _ in by_age], fmt="o-", color=ORANGE,
            capsize=3, label="Reading 1 minus mean of readings 2 and 3")
d32_age = [d32[band == b].mean() for b in range(len(bins) - 1)]
d32_se = [d32[band == b].std(ddof=1) / np.sqrt((band == b).sum()) for b in range(len(bins) - 1)]
ax.errorbar(mid, d32_age, yerr=[1.96 * e for e in d32_se], fmt="s--", color=AQUA, capsize=3,
            label="Reading 2 minus reading 3")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(mid)
ax.set_xticklabels([f"{bins[b]}–{bins[b + 1] - 1}" for b in range(len(bins) - 1)])
ax.set_xlabel("Age band (years)")
ax.set_ylabel("Mean difference (mmHg), 95% CI")
ax.set_title("The first reading is not a replicate")
ax.legend(fontsize=9, loc="upper left")
ax = axes[1]
labs = list(outcomes)
for i, lab in enumerate(labs):
    b, se, _ = res[lab]
    col = ORANGE if lab == "reading 1" else BLUE
    ax.errorbar(b, i, xerr=1.96 * se, fmt="o", color=col, capsize=3)
ax.set_yticks(range(len(labs)))
ax.set_yticklabels([l[0].upper() + l[1:] for l in labs])
ax.invert_yaxis()
ax.set_xlabel("mmHg per year, 95% CI")
ax.set_title("Age slope of SBP, by reading used as outcome")
fig.savefig(FIG / "fig1_nhanes_readings.png"); plt.close(fig)
