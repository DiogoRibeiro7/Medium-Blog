"""Experiment 4 - real missing values: NHANES 2009-2012, adults aged 20-79.

Data: NHANES::NHANESraw (the actual survey sample, before the package's
resampling), analysed unweighted, so estimates describe this sample rather
than the US population.

* missing-data patterns and who is missing
* complete-case analysis versus MICE (PMM / logistic) for
  SBP ~ age + age^2 + female + BMI + poverty + cholesterol + race + ever-smoker,
  plus mean SBP and the share with SBP >= 140 mmHg
* number of imputations from a pilot run (von Hippel's rule)
* delta-adjustment sensitivity analysis for SBP and for the poverty ratio
  (m = 50 for these runs)

Run:  python 04_nhanes_real.py      (about 6 minutes)
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import BLUE, FIG, INK2, ORANGE, RES, load_nhanes, style, write
from mi import mice, ols, pool

warnings.filterwarnings("ignore")
style()
d = load_nhanes()
d["age2"] = (d.age - 50) ** 2 / 100
N = len(d)
ANALYSIS = ["age", "age2", "female", "bmi", "poverty", "chol", "black", "hispanic",
            "otherrace", "smoker"]
INCOMPLETE = {"sbp": "pmm", "bmi": "pmm", "poverty": "pmm", "chol": "pmm",
              "smoker": "logreg", "educ": "pmm", "dbp": "pmm", "pulse": "pmm",
              "hdl": "pmm", "weight": "pmm", "income": "pmm", "homeown": "logreg"}
COMPLETE = ["age", "age2", "female", "black", "hispanic", "otherrace", "year2011"]
cols = list(INCOMPLETE) + COMPLETE
A = d[cols].values.astype(float)
ix = {c: i for i, c in enumerate(cols)}
methods = {ix[c]: m for c, m in INCOMPLETE.items()}
out = [f"NHANES 2009-2012, ages 20-79, unweighted: N = {N}"]

# ------------------------------------------------------------- 1. patterns
core = ["sbp", "bmi", "poverty", "chol", "smoker"]
pat = d[core].isna().astype(int).astype(str).agg("".join, axis=1)
pc = pat.value_counts()
out.append(f"Missing share: " + ", ".join(f"{c} {d[c].isna().mean():.1%}" for c in core))
out.append(f"Complete cases for the analysis model: {(pat == '00000').sum()} "
           f"({(pat == '00000').mean():.1%})")
out.append("Most frequent patterns over (sbp, bmi, poverty, chol, smoker), 1 = missing:")
for p_, n_ in pc.head(8).items():
    out.append(f"  {p_}: {n_} ({n_ / N:.1%})")
both = d.sbp.isna() & d.chol.isna() & d.bmi.isna()
out.append(f"SBP, BMI and cholesterol all missing together: {both.sum()} "
           f"({both.sum() / d.sbp.isna().sum():.0%} of rows missing SBP)")
out.append(f"Poverty missing while income observed: "
           f"{(d.poverty.isna() & d.income.notna()).sum()}; both missing: "
           f"{(d.poverty.isna() & d.income.isna()).sum()}")

# ------------------------------------------------- 2. who is missing
ageband = pd.cut(d.age, [19, 34, 49, 64, 79], labels=["20-34", "35-49", "50-64", "65-79"])
out.append("SBP missing by age band: " + ", ".join(
    f"{k} {v:.1%}" for k, v in d.sbp.isna().groupby(ageband, observed=True).mean().items()))
race = np.select([d.black == 1, d.hispanic == 1, d.otherrace == 1], ["Black", "Hispanic", "Other"],
                 "White")
out.append("Poverty missing by race group: " + ", ".join(
    f"{k} {v:.1%}" for k, v in d.poverty.isna().groupby(race).mean().items()))
edu = d.educ.map({1: "<9th", 2: "9-11th", 3: "HS", 4: "Some college", 5: "College"})
out.append("Poverty missing by education: " + ", ".join(
    f"{k} {v:.1%}" for k, v in d.poverty.isna().groupby(edu).mean().items()))
# observed outcome among rows missing poverty vs not
out.append(f"Mean observed SBP: poverty observed {d.sbp[d.poverty.notna()].mean():.1f}, "
           f"poverty missing {d.sbp[d.poverty.isna()].mean():.1f} mmHg; mean age "
           f"{d.age[d.poverty.notna()].mean():.1f} vs {d.age[d.poverty.isna()].mean():.1f}")
# logistic regression of R(poverty) on complete variables + observed SBP rows only
from sklearn.linear_model import LogisticRegression
sub = d[d.sbp.notna()]
Xr = sub[["age", "female", "black", "hispanic", "otherrace", "sbp"]].values
Xr = (Xr - Xr.mean(0)) / Xr.std(0)
lr = LogisticRegression(C=1e6).fit(Xr, sub.poverty.isna())
out.append("Logistic model for 'poverty missing' (standardised coefficients, rows with SBP): "
           + ", ".join(f"{k} {v:+.2f}" for k, v in zip(["age", "female", "black", "hispanic",
                                                         "other", "sbp"], lr.coef_[0])))


# ------------------------------------------------- 3. estimates
def analyse(Z):
    X = np.column_stack([np.ones(len(Z))] + [Z[:, ix[c]] for c in ANALYSIS])
    b, v = ols(X, Z[:, ix["sbp"]], robust=True)
    s = Z[:, ix["sbp"]]
    hyp = (s >= 140).mean()
    return (np.r_[b, s.mean(), hyp],
            np.r_[v, s.var(ddof=1) / len(s), hyp * (1 - hyp) / len(s)])


names = ["const"] + ANALYSIS + ["mean SBP", "SBP>=140"]
cc = d[["sbp"] + ANALYSIS].dropna()
Zcc = np.full((len(cc), len(cols)), np.nan)
for c in ["sbp"] + ANALYSIS:
    Zcc[:, ix[c]] = cc[c].values
est_cc, var_cc = analyse(Zcc)
s_avail = d.sbp.dropna()

pilot = mice(A, methods, m=20, maxit=10, seed=[2009, 0])
Q, U = zip(*(analyse(Z) for Z in pilot))
pp = pool(Q, U, N - len(ANALYSIS) - 1)
fmax = float(pp.fmi[:-2].max())
m_rule = int(np.ceil(1 + 0.5 * (fmax / 0.05) ** 2))
M = min(max(m_rule, 20), 100)
out.append(f"\nPilot m = 20: largest FMI among regression coefficients {fmax:.3f} "
           f"({names[int(np.argmax(pp.fmi[:-2]))]}); von Hippel's rule for CV(SE) = 0.05 "
           f"gives m = {m_rule}; using m = {M}")

imps = mice(A, methods, m=M, maxit=10, seed=[2009, 1])
Q, U = zip(*(analyse(Z) for Z in imps))
P = pool(Q, U, N - len(ANALYSIS) - 1)
lo, hi = P.ci()
out.append(f"\nComplete cases n = {len(cc)} vs MI n = {N} (m = {M})")
for i, k in enumerate(names):
    cc_se = np.sqrt(var_cc[i])
    out.append(f"  {k:10s} CC {est_cc[i]:8.3f} (SE {cc_se:.3f}) | MI {P.est[i]:8.3f} "
               f"(SE {P.se[i]:.3f}; 95% CI {lo[i]:.3f} to {hi[i]:.3f}; FMI {P.fmi[i]:.3f}; "
               f"SE ratio MI/CC {P.se[i] / cc_se:.2f})")
out.append(f"  available-case mean SBP {s_avail.mean():.2f}, share >= 140 "
           f"{(s_avail >= 140).mean():.4f} (n = {len(s_avail)})")
pd.DataFrame({"name": names, "cc": est_cc, "cc_se": np.sqrt(var_cc), "mi": P.est,
              "mi_se": P.se, "fmi": P.fmi}).to_csv(RES / "04_estimates.csv", index=False)


# ------------------------------------------------- 4. delta adjustment
MD = 50   # more imputations for the sensitivity curves, so their Monte Carlo noise is small


def run_delta(col, delta, bounds=None):
    imps = mice(A, methods, m=MD, maxit=10, seed=[2009, 2], delta={ix[col]: delta},
                bounds={ix[col]: bounds} if bounds else None)
    Q, U = zip(*(analyse(Z) for Z in imps))
    return pool(Q, U, N - len(ANALYSIS) - 1)


ipov, ihyp, imean = names.index("poverty"), names.index("SBP>=140"), names.index("mean SBP")
sbp_grid = [0, 2.5, 5, 7.5, 10, 15, 20]
pov_grid = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
rows = []
out.append("\nDelta adjustment on SBP (mmHg added to every imputed SBP):")
P50 = run_delta("sbp", 0.0)
out.append(f"(sensitivity runs use m = {MD}; MAR reference re-run with m = {MD}: share >= 140 "
           f"{P50.est[ihyp]:.4f}, poverty coef {P50.est[ipov]:.3f})")
for dl in sbp_grid:
    p_ = P50 if dl == 0 else run_delta("sbp", dl)
    l_, h_ = p_.ci()
    rows.append(("sbp", dl, p_.est[ihyp], l_[ihyp], h_[ihyp], p_.est[ipov], l_[ipov], h_[ipov],
                 p_.est[imean]))
    out.append(f"  delta {dl:5.1f}: mean SBP {p_.est[imean]:.2f}; share >= 140 "
               f"{p_.est[ihyp]:.4f} ({l_[ihyp]:.4f}-{h_[ihyp]:.4f}); poverty coef "
               f"{p_.est[ipov]:.3f} ({l_[ipov]:.3f} to {h_[ipov]:.3f})")
out.append("Delta adjustment on the poverty ratio (added to every imputed value, clipped to [0, 5]):")
for dl in pov_grid:
    p_ = P50 if dl == 0 else run_delta("poverty", dl, (0.0, 5.0))
    l_, h_ = p_.ci()
    rows.append(("poverty", dl, p_.est[ihyp], l_[ihyp], h_[ihyp], p_.est[ipov], l_[ipov],
                 h_[ipov], p_.est[imean]))
    out.append(f"  delta {dl:+.1f}: poverty coef {p_.est[ipov]:.3f} ({l_[ipov]:.3f} to "
               f"{h_[ipov]:.3f}); FMI {p_.fmi[ipov]:.3f}")
top = (d.poverty == 5).mean()
out.append(f"Observed poverty ratios at the top code 5.0: {top:.1%}; at 0: {(d.poverty == 0).mean():.1%}")
dd = pd.DataFrame(rows, columns=["var", "delta", "hyp", "hyp_lo", "hyp_hi", "pov", "pov_lo",
                                 "pov_hi", "mean_sbp"])
dd.to_csv(RES / "04_delta.csv", index=False)
cc_hyp = est_cc[ihyp]
write("04_nhanes_real.txt", "\n".join(out))

# figure
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
x = dd[dd["var"] == "sbp"]
ax = axes[0]
ax.fill_between(x.delta, 100 * x.hyp_lo, 100 * x.hyp_hi, color=BLUE, alpha=0.18, lw=0)
ax.plot(x.delta, 100 * x.hyp, "o-", color=BLUE, ms=5, label="MI with delta adjustment")
ax.axhline(100 * cc_hyp, color=ORANGE, lw=1.6, ls="--", label="Complete cases")
ax.set_xlabel("mmHg added to every imputed SBP (0 = MAR)")
ax.set_ylabel("Share with SBP ≥ 140 mmHg (%)")
ax.set_title("A prevalence moves with the MNAR assumption")
ax.legend(loc="upper left")
x = dd[dd["var"] == "poverty"]
ax = axes[1]
ax.fill_between(x.delta, x.pov_lo, x.pov_hi, color=BLUE, alpha=0.18, lw=0)
ax.plot(x.delta, x.pov, "o-", color=BLUE, ms=5, label="MI with delta adjustment")
ax.axhline(est_cc[ipov], color=ORANGE, lw=1.6, ls="--", label="Complete cases")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("Shift added to every imputed income-to-poverty ratio (0 = MAR)")
ax.set_ylabel("SBP difference per unit of poverty ratio (mmHg)")
ax.set_title("A regression coefficient barely does")
ax.legend(loc="lower right")
fig.savefig(FIG / "fig5_nhanes_delta.png"); plt.close(fig)
