"""Experiment 2 - a validation sample: self-reported and measured weight and height.

carData::Davis (Davis 1990): 200 people who exercise regularly, with measured
weight and height and their own reports of both. The data set is itself a
validation study, so it shows what self-report error looks like, and it
contains a famous recording error in the measured values (row 12: weight 166
kg, height 57 cm, which are the reported 56 kg and 163 cm with the two
measurements swapped).

The analysis question is the allometric exponent in log weight = a + b log
height + sex, which is about 2 if weight scales with height squared (the
premise of the body-mass index). Then a validation design is simulated on
the real data: only a random quarter of the people keep their measured
values, and the correction methods must recover the full measured-data
estimate from the reports of everyone else.

Run:  python 02_davis_validation.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, FIG, INK2, ORANGE, AQUA, fetch, style, write
from me import design, ols

warnings.filterwarnings("ignore")
style()
dv = fetch("carData", "Davis")
c = dv.dropna().reset_index(drop=True)
out = [f"Davis: {len(dv)} people; {dv.repwt.isna().sum()} without reported weight, {dv.repht.isna().sum()} "
       f"without reported height; {len(c)} with all four values"]
r12 = dv[dv.rownames == 12].iloc[0]
out.append(f"Row 12: measured weight {r12.weight} kg, height {r12.height} cm; reported {r12.repwt} kg, {r12.repht} cm")
clean = c[c.rownames != 12].reset_index(drop=True)

for lab, dd in (("with row 12", c), ("without row 12", clean)):
    out.append(f"-- {lab} (n = {len(dd)})")
    for sex, s in dd.groupby("sex"):
        ew, eh = s.repwt - s.weight, s.repht - s.height
        bw = np.polyfit(s.weight, s.repwt, 1)[0]
        bh = np.polyfit(s.height, s.repht, 1)[0]
        out.append(f"  {sex}: n {len(s)}; weight error (reported - measured) mean {ew.mean():+.2f} kg, SD {ew.std(ddof=1):.2f}, "
                   f"slope of reported on measured {bw:.3f}, correlation {np.corrcoef(s.weight, s.repwt)[0, 1]:.4f}; "
                   f"height error mean {eh.mean():+.2f} cm, SD {eh.std(ddof=1):.2f}, slope {bh:.3f}, "
                   f"correlation {np.corrcoef(s.height, s.repht)[0, 1]:.4f}")
    fem = (dd.sex == "F").astype(float).values
    for wlab, (wt, ht) in (("measured weight and height", (dd.weight, dd.height)),
                           ("reported weight and height", (dd.repwt, dd.repht)),
                           ("reported weight, measured height", (dd.repwt, dd.height)),
                           ("measured weight, reported height", (dd.weight, dd.repht))):
        f = ols(design(np.log(ht.values), fem), np.log(wt.values))
        out.append(f"  log weight ~ log height + female, {wlab:34s} exponent {f.coef[1]:.3f} (SE {f.se[1]:.3f})")
    bm, br = dd.weight / (dd.height / 100) ** 2, dd.repwt / (dd.repht / 100) ** 2
    out.append(f"  BMI >= 25: measured {int((bm >= 25).sum())}, reported {int((br >= 25).sum())}, both "
               f"{int(((bm >= 25) & (br >= 25)).sum())}; reported BMI minus measured: mean {(br - bm).mean():+.2f}")


# ------------------------------------------- simulated internal validation
def analyses(dd, val, rng, m=20):
    """Exponent estimates when only rows `val` keep their measured values."""
    lw, lh = np.log(dd.weight.values), np.log(dd.height.values)
    rw, rh = np.log(dd.repwt.values), np.log(dd.repht.values)
    fem = (dd.sex == "F").astype(float).values
    res = {"Reports only (naive)": ols(design(rh, fem), rw, se=False).coef[1],
           "Validation rows only": ols(design(lh[val], fem[val]), lw[val], se=False).coef[1]}
    # regression calibration of height from reported height and sex (the outcome
    # stays out of the calibration model), reported weight as the outcome
    Crc = design(rh, fem)
    g = np.linalg.lstsq(Crc[val], lh[val], rcond=None)[0]
    hh = np.where(val, lh, Crc @ g)
    res["Calibrated height, reported weight"] = ols(design(hh, fem), np.where(val, lw, rw), se=False).coef[1]
    # multiple imputation of both measured values from both reports and sex
    C = design(np.column_stack([rh, rw]), fem)
    est = []
    nv = val.sum()
    for _ in range(m):
        draws = {}
        for target, X in (("h", C), ("w", None)):
            if target == "h":
                yv, Xf = lh, C
            else:
                Xf = np.column_stack([C, draws["h"]])
                yv = lw
            Xv = Xf[val]
            V = np.linalg.inv(Xv.T @ Xv)
            b = V @ Xv.T @ yv[val]
            s2 = np.sum((yv[val] - Xv @ b) ** 2) / rng.chisquare(nv - Xv.shape[1])
            bd = b + np.sqrt(s2) * np.linalg.cholesky(V) @ rng.standard_normal(len(b))
            draws[target] = np.where(val, yv, Xf @ bd + np.sqrt(s2) * rng.standard_normal(len(yv)))
        est.append(ols(design(draws["h"], fem), draws["w"], se=False).coef[1])
    res["Multiple imputation of both"] = np.mean(est)
    return res


rng = np.random.default_rng(12)
SHARE, REPS = 0.25, 2000
rows = []
for lab, dd in (("with row 12", c), ("without row 12", clean)):
    n = len(dd)
    k = int(round(SHARE * n))
    has12 = (dd.rownames == 12).values
    for rep in range(REPS):
        val = np.zeros(n, bool)
        val[rng.choice(n, k, replace=False)] = True
        for meth, b in analyses(dd, val, rng).items():
            rows.append((lab, rep, bool((val & has12).any()), meth, b))
sim = pd.DataFrame(rows, columns=["pool", "rep", "row12_validated", "method", "est"])
target = ols(design(np.log(clean.height.values), (clean.sex == "F").astype(float).values),
             np.log(clean.weight.values), se=False).coef[1]
out.append(f"\nSimulated internal validation: {REPS} random subsets of {SHARE:.0%} keep their measured values; "
           f"target = measured-data exponent without row 12, {target:.3f}")
for (pool, flag, meth), s in sim.groupby(["pool", "row12_validated", "method"], sort=False):
    tag = pool if pool == "without row 12" else f"{pool}, row 12 {'in' if flag else 'not in'} the validation subset"
    q = np.percentile(s.est, [5, 50, 95])
    out.append(f"  {tag:52s} {meth:36s} n {len(s):4d}; median {q[1]:.3f} (5-95%: {q[0]:.3f} to {q[2]:.3f}); "
               f"RMSE vs target {np.sqrt(np.mean((s.est - target) ** 2)):.3f}")
write("02_davis_validation.txt", "\n".join(out))

# --------------------------------------------------------------- figure 2
fig = plt.figure(figsize=(11.6, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.45], wspace=0.55, hspace=0.45)
scat = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
for ax, (m_, r_, unit, title) in zip(scat, (("weight", "repwt", "kg", "Weight"), ("height", "repht", "cm", "Height"))):
    ok = c.rownames != 12
    for sex, col in (("F", ORANGE), ("M", BLUE)):
        s = c[ok & (c.sex == sex)]
        ax.scatter(s[m_], s[r_] - s[m_], s=14, color=col, alpha=0.75, lw=0, label={"F": "Women", "M": "Men"}[sex])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel(f"Measured {m_} ({unit})")
    ax.set_ylabel(f"Reported − measured ({unit})")
    s12 = c[~ok].iloc[0]
    ax.set_title(f"{title} (row 12, off scale: {s12[r_] - s12[m_]:+.0f} {unit})", fontsize=10.5)
scat[0].legend(fontsize=8.5, loc="upper right", ncol=2)
ax = fig.add_subplot(gs[:, 1])
order = ["Reports only (naive)", "Calibrated height, reported weight", "Multiple imputation of both",
         "Validation rows only"]
groups = [("without row 12", None, BLUE, "Row 12 removed"),
          ("with row 12", False, AQUA, "Row 12 kept, not in the validation subset"),
          ("with row 12", True, ORANGE, "Row 12 kept and in the validation subset")]
for gi, (pool, flag, col, lab) in enumerate(groups):
    for mi, meth in enumerate(order):
        s = sim[(sim.pool == pool) & (sim.method == meth)]
        if flag is not None:
            s = s[s.row12_validated == flag]
        q = np.percentile(s.est, [5, 50, 95])
        yv = mi + (gi - 1) * 0.22
        ax.plot([q[0], q[2]], [yv, yv], color=col, lw=2.2)
        ax.plot(q[1], yv, "o", color=col, ms=5, label=lab if mi == 0 else None)
ax.axvline(target, color=INK2, ls="--", lw=0.9)
ax.text(target - 0.05, -0.5, f"measured data, row 12 removed: {target:.2f}", fontsize=8.5, color=INK2, ha="right", va="center")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([o.replace(", ", ",\n") for o in order], fontsize=9)
ax.set_ylim(3.45, -0.68)
ax.set_xlim(-1.2, 3.2)
ax.set_xlabel("Exponent of height: median and 5–95% over 2,000 subsets")
ax.set_title("Correcting from a 25% validation subset")
ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=1)
fig.savefig(FIG / "fig2_davis_validation.png"); plt.close(fig)
