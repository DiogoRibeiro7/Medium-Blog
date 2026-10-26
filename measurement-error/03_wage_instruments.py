"""Experiment 3 - a mismeasured confounder: ability in a wage regression.

wooldridge::wage2 (Blackburn & Neumark 1992): 935 men from the National
Longitudinal Survey, with log monthly earnings, schooling and two test scores:
IQ and the Knowledge of the World of Work test (KWW). Ability confounds the
return to schooling; IQ and KWW are two noisy indicators of it. If both
measure the same ability with independent errors, either can instrument the
other, and the two estimates of the return to schooling should agree.

Run:  python 03_wage_instruments.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, FIG, INK2, ORANGE, AQUA, VIOLET, fetch, style, write
from me import moment_correct, ols, simex, tsls

warnings.filterwarnings("ignore")
style()
w = fetch("wooldridge", "wage2")
CTRL = ["exper", "tenure", "married", "south", "urban", "black"]
y = w.lwage.values
n = len(w)
out = [f"wage2: n = {n}; controls {', '.join(CTRL)}; corr(IQ, KWW) = {np.corrcoef(w.IQ, w.KWW)[0, 1]:.3f}"]


def D(*cols):
    return np.column_stack([np.ones(n)] + [w[c].values for c in cols])


specs = {"no ability measure": [], "IQ": ["IQ"], "KWW": ["KWW"], "IQ and KWW": ["IQ", "KWW"]}
res = {}
for lab, extra in specs.items():
    f = ols(D("educ", *CTRL, *extra), y)
    res[lab] = (f.coef[1], f.se[1])
    out.append(f"OLS, {lab:20s} educ {f.coef[1]:.4f} (SE {f.se[1]:.4f})"
               + "".join(f"; {c} {f.coef[-len(extra) + i]:.4f} (SE {f.se[-len(extra) + i]:.4f})"
                         for i, c in enumerate(extra)))
iv1 = tsls(y, D("educ", *CTRL, "IQ"), D("educ", *CTRL, "KWW"))
iv2 = tsls(y, D("educ", *CTRL, "KWW"), D("educ", *CTRL, "IQ"))
fs1 = ols(D("educ", *CTRL, "KWW"), w.IQ.values)
fs2 = ols(D("educ", *CTRL, "IQ"), w.KWW.values)
out.append(f"2SLS, IQ instrumented by KWW: educ {iv1.coef[1]:.4f} (SE {iv1.se[1]:.4f}), IQ {iv1.coef[-1]:.4f} "
           f"(SE {iv1.se[-1]:.4f}); first-stage t of KWW {fs1.coef[-1] / fs1.se[-1]:.1f}")
out.append(f"2SLS, KWW instrumented by IQ: educ {iv2.coef[1]:.4f} (SE {iv2.se[1]:.4f}), KWW {iv2.coef[-1]:.4f} "
           f"(SE {iv2.se[-1]:.4f}); first-stage t of IQ {fs2.coef[-1] / fs2.se[-1]:.1f}")

# bootstrap the two IV estimates and their difference
rng = np.random.default_rng(4)
bs = []
for _ in range(2000):
    i = rng.integers(0, n, n)
    wi = w.iloc[i]
    Di = lambda *cols: np.column_stack([np.ones(n)] + [wi[c].values for c in cols])
    a = tsls(wi.lwage.values, Di("educ", *CTRL, "IQ"), Di("educ", *CTRL, "KWW")).coef[1]
    b = tsls(wi.lwage.values, Di("educ", *CTRL, "KWW"), Di("educ", *CTRL, "IQ")).coef[1]
    bs.append((a, b))
bs = np.array(bs)
dif = bs[:, 0] - bs[:, 1]
out.append(f"Bootstrap (2000): IV difference (IQ-by-KWW minus KWW-by-IQ) {iv1.coef[1] - iv2.coef[1]:+.4f}, "
           f"SE {dif.std(ddof=1):.4f}; 95% percentile interval {np.percentile(dif, 2.5):+.4f} to "
           f"{np.percentile(dif, 97.5):+.4f}")

# reliability sweep: correct the IQ coefficient for an assumed reliability
viq = w.IQ.var()
fz = ols(D("educ", *CTRL), w.IQ.values, se=False)
r2_iq = 1 - np.var(w.IQ.values - D("educ", *CTRL) @ fz.coef) / np.var(w.IQ.values)
out.append(f"R^2 of IQ on schooling and the controls: {r2_iq:.3f}; an assumed reliability at or below this "
           f"makes the error variance exceed the variance of IQ the other regressors leave unexplained, and "
           f"the correction is undefined")
lams = np.round(np.arange(np.ceil((r2_iq + 0.02) * 100) / 100, 1.0001, 0.01), 2)
sweep = []
for lam in lams:
    f = moment_correct(w.IQ.values, w[["educ"] + CTRL].values, y, (1 - lam) * viq)
    # moment_correct orders the design as [1, IQ, educ, controls]
    sweep.append((lam, f.coef[2], f.coef[1]))
sweep = pd.DataFrame(sweep, columns=["reliability", "educ", "iq"])
sweep.to_csv(__import__("common").RES / "03_reliability_sweep.csv", index=False)
for lam in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
    r = sweep[np.isclose(sweep.reliability, lam)].iloc[0]
    out.append(f"  IQ reliability {lam:.1f}: corrected educ {r.educ:.4f}, IQ {r.iq:.4f}")
k = np.argmin(np.abs(sweep.educ - iv1.coef[1]))
out.append(f"Reliability of IQ at which the corrected educ coefficient equals the IV estimate "
           f"{iv1.coef[1]:.4f}: {sweep.reliability[k]:.2f} (IQ coefficient there {sweep.iq[k]:.4f})")
k2 = np.argmin(np.abs(sweep.iq - iv1.coef[-1]))
out.append(f"Reliability at which the corrected IQ coefficient equals its IV estimate {iv1.coef[-1]:.4f}: "
           f"{sweep.reliability[k2]:.2f}")
sx = simex(w.IQ.values, w[["educ"] + CTRL].values, y, 0.1 * viq, B=200, rng=5, extrapolant="rational")
out.append(f"SIMEX (rational) with IQ reliability 0.9: educ {sx.coef[2]:.4f}, IQ {sx.coef[1]:.4f}")
write("03_wage_instruments.txt", "\n".join(out))

# --------------------------------------------------------------- figure 3
fig, ax = plt.subplots(figsize=(9.2, 4.8))
ax.plot(sweep.reliability, sweep.educ, color=BLUE, lw=2.2,
        label="OLS with IQ, corrected for an assumed reliability of IQ")
for (lab, val, col, ls) in (("OLS without an ability measure", res["no ability measure"][0], INK2, ":"),
                            ("2SLS, IQ instrumented by KWW", iv1.coef[1], ORANGE, "--"),
                            ("2SLS, KWW instrumented by IQ", iv2.coef[1], VIOLET, "--")):
    ax.axhline(val, color=col, ls=ls, lw=1.6, label=lab)
ax.axhspan(iv1.coef[1] - 1.96 * iv1.se[1], iv1.coef[1] + 1.96 * iv1.se[1], color=ORANGE, alpha=0.08,
           label="95% CI of the IQ-by-KWW estimate")
ax.plot(1.0, res["IQ"][0], "o", color=BLUE)
ax.annotate("OLS with IQ as measured", (1.0, res["IQ"][0]), xytext=(-8, 10), textcoords="offset points",
            ha="right", fontsize=9, color=INK2)
ax.set_xlabel("Assumed reliability of IQ as a measure of the confounder")
ax.set_ylabel("Return to a year of schooling (log points)")
ax.set_title("How much of the return to schooling is ability depends on how well IQ measures it")
ax.set_xlim(r2_iq, 1.02)
ax.axvspan(r2_iq, lams[0], color=INK2, alpha=0.12)
ax.set_ylim(-0.02, 0.1)
ax.legend(fontsize=8.5, loc="upper left")
fig.savefig(FIG / "fig3_wage_reliability.png"); plt.close(fig)
