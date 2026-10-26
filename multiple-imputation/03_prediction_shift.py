"""Experiment 3 - predicting with missing values when the missingness changes.

Real CPS 1988 wages. Train and test sets of 5,000 men each, drawn without
replacement, 30 repetitions. In training, 40% of education is missing MNAR
(low education more likely missing). Three deployment scenarios for the test set:

  same     the training mechanism (MNAR-X, 40%)
  MCAR     40% missing completely at random (e.g. a form change)
  complete education always recorded

Run:  python 03_prediction_shift.py      (about 3 minutes)
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from common import BLUE, FIG, INK2, ORANGE, load_cps, style, write

warnings.filterwarnings("ignore")
style()
POP = load_cps()
FEAT = ["educ", "exper", "afam", "smsa", "parttime", "northeast", "midwest", "west"]
NTR = NTE = 5000
REPS = 30


def mnar(e, rng, share=0.4, gamma=1.5):
    z = -(e - e.mean()) / e.std()
    c = optimize.brentq(lambda c: expit(c + gamma * z).mean() - share, -10, 10)
    return rng.random(len(e)) < expit(c + gamma * z)


def lin_design(X, ind=None):
    exper = X[:, 1]
    cols = [X, (exper ** 2 / 100)[:, None]]
    if ind is not None:
        cols.append(ind[:, None])
    return np.hstack(cols)


def condmean_fit(X, miss):
    """Regression of educ on the other features (no outcome), fitted on observed rows."""
    A = np.column_stack([np.ones(len(X)), X[:, 1:]])
    return np.linalg.lstsq(A[~miss], X[~miss, 0], rcond=None)[0]


def condmean_apply(X, miss, b):
    A = np.column_stack([np.ones(len(X)), X[:, 1:]])
    Z = X.copy()
    Z[miss, 0] = A[miss] @ b
    return Z


def hgb(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)


rows = []
for rep in range(REPS):
    rng = np.random.default_rng([3, rep])
    idx = rng.permutation(len(POP))
    tr, te = POP.iloc[idx[:NTR]], POP.iloc[idx[NTR:NTR + NTE]]
    Xtr, ytr = tr[FEAT].values.astype(float), tr.logwage.values
    Xte, yte = te[FEAT].values.astype(float), te.logwage.values
    mtr = mnar(Xtr[:, 0], rng)
    tests = {"same": mnar(Xte[:, 0], rng), "MCAR": rng.random(NTE) < 0.4,
             "complete": np.zeros(NTE, bool)}
    Xtr_nan = Xtr.copy(); Xtr_nan[mtr, 0] = np.nan
    mu = np.nanmean(Xtr_nan[:, 0])
    Xtr_mean = Xtr_nan.copy(); Xtr_mean[mtr, 0] = mu
    bcm = condmean_fit(Xtr, mtr)
    Xtr_cm = condmean_apply(Xtr, mtr, bcm)

    models = {
        "Boosting, native NaN": (hgb(rep).fit(Xtr_nan, ytr), "nan"),
        "Boosting, mean imputed": (hgb(rep).fit(Xtr_mean, ytr), "mean"),
        "Boosting, mean + indicator": (hgb(rep).fit(np.column_stack([Xtr_mean, mtr]), ytr), "mean+ind"),
        "Boosting, regression imputed": (hgb(rep).fit(Xtr_cm, ytr), "cm"),
        "Linear, mean + indicator": (LinearRegression().fit(lin_design(Xtr_mean, mtr), ytr), "lin+ind"),
        "Linear, regression imputed": (LinearRegression().fit(lin_design(Xtr_cm), ytr), "lin+cm"),
    }
    oracle = hgb(rep).fit(Xtr, ytr)
    for scen, mte in tests.items():
        Xn = Xte.copy(); Xn[mte, 0] = np.nan
        Xm = Xn.copy(); Xm[mte, 0] = mu
        Xc = condmean_apply(Xte, mte, bcm)
        feats = {"nan": Xn, "mean": Xm, "mean+ind": np.column_stack([Xm, mte]), "cm": Xc,
                 "lin+ind": lin_design(Xm, mte.astype(float)), "lin+cm": lin_design(Xc)}
        preds = {k: mod.predict(feats[kind]) for k, (mod, kind) in models.items()}
        preds["Oracle (no missing values anywhere)"] = oracle.predict(Xte)
        for k, p in preds.items():
            err = (yte - p) ** 2
            low = Xte[:, 0] < 12
            rows.append((rep, scen, k, np.sqrt(err.mean()),
                         np.sqrt(err[mte].mean()) if mte.any() else np.nan,
                         np.sqrt(err[low].mean()), np.mean(p[mte] - yte[mte]) if mte.any() else np.nan))

d = pd.DataFrame(rows, columns=["rep", "scenario", "model", "rmse", "rmse_missing",
                                "rmse_low_educ", "bias_missing"])
d.to_csv("results/03_prediction.csv.gz", index=False)
g = d.groupby(["scenario", "model"], sort=False).mean(numeric_only=True)
out = [f"Real CPS 1988; train {NTR}, test {NTE}, {REPS} repetitions; training educ 40% "
       f"missing MNAR (low education more likely missing). RMSE of log wage."]
order = list(d.model.unique())
for scen in ("same", "MCAR", "complete"):
    out.append(f"-- test scenario: {scen}")
    for k in order:
        r = g.loc[(scen, k)]
        out.append(f"  {k:38s} RMSE all {r.rmse:.4f}"
                   + (f" | rows with educ missing {r.rmse_missing:.4f} (mean error "
                      f"{r.bias_missing:+.3f})" if not np.isnan(r.rmse_missing) else "")
                   + f" | educ < 12 {r.rmse_low_educ:.4f}")
piv = d.pivot_table(index=["rep", "model"], columns="scenario", values="rmse").reset_index()
for k in order[:-1]:
    x = piv[piv.model == k]
    out.append(f"  {k:38s} RMSE change same -> MCAR {100 * (x.MCAR / x.same - 1).mean():+.1f}%, "
               f"same -> complete {100 * (x.complete / x.same - 1).mean():+.1f}%")
write("03_prediction_shift.txt", "\n".join(out))

fig, ax = plt.subplots(figsize=(9.0, 4.4))
names = [k for k in order if not k.startswith("Oracle")]
ypos = np.arange(len(names))[::-1]
for yi, k in zip(ypos, names):
    a_, b_ = g.loc[("same", k)].bias_missing, g.loc[("MCAR", k)].bias_missing
    ax.plot([a_, b_], [yi, yi], color="#cfcec9", lw=2, zorder=1)
    ax.scatter(a_, yi, s=70, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8,
               label="Test missingness as in training" if yi == ypos[0] else None)
    ax.scatter(b_, yi, s=70, color=ORANGE, zorder=3, edgecolor="white", linewidth=0.8,
               label="Test missingness completely at random" if yi == ypos[0] else None)
ax.axvline(0, color=INK2, lw=0.9)
ax.set_yticks(ypos); ax.set_yticklabels(names)
ax.set_xlabel("Mean prediction error on rows with education missing (log points)")
ax.set_title("A model that learned the missingness is calibrated only while it lasts")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper right", fontsize=9)
fig.savefig(FIG / "fig4_prediction_shift.png"); plt.close(fig)
