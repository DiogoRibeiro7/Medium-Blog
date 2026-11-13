"""Evidence - what the zeros of three classic count data sets say.

For physician office visits in the NMES 1988 (the pscl vignette's example),
articles of PhD biochemists (the classic zero-inflation example) and doctor
visits in the Australian Health Survey 1977-78, the script fits

  Poisson; NB2; NB2 whose dispersion depends on the regressors of the zero part
  ("NB, dispersion varies"); zero-inflated Poisson and NB2; hurdle NB2,

with the same regressors in the zero part of the zero-inflated and hurdle models
and in the dispersion equation (for NMES, the reduced set of the pscl
vignette). It prints, for each data set and model, the number of parameters,
log-likelihood, AIC, BIC, expected numbers of zeros and ones; how much of the
Poisson model's shortfall of zeros the NB already closes; Vuong statistics in
the comparisons usually reported; where the log-likelihood gain of the
zero-inflated and hurdle models over the NB comes from (zeros, ones, larger
counts); how well each model's P(Y = 0) is calibrated within deciles of the NB's
P(Y = 0); the share of zeros the zero-inflated model calls structural; and
decision quantities by model (mean, average effect of one regressor on the mean
and on P(Y > 0)).

Figures: fig2_rootograms.png (hanging rootograms, Poisson and NB),
fig3_where_the_gain_is.png (log-likelihood gain over the NB by count) and
fig4_who_the_zeros_are.png (observed and predicted
shares of zeros by insurance and chronic conditions, NMES).

Run:  python 01_three_datasets.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, RES, VIOLET, YELLOW, style, write
from counts import expected_counts, hurdle, negbin, poisson, vuong, zeroinfl
from data import biochemists, doctorvisits, nmes

warnings.filterwarnings("ignore")
style()

y1, X1, Z1, _ = nmes()
y2, X2 = biochemists()
y3, X3 = doctorvisits()
# data set: (y, X, Z for the zero part and dispersion, focal regressor (column, label, step))
SETS = {
    "NMES 1988, office visits": (y1, X1, Z1, (4, "one more chronic condition", 1.0)),
    "Biochemists, articles": (y2, X2, X2, (5, "ten more mentor articles", 10.0)),
    "Australia 1977-78, doctor visits": (y3, X3, X3, (5, "one more illness in the past two weeks", 1.0)),
}
MODELS = ["Poisson", "NB", "NB, dispersion varies", "ZIP", "ZINB", "Hurdle-NB"]
out = []
FITS = {}


def ame(fit, X, Z, col, step, which="mean"):
    """Average change in E[Y] (or in P(Y > 0)) when column col of X, and the same
    variable in the zero part or dispersion equation, rises by step."""
    Xa = X.copy()
    Xa[:, col] += step
    Za = Z.copy()
    for j in range(Z.shape[1]):
        if np.array_equal(Z[:, j], X[:, col]):
            Za[:, j] += step
    key = {"NB, dispersion varies": "W", "ZIP": "Z", "ZINB": "Z", "Hurdle-NB": "Z"}.get(fit.model)
    kw0, kw1 = ({key: Z}, {key: Za}) if key else ({}, {})
    if which == "mean":
        return float(np.mean(fit.mean(X=Xa, **kw1) - fit.mean(X=X, **kw0)))
    return float(np.mean(fit.pmf(0, X=X, **kw0) - fit.pmf(0, X=Xa, **kw1)))


for name, (y, X, Z, (col, lab, step)) in SETS.items():
    n = len(y)
    fits = {"Poisson": poisson(X, y), "NB": negbin(X, y), "NB, dispersion varies": negbin(X, y, Z),
            "ZIP": zeroinfl(X, y, Z, "poisson"), "ZINB": zeroinfl(X, y, Z), "Hurdle-NB": hurdle(X, y, Z)}
    FITS[name] = fits
    obs0, obs1 = int((y == 0).sum()), int((y == 1).sum())
    out.append(f"== {name}: n {n}; zeros {obs0} ({obs0 / n:.3f}); ones {obs1}; mean {y.mean():.4f}; variance "
               f"{y.var(ddof=1):.4f}; largest count {int(y.max())}")
    for k, f in fits.items():
        out.append(f"   {k:22s} parameters {f.nparams:2d}; log-likelihood {f.loglik:10.2f}; AIC {f.aic:9.1f}; BIC "
                   f"{f.bic(n):9.1f}; expected zeros {f.pmf(0).sum():7.1f}; expected ones {f.pmf(1).sum():7.1f}; "
                   f"converged {f.converged}")
    out.append("   theta of the count distribution (larger means less overdispersed): " + "; ".join(
        f"{k} {np.exp(fits[k].block('dispersion')[0]):.3f}" for k in ("NB", "ZINB", "Hurdle-NB")))
    e0p, e0n = fits["Poisson"].pmf(0).sum(), fits["NB"].pmf(0).sum()
    out.append(f"   Poisson shortfall of zeros {obs0 - e0p:.1f}; the NB closes {e0n - e0p:.1f} of it "
               f"({(e0n - e0p) / (obs0 - e0p):.3f}); NB minus observed zeros {e0n - obs0:+.1f}")
    v = lambda a, b, c=None: "%+.2f (p %.3g)" % vuong(fits[a], fits[b], c)
    out.append(f"   Vuong: ZIP vs Poisson {v('ZIP', 'Poisson')}; ZINB vs NB {v('ZINB', 'NB')}, AIC-corrected "
               f"{v('ZINB', 'NB', 'aic')}, BIC-corrected {v('ZINB', 'NB', 'bic')}; ZINB vs NB with varying dispersion "
               f"{v('ZINB', 'NB, dispersion varies')}; hurdle NB vs ZINB {v('Hurdle-NB', 'ZINB')}")
    out.append(f"   lowest AIC: {min(fits, key=lambda k: fits[k].aic)}; lowest BIC: {min(fits, key=lambda k: fits[k].bic(n))}")
    for k in ("ZINB", "Hurdle-NB", "NB, dispersion varies"):
        g = fits[k].ll_i - fits["NB"].ll_i
        out.append(f"   log-likelihood gain of {k} over the NB: total {g.sum():+.2f}; from the zeros {g[y == 0].sum():+.2f}; "
                   f"from the ones {g[y == 1].sum():+.2f}; from counts of 2 or more {g[y >= 2].sum():+.2f}")
    pi = fits["ZINB"].extra["inflation"]()
    p0 = fits["ZINB"].pmf(0)
    dz = np.abs(p0 - fits["Hurdle-NB"].pmf(0))
    out.append(f"   each person's P(Y = 0), ZINB against hurdle NB: mean |difference| {dz.mean():.4f}, largest {dz.max():.4f}; "
               f"correlation {np.corrcoef(p0, fits['Hurdle-NB'].pmf(0))[0, 1]:.4f}")
    out.append(f"   ZINB structural-zero probability: mean {pi.mean():.4f}, 10th-90th percentile {np.quantile(pi, .1):.4f}-"
               f"{np.quantile(pi, .9):.4f}; structural zeros as a share of the expected zeros {pi.sum() / p0.sum():.3f}")
    # calibration of P(0) within deciles of the NB's P(0)
    q = np.quantile(fits["NB"].pmf(0), np.linspace(0, 1, 11))
    grp = np.clip(np.searchsorted(q, fits["NB"].pmf(0), side="right") - 1, 0, 9)
    cal = {}
    for k in ("Poisson", "NB", "NB, dispersion varies", "ZINB", "Hurdle-NB"):
        p = fits[k].pmf(0)
        err = np.array([p[grp == j].mean() - (y[grp == j] == 0).mean() for j in range(10)])
        cal[k] = err
        out.append(f"   P(Y = 0) within deciles of the NB's P(Y = 0), {k:22s}: largest |predicted - observed| "
                   f"{np.abs(err).max():.4f}; mean |predicted - observed| {np.abs(err).mean():.4f}")
    FITS[name]["_cal"] = (grp, cal)
    # decision quantities
    for k in MODELS:
        f = fits[k]
        out.append(f"   {k:22s} mean of fitted means {f.mean().mean():.4f} (observed {y.mean():.4f}); effect of {lab}: "
                   f"on the mean {ame(f, X, Z, col, step):+.4f}, on P(Y > 0) {ame(f, X, Z, col, step, 'pos'):+.4f}")
write("01_three_datasets.txt", "\n".join(out))

# --------------------------------------------------------------- figure 2: hanging rootograms
KMAX = {"NMES 1988, office visits": 20, "Biochemists, articles": 10, "Australia 1977-78, doctor visits": 9}
fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.4), gridspec_kw={"hspace": 0.55, "wspace": 0.25})
for j, (name, (y, X, Z, _)) in enumerate(SETS.items()):
    km = KMAX[name]
    obs = np.array([(y == k).sum() for k in range(km + 1)])
    for i, model in enumerate(("Poisson", "NB")):
        ax = axes[i, j]
        ex = expected_counts(FITS[name][model], km)
        top = np.sqrt(ex)
        ax.bar(np.arange(km + 1), np.sqrt(obs), bottom=top - np.sqrt(obs), width=0.8, color=BLUE if i else ORANGE,
               alpha=0.55, lw=0)
        ax.plot(np.arange(km + 1), top, "o-", color=INK, ms=3.5, lw=1.4)
        ax.axhline(0, color=INK2, lw=0.8)
        lo, hi = min((top - np.sqrt(obs)).min(), 0), max(top.max(), np.sqrt(obs).max())
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_title(f"{name.split(',')[0]}: {model}", fontsize=10.5)
        ax.set_xlabel("Count")
        if j == 0:
            ax.set_ylabel("√frequency")
        gap0 = np.sqrt(ex[0]) - np.sqrt(obs[0])
        ax.annotate(f"zeros: expected {ex[0]:.0f}\nobserved {obs[0]}", (0, top[0]), xytext=(0.97, 0.95),
                    textcoords="axes fraction", ha="right", va="top", fontsize=8.5, color=INK2)
fig.text(0.5, -0.02, "Hanging rootograms: each bar is the square root of the observed frequency, hung from the square root "
         "of the expected frequency (the dots).\nA bar that ends below zero means the model expects too few of that count; "
         "one that ends above zero, too many.", ha="center", fontsize=8.8, color=INK2)
fig.savefig(FIG / "fig2_rootograms.png")
plt.close(fig)

# --------------------------------------------------------------- figure 3: where the gain is
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), gridspec_kw={"wspace": 0.28})
cats = ["0", "1", "2", "3", "4", "5–9", "10+"]
edges = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 9), (10, 10 ** 6)]
for ax, (name, (y, *_)) in zip(axes, SETS.items()):
    w = 0.36
    for s, (k, col) in enumerate((("ZINB", ORANGE), ("Hurdle-NB", VIOLET))):
        g = FITS[name][k].ll_i - FITS[name]["NB"].ll_i
        vals = [g[(y >= a) & (y <= b)].sum() for a, b in edges]
        ax.bar(np.arange(len(cats)) + (s - 0.5) * w, vals, width=w, color=col, label=f"{k} minus NB")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(cats)), cats)
    ax.set_xlabel("Observed count")
    ax.set_title(name.split(",")[0], fontsize=10.5)
axes[0].set_ylabel("Log-likelihood gain over the NB\n(sum over observations)")
axes[0].legend(fontsize=8.5, loc="lower right")
fig.savefig(FIG / "fig3_where_the_gain_is.png")
plt.close(fig)

# --------------------------------------------------------------- figure 4: who the extra zeros are (NMES)
name = "NMES 1988, office visits"
y, X, Z, _ = SETS[name]
chron = np.minimum(X[:, 4], 5).astype(int)
ins = X[:, 7].astype(bool)
fits = FITS[name]
show = [("NB", BLUE), ("NB, dispersion varies", AQUA), ("ZINB", YELLOW), ("Hurdle-NB", VIOLET)]
out = ["NMES 1988: share of people with no office visit, by private insurance and number of chronic conditions (5 = five "
       "or more): observed (count of people), and predicted by each model"]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, flag, title in zip(axes, (False, True), ("No private insurance", "Private insurance")):
    obs, lo, hi, cnt = [], [], [], []
    for c in range(6):
        m = (chron == c) & (ins == flag)
        k, nn = int((y[m] == 0).sum()), int(m.sum())
        p_ = k / nn
        se = np.sqrt(p_ * (1 - p_) / nn)
        obs.append(p_), lo.append(p_ - 1.96 * se), hi.append(p_ + 1.96 * se), cnt.append(nn)
    preds = {k: [fits[k].pmf(0)[(chron == c) & (ins == flag)].mean() for c in range(6)] for k, _ in show}
    for c in range(6):
        out.append(f"   {title.lower():21s} chronic {c}: observed {obs[c]:.3f} (n {cnt[c]}); " +
                   "; ".join(f"{k} {preds[k][c]:.3f}" for k, _ in show))
    for k, col in show:
        ax.plot(range(6), preds[k], "o-", color=col, ms=4, lw=1.8, label=k)
    ax.errorbar(range(6), obs, yerr=[np.array(obs) - lo, np.array(hi) - obs], fmt="s", color=INK, ms=6, capsize=3,
                lw=1.2, label="Observed (95% interval)", zorder=5)
    ax.set_xticks(range(6), ["0", "1", "2", "3", "4", "5+"])
    ax.set_xlabel("Number of chronic conditions")
    ax.set_title(title, fontsize=10.5)
axes[0].set_ylabel("Share with no physician office visit")
axes[1].legend(fontsize=8.5, loc="upper right")
fig.savefig(FIG / "fig4_who_the_zeros_are.png")
plt.close(fig)
with open(RES / "01_three_datasets.txt", "a") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
