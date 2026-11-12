"""Experiment 7 - prediction intervals when the truth is known: who gets the coverage.

Model: y = 1 + x + sigma(x) * eps, x drawn with replacement from the incomes of
single-person households in wooldridge::k401ksubs ($1,000s), sigma scaled so
that E sigma^2 = 1. Four scenarios:
  A  constant variance, normal errors
  B  SD proportional to x, normal errors
  C  SD proportional to x, Student t errors with 4 degrees of freedom
     (scaled to unit variance): heavy tails
  D  SD peaks at incomes near 40 and is flat elsewhere, normal errors: a
     variance function the power model cannot represent
Fitting samples of n = 100 and n = 1,000, calibration samples of n / 2, 300
replications. Six 90% intervals (as in 06_ames_intervals.py): OLS normal
theory, split conformal, location-scale (WLS mean, power variance function
log e^2 = g0 + g1 log x, normal quantile), scaled conformal (the same mean and
sigma(x), calibrated), linear quantile regression at 5% and 95%, and
conformalised quantile regression.

Because the truth is known, coverage at each x is computed exactly,
P(lo(x) <= y <= hi(x) | x), on a grid of 2,000 incomes drawn from the same
design; reported are its mean (marginal coverage), its mean within deciles of
x, the worst decile, and the mean width relative to the oracle interval
mu(x) +/- q sigma(x).

Run:  python 07_simulation_intervals.py      (about 5 minutes on 2 cores)
"""

import warnings
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, RES, VIOLET, YELLOW, fetch, style, write
from het import ols, quantreg, wls

warnings.filterwarnings("ignore")
ALPHA = 0.10
REPS = 300
NS = (100, 1000)
SCEN = ("A", "B", "C", "D")
SNAME = {"A": "A: constant variance", "B": "B: SD proportional to x", "C": "C: SD proportional to x, t(4) errors",
         "D": "D: SD peaks at incomes near 40"}
METHODS = ["OLS normal theory", "Split conformal", "Location-scale", "Scaled conformal", "Quantile regression",
           "CQR"]
k = fetch("wooldridge", "k401ksubs")
INCOME = k[k.fsize == 1].inc.values.astype(float)
EDGES = np.quantile(INCOME, np.linspace(0, 1, 11)[1:-1])
T_SCALE = np.sqrt(4 / 2)                     # sd of t(4); errors divided by it have unit variance
LOGCHI = -stats.chi2(1).expect(np.log)


def sd_raw(s, x):
    if s == "A":
        return np.ones_like(x)
    if s in ("B", "C"):
        return x
    return 1 + 3 * np.exp(-0.5 * ((x - 40) / 8) ** 2)


SCALE = {s: np.sqrt(np.mean(sd_raw(s, INCOME) ** 2)) for s in SCEN}


def sigma(s, x):
    return sd_raw(s, x) / SCALE[s]


def draw_err(s, n, rng):
    return rng.standard_t(4, size=n) / T_SCALE if s == "C" else rng.standard_normal(n)


def cdf(s, z):
    return stats.t.cdf(z * T_SCALE, 4) if s == "C" else stats.norm.cdf(z)


def ppf(s, p):
    return stats.t.ppf(p, 4) / T_SCALE if s == "C" else stats.norm.ppf(p)


def cq(scores):
    return np.sort(scores)[int(np.ceil((len(scores) + 1) * (1 - ALPHA))) - 1]


def one(task):
    s, n, rep = task
    rng = np.random.default_rng([7, SCEN.index(s), NS.index(n), rep])
    m = n // 2
    x = rng.choice(INCOME, n + m, replace=True)
    y = 1 + x + sigma(s, x) * draw_err(s, n + m, rng)
    tr, ca = np.arange(n), np.arange(n, n + m)
    xg = rng.choice(INCOME, 2000, replace=True)
    Xa = np.column_stack([np.ones(n + m), x])
    Xg = np.column_stack([np.ones(len(xg)), xg])
    f = ols(Xa[tr], y[tr])
    s_hat = np.sqrt(np.sum(f.resid ** 2) / f.df)
    yh, yhg = Xa @ f.coef, Xg @ f.coef
    hg = np.einsum("ij,jk,ik->i", Xg, f.XtX_inv, Xg)
    D = np.column_stack([np.ones(n), np.log(x[tr])])
    g = np.linalg.lstsq(D, np.log(f.resid ** 2 + 1e-12), rcond=None)[0]
    sig_a = np.sqrt(np.exp(g[0] + g[1] * np.log(x) + LOGCHI))
    sig_g = np.sqrt(np.exp(g[0] + g[1] * np.log(xg) + LOGCHI))
    fw = wls(Xa[tr], y[tr], 1 / sig_a[tr] ** 2)
    yw, ywg = Xa @ fw.coef, Xg @ fw.coef
    ints = {}
    tq = stats.t.ppf(1 - ALPHA / 2, f.df)
    ints["OLS normal theory"] = (yhg - tq * s_hat * np.sqrt(1 + hg), yhg + tq * s_hat * np.sqrt(1 + hg))
    q = cq(np.abs(y[ca] - yh[ca]))
    ints["Split conformal"] = (yhg - q, yhg + q)
    z = stats.norm.ppf(1 - ALPHA / 2)
    ints["Location-scale"] = (ywg - z * sig_g, ywg + z * sig_g)
    qs = cq(np.abs(y[ca] - yw[ca]) / sig_a[ca])
    ints["Scaled conformal"] = (ywg - qs * sig_g, ywg + qs * sig_g)
    b_lo, b_hi = quantreg(Xa[tr], y[tr], ALPHA / 2), quantreg(Xa[tr], y[tr], 1 - ALPHA / 2)
    ints["Quantile regression"] = (Xg @ b_lo, Xg @ b_hi)
    qc = cq(np.maximum(Xa[ca] @ b_lo - y[ca], y[ca] - Xa[ca] @ b_hi))
    ints["CQR"] = (Xg @ b_lo - qc, Xg @ b_hi + qc)
    mu, sg = 1 + xg, sigma(s, xg)
    oracle_w = np.mean(2 * ppf(s, 1 - ALPHA / 2) * sg)
    dec = np.digitize(xg, EDGES)
    rows = []
    for meth, (lo, hi) in ints.items():
        hi = np.maximum(hi, lo)
        cc = cdf(s, (hi - mu) / sg) - cdf(s, (lo - mu) / sg)
        rows.append((s, n, rep, meth, cc.mean(), np.mean(hi - lo) / oracle_w,
                     *[cc[dec == d].mean() if np.any(dec == d) else np.nan for d in range(10)]))
    return rows


if __name__ == "__main__":
    tasks = [(s, n, r) for s in SCEN for n in NS for r in range(REPS)]
    with Pool(2) as pool:
        res = pool.map(one, tasks, chunksize=20)
    df = pd.DataFrame([r for rr in res for r in rr],
                      columns=["scenario", "n", "rep", "method", "coverage", "width_ratio", *[f"d{d}" for d in range(10)]])
    S = df.groupby(["scenario", "n", "method"], sort=False).mean(numeric_only=True).drop(columns="rep").reset_index()
    S["worst_decile"] = S[[f"d{d}" for d in range(10)]].min(axis=1)
    S.to_csv(RES / "07_intervals_summary.csv", index=False)
    out = [f"90% prediction intervals, exact conditional coverage on known truth ({REPS} replications per cell; "
           f"fitting n, calibration n/2)"]
    for (s, n), t in S.groupby(["scenario", "n"], sort=False):
        out.append(f"-- {SNAME[s]}, n {n}")
        for _, r in t.iterrows():
            out.append(f"   {r.method:20s} coverage {r.coverage:.3f}; worst x-decile {r.worst_decile:.3f}; width / "
                       f"oracle {r.width_ratio:.3f}; by decile of x " + " ".join(f"{r[f'd{d}']:.3f}" for d in range(10)))
    write("07_simulation_intervals.txt", "\n".join(out))

    # --------------------------------------------------------------- figure
    style()
    COL = {"OLS normal theory": INK2, "Split conformal": BLUE, "Location-scale": AQUA, "Scaled conformal": ORANGE,
           "Quantile regression": YELLOW, "CQR": VIOLET}
    MK = {"OLS normal theory": "o", "Split conformal": "s", "Location-scale": "^", "Scaled conformal": "D",
          "Quantile regression": "v", "CQR": "P"}
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.2), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, s in zip(axes, SCEN):
        t = S[(S.scenario == s) & (S.n == 1000)]
        for meth in METHODS:
            r = t[t.method == meth].iloc[0]
            ax.plot(range(1, 11), [r[f"d{d}"] for d in range(10)], marker=MK[meth], color=COL[meth], lw=1.5, ms=4,
                    label=meth)
        ax.axhline(1 - ALPHA, color=INK, lw=0.8, ls=":")
        ax.set_title(SNAME[s].replace("proportional to x", "∝ x"), fontsize=10)
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel("Decile of income x")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_ylabel("Coverage at x of a 90% interval\n(n = 1,000, exact, mean of 300 samples)")
    axes[0].set_ylim(0.55, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.savefig(FIG / "fig8_interval_simulation.png"); plt.close(fig)
