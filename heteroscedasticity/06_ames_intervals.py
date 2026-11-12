"""Case 6 - the variance as information: prediction intervals and a review budget for house valuations.

modeldata::ames (De Cock 2011): 2,930 residential sales in Ames, Iowa,
2006-2010; kept are the 2,413 normal sales. The mean model is a hedonic
regression of log sale price on log living area, year built, year remodelled,
log lot area, basement area, garage size, overall condition (1-9), full baths,
fireplaces, building type and neighbourhood (neighbourhoods with fewer than
30 sales pooled), 34 coefficients.

200 random splits: half the sales to fit, a quarter to calibrate, a quarter
to test. Seven 90% prediction intervals, all fitted on the same half:
  Dollars, constant      OLS on the sale price in dollars, split conformal
                          (calibration quantile of |residual|)
  Log, normal theory     OLS on log price, yhat +/- t s sqrt(1 + h)
  Log, conformal         split conformal on log price
  Log, location-scale    WLS on log price with weights 1/sigma^2(x); sigma^2(x)
                          from a regression of log e^2 / (1 - h) (squared
                          standardised OLS residuals) on the fitted value, its
                          square, year built and log leverage (an exponential
                          variance function); yhat +/- z sigma(x)
  Log, scaled conformal  the same mean and sigma(x), calibrated: the
                          calibration quantile of |y - yhat| / sigma(x)
                          (locally weighted conformal, Lei et al. 2018)
  Log, quantile reg.     linear quantile regressions at 5% and 95%
  Log, CQR               the same, conformalised (Romano, Patterson and Candes 2019)
Reported on the test quarter: coverage overall and within groups (predicted
price quintile, year built, quintile of sigma(x)), width, and the interval
score of Gneiting and Raftery (2007).

The review budget: a valuer can check by hand only a share of the houses.
The point valuation is exp(yhat) from the log OLS fit; a valuation is "off" if
it misses the sale price by more than 20%. Houses are ranked for review by
(a) chance, (b) sqrt(1 + h), the only thing that varies in the classical
interval, (c) sigma(x) from the variance model, (d) the width of the
quantile-regression interval; reported is the share of off valuations that
the reviewed houses contain.

Run:  python 06_ames_intervals.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, RES, VIOLET, YELLOW, fetch, style, write
from het import ols, quantreg, wls

warnings.filterwarnings("ignore")
style()
ALPHA = 0.10
SPLITS = 200
OFF = 0.20
a = fetch("modeldata", "ames")
n_all = len(a)
a = a[a.Sale_Condition == "Normal"].reset_index(drop=True)
COND = {"Very_Poor": 1, "Poor": 2, "Fair": 3, "Below_Average": 4, "Average": 5, "Above_Average": 6, "Good": 7,
        "Very_Good": 8, "Excellent": 9, "Very_Excellent": 10}
a["cond"] = a.Overall_Cond.map(COND)
counts = a.Neighborhood.value_counts()
a["nb"] = np.where(a.Neighborhood.map(counts) >= 30, a.Neighborhood, "Other")
num = np.column_stack([np.log(a.Gr_Liv_Area), a.Year_Built, a.Year_Remod_Add, np.log(a.Lot_Area),
                       a.Total_Bsmt_SF / 1000, a.Garage_Cars, a.cond, a.Full_Bath, a.Fireplaces])
X = np.column_stack([np.ones(len(a)), num, pd.get_dummies(a.Bldg_Type, drop_first=True).astype(float).values,
                     pd.get_dummies(a.nb, drop_first=True).astype(float).values])
price = a.Sale_Price.values.astype(float)
y = np.log(price)
yb = a.Year_Built.values
YB_BANDS = ["before 1940", "1940-1979", "1980 or later"]
yb_group = np.digitize(yb, [1940, 1980])
LOGCHI = -stats.chi2(1).expect(np.log)            # -E log chi2_1 = 1.2704: puts exp(Z g) on the variance scale

out = [f"ames: {n_all} sales, {len(a)} normal sales; {a.nb.nunique()} neighbourhood groups "
       f"({(a.nb == 'Other').sum()} sales pooled as Other); {X.shape[1]} coefficients; sale price median "
       f"{np.median(price):,.0f}, range {price.min():,.0f}-{price.max():,.0f}"]

# ---------------------------------------------------- whole-sample description
f_log = ols(X, y)
f_usd = ols(X, price)
qfit = pd.qcut(X @ f_log.coef, 5, labels=False)
out.append("Whole sample, OLS residual SD by quintile of the fitted value: log scale " + ", ".join(
    f"{f_log.resid[qfit == i].std(ddof=1):.3f}" for i in range(5)) + "; dollar scale " + ", ".join(
    f"{f_usd.resid[pd.qcut(X @ f_usd.coef, 5, labels=False) == i].std(ddof=1):,.0f}" for i in range(5)))
out.append("Whole sample, log-scale residual SD by year built: " + ", ".join(
    f"{YB_BANDS[i]} {f_log.resid[yb_group == i].std(ddof=1):.3f} (n {np.sum(yb_group == i)})" for i in range(3)))
out.append(f"Whole sample, log-scale residuals: excess kurtosis {stats.kurtosis(f_log.resid):.2f}; "
           f"share of valuations exp(yhat) off by more than {OFF:.0%}: "
           f"{np.mean(np.abs(np.exp(X @ f_log.coef) - price) / price > OFF):.3f}")


def zvar(yh, lev, idx):
    return np.column_stack([np.ones(len(idx)), yh[idx], yh[idx] ** 2, (yb[idx] - 1970) / 10, np.log(lev[idx])])


def cq(scores, level=1 - ALPHA):
    k = int(np.ceil((len(scores) + 1) * level))
    return np.sort(scores)[k - 1]


def iscore(lo, hi, yy):
    return (hi - lo) + 2 / ALPHA * np.maximum(lo - yy, 0) + 2 / ALPHA * np.maximum(yy - hi, 0)


METHODS = ["Dollars, constant", "Log, normal theory", "Log, conformal", "Log, location-scale",
           "Log, scaled conformal", "Log, quantile reg.", "Log, CQR"]
BUDGETS = np.round(np.arange(0.0, 0.51, 0.025), 3)
RANKS = ["Chance", "Leverage, sqrt(1 + h)", "Variance model, sigma(x)", "Quantile-regression width"]
rng = np.random.default_rng(2024)
rows, cap_rows, var_coef = [], [], []
for sp in range(SPLITS):
    perm = rng.permutation(len(a))
    n = len(a)
    tr, ca, te = perm[: n // 2], perm[n // 2: 3 * n // 4], perm[3 * n // 4:]
    yt = y[te]
    f = ols(X[tr], y[tr])
    yh = X @ f.coef
    s = np.sqrt(np.sum(f.resid ** 2) / f.df)
    lev = np.einsum("ij,jk,ik->i", X, f.XtX_inv, X)          # x'(X'X)^-1 x with X'X from the fitting half
    h_te = lev[te]
    # variance model on the standardised OLS residuals of the fitting half, then WLS
    g = np.linalg.lstsq(zvar(yh, lev, tr), np.log(f.resid ** 2 / (1 - lev[tr]) + 1e-12), rcond=None)[0]
    var_coef.append(g)
    sig = np.sqrt(np.exp(zvar(yh, lev, np.arange(n)) @ g + LOGCHI))
    fw = wls(X[tr], y[tr], 1 / sig[tr] ** 2)
    yw = X @ fw.coef
    ints = {}
    fu = ols(X[tr], price[tr])
    pu = X @ fu.coef
    qu = cq(np.abs(price[ca] - pu[ca]))
    ints["Dollars, constant"] = (np.log(np.maximum(pu[te] - qu, 1.0)), np.log(pu[te] + qu))
    tq = stats.t.ppf(1 - ALPHA / 2, f.df)
    ints["Log, normal theory"] = (yh[te] - tq * s * np.sqrt(1 + h_te), yh[te] + tq * s * np.sqrt(1 + h_te))
    q = cq(np.abs(y[ca] - yh[ca]))
    ints["Log, conformal"] = (yh[te] - q, yh[te] + q)
    z = stats.norm.ppf(1 - ALPHA / 2)
    ints["Log, location-scale"] = (yw[te] - z * sig[te], yw[te] + z * sig[te])
    qs = cq(np.abs(y[ca] - yw[ca]) / sig[ca])
    ints["Log, scaled conformal"] = (yw[te] - qs * sig[te], yw[te] + qs * sig[te])
    b_lo, b_hi = quantreg(X[tr], y[tr], ALPHA / 2), quantreg(X[tr], y[tr], 1 - ALPHA / 2)
    lo_q, hi_q = X @ b_lo, X @ b_hi
    ints["Log, quantile reg."] = (lo_q[te], hi_q[te])
    qc = cq(np.maximum(lo_q[ca] - y[ca], y[ca] - hi_q[ca]))
    ints["Log, CQR"] = (lo_q[te] - qc, hi_q[te] + qc)
    # groups on the test quarter, cut points from the fitting half
    g_fit = np.digitize(yh[te], np.quantile(yh[tr], [0.2, 0.4, 0.6, 0.8]))
    g_sig = np.digitize(sig[te], np.quantile(sig[tr], [0.2, 0.4, 0.6, 0.8]))
    for m, (lo, hi) in ints.items():
        cov = (yt >= lo) & (yt <= hi)
        r = {"split": sp, "method": m, "coverage": cov.mean(), "width_log": np.mean(hi - lo),
             "width_usd": np.mean(np.exp(hi) - np.exp(lo)), "interval_score": np.mean(iscore(lo, hi, yt))}
        for i in range(5):
            r[f"cov_price_q{i + 1}"] = cov[g_fit == i].mean()
            r[f"cov_sigma_q{i + 1}"] = cov[g_sig == i].mean()
            r[f"width_usd_price_q{i + 1}"] = np.mean((np.exp(hi) - np.exp(lo))[g_fit == i])
        for i in range(3):
            r[f"cov_yb{i}"] = cov[yb_group[te] == i].mean()
        rows.append(r)
    # review budget
    off = np.abs(np.exp(yh[te]) - price[te]) / price[te] > OFF
    scores = {"Leverage, sqrt(1 + h)": np.sqrt(1 + h_te), "Variance model, sigma(x)": sig[te],
              "Quantile-regression width": hi_q[te] - lo_q[te]}
    for rk, sc in scores.items():
        order = np.argsort(-sc, kind="stable")
        cum = np.r_[0, np.cumsum(off[order])]
        for b in BUDGETS:
            k = int(round(b * len(te)))
            cap_rows.append((sp, rk, b, cum[k] / max(off.sum(), 1), off.mean()))
    for b in BUDGETS:
        cap_rows.append((sp, "Chance", b, b, off.mean()))

D = pd.DataFrame(rows)
C = pd.DataFrame(cap_rows, columns=["split", "rank", "budget", "captured", "off_rate"])
summ = D.groupby("method", sort=False).mean(numeric_only=True).drop(columns="split").reindex(METHODS)
summ.to_csv(RES / "06_ames_intervals_summary.csv")
G = np.array(var_coef)
out.append(f"Variance model, log sigma^2 = g0 + g1 yhat + g2 yhat^2 + g3 (year built - 1970)/10 + g4 log h; median "
           f"over splits: {np.round(np.median(G, axis=0), 3)}; g4 positive in {np.mean(G[:, 4] > 0):.0%} of splits")
out.append(f"{SPLITS} splits (fit {len(a) // 2}, calibrate {len(a) * 3 // 4 - len(a) // 2}, test "
           f"{len(a) - len(a) * 3 // 4}); 90% intervals; means over splits:")
for m in METHODS:
    r = summ.loc[m]
    out.append(f"-- {m}: coverage {r.coverage:.3f}; mean width {r.width_log:.3f} log points, "
               f"{r.width_usd:,.0f} dollars; interval score {r.interval_score:.4f}")
    out.append("   coverage by predicted-price quintile " + ", ".join(f"{r[f'cov_price_q{i}']:.3f}" for i in range(1, 6))
               + "; by year built " + ", ".join(f"{YB_BANDS[i]} {r[f'cov_yb{i}']:.3f}" for i in range(3))
               + "; by sigma(x) quintile " + ", ".join(f"{r[f'cov_sigma_q{i}']:.3f}" for i in range(1, 6)))
    out.append("   mean width in dollars by predicted-price quintile " + ", ".join(
        f"{r[f'width_usd_price_q{i}']:,.0f}" for i in range(1, 6)))
    worst = min([r[f"cov_price_q{i}"] for i in range(1, 6)] + [r[f"cov_yb{i}"] for i in range(3)]
                + [r[f"cov_sigma_q{i}"] for i in range(1, 6)])
    out.append(f"   worst group coverage {worst:.3f}")
base = summ.loc["Log, conformal", "interval_score"]
out.append("Interval score relative to log conformal: " + "; ".join(
    f"{m} {summ.loc[m, 'interval_score'] / base:.3f}" for m in METHODS))
CS = C.groupby(["rank", "budget"]).captured.mean().unstack(0)
out.append(f"Review budget: valuations off by more than {OFF:.0%}: {C.off_rate.mean():.3f} of test houses on average")
for b in (0.05, 0.10, 0.20, 0.30):
    out.append(f"   review {b:.0%} of houses: share of off valuations caught: " + "; ".join(
        f"{rk} {CS.loc[b, rk]:.3f}" for rk in RANKS))
CS.to_csv(RES / "06_ames_review_budget.csv")
write("06_ames_intervals.txt", "\n".join(out))

# --------------------------------------------------------------- figure: conditional coverage
pct = FuncFormatter(lambda v, _: f"{v:.0%}")
COL = {"Dollars, constant": INK2, "Log, conformal": BLUE, "Log, scaled conformal": ORANGE,
       "Log, quantile reg.": YELLOW, "Log, CQR": VIOLET}
MK = {"Dollars, constant": "o", "Log, conformal": "s", "Log, scaled conformal": "D", "Log, quantile reg.": "v",
      "Log, CQR": "^"}
LAB = {"Dollars, constant": "Constant width in dollars (conformal)", "Log, conformal": "Constant width in logs (conformal)",
       "Log, scaled conformal": "Width from the variance model (scaled conformal)",
       "Log, quantile reg.": "Linear quantile regression, 5%-95%", "Log, CQR": "Conformalised quantile regression"}
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True,
                         gridspec_kw={"width_ratios": [5, 3, 5], "wspace": 0.1})
panels = [("cov_price_q{}", range(1, 6), ["1\ncheapest", "2", "3", "4", "5\ndearest"], "Predicted-price quintile"),
          ("cov_yb{}", range(3), ["before\n1940", "1940-\n1979", "1980 or\nlater"], "Year built"),
          ("cov_sigma_q{}", range(1, 6), ["1\nlowest", "2", "3", "4", "5\nhighest"],
           "Quintile of the modelled spread σ(x)")]
for ax, (key, idx, ticks, xl) in zip(axes, panels):
    for m in COL:
        vals = [summ.loc[m, key.format(i)] for i in idx]
        ax.plot(range(len(vals)), vals, marker=MK[m], color=COL[m], lw=1.6, ms=5, label=LAB[m])
    ax.axhline(1 - ALPHA, color=INK, lw=0.8, ls=":")
    ax.set_xticks(range(len(ticks)), ticks, fontsize=9)
    ax.set_xlabel(xl)
    ax.yaxis.set_major_formatter(pct)
axes[0].set_ylabel("Coverage of a 90% prediction interval\n(test houses, mean of 200 splits)")
axes[0].set_title("Same average coverage, different houses left out", loc="left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.04), frameon=False)
fig.savefig(FIG / "fig7_ames_coverage.png"); plt.close(fig)

# --------------------------------------------------------------- figure: review budget
fig, ax = plt.subplots(figsize=(6.8, 4.6))
RC = {"Chance": INK2, "Leverage, sqrt(1 + h)": YELLOW, "Variance model, sigma(x)": ORANGE,
      "Quantile-regression width": VIOLET}
RL = {"Chance": "Review at random", "Leverage, sqrt(1 + h)": "Rank by leverage (the classical interval)",
      "Variance model, sigma(x)": "Rank by the variance model's σ(x)",
      "Quantile-regression width": "Rank by quantile-regression width"}
for rk in RANKS:
    ax.plot(CS.index, CS[rk], color=RC[rk], lw=2, ls="--" if rk == "Chance" else "-", label=RL[rk])
ax.axvline(0.10, color=INK2, lw=0.8, ls=":")
for rk, off_xy in (("Variance model, sigma(x)", (-30, 4)), ("Leverage, sqrt(1 + h)", (9, -4)), ("Chance", (6, -12))):
    v = CS.loc[0.10, rk]
    ax.plot([0.10], [v], "o", color=RC[rk], ms=6)
    ax.annotate(f"{v:.0%}", (0.10, v), xytext=off_xy, textcoords="offset points", fontsize=9, color=RC[rk])
ax.xaxis.set_major_formatter(pct)
ax.yaxis.set_major_formatter(pct)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.9)
ax.set_xlabel("Share of houses a valuer checks by hand")
ax.set_ylabel(f"Share of valuations off by more than {OFF:.0%}\nthat the checked houses contain")
ax.set_title("Ranking by predicted spread finds the bad valuations first", loc="left")
ax.legend(fontsize=8.5, loc="upper left")
fig.savefig(FIG / "fig9_ames_review.png"); plt.close(fig)
