"""Efficiency rankings across analyst choices (specification curve).

Choices: outcome x input x window x method. Ranks are among countries with data in every window.
Usage: python3 -I efficiency_speccurve.py PANEL_CSV OUT_DIR
"""
import sys
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from scipy.stats import kendalltau

panel, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(panel)

WINDOWS = {"2008-10": (2008, 2010), "2013-15": (2013, 2015), "2015-19": (2015, 2019), "2017-19": (2017, 2019)}
OUTCOMES = {  # name: (column, higher_is_better)
    "treatable mortality": ("trtm", False),
    "avoidable mortality": ("avm", False),
    "life expectancy": ("le0", True),
}
INPUTS = {"spend per capita": "hexp_pc_const", "spend % GDP": "hexp_gdp"}
COV = ["gdp_pc", "pop65", "alcohol"]


def window_means(w):
    a, b = WINDOWS[w]
    s = d[d.year.between(a, b)].groupby("iso")[["trtm", "avm", "le0", "hexp_pc_const", "hexp_gdp"] + COV].mean()
    return s


def dea_vrs_output(x, y):
    """Output-oriented VRS DEA, single input/output. Returns efficiency in (0,1]."""
    n = len(x)
    eff = np.empty(n)
    for o in range(n):
        # variables: lambda_1..n, phi ; maximise phi
        c = np.zeros(n + 1); c[-1] = -1
        A_ub = [np.append(-y, y[o]), np.append(x, 0)]  # -sum l*y + phi*y_o <= 0 ; sum l*x <= x_o
        b_ub = [0, x[o]]
        A_eq = [np.append(np.ones(n), 0)]; b_eq = [1]
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n + [(1, None)], method="highs")
        eff[o] = 1 / r.x[-1]
    return eff


def cols(df, ycol, hib, xcol, adjust):
    ly, lx = np.log(df[ycol]), np.log(df[xcol])
    X = pd.DataFrame({"lx": lx})
    if adjust:
        X["lgdp"] = np.log(df.gdp_pc); X["pop65"] = df.pop65; X["alcohol"] = df.alcohol
    r = sm.OLS(ly, sm.add_constant(X)).fit()
    e = r.resid
    return np.exp(e - e.max()) if hib else np.exp(e.min() - e)


METHODS = ["COLS raw", "COLS adjusted", "DEA-VRS"]
ranks, scores = {}, {}
means = {w: window_means(w) for w in WINDOWS}
common = sorted(set.intersection(*[set(m.dropna(subset=["trtm", "avm", "le0", "hexp_pc_const", "hexp_gdp"] + COV).index) for m in means.values()]))
dropped = sorted(set(d.iso) - set(common))
print("countries:", len(common), "dropped:", dropped)

for (on, (ycol, hib)), (inn, xcol), w, m in itertools.product(OUTCOMES.items(), INPUTS.items(), WINDOWS, METHODS):
    df = means[w].loc[common]
    if m == "DEA-VRS":
        y = df[ycol].values if hib else 1 / df[ycol].values
        s = pd.Series(dea_vrs_output(df[xcol].values, y), index=df.index)
    else:
        s = cols(df, ycol, hib, xcol, adjust=(m == "COLS adjusted"))
    key = (on, inn, w, m)
    scores[key] = s
    ranks[key] = s.rank(ascending=False, method="min")

R = pd.DataFrame(ranks); R.columns = pd.MultiIndex.from_tuples(R.columns, names=["outcome", "input", "window", "method"])
S = pd.DataFrame(scores); S.columns = R.columns
R.to_csv(f"{out}/efficiency_ranks.csv"); S.to_csv(f"{out}/efficiency_scores.csv")
print("specs:", R.shape[1])

# per-country summary
summ = pd.DataFrame({"median": R.median(axis=1), "min": R.min(axis=1), "max": R.max(axis=1),
                     "p10": R.quantile(.1, axis=1), "p90": R.quantile(.9, axis=1),
                     "share_top10": (R <= 10).mean(axis=1), "share_bottom10": (R > len(common) - 10).mean(axis=1)})
summ["range"] = summ["max"] - summ["min"]
summ = summ.sort_values("median")
summ.to_csv(f"{out}/efficiency_summary.csv")
pd.set_option("display.width", 200)
print(summ.round(2).to_string())

# which choice matters most: mean |rank change| switching one factor, others fixed
levels = {n: list(R.columns.get_level_values(n).unique()) for n in R.columns.names}
effects = {}
for f in R.columns.names:
    others = [n for n in R.columns.names if n != f]
    diffs = []
    for combo in itertools.product(*[levels[n] for n in others]):
        cols_ = []
        for lv in levels[f]:
            key = dict(zip(others, combo)); key[f] = lv
            cols_.append(R[tuple(key[n] for n in R.columns.names)])
        for a, b in itertools.combinations(cols_, 2):
            diffs.append((a - b).abs().mean())
    effects[f] = np.mean(diffs)
eff = pd.Series(effects).sort_values(ascending=False)
eff.name = "mean_abs_rank_change_when_switching"
eff.to_csv(f"{out}/efficiency_factor_effects.csv")
print("\nMean |rank change| when switching one choice:\n", eff.round(2).to_string())

# agreement between specs: Kendall tau across all pairs
specs = list(R.columns)
taus = []
for i, j in itertools.combinations(range(len(specs)), 2):
    taus.append(kendalltau(R.iloc[:, i], R.iloc[:, j])[0])
taus = np.array(taus)
print("\nKendall tau between spec pairs: median %.2f, 10th pct %.2f, 90th pct %.2f, min %.2f" % (
    np.median(taus), np.quantile(taus, .1), np.quantile(taus, .9), taus.min()))
# top-5 stability
top5 = (R <= 5).mean(axis=1).sort_values(ascending=False)
print("\nShare of specs with country in top 5:\n", top5.head(10).round(2).to_string())
# is any country top-10 in every spec?
print("\nCountries in top 10 in all specs:", list(R.index[(R <= 10).all(axis=1)]))
print("Countries in top 10 in at least one spec:", int((R <= 10).any(axis=1).sum()), "of", len(common))
