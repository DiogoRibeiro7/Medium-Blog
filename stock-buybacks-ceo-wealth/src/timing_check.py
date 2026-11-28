"""Timing check: do aggregate buybacks and CEO pay ratios move together, beyond the market?"""
import json, sys, math, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from series import BUYBACKS, PAY_RATIO, PRICE_RETURN, REAL_PAY_2024USD_K

yrs = sorted(BUYBACKS)
bb = np.array([BUYBACKS[y] for y in yrs])
real = np.array([PAY_RATIO[y][0] for y in yrs])
gran = np.array([PAY_RATIO[y][1] for y in yrs])
ret = np.array([PRICE_RETURN[y] for y in yrs])
out = {}

def corr(a, b):
    r, p = stats.pearsonr(a, b); s, ps = stats.spearmanr(a, b)
    return dict(n=len(a), pearson=round(r, 3), p_pearson=round(p, 3), spearman=round(s, 3), p_spearman=round(ps, 3))

out["levels_2007_2024"] = {"realized": corr(bb, real), "granted": corr(bb, gran)}
# growth rates (log differences)
dbb = np.diff(np.log(bb)); dre = np.diff(np.log(real)); dgr = np.diff(np.log(gran)); r1 = ret[1:]
out["logdiff_2008_2024"] = {"realized": corr(dbb, dre), "granted": corr(dbb, dgr)}

def partial(y, x, z):
    X = np.column_stack([np.ones(len(z)), z])
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return dict(n=len(y), partial_r=round(r, 3), p=round(p, 3))

out["partial_given_price_return_logdiff"] = {"realized": partial(dre, dbb, r1), "granted": partial(dgr, dbb, r1)}
out["market_vs_each_logdiff"] = {"buybacks": corr(dbb, r1), "realized": corr(dre, r1), "granted": corr(dgr, r1)}

# Long-run timing
allyrs = sorted(PAY_RATIO)
R = {y: PAY_RATIO[y][0] for y in allyrs}; G = {y: PAY_RATIO[y][1] for y in allyrs}
out["ratio_path"] = {
    "realized_1982": R[1982], "realized_1989": R[1989], "realized_2000": R[2000],
    "realized_2007": R[2007], "realized_2015": R[2015], "realized_2021": R[2021], "realized_2024": R[2024],
    "granted_1982": G[1982], "granted_2000": G[2000], "granted_2015": G[2015], "granted_2024": G[2024],
}
tot = R[2024] - R[1982]
out["share_of_1982_2024_rise_realized_ratio_already_by_2000"] = round((R[2000] - R[1982]) / tot, 3)
out["buyback_change_2015_2024_pct"] = round(100 * (BUYBACKS[2024] / BUYBACKS[2015] - 1), 1)
out["realized_ratio_change_2015_2024_pct"] = round(100 * (R[2024] / R[2015] - 1), 1)
out["granted_ratio_change_2015_2024_pct"] = round(100 * (G[2024] / G[2015] - 1), 1)
out["buyback_change_2007_2024_pct"] = round(100 * (BUYBACKS[2024] / BUYBACKS[2007] - 1), 1)
out["granted_ratio_change_2007_2024_pct"] = round(100 * (G[2024] / G[2007] - 1), 1)

# Block means by period
def mean(d, a, b): return round(float(np.mean([d[y] for y in range(a, b + 1)])), 1)
out["period_means"] = {
    "2007_2014": {"buybacks": mean(BUYBACKS, 2007, 2014), "realized": mean(R, 2007, 2014), "granted": mean(G, 2007, 2014)},
    "2015_2024": {"buybacks": mean(BUYBACKS, 2015, 2024), "realized": mean(R, 2015, 2024), "granted": mean(G, 2015, 2024)},
}
rp = REAL_PAY_2024USD_K
out["real_dollar_pay_change_2000_2024_pct"] = {"realized": round(100 * (rp[2024][0] / rp[2000][0] - 1), 1), "granted": round(100 * (rp[2024][1] / rp[2000][1] - 1), 1)}
# Order of magnitude: CEO realized pay (350 firms, 2024, USD 22.98m) vs S&P 500 buybacks
ceo_total_bn = 22.98e6 * 350 / 1e9
out["ceo_pay_total_350_bn_2024"] = round(ceo_total_bn, 2)
out["ceo_pay_over_buybacks_pct_2024"] = round(100 * ceo_total_bn / BUYBACKS[2024], 2)

# permutation test on levels (realized), 20000 shuffles of the pay series
rng = np.random.default_rng(7)
def perm(x, y, B=20000):
    obs = stats.spearmanr(x, y)[0]
    cnt = sum(abs(stats.spearmanr(x, rng.permutation(y))[0]) >= abs(obs) for _ in range(B))
    return round((cnt + 1) / (B + 1), 4)
out["permutation_p_spearman_levels"] = {"realized": perm(bb, real), "granted": perm(bb, gran)}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/timing_check.json", "w"), indent=2)
print(json.dumps(out, indent=2))
