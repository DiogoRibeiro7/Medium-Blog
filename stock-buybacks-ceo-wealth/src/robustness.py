"""Robustness of the buyback / CEO-pay-ratio co-movement (log differences, 2008-2024)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from series import BUYBACKS, PAY_RATIO

yrs = sorted(BUYBACKS)
bb = np.array([BUYBACKS[y] for y in yrs]); gr = np.array([PAY_RATIO[y][1] for y in yrs]); re = np.array([PAY_RATIO[y][0] for y in yrs])
dy = np.array(yrs[1:])
d = lambda a: np.diff(np.log(a))
db, dg, dr = d(bb), d(gr), d(re)

def r(mask, y):
    x = db[mask]; z = y[mask]
    return round(float(stats.pearsonr(x, z)[0]), 3), round(float(stats.spearmanr(x, z)[0]), 3), int(mask.sum())

out = {}
allm = np.ones(len(dy), bool)
out["all"] = {"granted": r(allm, dg), "realized": r(allm, dr)}
for label, drop in {"drop_2008_2009": [2008, 2009], "drop_2010": [2010], "drop_2008_2009_2010": [2008, 2009, 2010], "drop_2020_2021": [2020, 2021],
                    "drop_2008_2009_2020": [2008, 2009, 2020]}.items():
    m = ~np.isin(dy, drop)
    out[label] = {"granted": r(m, dg), "realized": r(m, dr)}
loo = []
for i, y in enumerate(dy):
    m = np.ones(len(dy), bool); m[i] = False
    loo.append((int(y), r(m, dg)[0], r(m, dr)[0]))
out["leave_one_out_granted_pearson_range"] = [min(t[1] for t in loo), max(t[1] for t in loo)]
out["leave_one_out_realized_pearson_range"] = [min(t[2] for t in loo), max(t[2] for t in loo)]
# one-year lag: do buyback changes lead pay-ratio changes, or the reverse?
out["lead_lag_granted"] = {
    "bb_t_vs_gr_t+1": [round(float(stats.pearsonr(db[:-1], dg[1:])[0]), 3), round(float(stats.pearsonr(db[:-1], dg[1:])[1]), 3)],
    "gr_t_vs_bb_t+1": [round(float(stats.pearsonr(dg[:-1], db[1:])[0]), 3), round(float(stats.pearsonr(dg[:-1], db[1:])[1]), 3)],
}
out["lead_lag_realized"] = {
    "bb_t_vs_re_t+1": [round(float(stats.pearsonr(db[:-1], dr[1:])[0]), 3), round(float(stats.pearsonr(db[:-1], dr[1:])[1]), 3)],
    "re_t_vs_bb_t+1": [round(float(stats.pearsonr(dr[:-1], db[1:])[0]), 3), round(float(stats.pearsonr(dr[:-1], db[1:])[1]), 3)],
}
# 2015-2024 only (the 'last decade' of the theory)
m = dy >= 2016
out["last_decade_2016_2024"] = {"granted": r(m, dg), "realized": r(m, dr)}
m = dy <= 2015
out["first_period_2008_2015"] = {"granted": r(m, dg), "realized": r(m, dr)}
json.dump(out, open(sys.argv[1], "w"), indent=2); print(json.dumps(out, indent=2))
