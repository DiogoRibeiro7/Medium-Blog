"""Why does the within-country effect vanish? Year FE, long differences, power.

Usage: python3 -I panel_robustness.py PANEL_CSV OUT_DIR
"""
import sys
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

panel, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(panel).sort_values(["iso", "year"])
d = d[d.year <= 2019].copy()
d["l_trtm"] = np.log(d.trtm)
d["l_h"] = np.log(d.hexp_pc_const)
d["l_gdp"] = np.log(d.gdp_pc)
g = d.groupby("iso")
d["l_h_avg12"] = pd.concat([g.l_h.shift(1), g.l_h.shift(2)], axis=1).mean(axis=1)
rows = []


def fit(name, y, xs, data, entity=True, time=True, other=None):
    x = data.dropna(subset=[y] + xs).set_index(["iso", "year"])
    r = PanelOLS(x[y], x[xs], entity_effects=entity, time_effects=time).fit(cov_type="clustered", cluster_entity=True)
    v = xs[0]
    rows.append(dict(model=name, coef=r.params[v], se=r.std_errors[v], lo=r.params[v] - 1.96 * r.std_errors[v],
                     hi=r.params[v] + 1.96 * r.std_errors[v], n=len(x), countries=x.index.get_level_values(0).nunique()))


X = ["l_h_avg12", "l_gdp", "pop65", "alcohol"]
fit("Country FE + year FE", "l_trtm", X, d)
fit("Country FE only (year shocks not absorbed)", "l_trtm", X, d, time=False)
fit("Year FE only (between + within)", "l_trtm", X, d, entity=False)

# country-specific linear trends via demeaning
d2 = d.dropna(subset=["l_trtm"] + X).copy()
for c in ["l_trtm"] + X:
    d2[c + "_dt"] = d2.groupby("iso")[c].transform(lambda s: s - np.polyval(np.polyfit(d2.loc[s.index, "year"], s, 1), d2.loc[s.index, "year"]))
fit("Country FE + country trends + year FE", "l_trtm_dt", [c + "_dt" for c in X], d2)

# long differences (5-year) with year FE
ld = d.set_index(["iso", "year"])[["l_trtm", "l_h", "l_gdp", "pop65", "alcohol"]]
frames = []
for t in range(2005, 2020, 5):
    a, b = ld.xs(t, level="year"), ld.xs(t - 5, level="year")
    diff = (a - b)
    diff["l_h_prev"] = b.l_h  # for completeness
    diff["year"] = t
    frames.append(diff.reset_index())
L = pd.concat(frames).rename(columns={"l_h": "l_h_avg12"})
fit("5-year long differences + period FE", "l_trtm", ["l_h_avg12", "l_gdp", "pop65", "alcohol"],
    L.assign(iso=L.iso), entity=False, time=True)

# lagged-spend long difference: spend change over the *previous* 5 years -> outcome change
fr2 = []
for t in range(2010, 2020, 5):
    out_ch = ld.xs(t, level="year").l_trtm - ld.xs(t - 5, level="year").l_trtm
    sp_ch = ld.xs(t - 5, level="year").l_h - ld.xs(t - 10, level="year").l_h
    gd = ld.xs(t, level="year").l_gdp - ld.xs(t - 5, level="year").l_gdp
    fr2.append(pd.DataFrame({"l_trtm": out_ch, "l_h_avg12": sp_ch, "l_gdp": gd, "year": t}).reset_index())
L2 = pd.concat(fr2)
fit("Outcome change on previous-5y spending change", "l_trtm", ["l_h_avg12", "l_gdp"], L2, entity=False, time=True)

res = pd.DataFrame(rows)
res["mde_80pct_power"] = 2.8 * res.se
res.to_csv(f"{out}/robustness.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string())
# data facts relevant to attenuation: share of variance that is within-country after year FE
x = d.dropna(subset=["l_h", "l_trtm"]).copy()
for c in ["l_h", "l_trtm"]:
    x[c + "_w"] = x[c] - x.groupby("iso")[c].transform("mean") - x.groupby("year")[c].transform("mean") + x[c].mean()
print("share of total variance left after country+year demeaning: spend %.3f, treatable %.3f" % (
    x.l_h_w.var() / x.l_h.var(), x.l_trtm_w.var() / x.l_trtm.var()))
