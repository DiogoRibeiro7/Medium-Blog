"""Spending vs treatable mortality: cross-section, two-way FE panel, dose-response, change-on-change.

Usage: python3 -I panel_models.py PANEL_CSV OUT_DIR
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

panel, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(panel)
d["l_trtm"] = np.log(d.trtm)
d["l_prevm"] = np.log(d.prevm)
d["l_avm"] = np.log(d.avm)
d["l_le"] = np.log(d.le0)
d["l_h"] = np.log(d.hexp_pc_const)
d["l_gdp"] = np.log(d.gdp_pc)
d = d.sort_values(["iso", "year"])
g = d.groupby("iso")
for k in range(1, 4):
    d[f"l_h_L{k}"] = g.l_h.shift(k)
d["l_h_avg12"] = d[["l_h_L1", "l_h_L2"]].mean(axis=1)
d["l_h_avg03"] = d[["l_h", "l_h_L1", "l_h_L2", "l_h_L3"]].mean(axis=1)

rows = []


def add(name, res, var, n, nc, note=""):
    b, se = res.params[var], res.std_errors[var] if hasattr(res, "std_errors") else res.bse[var]
    rows.append(dict(model=name, term=var, coef=b, se=se, n=n, countries=nc, note=note))


def fe(formula_y, xs, data, cluster=True):
    x = data.dropna(subset=[formula_y] + xs).set_index(["iso", "year"])
    mod = PanelOLS(x[formula_y], x[xs], entity_effects=True, time_effects=True)
    r = mod.fit(cov_type="clustered", cluster_entity=True)
    return r, len(x), x.index.get_level_values(0).nunique()


main = d[d.year <= 2019]  # exclude pandemic years

# ---- 1. cross-section 2015-19 average
cs = main[main.year.between(2015, 2019)].groupby("iso")[["l_trtm", "l_prevm", "l_le", "l_h", "l_gdp", "pop65", "alcohol"]].mean().dropna(subset=["l_trtm", "l_h"])
for nm, f in [("CS raw", "l_trtm ~ l_h"), ("CS +income", "l_trtm ~ l_h + l_gdp"),
              ("CS +income+age+alcohol", "l_trtm ~ l_h + l_gdp + pop65 + alcohol")]:
    r = smf.ols(f, cs.dropna()).fit(cov_type="HC3")
    rows.append(dict(model=nm, term="l_h", coef=r.params["l_h"], se=r.bse["l_h"], n=int(r.nobs), countries=int(r.nobs), note=f"R2={r.rsquared:.2f}"))
cs.to_csv(f"{out}/cross_section_2015_19.csv")

# ---- 2. two-way FE panel 2000-2019
for nm, xs in [("FE contemporaneous", ["l_h"]),
               ("FE +income", ["l_h", "l_gdp"]),
               ("FE +income+age+alcohol", ["l_h", "l_gdp", "pop65", "alcohol"]),
               ("FE spend lag1", ["l_h_L1", "l_gdp", "pop65", "alcohol"]),
               ("FE spend avg lag1-2", ["l_h_avg12", "l_gdp", "pop65", "alcohol"]),
               ("FE spend avg lag0-3", ["l_h_avg03", "l_gdp", "pop65", "alcohol"])]:
    r, n, nc = fe("l_trtm", xs, main)
    add(nm, r, xs[0], n, nc)

# distributed lags
r, n, nc = fe("l_trtm", ["l_h", "l_h_L1", "l_h_L2", "l_h_L3", "l_gdp", "pop65", "alcohol"], main)
for v in ["l_h", "l_h_L1", "l_h_L2", "l_h_L3"]:
    add("FE distributed lags", r, v, n, nc)
cum = sum(r.params[v] for v in ["l_h", "l_h_L1", "l_h_L2", "l_h_L3"])
rows.append(dict(model="FE distributed lags", term="cumulative 0-3", coef=cum, se=np.nan, n=n, countries=nc, note="sum of coefficients"))

# placebo / comparison outcomes
for y in ["l_prevm", "l_avm", "l_le"]:
    r, n, nc = fe(y, ["l_h_avg12", "l_gdp", "pop65", "alcohol"], main)
    add(f"FE outcome={y}", r, "l_h_avg12", n, nc)

# robustness: drop LUX/IRL (GDP distortions), drop USA, post-2008 only, with smoking
for nm, sub in [("FE drop LUX,IRL", main[~main.iso.isin(["LUX", "IRL"])]),
                ("FE drop USA", main[main.iso != "USA"]),
                ("FE 2008-2019", main[main.year >= 2008]),
                ("FE drop CEE+Baltics early catch-up", main[~main.iso.isin(["EST", "LVA", "LTU", "POL", "SVK", "CZE", "HUN", "SVN"])])]:
    r, n, nc = fe("l_trtm", ["l_h_avg12", "l_gdp", "pop65", "alcohol"], sub)
    add(nm, r, "l_h_avg12", n, nc)
r, n, nc = fe("l_trtm", ["l_h_avg12", "l_gdp", "pop65", "alcohol", "smoke"], main)
add("FE +smoking", r, "l_h_avg12", n, nc, "smaller sample")

# ---- 3. dose-response: does the effect fade at higher spending?
main = main.copy()
main["l_h_c"] = main.l_h_avg12 - main.l_h_avg12.mean()
main["l_h_c2"] = main.l_h_c ** 2
r, n, nc = fe("l_trtm", ["l_h_c", "l_h_c2", "l_gdp", "pop65", "alcohol"], main)
add("FE quadratic", r, "l_h_c", n, nc)
add("FE quadratic", r, "l_h_c2", n, nc)
# spline-ish: separate slopes below/above median spending level
med = main.groupby("iso").l_h_avg12.mean().median()
main["lo"] = np.minimum(main.l_h_avg12, med)
main["hi"] = np.maximum(main.l_h_avg12 - med, 0)
r, n, nc = fe("l_trtm", ["lo", "hi", "l_gdp", "pop65", "alcohol"], main)
add("FE piecewise at median", r, "lo", n, nc, f"knot={np.exp(med):.0f} USD PPP")
add("FE piecewise at median", r, "hi", n, nc)
# test hi - lo
diff = r.params["hi"] - r.params["lo"]
cov = r.cov
se = np.sqrt(cov.loc["hi", "hi"] + cov.loc["lo", "lo"] - 2 * cov.loc["hi", "lo"])
rows.append(dict(model="FE piecewise at median", term="hi - lo", coef=diff, se=se, n=n, countries=nc, note=""))

# ---- 4. change-on-change, 2009 -> 2019 (post-crisis austerity episode)
a = d[d.year.isin([2009, 2019])].pivot(index="iso", columns="year", values=["l_trtm", "l_h", "l_gdp", "pop65"])
ch = pd.DataFrame({"d_trtm": a["l_trtm"][2019] - a["l_trtm"][2009], "d_h": a["l_h"][2019] - a["l_h"][2009],
                   "d_gdp": a["l_gdp"][2019] - a["l_gdp"][2009], "d_pop65": a["pop65"][2019] - a["pop65"][2009],
                   "trtm09": a["l_trtm"][2009]}).dropna()
for nm, f in [("Delta 2009-19 raw", "d_trtm ~ d_h"), ("Delta 2009-19 +income+age", "d_trtm ~ d_h + d_gdp + d_pop65"),
              ("Delta 2009-19 +initial level", "d_trtm ~ d_h + d_gdp + d_pop65 + trtm09")]:
    r = smf.ols(f, ch).fit(cov_type="HC3")
    rows.append(dict(model=nm, term="d_h", coef=r.params["d_h"], se=r.bse["d_h"], n=int(r.nobs), countries=int(r.nobs), note=f"R2={r.rsquared:.2f}"))
ch.to_csv(f"{out}/change_2009_2019.csv")

res = pd.DataFrame(rows)
res["t"] = res.coef / res.se
res.to_csv(f"{out}/panel_results.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string())
