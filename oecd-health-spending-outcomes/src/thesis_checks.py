"""Checks for the 'quality before spending' thesis.
Usage: python3 -I thesis_checks.py RESULTS_DIR
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

res = sys.argv[1]
df = pd.read_csv(f"{res}/pandemic_all_countries.csv", index_col=0)
df["l_prevm"] = np.log(df.prevm)
print("n with crude excess:", len(df), "| with treatable mortality:", int(df.trtm.notna().sum()))

print("\ncorr(log treatable, log preventable) = %.2f" % df.l_trtm.corr(df.l_prevm))
def fit(f, data, label=""):
    z = data.dropna(subset=[f.split("~")[0].strip()] + [t.strip() for t in f.split("~")[1].split("+")])
    r = smf.ols(f, z).fit(cov_type="HC3")
    print(f"{label:14s}{f:58s} n={int(r.nobs):2d} R2={r.rsquared:.2f}  " + "  ".join(
        f"{k}={v:7.1f}(se {r.bse[k]:.1f})" for k, v in r.params.items() if k not in ("Intercept",)))
    return r

print("\n-- horse race (outcome: crude excess deaths per 100k per year) --")
for f in ["excess_crude ~ l_trtm + pop65",
          "excess_crude ~ l_prevm + pop65",
          "excess_crude ~ l_trtm + l_prevm + pop65",
          "excess_crude ~ l_trtm + l_h + pop65",
          "excess_crude ~ l_trtm + l_h + l_gdp + pop65",
          "excess_crude ~ l_trtm + l_h + l_gdp + pop65 + sh_public",
          "excess_crude ~ l_trtm + l_h + l_gdp + pop65 + alcohol"]:
    fit(f, df)

print("\n-- leave-one-out: coefficient on log treatable (model with income and 65+) --")
f = "excess_crude ~ l_trtm + l_gdp + pop65"
z = df.dropna(subset=["excess_crude", "l_trtm", "l_gdp", "pop65"])
co = {}
for c in z.index:
    r = smf.ols(f, z.drop(c)).fit(cov_type="HC3"); co[c] = (r.params["l_trtm"], r.bse["l_trtm"])
co = pd.DataFrame(co, index=["coef", "se"]).T.sort_values("coef")
print("full-sample:", round(smf.ols(f, z).fit().params["l_trtm"], 1), "| LOO range: %.1f to %.1f" % (co.coef.min(), co.coef.max()))
print("lowest 3:", co.head(3).round(1).to_dict("index")); print("highest 3:", co.tail(3).round(1).to_dict("index"))
print("all LOO t-stats > 2:", bool((co.coef / co.se > 2).all()), "| min t = %.2f" % (co.coef / co.se).min())

print("\n-- bootstrap (countries resampled, 5000 draws) --")
rng = np.random.default_rng(20261128)
b = []
for _ in range(5000):
    s = z.sample(len(z), replace=True, random_state=int(rng.integers(1e9)))
    try:
        b.append(smf.ols(f, s).fit().params["l_trtm"])
    except Exception:
        pass
b = np.array(b); print("coef on log treatable: median %.1f, 95%% interval %.1f to %.1f" % (np.median(b), *np.quantile(b, [.025, .975])))

print("\n-- rank-based view: treatable-mortality rank vs spending rank vs excess rank (Spearman) --")
print(df[["excess_crude", "trtm", "hexp_pc_const", "gdp_pc", "prevm", "le0"]].corr(method="spearman").round(2).loc["excess_crude"].to_string())

print("\n-- pre-pandemic life expectancy as predictor --")
fit("excess_crude ~ le0 + pop65", df)
fit("excess_crude ~ le0 + l_h + l_gdp + pop65", df)

print("\n-- relative standing: treatable rank vs excess rank, biggest gaps --")
d2 = df.dropna(subset=["trtm", "excess_crude"]).copy()
d2["r_trtm"] = d2.trtm.rank(); d2["r_ex"] = d2.excess_crude.rank(); d2["gap"] = d2.r_ex - d2.r_trtm
print(d2.sort_values("gap")[["r_trtm", "r_ex", "gap"]].head(5).to_string()); print(d2.sort_values("gap")[["r_trtm", "r_ex", "gap"]].tail(5).to_string())
