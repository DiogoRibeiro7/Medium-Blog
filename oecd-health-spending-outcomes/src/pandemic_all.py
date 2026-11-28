"""Extend pandemic analysis to all countries with weekly excess deaths (all-age), validate vs age-standardised,
and combine with pre-pandemic efficiency ranks.
Usage: python3 -I pandemic_all.py RAW_DIR PANEL_CSV RESULTS_DIR FIG_DIR
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

raw, panel, res, figdir = sys.argv[1:5]
YEARS = 148 / 52.1775  # weeks from 2020-W10 to 2022-W52, in years

e = pd.read_csv(f"{raw}/excess.csv", dtype=str)
e = e[(e.SEX == "_T") & (e.UNIT_MEASURE == "DT") & (e.AGE == "_T")].copy()
e["val"] = e.OBS_VALUE.astype(float)
e["year"] = e.TIME_PERIOD.str[:4].astype(int); e["week"] = e.TIME_PERIOD.str[-2:].astype(int)
e = e[e.year.between(2020, 2022) & ~((e.year == 2020) & (e.week < 10))]
n = e.groupby("REF_AREA").size()
full = n[n >= 0.97 * 148].index
print("all-age coverage complete:", len(full), "| incomplete:", sorted(set(n.index) - set(full)))
ex = e[e.REF_AREA.isin(full)].groupby("REF_AREA").val.sum()

p = pd.read_csv(f"{raw}/pophist.csv", dtype=str)
p = p[(p.MEASURE == "POP") & (p.SEX == "_T") & (p.UNIT_MEASURE == "PS") & (p.AGE == "_T") & p.TIME_PERIOD.isin(["2020", "2021", "2022"])]
pop = p.assign(v=p.OBS_VALUE.astype(float)).groupby("REF_AREA").v.mean()
crude = (ex / pop.reindex(ex.index) * 1e5 / YEARS).rename("excess_crude")

d = pd.read_csv(panel)
pre = d[d.year.between(2017, 2019)].groupby("iso")[["trtm", "prevm", "le0", "hexp_pc_const", "sh_public", "gdp_pc", "pop65", "alcohol"]].mean()
eff = pd.read_csv(f"{res}/efficiency_summary.csv", index_col=0)[["median", "p10", "p90"]].rename(columns={"median": "eff_rank"})
std = pd.read_csv(f"{res}/pandemic_excess.csv", index_col=0)[["excess_std"]]
df = pd.concat([crude, std, pre, eff], axis=1, join="inner" if False else "outer").loc[crude.index]
df["l_h"] = np.log(df.hexp_pc_const); df["l_gdp"] = np.log(df.gdp_pc); df["l_trtm"] = np.log(df.trtm)
df.to_csv(f"{res}/pandemic_all_countries.csv")
print("countries with crude excess:", len(df), "| with standardised:", int(df.excess_std.notna().sum()))
both = df.dropna(subset=["excess_std"])
print("validation (n=%d): Pearson crude vs standardised %.2f, Spearman %.2f" % (
    len(both), both.excess_crude.corr(both.excess_std), both.excess_crude.corr(both.excess_std, method="spearman")))
print(df[["excess_crude", "excess_std", "pop65", "hexp_pc_const", "trtm", "eff_rank"]].round(1).sort_values("excess_crude").to_string())

rows = []
def run(label, f, data):
    t = f.split("~")[1].split("+")[0].strip()
    z = data.dropna(subset=[f.split("~")[0].strip()] + [x.strip() for x in f.split("~")[1].split("+")])
    r = smf.ols(f, z).fit(cov_type="HC3")
    rows.append(dict(sample=label, formula=f, coef=r.params[t], se=r.bse[t], n=int(r.nobs), r2=r.rsquared))

for lab, data in [("all", df), ("drop MEX", df.drop("MEX")), ("drop MEX,USA", df.drop(["MEX", "USA"]))]:
    for f in ["excess_crude ~ l_h + pop65", "excess_crude ~ l_h + l_gdp + pop65",
              "excess_crude ~ l_trtm + pop65", "excess_crude ~ l_trtm + l_gdp + pop65",
              "excess_crude ~ eff_rank + pop65"]:
        run(lab, f, data)
R = pd.DataFrame(rows); R.to_csv(f"{res}/pandemic_all_models.csv", index=False)
pd.set_option("display.width", 220)
print(R.round(2).to_string())
j = df.dropna(subset=["eff_rank"])
print("\nSpearman eff_rank vs crude excess: %.2f (n=%d); without MEX: %.2f" % (
    j.eff_rank.corr(j.excess_crude, method="spearman"), len(j), j.drop("MEX").eff_rank.corr(j.drop("MEX").excess_crude, method="spearman")))

# ---- figure: pre-pandemic efficiency vs pandemic excess
BLUE, GREY, INK, MUTED = "#257de0", "#9c9c9c", "#191919", "#666666"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc", "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8, 5.6), dpi=160)
k = j.copy()
ax.hlines(k.excess_crude, k.p10, k.p90, color="#d9d9d9", lw=2, zorder=1)
ax.scatter(k.eff_rank, k.excess_crude, s=38, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
for c, r_ in k.iterrows():
    ax.annotate(c, (r_.eff_rank, r_.excess_crude), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK)
ax.set_xlabel("Pre-pandemic efficiency rank (median over 72 specs; 1 = best). Bar = 10th-90th percentile")
ax.set_ylabel("Excess deaths per 100,000 per year, Mar 2020 - Dec 2022")
ax.set_title("Countries ranked efficient before 2020 had fewer excess deaths", loc="left", fontsize=11, color=INK)
ax.grid(color="#eeeeee", lw=0.8, zorder=0)
fig.tight_layout(); fig.savefig(f"{figdir}/fig3_efficiency_vs_pandemic.png"); plt.close(fig)
