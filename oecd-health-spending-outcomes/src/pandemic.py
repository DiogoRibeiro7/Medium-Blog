"""Pandemic stress test: age-standardised excess mortality 2020-2022 vs pre-pandemic spending and outcomes.

Usage: python3 -I pandemic.py RAW_DIR PANEL_CSV OUT_DIR
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

raw, panel, out = sys.argv[1:4]
YEARS = 148 / 52.1775  # weeks from 2020-W10 to 2022-W52, in years

# ---- excess deaths by age band (counts), both sexes, weekly -> yearly sums
e = pd.read_csv(f"{raw}/excess.csv", dtype=str)
e = e[(e.SEX == "_T") & (e.UNIT_MEASURE == "DT") & (e.AGE != "_T")].copy()
e["val"] = e.OBS_VALUE.astype(float)
e["year"] = e.TIME_PERIOD.str[:4].astype(int)
e["week"] = e.TIME_PERIOD.str[-2:].astype(int)
e = e[(e.year.between(2020, 2022)) & ~((e.year == 2020) & (e.week < 10))]
# keep only countries with complete weekly coverage in the window for every age band
cnt = e.groupby(["REF_AREA", "AGE"]).size().unstack()
full = cnt[(cnt >= 0.97 * 148).all(axis=1)].index  # 2020 W10-W53 (~44) + 52 + 52 weeks ~ 148
ex = e[e.REF_AREA.isin(full)].groupby(["REF_AREA", "AGE"]).val.sum().unstack()
ex.columns = ["ex_le44", "ex_45_64", "ex_ge65"]  # order of AGE codes: Y45T64, Y_GE65, Y_LE44 -> reassigned below
ex = e[e.REF_AREA.isin(full)].groupby(["REF_AREA", "AGE"]).val.sum().unstack().rename(
    columns={"Y_LE44": "ex_le44", "Y45T64": "ex_45_64", "Y_GE65": "ex_ge65"})

# ---- population by band (mean 2020-22)
p = pd.read_csv(f"{raw}/pophist.csv", dtype=str)
p = p[(p.MEASURE == "POP") & (p.SEX == "_T") & (p.UNIT_MEASURE == "PS") & p.TIME_PERIOD.isin(["2020", "2021", "2022"])].copy()
p["v"] = p.OBS_VALUE.astype(float)
pp = p.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns="AGE", values="v")
pp["mid"] = pp[["Y45T49", "Y50T54", "Y55T59", "Y60T64"]].sum(axis=1)
pp["old"] = pp["Y_GE65"]
pp["young"] = pp["_T"] - pp["mid"] - pp["old"]
pop = pp[["young", "mid", "old"]].groupby("REF_AREA").mean()
pop.columns = ["ex_le44", "ex_45_64", "ex_ge65"]
c = ex.index.intersection(pop.index)
ex, pop = ex.loc[c], pop.loc[c]
rate = ex / pop * 1e5 / YEARS  # per 100k per year
w = (pop / pop.sum(axis=1).values[:, None]).mean()  # reference age weights: mean share across countries
rate["excess_std"] = (rate * w.values).sum(axis=1)  # age-standardised excess deaths per 100k per year
rate["excess_ge65"] = rate.ex_ge65
print("age weights:", w.round(3).to_dict(), "| countries:", len(rate))

# ---- pre-pandemic covariates (2017-19 means) from the panel
d = pd.read_csv(panel)
pre = d[d.year.between(2017, 2019)].groupby("iso")[["trtm", "prevm", "le0", "hexp_pc_const", "hexp_gdp", "sh_public", "gdp_pc", "pop65", "alcohol"]].mean()
le = d.pivot(index="iso", columns="year", values="le0")
pre["dle_2019_21"] = le[2021] - le[2019]
df = rate.join(pre, how="inner")
for c_ in ["trtm", "hexp_pc_const", "gdp_pc", "prevm"]:
    df["l_" + c_] = np.log(df[c_])
df.to_csv(f"{out}/pandemic_excess.csv")
print(df[["excess_std", "excess_ge65", "dle_2019_21", "hexp_pc_const", "trtm"]].round(1).sort_values("excess_std").to_string())

rows = []
for nm, f in [("excess ~ log spend", "excess_std ~ l_hexp_pc_const"),
              ("excess ~ log spend + log GDP", "excess_std ~ l_hexp_pc_const + l_gdp_pc"),
              ("excess ~ log spend + GDP + 65+ share + alcohol", "excess_std ~ l_hexp_pc_const + l_gdp_pc + pop65 + alcohol"),
              ("excess ~ pre-pandemic treatable mortality", "excess_std ~ l_trtm"),
              ("excess ~ treatable mortality + log GDP", "excess_std ~ l_trtm + l_gdp_pc"),
              ("excess ~ preventable mortality", "excess_std ~ l_prevm"),
              ("excess ~ public share", "excess_std ~ sh_public"),
              ("excess 65+ ~ log spend + log GDP", "excess_ge65 ~ l_hexp_pc_const + l_gdp_pc"),
              ("dLE 2019-21 ~ log spend + log GDP", "dle_2019_21 ~ l_hexp_pc_const + l_gdp_pc")]:
    z = df.dropna(subset=[f.split("~")[0].strip()] + [t.strip() for t in f.split("~")[1].split("+")])
    r = smf.ols(f, z).fit(cov_type="HC3")
    term = f.split("~")[1].split("+")[0].strip()
    rows.append(dict(model=nm, term=term, coef=r.params[term], se=r.bse[term], p=r.pvalues[term], n=int(r.nobs), r2=r.rsquared))
res = pd.DataFrame(rows)
res.to_csv(f"{out}/pandemic_models.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string())

# rank correlation between pre-pandemic efficiency (median rank over specs) and pandemic excess
er = pd.read_csv(f"{out}/efficiency_summary.csv", index_col=0)
j = df.join(er["median"], how="inner")
print("\nSpearman: pre-pandemic efficiency median rank vs excess_std: %.2f (n=%d)" % (j["median"].corr(j.excess_std, method="spearman"), len(j)))
print("Spearman: spend vs excess_std: %.2f" % df.hexp_pc_const.corr(df.excess_std, method="spearman"))
