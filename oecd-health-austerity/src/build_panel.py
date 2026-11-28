"""Build the OECD health spending / outcomes panel from raw SDMX CSV extracts.

Usage: python3 -I build_panel.py RAW_DIR OUT_CSV
"""
import sys
import pandas as pd

raw, out = sys.argv[1], sys.argv[2]

OECD = ("AUS AUT BEL CAN CHL COL CRI CZE DNK EST FIN FRA DEU GRC HUN ISL IRL ISR "
        "ITA JPN KOR LVA LTU LUX MEX NLD NZL NOR POL PRT SVK SVN ESP SWE CHE TUR GBR USA").split()
assert len(OECD) == 38


def load(name, **kw):
    return pd.read_csv(f"{raw}/{name}", dtype=str, **kw)


# --- outcomes: avoidable mortality, age-standardised rates per 100k, both sexes
am = load("am.csv")
am = am[(am.UNIT_MEASURE == "DT_10P5HB") & (am.SEX == "_T")]
am = am.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns="MEASURE",
                    values="OBS_VALUE", aggfunc="first").astype(float)
am.columns = ["avm", "prevm", "trtm"]

le = load("DF_LE.csv")
le = le[(le.MEASURE == "LFEXP") & (le.AGE == "Y0") & (le.SEX == "_T")]
le = le.set_index(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.astype(float).rename("le0")

# --- spending
sha = load("sha_slim.csv")
sha["value"] = sha.value.astype(float)


def pick(unit, fs, fn, price, name):
    s = sha[(sha.unit == unit) & (sha.fs == fs) & (sha.fn == fn) & (sha.price == price)]
    return s.set_index(["area", "year"]).value.rename(name)


sp = pd.concat([
    pick("USD_PPP_PS", "_T", "_T", "Q", "hexp_pc_const"),
    pick("USD_PPP_PS", "_T", "_T", "V", "hexp_pc_curr"),
    pick("PT_B1GQ", "_T", "_T", "_Z", "hexp_gdp"),
    pick("PT_EXP_HLTH", "HF1", "_T", "_Z", "sh_public"),
    pick("PT_EXP_HLTH", "HF3", "_T", "_Z", "sh_oop"),
    pick("PT_EXP_HLTH", "_T", "HC6", "_Z", "sh_prev"),
], axis=1)
sp.index = sp.index.set_names(["REF_AREA", "TIME_PERIOD"])

# --- covariates
gdp = load("gdp_all.csv")
gdp = gdp[(gdp.TRANSACTION == "B1GQ_POP") & (gdp.UNIT_MEASURE == "USD_PPP_PS")]
gdp = gdp.set_index(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.astype(float).rename("gdp_pc")
gdp = gdp[~gdp.index.duplicated()]

pop = load("pophist.csv")
pop = pop[(pop.MEASURE == "POP") & (pop.SEX == "_T") & (pop.UNIT_MEASURE == "PT_POP") & (pop.AGE == "Y_GE65")]
pop = pop.set_index(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.astype(float).rename("pop65")

tob = load("DF_HEALTH_LVNG_TC.csv")
tob = tob[(tob.MEASURE == "SP_DS") & (tob.SEX == "_T") & (tob.AGE == "Y_GE15")]
tob = tob.set_index(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.astype(float).rename("smoke")
tob = tob[~tob.index.duplicated()]

alc = load("DF_HEALTH_LVNG_AC.csv")
alc = alc[(alc.MEASURE == "AC") & (alc.AGE == "Y_GE15") & (alc.SEX == "_T")]
alc = alc.set_index(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.astype(float).rename("alcohol")
alc = alc[~alc.index.duplicated()]

df = pd.concat([am, le, sp, gdp, pop, tob, alc], axis=1).reset_index()
df.columns = ["iso", "year"] + list(df.columns[2:])
df["year"] = df.year.astype(int)
df = df[df.iso.isin(OECD) & df.year.between(2000, 2023)].sort_values(["iso", "year"])
df.to_csv(out, index=False)

print(df.shape, df.iso.nunique(), "countries")
print(df.notna().mean().round(2).to_string())
core = df.dropna(subset=["trtm", "hexp_pc_const", "gdp_pc"])
print("core rows", len(core), "countries", core.iso.nunique())
print(core.groupby("iso").size().sort_values().head(12).to_string())
