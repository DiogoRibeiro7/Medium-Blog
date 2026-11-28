"""Austerity episodes as a natural experiment: did the post-2009 cuts in real health spending move mortality?

Designs
  A. Dose-response long differences: change in outcome from 2009 to 2009+k on the 2009-2013 change in real
     health spending per head, k = 1..10, with and without pre-trend and GDP controls (HC3).
  B. Cut-versus-no-cut comparison of the same long differences, with a permutation (placebo) test.
  C. Synthetic control for the cut countries, with in-space placebos and post/pre RMSPE ratios.
  D. Robustness: dose definitions, leave-one-out, series breaks kept in, alternative outcomes.

Usage: python3 -I austerity_event.py PANEL_CSV OUT_DIR
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

panel, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(panel).sort_values(["iso", "year"])

H = d.pivot(index="year", columns="iso", values="hexp_pc_const")
G = d.pivot(index="year", columns="iso", values="gdp_pc")
OUT = {
    "treatable": d.pivot(index="year", columns="iso", values="trtm"),
    "preventable": d.pivot(index="year", columns="iso", values="prevm"),
    "avoidable": d.pivot(index="year", columns="iso", values="avm"),
    "life_exp": d.pivot(index="year", columns="iso", values="le0"),
}
SH_PUB = d.pivot(index="year", columns="iso", values="sh_public")
LOGOUT = {"treatable", "preventable", "avoidable"}


def series(name):
    y = OUT[name]
    return np.log(y) * 100 if name in LOGOUT else y  # log points x100, or years of life expectancy


# ---- Series-break screen: a one-year move in real health spending far larger than the move in GDP per head
dh = np.log(H).diff() * 100
dg = np.log(G).diff() * 100
gap = (dh - dg).loc[2008:2014]
flag = gap.abs().max()
BREAKS = sorted(flag[flag > 12].index)
screen = pd.DataFrame({"max_abs_gap_pp": flag.round(1), "flagged": flag > 12}).sort_values("max_abs_gap_pp", ascending=False)
screen.to_csv(f"{out}/austerity_break_screen.csv")

# ---- Dose definitions
dose_main = np.log(H.loc[2013] / H.loc[2009]) * 100
doses = {
    "2009-2013": dose_main,
    "2009-2012": np.log(H.loc[2012] / H.loc[2009]) * 100,
    "2009-2014": np.log(H.loc[2014] / H.loc[2009]) * 100,
    "peak-to-trough": np.log(H.loc[2009:2015].min() / H.loc[2009:2015].max()) * 100,
}
gdp_chg = np.log(G.loc[2013] / G.loc[2009]) * 100
pub_chg = SH_PUB.loc[2013] - SH_PUB.loc[2009]
pd.DataFrame({"dose": dose_main, "gdp_change_2009_2013": gdp_chg, "public_share_change": pub_chg,
              "flagged_break": [c in BREAKS for c in dose_main.index]}).round(2).to_csv(f"{out}/austerity_doses.csv")

CUT = 5.0  # a "cut" is a fall of at least 5% in real spending per head, 2009 to 2013
cut_set = sorted(c for c in dose_main.index if dose_main[c] <= -CUT and c not in BREAKS)


def long_diff(name, k, base=2009):
    y = series(name)
    return y.loc[base + k] - y.loc[base]


def reg(y, X, drop=()):
    df = pd.concat([y.rename("y"), X], axis=1).drop(index=list(drop), errors="ignore").dropna()
    m = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(cov_type="HC3")
    return m, len(df)


# ---- A. dose-response path
rows = []
for name in OUT:
    pre = series(name).loc[2009] - series(name).loc[2004]
    for spec in ["raw", "pretrend", "pretrend+gdp"]:
        for k in range(1, 11):
            X = pd.DataFrame({"dose": dose_main})
            if spec != "raw":
                X["pre"] = pre
            if spec == "pretrend+gdp":
                X["gdp"] = gdp_chg
            m, n = reg(long_diff(name, k), X, drop=BREAKS)
            rows.append(dict(outcome=name, spec=spec, horizon=k, coef=m.params["dose"], se=m.bse["dose"], n=n))
A = pd.DataFrame(rows)
A["mde"] = 2.8 * A.se
A.round(4).to_csv(f"{out}/austerity_dose_response.csv", index=False)

# ---- B. cut vs no-cut with permutation test (difference in mean long differences)
rows = []
for name in OUT:
    for k in (3, 5, 8, 10):
        y = long_diff(name, k).drop(index=BREAKS, errors="ignore").dropna()
        grp = pd.Series({c: c in cut_set for c in y.index})
        if grp.sum() < 2:
            continue
        obs = y[grp].mean() - y[~grp].mean()
        rng = np.random.default_rng(7)
        perm = []
        for _ in range(20000):
            g = rng.permutation(grp.values)
            perm.append(y[g].mean() - y[~g].mean())
        perm = np.array(perm)
        rows.append(dict(outcome=name, horizon=k, n_cut=int(grp.sum()), n_other=int((~grp).sum()),
                         diff=obs, p_perm=(np.abs(perm) >= abs(obs)).mean(),
                         perm_sd=perm.std(), mde=2.8 * perm.std()))
B = pd.DataFrame(rows)
B.round(4).to_csv(f"{out}/austerity_cut_vs_other.csv", index=False)

# ---- C. synthetic control
POST0 = 2010


def synth(unit, name, donors, pre_years, post_years):
    y = series(name)
    cols = [c for c in donors if c != unit and y.loc[pre_years, c].notna().all() and y.loc[post_years, c].notna().all()]
    if y.loc[pre_years, unit].isna().any() or y.loc[post_years, unit].isna().any() or len(cols) < 4:
        return None
    # demean by the 2009 level so the match is on trajectory, not level
    base = y.loc[2009]
    Y0 = (y[cols] - base[cols])
    y1 = (y[unit] - base[unit])
    A_ = Y0.loc[pre_years].values
    b_ = y1.loc[pre_years].values

    def loss(w):
        return np.sum((A_ @ w - b_) ** 2)

    n = len(cols)
    r = minimize(loss, np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500, "ftol": 1e-10})
    w = r.x
    gap = y1 - (Y0 @ w)
    pre_rmspe = np.sqrt(np.mean(gap.loc[pre_years] ** 2))
    post_rmspe = np.sqrt(np.mean(gap.loc[post_years] ** 2))
    return dict(weights=pd.Series(w, index=cols), gap=gap, pre=pre_rmspe, post=post_rmspe,
                avg_post=gap.loc[post_years].mean())


def run_sc(name, unit, pre_years, post_years, pool):
    base = synth(unit, name, pool, pre_years, post_years)
    if base is None:
        return None
    plac = []
    for c in pool:
        r = synth(c, name, pool, pre_years, post_years)
        if r is not None:
            plac.append((c, r["avg_post"], r["pre"], r["post"]))
    # placebos with poor pre-fit (>2x the unit's pre RMSPE) are dropped, as is standard
    keep = [p for p in plac if p[2] <= 2 * max(base["pre"], 1e-9) or p[2] <= np.median([q[2] for q in plac])]
    ratio_unit = base["post"] / max(base["pre"], 1e-9)
    ratios = [p[3] / max(p[2], 1e-9) for p in keep]
    p_val = (np.sum(np.array(ratios) >= ratio_unit) + 1) / (len(ratios) + 1)
    return base, plac, keep, ratio_unit, p_val


no_cut_pool = sorted(c for c in dose_main.index if dose_main[c] > 0 and c not in BREAKS)
sc_rows, sc_gaps, sc_weights = [], {}, {}
for name in ["treatable", "life_exp"]:
    pre_years = list(range(2004, 2010)) if name == "treatable" else list(range(2000, 2010))  # treatable series start later for ITA, IRL, PRT, GRC
    post_years = list(range(2011, 2017))
    for unit in ["ESP", "ITA", "PRT", "ISL", "GRC", "SVN", "IRL"]:
        res = run_sc(name, unit, pre_years, post_years, no_cut_pool)
        if res is None:
            sc_rows.append(dict(outcome=name, unit=unit, status="insufficient pre-period data"))
            continue
        base, plac, keep, ratio_unit, p_val = res
        unit_pre = base["pre"]
        sc_rows.append(dict(outcome=name, unit=unit, status="ok", dose=round(dose_main[unit], 1),
                            avg_gap_2011_16=base["avg_post"], pre_rmspe=unit_pre, post_rmspe=base["post"],
                            rmspe_ratio=ratio_unit, p_placebo=p_val, n_placebos=len(keep),
                            placebo_sd=np.std([p[1] for p in keep]),
                            top_donors=", ".join(f"{k} {v:.2f}" for k, v in base["weights"].sort_values(ascending=False).head(4).items())))
        sc_gaps[(name, unit)] = base["gap"]
        sc_weights[(name, unit)] = base["weights"]
        # placebo gaps for plotting
        sc_gaps[(name, unit, "placebo")] = pd.DataFrame({c: synth(c, name, no_cut_pool, pre_years, post_years)["gap"]
                                                         for c, *_ in keep})
SC = pd.DataFrame(sc_rows)
SC.round(3).to_csv(f"{out}/austerity_synth.csv", index=False)
pd.concat({f"{k[0]}|{k[1]}": v for k, v in sc_gaps.items() if len(k) == 2}, axis=1).to_csv(f"{out}/austerity_synth_gaps.csv")
for k, v in sc_gaps.items():
    if len(k) == 3:
        v.to_csv(f"{out}/austerity_placebo_{k[0]}_{k[1]}.csv")

# ---- D. robustness
rows = []
for dname, dz in doses.items():
    for name in ["treatable", "life_exp"]:
        X = pd.DataFrame({"dose": dz, "pre": series(name).loc[2009] - series(name).loc[2004]})
        m, n = reg(long_diff(name, 5), X, drop=BREAKS)
        rows.append(dict(check=f"dose {dname}", outcome=name, coef=m.params["dose"], se=m.bse["dose"], n=n))
for name in ["treatable", "life_exp"]:
    X = pd.DataFrame({"dose": dose_main, "pre": series(name).loc[2009] - series(name).loc[2004]})
    m, n = reg(long_diff(name, 5), X, drop=())
    rows.append(dict(check="breaks kept in", outcome=name, coef=m.params["dose"], se=m.bse["dose"], n=n))
    base_c = reg(long_diff(name, 5), X, drop=BREAKS)[0].params["dose"]
    for c in dose_main.index:
        if c in BREAKS:
            continue
        m, n = reg(long_diff(name, 5), X, drop=list(BREAKS) + [c])
        rows.append(dict(check=f"leave out {c}", outcome=name, coef=m.params["dose"], se=m.bse["dose"], n=n))
    # post-2013 reversal: did spending recover, and did the horizon matter
    X2 = pd.DataFrame({"dose": dose_main, "pre": series(name).loc[2009] - series(name).loc[2004],
                       "pub": pub_chg})
    m, n = reg(long_diff(name, 5), X2, drop=BREAKS)
    rows.append(dict(check="plus change in public share", outcome=name, coef=m.params["dose"], se=m.bse["dose"], n=n))
# influence of Greece (dose -40 log points, GDP -26): long horizons, with and without
for name in ["treatable", "life_exp"]:
    for k in (5, 8):
        X = pd.DataFrame({"dose": dose_main, "pre": series(name).loc[2009] - series(name).loc[2004]})
        for lab, dr in (("with Greece", list(BREAKS)), ("without Greece", list(BREAKS) + ["GRC"])):
            m, n = reg(long_diff(name, k), X, drop=dr)
            rows.append(dict(check=f"h={k} {lab}", outcome=name, coef=m.params["dose"], se=m.bse["dose"], n=n))
R = pd.DataFrame(rows)
R.round(4).to_csv(f"{out}/austerity_robustness.csv", index=False)

# ---- Placebo in time: the same regression for a fake "2004-2008" episode
rows = []
dose_pl = np.log(H.loc[2008] / H.loc[2004]) * 100
for name in ["treatable", "life_exp"]:
    y = series(name)
    for k in (3, 5):
        X = pd.DataFrame({"dose": dose_pl, "pre": y.loc[2004] - y.loc[2000]})
        m, n = reg(y.loc[2004 + k] - y.loc[2004], X, drop=BREAKS)
        rows.append(dict(outcome=name, horizon=k, coef=m.params["dose"], se=m.bse["dose"], n=n))
pd.DataFrame(rows).round(4).to_csv(f"{out}/austerity_placebo_time.csv", index=False)

print("breaks flagged:", BREAKS)
print("cut set:", cut_set, "| no-cut pool:", len(no_cut_pool))
print(screen.head(8).to_string())
print("\nA (pretrend, h=3,5,8):")
print(A[(A.spec == "pretrend") & (A.horizon.isin([3, 5, 8]))].round(3).to_string(index=False))
print("\nB:"); print(B.round(3).to_string(index=False))
print("\nC:"); print(SC.round(3).to_string(index=False))
print("\nD (selected):"); print(R[~R.check.str.startswith("leave")].round(3).to_string(index=False))
print("\ncorr(dose, GDP change), n=%d: %.2f" % (len(dose_main.drop(BREAKS).dropna()), dose_main.drop(BREAKS).corr(gdp_chg.drop(BREAKS))))
lo = R[R.check.str.startswith("leave") & (R.outcome == "treatable")]
print("\nleave-one-out treatable range:", lo.coef.min().round(4), lo.coef.max().round(4))
print("\nplacebo time:"); print(pd.DataFrame(rows).round(3).to_string(index=False))
