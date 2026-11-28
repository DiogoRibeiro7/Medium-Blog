"""Six figures for the austerity event-study article.
Usage: python3 -I austerity_figures.py RESULTS_DIR AUS_DIR OUT_DIR PANEL_CSV
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res, aus, out, panel = sys.argv[1:5]
BLUE, RED, GREY, LIGHT, INK, MUTED = "#257de0", "#c4452d", "#9c9c9c", "#dddddd", "#191919", "#666666"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

d = pd.read_csv(panel).sort_values(["iso", "year"])
H = d.pivot(index="year", columns="iso", values="hexp_pc_const")
T = d.pivot(index="year", columns="iso", values="trtm")
L = d.pivot(index="year", columns="iso", values="le0")
doses = pd.read_csv(f"{aus}/austerity_doses.csv", index_col=0)
brk = list(doses.index[doses.flagged_break])
cut = ["GRC", "PRT", "ESP", "ITA", "ISL"]
pool = [c for c in doses.index if doses.dose[c] > 0 and c not in brk]
A = pd.read_csv(f"{aus}/austerity_dose_response.csv")
grid = dict(color="#eeeeee", lw=0.8, zorder=0)

# Fig 1: spending paths
yrs = list(range(2005, 2020))
idx = H.loc[yrs] / H.loc[2009] * 100
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=160)
lo, hi = idx[pool].quantile(0.1, axis=1), idx[pool].quantile(0.9, axis=1)
ax.fill_between(yrs, lo, hi, color=LIGHT, zorder=1, label="Countries without cuts (10th-90th percentile)")
ax.plot(yrs, idx[pool].median(axis=1), color=GREY, lw=2, zorder=2)
cols = {"GRC": RED, "PRT": "#e08a1e", "ESP": BLUE, "ITA": "#1c9b7a", "ISL": "#7a4fbf"}
for c in cut:
    ax.plot(yrs, idx[c], color=cols[c], lw=2, zorder=3)
    ax.annotate(c, (2019, idx[c].iloc[-1]), xytext=(5, {"PRT": 6, "ESP": -6}.get(c, 0)), textcoords="offset points", fontsize=9, color=cols[c], va="center")
ax.axvline(2009, color="#bbbbbb", lw=1, ls=":")
ax.set_ylabel("Real health spending per person, 2009 = 100")
ax.set_title("Five countries cut real health spending after 2009; most others kept growing", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(**grid); fig.tight_layout(); fig.savefig(f"{out}/fig1_spending_paths.png"); plt.close(fig)

# Fig 2: dose-response paths
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), dpi=160)
for ax, (name, ylab, ttl) in zip(axs, [("treatable", "Treatable mortality, % per 1% of spending", "Treatable mortality"),
                                        ("life_exp", "Life expectancy, years per 1% of spending", "Life expectancy")]):
    a = A[(A.outcome == name) & (A.spec == "pretrend")].sort_values("horizon")
    ax.axhline(0, color="#bbbbbb", lw=1)
    ax.fill_between(a.horizon, a.coef - 1.96 * a.se, a.coef + 1.96 * a.se, color=BLUE, alpha=0.15, lw=0)
    ax.plot(a.horizon, a.coef, color=BLUE, lw=2, marker="o", ms=4)
    ax.set_xlabel("Years after 2009"); ax.set_ylabel(ylab); ax.set_title(ttl, loc="left", fontsize=11, color=INK)
    ax.set_xticks(range(1, 11)); ax.grid(**grid)
fig.suptitle("Spending change 2009-13 against later mortality change, 95% intervals (HC3)", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f"{out}/fig2_dose_response.png"); plt.close(fig)

# Fig 3: synthetic control gaps, life expectancy
fig, axs = plt.subplots(1, 5, figsize=(13, 3.6), dpi=160, sharey=True)
for ax, u in zip(axs, ["ESP", "ITA", "PRT", "ISL", "GRC"]):
    pl = pd.read_csv(f"{aus}/austerity_placebo_life_exp_{u}.csv", index_col=0)
    g = pd.read_csv(f"{aus}/austerity_synth_gaps.csv", header=[0], index_col=0)[f"life_exp|{u}"]
    pl, g = pl.loc[:2019], g.loc[:2019]
    for c in pl.columns:
        ax.plot(pl.index, pl[c], color=LIGHT, lw=1, zorder=1)
    ax.plot(g.index, g.values, color=cols[u], lw=2.2, zorder=3)
    ax.axhline(0, color="#bbbbbb", lw=1); ax.axvline(2010, color="#bbbbbb", lw=1, ls=":")
    ax.set_title(u, loc="left", fontsize=11, color=INK); ax.set_xticks([2000, 2010, 2019]); ax.set_ylim(-1.6, 1.6); ax.grid(**grid)
axs[0].set_ylabel("Gap in life expectancy vs synthetic control, years")
fig.suptitle("Life expectancy gap against a synthetic control (grey: the same exercise run on countries without cuts)", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f"{out}/fig3_synthetic_control.png"); plt.close(fig)

# Fig 4: dose vs GDP
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=160)
x = doses[~doses.flagged_break]
ax.scatter(x.gdp_change_2009_2013, x.dose, s=34, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
b, a0 = np.polyfit(x.gdp_change_2009_2013, x.dose, 1)
xs = np.linspace(x.gdp_change_2009_2013.min(), x.gdp_change_2009_2013.max(), 50)
ax.plot(xs, a0 + b * xs, color=GREY, lw=2, zorder=2)
for c in ["GRC", "PRT", "ESP", "ITA", "ISL", "KOR", "CHL", "EST", "USA", "DEU"]:
    if c in x.index:
        ax.annotate(c, (x.gdp_change_2009_2013[c], x.dose[c]), xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
ax.set_xlabel("Change in real GDP per person, 2009-13, log points x 100")
ax.set_ylabel("Change in real health spending per person, 2009-13, log points x 100")
ax.set_title("Where health spending was cut, the whole economy was shrinking (correlation 0.79)", loc="left", fontsize=11, color=INK)
ax.grid(**grid); fig.tight_layout(); fig.savefig(f"{out}/fig4_spending_vs_gdp.png"); plt.close(fig)

# Fig 5: one elasticity, four designs
pr = pd.read_csv(f"{res}/panel_results.csv").set_index("model")
a5 = A[(A.outcome == "treatable") & (A.horizon == 5)].set_index("spec")
rows = [("Cross-section 2015-19", pr.loc["CS raw", "coef"], pr.loc["CS raw", "se"]),
        ("Two-way fixed effects, lagged spending", pr.loc["FE spend avg lag1-2", "coef"], pr.loc["FE spend avg lag1-2", "se"]),
        ("Cuts 2009-13: long difference, pre-trend", a5.loc["pretrend", "coef"], a5.loc["pretrend", "se"]),
        ("Cuts 2009-13: also controlling for GDP", a5.loc["pretrend+gdp", "coef"], a5.loc["pretrend+gdp", "se"])]
fig, ax = plt.subplots(figsize=(8, 3.8), dpi=160)
for i, (lab, c, se) in enumerate(rows[::-1]):
    ax.plot([c - 1.96 * se, c + 1.96 * se], [i, i], color=BLUE, lw=2.5, solid_capstyle="round")
    ax.plot(c, i, "o", color=BLUE, ms=6)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.axvline(0, color="#bbbbbb", lw=1); ax.set_xlabel("Elasticity of treatable mortality with respect to health spending, 95% interval")
fig.suptitle("Four designs, one question: how do treatable deaths respond to spending?", x=0.01, ha="left", fontsize=11, color=INK)
ax.grid(axis="x", **grid); fig.tight_layout(); fig.savefig(f"{out}/fig5_four_designs.png"); plt.close(fig)

# Fig 6: trajectories, cut vs other
others = [c for c in pool if c not in cut]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), dpi=160)
for ax, (Y, ttl, yl) in zip(axs, [(T, "Treatable deaths per 100,000, 2009 = 100", "Index"), (L, "Life expectancy, years, change since 2009", "Years")]):
    for grp, col, lab in [(["ESP", "ITA", "ISL"], RED, "Cut: Spain, Italy, Iceland"), (others, GREY, "No cut (median)")]:
        yy = Y.loc[2004:2019, grp].dropna(axis=1, how="any")
        yy = (yy / yy.loc[2009] * 100) if Y is T else (yy - yy.loc[2009])
        ax.plot(yy.index, yy.median(axis=1), color=col, lw=2.2, label=lab)
    ax.axvline(2009, color="#bbbbbb", lw=1, ls=":"); ax.set_title(ttl, loc="left", fontsize=10.5, color=INK); ax.grid(**grid)
axs[0].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{out}/fig6_trajectories.png"); plt.close(fig)
print("ok")
