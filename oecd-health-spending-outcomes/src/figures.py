"""Two figures: spending vs treatable mortality (cross-section) and rank ranges across 72 specs.
Usage: python3 -I figures.py RESULTS_DIR OUT_DIR PANEL_CSV
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res, out, panel = sys.argv[1], sys.argv[2], sys.argv[3]
BLUE, GREY, INK, MUTED = "#257de0", "#9c9c9c", "#191919", "#666666"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: cross-section scatter
cs = pd.read_csv(f"{res}/cross_section_2015_19.csv", index_col=0)
x, y = np.exp(cs.l_h), np.exp(cs.l_trtm)
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=160)
ax.scatter(x, y, s=34, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
b, a = np.polyfit(cs.l_h, cs.l_trtm, 1)
xs = np.linspace(cs.l_h.min(), cs.l_h.max(), 100)
ax.plot(np.exp(xs), np.exp(a + b * xs), color=GREY, lw=2, zorder=2)
for c in ["USA", "CHE", "KOR", "MEX", "LTU", "ESP", "DEU", "JPN"]:
    if c in cs.index:
        ax.annotate(c, (x[c], y[c]), xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1500, 3000, 6000, 12000]); ax.set_xticklabels(["1.5k", "3k", "6k", "12k"])
ax.set_yticks([40, 60, 100, 150]); ax.set_yticklabels(["40", "60", "100", "150"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Health spending per person, USD PPP (constant prices), 2015-19 mean, log scale")
ax.set_ylabel("Treatable deaths per 100,000, log scale")
ax.set_title("Across countries, more spending goes with fewer treatable deaths (elasticity -0.51)", loc="left", fontsize=11, color=INK)
ax.grid(color="#eeeeee", lw=0.8, zorder=0)
fig.tight_layout(); fig.savefig(f"{out}/fig1_spending_vs_treatable.png"); plt.close(fig)

# Fig 2: rank ranges
s = pd.read_csv(f"{res}/efficiency_summary.csv", index_col=0).sort_values("median", ascending=False)
fig, ax = plt.subplots(figsize=(7.5, 8), dpi=160)
yy = np.arange(len(s))
ax.hlines(yy, s.p10, s.p90, color=GREY, lw=3, zorder=2)
ax.scatter(s["median"], yy, s=40, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
ax.set_yticks(yy); ax.set_yticklabels(s.index, fontsize=8)
ax.invert_xaxis() if False else None
ax.set_xlim(0.5, 36.5); ax.set_xlabel("Efficiency rank (1 = best); dot = median, bar = 10th-90th percentile over 72 specifications")
ax.set_title("Efficiency rank depends on analyst choices, except at the extremes", loc="left", fontsize=11, color=INK)
ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
fig.tight_layout(); fig.savefig(f"{out}/fig2_rank_ranges.png"); plt.close(fig)

# Fig 4: within-country estimates of the spending elasticity under different designs
rb = pd.read_csv(f"{res}/robustness.csv")
rb = rb.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
yy = np.arange(len(rb))
ax.axvline(0, color="#cccccc", lw=1.2, zorder=1)
ax.hlines(yy, rb.lo, rb.hi, color=GREY, lw=3, zorder=2)
ax.scatter(rb.coef, yy, s=40, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
ax.set_yticks(yy); ax.set_yticklabels(rb.model, fontsize=8.5)
ax.set_xlabel("Spending elasticity of treatable mortality (estimate, 95% interval)")
ax.set_title("Within countries the spending effect is small and mostly near zero", loc="left", fontsize=11, color=INK)
ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
fig.tight_layout(); fig.savefig(f"{out}/fig4_within_country_estimates.png"); plt.close(fig)

# Fig 5: pre-pandemic treatable mortality vs excess deaths 2020-22
pa = pd.read_csv(f"{res}/pandemic_all_countries.csv", index_col=0).dropna(subset=["trtm", "excess_crude"])
fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=160)
ax.scatter(pa.trtm, pa.excess_crude, s=38, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
b, a = np.polyfit(np.log(pa.trtm), pa.excess_crude, 1)
xs = np.linspace(pa.trtm.min(), pa.trtm.max(), 100)
ax.plot(xs, a + b * np.log(xs), color=GREY, lw=2, zorder=2)
for c, r_ in pa.iterrows():
    ax.annotate(c, (r_.trtm, r_.excess_crude), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK)
ax.set_xscale("log"); ax.set_xticks([40, 60, 100, 150]); ax.set_xticklabels(["40", "60", "100", "150"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Treatable deaths per 100,000, 2017-19 mean (log scale)")
ax.set_ylabel("Excess deaths per 100,000 per year, Mar 2020 - Dec 2022")
ax.set_title("Routine-care outcomes before 2020 line up with pandemic excess deaths", loc="left", fontsize=11, color=INK)
ax.grid(color="#eeeeee", lw=0.8, zorder=0)
fig.tight_layout(); fig.savefig(f"{out}/fig5_treatable_vs_excess.png"); plt.close(fig)
