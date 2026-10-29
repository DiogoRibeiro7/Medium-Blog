"""Figures for the article on job duration by age at hire.

fig1_two_estimands.png        The 2019-2021 break in the published table, and what averaging per worker does by itself
fig2_survival_from_hire.png   Share of hires still with the employer, by age at start, under both published estimands
fig3_restricted_mean.png      Expected years with the employer within H years of the hire, with worst cases
fig4_hires_per_seat.png       Hires needed to keep one position filled for H years, by age at hire
fig5_stock_vs_flow.png        Median tenure of current jobs (stock) against the median duration of new hires (flow)

Run after tenure_numbers.py (which writes results/model.npz):  python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tenure_core import RESULTS, interval_hazards, knots_from_F, nlsy79_rows, pick, read_csv, rmst, survival_at

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
# categorical slots in fixed order, one per age-at-start group, always with direct labels
AGE = {"18-24": "#2a78d6", "25-34": "#eb6834", "35-44": "#1baf7a", "45-54": "#eda100", "55-58": "#e87ba4"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})

rows = nlsy79_rows()
HIRE = {g: pick(rows, "2019-08-22", g) for g in ["18-24", "25-34", "35-44", "45-52"]}
WORKER = {g: pick(rows, "2025-08-26", g) for g in ["18-24", "25-34", "35-44", "45-54", "55-58"]}
M = np.load(RESULTS / "model.npz")
BANDS = ["18-24", "25-34", "35-44", "45-54", "55-58"]


def source(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def curve(ax, r: dict, color: str, label: str, dy: float = 0, leader: bool = False) -> None:
    t, S = knots_from_F(r["F"])
    x = np.linspace(0, t[-1], 400)
    ax.plot(x, 100 * survival_at(x, t, S), color=color, lw=2.2)
    ax.plot(t, 100 * S, "o", color=color, ms=5, mec=SURFACE, mew=1)
    if leader:   # crowded end point: label placed above, with a thin leader line
        ax.annotate(label, (t[-1], 100 * S[-1]), xytext=(t[-1] + 1.2, 100 * S[-1] + 13), textcoords="data",
                    color=INK2, fontsize=8.5, va="center",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8, "shrinkA": 2, "shrinkB": 4})
    else:
        ax.annotate(label, (t[-1], 100 * S[-1]), xytext=(6, dy), textcoords="offset points",
                    color=INK2, fontsize=8.5, va="center")


# ---------------------------------------------------------------------------
# Figure 1: survival from hire
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
LABEL_DY = {("hire", "25-34"): 7, ("hire", "18-24"): 0, ("worker", "45-54"): 6}
for ax, table, title, kind in [(axes[0], HIRE, "Hire-weighted (2019 release)", "hire"),
                               (axes[1], WORKER, "Worker-averaged (2025 release)", "worker")]:
    for g, r in table.items():
        key = "45-54" if g == "45-52" else g
        curve(ax, r, AGE[key], g, LABEL_DY.get((kind, g), 0), leader=(kind == "worker" and g == "55-58"))
    ax.set_title(title)
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 1, 2, 5, 10, 15])
    ax.set_xlabel("Years since the start of the job")
axes[0].set_ylabel("Jobs still in progress (%)")
fig.suptitle("Share of new jobs that last at least t years, by age at the start of the job",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
source(fig, "Data: BLS, NLSY79 (cohort born 1957-64), Table 2 of the 2019 and 2025 releases. Points are published cells; "
            "lines are constant hazards between them. Diogo Ribeiro.")
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(OUT / "fig2_survival_from_hire.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: the two estimands
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
releases = ["2017-08-24", "2019-08-22", "2021-08-31", "2023-08-22", "2025-08-26"]
years = [2017, 2019, 2021, 2023, 2025]
older = {"2017-08-24": "45-50", "2019-08-22": "45-52", "2021-08-31": "45-54", "2023-08-22": "45-56", "2025-08-26": "45-54"}
ax.axvspan(2019.15, 2020.85, color=GRID, alpha=0.6, lw=0)
ax.text(2020, 9, "wording changes\nfrom 'jobs' to\n'the average worker'", ha="center", va="center", fontsize=8, color=INK2)
for g in ["18-24", "25-34", "35-44", "45+"]:
    vals = [100 * pick(rows, rel, older[rel] if g == "45+" else g)["F"][0] for rel in releases]
    col = AGE["45-54" if g == "45+" else g]
    ax.plot(years, vals, "-o", color=col, lw=2.2, ms=5.5, mec=SURFACE, mew=1)
    ax.annotate(g, (2025, vals[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK2, va="center")
ax.set_xticks(years)
ax.set_xlim(2016.4, 2026.3)
ax.set_ylim(0, 78)
ax.set_ylabel("Jobs that ended within a year (%)")
ax.set_xlabel("BLS release (same cohort, same jobs)")
ax.set_title("The published first-year exit share, release by release")

ax = axes[1]
pooled_pub = [HIRE[g]["F"][0] for g in ["18-24", "25-34", "35-44", "45-52"]] + [np.nan]
person_pub = [WORKER[g]["F"][0] for g in BANDS]
zero_person = M["zero_person"]
y = np.arange(len(BANDS))[::-1]
for i, g in enumerate(BANDS):
    p0 = 100 * (pooled_pub[i] if not np.isnan(pooled_pub[i]) else M["zero_pooled"][i])
    ax.plot([100 * zero_person[i], p0], [y[i], y[i]], color=GRID, lw=3, zorder=1)
    ax.plot(p0, y[i], "o", ms=8, color=INK2 if not np.isnan(pooled_pub[i]) else "none",
            mec=INK2, mew=1.6, zorder=3)
    ax.plot(100 * person_pub[i], y[i], "s", ms=11, color="none", mec=INK, mew=1.4, zorder=4)
    ax.plot(100 * zero_person[i], y[i], "D", ms=5, color=INK, zorder=5)
ax.set_yticks(y)
ax.set_yticklabels(BANDS)
ax.set_xlim(10, 78)
ax.set_xlabel("Jobs that ended within a year (%)")
ax.set_title("Averaging per worker, with identical workers")
ax.grid(axis="y", visible=False)
handles = [plt.Line2D([], [], marker="o", ls="", color=INK2, ms=7, label="pooled over jobs, published 2019"),
           plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, ms=7, label="pooled over jobs, model (55-58)"),
           plt.Line2D([], [], marker="s", ls="", mfc="none", mec=INK, ms=9, label="average worker, published 2025"),
           plt.Line2D([], [], marker="D", ls="", color=INK, ms=5, label="average worker, model with identical workers")]
ax.legend(handles=handles, loc="lower right", fontsize=7.8)
fig.suptitle("Same cohort, same jobs, two statistics: the drop between releases is what averaging per worker does",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
source(fig, "Data: BLS, NLSY79 Table 2, releases 2017-2025. Right panel: career model calibrated to the pooled cells, "
            "frailty variance 0.001, then averaged per worker. Diogo Ribeiro.")
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(OUT / "fig1_two_estimands.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: stock versus flow
# ---------------------------------------------------------------------------

cps = {r["age_group"]: float(r["jan2026"]) for r in read_csv("cps_median_tenure.csv")}
age_labels = ["20-24", "25-34", "35-44", "45-54", "55-64", "65+"]
x = np.arange(len(age_labels))
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.plot(x, [cps[a] for a in age_labels], "-o", color=BLUE, lw=2.2, ms=7, mec=SURFACE, mew=1,
        label="stock: median tenure of current jobs, CPS January 2026")
ax.plot(x, M["stock"], "-s", color=ORANGE, lw=2, ms=6, mec=SURFACE, mew=1,
        label="stock: same statistic in the career model")
ax.plot(x, M["stock_flat"], "--", color=MUTED, lw=1.8, marker="^", ms=6,
        label="stock: career model with the 25-34 hiring curve at every age")
flow_x = np.array([0, 1, 2, 3, 4])  # age at start 18-24 ... 55-58 plotted under the nearest current-age group
ax.plot(flow_x, M["flow"], "-D", color=AQUA, lw=2, ms=6, mec=SURFACE, mew=1,
        label="flow: median time a new hire stays, by age at hire (18-24 ... 55-58)")
ax.set_xticks(x)
ax.set_xticklabels(age_labels)
ax.set_xlabel("Age group (current age for the stock lines, age at hire for the flow line)")
ax.set_ylabel("Years")
ax.set_ylim(0, 11)
ax.legend(loc="upper left", fontsize=8.3)
ax.set_title("What a cross-section of current jobs shows, and what a new hire does")
source(fig, "Data: BLS Employee Tenure in 2026 (Table 1); career model calibrated to NLSY79 Table 2. Diogo Ribeiro.")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "fig5_stock_vs_flow.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: restricted mean time with the employer, data only
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9.5, 5))
Hs = np.linspace(0.05, 15, 300)
dy = {"35-44": -7, "25-34": 0, "18-24": 0}
for g in ["18-24", "25-34", "35-44"]:
    t, S = knots_from_F(WORKER[g]["F"])
    ax.plot(Hs, [rmst(H, t, S) for H in Hs], color=AGE[g], lw=2.2)
    ax.annotate(g, (15, rmst(15, t, S)), xytext=(6, dy[g]), textcoords="offset points", fontsize=8.5, color=INK2, va="center")
for g, up in [("45-54", 7), ("55-58", 0)]:
    t, S = knots_from_F(WORKER[g]["F"])
    obs = Hs[Hs <= t[-1]]
    ax.plot(obs, [rmst(H, t, S) for H in obs], color=AGE[g], lw=2.6)
    beyond = Hs[Hs >= t[-1]]
    worst = np.full(beyond.size, rmst(t[-1], t, S))
    carry = [rmst(H, t, S, tail_hazard=interval_hazards(t, S)[-1]) for H in beyond]
    ax.plot(beyond, worst, color=AGE[g], lw=1.6, ls=":")
    ax.plot(beyond, carry, color=AGE[g], lw=1.8, ls="--")
    ax.plot(t[-1], rmst(t[-1], t, S), "o", color=AGE[g], ms=6, mec=SURFACE)
    ax.annotate(f"{g}, exit rate continues", (15, carry[-1]), xytext=(6, up), textcoords="offset points",
                fontsize=8.5, color=INK2, va="center")
    ax.annotate(f"{g}, worst case", (15, worst[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=INK2, va="center")
ax.set_xlim(0, 20.5)
ax.set_ylim(0, 6.6)
ax.set_xticks([0, 1, 2, 5, 10, 15])
ax.set_xlabel("Horizon H (years after the hire)")
ax.set_ylabel("Expected years with the employer within H")
ax.set_title("Expected time with the employer within the first H years (worker-averaged estimand)")
handles = [plt.Line2D([], [], color=INK2, lw=2.2, label="published cells, constant hazard between them"),
           plt.Line2D([], [], color=INK2, lw=1.8, ls="--", label="after the data stop: last observed exit rate continues"),
           plt.Line2D([], [], color=INK2, lw=1.6, ls=":", label="after the data stop: every remaining hire leaves at once")]
ax.legend(handles=handles, loc="upper left", fontsize=8.2)
source(fig, "Data: BLS, NLSY79 Table 2, 2025 release. Diogo Ribeiro.")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "fig3_restricted_mean.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: hires per seat
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9.5, 4.8))
grid = M["seat_grid"]
seat_dy = {"35-44": 6, "55-58": -4, "45-54": -6}
for g in BANDS:
    c = M[f"seats_{g}"]
    ax.plot(np.concatenate([[0], grid]), np.concatenate([[1], c]), color=AGE[g], lw=2.2)
    ax.annotate(g, (grid[-1], c[-1]), xytext=(6, seat_dy.get(g, 0)), textcoords="offset points",
                fontsize=8.5, color=INK2, va="center")
ax.set_xlim(0, 11)
ax.set_ylim(0, 9.8)
ax.set_xticks([0, 1, 2, 5, 10])
ax.set_xlabel("Years the position must stay filled")
ax.set_ylabel("Expected number of hires")
ax.set_title("Hires needed to keep one position filled, if every replacement is hired at the same age")
source(fig, "Career model calibrated to BLS NLSY79 Table 2 (hire-weighted durations, followed to retirement). Diogo Ribeiro.")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "fig4_hires_per_seat.png", dpi=200)
plt.close(fig)

print("wrote", ", ".join(sorted(p.name for p in OUT.glob("fig*.png"))))
