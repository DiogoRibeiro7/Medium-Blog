"""Draw the four figures for the article from ``results/``, numbered in the order the article shows them.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import pairs as pr  # noqa: E402

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GROUP_COLOUR = {"Eurostat": SLOTS[0], "OECD": SLOTS[1]}
GROUP_LABEL = {"Eurostat": "PPPs compiled by Eurostat", "OECD": "PPPs compiled by the OECD"}
SOURCE = "Sources: OECD national accounts (Tables 1 and 4), snapshot 13 Nov 2026; author's calculations."
NAMES = {
    "AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria", "HRV": "Croatia", "CYP": "Cyprus", "CZE": "Czechia",
    "DNK": "Denmark", "EST": "Estonia", "FIN": "Finland", "FRA": "France", "DEU": "Germany", "GRC": "Greece",
    "HUN": "Hungary", "IRL": "Ireland", "ITA": "Italy", "LVA": "Latvia", "LTU": "Lithuania", "LUX": "Luxembourg",
    "MLT": "Malta", "NLD": "Netherlands", "POL": "Poland", "PRT": "Portugal", "ROU": "Romania", "SVK": "Slovakia",
    "SVN": "Slovenia", "ESP": "Spain", "SWE": "Sweden", "GBR": "United Kingdom", "CHE": "Switzerland",
    "NOR": "Norway", "ISL": "Iceland", "TUR": "Türkiye", "SRB": "Serbia", "MKD": "North Macedonia",
    "ALB": "Albania", "MNE": "Montenegro", "BIH": "Bosnia and Herz.", "USA": "United States", "JPN": "Japan",
    "KOR": "Korea", "CAN": "Canada", "AUS": "Australia", "NZL": "New Zealand", "ISR": "Israel", "MEX": "Mexico",
    "CHL": "Chile", "COL": "Colombia", "CRI": "Costa Rica",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "legend.labelcolor": INK2,
})


def tidy(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def finish(fig, name: str, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.005, SOURCE, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1() -> None:
    s = pd.read_csv(RES / "hfce_scores.csv", index_col=0)
    order = list(s[s.group == "OECD"].sort_values("drift").index) + list(s[s.group == "Eurostat"].sort_values("drift").index)
    v = s.loc[order, "drift"].values
    m = v[:, None] - v[None, :]          # row economy against column economy
    cmap = LinearSegmentedColormap.from_list("div", [SLOTS[0], "#efeee9", SLOTS[1]])
    fig, ax = plt.subplots(figsize=(10, 10.6))
    fig.subplots_adjust(left=0.16, right=0.90, top=0.88, bottom=0.14)
    im = ax.imshow(m, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-30, vmax=30), interpolation="nearest")
    labels = [NAMES.get(c, c) for c in order]
    ax.set_xticks(range(len(order)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), labels, fontsize=7)
    k = (s.group == "OECD").sum()
    for pos in (k - 0.5,):
        ax.axhline(pos, color=INK, linewidth=1.2)
        ax.axvline(pos, color=INK, linewidth=1.2)
    ax.text(k / 2 - 0.5, -1.2, "OECD", ha="center", fontsize=8.5, color=INK2, fontweight="bold")
    ax.text(k + (len(order) - k) / 2 - 0.5, -1.2, "Eurostat", ha="center", fontsize=8.5, color=INK2, fontweight="bold")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    cax = fig.add_axes([0.92, 0.30, 0.018, 0.40])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8, labelcolor=INK2)
    cb.set_label("drift of row against column, log points ×100", color=INK2, fontsize=8.5)
    finish(fig, "fig1_pair_matrix",
           "Household-consumption drift for every pair of economies, 2005–2019",
           "Row economy against column economy. Economies grouped by who compiles their PPPs, each group sorted by its "
           "drift against the US.\n946 pairs, built from 43 independent numbers.")


def fig2() -> None:
    s = pd.read_csv(RES / "hfce_scores.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.13)
    x = s.ln_pli_2005
    X = np.column_stack([np.ones(len(s)), (s.group == "Eurostat").astype(float), x])
    beta, *_ = np.linalg.lstsq(X, s.drift.values, rcond=None)
    xs = np.linspace(x.min(), x.max(), 20)
    for g in ("Eurostat", "OECD"):
        sub = s[s.group == g]
        ax.scatter(np.exp(sub.ln_pli_2005), sub.drift, s=42, color=GROUP_COLOUR[g], edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=GROUP_LABEL[g])
        e = 1.0 if g == "Eurostat" else 0.0
        ax.plot(np.exp(xs), beta[0] + beta[1] * e + beta[2] * xs, color=GROUP_COLOUR[g], linewidth=1.4,
                linestyle=(0, (4, 2)))
    label = {"USA": (5, 4), "JPN": (5, 4), "KOR": (5, -10), "CAN": (5, -10), "MEX": (5, 4), "DEU": (5, 4),
             "ITA": (5, 4), "SWE": (-5, -11), "IRL": (5, 4), "GBR": (-5, -11), "NOR": (5, 4), "BGR": (5, 4),
             "ROU": (5, -10), "TUR": (5, 4), "HRV": (5, 4), "POL": (-5, 6), "FRA": (5, -10), "CHE": (5, 4)}
    for c, off in label.items():
        if c in s.index:
            ax.annotate(NAMES[c], (np.exp(s.loc[c, "ln_pli_2005"]), s.loc[c, "drift"]), xytext=off,
                        textcoords="offset points", fontsize=8, color=INK2, ha="right" if off[0] < 0 else "left")
    ax.set_xscale("log")
    ax.set_xticks([40, 60, 80, 100, 150], ["40", "60", "80", "100", "150"])
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xlabel("price level of household consumption in 2005, United States = 100 (log scale)")
    ax.set_ylabel("drift 2005–2019 against the US (log points ×100)")
    ax.legend(loc="upper right", fontsize=9)
    tidy(ax, "both")
    finish(fig, "fig2_drift_by_group_and_price_level",
           "Two groups, and a price-level slope inside the European one",
           "Household-consumption drift against the US, 2005–2019. Dashed lines: one fitted slope, separate levels "
           "for the two groups.")


def fig3() -> None:
    st = pd.read_csv(RES / "annual_steps_hfce.csv")
    q = pd.read_csv(RES / "quiet_share.csv", index_col=0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.10, hspace=0.18)
    o = st[st.group == "OECD"]
    rng = np.random.default_rng(7)
    a1.scatter(o.year + rng.uniform(-0.18, 0.18, len(o)), o.step, s=22, color=SLOTS[1], edgecolor=SURFACE,
               linewidth=1, zorder=3)
    a1.axhline(0, color=BASE, linewidth=1)
    for y in pr.BENCHMARKS:
        a1.axvline(y, color=GRID, linewidth=1, zorder=0)
        a2.axvline(y, color=GRID, linewidth=1, zorder=0)
    a1.set_ylim(-8, 8)
    a1.set_ylabel("step against the US\n(log points ×100)")
    a1.set_title("Annual change in the household-consumption drift, eight OECD-compiled economies")
    a1.text(2023.9, -7.4, "vertical lines: benchmark years", color=MUTED, fontsize=8, ha="right")
    tidy(a1)
    for g, col in (("OECD group", SLOTS[1]), ("Eurostat group", SLOTS[0])):
        a2.plot(q.index, q[g], color=col, linewidth=2, marker="o", markersize=4,
                label="PPPs compiled by the OECD" if g == "OECD group" else "PPPs compiled by Eurostat")
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("share of economies with\na step below 0.3 points")
    a2.legend(loc="upper right", fontsize=8.5)
    a2.set_xlim(1995.5, 2024.5)
    tidy(a2)
    finish(fig, "fig3_construction_signature",
           "How the PPP time series were built shows up in the drift",
           "Before 2013 the OECD-compiled series move in pairs of equal steps around benchmark years, with near-zero "
           "steps in between.\nThe Eurostat-compiled series move every year.")


def fig4() -> None:
    g = pd.read_csv(RES / "group_gap_paths.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.11, right=0.82, top=0.84, bottom=0.12)
    lines = [("GDP", "GDP", SLOTS[0]), ("AIC", "Actual individual cons.", SLOTS[1]), ("HFCE", "Household consumption", SLOTS[2])]
    for col, lab, colour in lines:
        ax.plot(g.index, g[col], color=colour, linewidth=2, label=lab)
    ends = sorted(((g[c].iloc[-1], lab, colour) for c, lab, colour in lines), key=lambda x: x[0])
    pos = [ends[0][0]]
    for y, _, _ in ends[1:]:
        pos.append(max(y, pos[-1] + 2.2))
    for (y, lab, colour), yy in zip(ends, pos):
        ax.annotate(lab, xy=(g.index[-1], y), xytext=(g.index[-1] + 0.8, yy), fontsize=8.6, color=INK2, va="center",
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", color=colour, lw=1, shrinkA=0, shrinkB=2))
    for y in pr.BENCHMARKS:
        ax.axvline(y, color=GRID, linewidth=1, zorder=0)
    ax.axvspan(2015.5, 2016.5, color=GRID, alpha=0.6, linewidth=0)
    ax.text(2016, -15.2, "OECD moves to\nannual calculation", ha="center", fontsize=8, color=MUTED)
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xlim(1995, 2024)
    ax.set_ylim(-16, 8)
    ax.set_ylabel("Eurostat-group mean minus OECD-group mean\n(log points ×100, 2020 = 0)")
    ax.legend(loc="upper left", fontsize=8.5)
    tidy(ax)
    finish(fig, "fig4_group_gap_over_time",
           "The gap between the two groups, 1995–2024",
           "Mean drift of the economies whose PPPs Eurostat compiles minus the mean of those the OECD compiles. "
           "Vertical lines: benchmark years.")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        f()
