"""The five figures, drawn from the tables ranking_numbers.py writes to .cache/results/.

python ranking_numbers.py > output.txt
python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / ".cache" / "results"
FIGURES = HERE / "figures"

BIG = "#2a78d6"  # countries with 10 or more rated reports
THIN = "#eb6834"  # countries with 5-9 rated reports ("thin sample" on the page)
INK = "#2b2b2a"
MUTED = "#73726c"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"
SOURCE = "Data: Euro Top Tech, 'Happiness and taxes in Europe', read 22 Nov 2026. Analysis: Diogo Ribeiro."
TITLES = {"services": "Satisfaction with tax-paid services", "happiness": "Happiness"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
    }
)


def spread(values, gap):
    """Shift sorted label positions apart so neighbours are at least gap apart, keeping their mean."""
    v = np.asarray(values, float)
    order = np.argsort(v)
    out = v[order].copy()
    for _ in range(200):
        moved = False
        for k in range(1, len(out)):
            if out[k] - out[k - 1] < gap:
                mid = (out[k] + out[k - 1]) / 2
                out[k - 1], out[k] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    res = np.empty_like(out)
    res[order] = out
    return res


def label_side(ax, x, ys, names, gap, side):
    """Labels beside a column of points, spread vertically, with a thin leader where moved."""
    pos = spread(ys, gap)
    dx = -0.06 if side == "left" else 0.06
    for y0, y1, name in zip(ys, pos, names, strict=True):
        ax.annotate(
            name,
            (x, y0),
            xytext=(x + dx, y1),
            textcoords="data",
            ha="right" if side == "left" else "left",
            va="center",
            fontsize=8,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6, "shrinkA": 0, "shrinkB": 2},
        )


def legend_handles():
    big = plt.Line2D([], [], marker="o", ls="", color=BIG, ms=7, label="10 or more reports")
    thin = plt.Line2D([], [], marker="^", ls="", color=THIN, ms=8, label="5 to 9 reports")
    return [big, thin]


def footer(fig, extra=""):
    fig.text(0.01, 0.005, SOURCE + extra, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def fig1_funnel():
    f = pd.read_csv(RESULTS / "funnel.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    nn = np.geomspace(4, 130, 200)
    for ax, v in zip(axes, ("services", "happiness"), strict=True):
        g = f[f.rating == v]
        c, s = g.centre.iloc[0], g.sigma.iloc[0]
        for z, ls, lab in ((1.96, "--", "95% limits"), (3.09, ":", "99.8% limits")):
            ax.plot(nn, c + z * s / np.sqrt(nn), ls, color=MUTED, lw=1.2, label=lab)
            ax.plot(nn, c - z * s / np.sqrt(nn), ls, color=MUTED, lw=1.2)
        ax.axhline(c, color=MUTED, lw=1)
        thin = g.n < 10
        ax.scatter(g.n[~thin], g["mean"][~thin], s=36, color=BIG, zorder=3, edgecolor=SURFACE, lw=1)
        ax.scatter(g.n[thin], g["mean"][thin], s=48, marker="^", color=THIN, zorder=3, edgecolor=SURFACE, lw=1)
        order = g.sort_values("mean")
        label = set(order.country.iloc[:2]) | set(order.country.iloc[-2:]) | set(g.country[np.abs(g.z) > 1.96])
        nudge = {"Romania": (5, 5), "Portugal": (5, -11)}
        for _, r in g[g.country.isin(label)].iterrows():
            ax.annotate(
                r.country,
                (r.n, r["mean"]),
                xytext=nudge.get(r.country, (5, 3)),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.set_xticklabels(["5", "10", "20", "50", "100"])
        ax.set_xlabel("Rated reports from the country (log scale)")
        ax.set_ylabel("Country mean, 0-10")
        ax.set_title(TITLES[v], loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    h, lab = axes[0].get_legend_handles_labels()
    axes[1].legend(
        legend_handles() + h[:2], [x.get_label() for x in legend_handles()] + lab[:2], loc="lower right", fontsize=8
    )
    fig.suptitle(
        "Figure 1. Each country's mean against its number of reports, with the band sampling noise alone would fill",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    footer(fig, " Limits: report-weighted mean +/- z x sigma / sqrt(n), sigma at its posterior median.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGURES / "fig1_funnel.png", dpi=200)
    plt.close(fig)


def fig2_shrinkage():
    r = pd.read_csv(RESULTS / "ranks.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=False)
    for ax, v in zip(axes, ("services", "happiness"), strict=True):
        g = r[r.rating == v]
        for _, row in g.iterrows():
            col = THIN if row.thin else BIG
            ax.plot([0, 1], [row.raw, row.shrunk], color=col, lw=1.6 if row.thin else 1.1, alpha=0.9)
            ax.scatter(
                [0], [row.raw], color=col, s=22 if not row.thin else 30, marker="^" if row.thin else "o", zorder=3
            )
            ax.scatter(
                [1], [row.shrunk], color=col, s=22 if not row.thin else 30, marker="^" if row.thin else "o", zorder=3
            )
        gap = 0.045 * (g.raw.max() - g.raw.min())
        ends = pd.concat([g.nlargest(3, "raw"), g.nsmallest(3, "raw")])
        label_side(
            ax, 0, ends.raw.to_numpy(), [f"{c} ({k})" for c, k in zip(ends.country, ends.n, strict=True)], gap, "left"
        )
        ends2 = pd.concat([g.nlargest(3, "shrunk"), g.nsmallest(3, "shrunk")])
        label_side(
            ax,
            1,
            ends2.shrunk.to_numpy(),
            [f"{c} ({k})" for c, k in zip(ends2.country, ends2.n, strict=True)],
            gap,
            "right",
        )
        ax.set_xlim(-0.8, 1.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Published mean", "Shrunk estimate"])
        ax.set_ylabel("0-10 scale")
        ax.set_title(TITLES[v], loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[1].legend(handles=legend_handles(), loc="lower right", fontsize=8)
    fig.suptitle(
        "Figure 2. From published means to shrunk estimates: the smaller the sample, the further it moves toward the "
        "middle",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    footer(fig, " Labels show rated reports in brackets.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGURES / "fig2_shrinkage.png", dpi=200)
    plt.close(fig)


def fig3_rank_intervals():
    r = pd.read_csv(RESULTS / "ranks.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.4))
    for ax, v in zip(axes, ("services", "happiness"), strict=True):
        g = r[r.rating == v].sort_values("shrunk_rank")
        y = np.arange(len(g))[::-1]
        for yi, (_, row) in zip(y, g.iterrows(), strict=True):
            col = THIN if row.thin else BIG
            ax.plot([row.rank_lo, row.rank_hi], [yi, yi], color=col, lw=3, solid_capstyle="round", alpha=0.85)
            ax.scatter(row.raw_rank, yi, marker="x", color=INK, s=22, zorder=3, lw=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c} ({k})" for c, k in zip(g.country, g.n, strict=True)], fontsize=8)
        ax.set_xlim(0.5, 29.5)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 29])
        ax.set_xlabel("Rank (1 = highest)")
        ax.set_title(TITLES[v], loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], color=BIG, lw=3, label="90% rank interval, 10+ reports"),
        plt.Line2D([], [], color=THIN, lw=3, label="90% rank interval, 5-9 reports"),
        plt.Line2D([], [], marker="x", ls="", color=INK, label="Place in the published table"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.025))
    fig.suptitle(
        "Figure 3. Where each country could rank: 90% intervals from the hierarchical model, ordered by shrunk "
        "estimate",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    footer(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(FIGURES / "fig3_rank_intervals.png", dpi=200)
    plt.close(fig)


def fig4_extremes():
    s = pd.read_csv(RESULTS / "extremes_simulation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, v in zip(axes, ("services", "happiness"), strict=True):
        g = s[s.rating == v]
        x = g["T"].to_numpy()
        ax.bar(x - 0.21, g.no_differences, width=0.4, color="#b0aea5", label="No real differences between countries")
        ax.bar(x + 0.21, g.as_estimated, width=0.4, color=BIG, label="Differences as estimated from the data")
        obs = int(g.observed.iloc[0])
        top = max(g.no_differences.max(), g.as_estimated.max())
        ax.annotate(
            f"published table: {obs}",
            (obs, top * 1.02),
            xytext=(obs + 0.15, top * 1.2),
            ha="left",
            fontsize=8.5,
            color=INK,
            arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1},
        )
        ax.axvline(6 * 12 / 29, color=THIN, lw=1.2, ls="--")
        ax.text(
            6 * 12 / 29 - 0.08,
            top * 1.36,
            "expected if size\ndid not matter: 2.5",
            color=INK,
            fontsize=8,
            va="top",
            ha="right",
        )
        ax.set_xticks(range(7))
        ax.set_xlabel("Thin countries among the 3 highest and 3 lowest places")
        ax.set_title(TITLES[v], loc="left")
        ax.set_ylim(0, top * 1.4)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of 20,000 simulated tables")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.04))
    fig.suptitle(
        "Figure 4. In simulated tables with the same sample sizes, the 12 small countries crowd the ends",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    footer(fig)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(FIGURES / "fig4_extremes_simulation.png", dpi=200)
    plt.close(fig)


def fig5_oecd():
    o = pd.read_csv(RESULTS / "oecd.csv")
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    thin = o.thin.astype(bool)
    ax.scatter(o.composite[~thin], o.services[~thin], s=40, color=BIG, edgecolor=SURFACE, lw=1, zorder=3)
    ax.scatter(o.composite[thin], o.services[thin], s=52, marker="^", color=THIN, edgecolor=SURFACE, lw=1, zorder=3)
    b = np.polyfit(o.composite, o.services, 1)
    xx = np.linspace(o.composite.min() - 2, o.composite.max() + 2, 50)
    ax.plot(xx, np.polyval(b, xx), color=MUTED, lw=1)
    nudge = {"Denmark": (-5, 4, "right"), "Norway": (5, -10, "left"), "Czechia": (-5, 4, "right")}
    for _, r in o.iterrows():
        dx, dy, ha = nudge.get(r.country, (5, 2, "left"))
        ax.annotate(
            f"{r.country} ({r.n})",
            (r.composite, r.services),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            ha=ha,
        )
    ax.set_xlabel("OECD 2023: percent satisfied (mean of healthcare, education, administration)")
    ax.set_ylabel("Engineers' rating of tax-paid services, 0-10")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles(), loc="lower right", fontsize=8)
    fig.suptitle(
        "Figure 5. Engineers' ratings against a population survey, 20 countries in both",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.text(
        0.01,
        0.005,
        "Data: Euro Top Tech (read 22 Nov 2026); OECD, Government at a Glance 2025 (Trust Survey 2023). Analysis: "
        "Diogo Ribeiro.",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGURES / "fig5_oecd.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    fig1_funnel()
    fig2_shrinkage()
    fig3_rank_intervals()
    fig4_extremes()
    fig5_oecd()


if __name__ == "__main__":
    main()
