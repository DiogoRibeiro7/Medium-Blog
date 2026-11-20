"""Draw the article's figures from the tables heat_numbers.py writes to results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import heat_core as hc  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LIGHT_BLUE, LIGHT_ORANGE, NEUTRAL = "#cde2fb", "#fbd5c4", "#f1f0ea"
DARK_BLUE, DARK_ORANGE = "#184f94", "#a8431b"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)
CREDIT = "Data: Eurostat, household electricity (nrg_pc_204, band DD) and gas prices (nrg_pc_202, band D2), all taxes."
CREDIT += " Code: Medium-Blog/heat-pump-price-ratio. Figure: Diogo Ribeiro."
SPF_LOW, SPF_HIGH = hc.FIELD_SPF.values()


def frame(fig, title, subtitle, top_in=0.85, bottom_in=0.7, credit=CREDIT):
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(
        0.01,
        1 - 0.08 / h,
        title,
        fontsize=11,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.01,
        1 - 0.36 / h,
        subtitle,
        fontsize=8.5,
        color=INK2,
        ha="left",
        va="top",
        linespacing=1.4,
    )
    fig.text(0.01, 0.06 / h, credit, fontsize=7.2, color=MUTED, ha="left", va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def name(row):
    last = row.latest_half if "latest_half" in row else row.last_half
    return row["name"] + ("" if last == hc.LAST else f" (data to {last[:4]})")


def fig1_ratio():
    t = pd.read_csv(RES / "ratio_by_country.csv").sort_values("latest").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 7.6))
    y = np.arange(len(t))
    ax.hlines(
        y,
        t.lowest,
        t.highest,
        color=GRID,
        linewidth=6,
        label="Lowest to highest, 2007–2026",
    )
    ax.scatter(
        t["median"],
        y,
        marker="|",
        s=140,
        color=INK2,
        linewidth=1.6,
        zorder=3,
        label="Median, 2007–2026",
    )
    cheap = t.latest < SPF_LOW / hc.ETA
    ax.scatter(t.latest[cheap], y[cheap], s=60, color=BLUE, zorder=4, ec="white", lw=1.2,
               label=f"Latest year: a heat pump with SPF {SPF_LOW:.1f} is cheaper to run")  # fmt: skip
    ax.scatter(t.latest[~cheap], y[~cheap], s=60, color=ORANGE, zorder=4, ec="white", lw=1.2,
               label="Latest year: it is not")  # fmt: skip
    for spf, text, side in (
        (SPF_LOW, f"break-even at SPF {SPF_LOW:.1f}", -1),
        (SPF_HIGH, f"at SPF {SPF_HIGH:.1f}", 1),
    ):
        x = spf / hc.ETA
        ax.axvline(x, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.text(
            x + 0.04 * side,
            -0.75,
            text,
            fontsize=8,
            color=INK2,
            va="top",
            ha="left" if side > 0 else "right",
        )
    ax.set_yticks(y, [name(r) for _, r in t.iterrows()])
    ax.set_ylim(len(t) - 0.4, -1.3)
    ax.set_xlim(0.5, 5.6)
    grid(ax, "x")
    ax.set_xlabel("Electricity price ÷ gas price, per kWh, all taxes included")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.94),
        fontsize=8.2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
    )
    frame(
        fig,
        "In the latest year, electricity cost from 1.1 to 4.6 times as much as gas, per kWh",
        "Household prices, half-yearly, 2007 to 2026. Latest year: second half of 2025 and first half of 2026"
        " (earlier for Poland and the UK).\nDashed lines: the ratio below which a heat pump is cheaper to run than a"
        f" condensing boiler with {hc.ETA:.0%} seasonal efficiency.",
        top_in=1.0,
    )
    fig.savefig(FIG / "fig1_ratio.png")
    plt.close(fig)


def fig2_heatmap():
    p = pd.read_csv(RES / "panel.csv")
    halves = sorted(p.half.unique(), key=hc.half)
    order = p.groupby("country").ratio.median().sort_values(ascending=False).index
    m = p.pivot(index="country", columns="half", values="breakeven_spf").reindex(index=order, columns=halves)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "be", [DARK_BLUE, BLUE, LIGHT_BLUE, NEUTRAL, LIGHT_ORANGE, ORANGE, DARK_ORANGE]
    )
    cmap.set_bad("white")
    norm = mcolors.TwoSlopeNorm(vmin=0.7, vcenter=SPF_LOW, vmax=4.9)
    fig, ax = plt.subplots(figsize=(10, 7.4))
    im = ax.imshow(
        np.ma.masked_invalid(m.values),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(order)), [hc.NAMES[c] for c in order], fontsize=8.5)
    ticks = [i for i, h in enumerate(halves) if h.endswith("S1") and int(h[:4]) % 2 == 0]
    ax.set_xticks(ticks, [halves[i][:4] for i in ticks], fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(halves)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[1, 1.5, 2, SPF_LOW, 3.5, 4, 4.5])
    cb.set_label(
        "Seasonal performance factor a heat pump needs to match the boiler",
        fontsize=8.5,
        color=INK2,
    )
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    frame(
        fig,
        "How good a heat pump had to be to beat a gas boiler, by country and half-year",
        f"Electricity ÷ gas price × {hc.ETA:.2f} (condensing boiler). Blue: below {SPF_LOW:.1f}, the UK field median,"
        " so a typical heat pump was cheaper to run. Orange: above it.\nCountries ordered by their median ratio."
        " White: no data.",
        top_in=0.95,
    )
    fig.savefig(FIG / "fig2_heatmap.png")
    plt.close(fig)


def fig3_share():
    t = pd.read_csv(RES / "share_by_spf.csv")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        t.spf,
        t.all_half_years,
        color=BLUE,
        linewidth=2.2,
        label="All country half-years, 2007–2026",
    )
    ax.plot(
        t.spf,
        t.latest_year,
        color=ORANGE,
        linewidth=2.2,
        label="Latest year, 23 countries",
    )
    ax.plot(t.spf, t.all_half_years_boiler_085, color=MUTED, linewidth=1.6, linestyle=(0, (4, 3)),
            label="All country half-years, boiler at 85% instead of 90%")  # fmt: skip
    for spf, text in (
        (SPF_LOW, "UK field median\n(Electrification of Heat)"),
        (SPF_HIGH, "German field mean\n(Fraunhofer ISE)"),
    ):
        ax.axvline(spf, color=INK2, linewidth=0.9)
        ax.text(spf + 0.03, 0.05, text, fontsize=8, color=INK2, va="bottom")
        for col, c, off, ha in (
            ("all_half_years", BLUE, (8, -12), "left"),
            ("latest_year", ORANGE, (-6, 6), "right"),
        ):
            v = float(t.loc[(t.spf - spf).abs().idxmin(), col])
            ax.scatter([spf], [v], s=38, color=c, zorder=4, ec="white", lw=1)
            ax.annotate(
                f"{v:.0%}",
                (spf, v),
                textcoords="offset points",
                xytext=off,
                ha=ha,
                fontsize=8,
                color=c,
            )
    ax.set_xlim(2.0, 5.0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    grid(ax, "y")
    ax.set_xlabel("Seasonal performance factor of the heat pump (kWh of heat per kWh of electricity)")
    ax.set_ylabel("Share in which the heat pump was cheaper to run")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=8.2)
    frame(
        fig,
        "A heat pump as good as the field median beat a gas boiler in about three country half-years out of four",
        "Share of country half-years in which a heat pump of a given efficiency cost less to run than a condensing"
        f" boiler ({hc.ETA:.0%} seasonal efficiency),\nat household prices with all taxes, 25 countries, 2007-S2 to"
        " 2026-S1.",
        top_in=0.95,
    )
    fig.savefig(FIG / "fig3_share.png")
    plt.close(fig)


def fig4_taxes():
    t = pd.read_csv(RES / "taxes.csv").sort_values("ratio_all_taxes", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 7.4))
    y = np.arange(len(t))
    for i, r in t.iterrows():
        c = ORANGE if r.ratio_all_taxes > r.ratio_without_taxes else BLUE
        ax.annotate("", xy=(r.ratio_all_taxes, i), xytext=(r.ratio_without_taxes, i),
                    arrowprops={"arrowstyle": "-|>", "color": c, "lw": 1.8, "shrinkA": 3, "shrinkB": 4})  # fmt: skip
    ax.scatter(t.ratio_without_taxes, y, s=40, facecolor="white", edgecolor=INK2, lw=1.3, zorder=3,
               label="Before taxes and levies")  # fmt: skip
    ax.scatter(
        t.ratio_all_taxes,
        y,
        s=46,
        color=INK,
        zorder=3,
        label="With all taxes and levies (what households pay)",
    )
    ax.plot(
        [],
        [],
        color=ORANGE,
        lw=1.8,
        label="Taxes raise the ratio (penalise the heat pump)",
    )
    ax.plot([], [], color=BLUE, lw=1.8, label="Taxes lower the ratio (favour the heat pump)")
    ax.set_yticks(y, [name(r) for _, r in t.iterrows()])
    ax.set_ylim(len(t) - 0.4, -0.8)
    ax.set_xlim(0.8, 4.9)
    grid(ax, "x")
    ax.set_xlabel("Electricity price ÷ gas price, per kWh")
    ax.legend(loc="lower right", fontsize=8.2)
    frame(
        fig,
        "Taxes and levies widen the price gap in some countries and close it in others",
        "Ratio of household electricity to gas prices before any taxes and levies, and with all of them, latest year"
        " (second half of 2025 and\nfirst half of 2026; earlier for Poland and the UK).",
        top_in=0.95,
    )
    fig.savefig(FIG / "fig4_taxes.png")
    plt.close(fig)


def fig5_savings():
    t = pd.read_csv(RES / "latest.csv").sort_values("saving_34").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 7.4))
    y = np.arange(len(t))
    ax.hlines(y, t.saving_28, t.saving_34, color=GRID, linewidth=4)
    ax.scatter(t.saving_28, y, s=44, color=LIGHT_BLUE, ec=BLUE, lw=1.2, zorder=3,
               label=f"Heat pump with SPF {SPF_LOW:.1f} (UK field median)")  # fmt: skip
    ax.scatter(t.saving_34, y, s=50, color=BLUE, ec="white", lw=1.2, zorder=4,
               label=f"Heat pump with SPF {SPF_HIGH:.1f} (German field mean)")  # fmt: skip
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, [name(r) for _, r in t.iterrows()])
    ax.set_ylim(-0.8, len(t) - 0.2)
    grid(ax, "x")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel(f"Euros saved a year against a condensing gas boiler, home needing {hc.HEAT:,.0f} kWh of heat")
    ax.text(-30, len(t) - 0.6, "costs more ←", ha="right", fontsize=8, color=INK2)
    ax.text(30, len(t) - 0.6, "→ saves", ha="left", fontsize=8, color=INK2)
    ax.legend(loc="lower right", fontsize=8.2)
    frame(
        fig,
        "Running costs saved by a heat pump at the latest year's prices, from −312 to 1,706 euros",
        "Household prices with all taxes, latest year (second half of 2025 and first half of 2026; earlier for Poland"
        f" and the UK). Boiler at {hc.ETA:.0%}\nseasonal efficiency. Running costs only, at average prices per kWh"
        " (standing charges spread over the kWh); installation and maintenance are not included.",
        top_in=0.95,
    )
    fig.savefig(FIG / "fig5_savings.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig1_ratio()
    fig2_heatmap()
    fig3_share()
    fig4_taxes()
    fig5_savings()


if __name__ == "__main__":
    main()
