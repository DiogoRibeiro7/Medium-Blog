"""The six figures, drawn from results/ (run legacy_numbers.py first).

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import legacy_core as lc  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SHADE = "#efeeea"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#7b5cd6"
GREY_LINE = "#b9b8b3"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
    }
)


def heading(fig, title, subtitle):
    fig.text(0.01, 0.985, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.945, subtitle, fontsize=9.5, color=INK2, va="top")


def footer(fig, source):
    fig.text(0.01, 0.008, f"Source: {source} Figure: Diogo Ribeiro.", fontsize=8, color=MUTED, ha="left", va="bottom")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def shade_regime(ax, label_y):
    ax.axvspan(1926, 1974, color=SHADE, zorder=0)
    ax.text(1950, label_y, "Dictatorship, 1926–1974", ha="center", va="bottom", fontsize=9, color=INK2)


def end_label(ax, x, y, text, color, bold=False, dy=0.0):
    ax.text(x + 1.2, y + dy, text, va="center", fontsize=9, color=color if bold else INK2, fontweight="bold" if bold else "normal")


def fig1_relative_income():
    rel = pd.read_csv(RESULTS / "relative_gdp_longrun.csv", index_col=0)
    rel = rel.loc[1900:2022]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.07, right=0.88, top=0.83, bottom=0.11)
    shade_regime(ax, 101)
    others = {"ESP": (BLUE, 0), "GRC": (AQUA, 0), "POL": (VIOLET, 2.0), "HUN": (GREY_LINE, 1.2)}
    for c, (col, dy) in others.items():
        s = rel[c].dropna()
        ax.plot(s.index, s.values, color=col, lw=1.6, zorder=2)
        end_label(ax, s.index[-1], s.values[-1], lc.NAMES[c], col, dy=dy)
    s = rel["PRT"].dropna()
    ax.plot(s.index, s.values, color=ORANGE, lw=2.6, zorder=3)
    end_label(ax, s.index[-1], s.values[-1], "Portugal", ORANGE, bold=True, dy=-1.8)
    for y, off in ((1925, (0, -16)), (1960, (0, -16)), (1973, (-34, 10)), (2000, (0, -16))):
        ax.scatter([y], [s.loc[y]], s=30, color=ORANGE, zorder=4, edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(f"{s.loc[y]:.0f}% in {y}", (y, s.loc[y]), xytext=off, textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlim(1900, 2022)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of the GDP per head of twelve Western European countries")
    heading(
        fig,
        "Portugal began and ended the dictatorship near the bottom of Western Europe",
        "GDP per head relative to Austria, Belgium, Denmark, Finland, France, Germany, Italy, the Netherlands, Norway, Sweden,\n"
        "Switzerland and the United Kingdom combined. The catch-up came in 1960–1973, after the economy opened to trade.",
    )
    footer(fig, "Bolt and van Zanden (2024), Maddison Project Database 2023, GDP via Our World in Data; per head with OWID population.")
    save(fig, "fig1_relative_income")


def fig2_schooling():
    sch = pd.read_csv(RESULTS / "schooling_longrun.csv", index_col=0)
    sch.columns = sch.columns.astype(int)
    years = [y for y in sch.columns if 1900 <= y <= 2010]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.07, right=0.86, top=0.83, bottom=0.11)
    shade_regime(ax, 13.3)
    for c in ("ITA", "FIN", "GRC", "BGR", "IRL"):
        ax.plot(years, sch.loc[c, years], color=GREY_LINE, lw=1.4, zorder=2)
    end_label(ax, years[-1], sch.loc["FIN", years[-1]] - 0.1, "Five others (grey)", INK2)
    ax.plot(years, sch.loc["POL", years], color=VIOLET, lw=1.8, zorder=2)
    end_label(ax, years[-1], sch.loc["POL", years[-1]], "Poland", VIOLET, dy=-0.3)
    ax.plot(years, sch.loc["ESP", years], color=BLUE, lw=1.8, zorder=3)
    end_label(ax, years[-1], sch.loc["ESP", years[-1]], "Spain", BLUE, dy=-0.6)
    ax.plot(years, sch.loc["PRT", years], color=ORANGE, lw=2.6, zorder=4)
    end_label(ax, years[-1], sch.loc["PRT", years[-1]], "Portugal", ORANGE, bold=True)
    for y in (1925, 1975):
        v = sch.loc["PRT", y]
        ax.scatter([y], [v], s=30, color=ORANGE, zorder=5, edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(f"{v:.1f} years", (y, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlim(1900, 2010)
    ax.set_ylim(0, 14)
    ax.set_ylabel("Average years of schooling, ages 25–64")
    heading(
        fig,
        "Portugal's schooling gap with seven European peers doubled under the regime",
        "Average years of schooling of adults aged 25–64. The seven peers are the European countries with 2.0 to 3.5 years in 1925:\n"
        "their lead over Portugal grew from 1.8 years then to 3.5 years in 1975. Grey: Italy, Finland, Greece, Bulgaria, Ireland.",
    )
    footer(fig, "Barro and Lee (2015), Education Matters, long-run attainment 1870-2010 (barrolee/BarroLeeDataSet).")
    save(fig, "fig2_schooling")


def fig3_catch_up():
    cu = pd.read_csv(RESULTS / "schooling_catch_up.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 6.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.83, bottom=0.11)
    xs = np.linspace(0, 7.5, 50)
    slope = np.polyfit(cu["start"], cu["predicted_gain"], 1)
    ax.plot(xs, np.polyval(slope, xs), color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.text(7.4, np.polyval(slope, 7.4) - 0.25, "gain predicted by the 1925 level", ha="right", fontsize=8.5, color=INK2)
    com = ["BGR", "HUN", "POL", "RUS", "ALB"]
    dot = {"s": 62, "edgecolor": SURFACE, "linewidth": 1.5, "zorder": 3}
    rest = cu.drop(index=com + ["PRT", "ESP"])
    ax.scatter(rest["start"], rest["gain"], color=GREY_LINE, label="Other European countries", **dot)
    ax.scatter(cu.loc[com, "start"], cu.loc[com, "gain"], color=VIOLET, label="Ruled by communist parties after 1945", **dot)
    ax.scatter(cu.loc[["ESP"], "start"], cu.loc[["ESP"], "gain"], color=BLUE, label="Spain (dictatorship 1939–1975)", **dot)
    big = {**dot, "s": 90, "zorder": 4}
    ax.scatter(cu.loc[["PRT"], "start"], cu.loc[["PRT"], "gain"], color=ORANGE, label="Portugal (dictatorship 1926–1974)", **big)
    offsets = {
        "PRT": (8, -4), "ESP": (-36, -4), "TUR": (8, -4), "NOR": (-10, 8), "FRA": (8, -4), "RUS": (8, -2), "IRL": (8, 2),
        "POL": (8, 2), "BGR": (-52, 2), "GRC": (8, -4), "ALB": (8, -2), "HUN": (8, -10), "FIN": (8, -4), "ITA": (8, -6),
        "NLD": (8, -2), "SWE": (8, -2), "BEL": (8, -2), "CHE": (-20, 8), "GBR": (-40, 6), "DEU": (8, -6), "AUT": (-38, -4), "DNK": (8, -4),
    }  # fmt: skip
    for c, off in offsets.items():
        ax.annotate(lc.NAMES[c], (cu.loc[c, "start"], cu.loc[c, "gain"]), xytext=off, textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(1.2, 5.8)
    ax.set_xlabel("Average years of schooling, ages 25–64, in 1925")
    ax.set_ylabel("Gain in years of schooling, 1925–1975")
    ax.legend(loc="upper right", fontsize=9)
    heading(
        fig,
        "Under the regime, Portugal's adult schooling gained 1.6 years less than predicted",
        "Each dot is a European country. Gain in average schooling of adults aged 25–64 between 1925 and 1975, against the 1925\n"
        "level. The dashed line is the least-squares fit; Luxembourg is left out (its series starts implausibly low).",
    )
    footer(fig, "Barro and Lee (2015), Education Matters, long-run attainment 1870-2010 (barrolee/BarroLeeDataSet).")
    save(fig, "fig3_catch_up")


def fig4_gdp_since_1995():
    g = pd.read_csv(RESULTS / "gdp_per_head_pps.csv", index_col=0)
    g.columns = g.columns.astype(int)
    g = g.loc[:, 1995:]
    years = list(g.columns)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.07, right=0.87, top=0.83, bottom=0.11)
    ax.axhline(100, color=MUTED, lw=1)
    ax.text(1995.3, 101.5, "EU27 = 100", fontsize=8.5, color=INK2)
    over = {"CZ": ("Czechia", VIOLET, 1.0), "SI": ("Slovenia", AQUA, -1.5), "LT": ("Lithuania", BLUE, -0.6)}
    for c, (name, col, dy) in over.items():
        ax.plot(years, g.loc[c, years], color=col, lw=1.6, zorder=2)
        end_label(ax, years[-1], g.loc[c, years[-1]], name, col, dy=dy)
    for c, name, dy in (("PL", "Poland", -1.6), ("EE", "Estonia", -3.6), ("ES", "Spain", 4.6), ("EL", "Greece", 0.0)):
        ax.plot(years, g.loc[c, years], color=GREY_LINE, lw=1.3, zorder=1)
        end_label(ax, years[-1], g.loc[c, years[-1]], name, INK2, dy=dy)
    ax.plot(years, g.loc["PT", years], color=ORANGE, lw=2.6, zorder=3)
    end_label(ax, years[-1], g.loc["PT", years[-1]], "Portugal", ORANGE, bold=True, dy=1.0)
    ax.set_xlim(1995, years[-1])
    ax.set_ylim(25, 115)
    ax.set_ylabel("GDP per head in PPS, EU27 = 100")
    heading(
        fig,
        "Since 1995 Portugal has stayed at about 81% of the EU average while the east caught up",
        "GDP per head in purchasing power standards. Czechia, Slovenia and Lithuania were below Portugal in 1995 and are above\n"
        "it in 2025; Poland has drawn level and Estonia is close behind. Malta, not shown, also passed Portugal.",
    )
    footer(fig, "Eurostat nama_10_pc (snapshot in DiogoRibeiro7/data, eurostat-eu-convergence-education).")
    save(fig, "fig4_gdp_since_1995")


def fig5_education_by_age():
    low = pd.read_csv(RESULTS / "low_education_by_age.csv", index_col=[0, 1])
    low.columns = low.columns.astype(int)
    last = int(low.columns.max())
    ages = ["Y25-34", "Y35-44", "Y45-54", "Y55-64"]
    labels = ["25–34", "35–44", "45–54", "55–64"]
    series = [("PT", "Portugal", ORANGE), ("ES", "Spain", BLUE), ("EU27_2020", "EU27", INK2), ("CZ", "Czechia", VIOLET), ("PL", "Poland", AQUA)]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.8, bottom=0.12)
    x = np.arange(len(ages))
    width = 0.15
    for i, (g, name, col) in enumerate(series):
        vals = [low.loc[(g, a), last] for a in ages]
        pos = x + (i - 2) * width
        ax.bar(pos, vals, width * 0.9, color=col, label=name, zorder=2)
        if g == "PT":
            for p, v in zip(pos, vals, strict=True):
                ax.text(p, v + 1.0, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Aged {a}" for a in labels])
    ax.set_ylim(0, 68)
    ax.set_ylabel("% with at most lower secondary education")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=5)
    heading(
        fig,
        f"Portugal's education gap is now concentrated in the older half of the workforce ({last})",
        "Share of each age group with at most lower secondary education (ISCED 0–2). Portuguese aged 55–64 have the highest share\n"
        "in the EU; those aged 25–34 are close to the EU average.",
    )
    footer(fig, "Eurostat edat_lfse_03 (snapshot in DiogoRibeiro7/data, eurostat-eu-convergence-education).")
    save(fig, "fig5_education_by_age")


def fig6_accounting():
    acc = pd.read_csv(RESULTS / "development_accounting.csv")
    a = acc[acc["year"] == 2019].set_index("country").sort_values("hc")
    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.84, bottom=0.1)
    y = np.arange(len(a))
    ax.axvline(100, color=MUTED, lw=1)
    for i, (_c, r) in enumerate(a.iterrows()):
        ax.plot([100 * r.relative_output, 100 * r.relative_output_with_ref_h], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    dot = {"s": 58, "edgecolor": SURFACE, "linewidth": 1.5, "zorder": 3}
    ax.scatter(100 * a["relative_output"], y, color=BLUE, label="Actual", **dot)
    ax.scatter(100 * a["relative_output_with_ref_h"], y, color=ORANGE, label="With the reference countries' human capital", **dot)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lc.NAMES[c]} ({a.loc[c, 'hc']:.2f})" for c in a.index])
    for lab, c in zip(ax.get_yticklabels(), a.index, strict=True):
        if c == "PRT":
            lab.set_fontweight("bold")
            lab.set_color(INK)
    ax.set_xlim(30, 240)
    ax.set_xscale("log")
    ticks = [40, 60, 80, 100, 150, 200]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Output per worker, % of twelve Western European countries (log scale)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    heading(
        fig,
        "Human capital accounts for more of Portugal's gap than of any other country's here",
        "Development accounting for 2019: output per worker, actual and with the reference countries' human capital per worker,\n"
        "holding capital intensity and productivity fixed. Countries ordered by PWT human capital index (in brackets).",
    )
    footer(fig, "Feenstra, Inklaar and Timmer, Penn World Table 10.01, via the pwt10 package; capital share 1/3.")
    save(fig, "fig6_accounting")


def main():
    fig1_relative_income()
    fig2_schooling()
    fig3_catch_up()
    fig4_gdp_since_1995()
    fig5_education_by_age()
    fig6_accounting()


if __name__ == "__main__":
    main()
