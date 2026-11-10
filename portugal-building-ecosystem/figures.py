"""Draw the figures from the CSVs written by ``analysis.py``.

    python figures.py

Writes ``figures/fig1_*.png`` ... ``figures/fig6_*.png``. Colours are the first five slots of a
categorical palette checked for colour-vision deficiency; every line is labelled at its end, so
no figure depends on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2.0, "legend.frameon": False,
    "text.color": INK,
})


def end_label(ax, x, y, text, color, dy=0.0, ha="left"):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", ha=ha,
                fontsize=10, color=INK, fontweight="bold")
    ax.plot([x], [y], marker="o", ms=4.5, color=color, mec="white", mew=1.2, zorder=5)


def source(fig, text):
    fig.text(0.01, 0.005, text, fontsize=8.5, color=MUTED, ha="left", va="bottom")


def fig1_age() -> None:
    age = pd.read_csv(RES / "workforce_age.csv", index_col=0).loc[1998:]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.32})
    under50 = age["construction_total"] - age["construction_50plus"]
    series = [(under50, "aged under 50", BLUE, "-"), (age["construction_50plus"], "aged 50 and over", ORANGE, "--"),
              (age["construction_under25"], "aged under 25", AQUA, ":")]
    offsets = {"aged under 50": 12, "aged 50 and over": -10, "aged under 25": 0}
    for s, lab, c, ls in series:
        s = s.dropna()
        a.plot(s.index, s.values, color=c, ls=ls, label=lab)
        end_label(a, s.index[-1], s.values[-1], f"{lab}\n{s.values[-1]:.0f}k", c, dy=offsets[lab])
    a.set_title("People working in construction, by age (thousands)")
    a.set_xlim(1997, 2031)
    a.set_ylim(0, None)
    a.axvline(2008, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    a.text(2008.4, 535, "classification\nchange", fontsize=8.5, color=MUTED, va="top")
    a.legend(loc="upper right", fontsize=9)

    for col, lab, c, ls in (("construction_share_50plus", "construction", ORANGE, "-"),
                            ("all_share_50plus", "all industries", BLUE, "--")):
        s = age[col] * 100
        b.plot(s.index, s.values, color=c, ls=ls, label=lab)
        end_label(b, s.index[-1], s.values[-1], f"{lab}\n{s.values[-1]:.0f}%", c)
    b.set_title("Share of workers aged 50 and over (%)")
    b.set_xlim(1997, 2031)
    b.set_ylim(0, 50)
    b.axvline(2008, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    cross = age.index[age["construction_share_50plus"] > age["all_share_50plus"]].min()
    b.annotate(f"construction becomes the older\nworkforce in {cross}", (cross, age.loc[cross, "construction_share_50plus"] * 100),
               xytext=(1999, 42), fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    b.legend(loc="lower right", fontsize=9)
    source(fig, "Source: Eurostat, Labour Force Survey (lfsa_egana2d, lfsa_egana, lfsa_egan2). Under-25 values from 2012 are small samples.")
    fig.savefig(FIG / "fig1_workforce_age.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig2_labour_price_and_pipeline() -> None:
    m = pd.read_csv(RES / "cost_index_monthly.csv")
    m["t"] = m["year"] + (m["month"] - 0.5) / 12
    base = m[m.year == 2008][["labour", "materials"]].mean()
    g = pd.read_csv(RES / "graduates_construction.csv", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.3, "width_ratios": [1.25, 1]})
    for col, lab, c, ls in (("labour", "labour", ORANGE, "-"), ("materials", "materials", BLUE, "--")):
        s = m[col] / base[col] * 100
        a.plot(m["t"], s, color=c, ls=ls, label=lab)
        end_label(a, m["t"].iloc[-1], s.iloc[-1], f"{lab}\n{s.iloc[-1]:.0f}", c)
    a.axhline(100, color=MUTED, lw=0.8)
    a.set_title("Cost of building a new dwelling, by input (2008 = 100)")
    a.set_xlim(2000, 2030.5)
    a.legend(loc="upper left", fontsize=9)
    years = [y for y in g.index if y >= 2013]
    x = np.arange(len(years))
    voc = g.loc[years, "ED35_45"].fillna(0).values
    ter = g.loc[years, "ED5-8"].fillna(0).values
    w = 0.38
    b.bar(x - w / 2 - 0.02, ter, width=w, color=BLUE, label="tertiary (bachelor, master, doctorate)")
    b.bar(x + w / 2 + 0.02, voc, width=w, color=ORANGE, label="vocational upper or post-secondary")
    for i in (0, len(years) - 1):
        b.text(x[i] - w / 2, ter[i] + 60, f"{ter[i]:,.0f}", ha="center", fontsize=8.5, color=INK)
        b.text(x[i] + w / 2, voc[i] + 60, f"{voc[i]:,.0f}", ha="center", fontsize=8.5, color=INK)
    b.set_xticks(x, [str(y) if y % 2 == 1 else "" for y in years])
    b.set_title("Graduates in architecture and construction (number)")
    b.legend(loc="upper right", fontsize=9)
    b.grid(axis="x", visible=False)
    source(fig, "Sources: INE, Índice de custos de construção de habitação nova (0011748); Eurostat, educ_uoe_grad02 (ISCED-F 073).")
    fig.savefig(FIG / "fig2_labour_price_and_pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3_three_counts() -> None:
    t = pd.read_csv(RES / "workforce_three_counts.csv", index_col=0)
    occ = pd.read_csv(RES / "occupations.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    series = [("business_statistics", "business statistics (firms' accounts)", BLUE, "-"),
              ("national_accounts", "national accounts", AQUA, "--"),
              ("labour_force_survey", "Labour Force Survey (households)", ORANGE, "-."),
              (None, "of whom craft and trades workers (LFS)", MAGENTA, ":")]
    for col, lab, c, ls in series:
        s = (t[col] if col else occ["OC7"]).dropna()
        ax.plot(s.index, s.values, color=c, ls=ls, label=lab)
        end_label(ax, s.index[-1], s.values[-1], f"{s.values[-1]:.0f}k", c)
    ax.set_title("People working in construction, three ways of counting them (thousands)")
    ax.set_ylim(0, 600)
    ax.set_xlim(2007.5, 2026.5)
    ax.set_xticks(range(2008, 2026, 2))
    ax.legend(loc="lower left", fontsize=9.5)
    source(fig, "Sources: INE, Sistema de contas integradas das empresas (0008467, 0013839); Eurostat, nama_10_a64_e, lfsa_egan2, lfsa_eisn2.")
    fig.savefig(FIG / "fig3_three_counts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig4_promoters() -> None:
    lic = pd.read_csv(RES / "licensed_by_promoter.csv", index_col=0)
    form = pd.read_csv(RES / "licensed_building_form.csv", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.3, "width_ratios": [1.35, 1]})
    pub = lic["public"] + lic["other"]
    for s, lab, c, ls in ((lic["companies"], "private companies", BLUE, "-"),
                          (lic["individuals"], "private individuals", ORANGE, "--"),
                          (pub, "public bodies, cooperatives, other", AQUA, ":")):
        a.plot(s.index, s.values / 1000, color=c, ls=ls, label=lab)
        end_label(a, s.index[-1], s.values[-1] / 1000, f"{s.values[-1] / 1000:.1f}k", c)
    a.set_title("Dwellings licensed in new buildings, by promoter (thousands)")
    a.set_xlim(2001.5, 2028)
    a.set_ylim(0, None)
    a.legend(loc="upper right", fontsize=9)
    s = form["dwellings_per_apartment_building"]
    b.plot(s.index, s.values, color=BLUE)
    end_label(b, s.index[-1], s.values[-1], f"{s.values[-1]:.1f}", BLUE)
    i0, imin = s.index[0], s.idxmin()
    for i in (i0, imin):
        b.annotate(f"{s.loc[i]:.1f}", (i, s.loc[i]), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=9, color=INK)
    b.set_title("Dwellings per licensed apartment building")
    b.set_xlim(2001.5, 2027.5)
    b.set_ylim(0, 13)
    source(fig, "Source: INE, Inquérito aos projetos de obras de edificação e de demolição de edifícios (0013236, 0013235, 0013244, 0013245).")
    fig.savefig(FIG / "fig4_who_commissions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig5_layers() -> None:
    lay = pd.read_csv(RES / "layers_persons_employed.csv", index_col=0)
    cols = [("agencies_finders_68311_68312", "agencies and finders", ORANGE, "-"),
            ("buying_selling_own_68100", "buying and selling own property", MAGENTA, "-."),
            ("specialised_trades_43", "specialised trades", AQUA, "--"),
            ("building_contractors_41200", "building contractors", BLUE, "-"),
            ("developers_41100", "developers", YELLOW, ":")]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ends = []
    for col, lab, c, ls in cols:
        s = lay[col] / lay.loc[2008, col] * 100
        ax.plot(s.index, s.values, color=c, ls=ls, label=lab)
        ends.append((s.values[-1], lab, c, s.index[-1]))
    ax.set_yscale("log")
    ax.set_yticks([50, 70, 100, 150, 200, 250], ["50", "70", "100", "150", "200", "250"])
    ax.minorticks_off()
    ax.axhline(100, color=MUTED, lw=0.8)
    # spread the end labels so they do not collide
    ends.sort()
    placed = []
    for v, lab, c, x in ends:
        y = v
        while placed and np.log(y) - np.log(placed[-1]) < 0.06:
            y = placed[-1] * np.exp(0.06)
        placed.append(y)
        ax.plot([x], [v], marker="o", ms=4.5, color=c, mec="white", mew=1.2, zorder=5)
        ax.annotate(f"{lab} {v:.0f}", (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5,
                    fontweight="bold", color=INK)
    ax.set_xlim(2007.5, 2030.5)
    ax.set_title("Persons employed in each layer of the housing market (2008 = 100, log scale)")
    ax.legend(loc="upper left", fontsize=9)
    source(fig, "Source: INE, Sistema de contas integradas das empresas (0008467, 0013839), CAE Rev. 3 41.10, 41.20, 43, 68.10, 68.31.1-2.")
    fig.savefig(FIG / "fig5_layers_persons_employed.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig6_levels() -> None:
    rt = pd.read_csv(RES / "recovered_share.csv")
    keep = [
        ("A skills", "construction workers under 25 (LFS)", "workers under 25"),
        ("A skills", "craft and trades workers (LFS)", "craft and trades workers"),
        ("A skills", "construction employment (national accounts)", "all construction employment"),
        ("A skills", "engineering firms, persons employed", "engineering firms (persons)"),
        ("B firms", "building contractors with 10-249 persons", "contractors with 10-249 persons"),
        ("B firms", "building contractors (41.20), persons employed", "building contractors (persons)"),
        ("B firms", "specialised trades (43), persons employed", "specialised trades (persons)"),
        ("C promoters", "dwellings licensed by private individuals", "dwellings licensed by individuals"),
        ("C promoters", "developers, production (real)", "developers' real production"),
        ("C promoters", "developers, persons employed", "developers (persons)"),
        ("C promoters", "dwellings licensed by private companies", "dwellings licensed by companies"),
        ("D sellers", "real estate agencies, persons employed", "agencies (persons)"),
        ("D sellers", "buying and selling own property, persons employed", "buying and selling own property (persons)"),
    ]
    layer_names = {"A skills": "skills", "B firms": "firms", "C promoters": "promoters", "D sellers": "sellers"}
    rows = []
    for layer, series, lab in keep:
        sub = rt[(rt.layer == layer) & (rt.series == series)]
        r08 = sub[sub.ref_year == 2008]
        r02 = sub[sub.ref_year == 2002]
        rows.append((layer_names[layer], lab, float(r08.last_over_ref.iloc[0]),
                     float(r02.last_over_ref.iloc[0]) if len(r02) else np.nan))
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    y = np.arange(len(rows))[::-1]
    for yi, (layer, lab, v08, v02) in zip(y, rows):
        ax.plot([v08], [yi], marker="o", ms=8, color=BLUE, mec="white", mew=1.2, zorder=4)
        ax.annotate(f"{v08 * 100:.0f}%", (v08, yi), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        if not np.isnan(v02):
            ax.plot([v02], [yi], marker="D", ms=7, color=ORANGE, mec="white", mew=1.2, zorder=4)
            ax.plot([v02, v08], [yi, yi], color=GRID, lw=2, zorder=2)
            ax.annotate(f"{v02 * 100:.0f}%", (v02, yi), xytext=(-8, 0), textcoords="offset points", va="center", ha="right", fontsize=9, color=INK)
    ax.set_yticks(y, [f"{lab}  [{layer}]" for layer, lab, _, _ in rows], fontsize=9.5)
    ax.axvline(1, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2], ["10%", "20%", "50%", "100%", "200%"])
    ax.minorticks_off()
    ax.set_xlim(0.1, 3.2)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], marker="o", ls="", color=BLUE, label="latest year as a share of 2008")
    ax.plot([], [], marker="D", ls="", color=ORANGE, label="latest year as a share of 2002")
    ax.legend(loc="center left", fontsize=9)
    ax.set_title("How much of each layer is there today, against 2008 and 2002 (log scale)")
    source(fig, "Sources: as in the other figures; latest year 2024 for firm statistics, 2025 for the rest. See results/recovered_share.csv.")
    fig.savefig(FIG / "fig6_how_much_is_left.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig1_age()
    fig2_labour_price_and_pipeline()
    fig3_three_counts()
    fig4_promoters()
    fig5_layers()
    fig6_levels()
    print("figures written:", ", ".join(sorted(p.name for p in FIG.glob("fig*.png"))))


if __name__ == "__main__":
    main()
