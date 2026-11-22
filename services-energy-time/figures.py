"""The four figures, drawn from results/ (run services_numbers.py first).

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SOURCE = "Eurostat (snapshot in DiogoRibeiro7/data, eurostat-services-energy-time). Figure: Diogo Ribeiro."

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
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
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
    fig.text(0.01, 0.965, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.905, subtitle, fontsize=9.5, color=INK2, va="top")


def footer(fig):
    fig.text(0.01, 0.01, f"Source: {SOURCE}", fontsize=8, color=MUTED, ha="left", va="bottom")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1_shares():
    s = pd.read_csv(RESULTS / "eu_shares.csv")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for col, color, label in (
        ("services_gva_share", BLUE, "Value added (current prices)"),
        ("services_hours_share", ORANGE, "Hours worked"),
        ("services_energy_share", AQUA, "Final energy (commercial and public services)"),
    ):
        ax.plot(s.year, 100 * s[col], color=color, label=label)
        dy = {"services_gva_share": 6, "services_hours_share": -6}.get(col, 0)
        ax.annotate(f"{100 * s[col].iloc[-1]:.0f}%", (s.year.iloc[-1], 100 * s[col].iloc[-1]), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK2)  # fmt: skip
    ax.set_xlim(1995, 2027)
    ax.set_ylim(0, 80)
    ax.set_ylabel("Share of the EU total (%)")
    ax.legend(loc="center right", fontsize=9.5, bbox_to_anchor=(1.0, 0.42))
    heading(fig, "Services' share of EU output, hours and energy", "EU27 (2020 composition), 1995–2024. Services are NACE sections G to U.")
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.08, right=0.97)
    footer(fig)
    save(fig, "fig1_services_shares")


def fig2_indices():
    d = pd.read_csv(RESULTS / "eu_sector_index.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, prefix, title in ((axes[0], "services", "Services"), (axes[1], "industry", "Industry except construction")):
        for kind, color, label in (("output", BLUE, "Real value added"), ("hours", ORANGE, "Hours worked"), ("energy", AQUA, "Final energy")):
            col = f"{prefix}_{kind}"
            ax.plot(d.year, d[col], color=color, label=label)
            ax.annotate(f"×{d[col].iloc[-1]:.2f}", (d.year.iloc[-1], d[col].iloc[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK2)  # fmt: skip
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_xlim(1995, 2029)
        ax.set_title(title)
    axes[0].set_ylabel("Index, 1995 = 1")
    axes[0].legend(loc="upper left", fontsize=9.5)
    heading(
        fig,
        "Services grew by adding hours and energy; industry grew while shedding both",
        "EU27, 1995–2024. Services value added is a Törnqvist aggregate of sections G to U; energy for services is commercial and public services.",
    )
    fig.subplots_adjust(top=0.78, bottom=0.1, left=0.07, right=0.98, wspace=0.08)
    footer(fig)
    save(fig, "fig2_output_hours_energy")


def fig3_productivity():
    d = pd.read_csv(RESULTS / "productivity_by_industry.csv")
    d = d[d.code != "TOTAL"].sort_values("output_per_hour_growth")
    services = {"G-I", "J", "K", "L", "M_N", "O-Q", "R-U"}
    total = pd.read_csv(RESULTS / "productivity_by_industry.csv").set_index("code").loc["TOTAL", "output_per_hour_growth"]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = [BLUE if c in services else MUTED for c in d.code]
    ax.barh(d.label, 100 * d.output_per_hour_growth, color=colors, height=0.62)
    for y, v in enumerate(100 * d.output_per_hour_growth):
        ax.text(v + (0.05 if v >= 0 else -0.05), y, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
    ax.axvline(100 * total, color=INK2, lw=1, ls="--")
    ax.text(100 * total + 0.05, len(d) - 0.4, f"all industries {100 * total:.2f}", fontsize=9, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-1.0, 3.6)
    ax.set_xlabel("Real value added per hour worked, % per year")
    ax.grid(axis="y", visible=False)
    heading(
        fig,
        "Output per hour grew slowly in most services",
        "EU27, 1995–2024. Blue bars are services (sections G to U); grey bars are the rest of the economy.",
    )
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.39, right=0.97)
    footer(fig)
    save(fig, "fig3_output_per_hour")


def fig4_time_use():
    d = pd.read_csv(RESULTS / "time_use.csv").dropna(subset=["leisure_min", "consumption_pps"])
    per_hour = d.consumption_pps / (d.leisure_min * 365.25 / 60)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for wave, color in ((2000, ORANGE), (2010, BLUE)):
        w = d.time == wave
        a1.scatter(d.consumption_pps[w], d.leisure_min[w] / 60, s=40, color=color, edgecolor=SURFACE, linewidth=1.5, label=f"{wave} wave", zorder=3)
        a2.scatter(d.consumption_pps[w], per_hour[w], s=40, color=color, edgecolor=SURFACE, linewidth=1.5, label=f"{wave} wave", zorder=3)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks([4000, 8000, 16000, 32000])
        ax.set_xticklabels(["4,000", "8,000", "16,000", "32,000"])
        ax.minorticks_off()
        ax.set_xlabel("Actual individual consumption per person (PPS)")
    a1.set_ylim(0, 7)
    a1.set_ylabel("Hours per day")
    a1.set_title("Free time per person")
    a1.legend(loc="lower right", fontsize=9.5)
    a2.set_yscale("log")
    a2.set_yticks([2, 4, 8, 16])
    a2.set_yticklabels(["2", "4", "8", "16"])
    a2.minorticks_off()
    a2.set_ylabel("PPS per hour of free time")
    a2.set_title("Consumption per hour of free time")
    for _, r in d[d.geo.isin(["BG", "LU", "FI", "FR", "LT"])].iterrows():
        a1.annotate(f"{r.geo} {r.time}", (r.consumption_pps, r.leisure_min / 60), xytext=(4, 5), textcoords="offset points", fontsize=8, color=INK2)
    heading(
        fig,
        "Consumption varies sevenfold across countries; free time barely moves",
        "Harmonised European Time Use Surveys, 2000 and 2010 waves (leisure, social and associative life), against consumption in the survey year.",
    )
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.07, right=0.98, wspace=0.25)
    footer(fig)
    save(fig, "fig4_time_use")


def main():
    fig1_shares()
    fig2_indices()
    fig3_productivity()
    fig4_time_use()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
