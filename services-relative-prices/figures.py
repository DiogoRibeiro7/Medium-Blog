"""The six figures, drawn from results/ (run prices_numbers.py first).

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SOURCE = "Eurostat (snapshot in DiogoRibeiro7/data, eurostat-services-relative-prices). Figure: Diogo Ribeiro."

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

SHORT = {
    "CP0441": "Water supply", "CP0432": "Home repair services", "CP0723": "Car repair and servicing",
    "CP0733": "Air fares", "CP081": "Postal services", "CP0442": "Refuse collection", "CP1111": "Restaurants and cafés",
    "CP0941": "Sport and recreation services", "CP01": "Food", "CP063": "Hospital services", "CP1211": "Hairdressing",
    "CP0731": "Rail fares", "CP125": "Insurance", "CP10": "Education", "CP0314": "Clothing repair and cleaning",
    "CP0622": "Dental services", "CP041": "Rents", "CP0611": "Medicines", "CP0721": "Car spare parts",
    "CP0511": "Furniture", "CP03": "Clothing and footwear", "CP0931": "Games and toys",
    "CP0531_0532": "Household appliances", "CP0911": "Television and audio equipment", "CP0913": "Computers",
}  # fmt: skip
NAMES = {
    "A": "Agriculture", "B": "Mining", "C": "Manufacturing", "D": "Electricity and gas", "E": "Water and waste",
    "F": "Construction", "G": "Trade", "H": "Transport", "I": "Hotels and restaurants", "J": "Information and communication",
    "K": "Finance and insurance", "M": "Professional services", "N": "Administrative services", "O": "Public administration",
    "P": "Education", "Q": "Health and social work", "R": "Arts and recreation", "S": "Other services",
    "T": "Households as employers",
}  # fmt: skip


def heading(fig, title, subtitle):
    fig.text(0.01, 0.975, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.93, subtitle, fontsize=9.5, color=INK2, va="top")


def footer(fig):
    fig.text(0.01, 0.008, f"Source: {SOURCE}", fontsize=8, color=MUTED, ha="left", va="bottom")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1_consumer_items():
    d = pd.read_csv(RESULTS / "hicp_items.csv").sort_values("change_pct")
    fig, ax = plt.subplots(figsize=(9, 8.2))
    fig.subplots_adjust(left=0.30, right=0.93, top=0.86, bottom=0.08)
    y = np.arange(len(d))
    colors = [BLUE if k == "services" else ORANGE for k in d.kind]
    ax.barh(y, d.change_pct, color=colors, height=0.72, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, [SHORT[c] for c in d.code], fontsize=9.5)
    for yi, v in zip(y, d.change_pct, strict=True):
        ax.text(v + (3 if v >= 0 else -3), yi, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK2)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.axvline(75.4, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_ylim(-0.7, len(d) + 0.5)
    ax.text(77, len(d) - 0.05, "All items +75%", fontsize=8.5, color=INK2, va="center")
    ax.set_xlim(-110, 165)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Price change, 2001 to 2025 (%)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)], labels=["Services", "Goods"],
              loc="lower right", fontsize=9.5)  # fmt: skip
    heading(fig, "What a household pays: services against goods, EU27", "HICP annual average indices, items with an EU27 series from 2001")
    footer(fig)
    save(fig, "fig1_consumer_items")


def fig2_services_vs_manufacturing():
    g = pd.read_csv(RESULTS / "eu_services_vs_manufacturing.csv")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.subplots_adjust(left=0.08, right=0.79, top=0.84, bottom=0.11)
    series = (
        ("productivity", BLUE, "Productivity gap"),
        ("price", INK, "Relative price"),
        ("wage", ORANGE, "Wage gap"),
        ("margins", AQUA, "Margins"),
    )
    for col, color, label in series:
        lw = 2.6 if col == "price" else 2.0
        ax.plot(g.year, 100 * g[col], color=color, linewidth=lw)
        ax.annotate(f"{label} {100 * g[col].iloc[-1]:+.0f}", (g.year.iloc[-1], 100 * g[col].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK2, annotation_clip=False)  # fmt: skip
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlim(1995, 2024)
    ax.set_ylabel("Log points since 1995")
    heading(fig, "Services against manufacturing: productivity pushed, wages and margins held back",
            "EU27, price of value added of services (G to T without real estate) relative to manufacturing.\n"
            "Relative price = productivity gap + wage gap + margins")  # fmt: skip
    footer(fig)
    save(fig, "fig2_services_vs_manufacturing")


def fig3_industries():
    d = pd.read_csv(RESULTS / "eu_industries.csv")
    d = d[d.code.isin(list(NAMES))].sort_values("productivity")
    fig, ax = plt.subplots(figsize=(9, 7.4))
    fig.subplots_adjust(left=0.27, right=0.96, top=0.86, bottom=0.09)
    y = np.arange(len(d))
    for yi, p, q in zip(y, 100 * d.productivity, 100 * d.price, strict=True):
        ax.plot([p, q], [yi, yi], color=GRID, linewidth=2.5, zorder=1)
    ax.scatter(100 * d.productivity, y, s=60, color=BLUE, zorder=3, label="What the productivity gap alone implies", edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(100 * d.price, y, s=60, color=ORANGE, zorder=3, label="What the price did", edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y, [NAMES[c] for c in d.code], fontsize=9.5)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Log points, 1995 to 2024, against the whole economy (without real estate)")
    ax.legend(loc="lower right", fontsize=9.5)
    heading(fig, "Productivity sets the order of industry prices",
            "EU27, 19 NACE sections. The grey line is what wages and margins added or took away")  # fmt: skip
    footer(fig)
    save(fig, "fig3_industries")


def fig4_panel():
    p = pd.read_csv(RESULTS / "panel.csv")
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.86, bottom=0.11)
    other = p[p.geo != "PT"]
    pt = p[p.geo == "PT"]
    ax.scatter(100 * other.productivity, 100 * other.price, s=16, color=BLUE, alpha=0.45, edgecolor="none", label="Section in a member state")
    ax.scatter(100 * pt.productivity, 100 * pt.price, s=34, color=ORANGE, edgecolor=SURFACE, linewidth=1, label="Portugal", zorder=3)
    x = np.linspace(-200, 200, 2)
    b, a = np.polyfit(p.productivity, p.price, 1)
    ax.plot(x, a * 100 + b * x, color=INK, linewidth=1.6, label=f"Fitted slope {b:.2f}")
    ax.plot(x, x, color=INK2, linewidth=1, linestyle=(0, (3, 3)), label="Full pass-through (slope 1)")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    lim = 100 * max(p.productivity.abs().quantile(0.995), p.price.abs().quantile(0.995))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Productivity contribution, log points")
    ax.set_ylabel("Relative price change, log points")
    ax.legend(loc="upper left", fontsize=9)
    heading(fig, "The same pattern in 450 country-industry cells",
            "25 member states, 18 sections, 1995 to 2024, each against its own country's economy")  # fmt: skip
    footer(fig)
    save(fig, "fig4_panel")


def fig5_price_levels():
    d = pd.read_csv(RESULTS / "price_levels_2024.csv")
    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.86, bottom=0.11)
    ax.scatter(d.gdp_pc, d.services, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1, label="Services", zorder=3)
    ax.scatter(d.gdp_pc, d.goods, s=46, color=ORANGE, edgecolor=SURFACE, linewidth=1, label="Goods", zorder=3)
    for _, r in d.iterrows():
        if r.geo in ("PT", "BG", "RO", "DK", "LU", "IE", "DE", "PL", "EL"):
            ax.annotate(r.geo, (r.gdp_pc, r.services), xytext=(5, 3), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([50, 75, 100, 150, 200, 250], ["50", "75", "100", "150", "200", "250"])
    ax.set_yticks([40, 60, 80, 100, 140, 180], ["40", "60", "80", "100", "140", "180"])
    ax.minorticks_off()
    ax.axhline(100, color=MUTED, linewidth=1)
    ax.set_xlabel("GDP per head in purchasing power standards, EU27 = 100 (log scale)")
    ax.set_ylabel("Price level, EU27 = 100 (log scale)")
    ax.legend(loc="upper left", fontsize=9.5)
    heading(fig, "Across member states, services are priced like the local wage and goods like the world",
            "Price level indices for goods and for services against GDP per head, 27 member states, 2024")  # fmt: skip
    footer(fig)
    save(fig, "fig5_price_levels")


def fig6_persistence():
    d = pd.read_csv(RESULTS / "persistence.csv")
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.86, bottom=0.11)
    ax.scatter(d.drift_1995_2010, d.drift_2010_2024, s=16, color=BLUE, alpha=0.5, edgecolor="none")
    lim = max(d.drift_1995_2010.abs().quantile(0.995), d.drift_2010_2024.abs().quantile(0.995))
    x = np.linspace(-lim, lim, 2)
    ax.plot(x, x, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(lim * 0.30, lim * 0.78, "Drift continues (slope 1)", fontsize=9, color=INK2)
    r = np.corrcoef(d.drift_1995_2010, d.drift_2010_2024)[0, 1]
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Relative price drift, 1995 to 2010 (log points a year)")
    ax.set_ylabel("Relative price drift, 2010 to 2024 (log points a year)")
    heading(fig, "Past drift is a poor forecast of future drift",
            f"450 country-industry cells, each against its country's economy; correlation {r:.2f}")  # fmt: skip
    footer(fig)
    save(fig, "fig6_persistence")


if __name__ == "__main__":
    fig1_consumer_items()
    fig2_services_vs_manufacturing()
    fig3_industries()
    fig4_panel()
    fig5_price_levels()
    fig6_persistence()
