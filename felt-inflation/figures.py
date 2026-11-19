"""Draw the article's figures from the tables felt_numbers.py writes to results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT_ORANGE, LIGHT_BLUE = "#f8c9b2", "#c6dcf6"
KIND = {"frequent": ORANGE, "bill": AQUA, "rare": BLUE}
KIND_LABEL = {"frequent": "bought often", "bill": "paid as a bill", "rare": "bought rarely"}
NAMES = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "EL": "Greece",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
}
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
CREDIT = (
    "Data: Eurostat (HICP; European Commission consumer survey). Code: Medium-Blog/felt-inflation."
    " Figure: Diogo Ribeiro."
)


def frame(fig, title, subtitle, note=None, top_in=0.8, bottom_in=0.7):
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(0.01, 1 - 0.08 / h, title, fontsize=11, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 1 - 0.36 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top", linespacing=1.4)
    y = 0.06 / h
    fig.text(0.01, y, CREDIT, fontsize=7.2, color=MUTED, ha="left", va="bottom")
    if note:
        fig.text(0.01, y + 0.2 / h, note, fontsize=8, color=INK2, ha="left", va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def key():
    return pd.read_csv(RES / "key_numbers.csv").set_index("name")["value"]


def fig1_two_baskets():
    w = key()["weight_official_2026"]
    d = pd.read_csv(RES / "ea_two_baskets.csv").set_index("month").loc["2001-01":]
    x = pd.to_datetime(d.index)
    fig, ax = plt.subplots(figsize=(10, 4.9))
    ax.fill_between(x, d.everything_else, d.frequent, where=d.frequent >= d.everything_else, color=LIGHT_ORANGE, lw=0)
    ax.fill_between(x, d.everything_else, d.frequent, where=d.frequent < d.everything_else, color=LIGHT_BLUE, lw=0)
    ax.plot(x, d.frequent, color=ORANGE, lw=2, label="Bought at least once a month, paid at the till")
    ax.plot(x, d.everything_else, color=BLUE, lw=2, label="Everything else")
    ax.axhline(0, color=INK2, lw=0.8)
    grid(ax)
    ax.set_ylabel("Change on a year earlier (%)")
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=120))
    notes = [
        ("2002-01", "Euro notes\nand coins"),
        ("2008-07", "Food and oil\nspike"),
        ("2022-10", "Energy and\nfood shock"),
        ("2026-10", "Fuel after\nMarch 2026"),
    ]
    for mo, text in notes:
        t = pd.Timestamp(mo + "-01")
        ax.annotate(
            text,
            (t, d.loc[mo, "frequent"]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK2,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7},
        )
    ax.set_ylim(-1.5, 13.5)
    ax.legend(loc="upper left", fontsize=8.5)
    frame(
        fig,
        "Frequent purchases have risen faster than the rest of the basket in most months since 2002",
        "Euro area (20 countries), consumer prices, 12-month change. Orange: Eurostat's frequent"
        f" out-of-pocket purchases (FROOPP), {w * 100:.0f}% of spending in 2026.",
    )
    fig.savefig(FIG / "fig1_two_baskets.png")
    plt.close(fig)


def fig2_items():
    d = pd.read_csv(RES / "showcase.csv").sort_values("change")
    med = d.groupby("kind").change.median()
    fig, ax = plt.subplots(figsize=(10, 7.2))
    y = np.arange(len(d))
    for yi, r in zip(y, d.itertuples(), strict=True):
        ax.plot([0, r.change], [yi, yi], color=KIND[r.kind], lw=2.2, solid_capstyle="round")
        ax.plot(r.change, yi, "o", color=KIND[r.kind], ms=7, mec="white", mew=1.5)
        off = 6 if r.change >= 0 else -6
        ax.annotate(
            f"{r.change:+.0f}%",
            (r.change, yi),
            xytext=(off, 0),
            textcoords="offset points",
            ha="left" if r.change >= 0 else "right",
            va="center",
            fontsize=8,
            color=INK2,
        )
    ax.set_yticks(y, d.label)
    ax.axvline(0, color=INK2, lw=0.8)
    grid(ax, "x")
    ax.set_xlim(-110, 275)
    ax.set_xlabel("Change in price, December 2001 to December 2025 (%)")
    for k in ["frequent", "bill", "rare"]:
        ax.plot([], [], "o", color=KIND[k], label=KIND_LABEL[k])
    ax.legend(loc="lower right", fontsize=8.5)
    frame(
        fig,
        "The things people buy often rose more than the things they buy rarely",
        f"Euro area (20 countries). Median rise: bought often {med.frequent:+.0f}%, bills {med.bill:+.0f}%,"
        f" bought rarely {med.rare:+.0f}%."
        " Prices of rare purchases are adjusted for quality.",
    )
    fig.savefig(FIG / "fig2_items.png")
    plt.close(fig)


def fig3_countries():
    d = pd.read_csv(RES / "countries.csv").sort_values("gap")
    up = int((d.gap > 0).sum())
    fig, ax = plt.subplots(figsize=(10, 7.6))
    y = np.arange(len(d))
    for yi, r in zip(y, d.itertuples(), strict=True):
        ax.plot([r.all_items, r.frequent], [yi, yi], color=GRID, lw=2.5, zorder=1)
    ax.scatter(d.all_items, y, color=BLUE, s=40, zorder=2, label="All items (the official index)", ec="white", lw=1)
    ax.scatter(d.frequent, y, color=ORANGE, s=40, zorder=3, label="Frequent purchases (rebuilt)", ec="white", lw=1)
    for yi, r in zip(y, d.itertuples(), strict=True):
        ax.annotate(
            f"{r.gap:+.1f}",
            (max(r.frequent, r.all_items), yi),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK2,
        )
    ax.set_yticks(y, [NAMES[g] for g in d.geo])
    grid(ax, "x")
    ax.set_xlim(10, 72)
    ax.set_xlabel("Rise in prices, December 2020 to December 2025 (%)")
    ax.legend(loc="lower right", fontsize=8.5)
    frame(
        fig,
        f"In {up} of {len(d)} EU countries, frequent purchases rose more than the official index",
        "Numbers: frequent minus all items, percentage points. Frequent purchases rebuilt from Eurostat's class indices"
        " and weights.",
    )
    fig.savefig(FIG / "fig3_countries.png")
    plt.close(fig)


def fig4_correlations():
    d = pd.read_csv(RES / "correlations.csv").set_index("geo")
    d = d.sort_values("F12")
    more = int((d.F12 > d.X12).sum())
    fig, ax = plt.subplots(figsize=(10, 7.6))
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows(), strict=True):
        ax.plot([r.X12, r.F12], [yi, yi], color=GRID, lw=2.5, zorder=1)
    ax.scatter(d.X12, y, color=BLUE, s=40, zorder=2, ec="white", lw=1, label="Everything else")
    ax.scatter(d.F12, y, color=ORANGE, s=40, zorder=3, ec="white", lw=1, label="Frequent purchases")
    ax.set_yticks(y, [NAMES[g] for g in d.index])
    grid(ax, "x")
    ax.set_xlim(0.3, 0.92)
    ax.set_xlabel("Correlation between the survey answer and the 12-month price change, 2002–2025")
    ax.legend(loc="lower right", fontsize=8.5)
    frame(
        fig,
        f"Survey answers move with frequent purchases more than with the rest, in {more} of {len(d)} countries",
        "Monthly answers to 'How do you think consumer prices have developed over the last 12 months?'"
        " (balance of answers, seasonally adjusted).",
    )
    fig.savefig(FIG / "fig4_correlations.png")
    plt.close(fig)


def fig5_weights():
    d = pd.read_csv(RES / "weights.csv")
    d = d[d.model == "six"].reset_index(drop=True)
    labels = {
        "groceries": "Food, drinks, tobacco",
        "fuel": "Fuel for cars",
        "other_frequent": "Other frequent purchases",
        "energy": "Electricity, gas, heating",
        "bills": "Rent, water, phone, insurance",
        "rest": "Everything else",
    }
    colors = {
        "groceries": ORANGE,
        "fuel": ORANGE,
        "other_frequent": ORANGE,
        "energy": AQUA,
        "bills": AQUA,
        "rest": BLUE,
    }
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(d))[::-1]
    for yi, r in zip(y, d.itertuples(), strict=True):
        c = colors[r.basket]
        ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=2.2, alpha=0.45, solid_capstyle="round")
        ax.plot(r.weight, yi, "o", color=c, ms=8, mec="white", mew=1.5, zorder=3)
        ax.plot(r.spending, yi, "|", color=INK, ms=16, mew=2, zorder=4)
        ax.annotate(
            f"answers {r.weight:.2f}, spending {r.spending:.2f}",
            (max(r.hi, r.spending), yi),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK2,
        )
    ax.set_yticks(y, [labels[b] for b in d.basket])
    ax.axvline(0, color=INK2, lw=0.8)
    grid(ax, "x")
    ax.set_xlim(-0.2, 0.95)
    ax.set_ylim(-1.6, len(d) - 0.5)
    ax.set_xlabel("Share of the weight")
    ax.plot([], [], "o", color=MUTED, label="Weight in survey answers, with 95% interval (countries resampled)")
    ax.plot([], [], "|", color=INK, ms=12, mew=2, label="Weight in spending (the official index)")
    ax.legend(loc="lower right", fontsize=8.5)
    frame(
        fig,
        "Answers weigh food more than its share of spending, and bills hardly at all",
        "27 EU countries, 2002–2025. Slopes of the survey balance on six baskets' 12-month inflation,"
        " country fixed effects, scaled to sum to one.",
    )
    fig.savefig(FIG / "fig5_weights.png")
    plt.close(fig)


def fig6_memory():
    d = pd.read_csv(RES / "memory_residuals.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(d.year, d.res_short, color=BLUE, lw=2, marker="o", ms=5, label="Last 12 months only")
    ax.plot(d.year, d.res_long, color=ORANGE, lw=2, marker="o", ms=5, label="Last 12 months and last 3 years")
    grid(ax)
    ax.set_xticks(range(2002, 2026, 2))
    ax.set_ylabel("Answers minus model (balance points)")
    ax.set_ylim(-17, 27)
    ax.annotate(
        "After 2022, answers stay high\nas inflation falls back",
        (2024, d.set_index("year").at[2024, "res_short"]),
        xytext=(-130, 4),
        textcoords="offset points",
        fontsize=8,
        color=INK2,
        va="center",
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7},
    )
    ax.annotate(
        "Low inflation, answers\nlower than the model",
        (2016, d.set_index("year").at[2016, "res_short"]),
        xytext=(40, 22),
        textcoords="offset points",
        fontsize=8,
        color=INK2,
        va="center",
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7},
    )
    ax.legend(loc="upper left", fontsize=8.5)
    frame(
        fig,
        "Adding the last three years of frequent-purchase prices explains much of the post-2022 gap",
        "27 EU countries, average by year of the survey balance minus a model with country fixed effects."
        " Both models use frequent purchases and everything else.",
    )
    fig.savefig(FIG / "fig6_memory.png")
    plt.close(fig)


def fig7_felt():
    d = pd.read_csv(RES / "felt_curve.csv")
    k = key()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(d.w, d.since_2020, color=INK, lw=2)
    w0, w1 = k["weight_official_2026"], k["weight_answers"]
    marks = [(w0, f"Official weight, {w0:.2f}", BLUE), (w1, f"Weight in survey answers, {w1:.2f}", ORANGE)]
    for w, text, c in marks:
        v = float(np.interp(w, d.w, d.since_2020))
        ax.plot([w, w], [d.since_2020.min() - 1, v], color=c, lw=1.2, ls="--")
        ax.plot(w, v, "o", color=c, ms=8, mec="white", mew=1.5, zorder=3)
        ax.annotate(
            f"{text}: +{v:.1f}%",
            (w, v),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=INK2,
        )
    ax.axvspan(k["weight_answers_lo"], min(1.0, k["weight_answers_hi"]), color=LIGHT_ORANGE, alpha=0.5, lw=0)
    ax.text(
        k["weight_answers_lo"] + 0.01,
        d.since_2020.min() - 0.6,
        "95% interval of\nthe survey weight",
        ha="left",
        va="bottom",
        fontsize=8,
        color=INK2,
    )
    grid(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(d.since_2020.min() - 1, d.since_2020.max() + 1)
    ax.set_xlabel("Weight on frequent purchases (the rest goes to everything else)")
    ax.set_ylabel("Rise in prices, Dec 2020 to Oct 2026 (%)")
    frame(
        fig,
        f"Weighted the way survey answers weigh it, euro-area prices rose {np.interp(w1, d.w, d.since_2020):.1f}% since"
        f" 2020, not {np.interp(w0, d.w, d.since_2020):.1f}%",
        "Euro area (20 countries). Two-basket index chained each December, as the HICP is; with the official weights it"
        " reproduces the published index.",
    )
    fig.savefig(FIG / "fig7_felt.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig1_two_baskets()
    fig2_items()
    fig3_countries()
    fig4_correlations()
    fig5_weights()
    fig6_memory()
    fig7_felt()


if __name__ == "__main__":
    main()
