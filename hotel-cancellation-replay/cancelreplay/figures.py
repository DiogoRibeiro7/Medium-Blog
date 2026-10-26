"""The article's three figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .paths import FIGURES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BAR = "#c9c8c2"
SOURCE = "Data: Antonio, de Almeida & Nunes (2019), two Portuguese hotels. Analysis: Diogo Ribeiro."


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width, height):
    fig = plt.figure(figsize=(width, height), dpi=200, facecolor=SURFACE)
    plt.rcParams["font.family"] = "DejaVu Sans"
    return fig


def _titles(fig, title, subtitle, top=0.95):
    fig.text(0.02, top, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, top - 0.055, subtitle, fontsize=9.5, color=INK_2, ha="left", va="top")
    fig.text(0.02, 0.015, SOURCE, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def auc_by_question(results: dict, path):
    q = results["auc_by_question"]
    rows = [
        ("ever_cancel_all_columns", "Will it ever cancel?\nall columns, random split"),
        ("ever_cancel_booking_columns", "Will it ever cancel?\nbooking-time columns, random split"),
        ("after_30_day_batch_booking_columns", "Will it cancel after the 30-day batch?\nbooking-time columns, random split"),
        ("after_30_day_batch_grouped_by_block", "Will it cancel after the 30-day batch?\nsplit keeping each block together"),
        ("reference_out_of_time", "Will it cancel after the 30-day batch?\nlive replay, refit monthly"),
    ]
    fig = _figure(8.4, 5.2)
    ax = fig.add_axes([0.41, 0.13, 0.55, 0.66])
    _style(ax)
    y = np.arange(len(rows))[::-1]
    for (key, _), yy in zip(rows, y):
        r, c = q[key]["Resort Hotel"], q[key]["City Hotel"]
        ax.plot([min(r, c), max(r, c)], [yy, yy], color=GRID, linewidth=2, zorder=1)
        ax.scatter([r], [yy], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([c], [yy], s=70, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(max(r, c) + 0.012, yy, f"{min(r, c):.2f}–{max(r, c):.2f}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [label for _, label in rows], fontsize=8.8, color=INK)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("AUC (0.5 = no better than chance)", fontsize=9, color=INK_2)
    ax.scatter([], [], s=50, color=BLUE, label="Resort hotel (Algarve)")
    ax.scatter([], [], s=50, color=ORANGE, label="City hotel (Lisbon)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=8.5,
              labelcolor=INK_2, handletextpad=0.3)
    _titles(fig, "The same gradient-boosting model, four questions",
            "AUC depends far more on the question asked than on the algorithm")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


GROUP_TITLES = {
    "model": "The model and its inputs",
    "engineering": "How the data reach the model",
    "architecture": "Where the parts meet",
}


def value_by_variant(results: dict, path, clip=(-0.6, 0.45)):
    table = pd.DataFrame(results["variants"]).set_index("key")
    ref_share = table.loc["reference", "share_of_oracle"]
    order = []
    for group in ("model", "engineering", "architecture"):
        order.append(("header", group))
        order += [("row", k) for k in table.index if table.loc[k, "group"] == group]
    n = len(order)
    fig = _figure(9.0, 0.36 * n + 1.6)
    ax = fig.add_axes([0.43, 0.08, 0.54, 0.80])
    _style(ax)
    lo, hi = clip
    ypos = np.arange(n)[::-1]
    labels = []
    for (kind, key), y in zip(order, ypos):
        if kind == "header":
            labels.append("")
            ax.text(lo - 0.02, y, GROUP_TITLES[key].upper(), transform=ax.transData, ha="right",
                    va="center", fontsize=8.2, fontweight="bold", color=INK, clip_on=False)
            continue
        row = table.loc[key]
        labels.append(row["label"])
        share = row["share_of_oracle"]
        a = ref_share + row["diff_lo"]
        b = ref_share + row["diff_hi"]
        colour = BLUE
        if share < lo:
            ax.annotate("", xy=(lo, y), xytext=(lo + 0.07, y),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
            ax.text(lo + 0.08, y, f"{share:.2f}", va="center", fontsize=8.5, color=INK_2)
            continue
        if b < 0 or share < 0:
            colour = ORANGE
        ax.plot([max(a, lo), min(b, hi)], [y, y], color=colour, alpha=0.35, linewidth=2.5, solid_capstyle="round")
        ax.scatter([share], [y], s=55, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.axvline(ref_share, color=INK_2, linewidth=1, zorder=0)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), zorder=0)
    ax.text(ref_share, n - 0.1, f"reference {ref_share:.2f}", ha="center", va="bottom", fontsize=8.2, color=INK_2)
    ax.text(0, n - 0.1, "resell nothing", ha="center", va="bottom", fontsize=8.2, color=MUTED)
    ax.set_yticks(ypos, labels, fontsize=8.6, color=INK)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, n + 0.6)
    ax.set_xlabel("Share of the perfect-foresight profit captured, 854 arrival nights (95% interval for the difference from the reference)",
                  fontsize=8.2, color=INK_2)
    fig.text(0.02, 0.985, "Where the value of a cancellation model is decided", fontsize=13,
             fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.985 - 0.045 * 8 / (0.36 * n + 1.6) * 1.6,
             "Each line changes one decision in the reference system and replays it on the real outcomes",
             fontsize=9.5, color=INK_2, ha="left", va="top")
    fig.text(0.02, 0.008, SOURCE, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def nights_window(logs: dict, path, hotel="City Hotel", start="2016-10-01", end="2016-12-31"):
    ref = logs["reference"]
    bad = logs["all_bookings"]
    sel = lambda d: d.loc[(d["hotel"] == hotel) & (d["arrival"] >= start) & (d["arrival"] <= end)].set_index("arrival")  # noqa: E731
    r, b = sel(ref), sel(bad)
    fig = _figure(9.0, 4.4)
    ax = fig.add_axes([0.07, 0.15, 0.90, 0.62])
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.bar(r.index, r["released"], width=0.8, color=BAR, label="Rooms actually released after the batch")
    ax.step(r.index, r["q_distribution_blocks"], where="mid", color=BLUE, linewidth=2,
            label="Resold by the reference system")
    ax.step(b.index, b["q_distribution_blocks"], where="mid", color=ORANGE, linewidth=2,
            label="Resold by the same model trained on all bookings")
    ax.set_ylabel("Rooms per arrival night", fontsize=9, color=INK_2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=8.3, labelcolor=INK_2)
    ax.margins(x=0.01)
    _titles(fig, f"{hotel.replace(' Hotel', ' hotel')}, arrivals October to December 2016",
            "Same algorithm, same columns; only the rows it learned from differ", top=0.97)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def draw_all(results: dict, logs: dict, rp) -> None:
    FIGURES.mkdir(exist_ok=True)
    auc_by_question(results, FIGURES / "fig1_auc_by_question.png")
    nights_window(logs, FIGURES / "fig2_nights_city_hotel.png")
    value_by_variant(results, FIGURES / "fig3_value_by_variant.png")
