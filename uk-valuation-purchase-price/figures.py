"""Figures for the uk-valuation-purchase-price article.

    python figures.py     # reads results/*.csv; writes figures/fig*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"

# First three slots of a validated colour-blind-safe categorical palette (all pairs pass).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SOURCE = ("Contains HM Land Registry data © Crown copyright and database right 2026, licensed "
          "under the Open Government Licence v3.0. Chart: Diogo Ribeiro.")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False,
})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _titles(fig, title, subtitle=None):
    fig.text(0.01, 0.975, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.01, 0.925, subtitle, fontsize=9.5, color=INK2, ha="left", va="top")


def _finish(fig, name, top=0.88):
    fig.text(0.01, 0.01, SOURCE, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, top))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)


def _xpct(ax, label="Index-implied nominal gain at the sale date (m)"):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_xlabel(label)
    ax.axvline(0, color=AXIS, linewidth=1)


def _curve(split: str, lo: int = -15, hi: int = 14) -> pd.DataFrame:
    c = pd.read_csv(RES / "curve_predicted.csv")
    c = c[(c.split == split) & c.bin.between(lo, hi)].copy()
    c["x"] = c.bin + 0.5
    return c


# ----------------------------------------------------------------------------- fig 1
def fig1_near_purchase_price():
    """Observed shares just below, at and just above the old price, against an indexed model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, split in zip(axes, ["2008-2012", "2022-2025"]):
        c = _curve(split)
        ax.plot(c.x, 100 * c.obs_loss_within_2, color=ORANGE, label="Observed: within 2% below")
        ax.plot(c.x, 100 * c["M0-dist_loss_within_2"], color=ORANGE, linestyle=(0, (4, 3)),
                linewidth=1.6, label="Indexed valuation: within 2% below")
        ax.plot(c.x, 100 * (c.obs_gain_within_2 + c.obs_exact), color=BLUE,
                label="Observed: at or within 2% above")
        ax.plot(c.x, 100 * c["M0-dist_gain_within_2"], color=BLUE, linestyle=(0, (4, 3)),
                linewidth=1.6, label="Indexed valuation: within 2% above")
        ax.set_title(f"Resales {split}", loc="left")
        _xpct(ax)
        _grid(ax)
    axes[0].set_ylabel("Share of resales in each 1% band of m")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.9), ncol=2, fontsize=9)
    _titles(fig, "Sales land just above the purchase price more often than an indexed valuation implies",
            "Share of resales within 2% below the old price, and at or within 2% above it, by the gain "
            "the area index implies.\nDashed: the same shares implied by the index plus the spread of "
            "its errors in the calibration year.")
    _finish(fig, "fig1_near_purchase_price", top=0.72)


# ----------------------------------------------------------------------------- fig 2
def fig2_partial_dependence():
    p = pd.read_csv(RES / "pdp_m.csv")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for split, col in zip(["2008-2012", "2017-2019", "2022-2025"], [MUTED, AQUA, BLUE]):
        s = p[p.split == split].set_index("m").mean_pred
        s = 100 * (s - s.loc[0.0])
        ax.plot(100 * s.index, s.to_numpy(), color=col)
        ax.text(100 * s.index[-1] + 0.6, s.iloc[-1], f"test {split}", color=INK2, fontsize=9,
                va="center")
    _xpct(ax, "Index-implied nominal gain (m), indexed value held fixed")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylabel("Change in predicted price, % (relative to m = 0)")
    ax.set_xlim(-31, 39)
    _grid(ax)
    _titles(fig, "What the model with the purchase price learned about it",
            "Average change in M2's predicted price as the purchase price varies with the indexed value "
            "held fixed, by test period.\nEach model was trained only on resales before its test period.")
    _finish(fig, "fig2_partial_dependence", top=0.84)


# ----------------------------------------------------------------------------- fig 3
def fig3_loss_probability():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, split in zip(axes, ["2008-2012", "2022-2025"]):
        c = _curve(split, -20, 19)
        obs = 100 * (c.obs_loss_over_2 + c.obs_loss_within_2)
        dist = 100 * (c["M0-dist_loss_over_2"] + c["M0-dist_loss_within_2"])
        clf = 100 * (c["C2_loss_over_2"] + c["C2_loss_within_2"])
        ax.scatter(c.x, obs, s=16, color=INK, zorder=3, label="Observed")
        ax.plot(c.x, dist, color=BLUE, label="Indexed valuation + error distribution")
        ax.plot(c.x, clf, color=ORANGE, label="Classifier trained on earlier sales")
        ax.set_title(f"Resales {split}", loc="left")
        _xpct(ax)
        _grid(ax)
    axes[0].set_ylabel("Share of resales below the purchase price")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.885), ncol=3, fontsize=9)
    _titles(fig, "Predicting a sale below the purchase price",
            "Observed share of resales below the old price, and two predictions of it, by the gain the "
            "area index implies.")
    _finish(fig, "fig3_loss_probability", top=0.78)


# ----------------------------------------------------------------------------- fig 4
def fig4_interval_coverage():
    cov = pd.read_csv(RES / "coverage.csv")
    bands = ["below -10%", "-10% to 0", "0 to +10%", "+10% to +25%", "above +25%"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True)
    for ax, split in zip(axes, ["2008-2012", "2017-2019", "2022-2025"]):
        sub = cov[(cov.split == split) & (cov.group_type == "mband")]
        x = np.arange(len(bands))
        for k, (model, method, col, lab) in enumerate([
                ("M1", "global", BLUE, "M1, one correction"),
                ("M2", "global", ORANGE, "M2, one correction"),
                ("M2", "mondrian", AQUA, "M2, one correction per band")]):
            s = sub[(sub.model == model) & (sub.method == method)].set_index("group").reindex(bands)
            ax.plot(x, 100 * s.coverage, color=col, marker="o", ms=5, label=lab)
        ax.axhline(80, color=AXIS, linewidth=1)
        ax.set_xticks(x, ["< -10%", "-10–0", "0–10%", "10–25%", "> 25%"], fontsize=8.5)
        ax.set_title(f"Resales {split}", loc="left")
        ax.set_xlabel("Band of m")
        _grid(ax)
    axes[0].set_ylabel("Coverage of the 80% interval")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.885), ncol=3, fontsize=9)
    _titles(fig, "Interval coverage by distance from the purchase price",
            "Conformalised 80% intervals calibrated on the year before each test period.")
    _finish(fig, "fig4_interval_coverage", top=0.78)


def main() -> int:
    fig1_near_purchase_price()
    fig2_partial_dependence()
    fig3_loss_probability()
    fig4_interval_coverage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
