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
def fig1_error_by_year():
    met = pd.read_csv(RES / "metrics.csv")
    met = met[met.group_type == "year"].assign(year=lambda d: d.group.astype(int))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    styles = {"M0": (MUTED, "Indexed valuation (M0)"), "M1": (ORANGE, "Correction without m (M1)"),
              "M2": (BLUE, "Correction with m (M2)")}
    for split in ["2008-2012", "2017-2019", "2022-2025"]:
        sub = met[met.split == split]
        for model, (col, lab) in styles.items():
            s_ = sub[sub.model == model].sort_values("year")
            ax.plot(s_.year, 100 * s_.mdae, color=col, marker="o", ms=5,
                    label=lab if split == "2008-2012" else None)
        a, b = split.split("-")
        ax.text((int(a) + int(b)) / 2, 11.75, f"trained to {int(a) - 2}", ha="center",
                color=INK2, fontsize=9)
    ax.set_ylabel("Median absolute error, %")
    ax.set_ylim(8.0, 12.0)
    ax.set_xticks(range(2008, 2026))
    ax.set_xticklabels([str(y) if y in (2008, 2010, 2012, 2017, 2019, 2022, 2025) else ""
                        for y in range(2008, 2026)])
    ax.set_xlabel("Year of resale (test years only)")
    _grid(ax)
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    _titles(fig, "Out-of-time error of three valuation models, by year of resale",
            "Each test period is valued by models trained only on earlier resales and calibrated on the "
            "year before it.")
    _finish(fig, "fig1_error_by_year", top=0.88)


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
    _titles(fig, "What the model learned about the purchase price",
            "Average change in M2's predicted price when the purchase price varies and the indexed value "
            "is held fixed.\nThe model tested on 2008-2012 was trained on 1996-2006, when 0.3% of resales "
            "had m below zero.")
    _finish(fig, "fig2_partial_dependence", top=0.84)


# ----------------------------------------------------------------------------- fig 3
def fig3_loss_probability():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    lines = [("M0-dist", MUTED, "Indexed valuation + one error spread"),
             ("M2-dist-h", BLUE, "M2 + error spread by holding period"),
             ("C2", AQUA, "Classifier trained on earlier resales")]
    for ax, split in zip(axes, ["2008-2012", "2022-2025"]):
        c = _curve(split, -20, 19)
        c = c[c.n >= 1000]
        ax.scatter(c.x, 100 * (c.obs_loss_over_2 + c.obs_loss_within_2), s=16, color=INK, zorder=3,
                   label="Observed")
        for key, col, lab in lines:
            if f"{key}_loss_over_2" in c:
                ax.plot(c.x, 100 * (c[f"{key}_loss_over_2"] + c[f"{key}_loss_within_2"]), color=col,
                        label=lab)
        ax.set_title(f"Resales {split}", loc="left")
        _xpct(ax)
        _grid(ax)
    axes[0].set_ylabel("Share of resales below the purchase price")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.875), ncol=2, fontsize=9)
    _titles(fig, "Predicting a sale below the purchase price",
            "Observed share of resales below the old price and three predictions of it, by the gain the "
            "area index implies (1% bands with at least 1,000 resales).")
    _finish(fig, "fig3_loss_probability", top=0.74)


def main() -> int:
    fig1_error_by_year()
    fig2_partial_dependence()
    fig3_loss_probability()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
