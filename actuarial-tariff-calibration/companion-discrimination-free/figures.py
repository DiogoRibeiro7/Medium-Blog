"""Figures for the discrimination-free pricing companion. Run after fairness_numbers.py (reads figure_data.json).

fig1_who_the_women_are.png        Belgium: share of women in the exposure by driver age, engine power, bonus-malus
                                  level and fuel, against the portfolio share
fig2_premium_ratio_by_price.png   Belgium: women's average premium per policy-year over men's under the four prices,
                                  GLM and boosting, over 20 splits
fig3_gbm_effect_by_decile.png     Belgium: the boosting model's direct effect of sex by decile of the predicted
                                  probability that the policyholder is a woman, beside the GLM's constant effect
fig4_share_vs_tjur.png            The share of the direct effect restored by the unawareness price against Tjur's
                                  coefficient of discrimination: controlled experiment and Belgium
fig5_calibration_by_decile.png    Controlled experiment: true over charged claims by decile of P(D = 1 | x), for the
                                  unawareness and discrimination-free prices

Run:  python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
DATA = json.loads((HERE / "figure_data.json").read_text())

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#7a5cd6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
GRID = "#e6e5e1"
PRICES = ("best estimate", "unawareness", "implied unawareness", "discrimination-free")
PRICE_COL = {"best estimate": BLUE, "unawareness": ORANGE, "implied unawareness": VIOLET, "discrimination-free": AQUA}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "legend.handlelength": 3.0, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
})
CREDIT = ("Figure: Diogo Ribeiro. Data: beMTPL97 and freMTPL2 (CASdatasets). "
          "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/actuarial-tariff-calibration")


def credit(fig) -> None:
    fig.text(0.01, 0.005, CREDIT, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def figure_women() -> None:
    g = DATA["beMTPL97"]["groups"]
    panels = [("driver age", "driver age"), ("engine power (kW)", "engine power (kW)"),
              ("bonus-malus level", "bonus-malus level"), ("fuel", "fuel")]
    overall = sum(r[1] * r[2] for r in g["fuel"]) / sum(r[2] for r in g["fuel"])
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.2), gridspec_kw={"wspace": 0.28, "width_ratios": [7, 4, 6, 2.4]})
    for ax, (key, lab) in zip(axes, panels):
        rows = g[key]
        x = np.arange(len(rows))
        ax.bar(x, [r[1] for r in rows], width=0.62, color=BLUE)
        ax.axhline(overall, color=INK2, lw=1.2, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=0 if len(rows) < 5 else 35, fontsize=8.5)
        ax.set_ylim(0, 0.42)
        ax.set_title(lab)
        ax.grid(axis="x", visible=False)
        for xi, r in zip(x, rows):
            ax.text(xi, r[1] + 0.008, f"{r[1]:.2f}", ha="center", fontsize=7.5, color=INK2)
    axes[0].set_ylabel("share of women in the exposure")
    axes[0].text(len(g["driver age"]) - 0.6, overall + 0.012, "whole portfolio", fontsize=8, color=INK2, ha="right")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2)
    credit(fig)
    fig.savefig(OUT / "fig1_who_the_women_are.png")
    plt.close(fig)


def figure_ratio() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    for row, cls in enumerate(("GLM", "GBM")):
        r = DATA["beMTPL97"][cls]["ratio"]
        for j, k in enumerate(PRICES):
            y = row * 5 + j
            v = np.array(r[k])
            ax.plot([v.min(), v.max()], [y, y], color=PRICE_COL[k], lw=2.5, alpha=0.45, solid_capstyle="round")
            ax.plot(v.mean(), y, "o", color=PRICE_COL[k], ms=8, mec="white")
            ax.text(v.max() + 0.002, y, f"{v.mean():.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks([row * 5 + j for row in range(2) for j in range(4)])
    ax.set_yticklabels([f"{cls}: {k}" for cls in ("GLM", "GBM") for k in PRICES], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(1.0, color=MUTED, lw=1)
    ax.set_xlabel("women's average premium per policy-year over men's (dot: mean of 20 splits; bar: range)")
    ax.set_title("Belgium: what women pay relative to men under each price")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.9, bottom=0.17)
    credit(fig)
    fig.savefig(OUT / "fig2_premium_ratio_by_price.png")
    plt.close(fig)


def figure_effect() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for cls, col, mk in (("GLM", BLUE, "o"), ("GBM", ORANGE, "s")):
        dec = DATA["beMTPL97"][cls]["dec"]
        ax.plot(dec["p"], dec["eff"], "-", color=col, lw=2, marker=mk, ms=6, mec="white",
                label=f"{cls}: direct effect, by decile of its classifier's probability")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("mean predicted probability that the policyholder is a woman (decile)")
    ax.set_ylabel("log of a woman's price over a man's,\nsame rating factors")
    ax.set_title("Belgium: where each model puts the direct effect of sex")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.9, bottom=0.17)
    credit(fig)
    fig.savefig(OUT / "fig3_gbm_effect_by_decile.png")
    plt.close(fig)


def figure_share() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.plot([0, 0.7], [0, 0.7], ":", color=MUTED, lw=1.2)
    ax.text(0.66, 0.69, "share = Tjur's\ncoefficient", fontsize=8, color=INK2, ha="right", va="top")
    marks = {"additive": "o", "interaction": "D"}
    for s in DATA["synthetic"]["settings"]:
        if s["gamma"] != 0.2:
            continue
        mk = marks[s["shape"]]
        ax.plot(s["tjur"], s["GLM"]["share"], mk, color=BLUE, ms=8, mec="white")
        ax.plot(s["tjur"], s["GBM"]["share"], mk, color=ORANGE, ms=8, mec="white")
        ax.plot(s["tjur"], s["GLM"]["share_i"], mk, color=VIOLET, ms=6, mfc="none", mew=1.6)
    be = DATA["beMTPL97"]
    ax.plot(np.mean(be["GLM"]["tjur"]), be["GLM"]["pooled"], "*", color=INK, ms=13, mec="white")
    ax.annotate("Belgium, sex, GLM", (np.mean(be["GLM"]["tjur"]), be["GLM"]["pooled"]), xytext=(12, 18),
                textcoords="offset points", fontsize=8.5, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED})
    handles = [plt.Line2D([], [], ls="", marker="o", color=BLUE, ms=8, label="GLM refitted without D"),
               plt.Line2D([], [], ls="", marker="o", color=ORANGE, ms=8, label="boosting refitted without D"),
               plt.Line2D([], [], ls="", marker="o", color=VIOLET, mfc="none", mew=1.6, ms=7,
                          label="GLM fitted with D, averaged over the true P(D | x)"),
               plt.Line2D([], [], ls="", marker="o", color=INK2, ms=7, label="circles: proxy additive in the factors"),
               plt.Line2D([], [], ls="", marker="D", color=INK2, ms=7, label="diamonds: proxy through an interaction")]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    ax.set_xlim(-0.02, 0.7)
    ax.set_ylim(-0.04, 0.7)
    ax.set_xlabel("Tjur's coefficient of discrimination of P(D = 1 | x)")
    ax.set_ylabel("share of the direct effect restored by the unawareness price")
    ax.set_title("How much leaving D out leaves in")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
    credit(fig)
    fig.savefig(OUT / "fig4_share_vs_tjur.png")
    plt.close(fig)


def figure_calibration() -> None:
    sel = [s for s in DATA["synthetic"]["settings"] if s["shape"] == "additive" and s["gamma"] == 0.2 and s["kappa"] > 0]
    fig, axes = plt.subplots(1, len(sel), figsize=(13.2, 4.4), sharey=True, gridspec_kw={"wspace": 0.12})
    x = np.arange(1, 11)
    for ax, s in zip(axes, sel):
        for k, key, col, mk in (("unawareness", "U", ORANGE, "o"), ("discrimination-free", "DF", AQUA, "s")):
            ax.plot(x, s["GLM"][key]["dec"], "-", color=col, lw=2, marker=mk, ms=5.5, mec="white", label=k)
        ax.axhline(1, color=MUTED, lw=1)
        ax.set_title(f"proxy strength {s['kappa']:.0f}: Tjur {s['tjur']:.2f}")
        ax.set_xticks(x)
        ax.set_xlabel("decile of P(D = 1 | x), lowest first")
    axes[0].set_ylabel("true expected claims over charged premium")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.17)
    credit(fig)
    fig.savefig(OUT / "fig5_calibration_by_decile.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_women()
    figure_ratio()
    figure_effect()
    figure_share()
    figure_calibration()
