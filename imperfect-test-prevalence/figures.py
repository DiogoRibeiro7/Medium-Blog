"""Draw the article's five figures from the arrays prevalence_numbers.py saves in results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

import prevalence_core as C  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT_BLUE, LIGHT_ORANGE, SHADE = "#cde2fb", "#fbd9c9", "#f3f2ee"
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
    "Data: Bendavid et al. (2020), versions 1 and 2, as transcribed by Gelman and Carpenter (2020)."
    " Code: Medium-Blog/imperfect-test-prevalence. Figure: Diogo Ribeiro."
)
SIM_CREDIT = (
    "Simulation calibrated to Bendavid et al. (2020)."
    " Code: Medium-Blog/imperfect-test-prevalence. Figure: Diogo Ribeiro."
)
WEIGHT = C.PUBLISHED_WEIGHTED / (C.FIELD_POS / C.FIELD_N)
CLAIM = (C.PUBLISHED_RANGE[0] / WEIGHT, C.PUBLISHED_RANGE[1] / WEIGHT)


def frame(fig, title, subtitle, credit, top_in=1.0, bottom_in=0.7):
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(0.01, 1 - 0.08 / h, title, fontsize=11, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 1 - 0.36 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top", linespacing=1.4)
    fig.text(0.01, 0.06 / h, credit, fontsize=7.2, color=MUTED, ha="left", va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def percent(ax, axis="y", digits=0):
    fmt = FuncFormatter(lambda v, _: f"{100 * v:.{digits}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=9.5, color=INK, pad=8)


# ----------------------------------------------------------------------------------------------- figure 1


def fig_false_alarms():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"wspace": 0.28})
    ys, ns = C.V1_SPEC
    pp = stats.betabinom(C.FIELD_N, ns - ys + 1.0, ys + 1.0)
    k = np.arange(0, 91)
    pmf = pp.pmf(k)
    ax1.bar(k, pmf, width=0.85, color=np.where(k >= C.FIELD_POS, ORANGE, LIGHT_BLUE))
    ax1.axvline(C.FIELD_POS, color=INK2, linewidth=1, linestyle="--")
    ax1.text(C.FIELD_POS + 1.5, pmf.max() * 0.92, f"{C.FIELD_POS} positives observed", fontsize=8, color=INK2)
    ax1.text(
        C.FIELD_POS + 6,
        pmf.max() * 0.35,
        f"{100 * pp.sf(C.FIELD_POS - 1):.0f}% chance of {C.FIELD_POS} or more\nfalse positives alone",
        fontsize=8,
        color=ORANGE,
    )
    ax1.set_xlabel(f"Positive tests among {C.FIELD_N:,} people, none of them infected")
    ax1.set_ylabel("Probability")
    percent(ax1, "y", 1)
    grid(ax1)
    panel_title(ax1, "With the specificity known from 399 of 401 negatives")

    sizes = np.array([s[1] for s in C.SPEC_STUDIES])
    fp = sizes - np.array([s[0] for s in C.SPEC_STUDIES])
    lo = stats.beta.ppf(0.025, fp + 0.5, sizes - fp + 0.5)
    hi = stats.beta.ppf(0.975, fp + 0.5, sizes - fp + 0.5)
    lo = np.where(fp == 0, 0.0, lo)
    rate = fp / sizes
    yp, np_ = C.pooled(C.SPEC_STUDIES)
    pooled = (np_ - yp) / np_
    ax2.axhline(pooled, color=MUTED, linewidth=1)
    ax2.text(1500, pooled + 0.002, f"pooled: {100 * pooled:.2f}%", fontsize=8, color=INK2, ha="right")
    ax2.vlines(sizes, lo, hi, color=BLUE, linewidth=1.6)
    ax2.scatter(sizes, rate, color=BLUE, s=30, zorder=3, edgecolor="white", linewidth=1.2)
    ax2.set_xscale("log")
    ax2.set_xticks([30, 100, 300, 1000], ["30", "100", "300", "1,000"])
    ax2.set_ylim(-0.003, 0.2)
    ax2.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    percent(ax2, "y", 0)
    ax2.set_xlabel("Known negatives tested in the study (log scale)")
    ax2.set_ylabel("False-positive rate (1 − specificity)")
    big = int(np.argmax(sizes))
    ax2.annotate(
        "1,102 tested,\nno false positive",
        (sizes[big], 0),
        (500, 0.06),
        fontsize=8,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )
    grid(ax2)
    panel_title(ax2, "The 13 specificity studies of version 2, with 95% intervals")
    frame(
        fig,
        "Could the 50 positives all have been false? With 401 validation samples, it could not be ruled out",
        "Left: positives expected among 3,330 uninfected people when specificity is only known from 399 correct of 401."
        "\nRight: the validation studies pooled later disagree more than chance allows; pooling hides that.",
        CREDIT,
    )
    fig.savefig(FIG / "fig1_false_alarms.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 2


def density(grid_, mass):
    w = np.gradient(grid_)
    return mass / w


def fig_posteriors():
    p = np.load(RES / "posteriors.npz")
    h = np.load(RES / "hierarchical.npz")
    panels = (
        ("Version 1 validation, pooled (399/401)", p["grid"], p["v1"]),
        ("Version 2 validation, pooled (3,308/3,324)", p["grid"], p["v2"]),
        ("Hierarchical, between-study prior scale 0.3", h["grid"], h["tau_0.3"]),
        ("Hierarchical, between-study prior scale 1", h["grid"], h["tau_1.0"]),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, gridspec_kw={"hspace": 0.4, "wspace": 0.18})
    for ax, (title, g, m) in zip(axes.ravel(), panels, strict=True):
        keep = g <= 0.05
        d = density(g, m)
        ax.axvspan(*CLAIM, color=SHADE, zorder=0)
        ax.fill_between(g[keep], d[keep], color=LIGHT_BLUE, linewidth=0)
        ax.plot(g[keep], d[keep], color=BLUE, linewidth=1.8)
        c = np.cumsum(m)
        p_claim = 1 - np.interp(CLAIM[0], g, c)
        p_low = np.interp(0.005, g, c)
        ax.text(
            0.98,
            0.92,
            f"P(≥ {100 * CLAIM[0]:.2f}%) {100 * p_claim:.0f}%\nP(< 0.5%) {100 * p_low:.0f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=INK,
        )
        ax.set_xlim(0, 0.04)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        percent(ax, "x", 1)
        grid(ax, "x")
        panel_title(ax, title)
    for ax in axes[1]:
        ax.set_xlabel("Prevalence in the sample (unweighted)")
    axes[0, 1].text(
        CLAIM[0] + 0.0003,
        axes[0, 1].get_ylim()[1] * 0.04,
        "published range,\nunweighted",
        fontsize=8,
        color=INK2,
        va="bottom",
    )
    frame(
        fig,
        "The same 50 positives, four ways of treating the test: the posterior widens as the assumptions get honest",
        "Posterior density of prevalence. Shaded: the published 2.49%–4.16% range divided by the weighting factor"
        " (2.81% / 1.50%),\nso that it is comparable with an unweighted prevalence. Tails beyond 4% are not shown.",
        CREDIT,
    )
    fig.savefig(FIG / "fig2_posteriors.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 3


def fig_joint():
    p = np.load(RES / "posteriors.npz")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, sp, pi, title in (
        (ax1, p["specs_v1"], p["pis_v1"], "Specificity from 399 of 401"),
        (ax2, p["specs_v2"], p["pis_v2"], "Specificity from 3,308 of 3,324, pooled"),
    ):
        ax.scatter(sp, pi, s=5, color=BLUE, alpha=0.35, linewidth=0)
        ax.set_xlim(0.982, 1.0)
        ax.set_xticks(np.arange(0.982, 0.9999, 0.004))
        ax.set_ylim(0, 0.03)
        percent(ax, "x", 1)
        percent(ax, "y", 1)
        ax.set_xlabel("Specificity")
        grid(ax)
        grid(ax, "x")
        panel_title(ax, title)
    ax1.set_ylabel("Prevalence")
    frame(
        fig,
        "Prevalence is mostly a statement about specificity",
        "2,000 draws from each joint posterior. Each point of specificity is worth about 1.2 points of prevalence"
        " (one divided by the sensitivity).",
        CREDIT,
        top_in=0.85,
    )
    fig.savefig(FIG / "fig3_joint.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 4


def fig_design():
    d = np.load(RES / "design.npz")
    splits, scen, widths, resolve = d["splits"], d["scenarios"], d["widths"], d["resolve"]
    budget = int(d["budget"])
    colors = (BLUE, ORANGE, AQUA)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, gridspec_kw={"wspace": 0.32})
    for i, (pi, col) in enumerate(zip(scen, colors, strict=True)):
        ax1.plot(splits, widths[i], color=col, linewidth=2, marker="o", markersize=4)
        best = int(np.argmin(widths[i]))
        ax1.scatter([splits[best]], [widths[i, best]], s=70, facecolor="white", edgecolor=col, linewidth=2, zorder=4)
        ax1.text(splits[-1] + 60, widths[i, -1], f"{100 * pi:.1f}%", fontsize=8.5, color=INK, va="center")
        ax2.plot(splits, resolve[i], color=col, linewidth=2, marker="o", markersize=4)
        ax2.text(splits[-1] + 60, resolve[i, -1], f"{100 * pi:.1f}%", fontsize=8.5, color=INK, va="center")
    for ax in (ax1, ax2):
        ax.axvline(401, color=INK2, linewidth=1, linestyle="--")
        ax.set_xlabel(f"Known negatives tested, out of {budget:,} tests")
        grid(ax)
    ax1.text(415, ax1.get_ylim()[1] * 0.97, "Santa Clara's split", fontsize=8, color=INK2, va="top")
    percent(ax1, "y", 1)
    ax1.set_ylabel("Median width of the 90% interval (points)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.1f}"))
    panel_title(ax1, "Precision: narrowest split circled")
    percent(ax2, "y", 0)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Surveys whose lower bound clears half the truth")
    panel_title(ax2, "Resolution: can it rule out the near-zero answer?")
    frame(
        fig,
        "With 3,853 tests, a third on known negatives beats Santa Clara's tenth; at 0.5% no split is enough",
        "300 simulated surveys per split, for true prevalences of 0.5%, 1.2% and 3.0% (labels at the line ends), with"
        " 122 known positives,\nspecificity 99.5% and sensitivity 83% in the survey's own lab. The rest of the tests"
        " go to the field survey.",
        SIM_CREDIT,
    )
    fig.savefig(FIG / "fig4_design.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 5


def fig_floor():
    d = np.load(RES / "floor.npz")
    sizes, same, sigmas, floors = d["sizes"], d["same"], d["sigmas"], d["floors"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sizes, same[:, 1] - same[:, 0], color=BLUE, linewidth=2.2, marker="o", markersize=5)
    ax.text(
        sizes[4] * 1.12,
        same[4, 1] - same[4, 0] + 0.0002,
        "validation in the survey's own lab",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )
    for s, (lo, hi) in zip(sigmas, floors, strict=True):
        ax.axhline(hi - lo, color=ORANGE, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(
            sizes[-1] * 2.1,
            hi - lo + 0.0002,
            f"other labs only, between-lab sd {s:g}",
            fontsize=8,
            color=INK2,
            ha="right",
            va="bottom",
        )
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
    ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 2.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.1f}"))
    ax.set_xlabel("Known negatives tested (log scale)")
    ax.set_ylabel("Width of the 95% interval for prevalence (points)")
    grid(ax)
    frame(
        fig,
        "Validation from other labs stops helping at a floor set by how much labs differ",
        "Santa Clara's 50 positives in 3,330, with sensitivity from 130 of 157. Blue: known negatives tested in the"
        " survey's own lab.\nOrange: specificity known only as a draw from a population of labs, even with unlimited"
        " validation in those labs.",
        SIM_CREDIT,
    )
    fig.savefig(FIG / "fig5_validation_floor.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_false_alarms()
    fig_posteriors()
    fig_joint()
    fig_design()
    fig_floor()


if __name__ == "__main__":
    main()
