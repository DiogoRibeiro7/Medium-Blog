"""Draw the article's six figures from the arrays bayes_numbers.py saves in results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

import bayes_core as B  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LIGHT_BLUE, LIGHT_ORANGE = "#cde2fb", "#fbd9c9"
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
VACCINE_CREDIT = (
    "Data: Polack et al. (2020), NEJM, and the trial protocol (prior and analysis plan)."
    " Code: Medium-Blog/bayes-problem-first. Figure: Diogo Ribeiro."
)
ORING_CREDIT = (
    "Data: O-ring thermal distress on 23 shuttle flights, Dalal, Fowlkes and Hoadley (1989), via vcd::SpaceShuttle."
    " Code: Medium-Blog/bayes-problem-first. Figure: Diogo Ribeiro."
)
SIM_CREDIT = "Simulation. Code: Medium-Blog/bayes-problem-first. Figure: Diogo Ribeiro."


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


def looks(ax, y_text=None):
    for n in B.LOOKS:
        ax.axvline(n, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
        if y_text is not None:
            ax.text(n + 1.5, y_text, f"{n}", fontsize=7.5, color=INK2, ha="left", va="bottom")


# ----------------------------------------------------------------------------------------------- figure 1


def fig_prior_posterior():
    prior = np.load(RES / "prior_ve.npz")
    post = np.load(RES / "posterior_ve.npz")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"wspace": 0.28})

    ve = prior["ve"]
    shown = ve[ve > -3.0]
    off = np.mean(ve <= -3.0)
    bins = np.linspace(-3.0, 1.0, 81)
    h, edges = np.histogram(shown, bins=bins)
    h = h / ve.size
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax1.bar(centres, h, width=np.diff(edges) - 0.004, color=np.where(centres < 0, LIGHT_ORANGE, LIGHT_BLUE))
    ax1.axvline(B.VE_BAR, color=INK2, linewidth=1, linestyle="--")
    ax1.text(
        B.VE_BAR - 0.04, h.max() * 0.97, "VE = 30%,\nthe prior's mean", fontsize=8, color=INK2, ha="right", va="top"
    )
    ax1.text(
        -2.95, h.max() * 0.55, f"{100 * off:.0f}% of the prior lies\nbelow −300%, off the chart", fontsize=8, color=INK2
    )
    ax1.text(-1.6, h.max() * 0.2, "vaccine harmful", fontsize=8, color=ORANGE)
    percent(ax1, "x")
    ax1.set_xlabel("Vaccine efficacy")
    ax1.set_ylabel("Share of prior draws per 5-point bin")
    percent(ax1, "y", 1)
    grid(ax1)
    panel_title(ax1, "Before any case: the protocol's prior, Beta(0.700102, 1)")

    th, w = post["theta"], post["w"]
    ve_grid = B.ve_from_theta(th)
    keep = (ve_grid > 0.82) & (ve_grid < 0.995)
    # density on the VE scale: weight per theta cell divided by the VE width of the cell
    dens_grid = w[keep] / np.abs(np.gradient(ve_grid))[keep]
    a, b = B.conjugate(B.VACCINE_CASES, B.TOTAL_CASES, *B.PROTOCOL_PRIOR)
    vv = np.linspace(0.82, 0.995, 400)
    tt = B.theta_from_ve(vv)
    dens_exact = stats.beta.pdf(tt, a, b) / (2 - vv) ** 2  # |d theta / d VE| = 1 / (2 - VE)^2
    ax2.hist(
        post["sampler"],
        bins=np.linspace(0.82, 0.995, 71),
        density=True,
        color=LIGHT_BLUE,
        label="sampler, 16,000 draws",
    )
    ax2.plot(ve_grid[keep], dens_grid, color=BLUE, linewidth=2, label="grid, 20,000 points")
    ax2.plot(vv, dens_exact, color=ORANGE, linewidth=1.6, linestyle=(0, (4, 3)), label="conjugate algebra")
    ax2.set_xticks(np.arange(0.84, 1.001, 0.04))
    percent(ax2, "x")
    ax2.set_xlabel("Vaccine efficacy")
    ax2.set_ylabel("Posterior density")
    ax2.legend(loc="upper left", fontsize=8)
    grid(ax2)
    panel_title(ax2, "After 170 cases, 8 of them vaccinated: three routes, one answer")
    frame(
        fig,
        "The prior the vaccine trial wrote down, and the posterior it reached",
        "Left: prior draws of vaccine efficacy (VE). The prior is weak (worth 1.7 cases) and stretches far into harm."
        "\nRight: the posterior computed on a grid, by a Metropolis sampler and by the Beta-binomial formula.",
        VACCINE_CREDIT,
    )
    fig.savefig(FIG / "fig1_prior_posterior.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 2


def fig_accumulation():
    d = np.load(RES / "accumulation.npz")
    n = d["n"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"wspace": 0.28})
    for k in range(d["med"].shape[0]):
        ax1.plot(n, d["med"][k], color=BLUE, alpha=0.06, linewidth=0.8)
        ax1.plot(n, d["lo"][k], color=ORANGE, alpha=0.06, linewidth=0.8)
    ax1.plot(n, d["med"][0], color=BLUE, linewidth=2)
    ax1.plot(n, d["lo"][0], color=ORANGE, linewidth=2)
    ax1.text(172, d["med"][0, -1] + 0.015, "posterior median", fontsize=8, color=INK2, va="bottom", ha="right")
    ax1.text(172, d["lo"][0, -1] - 0.03, "lower end of the\n95% interval", fontsize=8, color=INK2, va="top", ha="right")
    ax1.axhline(B.VE_BAR, color=INK2, linewidth=1, linestyle="--")
    ax1.text(172, B.VE_BAR + 0.02, "VE = 30%", fontsize=8, color=INK2, ha="right")
    looks(ax1, y_text=-0.58)
    ax1.set_ylim(-0.6, 1.06)
    ax1.set_xlim(0, 175)
    percent(ax1)
    ax1.set_xlabel("Cases counted so far")
    ax1.set_ylabel("Vaccine efficacy")
    grid(ax1)
    panel_title(ax1, "The posterior narrows as cases arrive")

    miss = np.clip(1 - d["p_above"], 1e-300, 1)
    for k in range(miss.shape[0]):
        ax2.plot(n, miss[k], color=BLUE, alpha=0.08, linewidth=0.8)
    ax2.plot(n, miss[0], color=BLUE, linewidth=2)
    ax2.axhline(1 - B.THRESHOLD_INTERIM, color=ORANGE, linewidth=1.2)
    ax2.text(174, (1 - B.THRESHOLD_INTERIM) * 1.6, "interim bar: 0.5% left", fontsize=8, color=INK2, ha="right")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-10, 1.5)
    ax2.set_xlim(0, 175)
    looks(ax2)
    ax2.set_xlabel("Cases counted so far")
    ax2.set_ylabel("Posterior probability that VE is 30% or less")
    grid(ax2)
    panel_title(ax2, "Doubt that the vaccine clears 30%, on a log scale")
    frame(
        fig,
        "How the posterior moved as the trial's 170 cases came in",
        "Each thin line is one of 200 random orders of arrival consistent with the final count"
        " (8 vaccine, 162 placebo);"
        "\nthe thick line is one of them. Dotted verticals: the planned analyses at 32, 62, 92, 120 and 164 cases.",
        VACCINE_CREDIT,
    )
    fig.savefig(FIG / "fig2_accumulation.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 3

PRIOR_LABELS = {
    "protocol": "Protocol prior, worth 1.7 cases",
    "flat": "Flat prior, worth 2 cases",
    "sceptic": "Sceptic (centred on VE = 0), worth 100 cases",
    "enthusiast": "Enthusiast (centred on VE = 90%), worth 20 cases",
}


def fig_sensitivity():
    d = np.load(RES / "sensitivity.npz")
    n = d["n"]
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 6.6), sharex=True, sharey=True, gridspec_kw={"hspace": 0.32, "wspace": 0.1}
    )
    for ax, name in zip(axes.ravel(), PRIOR_LABELS, strict=True):
        med, lo, hi = d[name]
        ax.fill_between(n, lo, hi, color=LIGHT_BLUE, linewidth=0)
        ax.plot(n, med, color=BLUE, linewidth=2)
        ax.axhline(B.VE_BAR, color=INK2, linewidth=1, linestyle="--")
        ax.text(170, med[-1] + 0.04, f"{100 * med[-1]:.0f}%", fontsize=8.5, color=INK, ha="right", va="bottom")
        ax.set_ylim(-0.6, 1.02)
        ax.set_xlim(0, 172)
        percent(ax)
        grid(ax)
        panel_title(ax, PRIOR_LABELS[name])
    for ax in axes[1]:
        ax.set_xlabel("Cases counted so far")
    for ax in axes[:, 0]:
        ax.set_ylabel("Vaccine efficacy")
    frame(
        fig,
        "Four priors, one stream of cases: the data overrule weak priors fast and strong ones slowly",
        "Posterior median (line) and 95% credible interval (band) of vaccine efficacy"
        " along one random order of the 170 cases."
        "\nThe dashed line is VE = 30%. A prior's weight is its worth in cases divided by that plus the cases counted.",
        VACCINE_CREDIT,
    )
    fig.savefig(FIG / "fig3_prior_sensitivity.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 4


def fig_oring_prior():
    d = np.load(RES / "orings.npz")
    t = d["temps"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, curves, title in (
        (ax1, d["prior_vague"], "Vague prior: Normal(0, 10) on both coefficients"),
        (ax2, d["prior_wi"], "Weakly informative: Normal(−3, 1.5) and Normal(0, 1)"),
    ):
        ax.axvspan(53, 81, color="#f3f2ee", zorder=0)
        for c in curves:
            ax.plot(t, c, color=BLUE, alpha=0.35, linewidth=0.9)
        ax.axvline(B.LAUNCH_F, color=ORANGE, linewidth=1.2)
        ax.text(B.LAUNCH_F + 1, 1.02, "31°F", fontsize=8, color=INK2, va="bottom")
        ax.text(67, 1.02, "flights flown", fontsize=8, color=INK2, ha="center", va="bottom")
        ax.set_ylim(-0.02, 1.1)
        ax.set_xlim(25, 85)
        percent(ax)
        ax.set_xlabel("Launch temperature, °F")
        grid(ax)
        panel_title(ax, title)
    ax1.set_ylabel("Chance that one O-ring shows distress")
    frame(
        fig,
        "Before the data: what each prior believes about O-rings and temperature",
        "Sixty curves drawn from each prior (intercept: log-odds of distress at 70°F; slope: per 10°F)."
        "\nThe vague prior puts most curves at 0% or 100%: it is not neutral, it is extreme.",
        ORING_CREDIT,
    )
    fig.savefig(FIG / "fig4_oring_prior_predictive.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 5


def fig_oring_posterior():
    d = np.load(RES / "orings.npz")
    t = d["temps"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axvspan(53, 81, color="#f3f2ee", zorder=0)
    lo, med, hi = d["wi"]
    ax.fill_between(t, lo, hi, color=LIGHT_BLUE, linewidth=0)
    ax.plot(t, med, color=BLUE, linewidth=2, label="weakly informative prior: median and 90% band (shaded)")
    lo, med, hi = d["vague"]
    ax.plot(t, med, color=ORANGE, linewidth=2, label="vague prior: median and 90% band (dashed)")
    for edge in (lo, hi):
        ax.plot(t, edge, color=ORANGE, linewidth=1.2, linestyle=(0, (4, 3)))
    temp, y = d["temp"], d["distressed"]
    vals, counts = np.unique(np.column_stack([temp, y]), axis=0, return_counts=True)
    ax.scatter(
        vals[:, 0],
        vals[:, 1] / 6,
        s=28 + 22 * (counts - 1),
        color=INK,
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
        label="flights: share of 6 rings distressed",
    )
    ax.axvline(B.LAUNCH_F, color=INK2, linewidth=1, linestyle="--")
    ax.text(B.LAUNCH_F + 0.8, 1.03, "Challenger, 31°F", fontsize=8, color=INK2, va="bottom")
    ax.text(67, 1.03, "temperatures flown before, 53–81°F", fontsize=8, color=INK2, ha="center", va="bottom")
    ax.set_xlim(25, 85)
    ax.set_ylim(-0.02, 1.1)
    percent(ax)
    ax.set_xlabel("Launch temperature, °F")
    ax.set_ylabel("Chance that one O-ring shows distress")
    ax.legend(loc="center right", fontsize=8)
    grid(ax)
    frame(
        fig,
        "After the data: the posterior is tight where flights were flown and wide where Challenger flew",
        "Binomial logistic regression of distressed O-rings (of 6 per flight) on launch temperature,"
        " 23 flights before Challenger."
        "\nBetween 53 and 81°F the two priors agree; at 31°F the data say little and the prior's slope scale matters.",
        ORING_CREDIT,
    )
    fig.savefig(FIG / "fig5_oring_posterior.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 6


def fig_sbc():
    d = np.load(RES / "sbc.npz")
    names = [
        ("correct", "Correct sampler"),
        ("stopped_too_early", "Stopped too early"),
        ("data_counted_twice", "Data counted twice"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True, gridspec_kw={"wspace": 0.12})
    sims, bins = 2000, 20
    expect = sims / bins
    lo, hi = stats.binom.ppf([0.005, 0.995], sims, 1 / bins)
    for ax, (key, title) in zip(axes, names, strict=True):
        counts = d[key]
        ax.axhspan(lo, hi, color="#f3f2ee", zorder=0)
        ax.axhline(expect, color=MUTED, linewidth=1)
        ax.bar(np.arange(bins), counts, width=0.85, color=BLUE)
        ax.set_xticks([0, 19], ["below all draws", "above all draws"])
        ax.tick_params(axis="x", labelsize=8)
        left, right = ax.get_xticklabels()
        left.set_ha("left")
        right.set_ha("right")
        grid(ax)
        panel_title(ax, title)
    axes[0].set_ylabel("Simulated trials (of 2,000)")
    axes[1].set_xlabel("Rank of the true share among 99 posterior draws, in 20 bins")
    axes[0].text(19.4, hi + 6, "99% band if\nuniform", fontsize=7.5, color=INK2, ha="right", va="bottom")
    frame(
        fig,
        "Checking a sampler without algebra: where does the truth fall among the posterior draws?",
        "Simulation-based calibration: draw a true share from the prior, simulate 170 cases,"
        " run the sampler, rank the truth."
        "\nA correct sampler gives a flat histogram; a broken one piles the truth into the extremes.",
        SIM_CREDIT,
        bottom_in=0.85,
    )
    fig.savefig(FIG / "fig6_calibration.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_prior_posterior()
    fig_accumulation()
    fig_sensitivity()
    fig_oring_prior()
    fig_oring_posterior()
    fig_sbc()


if __name__ == "__main__":
    main()
