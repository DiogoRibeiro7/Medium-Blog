"""Numbers and figures for the article on ADHD medication decisions in older adults.

Everything here is arithmetic on stated assumptions, not clinical data:

1. Two hypothetical distributions of individual changes in systolic blood
   pressure with the same average rise (+2.0 mmHg, the pooled adult estimate in
   Mick, McManus and Goldberg, 2013). One is homogeneous; in the other about one
   person in six rises by 10 mmHg or more, the share reported among older adults
   by Michielsen et al. (2020): 10 of 64 patients.
2. Absolute versus relative risk: the odds ratio per year of medication use
   reported by Zhang et al. (2023) applied to two illustrative baseline risks.
3. The drug holiday as an experiment: the probability that comparing weeks on
   and off medication points the wrong way, as a function of the number of
   on/off pairs and of the true benefit in units of week-to-week noise.

Run:  python prescribing_individuals.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path("figures")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
CREDIT = "Diogo Ribeiro · illustrative calculation, assumptions in the text"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- 1. averages
MEAN = 2.0                 # pooled systolic rise, mmHg (Mick et al. 2013)
SHARE_LARGE = 10 / 64      # share with a rise of 10 mmHg or more (Michielsen et al. 2020)


def mixture_params():
    """Two-component mixture with overall mean MEAN and P(rise >= 10) = SHARE_LARGE.

    Most people: small changes around m1 with sd 5. A minority w: rises around
    13 with sd 3. m1 is set so the overall mean is MEAN; w is solved so the
    share above 10 mmHg matches SHARE_LARGE.
    """
    s1, m2, s2 = 5.0, 13.0, 5.0

    def share_above(w):
        m1 = (MEAN - w * m2) / (1 - w)
        return (1 - w) * stats.norm.sf(10, m1, s1) + w * stats.norm.sf(10, m2, s2)

    lo, hi = 0.01, 0.3
    for _ in range(100):
        mid = (lo + hi) / 2
        if share_above(mid) < SHARE_LARGE:
            lo = mid
        else:
            hi = mid
    w = (lo + hi) / 2
    m1 = (MEAN - w * m2) / (1 - w)
    return w, m1, s1, m2, s2


def section_averages():
    w, m1, s1, m2, s2 = mixture_params()
    homo_sd = 3.0
    x = np.linspace(-20, 30, 1001)
    homo = stats.norm.pdf(x, MEAN, homo_sd)
    het = (1 - w) * stats.norm.pdf(x, m1, s1) + w * stats.norm.pdf(x, m2, s2)
    het_above10 = (1 - w) * stats.norm.sf(10, m1, s1) + w * stats.norm.sf(10, m2, s2)
    het_above20 = (1 - w) * stats.norm.sf(20, m1, s1) + w * stats.norm.sf(20, m2, s2)
    het_below0 = (1 - w) * stats.norm.cdf(0, m1, s1) + w * stats.norm.cdf(0, m2, s2)
    homo_above10 = stats.norm.sf(10, MEAN, homo_sd)
    print("== 1. same average, different people")
    print(f"mixture: {1 - w:.3f} x N({m1:.2f}, {s1}) + {w:.3f} x N({m2}, {s2}); mean "
          f"{(1 - w) * m1 + w * m2:.2f}")
    print(f"share rising >= 10 mmHg: mixture {het_above10:.3f} (target {SHARE_LARGE:.3f}), "
          f"homogeneous N({MEAN}, {homo_sd}) {homo_above10:.4f}")
    print(f"mixture: share rising >= 20 mmHg {het_above20:.3f}; share whose pressure falls {het_below0:.3f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, homo, color=C1, lw=2)
    ax.plot(x, het, color=C2, lw=2)
    ax.fill_between(x, het, where=x >= 10, color=C2, alpha=0.25, lw=0)
    ax.axvline(MEAN, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(MEAN + 0.5, ax.get_ylim()[1] * 0.97, "average rise, both: +2 mmHg",
            fontsize=8.5, color=INK2, va="top")
    ax.text(5.6, stats.norm.pdf(5.6, MEAN, homo_sd) + 0.004, "similar responses",
            fontsize=9, color=INK2)
    ax.text(13.0, het[np.searchsorted(x, 13.0)] + 0.01,
            f"different responses:\nabout 1 in {round(1 / het_above10)} rise by\n10 mmHg or more",
            fontsize=9, color=INK2, va="bottom")
    style(ax)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-20, 30)
    ax.set_xlabel("change in systolic blood pressure on medication (mmHg)")
    ax.set_title("Two populations with the same average effect", loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.17)
    fig.text(0.01, 0.01, CREDIT, fontsize=7.5, color=MUTED)
    fig.savefig(OUT / "fig1_same_average.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------ 2. relative risk
OR_PER_YEAR = 1.04   # adjusted odds ratio of cardiovascular disease per year of ADHD medication
                     # use (Zhang et al., JAMA Psychiatry, 2023), ages 6 to 64


def apply_odds_ratio(p, odds_ratio):
    odds = p / (1 - p) * odds_ratio
    return odds / (1 + odds)


def section_relative():
    print("\n== 2. the same relative effect on different baselines")
    years = 5
    orr = OR_PER_YEAR ** years
    print(f"odds ratio per year {OR_PER_YEAR}; over {years} years of use: {orr:.3f}")
    for label, base in [("illustrative baseline risk 2%", 0.02), ("illustrative baseline risk 20%", 0.20)]:
        p1 = apply_odds_ratio(base, orr)
        print(f"{label}: risk with medication {p1:.2%}; absolute increase {p1 - base:.2%} "
              f"(one extra case per {1 / (p1 - base):.0f} people)")
    print("UK 1995 pill scare (Gigerenzer et al.): 1 in 7,000 vs 2 in 7,000 = relative +100%, "
          "absolute +1 in 7,000")


# ------------------------------------------------------------ 3. drug holiday
def p_wrong(d, k):
    """Probability that the mean on-minus-off difference over k pairs has the wrong sign.

    d is the true benefit in units of the standard deviation of one on-minus-off
    difference; with k independent pairs the mean difference is N(d, 1/k) in those units.
    """
    return stats.norm.cdf(-d * np.sqrt(k))


def power_t(d, k, alpha=0.05):
    """Power of a one-sided paired t-test with k pairs and standardised effect d."""
    if k < 2:
        return float("nan")
    df = k - 1
    tcrit = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(tcrit, df, d * np.sqrt(k)))


def section_holiday():
    print("\n== 3. the drug holiday as an experiment")
    for d in (0.3, 0.5, 1.0):
        row = ", ".join(f"k={k}: {p_wrong(d, k):.1%}" for k in (1, 2, 3, 4, 6, 8, 10))
        print(f"benefit d={d}: probability the comparison points the wrong way -> {row}")
    for d in (0.5, 1.0):
        row = ", ".join(f"k={k}: {power_t(d, k):.1%}" for k in (3, 4, 6, 8, 10))
        print(f"benefit d={d}: power of a one-sided paired t-test at 5% -> {row}")
    for d in (0.5, 1.0):
        k = next(k for k in range(1, 200) if p_wrong(d, k) <= 0.05)
        print(f"benefit d={d}: pairs needed for at most 5% chance of pointing the wrong way: {k}")

    ks = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ends = []
    for d, col, lab in [(0.3, C1, "small benefit (0.3)"), (0.5, C2, "moderate benefit (0.5)"),
                        (1.0, C3, "large benefit (1.0)")]:
        y = p_wrong(d, ks)
        ax.plot(ks, y, color=col, lw=2, marker="o", ms=5)
        ends.append((y[-1], col, lab))
    style(ax)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 0.5)
    ax.set_xticks(ks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("number of on/off pairs of weeks compared")
    ax.set_ylabel("chance the comparison points the wrong way")
    placed = []
    for y, col, lab in sorted(ends):
        yy = max(y, placed[-1] + 0.03) if placed else y
        placed.append(yy)
        ax.plot([10.65, 10.95], [yy, yy], color=col, lw=2.5, clip_on=False, solid_capstyle="round")
        ax.text(11.05, yy, lab, va="center", fontsize=9, color=INK2)
    ax.set_title("A single drug holiday is a noisy experiment", loc="left", fontsize=11.5, pad=12)
    fig.text(0.01, 0.01, "Benefit measured in units of the week-to-week noise of the difference. "
             + CREDIT, fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.72, top=0.88, bottom=0.17)
    fig.savefig(OUT / "fig2_drug_holiday.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    section_averages()
    section_relative()
    section_holiday()
