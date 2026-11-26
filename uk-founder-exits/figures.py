"""Draw the figures from the aggregates written by exits_numbers.py --figdata.

python figures.py --figdata figdata.json --out figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
GROUP_COLOURS = {"British": "#2a78d6", "non-British": "#eb6834", "mixed": "#1baf7a"}  # slots 1, 2, 3
GROUP_LABELS = {"British": "British founders", "non-British": "non-British founders", "mixed": "mixed teams"}
STATE_ORDER = [
    ("foreign_acquisition", "sold abroad", "#eb6834"),
    ("flip", "foreign parent of its own", "#6250d6"),
    ("offshore", "offshore parent", "#eda100"),
    ("uk_acquisition", "sold in the UK", "#2a78d6"),
    ("closure", "closed", "#52514e"),
    ("independent, raised again", "independent, raised again", "#1baf7a"),
    ("independent, no further allotment", "independent, no further allotment", "#d6d4cc"),
]
PANELS = [
    ("foreign_acquisition", "Sold to a foreign owner"),
    ("flip", "Moved under a foreign parent of its own"),
    ("uk_acquisition", "Sold to a UK owner"),
    ("offshore", "Moved under an offshore parent"),
    ("closure", "Closed"),
    ("next_allotment", "Allotted shares again, a year or more later"),
]
SPEC_ORDER = [
    "main (first allotment)",
    "origin at the first allotment with a resolution",
    "flips and offshore parents counted as foreign sales",
    "founders all living in the UK",
    "London only",
    "software only",
    "one founder with significant control",
    "incorporated 2016 to 2018",
    "incorporated 2019 to 2020",
]
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 10,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def source(fig, fd, extra="", y=-0.01):
    text = f"Source: Companies House, PSC snapshot of {fd['censor_date']}, filing histories; author's calculations."
    fig.text(0.01, y, (extra + " " if extra else "") + text, ha="left", va="top", fontsize=7.5, color=INK2)


def spread(values, gap):
    """Nudge label positions apart so that neighbours are at least `gap` apart, keeping their order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def curve(ax, cv, cause, group, band=True):
    c = cv[(cv.cause == cause) & (cv.group == group)].sort_values("years")
    colour = GROUP_COLOURS[group]
    if band:
        ax.fill_between(c.years, 100 * c.low, 100 * c.high, color=colour, alpha=0.14, lw=0)
    ax.plot(c.years, 100 * c.estimate, color=colour, lw=2, label=GROUP_LABELS[group])
    return c


def style_time_axis(ax):
    ax.set_xlim(0, 8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_any_foreign(fd, out):
    cv = pd.DataFrame(fd["curves"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ends = []
    for g in ("British", "non-British", "mixed"):
        c = curve(ax, cv, "any_foreign", g, band=g != "mixed")
        ends.append((g, c.years.iloc[-1], 100 * c.estimate.iloc[-1]))
    top = max(y for _, _, y in ends)
    for (g, x, y), ly in zip(ends, spread([y for _, _, y in ends], gap=0.09 * top), strict=True):
        ax.text(x + 0.08, ly, f"{GROUP_LABELS[g]}\n{y:.1f}%", color=INK2, fontsize=8.5, va="center")
    style_time_axis(ax)
    ax.set_xlim(0, 9.4)
    ax.set_xticks(range(0, 9))
    ax.set_xlabel("Years since the first allotment of new shares")
    ax.set_ylabel("Cumulative share under foreign or offshore control (%)")
    ax.set_title("UK technology companies that came under foreign or offshore control, by who founded them")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    n = fd["n"]
    source(fig, fd, f"Aalen-Johansen estimates, 95% bootstrap bands; companies: British {n['British']:,}, "
                    f"non-British {n['non-British']:,}, mixed {n['mixed']:,}.")  # fmt: skip
    fig.savefig(out / "fig1_foreign_control.png")
    plt.close(fig)


def fig_every_way_out(fd, out):
    cv = pd.DataFrame(fd["curves"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, (cause, title) in zip(axes.flat, PANELS, strict=True):
        for g in ("British", "non-British", "mixed"):
            curve(ax, cv, cause, g, band=g != "mixed")
        style_time_axis(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 2, 4, 6, 8])
    for ax in axes[1]:
        ax.set_xlabel("Years since the first allotment")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative incidence (%)")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Each way a funded company can stop being independent, and raising again", x=0.01, ha="left",
                 fontweight="bold", fontsize=12, color=INK)  # fmt: skip
    fig.tight_layout()
    source(fig, fd, "Each outcome is the first event after the first allotment; the others compete with it. "
                    "Axes differ between panels.")  # fmt: skip
    fig.savefig(out / "fig2_every_way_out.png")
    plt.close(fig)


def fig_states(fd, out):
    st = fd["states5"]
    groups = [g for g in ("British", "non-British", "mixed") if st.get(g, {}).get("n")]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, g in enumerate(groups[::-1]):
        left = 0.0
        for key, label, colour in STATE_ORDER:
            v = 100 * st[g].get(key, 0.0)
            ax.barh(i, v, left=left, color=colour, height=0.62, edgecolor=SURFACE, linewidth=1.5,
                    label=label if i == 0 else None)  # fmt: skip
            if v >= 6:
                ink = INK if colour in ("#eda100", "#d6d4cc") else "#ffffff"
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=ink)
            left += v
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{GROUP_LABELS[g]}\n({st[g]['n']:,})" for g in groups[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of companies (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Where companies stood five years after their first allotment of new shares")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.26))
    source(fig, fd, "Companies whose first allotment was at least five years before the snapshot.", y=-0.3)
    fig.savefig(out / "fig3_states_at_five_years.png")
    plt.close(fig)


def fig_hazard_ratios(fd, out):
    hr = pd.DataFrame(fd["sensitivity"])
    causes = [("foreign_acquisition", "Sold to a foreign owner"), ("uk_acquisition", "Sold to a UK owner"),
              ("closure", "Closed")]  # fmt: skip
    specs = [s for s in SPEC_ORDER if s in set(hr.spec)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.42 * len(specs) + 1.6), sharey=True)
    for ax, (cause, title) in zip(axes, causes, strict=True):
        h = hr[hr.cause == cause].set_index("spec")
        for i, s in enumerate(specs):
            if s not in h.index:
                continue
            r = h.loc[s]
            y = len(specs) - 1 - i
            colour = GROUP_COLOURS["British"] if i == 0 else INK2
            ax.plot([r.low, r.high], [y, y], color=colour, lw=1.6, solid_capstyle="round")
            ax.plot(r.hr, y, "o", color=colour, ms=6, mec=SURFACE, mew=1.5)
        ax.axvline(1, color=BASELINE, lw=1)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.7, 1, 1.4, 2])
        ax.set_xticklabels(["0.5", "0.7", "1", "1.4", "2"])
        ax.minorticks_off()
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Hazard ratio, British vs non-British founders")
    axes[0].set_yticks(range(len(specs)))
    axes[0].set_yticklabels(specs[::-1], fontsize=8.5)
    axes[0].tick_params(axis="y", length=0)
    fig.suptitle("The founder gap in each outcome, across choices of sample and definition", x=0.01, ha="left",
                 fontweight="bold", fontsize=12, color=INK)  # fmt: skip
    fig.tight_layout()
    source(fig, fd, "Cause-specific Cox models adjusted for incorporation year, activity, region, number of founders "
                    "and allotments in the first year; 95% intervals.")  # fmt: skip
    fig.savefig(out / "fig4_hazard_ratios.png")
    plt.close(fig)


def fig_origin(fd, out):
    og = pd.DataFrame(fd["origin5"])
    og = og[~og.group.str.contains(" minus ")]
    levels = list(dict.fromkeys(og.group))
    levels = [g for g in levels if g != "mixed"] + (["mixed"] if "mixed" in levels else [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(levels) + 1.6), sharey=True)
    for ax, (cause, title) in zip(axes, [("any_foreign", "Under foreign or offshore control"), ("closure", "Closed")],
                                  strict=True):  # fmt: skip
        o = og[og.cause == cause].set_index("group")
        for i, g in enumerate(levels):
            r = o.loc[g]
            y = len(levels) - 1 - i
            colour = GROUP_COLOURS["British"] if g == "British" else GROUP_COLOURS["non-British"]
            if g == "mixed":
                colour = GROUP_COLOURS["mixed"]
            ax.plot([100 * r.low, 100 * r.high], [y, y], color=colour, lw=1.6, solid_capstyle="round")
            ax.plot(100 * r.estimate, y, "o", color=colour, ms=6, mec=SURFACE, mew=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(left=0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Share five years after the first allotment (%)")
    axes[0].set_yticks(range(len(levels)))
    axes[0].set_yticklabels(levels[::-1])
    axes[0].tick_params(axis="y", length=0)
    fig.suptitle("Five-year outcomes by the nationality of the founders", x=0.01, ha="left", fontweight="bold",
                 fontsize=12, color=INK)  # fmt: skip
    fig.tight_layout()
    source(fig, fd, "Origins with at least 100 companies; 'several' means non-British founders of different origins.")
    fig.savefig(out / "fig5_founder_origin.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdata", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)
    fd = json.loads(Path(args.figdata).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_any_foreign, fig_every_way_out, fig_states, fig_hazard_ratios, fig_origin):
        f(fd, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
