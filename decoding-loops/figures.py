"""The six figures of the article, from results/summary.json (written by loops_numbers.py).

python figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9890", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SHADES = {
    "Pythia": ["#a9c8ef", "#7aaae6", "#4d8bdc", "#2a78d6", "#174d94"],
    "SmolLM2": ["#f5b49a", "#eb6834", "#a8401a"],
}
RULES_T = [("greedy", 0.0), ("T0.25", 0.25), ("T0.5", 0.5), ("T0.75", 0.75), ("T1", 1.0)]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": MUTED,
    }
)


def style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def ordered(summary):
    """Model keys by family, then size, with a colour for each."""
    out = []
    for fam in ("Pythia", "SmolLM2"):
        keys = sorted((m["params_m"], k) for k, m in summary["models"].items() if m["family"] == fam)
        out += [(k, fam, SHADES[fam][min(i, len(SHADES[fam]) - 1)]) for i, (_, k) in enumerate(keys)]
    return out


def label(key, summary):
    m = summary["models"][key]
    p = m["params_m"]
    return f"{m['family']} {p / 1000:g}B" if p >= 1000 else f"{m['family']} {p}M"


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def fig1_temperature(summary):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = [t for _, t in RULES_T]
    for key, _, col in ordered(summary):
        m = summary["models"][key]
        y = [100 * m[r]["rate"] for r, _ in RULES_T]
        ax.plot(x, y, color=col, lw=2, marker="o", ms=4, label=label(key, summary))
    ax.axhline(0, color=MUTED, lw=1.2, ls=":")
    ax.text(0.01, 2.5, "the books themselves: 0%", color=INK2, fontsize=8, ha="left")
    ax.set_xticks(x, ["0\n(greedy)", "0.25", "0.5", "0.75", "1\n(sample the\nmodel as is)"])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel("Continuations that end in a loop (%)")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "fig1_temperature.png")


def fig2_size(summary):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    panels = [
        (lambda m: 100 * m["greedy"]["rate"], "Greedy continuations that loop (%)", (-5, 103)),
        (lambda m: m["greedy"]["onset_median"], "Median token where the loop starts", None),
        (lambda m: m["greedy"]["period_median"], "Median length of the repeating block (tokens)", None),
    ]
    for ax, (f, title, ylim) in zip(axes, panels, strict=True):
        for fam, col in (("Pythia", BLUE), ("SmolLM2", ORANGE)):
            keys = [k for k, fm, _ in ordered(summary) if fm == fam]
            xs = [summary["models"][k]["params_m"] for k in keys]
            ys = [f(summary["models"][k]) for k in keys]
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=5, label=fam)
        if ylim:
            ax.set_ylim(*ylim)
            for fam, col in (("Pythia", BLUE), ("SmolLM2", ORANGE)):
                keys = [k for k, fm, _ in ordered(summary) if fm == fam]
                xs = [summary["models"][k]["params_m"] for k in keys]
                ax.plot(xs, [100 * summary["models"][k]["T1"]["rate"] for k in keys], color=col, lw=1.2, ls="--")
            ax.text(75, 4, "sampling at T = 1 (dashed): 0% at every size", fontsize=8, color=INK2)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xscale("log")
        sizes = sorted({summary["models"][k]["params_m"] for k in summary["models"]})
        ax.set_xticks(sizes, [f"{v / 1000:g}B" if v >= 1000 else f"{v}M" for v in sizes], fontsize=7, rotation=45)
        ax.minorticks_off()
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Parameters (millions)")
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    save(fig, "fig2_size.png")


def fig3_lockin(summary):
    keys = [k for k, _, _ in ordered(summary)]
    on = np.array([summary["greedy_onset"][k] for k in keys], dtype=float).T  # prompts x models
    books = summary["books"]
    cmap = LinearSegmentedColormap.from_list("onset", ["#0f3a75", "#2a78d6", "#7aaae6", "#bcd5f4"])
    masked = np.ma.masked_less(on, 0)
    fig, ax = plt.subplots(figsize=(7.6, 11))
    ax.set_facecolor("#e9dcc4")  # cells with no loop
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=256, interpolation="nearest")
    ax.set_xticks(range(len(keys)), [label(k, summary).replace(" ", "\n", 1) for k in keys], fontsize=7.5)
    ax.xaxis.tick_top()
    starts = [0] + [i for i in range(1, len(books)) if books[i] != books[i - 1]]
    for s in starts[1:]:
        ax.axhline(s - 0.5, color=SURFACE, lw=1.5)
    mids = [(s + e) / 2 - 0.5 for s, e in zip(starts, starts[1:] + [len(books)], strict=True)]
    ax.set_yticks(mids, [books[s] for s in starts], fontsize=7.5)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Token at which the greedy loop starts (beige: no loop)", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "fig3_lockin.png")


def fig4_position(summary):
    fig, ax = plt.subplots(figsize=(9, 5))
    shown = [("greedy", BLUE, "Greedy"), ("T0.5", AQUA, "Sampling, T = 0.5"), ("T1", ORANGE, "Sampling, T = 1"),
             ("human", MUTED, "The books themselves")]  # fmt: skip
    for rule, col, lab in shown:
        curves = np.array([m["position_p"][rule] for m in summary["models"].values() if "position_p" in m])
        x = np.arange(1, curves.shape[1] + 1)
        for c in curves:
            ax.plot(x, c, color=col, lw=0.6, alpha=0.35)
        ax.plot(x, curves.mean(axis=0), color=col, lw=2.4, label=lab)
    ax.set_xlim(1, 256)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Position in the continuation (tokens)")
    ax.set_ylabel("Probability the model gave the token it wrote")
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    fig.tight_layout()
    save(fig, "fig4_position.png")


def fig5_cycles(summary):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for key, _, col in ordered(summary):
        r = summary["models"][key].get("reinforcement")
        if r:
            y = [r["pre_p"], *r["cycle_p"]]
            ax.plot(range(len(y)), y, color=col, lw=2, marker="o", ms=4, label=label(key, summary))
    ax.set_xticks(range(7), ["before\nthe loop", "1st", "2nd", "3rd", "4th", "5th", "6th"])
    ax.set_xlabel("Repetition of the block")
    ax.set_ylabel("Mean probability of the chosen token")
    ax.set_ylim(0, 1)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    save(fig, "fig5_cycles.png")


def fig6_logprob(summary):
    rows = [("greedy", "Greedy"), ("T0.25", "T = 0.25"), ("T0.5", "T = 0.5"), ("T0.75", "T = 0.75"),
            ("top-p0.9", "Top-p 0.9"), ("T1", "T = 1"), ("human", "The books themselves")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, (rule, _) in enumerate(rows):
        for j, (key, _, col) in enumerate(ordered(summary)):
            v = summary["models"][key][rule]["logprob"]
            ax.scatter(v, i + (j - 3.5) * 0.06, color=col, s=26, zorder=3, edgecolor=SURFACE, lw=0.6)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Mean log-probability per token under the model that wrote it (nats; 0 = certain)")
    style(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", ls="", color=SHADES[f][-1], label=f) for f in SHADES]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(right=0.05)
    fig.tight_layout()
    save(fig, "fig6_logprob.png")


def main():
    FIGURES.mkdir(exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text())
    fig1_temperature(summary)
    fig2_size(summary)
    fig3_lockin(summary)
    fig4_position(summary)
    fig5_cycles(summary)
    fig6_logprob(summary)
    print(f"figures/fig1 to fig6 written ({len(summary['models'])} models)")


if __name__ == "__main__":
    main()
