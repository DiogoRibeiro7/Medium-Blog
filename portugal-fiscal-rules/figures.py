"""Figures for the rules comparison. Called by run_analysis.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
# Categorical slots in fixed order; each rule keeps its colour in every figure.
COLOURS = {
    "fixed": "#2a78d6",
    "rigid": "#eb6834",
    "debt_pt": "#1baf7a",
    "debt_strong": "#eda100",
    "spending": "#e87ba4",
    "structural": "#008300",
}
SHORT = {
    "fixed": "Fixed surplus",
    "rigid": "Rigid surplus",
    "debt_pt": "Debt rule (Portugal's strength)",
    "debt_strong": "Strong debt rule",
    "spending": "Spending path",
    "structural": "Structural balance",
}
HALO = [pe.withStroke(linewidth=3, foreground=SURFACE)]
plt.rcParams.update(
    {
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 11,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "legend.labelcolor": INK,
    }
)


def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8.5, color=INK2, ha="left", va="bottom")


def spread_labels(ys: list[float], gap: float) -> list[float]:
    """Nudge end-of-line labels apart so that none are closer than gap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for _ in range(50):
        moved = False
        for a, b in zip(order[:-1], order[1:], strict=True):
            if out[b] - out[a] < gap:
                mid = 0.5 * (out[a] + out[b])
                out[a], out[b] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    return list(out)


def fig_one_future(res: dict, keys: list[str], path: Path):
    st = res["one_future"]["rules"]
    yrs = list(range(2027, 2036))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, var, title, unit in (
        (axes[0], "pb", "Primary balance", "% of GDP"),
        (axes[1], "debt", "Gross public debt", "% of GDP"),
    ):
        ax.axvspan(2027.5, 2029.5, color=GRID, alpha=0.6, linewidth=0, zorder=0)
        ends, handles = [], []
        for k in keys:
            s = st[k][var]
            ys = [s[str(y)] if str(y) in s else s[y] for y in yrs]
            handles += ax.plot(yrs, ys, color=COLOURS[k], linewidth=2, solid_capstyle="round")
            ends.append(ys[-1])
        if var == "debt":
            lo_, hi_ = ax.get_ylim()
            pos = spread_labels(ends, gap=0.06 * (hi_ - lo_))
            for k, y0, y1 in zip(keys, ends, pos, strict=True):
                ax.annotate(SHORT[k], (yrs[-1], y0), xytext=(yrs[-1] + 0.35, y1), fontsize=9, color=INK,
                            va="center", path_effects=HALO)  # fmt: skip
        else:
            ax.legend(handles, [SHORT[k] for k in keys], loc="lower right", fontsize=9, handlelength=1.6)
        ax.set_title(title, pad=10)
        ax.set_ylabel(unit)
        ax.set_xlim(2026.6, 2040.6 if var == "debt" else 2035.4)
        ax.set_xticks(range(2027, 2036, 2))
        ax.spines["bottom"].set_bounds(2027, 2035)
    axes[0].axhline(0, color=INK2, linewidth=0.8)
    axes[0].text(2028.5, axes[0].get_ylim()[1], "2009–2010\nshocks", ha="center", va="top", fontsize=9, color=INK2)
    axes[1].axhline(60, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    axes[1].text(2027.1, 60.4, "60%", fontsize=9, color=INK2, va="bottom")
    fig.suptitle("One bad future, six rules: Portugal's 2008–2010 shocks arrive again in 2027–2029",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)  # fmt: skip
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.12, wspace=0.3)
    note(fig, "Same resampled history for every rule; each rule calibrated to an average primary surplus of 2.5% "
              "in 2028–2035. Source: European Commission, AMECO (autumn 2026); author's model.")  # fmt: skip
    fig.savefig(path / "fig1_one_bad_future.png", dpi=160)
    plt.close(fig)


def fig_scorecard(res: dict, keys: list[str], path: Path):
    m = res["main"]
    cols = [
        ("p_at_or_below_60_2035", "Chance debt is at or\nbelow 60% in 2035", 100, "%", "higher is better"),
        ("p90_2035", "Debt in a bad 2035\n(90th percentile)", 1, "% of GDP", "lower is better"),
        ("p_breach", "Chance of a deficit above\n3% at least once by 2035", 100, "%", "lower is better"),
        ("p_forced_tightening", "Chance of a tightening of 1 pp\nor more in a downturn", 100, "%", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 5.2), sharey=True)
    rows = list(range(len(keys)))[::-1]
    for ax, (var, title, scale, unit, better) in zip(axes, cols, strict=True):
        vals = [m[k][var] * scale for k in keys]
        lo, hi = min(vals), max(vals)
        pad = 0.25 * (hi - lo) if hi > lo else 1
        for r, k, v in zip(rows, keys, vals, strict=True):
            ax.hlines(r, lo - pad, v, color=GRID, linewidth=1.2, zorder=1)
            ax.scatter([v], [r], s=90, color=COLOURS[k], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(f"{v:.0f}" if var != "p90_2035" else f"{v:.1f}", (v, r), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)  # fmt: skip
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_xlabel(f"{unit}, {better}", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(keys) - 0.4)
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([SHORT[k] for k in keys], fontsize=10, color=INK)
    fig.suptitle("Six rules at the same average effort: no rule is best on every count",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)  # fmt: skip
    fig.subplots_adjust(left=0.17, right=0.99, top=0.80, bottom=0.17, wspace=0.18)
    note(fig, "10,000 futures resampled from Portugal's 1998–2025 shocks; every rule averages a 2.5% primary surplus "
              "in 2028–2035. Source: author's model.")  # fmt: skip
    fig.savefig(path / "fig2_scorecard.png", dpi=160)
    plt.close(fig)


def fig_tradeoff(res: dict, keys: list[str], path: Path):
    m, sw = res["main"], res["kappa_sweep"]
    ks = sorted(sw, key=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    panels = (
        (axes[0], "p90_2035", 1, "Debt in a bad 2035 (90th percentile, % of GDP)", "Protection against a bad 2035"),
        (axes[1], "p_at_or_below_60_2035", 100, "Chance debt is at or below 60% in 2035 (%)", "Chance of reaching 60%"),
    )
    for ax, var, scale, ylab, title in panels:
        xs = [100 * sw[k]["p_forced_tightening"] for k in ks]
        ys = [sw[k][var] * scale for k in ks]
        ax.plot(xs, ys, color=INK2, linewidth=1.6, linestyle=(0, (4, 3)), zorder=2)
        for k_, x_, y_ in zip(ks, xs, ys, strict=True):
            ax.scatter([x_], [y_], s=14, color=INK2, zorder=2)
            if float(k_) == max(float(z) for z in ks):
                ax.annotate(f"response {float(k_):.1f}", (x_, y_), xytext=(8, 0), textcoords="offset points",
                            fontsize=8.5, color=INK2, ha="left", va="center", path_effects=HALO)  # fmt: skip
        for k in keys:
            x_, y_ = 100 * m[k]["p_forced_tightening"], m[k][var] * scale
            ax.scatter([x_], [y_], s=110, color=COLOURS[k], edgecolor=SURFACE, linewidth=1.8, zorder=4)
            off = OFFSETS[var].get(k, (9, 4, "left"))
            ax.annotate(SHORT[k], (x_, y_), xytext=off[:2], textcoords="offset points", fontsize=9, color=INK,
                        ha=off[2], va="center", path_effects=HALO, zorder=5)  # fmt: skip
        ax.set_xlabel("Chance of a tightening of 1 pp or more in a downturn (%)")
        ax.set_ylabel(ylab)
        ax.set_title(title, pad=10)
        ax.set_xlim(-2, 100)
    for ax in axes:
        ax.text(0.02, 0.03, "Dashed: debt rules, response to debt\nfrom 0 (the fixed surplus) to 0.4",
                transform=ax.transAxes, va="bottom", fontsize=9, color=INK2)  # fmt: skip
    fig.suptitle("A stronger response to debt buys protection with tightening, and gives up the good outcomes",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)  # fmt: skip
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.25)
    note(fig, "Each point is a rule calibrated to the same average primary surplus, 2.5% in 2028–2035. "
              "Source: author's model, 10,000 futures.")  # fmt: skip
    fig.savefig(path / "fig3_tradeoff.png", dpi=160)
    plt.close(fig)


OFFSETS = {
    "p90_2035": {"structural": (9, 9, "left"), "debt_pt": (9, -13, "left")},
    "p_at_or_below_60_2035": {
        "rigid": (0, 13, "center"),
        "debt_pt": (-9, -11, "right"),
        "structural": (9, -11, "left"),
        "spending": (9, -6, "left"),
    },
}


def _tern(w):
    """Barycentric (debt, breach, tightening) to the plane: debt at the top, breach left, tightening right."""
    wd, wb, wt = w
    return wt + 0.5 * wd, wd * np.sqrt(3) / 2


def fig_weights(res: dict, keys: list[str], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    panels = (
        (axes[0], res["weights"], "Debt judged by a bad 2035"),
        (axes[1], res["weights_target"], "Debt judged by the chance of 60%"),
    )
    for ax, wm, title in panels:
        pts = {k: [] for k in keys}
        for g in wm["grid"]:
            pts[g["best"]].append(_tern(g["w"]))
        for k in keys:
            if pts[k]:
                xy = np.array(pts[k])
                ax.scatter(xy[:, 0], xy[:, 1], s=26, marker="h", color=COLOURS[k], linewidth=0)
        for k in keys:
            share = wm["share"][k]
            if share >= 0.03:
                xy = np.array(pts[k]).mean(axis=0)
                label = f"{SHORT[k]}\n{100 * share:.0f}% of weightings"
                ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=9, color=INK,
                        path_effects=[pe.withStroke(linewidth=3.5, foreground=SURFACE)])  # fmt: skip
        tri = np.array([_tern((0, 1, 0)), _tern((0, 0, 1)), _tern((1, 0, 0)), _tern((0, 1, 0))])
        ax.plot(tri[:, 0], tri[:, 1], color=INK2, linewidth=1)
        ax.text(*_tern((1, 0, 0)), "All weight\non debt\n\n", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.text(-0.02, -0.06, "All weight on\navoiding a 3% breach", ha="left", va="top", fontsize=9, color=INK2)
        ax.text(1.02, -0.06, "All weight on avoiding\ntightening in a downturn", ha="right", va="top", fontsize=9,
                color=INK2)  # fmt: skip
        ax.set_title(title, pad=4)
        ax.set_aspect("equal")
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.2, 1.02)
        ax.axis("off")
    fig.suptitle("Which rule is best depends on how the three risks are weighed",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)  # fmt: skip
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.08, wspace=0.08)
    note(fig, "Each point is one weighting of the three risks, each rescaled to 0–1 across the six rules; the colour "
              "is the rule with the lowest weighted score. Source: author's model.")  # fmt: skip
    fig.savefig(path / "fig4_weights.png", dpi=160)
    plt.close(fig)


def fig_sensitivity(res: dict, keys: list[str], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    miss = sorted(res["by_gap_miss"], key=float)
    ax = axes[0]
    for k in keys:
        ys = [100 * res["by_gap_miss"][c][k]["p_forced_tightening"] for c in miss]
        ax.plot([100 * float(c) for c in miss], ys, color=COLOURS[k], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE)  # fmt: skip
    ax.legend([SHORT[k] for k in keys], loc="lower right", fontsize=9, handlelength=1.6, borderaxespad=0.2)
    ax.set_xlim(-4, 104)
    ax.set_ylim(-3, 88)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Share of a year's cyclical surprise that real-time estimates miss (%)")
    ax.set_ylabel("Chance of a tightening of 1 pp or more in a downturn (%)")
    ax.set_title("Rules that read the cycle depend on reading it well", pad=10)
    ax = axes[1]
    eff = sorted(res["by_effort"], key=float)
    ends = []
    for k in keys:
        ys = [res["by_effort"][e][k]["p90_2035"] for e in eff]
        ax.plot([float(e) for e in eff], ys, color=COLOURS[k], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE)  # fmt: skip
        ends.append(ys[-1])
    for k, y0, y1 in zip(keys, ends, spread_labels(ends, 1.3), strict=True):
        ax.annotate(SHORT[k], (3.0, y0), xytext=(3.05, y1), fontsize=9, color=INK, va="center", path_effects=HALO)
    ax.set_xlim(1.95, 3.75)
    ax.set_xticks([2.0, 2.5, 3.0])
    ax.spines["bottom"].set_bounds(2.0, 3.0)
    ax.set_xlabel("Average primary surplus in 2028–2035 (% of GDP)")
    ax.set_ylabel("Debt in a bad 2035 (90th percentile, % of GDP)")
    ax.set_title("More effort moves every rule, not the ranking", pad=10)
    fig.suptitle("What the comparison depends on", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.14, wspace=0.32)
    note(fig, "Baseline: half of the cyclical surprise missed, 2.5% average surplus. Source: author's model, "
              "10,000 futures per point.")  # fmt: skip
    fig.savefig(path / "fig5_sensitivity.png", dpi=160)
    plt.close(fig)


def draw_all(res: dict, rules: list, path: Path) -> None:
    keys = [r.key for r in rules]
    fig_one_future(res, keys, path)
    fig_scorecard(res, keys, path)
    fig_tradeoff(res, keys, path)
    fig_weights(res, keys, path)
    fig_sensitivity(res, keys, path)
