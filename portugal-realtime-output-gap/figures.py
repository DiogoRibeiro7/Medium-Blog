"""Figures for the real-time output gap analysis. Called by run_analysis.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
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


def title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)


def fig_vintages(res: dict, path: Path):
    paths = res["part1"]["pt_paths"]
    releases = sorted(int(k) for k in paths if int(k) != 2026)
    fig, ax = plt.subplots(figsize=(11.5, 6))
    for col, vy in zip(RAMP, releases, strict=False):
        p = paths[str(vy)] if str(vy) in paths else paths[vy]
        ys = sorted(int(k) for k in p)
        ax.plot(ys, [p[str(y)] if str(y) in p else p[y] for y in ys], color=col, linewidth=1.6, zorder=2)
        ax.scatter([vy], [p[str(vy)] if str(vy) in p else p[vy]], s=36, color=col, edgecolor=SURFACE, zorder=4)
    lt = paths["2026"] if "2026" in paths else paths[2026]
    ys = [y for y in sorted(int(k) for k in lt) if y <= 2025]
    ax.plot(ys, [lt[str(y)] if str(y) in lt else lt[y] for y in ys], color=INK, linewidth=2.6, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.8)
    first = paths[str(releases[0])] if str(releases[0]) in paths else paths[releases[0]]
    ax.annotate("Autumn 2011 release", (2008, first.get("2008", first.get(2008))), xytext=(2004.2, -3.2),
                fontsize=9, color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})  # fmt: skip
    ax.annotate("Autumn 2026 release (current)", (2019, lt.get("2019", lt.get(2019))), xytext=(2012.2, 3.4),
                fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8})  # fmt: skip
    ax.set_xlim(2003, 2025.5)
    ax.set_xticks(range(2004, 2026, 3))
    ax.set_ylim(-7.2, 4.4)
    ax.set_ylabel("Output gap, % of potential GDP")
    title(fig, "Portugal's output gap, as each Commission release saw it")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.11)
    note(fig, "Lighter lines: older autumn releases (2011–2018, 2021); dots: each release's estimate for its own year. "
              "Source: European Commission, AMECO (archive and autumn 2026).")  # fmt: skip
    fig.savefig(path / "fig1_portugal_vintages.png", dpi=160)
    plt.close(fig)


AGE_LABEL = {-1: "Budget-time\nforecast", 0: "Same\nyear", 1: "1 year\nlater", 2: "2 years\nlater",
             3: "3 years\nlater", 4: "4 years\nlater", 5: "5 years\nlater"}  # fmt: skip


def fig_ages(res: dict, path: Path):
    ages = res["part1"]["ages"]
    ks = sorted(int(k) for k in ages)
    g = lambda k, q: ages[str(k)]["eu"][q] if str(k) in ages else ages[k]["eu"][q]  # noqa: E731
    gp = lambda k, q: ages[str(k)]["pt"][q] if str(k) in ages else ages[k]["pt"][q]  # noqa: E731
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(ks))
    ax.fill_between(x, [g(k, "p10") for k in ks], [g(k, "p90") for k in ks], color="#cde2fb", linewidth=0,
                    label="EU: middle 80% of errors")  # fmt: skip
    ax.fill_between(x, [g(k, "p25") for k in ks], [g(k, "p75") for k in ks], color="#86b6ef", linewidth=0,
                    label="EU: middle 50%")  # fmt: skip
    ax.plot(x, [g(k, "median") for k in ks], color=BLUE, linewidth=2, label="EU: median error")
    ax.plot(x, [gp(k, "bias") for k in ks], color=ORANGE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, label="Portugal: average error")  # fmt: skip
    for xi, k in zip(x, ks, strict=True):
        ax.text(xi, g(k, "p90") + 0.12, f"±{g(k, 'mae'):.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(x[-1] + 0.15, g(ks[-1], "p90") + 0.12, "EU mean absolute error", ha="left", va="bottom", fontsize=8.5,
            color=INK2)  # fmt: skip
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([AGE_LABEL[k] for k in ks], fontsize=9.5)
    ax.set_xlim(-0.4, len(ks) + 0.9)
    ax.set_ylabel("Estimate minus current estimate (points)")
    ax.legend(loc="lower right", fontsize=9)
    title(fig, "Output-gap errors shrink slowly: the budget-time forecast is off by 1.3 points on average")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.15)
    note(fig, "27 EU countries, target years 2010–2022, autumn releases 2011–2018 and 2021, against autumn 2026. "
              "Source: European Commission, AMECO; author's calculations.")  # fmt: skip
    fig.savefig(path / "fig2_errors_by_age.png", dpi=160)
    plt.close(fig)


def fig_compression(res: dict, path: Path):
    pts = res["part1"]["scatter"]
    pts = pts["-1"] if "-1" in pts else pts[-1]
    lam = res["part1"]["process"]["eu_age-1"]["lam"]
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    lo, hi = -9, 7
    xs = np.array([p["latest"] for p in pts])
    ys = np.array([p["estimate"] for p in pts])
    inside = (xs >= lo) & (xs <= hi) & (ys >= lo) & (ys <= hi)
    ax.scatter(xs[inside], ys[inside], s=22, color=INK2, alpha=0.35, linewidth=0, label="EU country-years")
    ax.plot([lo, hi], [lo, hi], color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(5.1, 6.4, "Estimate = outcome", fontsize=9, color=INK2, ha="right")
    a = res["part1"]["process"]["eu_age-1"]
    b0 = np.mean(ys) - lam * np.mean(xs)
    ax.plot([lo, hi], [b0 + lam * lo, b0 + lam * hi], color=BLUE, linewidth=2)
    ax.text(hi - 0.2, b0 + lam * hi - 0.5, f"Fitted slope {lam:.2f}", fontsize=9.5, color=INK, ha="right", va="top")
    pt = [p for p in pts if p["country"] == "PRT"]
    ax.scatter([p["latest"] for p in pt], [p["estimate"] for p in pt], s=60, color=ORANGE, edgecolor=SURFACE,
               linewidth=1.2, zorder=4, label="Portugal")  # fmt: skip
    for p in pt:
        ax.annotate(str(p["year"]), (p["latest"], p["estimate"]), xytext=(5, -10), textcoords="offset points",
                    fontsize=8.5, color=INK, path_effects=HALO)  # fmt: skip
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Output gap, current estimate (%)")
    ax.set_ylabel("Output gap, forecast in the autumn before the year (%)")
    ax.legend(loc="upper left", fontsize=9)
    title(fig, "Budget-time forecasts flatten the cycle")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.12)
    note(fig, f"{int((~inside).sum())} points outside the frame are not drawn; the slope uses all {len(pts)}. "
              f"Share of each year's surprise missed: {a['c']:.2f}. Source: AMECO; author's calculations.")  # fmt: skip
    fig.savefig(path / "fig3_compression.png", dpi=160)
    plt.close(fig)


def fig_effort(res: dict, path: Path):
    ef = res["part1"]["effort"]["sb"]
    pts = ef["points"]
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    lo, hi = -4, 4
    ax.axvspan(0.5, hi, ymin=0, ymax=(0.5 - lo) / (hi - lo), color="#f9d9ca", linewidth=0, zorder=0)
    ax.axvspan(lo, 0.5, ymin=(0.5 - lo) / (hi - lo), ymax=1, color="#f9d9ca", linewidth=0, zorder=0)
    ax.axvline(0.5, color=INK2, linewidth=0.9)
    ax.axhline(0.5, color=INK2, linewidth=0.9)
    ax.plot([lo, hi], [lo, hi], color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    xs = np.array([p["now"] for p in pts])
    ys = np.array([p["realtime"] for p in pts])
    inside = (xs >= lo) & (xs <= hi) & (ys >= lo) & (ys <= hi)
    ax.scatter(xs[inside], ys[inside], s=22, color=INK2, alpha=0.4, linewidth=0, label="EU country-years")
    pt = [p for p in pts if p["country"] == "PRT"]
    ax.scatter([p["now"] for p in pt], [p["realtime"] for p in pt], s=60, color=ORANGE, edgecolor=SURFACE,
               linewidth=1.2, zorder=4, label="Portugal")  # fmt: skip
    for p in pt:
        if p["year"] in (2016, 2017):
            continue
        dx, dy = {2013: (-30, 8), 2018: (8, -12)}.get(p["year"], (6, 6))
        ax.annotate(str(p["year"]), (p["now"], p["realtime"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=INK, path_effects=HALO)  # fmt: skip
    ax.text(hi - 0.1, lo + 0.15, "Judged short of 0.5\nin real time, met it\nas now measured", ha="right",
            va="bottom", fontsize=9, color=INK)  # fmt: skip
    ax.text(lo + 0.1, hi - 0.15, "Judged to meet 0.5 in real\ntime, short of it as now\nmeasured", ha="left",
            va="top", fontsize=9, color=INK)  # fmt: skip
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Change in the structural balance, current estimate (pp)")
    ax.set_ylabel("Change in the structural balance, estimated that autumn (pp)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08), fontsize=9)
    title(fig, f"The 0.5-point verdict changed in {100 * ef['flip_share']:.0f}% of country-years")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.12)
    note(fig, f"{ef['n']} country-years, 2013–2022; {int((~inside).sum())} outside the frame. Shaded: the two "
              "verdicts differ. Source: AMECO; author's calculations.")  # fmt: skip
    fig.savefig(path / "fig4_effort_verdicts.png", dpi=160)
    plt.close(fig)


def fig_rules(res: dict, keys: list[str], path: Path):
    sc = res["part2"]["scenarios"]
    show = [
        ("perfect", "Cycle measured exactly", BLUE, "o"),
        ("pt_budget", "Portugal's errors, 2011–2022", ORANGE, "^"),
        ("eu_budget", "EU errors at budget time", AQUA, "s"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    rows = list(range(len(keys)))[::-1]
    for ax, var, ttl in ((axes[0], "p_forced_tightening", "Chance of a tightening of 1 pp or more in a downturn"),
                         (axes[1], "p_breach", "Chance of a deficit above 3% at least once by 2035")):  # fmt: skip
        for r, k in zip(rows, keys, strict=True):
            vals = [100 * sc[s]["rules"][k][var] for s, *_ in show]
            ax.hlines(r, min(vals), max(vals), color=GRID, linewidth=3, zorder=1)
            for (_s, lab, col, mk), v in zip(show, vals, strict=True):
                ax.scatter([v], [r], s=80, color=col, marker=mk, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                           label=lab if r == rows[0] else None)  # fmt: skip
        ax.set_title(ttl, fontsize=10.5, pad=8)
        ax.set_xlabel("%")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-3, 100)
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([SHORT[k] for k in keys], color=INK)
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="center right", fontsize=9)
    axes[1].text(99, rows[1], "unaffected: does\nnot use the cycle", ha="right", va="center", fontsize=8.5, color=INK2)
    title(fig, "With EU-sized errors, every rule that reads the cycle squeezes in downturns")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.85, bottom=0.13, wspace=0.08)
    note(fig, "Six rules calibrated to an average primary surplus of 2.5% in 2028–2035; 10,000 futures. "
              "Source: author's model with errors estimated from AMECO releases.")  # fmt: skip
    fig.savefig(path / "fig5_rules_by_measurement.png", dpi=160)
    plt.close(fig)


def fig_bias(res: dict, keys: list[str], path: Path):
    known = res["part2"]["scenarios"]["pt_budget"]["rules"]
    unknown = res["part2"]["unknown_bias"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    rows = list(range(len(keys)))[::-1]
    for ax, var, scale, ttl, unit in (
        (axes[0], "effort", 1, "Average primary surplus, 2028–2035", "% of GDP"),
        (axes[1], "p_at_or_below_60_2035", 100, "Chance debt is at or below 60% in 2035", "%"),
    ):
        for r, k in zip(rows, keys, strict=True):
            a, b = known[k][var] * scale, unknown[k][var] * scale
            ax.annotate("", (b, r), xytext=(a, r), arrowprops={"arrowstyle": "->", "color": INK2, "lw": 1.4})
            ax.scatter([a], [r], s=70, color=BLUE, edgecolor=SURFACE, zorder=3,
                       label="Bias known and offset" if r == rows[0] else None)  # fmt: skip
            ax.scatter([b], [r], s=70, color=ORANGE, edgecolor=SURFACE, zorder=3,
                       label="Bias unknown" if r == rows[0] else None)  # fmt: skip
            if abs(a - b) > 1e-6:
                ax.text(b, r + 0.22, f"{b:.2f}" if scale == 1 else f"{b:.0f}", ha="center", fontsize=8.5, color=INK)
        ax.set_title(ttl, fontsize=10.5, pad=8)
        ax.set_xlabel(unit)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([SHORT[k] for k in keys], color=INK)
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="lower right", fontsize=9)
    title(fig, "A pessimistic reading of the cycle, if nobody corrects it, quietly lowers the effort")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.85, bottom=0.13, wspace=0.08)
    note(fig, "Portugal's real-time output gap was 0.7 points too low on average at budget time (2012–2022). "
              "Unknown: rules keep the level set for exact measurement. Source: author's model.")  # fmt: skip
    fig.savefig(path / "fig6_unknown_bias.png", dpi=160)
    plt.close(fig)


def draw_all(res: dict, rules: list, path: Path) -> None:
    keys = [r.key for r in rules]
    fig_vintages(res, path)
    fig_ages(res, path)
    fig_compression(res, path)
    fig_effort(res, path)
    fig_rules(res, keys, path)
    fig_bias(res, keys, path)
