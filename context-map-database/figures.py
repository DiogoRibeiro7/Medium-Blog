"""The three figures. House style shared with the other Medium-Blog companions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP4 = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # ordinal blue, light to dark
RAMP3 = ("#86b6ef", "#2a78d6", "#104281")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 200,
})

# -- Figure 1: the schema, and the same schema with one more arrow ------------------------
POS = {"Cloud": (0.50, 0.80), "Staging": (0.17, 0.30), "Production": (0.60, 0.30), "Incident": (0.88, -0.12)}
COLS = {
    "Cloud": "id · token · meaning · grants",
    "Staging": "id · token · meaning · grants\ns → Cloud",
    "Production": "id · token · meaning · grants\np → Cloud · u → Incident",
    "Incident": "id · token · meaning · grants\ni → Production",
}
W, H = 0.30, 0.17


def _box(ax, name, extra=""):
    x, y = POS[name]
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#f3f2ee", ec=INK2, lw=1.0, zorder=2))
    ax.text(x, y + H / 2 - 0.035, name, ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=3)
    ax.text(x, y - 0.02, COLS[name] + extra, ha="center", va="center", fontsize=7.6, color=INK2,
            linespacing=1.35, zorder=3)


def _arrow(ax, a, b, label, color=INK2, rad=0.0, ls="-", lw=1.4, lab_off=(0, 0), shrink=34):
    pa, pb = np.array(POS[a]), np.array(POS[b])
    arr = FancyArrowPatch(pa, pb, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=13,
                          color=color, lw=lw, linestyle=ls, shrinkA=shrink, shrinkB=shrink, zorder=1)
    ax.add_patch(arr)
    mid = (pa + pb) / 2
    dx, dy = pb - pa
    # matplotlib's arc3 puts the control point at mid + rad * (dy, -dx); the curve peaks halfway
    lp = mid + 0.5 * rad * np.array([dy, -dx]) + np.array(lab_off)
    ax.text(*lp, label, ha="center", va="center", fontsize=10.5, style="italic", color=color, zorder=4,
            bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none"))


def figure_schema(path_out: str = "fig1_schema.png") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    for k, ax in enumerate(axes):
        ax.set_xlim(-0.02, 1.08)
        ax.set_ylim(-0.27, 0.98)
        ax.set_axis_off()
        cloud_extra = "\nw → Production" if k == 1 else ""
        for name in POS:
            _box(ax, name, cloud_extra if name == "Cloud" else "")
        hot = ORANGE if k == 1 else INK2
        _arrow(ax, "Staging", "Cloud", "s", color=hot, lw=2.0 if k == 1 else 1.4, lab_off=(-0.035, 0.0))
        _arrow(ax, "Production", "Cloud", "p", rad=-0.18, lab_off=(-0.035, 0.0))
        _arrow(ax, "Incident", "Production", "i", rad=-0.25, lab_off=(0.03, -0.01))
        _arrow(ax, "Production", "Incident", "u", rad=-0.25, lab_off=(-0.03, 0.01))
        ax.text(0.98, 0.32, "u;i = id", fontsize=9.5, style="italic", color=INK, ha="right")
        if k == 1:
            _arrow(ax, "Cloud", "Production", "w", color=ORANGE, rad=-0.18, lw=2.0, ls=(0, (4, 2)),
                   lab_off=(0.035, 0.0))
            ax.text(0.98, 0.25, "w;p = id", fontsize=9.5, style="italic", color=INK, ha="right")
            ax.text(0.02, -0.23, "Hom(Staging, Production) = { s;w }", fontsize=10, color=ORANGE,
                    fontweight="bold", ha="left")
            ax.set_title("(b) one more arrow, w: Cloud → Production", loc="left")
        else:
            ax.text(0.02, -0.23, "Hom(Staging, Production) = ∅", fontsize=10, color=INK, fontweight="bold",
                    ha="left")
            ax.set_title("(a) the context schema: four tables, four foreign keys", loc="left")
    fig.tight_layout()
    out = FIG / path_out
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# -- Figure 2: learned transports ----------------------------------------------------------
def figure_drift(curve: list[dict], power: dict, path_out: str = "fig2_learned_loops.png") -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    dims = sorted({r["d"] for r in curve})
    for col, d in zip(RAMP3, dims):
        rows = [r for r in curve if r["d"] == d]
        n = np.array([r["n"] for r in rows])
        ax1.plot(n, [r["rms_analytic"] for r in rows], color=col, lw=2.0, zorder=2)
        ax1.scatter(n, [r["rms_sim"] for r in rows], s=46, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax1.text(n[-1] * 1.12, rows[-1]["rms_analytic"], f"d = {d}", color=INK, va="center", fontsize=9.5)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlim(None, 16384 * 2.6)
    ax1.set_xlabel("pairs used to fit each adapter, n")
    ax1.set_ylabel("RMS relative loop drift, |B̂Âx − x| / |x|")
    ax1.set_title("(a) A loop that should return exactly never does", loc="left")
    ax1.text(0.03, 0.05, "points: simulation   lines: σ√(d(1/(n−d−1) + 1/(n−m−1)))\nσ = 0.1, m = 2d; written maps give 8e−16",
             transform=ax1.transAxes, fontsize=8.6, color=INK2)

    ks = np.array(power["multiples"])
    for col, row in zip(RAMP4, power["by_n"]):
        ax2.plot(ks, row["power"], color=col, lw=2.0, marker="o", ms=4.5, label=f"n = {row['n']:,}")
    last = power["by_n"][1]
    ax2.plot(ks, last["power_noise_overestimated_10pct"], color=ORANGE, lw=2.0, ls=(0, (4, 2)),
             label=f"n = {last['n']:,}, noise overstated by 10%")
    ax2.axhline(0.8, color=INK2, lw=0.9, ls=":")
    ax2.axhline(power["alpha"], color=INK2, lw=0.9, ls=":")
    ax2.text(1.0, 0.82, "80% power", ha="right", fontsize=8.5, color=INK2)
    ax2.text(1.0, 0.07, "5% false alarms", ha="right", fontsize=8.5, color=INK2)
    ax2.set_xlabel("size of the break, as a multiple of the drift with no break")
    ax2.set_ylabel("share of broken laws flagged")
    ax2.set_title("(b) The loop check is a test, and needs the noise level", loc="left")
    ax2.set_ylim(-0.03, 1.05)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.97), fontsize=8.6)
    fig.tight_layout()
    out = FIG / path_out
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# -- Figure 3: grounding ---------------------------------------------------------------------
def figure_grounding(study: dict, path_out: str = "fig3_grounding.png") -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for col, lev in zip(RAMP4, study["levels"]):
        ref = np.array([c["refused"] for c in lev["curve"]])
        una = np.array([c["unauthorised"] for c in lev["curve"]])
        keep = una >= 1e-4  # below this the curve rests on a handful of events
        ax1.plot(ref[keep], una[keep], color=col, lw=2.0, label=f"context read right {lev['accuracy']:.0%} of the time")
        ax1.scatter([lev["argmax"]["refused"]], [lev["argmax"]["unauthorised"]], marker="o", s=60, color=col,
                    edgecolor=INK, linewidth=0.8, zorder=4)
        ax1.scatter([lev["bayes_calibrated"]["refused"]], [lev["bayes_calibrated"]["unauthorised"]], marker="s",
                    s=56, color=col, edgecolor=INK, linewidth=0.8, zorder=4)
        ax1.scatter([lev["bayes_overconfident"]["refused"]], [lev["bayes_overconfident"]["unauthorised"]],
                    marker="^", s=66, color=col, edgecolor=INK, linewidth=0.8, zorder=4)
    ax1.set_yscale("log")
    ax1.set_xlabel("permitted requests refused")
    ax1.set_ylabel("unpermitted requests carried out")
    ax1.set_title("(a) The database is exact; the decision is not", loc="left")
    h, lab = ax1.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker=m, ls="none", color="#bdbcb6", markeredgecolor=INK, ms=7.5) for m in "os^"]
    lab += ["most likely context", "cost rule, t = 0.9", "cost rule, overconfident model"]
    ax1.legend(h, lab, fontsize=8.3, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax1.set_ylim(1e-4, 0.7)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    def _pct(v, _):
        dec = max(0, int(-np.floor(np.log10(v * 100) + 1e-9))) if v > 0 else 0
        return f"{100 * v:.{dec}f}%"

    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_pct))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    lev = [L for L in study["levels"] if abs(L["target_accuracy"] - 0.95) < 1e-9][0]
    src = list(lev["argmax_sources"].items())[:6][::-1]
    ax2.barh([k for k, _ in src], [v for _, v in src], color=BLUE, height=0.62)
    for y, (k, v) in enumerate(src):
        ax2.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=9, color=INK)
    ax2.set_xlim(0, max(v for _, v in src) * 1.25)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("share of unpermitted actions carried out")
    ax2.set_title(f"(b) Where they come from ({lev['accuracy']:.0%} accuracy)", loc="left")
    fig.tight_layout()
    out = FIG / path_out
    fig.savefig(out)
    plt.close(fig)
    return str(out)
