"""Draw the article's figures from the JSON files in results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import theory  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
COLOR = {
    "reference (real)": INK2,
    "training copy": INK,
    "independent": "#2a78d6",
    "copula": "#eb6834",
    "cart": "#1baf7a",
    "smote": "#eda100",
    "ctgan": "#e87ba4",
    "tvae": "#6250d6",
}
LABEL = {
    "reference (real)": "independent real sample",
    "training copy": "training data itself",
    "independent": "independent marginals",
    "copula": "Gaussian copula",
    "cart": "sequential CART",
    "smote": "SMOTE interpolation",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
}
MARKER = {"reference (real)": "o", "training copy": "X"}
SOURCE = "Source: UCI Adult (PMLB copy); analysis and figure by Diogo Ribeiro"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def _save(fig, name: str, source: str = SOURCE) -> None:
    fig.text(0.01, 0.005, source, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _point(ax, x, y, name, dx=6, dy=4, ha="left"):
    ax.scatter(
        [x], [y], s=70, marker=MARKER.get(name, "o"), color=COLOR[name],
        edgecolor=SURFACE, linewidth=1.5, zorder=3,
    )
    ax.annotate(
        LABEL[name], (x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=8.5, color=INK, ha=ha, va="center",
    )


def fig_axes() -> None:
    a = json.loads((RES / "axes.json").read_text())
    g = a["generators"]
    offsets = {  # label placement (dx, dy, ha) per panel, tuned to avoid overlaps
        "a": {"reference (real)": (6, -12, "left"), "training copy": (0, 11, "left"),
              "independent": (-8, 0, "right"), "copula": (-8, 0, "right"),
              "cart": (6, -11, "left"), "smote": (8, 0, "left"),
              "ctgan": (8, 0, "left"), "tvae": (8, 0, "left")},
        "b": {"reference (real)": (-6, -13, "right"), "training copy": (-8, 0, "right"),
              "independent": (8, 0, "left"), "copula": (0, -13, "center"),
              "cart": (-8, 0, "right"), "smote": (-8, 0, "right"),
              "ctgan": (0, -13, "center"), "tvae": (8, 2, "left")},
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # panel a: square-root x scale spreads the generators close to the data
    y_lo, y_hi = 0.80, 0.94
    for name, r in g.items():
        dx, dy, ha = offsets["a"].get(name, (8, 0, "left"))
        x, y = r["pairwise_tvd_mean"], r["tstr"]["auc"]
        if y < y_lo:  # off the zoomed scale: an arrow on the bottom edge, with the value
            ax1.annotate("", xy=(x, y_lo + 0.002), xytext=(x, y_lo + 0.014),
                         arrowprops=dict(arrowstyle="-|>", color=COLOR[name], lw=1.5))
            ax1.annotate(f"{LABEL[name]} (AUC {y:.2f})", (x, y_lo + 0.014), xytext=(0, 6),
                         textcoords="offset points", fontsize=8.5, color=INK, ha="center")
            continue
        _point(ax1, x, y, name, dx, dy, ha)
    xmax = max(r["pairwise_tvd_mean"] for r in g.values()) * 1.3
    pad = 0.0004  # keeps the marker at zero clear of the axis
    ax1.set_xscale("function", functions=(lambda v: np.sqrt(np.clip(v + pad, 0, None)),
                                          lambda u: np.square(u) - pad))
    ax1.set_xlim(-pad, xmax)
    ax1.set_xticks([0, 0.01, 0.02, 0.05, 0.1, 0.2])
    ax1.set_xticklabels(["0", "0.01", "0.02", "0.05", "0.10", "0.20"])
    ax1.axhline(a["trtr"]["auc"], color=MUTED, lw=1, ls="--")
    ax1.text(xmax * 0.97, a["trtr"]["auc"] + 0.003, "model trained on the real data",
             fontsize=8, color=INK2, ha="right", va="bottom")
    ax1.set_xlabel("Mean two-way table distance to the training data (square-root scale)")
    ax1.set_ylabel("AUC on real holdout, model trained on synthetic")
    ax1.set_title("a. Fidelity against predictive utility")
    ax1.set_ylim(y_lo, y_hi)

    # panel b: zoom on the useful generators; independent marginals marked on the left edge
    x_lo, x_hi = 0.83, 0.94
    for name, r in g.items():
        dx, dy, ha = offsets["b"].get(name, (8, 0, "left"))
        x, y = r["tstr"]["auc"], max(r["mia_tpr_at_1pct"], 1e-3)
        if x < x_lo:
            ax2.annotate("", xy=(x_lo + 0.001, y), xytext=(x_lo + 0.009, y),
                         arrowprops=dict(arrowstyle="-|>", color=COLOR[name], lw=1.5))
            ax2.annotate(f"{LABEL[name]}\n(AUC {x:.2f})", (x_lo + 0.001, y), xytext=(0, 10),
                         textcoords="offset points", fontsize=8.5, color=INK, ha="left", va="bottom")
            continue
        _point(ax2, x, y, name, dx, dy, ha)
    ax2.axhline(0.01, color=MUTED, lw=1, ls=":")
    ax2.text(x_hi - 0.001, 0.0112, "chance: 1%", fontsize=8, color=INK2, ha="right", va="bottom")
    ax2.set_yscale("log")
    ax2.set_ylim(0.004, 1.6)
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_xlabel("AUC on real holdout, model trained on synthetic")
    ax2.set_ylabel("Members identified at a 1% false-positive rate")
    ax2.set_title("b. Predictive utility against membership leakage")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig3_three_axes")


def fig_dcr() -> None:
    a = json.loads((RES / "axes.json").read_text())
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    order = [n for n in COLOR if n in a["generators"] and n != "training copy"]
    for name in order:
        q = np.array(a["generators"][name]["dcr_quantiles"])
        p = np.linspace(0, 1, len(q))
        lw = 3 if name == "reference (real)" else 2
        ax.plot(q, p, color=COLOR[name], lw=lw, label=LABEL[name], ls="-" if name != "reference (real)" else "--")
    ax.set_xlim(-0.02, 1.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Distance from a synthetic record to the closest training record")
    ax.set_ylabel("Share of synthetic records within that distance")
    ax.set_title("Distance to the closest training record (DCR)")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig2_dcr")


def fig_coefficients() -> None:
    a = json.loads((RES / "axes.json").read_text())
    terms = a["terms"][1:]
    names = [n for n in COLOR if n in a["generators"] and n != "training copy"]
    pretty = {"age": "age (per year)", "education-num": "education (per grade)",
              "hours-per-week": "hours worked (per hour)", "male": "male", "married": "married"}
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    band = 1.959963984540054 * np.sqrt(2)
    ax.axvspan(-band, band, color=GRID, alpha=0.8, lw=0)
    ax.axvline(0, color=MUTED, lw=1)
    n = len(names)
    for i, t in enumerate(terms):
        for j, name in enumerate(names):
            z = a["generators"][name]["coef_diff_in_real_se"][i + 1]
            y = i + (j - (n - 1) / 2) * 0.11
            ax.scatter([z], [y], s=34, color=COLOR[name], marker=MARKER.get(name, "o"),
                       edgecolor=SURFACE, linewidth=1, zorder=3, label=LABEL[name] if i == 0 else None)
    ax.set_xscale("symlog", linthresh=3)
    ax.set_xticks([-50, -20, -10, -3, 0, 3, 10, 20, 50])
    ax.set_xticklabels(["-50", "-20", "-10", "-3", "0", "3", "10", "20", "50"])
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([pretty[t] for t in terms])
    ax.invert_yaxis()
    ax.set_xlabel("Synthetic minus real coefficient, in standard errors of the real estimate (symmetric log scale)")
    ax.set_title("Logistic regression of income > $50K: synthetic estimates against the real one", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color=GRID, label="spread between two real samples"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig1_coefficients")


def fig_mechanism() -> None:
    d = json.loads((RES / "coverage_mechanism.json").read_text())
    rows = {tuple(r["key"]): r for r in d["rows"]}

    def cov(key):
        return float(np.mean(rows[key]["coverage"][1:]))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    ks = np.array([1, 2, 5, 10, 20])
    kk = np.linspace(1, 20, 200)
    ax = axes[0]
    for mode, col in (("simple", "#2a78d6"), ("proper", "#eb6834")):
        ax.plot(kk, [theory.size_coverage(k, mode == "proper") for k in kk], color=col, lw=2,
                label=f"parametric, {mode}: theory")
        ax.scatter(ks, [cov(("size", "parametric", mode, k)) for k in ks], color=col, s=40,
                   edgecolor=SURFACE, zorder=3, label=f"parametric, {mode}: simulation")
    ax.scatter(ks, [cov(("size", "cart", "simple", k)) for k in ks], color="#1baf7a", marker="D", s=34,
               edgecolor=SURFACE, zorder=3, label="sequential CART, simple: simulation")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Synthetic rows per real row (one data set, analysed as real)")
    ax.set_ylabel("Coverage of nominal 95% intervals")
    ax.set_title("a. More synthetic rows, worse intervals")
    ax.legend(fontsize=7.5, loc="lower left")

    ms = np.array([1, 2, 5, 10, 20])
    mm = np.arange(1, 21)
    rule_col = {"naive": "#2a78d6", "simple": "#1baf7a", "proper": "#eb6834", "raghunathan": "#6250d6"}
    rule_lab = {"naive": "as if real", "simple": "simple-synthesis rule",
                "proper": "proper-synthesis rule", "raghunathan": "Raghunathan-Reiter-Rubin"}
    for ax, mode, title in ((axes[1], "simple", "b. Simple synthesis"), (axes[2], "proper", "c. Proper synthesis")):
        for rule in ("naive", "simple", "proper", "raghunathan"):
            m_ok = ms if rule != "raghunathan" else ms[1:]
            pts = [cov(("rules", "parametric", mode, rule, m)) for m in m_ok]
            ax.scatter(m_ok, pts, color=rule_col[rule], s=40, edgecolor=SURFACE, zorder=3, label=rule_lab[rule])
            if rule != "raghunathan":
                ax.plot(mm, [theory.coverage(rule, 1.0, m, mode == "proper") for m in mm],
                        color=rule_col[rule], lw=1.5, alpha=0.8)
            else:
                ax.plot(m_ok, pts, color=rule_col[rule], lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(ms)
        ax.set_xticklabels([str(m) for m in ms])
        ax.set_xlabel("Number of synthetic data sets m")
        ax.set_title(title)
    axes[1].legend(fontsize=7.5, loc="lower left")
    for ax in axes:
        ax.axhline(0.95, color=MUTED, lw=1, ls="--")
        ax.set_ylim(0, 1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig4_coverage_mechanism",
          source="Simulation: known mechanism, n = 1,000, 2,000 replications; lines are closed forms. Figure by Diogo Ribeiro")


def fig_adult() -> None:
    d = json.loads((RES / "coverage_adult.json").read_text())
    rows = {tuple(r["key"]): r for r in d["rows"]}
    terms = d["terms"][1:]
    pretty = ["age", "education", "hours", "male", "married"]
    gens = ["copula", "cart", "smote", "independent"]
    row_keys = [("real",)]
    row_labels = ["real sample"]
    for g in gens:
        for mode in ("simple", "proper"):
            row_keys.append((g, mode))
            row_labels.append(f"{LABEL[g]}, {mode}")
    panels = [("naive", 1, "a. One data set, analysed as real"),
              ("matched", d["m"], f"b. m = {d['m']}, rule matched to the synthesis")]
    cmap = LinearSegmentedColormap.from_list("cov", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    for ax, (rule, m, title) in zip(axes, panels):
        mat = np.zeros((len(row_keys), len(terms)))
        for i, key in enumerate(row_keys):
            if key == ("real",):
                mat[i] = rows[key]["coverage"][1:]
            else:
                r = key[1] if rule == "matched" else rule
                mat[i] = rows[(key[0], key[1], r, m)]["coverage"][1:]
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color=SURFACE if v > 0.55 else INK)
        ax.set_xticks(range(len(terms)))
        ax.set_xticklabels(pretty)
        ax.set_yticks(range(len(row_keys)))
        ax.set_yticklabels(row_labels)
        ax.set_title(title)
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(0.5, color=SURFACE, lw=3)
    fig.suptitle("Coverage of nominal 95% intervals for the population coefficients", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.985), h_pad=0.2)
    _save(fig, "fig5_coverage_adult",
          source=f"Adult as population ({d['population']:,} rows); samples of {d['n']:,}; "
                 f"{d['reps']:,} replications. Figure by Diogo Ribeiro")


def main() -> None:
    fig_coefficients()
    fig_dcr()
    fig_axes()
    fig_mechanism()
    fig_adult()


if __name__ == "__main__":
    main()
