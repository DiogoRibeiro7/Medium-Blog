"""The four figures, in the house style of the other Medium-Blog companions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FLAT, HYP = ORANGE, BLUE
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 200,
})


def _poincare(Z: np.ndarray) -> np.ndarray:
    """Polar tangent coordinates (radius r = |z|) to the Poincare disk (radius tanh(r/2))."""
    r = np.linalg.norm(Z, axis=1, keepdims=True)
    return np.tanh(r / 2.0) * Z / np.maximum(r, 1e-12)


def _geodesic_segments(Z: np.ndarray, edges, k: int = 24) -> list[np.ndarray]:
    """Hyperbolic geodesics between embedded points, drawn in the Poincare disk: interpolate on
    the hyperboloid, x(t) = (sinh((1-t)D) x + sinh(tD) y) / sinh(D), then map to the disk."""
    r = np.linalg.norm(Z, axis=1)
    u = Z / np.maximum(r, 1e-12)[:, None]
    H = np.column_stack([np.cosh(r), np.sinh(r)[:, None] * u])  # points on the hyperboloid
    t = np.linspace(0.0, 1.0, k)[:, None]
    segs = []
    for i, j in edges:
        x, y = H[i], H[j]
        D = np.arccosh(max(x[0] * y[0] - x[1:] @ y[1:], 1.0))
        if D < 1e-9:
            continue
        pts = (np.sinh((1 - t) * D) * x + np.sinh(t * D) * y) / np.sinh(D)
        segs.append(pts[:, 1:] / (1.0 + pts[:, :1]))
    return segs


def figure_two_dimensions(g, root, fits, metrics, path_out="fig1_two_dimensions.png") -> str:
    depth = g.dist[root]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("depth", ["#86b6ef", "#2a78d6", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.6))
    for ax, geo in zip(axes, ("euclidean", "hyperbolic")):
        P = fits[geo].X  # hyperbolic: polar view, distance from the centre = hyperbolic radius
        segs = [[P[i], P[j]] for i, j in g.edges]
        ax.add_collection(LineCollection(segs, colors=INK2, linewidths=0.35, alpha=0.45, zorder=1))
        sc = ax.scatter(P[:, 0], P[:, 1], c=depth, cmap=cmap, s=7, linewidths=0, zorder=2)
        ax.scatter(*P[root], s=60, color=ORANGE, edgecolor=INK, linewidth=0.8, zorder=3)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        m = metrics[geo]
        pad = 0.05 * np.ptp(P, axis=0).max()
        ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
        ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
        if geo == "hyperbolic":
            rmax = float(np.linalg.norm(P, axis=1).max())
            for rr in range(2, int(rmax) + 1, 2):
                ax.add_patch(plt.Circle((0, 0), rr, fill=False, color=GRID, lw=0.8, zorder=0))
                ax.text(rr * 0.71, rr * 0.71, f"r={rr}", fontsize=7.5, color=INK2)
            ax.set_title(f"(b) hyperbolic plane, polar view\naverage distortion {m['D_avg']:.3f}, MAP {m['MAP']:.3f}", loc="left")
        else:
            ax.set_title(f"(a) flat plane\naverage distortion {m['D_avg']:.3f}, MAP {m['MAP']:.3f}", loc="left")
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("depth below the root (orange dot)")
    cb.outline.set_visible(False)
    out = FIG / path_out
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


SHORT = {"mammals": "WordNet mammals", "food": "WordNet food", "closure": "mammals, transitive closure",
         "tree": "complete tree", "grid": "grid"}


def _small_multiples(sweep, keys, dims, metric, ylabel, logy, path_out, title_fmt):
    fig, axes = plt.subplots(1, len(keys), figsize=(14.0, 3.9), sharey=True)
    for ax, k in zip(axes, keys):
        e = [sweep[k][d]["euclidean"][metric] for d in dims]
        h = [sweep[k][d]["hyperbolic"][metric] for d in dims]
        ax.plot(dims, e, color=FLAT, marker="o", ms=4.5, label="flat")
        ax.plot(dims, h, color=HYP, marker="o", ms=4.5, label="hyperbolic")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(dims)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_title(title_fmt(k), loc="left", fontsize=10)
        ax.set_xlabel("dimensions")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = FIG / path_out
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def figure_distortion(sweep, graphs, keys, dims, path_out="fig2_distortion_by_dimension.png") -> str:
    return _small_multiples(sweep, keys, dims, "D_avg", "average distortion (log scale)", True, path_out,
                            lambda k: SHORT[k])


def figure_ranking(sweep, graphs, keys, dims, path_out="fig3_ranking_by_dimension.png") -> str:
    return _small_multiples(sweep, keys, dims, "MAP", "neighbour-ranking MAP", False, path_out,
                            lambda k: SHORT[k])


def figure_dial(dial, fractions, path_out="fig4_cross_links.png") -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    xs = [f * 100 for f in fractions]
    xs_plot = [max(x, 0.2) for x in xs]  # 0% drawn at the left edge of a log axis
    for d, col, lab in ((5, "#86b6ef", "5 dimensions"), (50, "#0d366b", "50 dimensions")):
        mean, lo, hi = [], [], []
        for f in fractions:
            rs = [r[f"d{d}"]["euclidean"]["D_avg"] / r[f"d{d}"]["hyperbolic"]["D_avg"]
                  for r in dial if r["fraction"] == f and f"d{d}" in r]
            mean.append(np.mean(rs))
            lo.append(min(rs))
            hi.append(max(rs))
        ax1.plot(xs_plot, mean, color=col, marker="o", ms=5, label=lab)
        if d == 50:
            ax1.fill_between(xs_plot, lo, hi, color=col, alpha=0.15, lw=0)
    ax1.axhline(1.0, color=INK2, lw=1.0, ls=":")
    ax1.text(xs_plot[-1], 0.82, "no advantage", ha="right", va="top", fontsize=8.5, color=INK2)
    ax1.set_xscale("log")
    ax1.set_xticks(xs_plot)
    ax1.set_xticklabels(["0"] + [f"{x:g}" for x in xs[1:]])
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlabel("random cross-links added, % of nodes")
    ax1.set_ylabel("flat distortion / hyperbolic distortion")
    ax1.set_title("(a) A few cross-links erase the high-dimension advantage", loc="left")
    ax1.legend(loc="upper right", fontsize=9)

    for geo, col, lab in (("euclidean", FLAT, "flat"), ("hyperbolic", HYP, "hyperbolic")):
        m = [np.mean([r["d50"][geo]["MAP"] for r in dial if r["fraction"] == f]) for f in fractions]
        ax2.plot(xs_plot, m, color=col, marker="o", ms=5, label=lab)
    ax2.set_xscale("log")
    ax2.set_xticks(xs_plot)
    ax2.set_xticklabels(["0"] + [f"{x:g}" for x in xs[1:]])
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("random cross-links added, % of nodes")
    ax2.set_ylabel("neighbour-ranking MAP, 50 dimensions")
    ax2.set_title("(b) Ranking keeps a margin longer, then narrows", loc="left")
    ax2.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = FIG / path_out
    fig.savefig(out)
    plt.close(fig)
    return str(out)
