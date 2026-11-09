"""Figures for the article, from the files in ``results/``.

    python figures.py

fig1  the matrix and which cells the usual backtest uses (schematic)
fig2  simulated matrices for five kinds of change (mean excess log loss)
fig3  volatility falls: the same transfers under three scores
fig4  US industrial production growth and CPI inflation, 1961-2026
fig5  today's model on past blocks: excess log loss and 90% coverage
fig6  forward cost of each window scheme: no change versus one break
fig7  the backward rule in real time: inflation scores and production windows
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
DIVERGING = LinearSegmentedColormap.from_list(
    "bluegrayred", ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f4bcbb", "#e34948", "#8f1d1d"])
SEQ = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "font.family": "DejaVu Sans",
})


def credit(fig: plt.Figure) -> None:
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=7, color=INK2)


def heat(ax, M, norm, cmap=DIVERGING, labels=None, title="", every=1, clip_note=None):
    K = M.shape[0]
    im = ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest")
    for i in range(K):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK2, lw=0.6, hatch="////", alpha=0.35))
    ax.set_title(title, loc="left")
    if labels is None:
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([str(i + 1) for i in range(K)], fontsize=8)
        ax.set_yticklabels([str(i + 1) for i in range(K)], fontsize=8)
    else:
        idx = list(range(0, K, every))
        ax.set_xticks(idx)
        ax.set_yticks(idx)
        ax.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=8)
        ax.set_yticklabels([labels[i] for i in idx], fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


# --------------------------------------------------------------------------- fig1


def fig1() -> None:
    K = 8
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    for i in range(K):
        for j in range(K):
            fc = "#d8d7d2" if i == j else ("#cde2fb" if j > i else "#f7d3d2")
            ax.add_patch(Rectangle((j, K - 1 - i), 1, 1, fc=fc, ec=SURFACE, lw=2))
            if i == j:
                ax.add_patch(Rectangle((j, K - 1 - i), 1, 1, fill=False, ec=INK2, lw=0, hatch="////", alpha=0.4))
    for i in range(K - 1):
        ax.add_patch(Rectangle((i + 1, K - 1 - i), 1, 1, fc="none", ec=BLUE, lw=2.4))
    ax.add_patch(Rectangle((0.04, 0.04), K - 1 - 0.08, 0.92, fc="none", ec=RED, lw=2.6))
    ax.text(6.0, 7.0, "FORWARD\ntest after train", ha="center", va="center", fontsize=10, color="#1c5cab",
            fontweight="bold")
    ax.text(2.0, 3.0, "BACKWARD\ntest before train", ha="center", va="center", fontsize=10, color="#b8302f",
            fontweight="bold")
    ax.text(3.5, 0.5, "last row: today's model on every past block", ha="center", va="center", fontsize=9,
            color="#8f1d1d")
    ax.annotate("one step ahead: the cells\na rolling backtest uses", xy=(3.6, 5.5), xytext=(6.4, 5.45),
                ha="center", fontsize=9, color="#1c5cab",
                arrowprops=dict(arrowstyle="-", color="#1c5cab", lw=0.9))
    ax.set_xlim(0, K)
    ax.set_ylim(0, K)
    ax.set_xticks(np.arange(K) + 0.5)
    ax.set_xticklabels([f"{j + 1}" for j in range(K)])
    ax.set_yticks(np.arange(K) + 0.5)
    ax.set_yticklabels([f"{K - i}" for i in range(K)])
    ax.set_xlabel("test block (time runs left to right)")
    ax.set_ylabel("training block (time runs top to bottom)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal")
    ax.set_title("Train on block i, score on block j", loc="left")
    fig.text(0.02, 0.035, "Hatched diagonal: the block's own model, cross-validated inside the block. It is the "
             "benchmark\nfor its column, so every cell is read as \"how much worse than a model of that period\".",
             fontsize=8.5, color=INK2, ha="left")
    credit(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig1_matrix_schematic.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- fig2


def fig2() -> None:
    d = np.load(RES / "matrix_sims.npz")
    names = [("stationary", "No change"), ("mean_shift", "Mean shifts up"),
             ("variance_down", "Volatility halves"), ("persistence_up", "Persistence rises"),
             ("drift", "Persistence drifts")]
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.9))
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=1.0)
    for ax, (k, lab) in zip(axes, names):
        im = heat(ax, d[f"{k}__nlpd__mean"], norm, title=lab)
        if k not in ("stationary", "drift"):
            ax.axhline(4.5, color=INK, lw=0.8, ls=(0, (3, 2)))
            ax.axvline(4.5, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.set_xlabel("test block")
    axes[0].set_ylabel("training block")
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("excess log loss (nats per forecast)")
    cb.outline.set_visible(False)
    fig.suptitle("Simulated AR(1), ten blocks of 60 observations, 500 replications; dashed line = break",
                 x=0.01, ha="left", fontsize=10, color=INK2, y=0.995)
    credit(fig)
    fig.savefig(FIG / "fig2_simulated_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- fig3


def fig3() -> None:
    d = np.load(RES / "matrix_sims.npz")
    k = "variance_down"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    panels = [
        (d[f"{k}__nlpd__mean"], TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=1.0), DIVERGING,
         "Log loss: excess over native", "nats per forecast"),
        (d[f"{k}__sqerr__mean"], TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=1.6), DIVERGING,
         "Squared error: ratio to native", "ratio"),
        (d[f"{k}__cover90__mean"], TwoSlopeNorm(vmin=0.5, vcenter=0.9, vmax=1.0), DIVERGING.reversed(),
         "Share inside the 90% interval", "coverage (nominal 0.90)"),
    ]
    for ax, (M, norm, cmap, title, cl) in zip(axes, panels):
        im = heat(ax, M, norm, cmap=cmap, title=title)
        ax.axhline(4.5, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.axvline(4.5, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.set_xlabel("test block")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label(cl, fontsize=8)
        cb.outline.set_visible(False)
    axes[0].set_ylabel("training block")
    fig.suptitle("Volatility halves after block 5 (innovation s.d. 2 → 1). Lower-left: calm models on the "
                 "turbulent past; upper-right: turbulent models on the calm present",
                 x=0.01, ha="left", fontsize=9.5, color=INK2)
    credit(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(FIG / "fig3_three_scores.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- fig4


def fig4() -> None:
    m = pd.read_csv(RES / "application_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=3.0)
    titles = {"ip_growth": "Industrial production growth", "cpi_inflation": "CPI inflation"}
    for ax, (name, g) in zip(axes, m.groupby("series", sort=False)):
        K = int(g.train_block.max()) + 1
        M = np.full((K, K), np.nan)
        for r in g.itertuples():
            M[r.train_block, r.test_block] = r.excess_nlpd
        labels = [f"{s[:4]}–{e[2:4]}" for s, e in
                  g.drop_duplicates("test_block").sort_values("test_block")[["test_start", "test_end"]].to_numpy()]
        im = heat(ax, np.clip(M, -0.5, 3.0), norm, labels=labels, title=titles[name])
        big = np.argwhere(M > 3.0)
        for i, j in big:
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white")
        ax.set_xlabel("test block")
    axes[0].set_ylabel("training block")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, extend="max")
    cb.set_label("excess log loss, nats per month (above 3: value printed)")
    cb.outline.set_visible(False)
    fig.suptitle("United States, monthly, AR(3) with Gaussian errors fitted on five-year blocks, 12-month gap",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    credit(fig)
    fig.savefig(FIG / "fig4_us_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- fig5


def fig5() -> None:
    g = pd.read_csv(RES / "application_matrix.csv")
    t = pd.read_csv(RES / "application_matrix_t.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), sharex="col")
    titles = {"ip_growth": "Industrial production growth", "cpi_inflation": "CPI inflation"}
    for c, name in enumerate(["ip_growth", "cpi_inflation"]):
        gg = g[g.series == name]
        K = int(gg.train_block.max()) + 1
        last_g = gg[(gg.train_block == K - 1) & (gg.test_block < K - 1)].sort_values("test_block")
        tt = t[t.series == name]
        last_t = tt[(tt.train_block == K - 1) & (tt.test_block < K - 1)].sort_values("test_block")
        x = np.arange(K - 1)
        labs = [f"{s[:4]}–{e[2:4]}" for s, e in last_g[["test_start", "test_end"]].to_numpy()]
        ax = axes[0, c]
        yg = last_g.excess_nlpd.to_numpy()
        ax.plot(x, np.clip(yg, -0.5, 3.0), color=BLUE, lw=2, marker="o", ms=7, label="Gaussian errors", zorder=3)
        ax.plot(x, last_t.excess_nlpd.to_numpy(), color=ORANGE, lw=2, marker="s", ms=7, label="Student-t(5) errors", zorder=3)
        for xi, v in zip(x, yg):
            if v > 3.0:
                ax.annotate(f"{v:.1f} (off scale)", xy=(xi, 3.0), xytext=(xi - 3.4, 2.2), fontsize=8, color=INK2,
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylim(-0.5, 3.1)
        ax.set_title(f"{titles[name]}: model of {last_g.train_start.iloc[0][:4]}–{last_g.train_end.iloc[0][2:4]} on the past",
                     loc="left")
        ax.set_ylabel("excess log loss (nats per month)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(frameon=False, loc="upper center", fontsize=8.5)
        ax2 = axes[1, c]
        ax2.plot(x, last_g.cover90.to_numpy(), color=BLUE, lw=2, marker="o", ms=7, label="Gaussian errors")
        ax2.plot(x, last_t.cover90.to_numpy(), color=ORANGE, lw=2, marker="s", ms=7, label="Student-t(5) errors")
        ax2.axhline(0.9, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax2.text(x[0] - 0.2, 0.905, "nominal 0.90", va="bottom", fontsize=8, color=INK2)
        ax2.set_ylim(0.4, 1.02)
        ax2.set_ylabel("share inside the 90% interval")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labs, rotation=90, fontsize=8)
        ax2.grid(axis="y", color=GRID, lw=0.6)
        ax2.set_xlabel("test block (oldest on the left)")
    fig.suptitle("The last row of the matrix: the most recent five-year model scored on every earlier block",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    credit(fig)
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    fig.savefig(FIG / "fig5_last_row.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- fig6


def fig6() -> None:
    df = pd.read_csv(RES / "rule_sims.csv.gz")
    df = df[df.phase == "all"]
    breaks = ["mean_shift", "variance_down", "variance_up", "persistence_up"]
    piv = df.pivot_table(index=["scenario", "seed"], columns="scheme", values="nlpd")
    rows = {}
    for scheme in piv.columns:
        if scheme == "oracle":
            continue
        stat = (piv.loc["stationary"][scheme] - piv.loc["stationary"]["expanding"]).mean()
        brk = np.mean([(piv.loc[s][scheme] - piv.loc[s]["oracle"]).mean() for s in breaks])
        rows[scheme] = (stat, brk)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    dial = [("backward_z1", 1.0), ("backward_z1.645", 1.645), ("backward_z2", 2.0), ("backward", 2.5), ("backward_z3", 3.0)]
    xs = [rows[k][0] for k, _ in dial]
    ys = [rows[k][1] for k, _ in dial]
    ax.plot(xs, ys, color=RED, lw=2, zorder=2)
    for (k, zc), x_, y_ in zip(dial, xs, ys):
        ax.scatter([x_], [y_], s=70, color=RED, edgecolor=SURFACE, lw=2, zorder=3)
        ax.annotate(f"z > {zc:g}", (x_, y_), xytext=(8, -3), textcoords="offset points", fontsize=8.5, color="#b8302f")
    others = {"expanding": ("expanding", INK2), "rolling_60": ("rolling 60", BLUE), "rolling_120": ("rolling 120", BLUE),
              "rolling_240": ("rolling 240", BLUE), "rolling_360": ("rolling 360", BLUE), "avew": ("pool over windows", AQUA),
              "forward_cv": ("forward validation", VIOLET), "roc": ("reverse CUSUM of squares", YELLOW),
              "backward_b120": ("backward, 120-row blocks", MAGENTA)}
    offsets = {"forward_cv": (-10, -16), "rolling_240": (8, -4), "rolling_120": (8, 6),
               "backward_b120": (8, 4)}
    for k, (lab, col) in others.items():
        x_, y_ = rows[k]
        ax.scatter([x_], [y_], s=70, color=col, edgecolor=SURFACE, lw=2, zorder=3, marker="D" if k.startswith("rolling") else "o")
        if k == "avew":
            ax.annotate(lab, (x_, y_), xytext=(0.0055, 0.104), textcoords="data", fontsize=8.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
            continue
        dx, dy = offsets.get(k, (8, 4))
        ax.annotate(lab, (x_, y_), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK,
                    ha="right" if dx < 0 else "left")
    ax.text(0.98, 0.97, "red line: backward rule\n(60-row blocks), one point\nper threshold", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color="#b8302f")
    ax.set_xlabel("cost when nothing changes (nats per forecast over expanding window)")
    ax.set_ylabel("cost after one break (nats per forecast over the known-break window,\naverage of four kinds of break)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_title("What each window scheme pays when nothing changes and when something does", loc="left")
    credit(fig)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_window_frontier.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- fig7


def fig7() -> None:
    f = pd.read_csv(RES / "application_forward.csv", parse_dates=["date"])
    tip = pd.read_csv(RES / "application_forward_t_ip_growth.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2))
    ax = axes[0]
    c = f[f.series == "cpi_inflation"].pivot(index="date", columns="scheme", values="nlpd")
    for k, lab, col in [("backward", "backward rule (z > 2.5)", RED), ("avew", "pool over windows", AQUA),
                        ("forward_cv", "forward validation", VIOLET), ("rolling_120", "rolling 120 months", BLUE)]:
        ax.plot(c.index, (c["expanding"] - c[k]).cumsum(), color=col, lw=2, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("cumulative log-score gain\nover the expanding window (nats)")
    ax.set_title("CPI inflation, one month ahead, Gaussian AR(3): cumulative gain over the expanding window", loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax = axes[1]
    w = f[(f.series == "ip_growth") & (f.scheme == "backward")].set_index("date")["window"] / 12
    wt = tip[tip.scheme == "backward"].set_index("date")["window"] / 12
    ax.step(w.index, w, where="post", color=BLUE, lw=2, label="Gaussian errors")
    ax.step(wt.index, wt, where="post", color=ORANGE, lw=2, label="Student-t(5) errors")
    for d0, d1, lab in [("2007-12-01", "2009-06-01", "2008–09 recession"), ("2020-02-01", "2020-04-01", "COVID-19")]:
        ax.axvspan(pd.Timestamp(d0), pd.Timestamp(d1), color="#d8d7d2", lw=0)
        ax.text(pd.Timestamp(d0), 63, lab, fontsize=8, color=INK2, ha="right", va="top")
    ax.set_ylabel("training window chosen\nby the backward rule (years)")
    ax.set_title("Industrial production: window chosen by the backward rule (z > 2.5), re-chosen every January",
                 loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.set_ylim(0, 70)
    ax.grid(axis="y", color=GRID, lw=0.6)
    credit(fig)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_rule_in_real_time.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
        print(f"{f.__name__} done")


if __name__ == "__main__":
    main()
