"""Figures for the article, from ``results/``.

    python figures.py

fig1  a lasting change and a temporary one look the same until the episode ends
fig2  how often the walk stops, and with what weight, after regimes and after episodes
fig3  gain of each scheme in each simulated scenario
fig4  rejection strength against persistence in the eight test series
fig5  gain of each scheme on each US series (pre-registered scoring)
fig6  mean gain against the weight given to the backward window
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import IN_SAMPLE, SHORT, TEST  # noqa: E402
from sims import simulate  # noqa: E402

RES, FIG = HERE / "results", HERE / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog"
SCHEME_STYLE = {
    "B": ("backward rule", RED, "o"),
    "EWP": ("evidence-weighted pool", VIOLET, "D"),
    "P0.75": ("fixed pool, 0.75", "#86b6ef", "s"),
    "P0.5": ("fixed pool, 0.5", BLUE, "s"),
    "P0.25": ("fixed pool, 0.25", "#184f95", "s"),
    "AveW": ("pool over windows", AQUA, "^"),
}
REGIMES = ["variance_up", "variance_down", "mean_shift", "persistence_up"]
EPISODES = ["burst", "level_episode", "crash"]
SCEN_LABEL = {"stationary": "No change", "variance_up": "Volatility rises", "variance_down": "Volatility falls",
              "mean_shift": "Level shifts", "persistence_up": "Persistence rises", "burst": "Volatility burst",
              "level_episode": "Level jump", "crash": "Crash and rebound"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})


def credit(fig):
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=7, color=INK2)


def fig1():
    n, at, ep = 721, 481, 24
    one = np.ones(n)
    regime_sd = np.where(np.arange(n) >= at, 2.0, 1.0)
    episode_sd = np.where((np.arange(n) >= at) & (np.arange(n) < at + ep), 2.0, 1.0)
    yr = simulate(0 * one, 0.5 * one, regime_sd, {}, np.random.default_rng(31))
    ye = simulate(0 * one, 0.5 * one, episode_sd, {}, np.random.default_rng(31))
    x = np.arange(n) - at
    sl = slice(at - 120, at + 120)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(x[sl], yr[sl], color=RED, lw=1.4, label="lasting change: volatility doubles for good")
    ax.plot(x[sl], ye[sl], color=BLUE, lw=1.4, label="temporary episode: volatility doubles for 24 periods")
    ax.axvspan(0, ep, color="#d8d7d2", lw=0, zorder=0)
    ax.axvline(18, color=INK, lw=1, ls=(0, (3, 2)))
    ax.annotate("a decision taken here\nsees identical data", xy=(18, ax.get_ylim()[0] * 0.75), xytext=(-55, ax.get_ylim()[0] * 0.7),
                fontsize=9, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.text(ep + 1, ax.get_ylim()[0] * 0.95, "the paths separate only\nafter the episode ends", fontsize=9, color=INK2,
            va="bottom")
    ax.set_xlabel("periods since the change")
    ax.set_ylabel("simulated series (same random draws)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("A regime and an episode are the same history until the episode ends", loc="left")
    credit(fig)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_regime_or_episode.png", dpi=200)
    plt.close(fig)


def fig2():
    w = pd.read_csv(RES / "sims_weights.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    cols = {"variance_up": RED, "variance_down": ORANGE, "mean_shift": YELLOW, "persistence_up": MAGENTA,
            "burst": BLUE, "level_episode": AQUA, "crash": VIOLET}
    for k, c in cols.items():
        d = w[w.scenario == k].sort_values("since")
        ls = "-" if k in REGIMES else (0, (4, 2))
        axes[0].plot(d.since, d.share_stopped, color=c, lw=2, ls=ls, label=SCEN_LABEL[k])
        axes[1].plot(d.since, d.median_weight_when_stopped, color=c, lw=2, ls=ls, label=SCEN_LABEL[k])
    axes[0].set_ylabel("share of selections where the walk stopped")
    axes[1].set_ylabel("median weight on the backward window\nwhen it stopped, Φ(z* − 2.498)")
    for ax in axes:
        ax.set_xlabel("periods since the change")
        ax.grid(color=GRID, lw=0.6)
        ax.axvspan(0, 24, color="#d8d7d2", lw=0, zorder=0)
    axes[1].axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=7,
               title="solid: lasting change; dashed: temporary episode", title_fontsize=8.5)
    axes[0].set_title("How often the backward walk rejects the past", loc="left")
    axes[1].set_title("How much weight the evidence asks for", loc="left")
    fig.suptitle("Simulated AR(1), 300 replications per scenario; grey band = the 24 periods of an episode",
                 x=0.01, ha="left", fontsize=9.5, color=INK2)
    credit(fig)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(FIG / "fig2_weights_after_change.png", dpi=200)
    plt.close(fig)


def fig3():
    df = pd.read_csv(RES / "sims.csv.gz")
    df = df[df.phase == "all"]
    piv = df.pivot_table(index=["scenario", "seed"], columns="scheme", values="nlpd")
    order = ["variance_up", "variance_down", "mean_shift", "persistence_up", "burst", "level_episode", "crash"]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for yi, sc in enumerate(order):
        p = piv.loc[sc]
        for k, (lab, col, mk) in SCHEME_STYLE.items():
            g = (p["E"] - p[k]).mean()
            ax.scatter(g, yi, color=col, marker=mk, s=70, edgecolor=SURFACE, lw=1.5, zorder=3,
                       label=lab if yi == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axhline(3.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SCEN_LABEL[s] for s in order])
    ax.invert_yaxis()
    ax.text(0.99, 0.02, "lasting regimes above the line, temporary episodes below", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("gain over the expanding window (nats per forecast, 240 forecasts after the change)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 0.07))
    ax.set_title("Simulations: who gains after a lasting change and who loses after a temporary one", loc="left")
    credit(fig)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_simulated_gains.png", dpi=200)
    plt.close(fig)


def fig4():
    t = pd.read_csv(RES / "premise_triples.csv")
    d = t[(t.series.isin(TEST)) & (t.block_len == 60)].dropna(subset=["R", "P"])
    real = {"UNRATE", "PAYEMS", "W875RX1", "DPCERA3M086SBEA", "HOUST"}
    fig, ax = plt.subplots(figsize=(11, 6.4))
    for grp, col, mk, lab in ((real, ORANGE, "o", "real activity: unemployment, payrolls, income, spending, housing"),
                              (set(TEST) - real, BLUE, "s", "prices, money and rates: PPI, M2, 10-year yield")):
        g = d[d.series.isin(grp)]
        ax.scatter(np.sign(g.R) * np.log1p(np.abs(g.R)), g.P.clip(-3, 3), color=col, marker=mk, s=55,
                   edgecolor=SURFACE, lw=1.2, label=lab, zorder=3)
    for r in d[d.R > 10].itertuples():
        if r.series in ("UNRATE", "DPCERA3M086SBEA"):
            continue
        ax.annotate(f"{SHORT[r.series]} {r.block_start[:4]}–{r.block_end[2:4]}",
                    (np.log1p(r.R), np.clip(r.P, -3, 3)), xytext=(-6, 8), textcoords="offset points", fontsize=8,
                    ha="right", color=INK2)
    ax.annotate("unemployment and real spending 2016–21", (np.log1p(15.5), -1.37), xytext=(np.log1p(4.2), -2.1),
                fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.text(np.log1p(2.7), -2.95, "values below −3 drawn at −3", fontsize=8, color=INK2, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(np.log1p(2.498), color=INK, lw=0.9, ls=(0, (3, 2)))
    ticks = [-3, -1, 0, 1, 2.5, 5, 10, 20, 40]
    ax.set_xticks([np.sign(v) * np.log1p(abs(v)) for v in ticks])
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_xlabel("rejection strength R: z statistic of block k's model on block k − 1 (log scale)")
    ax.set_ylabel("persistence P (nats per month, clipped at ±3)\n> 0: the change lasted into block k + 1")
    ax.text(np.log1p(2.498) + 0.04, 2.85, "rule's threshold", fontsize=8.5, color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Eight new US series, five-year blocks: a stronger rejection does not predict a lasting change", loc="left")
    credit(fig)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_strength_vs_persistence.png", dpi=200)
    plt.close(fig)


def fig5():
    g = pd.read_csv(RES / "forward_gains.csv")
    g = g[(g.errors == "gaussian") & (g.scoring == "no_covid")]
    ids = [*TEST, *IN_SAMPLE]
    fig, ax = plt.subplots(figsize=(11, 7.2))
    for yi, sid in enumerate(ids):
        r = g[g.series == sid].set_index("scheme")
        for k, (lab, col, mk) in SCHEME_STYLE.items():
            v = r.loc[k, "gain"]
            ax.scatter(np.clip(v, -0.2, 0.25), yi, color=col, marker=mk, s=70, edgecolor=SURFACE, lw=1.5, zorder=3,
                       label=lab if yi == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axhline(len(TEST) - 0.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([SHORT[s] + (" (in-sample)" if s in IN_SAMPLE else "") for s in ids])
    ax.invert_yaxis()
    ax.set_xlabel("gain over the expanding window (nats per month, 1985–2026, March–December 2020 not scored)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("The pre-registered test: AR(3), Gaussian errors, five-year blocks", loc="left")
    credit(fig)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_us_gains.png", dpi=200)
    plt.close(fig)


def fig6():
    g = pd.read_csv(RES / "forward_gains.csv")
    g = g[(g.errors == "gaussian") & (g.scoring == "no_covid") & (g.series.isin(TEST))]
    real_mean = {k: g[g.scheme == k].gain.mean() for k in ["E", "P0.25", "P0.5", "P0.75", "B", "EWP"]}
    w = pd.read_csv(RES / "forward_weights.csv")
    w_ewp = w[w.stopped & w.series.isin(TEST)].weight.mean()
    s = pd.read_csv(RES / "sims.csv.gz")
    s = s[s.phase == "all"]
    piv = s.pivot_table(index=["scenario", "seed"], columns="scheme", values="nlpd")

    def sim_mean(scen, k):
        return np.mean([(piv.loc[x]["E"] - piv.loc[x][k]).mean() for x in scen])

    xs = [0, 0.25, 0.5, 0.75, 1.0]
    keys = ["E", "P0.25", "P0.5", "P0.75", "B"]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for lab, vals, col, ls in (
        ("eight US test series (pre-registered scoring)", [real_mean[k] for k in keys], BLUE, "-"),
        ("simulations, lasting regimes", [sim_mean(REGIMES, k) for k in keys], RED, (0, (4, 2))),
        ("simulations, temporary episodes", [sim_mean(EPISODES, k) for k in keys], ORANGE, (0, (1, 1.5))),
    ):
        ax.plot(xs, vals, color=col, lw=2, ls=ls, marker="o", ms=7, label=lab)
    ax.scatter([w_ewp], [real_mean["EWP"]], color=VIOLET, marker="D", s=90, zorder=4, edgecolor=SURFACE, lw=1.5,
               label=f"evidence-weighted pool on the US series (mean weight {w_ewp:.2f} when the walk stopped)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0\nexpanding", "0.25", "0.5", "0.75", "1\nbackward rule"])
    ax.set_xlabel("fixed weight on the backward window whenever the walk stops")
    ax.set_ylabel("mean gain over the expanding window (nats per forecast)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("How much to trust the backward window: the answer depends on the world, not on the evidence", loc="left")
    credit(fig)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_weight_curve.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print(f.__name__, "done")


if __name__ == "__main__":
    main()
