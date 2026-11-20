"""Figures of the article, from the files written by evaluate.py and details.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import success_core as sc  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab"]
LABEL = {
    "snaive": "Seasonal naive", "ma28": "4-week average", "dow4": "Weekday average", "ses": "Exp. smoothing",
    "lgb_l2": "GBM, squared", "lgb_l1": "GBM, absolute", "lgb_tweedie": "GBM, Tweedie",
    "lgbw_tweedie": "Weekly GBM, Tweedie", "lgbw_q80": "Weekly GBM, 0.8 quantile",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold", "axes.titlesize": 14,
    "axes.titlelocation": "left", "axes.titlecolor": INK,
})


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_ranks(metrics: pd.DataFrame, out: Path) -> None:
    specs = ["S0", "S1", "S2", "S3", "S4"]
    rows = {"S0": "Default: daily error", "S1": "Unit and timing: week B error", "S2": "Cost: 4-to-1 order cost",
            "S3": "Tolerance: quarters with >1 bad week", "S4": "Stability: revisions"}
    models = sc.ALL_MODELS
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for i, s in enumerate(specs):
        d = metrics[metrics["spec"] == s].set_index("model")["value"]
        vals = d.reindex(models).round(4)  # the two averages tie up to float rounding
        order = vals.dropna().rank(method="min").astype(int)
        for j, m in enumerate(models):
            if m not in order.index:
                ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color="#efeeea", lw=0))
                ax.text(j + 0.48, i + 0.46, "not built", ha="center", va="center", fontsize=9, color=INK2)
                continue
            r = order[m]
            col = SEQ[-1 - min(r - 1, len(SEQ) - 1)]
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=col, lw=0))
            ax.text(j + 0.48, i + 0.46, str(r), ha="center", va="center", fontsize=13,
                    color="white" if r <= 4 else INK, fontweight="semibold" if r == 1 else "normal")
            if r == 1:
                ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, fill=False, ec=INK, lw=2))
    ax.set_xlim(0, len(models))
    ax.set_ylim(len(specs), 0)
    ax.set_yticks([i + 0.46 for i in range(len(specs))], [rows[s] for s in specs])
    ax.set_xticks([j + 0.48 for j in range(len(models))], [LABEL[m] for m in models], rotation=30, ha="right")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Rank of each candidate under each definition of success (1 = best, outlined)")
    save(fig, out, "fig2_rank_by_definition.png")


def fig_bias(bias: pd.DataFrame, out: Path) -> None:
    models = [("lgb_l1", BLUE), ("lgb_tweedie", ORANGE), ("ma28", AQUA)]
    groups = list(dict.fromkeys(bias["zeros"]))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.26
    for k, (m, c) in enumerate(models):
        d = bias[bias["model"] == m].set_index("zeros").reindex(groups)
        x = np.arange(len(groups)) + (k - 1) * (width + 0.02)
        ax.bar(x, 100 * d["bias"], width, color=c, label=LABEL[m])
        for xi, v in zip(x, 100 * d["bias"]):
            ax.text(xi, v - 1.2 if v < 0 else v + 0.6, f"{v:+.0f}%", ha="center", va="top" if v < 0 else "bottom",
                    fontsize=10, color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(np.arange(len(groups)), [f"{g} zero days" for g in groups])
    ax.set_ylabel("Week B forecast minus sales, % of sales")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-70, 4)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Weekly totals of daily forecasts, by how intermittent the series is")
    save(fig, out, "fig1_weekly_bias_by_intermittency.png")


def fig_full_spec(metrics: pd.DataFrame, blends: pd.DataFrame, out: Path) -> None:
    piv = metrics[metrics["spec"].isin(["S2", "S3", "S4"])].pivot(index="model", columns="spec", values="value")
    inc = piv.loc[sc.INCUMBENT]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.axvspan(0, inc["S4"], color="#eef4fc", lw=0)
    ax.axvline(inc["S4"], color=INK2, lw=1, ls="--")
    ax.text(inc["S4"] + 0.001, 0.199, "left of the line: no more revision\nthan the 4-week average", ha="left", va="top",
            fontsize=10, color=INK2)
    ax.plot(blends["S4"], blends["S2"], color=ORANGE, lw=2, zorder=2)
    for _, b in blends.iterrows():
        if b["w"] in (0.2, 0.4, 0.6, 0.8):
            ax.plot(b["S4"], b["S2"], "o", color=ORANGE, ms=6, zorder=3)
            left = b["w"] <= 0.4
            ax.annotate(f"w = {b['w']:.1f}", (b["S4"], b["S2"]), xytext=(-8, -2) if left else (-4, -15),
                        textcoords="offset points", fontsize=9, color=INK2, ha="right" if left else "center")
    ax.text(0.1235, 0.1505, "orange: blends, w x weekly Tweedie + (1 - w) x 4-week average", fontsize=10, color=INK)
    offsets = {"ma28": (8, 4), "lgb_tweedie": (-8, -16), "lgbw_tweedie": (8, -14), "lgb_l2": (8, 2)}
    for m in sc.ALL_MODELS:
        ok = piv.loc[m, "S3"] <= inc["S3"]
        ax.plot(piv.loc[m, "S4"], piv.loc[m, "S2"], "o", ms=9, color=BLUE if ok else SURFACE,
                mec=BLUE, mew=2, zorder=4)
        if m == "dow4":
            continue
        if m == "lgb_tweedie":
            ax.annotate(LABEL[m], (piv.loc[m, "S4"], piv.loc[m, "S2"]), xytext=(0.1445, 0.1545), ha="center",
                        fontsize=10, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
            continue
        dx, dy = offsets.get(m, (8, 2))
        ax.annotate(LABEL[m] if m != "ma28" else "4-week average (= weekday average)", (piv.loc[m, "S4"], piv.loc[m, "S2"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=10, color=INK, ha="right" if dx < 0 else "left")
    ax.plot([], [], "o", ms=9, color=BLUE, label="no more quarters with >1 bad week than the 4-week average")
    ax.plot([], [], "o", ms=9, color=SURFACE, mec=BLUE, mew=2, label="more quarters with >1 bad week")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.set_xlim(0.11, 0.21)
    ax.set_ylim(0.14, 0.20)
    ax.set_xlabel("Revisions of the same week between runs, per unit sold (S4)")
    ax.set_ylabel("Cost of the week B order, per unit sold (S2)")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Cost against stability; seasonal naive (revisions 0.39) is off the chart")
    save(fig, out, "fig3_cost_against_stability.png")


def fig_allowance(frontier: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = 100 * frontier["allowance"].to_numpy()
    ax.step(x, frontier["S2"], where="post", color=BLUE, lw=2)
    changes = frontier[frontier["winner"] != frontier["winner"].shift()]
    for _, r in changes.iterrows():
        ax.plot(100 * r["allowance"], r["S2"], "o", color=BLUE, ms=7)
        ax.annotate(f"{LABEL[r['winner']]}\n{r['S2']:.3f}", (100 * r["allowance"], r["S2"]), xytext=(6, 6),
                    textcoords="offset points", fontsize=10, color=INK)
    ax.set_xlabel("Extra revision the planners accept, % above the 4-week average")
    ax.set_ylabel("Cost of the chosen model, per unit sold")
    ax.set_ylim(0.150, 0.190)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Which model the full specification picks, as the stability clause is loosened")
    save(fig, out, "fig4_stability_allowance.png")


def fig_tolerance(details: Path, out: Path) -> None:
    txt = details.read_text().splitlines()
    model, floor = [], []
    labels = []
    for line in txt:
        if line.strip().startswith("mean weekly sales") and "of series-quarters with more than one bad week" in line:
            labels.append(line.split(":")[0].replace("  mean weekly sales ", "").strip())
            model.append(float(line.split(",")[1].split()[0]))
        if line.strip().startswith("mean weekly sales") and "floor" in line:
            floor.append(float(line.split(",")[1].split()[0]))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, 100 * np.array(model), 0.38, color=BLUE, label="Weekly GBM, Tweedie (best on S3)")
    ax.bar(x + 0.2, 100 * np.array(floor), 0.38, color=ORANGE, label="Perfect mean forecast, Poisson sales")
    for xi, v in zip(x - 0.2, model):
        ax.text(xi, 100 * v + 1, f"{100 * v:.0f}%", ha="center", fontsize=10, color=INK)
    for xi, v in zip(x + 0.2, floor):
        ax.text(xi, 100 * v + 1, f"{100 * v:.0f}%", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x, [f"{lab.replace('-up', '+')}" for lab in labels])
    ax.set_xlabel("Mean weekly sales of the series (units)")
    ax.set_ylabel("Series-quarters with more than one bad week")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 88)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    ax.set_title("What the tolerance clause can ask for, by sales volume")
    save(fig, out, "fig5_tolerance_by_volume.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.results / "metrics.csv")
    fig_ranks(metrics, args.out)
    fig_bias(pd.read_csv(args.results / "bias_by_intermittency.csv"), args.out)
    fig_full_spec(metrics, pd.read_csv(args.results / "blends.csv"), args.out)
    fig_allowance(pd.read_csv(args.results / "stability_allowance.csv"), args.out)
    fig_tolerance(args.results / "details.txt", args.out)


if __name__ == "__main__":
    main()
