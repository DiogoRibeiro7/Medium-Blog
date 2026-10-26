"""Draw the article's figures from the results of the experiments."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

from common import FIGURES, RESULTS, load  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BAND = "#e9e8e3"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ESTIMATORS = ["lag", "lag_minus_lead", "controlled", "joint"]
EST_LABEL = {"lag": "lag slope (earlier article)",
             "lag_minus_lead": "lag minus lead",
             "controlled": "controlled slope",
             "joint": "joint with clustering"}
EST_COLOR = dict(zip(ESTIMATORS, SLOT))
EST_MARKER = dict(zip(ESTIMATORS, ["o", "s", "^", "D"]))
CREDIT = ("Diogo Ribeiro · data: Technion SEE Center, Anonymous Bank (1999) "
          "· code: github.com/DiogoRibeiro7/Medium-Blog")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11.5, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False,
    "legend.fontsize": 9.5,
})


def label_ends(ax, items, gap: float = 0.055) -> None:
    """Direct labels at line ends with a colour key, spread apart."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    order = sorted(items, key=lambda it: it[1])
    pos = [it[1] for it in order]
    for _ in range(300):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap * span:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = mid - gap * span / 2, mid + gap * span / 2
                moved = True
        if not moved:
            break
    key = offset_copy(ax.transData, fig=ax.figure, x=11, y=0, units="points")
    txt = offset_copy(ax.transData, fig=ax.figure, x=18, y=0, units="points")
    for (x, y, text, color), yy in zip(order, pos):
        ax.plot([x], [yy], marker="s", ms=6, color=color, mec=color,
                transform=key, clip_on=False, ls="none")
        ax.text(x, yy, text, transform=txt, va="center", fontsize=9,
                color=INK_2)


def finish(fig, name: str, credit_y: float = -0.06) -> None:
    fig.text(0.995, credit_y, CREDIT, ha="right", va="top", fontsize=7.5,
             color=MUTED)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def line(ax, x, y, color, lw=2.0, ls="-", marker="o"):
    ax.plot(x, y, color=color, lw=lw, ls=ls, marker=marker, ms=6,
            mec=SURFACE, mew=1.0)


# --------------------------------------------------------------------------


def fig1_ids() -> None:
    r2 = load("r2_ids")
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.0),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.9],
                                          "wspace": 0.62})
    ax, bx, cx = axes

    c = pd.DataFrame(r2["curves"]["50"])
    w = c.pivot(index="window_min", columns="group", values="again")
    x = w.index.to_numpy(float)
    ax.fill_between(x, w["served"], w["abandoned"], color=BAND, lw=0)
    line(ax, x, w["abandoned"], SLOT[0])
    line(ax, x, w["served"], SLOT[1])
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 30, 60, 90, 120, 180, 240])
    ax.set_xlabel("window after the call ended (minutes)")
    ax.set_ylabel("share with another call from the same ID")
    ax.set_title("Calling again after hanging up vs after service")
    ax.text(97, 0.665, "excess", color=INK_2, fontsize=9, va="center")
    label_ends(ax, [(240, w["abandoned"].iloc[-1], "after hanging up",
                     SLOT[0]),
                    (240, w["served"].iloc[-1], "after service", SLOT[1])])
    ax.text(0.0, -0.2, "queued calls from IDs with at most 50 calls a year",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)

    items = []
    for i, (vol, name) in enumerate((("10", "≤10 calls a year"),
                                     ("50", "≤50 calls a year"))):
        c = pd.DataFrame(r2["curves"][vol])
        w = c.pivot(index="window_min", columns="group",
                    values=["again", "again_weighted"])
        x = w.index.to_numpy(float)
        ex = (w["again"]["abandoned"] - w["again"]["served"]).to_numpy()
        ew = (w["again_weighted"]["abandoned"]
              - w["again_weighted"]["served"]).to_numpy()
        line(bx, x, ex, SLOT[i])
        keep = x <= 60
        line(bx, x[keep], ew[keep], SLOT[i], lw=1.6, ls=":", marker="s")
        items.append((240, ex[-1], name, SLOT[i]))
    bx.set_xlim(0, 240)
    bx.set_ylim(0, 0.25)
    bx.set_xticks([0, 30, 60, 90, 120, 180, 240])
    bx.set_xlabel("window after the call ended (minutes)")
    bx.set_ylabel("excess share calling again")
    bx.set_title("Excess after hanging up")
    label_ends(bx, items, gap=0.07)
    bx.text(74, 0.222, "dotted: corrected for callbacks\nthe IDs cannot "
            "see", fontsize=8.5, color=INK_2, va="center")
    bx.text(0.0, -0.2, "IDs with at most 10 or 50 calls a year",
            transform=bx.transAxes, fontsize=8.5, color=MUTED)

    q = np.array(r2["delay_quantiles_min"])
    p = np.linspace(0, 1, q.size)
    cx.plot(q, p, color=SLOT[0], lw=2.2)
    s = r2["delay_summary"]
    for t in (10, 30, 60):
        share = s[f"share_within_{t}"]
        cx.plot([t, t], [0, share], color=MUTED, lw=1, ls=":")
        cx.plot([0, t], [share, share], color=MUTED, lw=1, ls=":")
        cx.text(t + 4, share - 0.06, f"{share:.0%} within {t} min",
                fontsize=9, color=INK_2)
    cx.axvline(s["median"], color=INK_2, lw=1)
    cx.text(s["median"] + 3, 0.08, f"median {s['median']:.0f} min",
            fontsize=9, color=INK_2)
    cx.set_xlim(0, 180)
    cx.set_ylim(0, 1)
    cx.set_xticks([0, 30, 60, 90, 120, 150, 180])
    cx.set_xlabel("minutes from hanging up to the next call")
    cx.set_ylabel("cumulative share")
    cx.set_title("How long until they call again")
    cx.text(0.0, -0.2, f"{s['n']:,} abandoned calls followed by another "
            "within 3 hours", transform=cx.transAxes, fontsize=8.5,
            color=MUTED)
    finish(fig, "fig1_caller_ids", credit_y=-0.1)


def fig2_slopes() -> None:
    r3 = load("r3_slopes")["series"]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    lags = [-3, -2, -1, 1, 2, 3]
    names = (("offered", "all offered calls"),
             ("non_redials", "offered minus identified redials"),
             ("id_redials", "identified redials alone"))
    items = []
    for i, (key, name) in enumerate(names):
        v = np.array([r3[key][f"lag_{L}"]["value"] for L in lags])
        sd = np.array([r3[key][f"lag_{L}"]["sd"] for L in lags])
        dx = (i - 1) * 0.09
        for side in (slice(0, 3), slice(3, 6)):
            xs = np.array(lags[side]) + dx
            ax.vlines(xs, v[side] - 1.645 * sd[side],
                      v[side] + 1.645 * sd[side], color=SLOT[i], lw=1.4,
                      alpha=0.8)
            line(ax, xs, v[side], SLOT[i])
        items.append((3 + dx, v[-1], name, SLOT[i]))
    ax.axhline(0, color=INK_2, lw=0.9)
    ax.axvline(0, color=GRID, lw=6)
    ax.set_xticks(lags)
    ax.set_xticklabels(["−3", "−2", "−1", "+1", "+2", "+3"])
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-0.12, 0.58)
    ax.set_xlabel("half hours from the abandonments to the surplus "
                  "(negative: the abandonments come later)")
    ax.set_ylabel("slope of surplus calls on abandonments")
    ax.set_title("The slope is almost as large backwards as forwards")
    ax.text(-3.35, 0.53, "leads: redials cannot act here",
            fontsize=9, color=INK_2)
    ax.text(0.25, 0.53, "lags: where the earlier article looked",
            fontsize=9, color=INK_2)
    label_ends(ax, items, gap=0.07)
    ax.text(0.0, -0.15, "247 working days, 34 half hours each; bars are "
            "90% day-cluster bootstrap intervals",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    finish(fig, "fig2_lag_lead_slopes", credit_y=-0.08)


def fig3_simcheck() -> None:
    r4 = load("r4_simcheck")
    f = pd.DataFrame(r4["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharey=True)
    for ax, world in zip(axes, ("independent demand", "clustered demand")):
        g = f[f.world == world]
        for p in (0.0, 0.15, 0.3):
            ax.plot([p - 0.05, p + 0.05], [p, p], color=INK, lw=1.2,
                    ls="--", zorder=1)
        ax.text(0.355, 0.3, "truth", fontsize=9, color=INK_2,
                va="center")
        for i, est in enumerate(ESTIMATORS):
            dx = (i - 1.5) * 0.022
            gg = g[g.estimator == est]
            for p, h in gg.groupby("p_true"):
                xs = p + dx + np.linspace(-0.004, 0.004, len(h))
                ax.scatter(xs, h["p_hat"], s=10, color=EST_COLOR[est],
                           alpha=0.35, lw=0)
                ax.plot([p + dx], [h["p_hat"].mean()],
                        marker=EST_MARKER[est], ms=8, color=EST_COLOR[est],
                        mec=SURFACE, mew=1.2, ls="none")
        ax.set_xticks([0, 0.15, 0.3])
        ax.set_xlim(-0.06, 0.38)
        ax.set_ylim(-0.03, 0.82)
        ax.set_xlabel("true redial probability")
        ax.set_title("Demand independent across half hours"
                     if world == "independent demand" else
                     "Demand clustered as in the real data")
    axes[0].set_ylabel("estimated redial probability")
    handles = [plt.Line2D([], [], marker=EST_MARKER[e], color=EST_COLOR[e],
                          ls="none", ms=8, mec=SURFACE)
               for e in ESTIMATORS]
    fig.legend(handles, [EST_LABEL[e] for e in ESTIMATORS], ncol=4,
               loc="upper left", bbox_to_anchor=(0.06, 1.04))
    fig.text(0.06, -0.03, f"{r4['histories']} simulated histories of "
             f"{r4['days']} days per cell (small dots), mean (large marker); "
             "clustered world: AR(1) with ρ = "
             f"{r4['clustering_rho']:.2f}, σ = {r4['clustering_sigma']:.2f}",
             fontsize=8.5, color=MUTED)
    fig.subplots_adjust(wspace=0.08)
    finish(fig, "fig3_simulation_check", credit_y=-0.07)


def fig4_real() -> None:
    r5 = load("r5_estimate")
    f = pd.DataFrame(r5["rows"])
    delays = list(dict.fromkeys(f["delay"]))
    short = {d: d.replace("exponential, mean ", "exponential delays, mean ")
             .replace("empirical, from caller IDs",
                      "delays of identified redials")
             for d in delays}
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    y = 0.0
    ticks, labels = [], []
    for est in ESTIMATORS:
        ax.text(0.005, y, EST_LABEL[est], fontsize=10, color=INK,
                weight="bold", va="center")
        y += 0.9
        for d in delays:
            r = f[(f.estimator == est) & (f.delay == d)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r.p_lo, r.p_hi], [y, y], color=EST_COLOR[est], lw=2)
            ax.plot([r.p_hat], [y], marker=EST_MARKER[est], ms=8,
                    color=EST_COLOR[est], mec=SURFACE, mew=1.2)
            ticks.append(y)
            labels.append(short[d])
            y += 1
        y += 0.5
    ex = r5.get("id_excess") or {}
    if "50" in ex:
        vals = [ex[v][k][w] for v in ("10", "50")
                for k in ("raw", "weighted") for w in ("30", "60")]
        ax.axvspan(min(vals), max(vals), color=BAND, lw=0, zorder=0)
        ax.text(max(vals) + 0.005, y - 0.6,
                "excess calling again after hanging up,\nmeasured from "
                "caller IDs", fontsize=8.5, color=INK_2, va="center")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(y - 0.2, -0.7)
    ax.set_xlim(0, 0.86)
    ax.axvline(0.8, color=MUTED, lw=1, ls=":")
    ax.text(0.795, -0.35, "edge of the simulated grid", fontsize=8.5,
            color=MUTED, ha="right", va="center")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("redial probability estimated from the half-hour table "
                  "(point and 90% interval)")
    ax.set_title("The same table gives anything from 0.15 to 0.74")
    ax.text(0.0, -0.11, "Grey band: abandoned minus served, share with "
            "another call from the same ID within 30 or 60 minutes,\nIDs "
            "with at most 10 or 50 calls a year, raw and corrected for "
            "unseen callbacks", transform=ax.transAxes, va="top",
            fontsize=8.5, color=MUTED, ha="left")
    finish(fig, "fig4_real_estimates", credit_y=0.0)


def main() -> None:
    if (RESULTS / "r2_ids.json").exists():
        fig1_ids()
    fig2_slopes()
    fig3_simcheck()
    if (RESULTS / "r5_estimate.json").exists():
        fig4_real()


if __name__ == "__main__":
    main()
