"""Draw the article's figures from the results of the experiments."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
# Categorical slots in fixed order; colour follows the generator everywhere.
COLOR = {
    "Erlang A": "#2a78d6",
    "Erlang C": "#eb6834",
    "Gaussian copula": "#1baf7a",
    "CART": "#eda100",
    "Gaussian mixture": "#e87ba4",
    "SMOTE": "#008300",
    "Reality": INK,
    "Real (holdout)": INK,
    "Empirical average": INK,
}
MECHANISTIC = {"Erlang A", "Erlang C"}
MARKER = {"Erlang A": "o", "Erlang C": "o", "Gaussian copula": "s",
          "CART": "D", "Gaussian mixture": "^", "SMOTE": "v", "Reality": "o"}
CREDIT = ("Diogo Ribeiro · simulation, code in "
          "github.com/DiogoRibeiro7/Medium-Blog")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11.5, "axes.titlecolor": INK, "axes.titlelocation":
    "left", "legend.frameon": False, "legend.fontsize": 9.5,
})


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def style(name: str) -> dict:
    data_driven = name not in MECHANISTIC and name != "Reality"
    return {"color": COLOR.get(name, MUTED), "lw": 2.4 if name == "Reality"
            else 2.0, "ls": "--" if data_driven else "-",
            "marker": MARKER.get(name, "o"), "ms": 6,
            "mec": SURFACE, "mew": 1.2}


def label_end(ax, x, y, text, color, dy=0.0) -> None:
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK_2 if color != INK else INK)


def label_ends(ax, items, gap: float = 0.045, log: bool = False) -> None:
    """Direct labels at line ends, pushed apart so they never overlap.

    ``items`` is a list of (x, y, text, color); ``gap`` is the minimum
    vertical distance as a share of the axis height.
    """
    lo, hi = ax.get_ylim()
    tf = (lambda v: np.log10(v)) if log else (lambda v: v)
    inv = (lambda v: 10 ** v) if log else (lambda v: v)
    span = tf(hi) - tf(lo)
    order = sorted(items, key=lambda it: tf(it[1]))
    pos = [tf(it[1]) for it in order]
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap * span:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1] = mid - gap * span / 2
                pos[i] = mid + gap * span / 2
                moved = True
        if not moved:
            break
    shifted = offset_copy(ax.transData, fig=ax.figure, x=7, y=0,
                          units="points")
    for (x, y, text, color), yy in zip(order, pos):
        ax.text(x, inv(yy), text, transform=shifted, va="center",
                fontsize=9, color=INK if color == INK else INK_2)


def finish(fig, name: str) -> None:
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=7.5,
             color=MUTED)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------


def fig1_fidelity() -> None:
    e1 = load("e1_fidelity")
    sc = e1["scatter"]
    order = ["Real (holdout)", "Erlang C", "Erlang A", "Gaussian copula",
             "CART", "SMOTE", "Gaussian mixture"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6))
    real = sc["Real (holdout)"]
    for ax, name in zip(axes.flat, order):
        ax.scatter(real["load"], real["sl"], s=7, color="#cfcec9", lw=0)
        col = INK if name == "Real (holdout)" else COLOR[name]
        ax.scatter(sc[name]["load"], sc[name]["sl"], s=7, color=col, lw=0,
                   alpha=0.85)
        ax.set_title("Real, 40 held-out days" if name == "Real (holdout)"
                     else f"{name} (grey: real)")
        ax.set_xlim(0.3, 1.45)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("offered load per agent")
        ax.set_ylabel("service level (20 s)")
    ax = axes.flat[7]
    f = pd.DataFrame(e1["rows"])
    floor = f[f.generator == "Real (independent)"]["auc"].mean()
    ax.set_ylim(0.45, 1.0)
    items = []
    for name in order[1:]:
        g = (f[f.generator == name].groupby("n_days")["auc"]
             .agg(["mean", "std"]))
        st = style(name)
        ax.errorbar(g.index, g["mean"], yerr=g["std"], capsize=0,
                    elinewidth=1, **st)
        items.append((g.index[-1], g["mean"].iloc[-1], name, st["color"]))
    label_ends(ax, items, gap=0.06)
    ax.axhline(floor, color=INK, lw=1.2)
    ax.text(28, floor - 0.012, "real vs real", va="top", fontsize=9,
            color=INK)
    ax.set_xscale("log")
    ax.set_xticks([30, 90, 270, 810], ["30", "90", "270", "810"])
    ax.minorticks_off()
    ax.set_xlim(25, 4000)
    ax.set_xlabel("days of training data")
    ax.set_ylabel("discriminator AUC")
    ax.set_title("How easily a classifier spots the fakes")
    fig.tight_layout()
    finish(fig, "fig1_fidelity")


def fig2_surge() -> None:
    e2 = load("e2_surge")
    f = pd.DataFrame(e2["rows"])
    q = e2["training_day_ratio_quantiles"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    mean = f.groupby(["generator", "m"])["sl"].mean()
    for a in (ax, bx):
        a.axvspan(q[1], q[2], color="#efeeea", lw=0, zorder=0)
        a.set_xlabel("fresh demand relative to normal (m)")
        a.set_ylim(0, 1)
        a.set_xlim(0.85, 1.78)
    ax.text((q[1] + q[2]) / 2, 0.03, "90% of\ntraining days", ha="center",
            fontsize=8.5, color=INK_2)
    items = []
    for name in ["Erlang C", "Erlang A", "Gaussian copula", "CART",
                 "Gaussian mixture", "Reality"]:
        s = mean.loc[name]
        st = style(name)
        ax.plot(s.index, s.values, **st, zorder=3)
        items.append((s.index[-1], s.values[-1], name, st["color"]))
    label_ends(ax, items)
    ax.set_ylabel("service level (share answered within 20 s)")
    ax.set_title("Asking each generator about a demand surge")
    red = pd.DataFrame(e2["redials"])
    red = red[red.m != "in-sample"].astype({"m": float})
    ramp = {0.0: "#86b6ef", 0.3: "#3987e5", 0.6: "#1c5cab", 0.9: "#0d366b"}
    items = []
    for p, g in red.groupby("p"):
        s = g.groupby("m")["sl"].mean()
        bx.plot(s.index, s.values, color=ramp[p], lw=2, marker="o", ms=5,
                mec=SURFACE)
        items.append((s.index[-1], s.values[-1], f"redials p = {p:.1f}",
                      ramp[p]))
    s = mean.loc["Reality"]
    bx.plot(s.index, s.values, **style("Reality"))
    items.append((s.index[-1], s.values[-1], "Reality", INK))
    label_ends(bx, items)
    bx.set_title("Erlang A plus redials: one past, several futures")
    fig.tight_layout()
    finish(fig, "fig2_surge")


def fig3_intervention() -> None:
    e3 = load("e3_intervention")
    f = pd.DataFrame(e3["rows"])
    eff = f.groupby(["method", "k"])["sl_effect"].mean()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    lo = min(eff.min(), -0.05) - 0.03
    hi = eff.max() + 0.03
    panels = [
        (ax, ["Erlang C", "Erlang A", "Gaussian copula (adjusted)",
              "CART (adjusted)", "Gaussian mixture (adjusted)", "Reality"],
         "Conditioning on offered calls and agents"),
        (bx, ["Gaussian copula (naive)", "CART (naive)",
              "Gaussian mixture (naive)", "Reality"],
         "Conditioning on agents only"),
    ]
    for a, methods, title in panels:
        a.set_ylim(lo, hi)
        a.axhline(0, color=MUTED, lw=1)
        a.axvline(0, color=MUTED, lw=1)
        items = []
        for m in methods:
            base = m.split(" (")[0]
            st = style(base)
            s = eff.loc[m]
            a.plot(s.index, s.values, **st)
            items.append((s.index[-1], s.values[-1], base, st["color"]))
        label_ends(a, items)
        a.set_title(title)
        a.set_xlabel("agents added to every half hour (k)")
        a.set_xlim(-6.5, 9.2)
    ax.set_ylabel("change in service level vs current staffing")
    fig.tight_layout()
    finish(fig, "fig3_intervention")


def fig4_error_budget() -> None:
    e4 = load("e4_error_budget")
    truth = e4["truth"]["sl"]
    a = pd.DataFrame(e4["panel_a"])
    b = pd.DataFrame(e4["panel_b"])
    rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))  # noqa: E731
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    order = ["Erlang A", "Gaussian copula", "CART", "SMOTE",
             "Gaussian mixture"]
    sa = {n: (lambda g: (g["estimate"] - truth)
              .groupby(g["synthetic_days"]).agg(rmse))(a[a.generator == n])
          for n in order}
    sb = {n: (lambda g: (g["estimate"] - truth)
              .groupby(g["real_days"]).agg(rmse))(b[b.generator == n])
          for n in ["Empirical average"] + order}
    vals = np.concatenate([s.values for s in list(sa.values())
                           + list(sb.values())])
    for a_ in (ax, bx):
        a_.set_yscale("log")
        a_.set_xscale("log")
        a_.set_ylim(vals.min() / 1.5, vals.max() * 1.5)
    items = []
    for name in order:
        s, st = sa[name], style(name)
        ax.plot(s.index, s.values, **st)
        items.append((s.index[-1], s.values[-1], name, st["color"]))
    label_ends(ax, items, log=True)
    ax.set_xlabel("synthetic days generated (real data fixed at 90 days)")
    ax.set_ylabel("RMSE of the long-run service level")
    ax.set_title("More simulation")
    ax.set_xlim(0.8, 3000)
    items = []
    for name in ["Empirical average"] + order:
        s = sb[name]
        st = style(name) if name != "Empirical average" else {
            "color": INK, "lw": 2.4, "ls": "-", "marker": "o", "ms": 6,
            "mec": SURFACE, "mew": 1.2}
        bx.plot(s.index, s.values, **st)
        items.append((s.index[-1], s.values[-1],
                      "average of the real days" if name ==
                      "Empirical average" else name, st["color"]))
    label_ends(bx, items, log=True)
    bx.set_xlabel("real days available (synthetic sample large)")
    bx.set_title("More data")
    bx.set_xlim(7, 6000)
    fig.tight_layout()
    finish(fig, "fig4_error_budget")


def fig5_calibration() -> None:
    e5 = load("e5_calibration")
    f = pd.DataFrame(e5["rows"])
    f["patience"] = 1 / f["theta"]
    colors = {"ratio": "#6250d6", "SMM (abandonment)": "#e34948",
              "SMM (waiting)": "#008300"}
    markers = {"ratio": "o", "SMM (abandonment)": "s", "SMM (waiting)": "D"}
    offsets = {"ratio": 0.0, "SMM (abandonment)": -0.07,
               "SMM (waiting)": 0.07}
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for name, g in f.groupby("estimator"):
        s = g.groupby("n_days")["patience"].agg(["mean", "std"])
        x = s.index * np.exp(offsets[name])
        ax.errorbar(x, s["mean"], yerr=s["std"], color=colors[name],
                    marker=markers[name], ms=6, mec=SURFACE, lw=2,
                    elinewidth=1.2, capsize=0, label=name)
    items = []
    for name, theta in e5["limits"].items():
        lim = 1 / theta
        ax.plot([600, 2200], [lim, lim], color=colors[name], lw=2)
        items.append((2200, lim, f"{name}, {e5['limit_days']} days: "
                      f"{lim:.0f} s", colors[name]))
    ref = 1 / e5["references"]["one over mean patience"]
    ax.axhline(ref, color=INK, lw=1.4)
    items.append((2200, ref, f"true mean patience: {ref:.0f} s", INK))
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 90, 270, 1000], ["10", "30", "90", "270",
                                             "1,000"])
    ax.minorticks_off()
    ax.set_xlim(7, 12000)
    label_ends(ax, items, gap=0.05)
    ax.legend(loc="upper left")
    ax.set_xlabel("days of data used for calibration")
    ax.set_ylabel("calibrated mean patience 1/θ (seconds)")
    ax.set_title("Three consistent estimators of one parameter")
    fig.tight_layout()
    finish(fig, "fig5_calibration")


def fig6_decision() -> None:
    e7 = load("e7_decision")
    curves = {n: pd.Series({int(k): v for k, v in c.items()}).sort_index()
              for n, c in e7["curves"].items()}
    show = {"Reality": "Reality",
            "Erlang A (point)": "Erlang A, surge fixed at 1.2",
            "Erlang A (randomised)": "Erlang A, randomised per day",
            "Erlang A (robust)": "Erlang A, 90% of scenarios",
            "Gaussian copula": "Gaussian copula"}
    styles = {
        "Reality": dict(color=INK, lw=2.4, marker="o"),
        "Erlang A (point)": dict(color="#2a78d6", lw=2, marker="o"),
        "Erlang A (randomised)": dict(color="#6250d6", lw=2, marker="s"),
        "Erlang A (robust)": dict(color="#e34948", lw=2, marker="D"),
        "Gaussian copula": dict(color="#1baf7a", lw=2, marker="s", ls="--"),
    }
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    for name, s in curves.items():
        ax.plot(s.index, s.values, ms=5, mec=SURFACE, **styles[name],
                label=show[name])
    ax.axhline(e7["coverage"], color=MUTED, lw=1)
    ax.legend(loc="lower right")
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("agents added to every half hour on launch days (k)")
    ax.set_ylabel(f"share of days with service level ≥ {e7['target']:.0%}")
    ax.set_title("What each method expects, and what happens")
    real = curves["Reality"]
    bx.plot(real.index, real.values, color="#cfcec9", lw=2)
    bx.axhline(e7["coverage"], color=MUTED, lw=1)
    items = []
    for name, info in e7["chosen"].items():
        k = info["k"]
        if k is None or name == "Reality":
            continue
        y = real[k]
        bx.plot([k], [y], ls="none", ms=9, mec=SURFACE,
                **{key: v for key, v in styles[name].items()
                   if key in ("color", "marker")})
        items.append((k, y, f"{show[name]}: k = {k}", styles[name]["color"]))
    k_star = e7["chosen"]["Reality"]["k"]
    bx.plot([k_star], [real[k_star]], marker="o", ms=9, color=INK,
            mec=SURFACE, ls="none")
    items.append((k_star, real[k_star], f"smallest k that works: {k_star}",
                  INK))
    bx.set_ylim(0.3, 1.02)
    label_ends(bx, items, gap=0.07)
    bx.set_xlim(-0.5, max(real.index) + 6)
    bx.set_xlabel("k chosen")
    bx.set_title("Each choice scored in reality")
    fig.tight_layout()
    finish(fig, "fig6_decision")


def main() -> None:
    for fn in (fig1_fidelity, fig2_surge, fig3_intervention,
               fig4_error_budget, fig5_calibration, fig6_decision):
        try:
            fn()
            print(f"drew {fn.__name__}")
        except FileNotFoundError as exc:
            print(f"skipped {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
