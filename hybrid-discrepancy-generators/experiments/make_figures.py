"""Draw the article's figures from the results of the experiments."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

from common import FIGURES, load  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BAND = "#efeeea"
COLOR = {
    "Reality": INK,
    "Erlang A": "#2a78d6",
    "Erlang A + boosting": "#eb6834",
    "Erlang A + boosting (link scale)": "#eb6834",
    "Erlang A + Gaussian process": "#1baf7a",
    "Erlang A + linear": "#eda100",
    "CART": "#e87ba4",
    "Erlang A + resampled day levels": "#008300",
    "Erlang A + estimated redials": "#6250d6",
    "Erlang A + both": "#e34948",
    "Boosting on the data": MUTED,
    "True queue given the inputs": INK,
    "Erlang A given reality's inputs": "#2a78d6",
}
DASHED = {"Erlang A + boosting", "Erlang A + Gaussian process",
          "Erlang A + linear", "Boosting on the data", "CART"}
DOTTED = {"Erlang A + boosting (link scale)", "True queue given the inputs",
          "Erlang A given reality's inputs"}
MARKER = {"Reality": "o", "Erlang A": "o", "Erlang A + boosting": "s",
          "Erlang A + boosting (link scale)": "s",
          "Erlang A + Gaussian process": "^", "Erlang A + linear": "D",
          "Erlang A + resampled day levels": "v",
          "Erlang A + estimated redials": "P", "Erlang A + both": "X",
          "Boosting on the data": "o", "True queue given the inputs": "",
          "Erlang A given reality's inputs": "", "CART": "D"}
SHORT = {"Erlang A + boosting": "+ boosting",
         "Erlang A + boosting (link scale)": "+ boosting, logit scale",
         "Erlang A + Gaussian process": "+ Gaussian process",
         "Erlang A + linear": "+ linear",
         "Erlang A + resampled day levels": "+ resampled day levels",
         "Erlang A + estimated redials": "+ estimated redials",
         "Erlang A + both": "+ day levels and redials",
         "Erlang A given reality's inputs": "Erlang A given reality's inputs"}
CREDIT = ("Diogo Ribeiro · simulation, code in "
          "github.com/DiogoRibeiro7/Medium-Blog")

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


def style(name: str) -> dict:
    ls = ":" if name in DOTTED else ("--" if name in DASHED else "-")
    return {"color": COLOR.get(name, MUTED), "ls": ls,
            "lw": 2.4 if name == "Reality" else 2.0,
            "marker": MARKER.get(name, "o"), "ms": 6, "mec": SURFACE,
            "mew": 1.0}


def label(name: str) -> str:
    return SHORT.get(name, name)


def label_ends(ax, items, gap: float = 0.05, log: bool = False) -> None:
    """Direct labels at line ends with a colour key, spread apart."""
    lo, hi = ax.get_ylim()
    tf = np.log10 if log else (lambda v: v)
    inv = (lambda v: 10 ** v) if log else (lambda v: v)
    span = tf(hi) - tf(lo)
    order = sorted(items, key=lambda it: tf(it[1]))
    pos = [tf(it[1]) for it in order]
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
        ax.plot([x], [inv(yy)], marker="s", ms=6, color=color, mec=color,
                transform=key, clip_on=False, ls="none")
        ax.text(x, inv(yy), text, transform=txt, va="center", fontsize=9,
                color=INK if color == INK else INK_2)


def finish(fig, name: str) -> None:
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=7.5,
             color=MUTED)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------


def fig1_fit() -> None:
    h1 = load("h1_fit")
    pred = pd.DataFrame(h1["prediction"])
    pred = pred[~pred.model.str.endswith("(impossible)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 0.9]})
    ax, bx, cx = axes
    models = ["True queue given the inputs", "Erlang A",
              "Boosting on the data", "Erlang A + boosting",
              "Erlang A + Gaussian process", "Erlang A + linear"]
    mean = pred.groupby(["model", "n_days"])["day_rmse_sl"].mean()
    vals = mean.loc[models].to_numpy()
    ax.set_ylim(vals.min() * 0.92, vals.max() * 1.05)
    items = []
    for m in models:
        s = mean.loc[m]
        st = style(m)
        if m == "True queue given the inputs":
            ax.axhline(s.mean(), color=INK, ls=":", lw=1.6)
            items.append((s.index.max(), s.mean(), "true queue given the "
                          "inputs", INK))
            continue
        ax.plot(s.index, s.values, **st)
        items.append((s.index[-1], s.values[-1], label(m), st["color"]))
    ax.set_xscale("log")
    ticks = sorted(pred["n_days"].unique())
    ax.set_xticks(ticks, [str(int(t)) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] * 0.8, ticks[-1] * 6)
    label_ends(ax, items, gap=0.06)
    ax.set_xlabel("days of training data")
    ax.set_ylabel("RMSE of the day's service level")
    ax.set_title("Predicting held-out days from their inputs")

    c = h1["curve"]
    rng_ = h1["train_load_range"]
    if c:
        load_ = np.array(c["load"])
        bx.axvspan(rng_["p1"], rng_["p99"], color=BAND, lw=0, zorder=0)
        bx.axhline(0, color=MUTED, lw=1)
        rb = pd.DataFrame(h1["residual_bins"])
        rb = rb[rb.rows >= 10]
        bx.scatter(rb["load"], rb["mean_residual"], s=18 + rb["rows"] / 6,
                   color="#cfcec9", zorder=2, lw=0)
        items = []
        for key, name in (("boosting", "Erlang A + boosting"),
                          ("Gaussian process", "Erlang A + Gaussian process"),
                          ("linear", "Erlang A + linear")):
            if key in c:
                st = style(name)
                bx.plot(load_, c[key], color=st["color"], ls=st["ls"],
                        lw=2)
                items.append((load_[-1], c[key][-1], label(name),
                              st["color"]))
        lo = min(min(c[k]) for k in ("boosting", "Gaussian process",
                                     "linear") if k in c)
        hi = max(max(c[k]) for k in ("boosting", "Gaussian process",
                                     "linear") if k in c)
        lo = min(lo, rb["mean_residual"].min())
        hi = max(hi, rb["mean_residual"].max())
        bx.set_ylim(lo - 0.02, hi + 0.02)
        bx.set_xlim(load_[0] - 0.05, load_[-1] + 0.75)
        label_ends(bx, items, gap=0.07)
        bx.text((rng_["p1"] + rng_["p99"]) / 2, lo - 0.012,
                "98% of training\nhalf hours", ha="center", fontsize=8.5,
                color=INK_2)
    bx.set_xlabel("offered load per agent")
    bx.set_ylabel("correction to the service level")
    bx.set_title("What each correction does past the data")

    gen = pd.DataFrame(h1["generation"]).dropna(subset=["auc"])
    order = ["Erlang A", "Erlang A + boosting",
             "Erlang A + boosting (link scale)",
             "Erlang A + Gaussian process", "Erlang A + linear",
             "Erlang A + resampled day levels",
             "Erlang A + estimated redials", "Erlang A + both", "CART"]
    order = [o for o in order if o in set(gen.model)]
    lengths = sorted(gen["n_days"].unique())
    y = np.arange(len(order))[::-1]
    for k, n in enumerate(lengths):
        s = gen[gen.n_days == n].groupby("model")["auc"].mean() \
            .reindex(order)
        cx.scatter(s.values, y, s=40,
                   marker="o" if k == len(lengths) - 1 else "o",
                   facecolors=[COLOR[o] for o in order] if k == len(lengths)
                   - 1 else "none",
                   edgecolors=[COLOR[o] for o in order], lw=1.5, zorder=3,
                   label=f"{int(n)} days")
    cx.axvline(0.54, color=INK, ls=":", lw=1.4)
    cx.text(0.545, y[0] + 0.6, "real vs real", fontsize=8.5, color=INK)
    cx.set_yticks(y, [label(o) for o in order])
    cx.set_xlim(0.5, max(0.82, gen["auc"].max() + 0.02))
    cx.legend(loc="lower right", title="training data",
              title_fontsize=9)
    cx.set_xlabel("discriminator AUC of 40 generated days")
    cx.set_title("Used as generators")
    cx.grid(axis="y", visible=False)
    fig.tight_layout()
    finish(fig, "fig1_fit")


def fig2_surge() -> None:
    h2 = load("h2_surge")
    f = pd.DataFrame(h2["rows"])
    mean = f.groupby(["model", "m"])["sl"].mean()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    panels = [
        (ax, ["Erlang A", "Erlang A + boosting",
              "Erlang A + Gaussian process", "Erlang A + linear",
              "Reality"], "Correcting the outputs"),
        (bx, ["Erlang A", "Erlang A + resampled day levels",
              "Erlang A + estimated redials", "Erlang A + both",
              "Erlang A given reality's inputs", "Reality"],
         "Correcting the inputs and the structure"),
    ]
    for a, names, title in panels:
        a.set_ylim(0, 1)
        a.axvspan(0.83, 1.17, color=BAND, lw=0, zorder=0)
        items = []
        for n in names:
            s = mean.loc[n]
            st = style(n)
            a.plot(s.index, s.values, **st, zorder=3)
            items.append((s.index[-1], s.values[-1], label(n), st["color"]))
        label_ends(a, items, gap=0.05)
        a.set_xlim(0.86, 1.95)
        a.set_xlabel("fresh demand relative to normal (m)")
        a.set_title(title)
    ax.text(1.0, 0.03, "90% of\ntraining days", ha="center", fontsize=8.5,
            color=INK_2)
    ax.set_ylabel("service level (share answered within 20 s)")
    fig.tight_layout()
    finish(fig, "fig2_surge")


def fig3_confounding() -> None:
    h4 = load("h4_confounding")
    f = pd.DataFrame(h4["rows"])
    ratio = f[f.model == "ratio estimate"]["patience"].mean()
    g = f[f.model != "ratio estimate"]
    names = {"Erlang A alone": ("Erlang A", "Erlang A alone"),
             "+ boosting": ("Erlang A + boosting", "+ boosting"),
             "+ Gaussian process": ("Erlang A + Gaussian process",
                                    "+ Gaussian process")}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.0))
    for metric, a, title, ylab in (
            ("day_rmse_sl", ax, "Fit to held-out days",
             "RMSE of the day's service level"),
            ("surge_sl", bx, "Forecast for a 30% surge",
             "service level")):
        s = g.groupby(["model", "patience"])[metric].mean()
        items = []
        for m, (key, lab) in names.items():
            st = style(key)
            a.plot(s.loc[m].index, s.loc[m].values, **st)
            items.append((s.loc[m].index[-1], s.loc[m].values[-1], lab,
                          st["color"]))
        if metric == "surge_sl":
            real = g["surge_sl_reality"].mean()
            a.plot([min(h4["grid"]), max(h4["grid"])], [real, real],
                   color=INK, lw=2)
            items.append((max(h4["grid"]), real, "reality", INK))
            lo, hi = a.get_ylim()
            a.set_ylim(min(lo, real - 0.03), hi)
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi)
        label_ends(a, items, gap=0.07)
        a.axvline(ratio, color=MUTED, lw=1)
        a.axvline(h4["true_mean_patience"], color=MUTED, lw=1, ls=":")
        a.text(ratio, hi, " calibrated", fontsize=8.5, color=INK_2,
               va="top", ha="right", rotation=90)
        a.text(h4["true_mean_patience"], hi, " reality's mean", fontsize=8.5,
               color=INK_2, va="top", ha="left", rotation=90)
        a.set_xlim(min(h4["grid"]) - 10, max(h4["grid"]) * 1.45)
        a.set_xlabel("mean patience set in Erlang A, 1/θ (seconds)")
        a.set_ylabel(ylab)
        a.set_title(title)
    fig.tight_layout()
    finish(fig, "fig3_confounding")


def fig4_redials() -> None:
    h3 = load("h3_redials")
    f = pd.DataFrame(h3["estimates"])
    b = h3["binding"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.0),
                                 gridspec_kw={"width_ratios": [1, 1.2]})
    grid = np.array(b["grid"])
    sims = np.array(b["beta_simulated"])
    fine = np.linspace(grid[0], grid[-1], 200)
    fit = np.maximum.accumulate(np.polyval(np.polyfit(grid, sims, 2), fine))
    ax.plot(fine, fit, color="#6250d6", lw=2)
    ax.scatter(grid, sims, color="#6250d6", s=36, zorder=3, ec=SURFACE)
    beta = b["beta_observed"]
    p_hat = float(np.interp(beta, fit, fine))
    ax.axhline(beta, color=INK, lw=1.4)
    ax.plot([p_hat, p_hat], [ax.get_ylim()[0], beta], color=INK, lw=1,
            ls=":")
    ax.text(0.02, beta, f"  observed coefficient {beta:.3f}", va="bottom",
            fontsize=9, color=INK)
    ax.text(p_hat, ax.get_ylim()[0], f" p̂ = {p_hat:.2f}", va="bottom",
            fontsize=9, color=INK)
    aux = pd.DataFrame(h3["auxiliary"])
    pl = aux[aux.world == "no redials"]["beta"]
    if len(pl):
        ax.axhspan(pl.mean() - 2 * pl.std(), pl.mean() + 2 * pl.std(),
                   color=BAND, lw=0, zorder=0)
        ax.text(0.98, pl.mean(), "world without redials\n(mean ± 2 sd)",
                ha="right", va="center", fontsize=8.5, color=INK_2,
                transform=ax.get_yaxis_transform())
    ax.set_xlabel("redial probability p in the simulator")
    ax.set_ylabel("auxiliary coefficient")
    ax.set_title("Binding function, one 90-day history")

    e = f[(f.world == "redials 0.6") & (f.delay == 600.0)]
    s = e.groupby("n_days")["p_hat"].agg(["mean", "std"])
    bx.errorbar(s.index, s["mean"], yerr=s["std"], color="#6250d6",
                marker="P", ms=8, lw=2, capsize=0, mec=SURFACE,
                label="redials at p = 0.6, delay assumed right")
    pl = f[f.world == "no redials"].groupby("n_days")["p_hat"] \
        .agg(["mean", "std"])
    bx.errorbar(pl.index * 1.08, pl["mean"], yerr=pl["std"], color=MUTED,
                marker="o", ms=7, lw=0, elinewidth=1.5, capsize=0,
                mec=SURFACE, label="no redials in the world")
    for delay, mk in ((300.0, "v"), (1200.0, "^")):
        d = f[(f.world == "redials 0.6") & (f.delay == delay)] \
            .groupby("n_days")["p_hat"].agg(["mean", "std"])
        bx.errorbar(d.index * (0.9 if delay == 300 else 1.17), d["mean"],
                    yerr=d["std"], color="#e34948", marker=mk, ms=7, lw=0,
                    elinewidth=1.5, capsize=0, mec=SURFACE,
                    label=f"delay assumed {delay:.0f} s (truth 600 s)")
    bx.axhline(h3["true_p"], color=INK, lw=1.2)
    bx.set_xscale("log")
    ticks = sorted(e["n_days"].unique())
    bx.set_xticks(ticks, [str(int(t)) for t in ticks])
    bx.minorticks_off()
    bx.set_ylim(-0.05, 1.0)
    bx.set_xlabel("days of data")
    bx.set_ylabel("estimated redial probability p̂")
    bx.set_title("Estimates across histories (mean ± sd)")
    bx.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    finish(fig, "fig4_redials")


def fig5_intervention() -> None:
    h5 = load("h5_intervention")
    h6 = load("h6_decision")
    f = pd.DataFrame(h5["rows"])
    eff = f.groupby(["model", "k"])["sl_effect"].mean()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    names = ["Erlang A", "Erlang A + boosting", "Erlang A + Gaussian process",
             "Erlang A + resampled day levels",
             "Erlang A + estimated redials", "Erlang A + both", "Reality"]
    names = [n for n in names if n in eff.index.get_level_values(0)]
    vals = eff.loc[names].to_numpy()
    ax.set_ylim(vals.min() - 0.03, vals.max() + 0.03)
    items = []
    for n in names:
        s = eff.loc[n]
        st = style(n)
        ax.plot(s.index, s.values, **st)
        items.append((s.index[-1], s.values[-1], label(n), st["color"]))
    label_ends(ax, items, gap=0.05)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(min(h5["ks"]) - 0.5, max(h5["ks"]) + 5.5)
    ax.set_xlabel("agents added to every half hour (k)")
    ax.set_ylabel("change in service level vs current staffing")
    ax.set_title("Effect of staffing, as a generator")

    curves = {n: pd.Series({int(k): v for k, v in c.items()}).sort_index()
              for n, c in h6["curves"].items()}
    for n, s in curves.items():
        st = style(n)
        bx.plot(s.index, s.values, **{**st, "ms": 5}, label=label(n))
    bx.axhline(h6["coverage"], color=MUTED, lw=1)
    for n, info in h6["chosen"].items():
        k = info["k"]
        if k is None:
            continue
        bx.plot([k], [curves[n][k]], marker="o", ms=12, mfc="none",
                mec=COLOR.get(n, MUTED), mew=1.6, ls="none")
    bx.set_ylim(0.2, 1.02)
    bx.set_xlabel("agents added on launch days (k)")
    bx.set_ylabel(f"share of days with service level ≥ "
                  f"{h6['target']:.0%}")
    bx.set_title("Launch staffing: circles mark each choice")
    bx.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    finish(fig, "fig5_intervention")


def main() -> None:
    for fn in (fig1_fit, fig2_surge, fig3_confounding, fig4_redials,
               fig5_intervention):
        try:
            fn()
            print(f"drew {fn.__name__}")
        except FileNotFoundError as exc:
            print(f"skipped {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
