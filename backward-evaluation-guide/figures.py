"""Figures for the guide, from ``results/``.

    python figures.py

fig1  one cell taken apart: today's CPI model on 1976-81, as issued, shifted, and rescaled
fig2  the simulated checks: what each kind of change looks like in the three parts
fig3  CPI inflation: the matrix, its two regimes, and today's model on the past
fig4  industrial production: the same matrix under Gaussian and Student-t errors
fig5  daily data: today's model on earlier years, by family, in three parts
fig6  Formula One: the matrices of LightGBM and of the level-normalised LightGBM
fig7  the forward check: GBM minus GBMn, article by article
fig8  monitoring: today's model on the year before, every 91 days
fig9  the platform: the matrix of the seasonal linear model averaged over 245 articles
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
PART_COLOR = {"level": BLUE, "spread": ORANGE, "structure": AQUA}
PART_LABEL = {"level": "level: removed by a constant shift", "spread": "spread: removed by a rescaling",
              "structure": "structure: what neither removes"}
FAM_LABEL = {"AR": "autoregression", "SLM": "seasonal linear", "ETS": "exponential smoothing",
             "GBM": "LightGBM", "GBMn": "LightGBM, level-normalised"}
DOM = {"elec": "Electricity, 25 countries", "dev_web": "Wikipedia, 304 development articles",
       "test_web": "Wikipedia, 245 test articles"}
LISTS = {"us_state": "US states", "city": "cities", "solar_system": "Solar System bodies",
         "programming_language": "programming languages", "instrument": "musical instruments",
         "sport": "sports", "mammal": "mammals"}
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog"
DIVERGE = LinearSegmentedColormap.from_list("div", ["#1f5fa8", "#9cc0e8", "#f1f0ec", "#f3a77f", "#c4421a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})


def credit(fig):
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=7, color=INK2)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def heat(ax, E, labels, vmax, title, ticks_every=1, cbar=True, fig=None):
    E = np.array(E, float)
    norm = TwoSlopeNorm(vmin=-vmax / 4, vcenter=0.0, vmax=vmax)
    im = ax.imshow(np.clip(E, -vmax / 4, vmax), cmap=DIVERGE, norm=norm, aspect="equal")
    k = len(labels)
    idx = list(range(0, k, ticks_every))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=8)
    ax.set_yticks(idx)
    ax.set_yticklabels([labels[i] for i in idx], fontsize=8)
    ax.set_xlabel("scored block (test)")
    ax.set_ylabel("block the model was fitted on (train)")
    ax.set_title(title, loc="left", fontsize=10.5)
    for s in ax.spines.values():
        s.set_visible(False)
    if cbar and fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("excess log loss, nats per row", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
    return im


def square(cells: pd.DataFrame, value: str = "excess") -> tuple[np.ndarray, list[str]]:
    labels = sorted(set(cells.train_label) | set(cells.test_label))
    pos = {lab: i for i, lab in enumerate(labels)}
    E = np.zeros((len(labels), len(labels)))
    for r in cells.itertuples():
        E[pos[r.train_label], pos[r.test_label]] = getattr(r, value)
    return E, labels


# --------------------------------------------------------------------------- fig1


def fig1():
    from bwdiag import LinearModel, lag_rows, nlpd, repair, transfer_matrix  # noqa: F401
    from macro import BLOCK, GAP, P, load
    from scipy.stats import t as student_t

    s = load()["cpi_inflation"]
    X, z, t = lag_rows(s.to_numpy(), P)
    dates = s.index[t]
    from bwdiag import blocks_from_end
    blocks = blocks_from_end(len(z), BLOCK)
    (a, b), (c, d) = blocks[-1], blocks[3]
    m = LinearModel(df=5.0).fit(X[a:b], z[a:b])
    mean, scale = m.predict(X[c:d])
    zz = z[c:d]
    rep = repair(zz, mean, scale, 5.0)
    shifted = mean + rep.shift
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    steps = [("as issued", mean, scale, rep.raw),
             ("after the best constant shift", shifted, scale, rep.raw - rep.level),
             ("after the shift and the best rescaling", shifted, scale * rep.factor, rep.repaired)]
    x = np.arange(len(zz))
    for ax, (name, mu, sc, loss) in zip(axes, steps):
        q = student_t.ppf(0.95, 5.0)
        ax.fill_between(x, mu - q * sc, mu + q * sc, color=BLUE, alpha=0.18, lw=0, label="90% predictive interval")
        ax.plot(x, mu, color=BLUE, lw=1.6, label="forecast (centre)")
        ax.plot(x, zz, color=INK, lw=0, marker="o", ms=3.2, label="actual inflation")
        ax.set_title(f"{name}\nmean log loss {loss:.2f} nats", loc="left", fontsize=10)
        ticks = [i for i in range(len(zz)) if dates[c + i].month == 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{dates[c + i]:%Y}" for i in ticks])
        grid(ax)
    axes[0].set_ylabel("CPI inflation, % per year")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"One cell taken apart: the model of {dates[a]:%Y}–{dates[b - 1]:%Y} forecasting "
                 f"{dates[c]:%b %Y}–{dates[d - 1]:%b %Y}, one month ahead", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    cell = pd.read_csv(RES / "macro_cells.csv")
    cell = cell[(cell.series == "cpi_inflation") & (cell.dist == "t") & (cell.train == len(blocks) - 1)
                & (cell.test == 3)].iloc[0]
    fig.text(0.01, 0.035, f"Excess over the block's own model {cell.excess:+.2f} nats = level {cell.level:.2f} (a shift "
             f"of {rep.shift:+.1f} points) + spread {cell.spread:.2f} (scale × {rep.factor:.2f}) + structure "
             f"{cell.structure:+.2f}. Student-t(5) forecasts; the interval is the central 90%.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    credit(fig)
    save(fig, "fig1_one_cell.png")


# --------------------------------------------------------------------------- fig2


def fig2():
    c = pd.read_csv(RES / "sims_cells.csv.gz")
    order = ["none", "level", "spread", "dynamics", "episode"]
    names = {"none": "no change", "level": "level shift", "spread": "spread doubles",
             "dynamics": "persistence 0.2 → 0.8", "episode": "one-block episode"}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.36
    for k, s in enumerate(order):
        for side, cr in enumerate((True, False)):
            x = c[(c.scenario == s) & (c.crosses == cr)]
            if s == "none":
                x = c[c.scenario == s] if cr else x.iloc[0:0]
            if not len(x):
                continue
            m = x[["level", "spread", "structure"]].mean()
            pos = k + (side - 0.5) * (w + 0.04)
            bottom = 0.0
            for p in ("level", "spread"):
                ax.bar(pos, m[p], w, bottom=bottom, color=PART_COLOR[p], alpha=1.0 if cr else 0.45)
                bottom += m[p]
            ax.bar(pos, m["structure"], w, bottom=bottom if m["structure"] > 0 else 0.0, color=PART_COLOR["structure"],
                   alpha=1.0 if cr else 0.45)
            tot = m.sum()
            ax.text(pos, max(bottom + max(m["structure"], 0), 0) + 0.015, f"{tot:+.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[s] for s in order])
    ax.axhline(0, color=INK2, lw=0.8)
    grid(ax)
    ax.set_ylabel("mean over cells, nats per row")
    ax.set_title("Simulated series with one known change: the excess log loss of each cell and its parts", loc="left")
    handles = [Patch(color=PART_COLOR[p], label=PART_LABEL[p]) for p in PART_COLOR]
    handles.append(Patch(color=INK2, alpha=0.45, label="lighter bars: cells within a regime"))
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.80, 0.97, "left bar: cells across the change; right bar: cells within a regime\n"
            "number: excess = level + spread + structure; 100 replicates, AR(1), 10 blocks of 120 rows",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    credit(fig)
    save(fig, "fig2_simulated_checks.png")


# --------------------------------------------------------------------------- fig3 / fig4


def macro_matrix(series, dist):
    c = pd.read_csv(RES / "macro_cells.csv")
    c = c[(c.series == series) & (c.dist == dist)]
    return c


def fig3():
    c = macro_matrix("cpi_inflation", "t")
    E, labels = square(c)
    years = [f"{lab[:4]}–{int(lab[:4]) + 5}" for lab in labels]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    heat(a, E, years, 1.5, "CPI inflation, AR(3) with Student-t errors: excess log loss\n"
         "boxes: the best split into two regimes", fig=fig)
    from bwdiag import Matrix  # regimes from the stored cells
    k = len(labels)
    M = Matrix([(i, i + 1) for i in range(k)], labels, E.copy(), np.zeros((k, k)), np.zeros((k, k)),
               np.zeros((k, k)), np.ones((k, k), int))
    reg, _ = M.regimes(2)
    for s, e in reg:
        a.add_patch(Rectangle((s - 0.5, s - 0.5), e - s + 1, e - s + 1, fill=False, ec=INK, lw=1.6))
    last = c[c.train == c.train.max()].sort_values("test")
    x = np.arange(len(last))
    bottom = np.zeros(len(last))
    for p in ("level", "spread"):
        b.bar(x, last[p], 0.7, bottom=bottom, color=PART_COLOR[p], label=PART_LABEL[p])
        bottom += last[p].to_numpy()
    st = last.structure.to_numpy()
    b.bar(x, st, 0.7, bottom=np.where(st > 0, bottom, 0), color=PART_COLOR["structure"], label=PART_LABEL["structure"])
    b.scatter(x, last.excess, color=INK, s=22, zorder=5, label="excess (sum of the parts)")
    b.set_xticks(x)
    b.set_xticklabels([years[j] for j in last.test], rotation=90, fontsize=8)
    b.axhline(0, color=INK2, lw=0.8)
    grid(b)
    b.set_ylabel("nats per month")
    b.set_title(f"Today's model ({years[-1]}) on each earlier block", loc="left", fontsize=10.5)
    b.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    credit(fig)
    save(fig, "fig3_cpi.png")


def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, dist, name in zip(axes, ("gaussian", "t"), ("Gaussian errors", "Student-t errors")):
        c = macro_matrix("ip_growth", dist)
        E, labels = square(c)
        years = [f"{lab[:4]}–{int(lab[:4]) + 5}" for lab in labels]
        heat(ax, E, years, 3.0, f"Industrial production growth, AR(3), {name}", fig=fig)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    credit(fig)
    save(fig, "fig4_ip_two_distributions.png")


# --------------------------------------------------------------------------- daily


def stacked(ax, x, rows, width=0.7, labels=True):
    bottom = np.zeros(len(x))
    for p in ("level", "spread"):
        v = np.array([r[p] for r in rows])
        ax.bar(x, v, width, bottom=bottom, color=PART_COLOR[p], label=PART_LABEL[p] if labels else None)
        bottom += v
    st = np.array([r["structure"] for r in rows])
    ax.bar(x, st, width, bottom=np.where(st > 0, bottom, 0), color=PART_COLOR["structure"],
           label=PART_LABEL["structure"] if labels else None)
    ex = np.array([r["excess"] for r in rows])
    ax.scatter(x, ex, color=INK, s=24, zorder=5, label="excess (sum of the parts)" if labels else None)


def fig5():
    per = pd.read_csv(RES / "daily_lastrow.csv")
    fams = ["AR", "SLM", "ETS", "GBM", "GBMn"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for k, (ax, dom) in enumerate(zip(axes, ("elec", "dev_web", "test_web"))):
        x = per[per.domain == dom]
        rows = [x[x.family == f][["excess", "level", "spread", "structure"]].mean().to_dict() for f in fams]
        stacked(ax, np.arange(len(fams)), rows, labels=k == 0)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([FAM_LABEL[f].replace(", ", ",\n").replace(" ", "\n", 1) if f != "GBMn" else
                            "LightGBM,\nlevel-\nnormalised" for f in fams], fontsize=8.5)
        ax.axhline(0, color=INK2, lw=0.8)
        grid(ax)
        ax.set_title(DOM[dom], loc="left", fontsize=10)
    axes[0].set_ylabel("today's model on earlier years, nats per day")
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.42, 0.025))
    fig.suptitle("Which models carry what they learned in the last year into earlier years (mean over years and series)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    credit(fig)
    save(fig, "fig5_families.png")


def fig6():
    c = pd.read_csv(RES / "daily_example_cells.csv")
    c = c[(c.domain == "test_web") & (c.series == "sport:Formula One")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.9))
    for ax, fam in zip(axes, ("GBM", "GBMn")):
        E, labels = square(c[c.family == fam])
        years = [lab[:4] + "–" + str(int(lab[:4]) + 1)[2:] for lab in labels]
        heat(ax, E, years, 1.0, f"Formula One article, {FAM_LABEL[fam]}", fig=fig)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    credit(fig)
    save(fig, "fig6_formula_one.png")


def fig7():
    w = pd.read_csv(RES / "daily_forward.csv")
    w = w[w.domain == "test_web"].copy()
    w["d"] = w.GBM - w.GBMn
    order = w.groupby("group").d.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 5.4))
    rng = np.random.default_rng(0)
    lo, hi = np.quantile(w.d, [0.005, 0.995])
    pad = 0.1 * (hi - lo)
    for i, g in enumerate(order):
        x = w[w.group == g].d.to_numpy()
        ax.scatter(np.clip(x, lo - pad, hi + pad), i + rng.uniform(-0.22, 0.22, len(x)), s=16, color=BLUE, alpha=0.55,
                   lw=0)
        m, h = x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))
        ax.errorbar(m, i, xerr=h, fmt="D", ms=8, color=INK, mfc=SURFACE, mew=1.6, capsize=4, zorder=5)
    m, h = w.d.mean(), 1.96 * w.d.std(ddof=1) / np.sqrt(len(w))
    ax.errorbar(m, len(order) + 0.4, xerr=h, fmt="D", ms=9, color=INK, mfc=INK, capsize=4)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(list(range(len(order))) + [len(order) + 0.4])
    ax.set_yticklabels([f"{LISTS[g]} ({(w.group == g).sum()})" for g in order] + [f"all {len(w)} articles"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(lo - pad * 1.2, hi + pad * 1.2)
    ax.set_xlabel("deployed log loss of LightGBM minus the level-normalised LightGBM, nats per day "
                  "(right of zero: the normalised model forecast better)")
    ax.set_title("The forward check: 245 test articles, refitted every 91 days, 2,993 days each", loc="left")
    ax.text(0.995, 0.985, "dots: articles (clipped at the 0.5% and 99.5% quantiles)\ndiamonds: mean ± 95% interval",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    grid(ax, "x")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    credit(fig)
    save(fig, "fig7_forward_check.png")


def fig8():
    mg = pd.read_csv(RES / "daily_monitor.csv", parse_dates=["date"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True)
    for ax, dom in zip(axes, ("elec", "dev_web", "test_web")):
        x = mg[mg.domain == dom]
        ax.plot(x.date, x.excess, color=INK, lw=1.8, marker="o", ms=3.5, label="excess")
        for p in ("level", "spread", "structure"):
            ax.plot(x.date, x[p], color=PART_COLOR[p], lw=1.3, label=PART_LABEL[p])
        ax.axhline(0, color=INK2, lw=0.8)
        grid(ax)
        ax.set_title(DOM[dom], loc="left", fontsize=10)
        ax.set_ylabel("nats per day")
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    axes[-1].set_xlabel("date of the reading (last day of data used)")
    fig.suptitle("Monitoring: the seasonal linear model fitted on the last 364 days, scored on the 364 days before a "
                 "28-day gap (mean over series)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    credit(fig)
    save(fig, "fig8_monitoring.png")


def fig9():
    mm = pd.read_csv(RES / "daily_mean_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.9))
    for ax, fam in zip(axes, ("SLM", "ETS")):
        x = mm[(mm.domain == "test_web") & (mm.family == fam)]
        E, labels = square(x)
        years = [lab[:4] + "–" + str(int(lab[:4]) + 1)[2:] for lab in labels]
        heat(ax, E, years, 1.5, f"{FAM_LABEL[fam].capitalize()}, mean over the 245 test articles", fig=fig)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    credit(fig)
    save(fig, "fig9_platform.png")


def main() -> None:
    fig1()
    fig2()
    fig3()
    fig4()
    for f in (fig5, fig6, fig7, fig8, fig9):
        try:
            f()
        except FileNotFoundError as e:
            print(f"{f.__name__}: skipped ({e.filename} missing)")


if __name__ == "__main__":
    main()
