"""Figures for the article. Reads results/, writes figures/fig*.png.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
DATA = HERE / "data"

# reference palette (light mode), validated: blue / orange / aqua clear all-pairs CVD checks
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})


def save(fig, name: str, source: str) -> None:
    fig.text(0.01, 0.005, source, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------


def fig1_static_surface() -> None:
    s = pd.read_csv(RES / "sim_static_surface.csv")
    base = s[s.lam == s.lam.min()]
    m = base.groupby(["rho_target", "top1_target"]).efficiency.mean().unstack()
    rhos, tops = m.index.to_numpy(), m.columns.to_numpy()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.pcolormesh(tops * 100, rhos, m.to_numpy(), cmap=cmap, vmin=0.3, vmax=1.0, shading="nearest")
    cs = ax.contour(tops * 100, rhos, m.to_numpy(), levels=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                    colors="white", linewidths=0.8)
    ax.clabel(cs, fmt="%.2f", fontsize=8, colors="white")
    best = m.idxmax(axis=1)
    ax.plot(best.to_numpy() * 100, rhos, color=ORANGE, lw=2)
    ax.annotate("most efficient top-1% share\nfor each degree of coupling", xy=(best.iloc[-6] * 100, rhos[-6]),
                xytext=(30, 0.33), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("Top 1% share of wealth (%)")
    ax.set_ylabel("Rank correlation of wealth with productivity")
    ax.set_title("Output relative to first best")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, colors=INK2)

    colors = {0.0: MUTED, 0.5: BLUE, 0.9: ORANGE, 1.0: INK}
    for rho, c in colors.items():
        sl = base[np.isclose(base.rho_target, rho)].groupby("top1_target").efficiency.mean()
        bx.plot(sl.index * 100, sl.values, color=c, lw=2)
        bx.text(sl.index[-1] * 100 + 0.8, sl.values[-1], f"ρ = {rho:g}", color=INK2, fontsize=9, va="center")
    k_top1 = 100 * (1 - norm.cdf(norm.ppf(0.99) - 2.0))
    bx.axvline(k_top1, color=MUTED, lw=1, zorder=0)
    bx.text(k_top1 - 0.8, 0.36, f"top-1% share of capital\nin the first-best allocation ({k_top1:.0f}%)",
            fontsize=8.5, color=INK2, ha="right")
    bx.set_xlim(2, 58)
    bx.set_ylim(0.3, 1.02)
    bx.set_xlabel("Top 1% share of wealth (%)")
    bx.set_ylabel("Output relative to first best")
    bx.set_title("Slices at four degrees of coupling")
    fig.tight_layout()
    save(fig, "fig1_static_surface",
         "Simulation: 20,000 owners, y = z·k^0.5, collateral limit k ≤ 1.25·a, mean of 4 seeds. "
         "Source: experiments.py (E1).")


def fig2_steady_states() -> None:
    inh = pd.read_csv(RES / "sim_inheritance.csv").groupby("rho_g").mean(numeric_only=True)
    per = pd.read_csv(RES / "sim_persistence.csv")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(inh.top1_wealth * 100, inh.efficiency, color=BLUE, marker="o", ms=7, lw=1.6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="Heir's ability: correlation with parent's (ρ_g)")
    for g, r in inh.iterrows():
        if g in (0.0, 1.0):
            ax.annotate(f"ρ_g = {g:g}", (r.top1_wealth * 100, r.efficiency), textcoords="offset points",
                        xytext=(8, -14), fontsize=8.5, color=INK2)
    for g, mk in ((0.0, "o"), (1.0, "s")):
        p = per[per.rho_g == g].groupby("psi").mean(numeric_only=True)
        ax.plot(p.top1_wealth * 100, p.efficiency, color=ORANGE, marker=mk, ms=7, lw=1.6,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"Persistence of ability within a life (ψ), ρ_g = {g:g}")
        for psi, r in p.iterrows():
            if g == 0.0 and psi < 0.98:
                ax.annotate(f"ψ = {psi:.2f}", (r.top1_wealth * 100, r.efficiency), textcoords="offset points",
                            xytext=(6, -15), fontsize=8.5, color=INK2, ha="left")
    ax.annotate("baseline ψ = 0.98", (inh.top1_wealth.iloc[0] * 100, inh.efficiency.iloc[0]),
                textcoords="offset points", xytext=(-10, 10), fontsize=8.5, color=INK2, ha="right")
    ax.set_xlim(4.5, 21.8)
    ax.set_xlabel("Top 1% share of wealth in steady state (%)")
    ax.set_ylabel("Output relative to first best")
    ax.set_title("Across simulated economies, more concentration comes with more efficiency")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig2_steady_states",
         "Simulation: 10,000 dynasties, 350-year burn-in, mean of the next 100 years, 3 seeds. "
         "Source: experiments.py (E2, E5).")


def fig3_reranking() -> None:
    d = pd.read_csv(RES / "sim_reranking.csv")
    d["regime"] = np.where((d.tau_k == 0) & (d.tau_a == 0), "No tax",
                           np.where(d.tau_k > 0, "25% capital income tax", "Revenue-equal wealth tax"))
    p = d.groupby(["regime", "t"]).mean(numeric_only=True).reset_index()
    colors = {"No tax": MUTED, "25% capital income tax": BLUE, "Revenue-equal wealth tax": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    for ax, col, title, ylab in ((axes[0], "coupling", "Coupling of wealth to productivity", "Spearman(wealth, productivity)"),
                                  (axes[1], "efficiency", "Output relative to first best", "Efficiency")):
        for name, c in colors.items():
            q = p[p.regime == name]
            ax.plot(q.t, q[col], color=c, lw=4 if name == "No tax" else 2, label=name)
        ax.set_title(title)
        ax.set_xlabel("Years after productivity is re-ranked")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 120)
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].text(118, 0.6, "no tax and the capital income tax\nnearly coincide", fontsize=8.5, color=INK2,
                 ha="right", va="top")
    fig.tight_layout()
    save(fig, "fig3_reranking",
         "Simulation: at t = 0 every dynasty's productivity is redrawn independently of its wealth; "
         "mean of 3 seeds. Source: experiments.py (E4).")


def fig4_coverage() -> None:
    s = pd.read_csv(DATA / "wid_shares.csv", keep_default_na=False)
    w = s[(s.variable == "shweal992j") & (s.percentile == "p99p100") & (s.country.str.len() == 2)]
    w = w[(w.year >= 1980) & (w.year <= 2024)]
    pub = w.groupby("year").country.nunique()
    obs = w[w.data_quality > 0].groupby("year").country.nunique().reindex(pub.index, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(pub.index, pub.values, color=MUTED, lw=2)
    ax.fill_between(obs.index, 0, obs.values, color=BLUE, alpha=0.85, lw=0)
    ax.text(1981, pub.values[0] - 14, "areas with a published top-1% wealth share", fontsize=9, color=INK2)
    peak = obs.idxmax()
    ax.annotate(f"…of which backed by a wealth source that year\n(peak: {obs.max()} countries in {peak})",
                xy=(peak, obs.max()), xytext=(1986, 70), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_ylim(0, 240)
    ax.set_xlim(1980, 2024)
    ax.set_ylabel("Countries and territories")
    ax.set_title("WID publishes wealth shares everywhere; it observes them in a few dozen places")
    fig.tight_layout()
    save(fig, "fig4_coverage",
         "Data: World Inequality Database bulk download (Nov 2026), shweal992j p99p100; "
         "'backed by a source' = data_quality > 0.")


def fig5_spec_curve() -> None:
    s = pd.read_csv(RES / "panel_specs.csv").sort_values("beta").reset_index(drop=True)
    x = np.arange(len(s))
    col = np.where(s["sign"] == "negative", BLUE, np.where(s["sign"] == "positive", ORANGE, MUTED))
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.06})
    ax.vlines(x, s.lo, s.hi, colors=col, lw=0.6, alpha=0.45)
    ax.scatter(x, s.beta, c=col, s=7, zorder=3, linewidths=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-2.2, 3.2)
    ax.set_ylabel("Δ annual growth (pp)\nper 1 SD more concentration")
    n_neg, n_pos = int((s["sign"] == "negative").sum()), int((s["sign"] == "positive").sum())
    ax.set_title(f"{len(s)} specifications: {n_neg} significantly negative, {n_pos} significantly positive, "
                 f"{len(s) - n_neg - n_pos} not distinguishable from zero")
    ax.text(5, 2.6, "95% interval excludes zero, negative", color=BLUE, fontsize=9)
    ax.text(5, 2.25, "95% interval excludes zero, positive", color=ORANGE, fontsize=9)
    ax.text(5, 1.9, "interval includes zero", color=MUTED, fontsize=9)
    rows = [
        ("estimator", "between", "Between countries"), ("estimator", "pooled", "Pooled, period effects"),
        ("estimator", "within", "Country fixed effects"),
        ("sample", "all", "All WID areas"), ("sample", "observed", "Source-based start values"),
        ("sample", "high_income", "High-income only"),
        ("measure", "wealth", "Wealth share"), ("measure", "income", "Income share"),
        ("outcome", "g_tfp", "TFP growth"), ("window", 10, "10-year windows"),
    ]
    for i, (k, v, label) in enumerate(rows):
        yy = len(rows) - i
        if k == "measure":
            hit = s[k].str.startswith(v)
        else:
            hit = s[k] == v
        bx.scatter(x[hit], np.full(hit.sum(), yy), s=5, color=INK2, linewidths=0)
        bx.text(-8, yy, label, ha="right", va="center", fontsize=8.5, color=INK2)
    bx.set_yticks([])
    bx.set_ylim(0.3, len(rows) + 0.7)
    bx.grid(False)
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("Specifications, sorted by estimate")
    bx.set_xlim(-2, len(s) + 2)
    save(fig, "fig5_spec_curve",
         "Data: WID (top shares at the start of each window), Penn World Table 11.0 (growth 1980-2019). "
         "Intervals: country-clustered (CR1) or HC1 (between). Source: analysis.py.")


def fig6_billionaires() -> None:
    b = pd.read_csv(RES / "billionaire_specs.csv")
    c = b[b.model == "composition"].copy()
    lab = {"g_gdppc": "GDP pc", "g_tfp": "TFP"}
    smp = {"all": "all countries", "with_billionaires": "only with billionaires", "no_offshore": "no financial centres"}
    c["label"] = [f"{lab[o]}, {h}y, {smp[sm]}" for o, h, sm in zip(c.outcome, c.horizon, c["sample"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, tr, title, unit in ((axes[0], "log1p", "Wealth entered as log(1 + % of GDP)", "pp per log point"),
                                (axes[1], "raw", "Wealth entered as % of GDP", "pp per 1% of GDP")):
        d = c[c["transform"] == tr].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        ax.axvline(0, color=INK2, lw=0.8)
        ax.hlines(y, d.diff_lo, d.diff_hi, color=BLUE, lw=2)
        ax.scatter(d.diff_inh_minus_self, y, color=BLUE, s=40, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(d.label, fontsize=8.5)
        ax.set_title(title)
        ax.set_xlabel(f"Inherited minus self-made coefficient ({unit})")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Inherited billionaire wealth predicts slower growth than self-made wealth in every specification, "
                 "and in none of them clearly", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_billionaires",
         "Data: Billionaire Characteristics Database (Freund & Oliver 2016) for 1996, 2001, 2014; WDI GDP; PWT 11.0. "
         "95% country-cluster bootstrap intervals (2,000 draws). Source: analysis.py.")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig1_static_surface()
    fig2_steady_states()
    fig3_reranking()
    fig4_coverage()
    fig5_spec_curve()
    fig6_billionaires()


if __name__ == "__main__":
    main()
