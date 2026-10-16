"""Figures for the article, drawn from results/ only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 9.5,
            "lines.linewidth": 2,
        }
    )
    return plt


def _source(fig, text="Source: Hansard Speeches 1979-2021 (Odell, v3.1.0). Author's analysis."):
    fig.text(0.01, 0.005, text, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def _save(fig, cfg: Config, name: str) -> str:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.figures_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path.name


def fig_naive_by_year(cfg: Config) -> str | None:
    p = cfg.stage1_dir / "naive_by_year.csv"
    if not p.exists():
        return None
    plt = _plt()
    df = pd.read_csv(p)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True, gridspec_kw={"hspace": 0.32})
    for ax, col, color, title in (
        (axes[0], "discipline_rate", BLUE, "Chair speeches the dictionary tags as discipline"),
        (axes[1], "timekeeping_rate", ORANGE, "Chair speeches the dictionary tags as time-keeping"),
    ):
        ax.plot(df["year"], df[col], color=color, lw=2)
        ax.set_title(title)
        ax.set_ylabel("per 1,000 Member speeches")
        ax.set_ylim(bottom=0)
        for e in cfg.eras[1:]:
            ax.axvline(e.start.year + (e.start.timetuple().tm_yday - 1) / 365, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    for e in cfg.eras:
        mid = (e.start.year + e.end.year) / 2
        axes[0].text(mid, axes[0].get_ylim()[1] * 0.97, e.name, ha="center", va="top", fontsize=8, color=INK_2)
    _source(fig)
    name = _save(fig, cfg, "fig1_naive_by_year.png")
    plt.close(fig)
    return name


def fig_naive_vs_corrected(cfg: Config) -> str | None:
    p = cfg.stage2_dir / "corrected_by_era.csv"
    if not p.exists():
        return None
    plt = _plt()
    df = pd.read_csv(p)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.errorbar(
        x + 0.12,
        df["rate_median"],
        yerr=[df["rate_median"] - df["rate_lo"], df["rate_hi"] - df["rate_median"]],
        fmt="o",
        color=BLUE,
        ms=7,
        elinewidth=2,
        capsize=0,
        label="corrected for the dictionary's errors (95% interval)",
    )
    ax.plot(x - 0.12, df["naive_rate"], "o", color=ORANGE, ms=7, label="dictionary count (naive)")
    ax.set_xticks(x, [f"{e}{'*' if p_ else ''}" for e, p_ in zip(df["era"], df["partial"], strict=True)])
    ax.set_ylabel("disciplinary interventions\nper 1,000 Member speeches")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.set_title("What the dictionary counts and what it is estimated to miss or invent")
    ax.legend(loc="upper right")
    _source(
        fig,
        "* Hoyle: Nov 2019 to Apr 2021 only. Source: Hansard Speeches 1979-2021 (Odell, v3.1.0); hand-labelled validation sample.",
    )
    name = _save(fig, cfg, "fig2_naive_vs_corrected.png")
    plt.close(fig)
    return name


def fig_precision_recall(cfg: Config) -> str | None:
    p = cfg.stage2_dir / "corrected_by_era.csv"
    if not p.exists():
        return None
    plt = _plt()
    df = pd.read_csv(p)
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for ax, key, title in ((axes[0], "precision", "Precision"), (axes[1], "recall", "Recall")):
        ax.errorbar(
            x,
            df[f"{key}_median"],
            yerr=[df[f"{key}_median"] - df[f"{key}_lo"], df[f"{key}_hi"] - df[f"{key}_median"]],
            fmt="o",
            color=BLUE,
            ms=7,
            elinewidth=2,
            capsize=0,
        )
        ax.set_xticks(x, df["era"], rotation=30, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    axes[0].set_ylabel("share, with 95% interval")
    fig.suptitle("The same dictionary, measured era by era", x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    _source(fig)
    name = _save(fig, cfg, "fig3_precision_recall.png")
    plt.close(fig)
    return name


def fig_decomposition(cfg: Config) -> str | None:
    p = cfg.stage2_dir / "decomposition.csv"
    if not p.exists():
        return None
    plt = _plt()
    df = pd.read_csv(p)
    ref = df["reference"].iloc[0]
    df = df[df["era"] != ref].reset_index(drop=True)
    y = np.arange(len(df))[::-1]
    h = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 0.75 * len(df) + 1.6))
    parts = (
        ("true", BLUE, "change in the true rate"),
        ("recall", ORANGE, "recall drift"),
        ("precision", AQUA, "precision drift"),
    )
    for k, (key, color, label) in enumerate(parts):
        yy = y + (1 - k) * h
        med = df[f"{key}_median"]
        ax.barh(yy, med, height=h * 0.86, color=color, label=label)
        ax.errorbar(
            med, yy, xerr=[med - df[f"{key}_lo"], df[f"{key}_hi"] - med], fmt="none", ecolor=INK_2, elinewidth=1
        )
    ax.scatter(
        df["naive_log_ratio"],
        y,
        marker="|",
        s=420,
        color=INK,
        linewidths=2.2,
        zorder=5,
        label="dictionary change (sum)",
    )
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks(y, df["era"])
    ticks = np.array([0.33, 0.5, 0.67, 1, 1.5, 2, 3])
    lo, hi = ax.get_xlim()
    ticks = ticks[(np.log(ticks) >= lo) & (np.log(ticks) <= hi)]
    ax.set_xticks(np.log(ticks), [f"×{t:g}" for t in ticks])
    ax.set_xlabel(f"ratio to the {ref} era (log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Where the change in the dictionary count comes from, relative to {ref}", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8.5, handletextpad=0.4, columnspacing=1.2)
    _source(fig)
    name = _save(fig, cfg, "fig4_decomposition.png")
    plt.close(fig)
    return name


def fig_sides(cfg: Config) -> str | None:
    p = cfg.stage2_dir / "sides.csv"
    if not p.exists():
        return None
    plt = _plt()
    df = pd.read_csv(p)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.errorbar(
        x + 0.12,
        df["ratio_opp_gov_median"],
        yerr=[df["ratio_opp_gov_median"] - df["ratio_opp_gov_lo"], df["ratio_opp_gov_hi"] - df["ratio_opp_gov_median"]],
        fmt="o",
        color=BLUE,
        ms=7,
        elinewidth=2,
        capsize=0,
        label="corrected (95% interval)",
    )
    ax.plot(x - 0.12, df["naive_ratio_opp_gov"], "o", color=ORANGE, ms=7, label="dictionary count")
    ax.axhline(1, color=INK_2, lw=0.8)
    ax.set_xticks(x, df["era"])
    ax.set_ylabel("Opposition rate ÷ Government rate")
    ax.grid(axis="x", visible=False)
    ax.set_title("How much more often the Chair intervened when the Opposition had the floor")
    ax.legend(loc="upper right")
    _source(fig)
    name = _save(fig, cfg, "fig5_sides.png")
    plt.close(fig)
    return name


def fig_transfer(cfg: Config) -> str | None:
    p = cfg.stage2_dir / "transfer.csv"
    if not p.exists():
        return None
    plt = _plt()
    df = pd.read_csv(p)
    eras = list(dict.fromkeys(df["era"]))
    methods = (
        ("dictionary", BLUE, "dictionary"),
        ("model_within_eras", ORANGE, "classifier, trained on all eras"),
        ("model_leave_era_out", AQUA, "classifier, era left out of training"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    x = np.arange(len(eras))
    for ax, key, title in ((axes[0], "precision", "Precision"), (axes[1], "recall", "Recall")):
        for k, (m, color, label) in enumerate(methods):
            sub = df[df["method"] == m].set_index("era").reindex(eras)
            xx = x + (k - 1) * 0.22
            ax.errorbar(
                xx,
                sub[key],
                yerr=[sub[key] - sub[f"{key}_lo"], sub[f"{key}_hi"] - sub[key]],
                fmt="o",
                color=color,
                ms=6,
                elinewidth=1.6,
                capsize=0,
                label=label,
            )
        ax.set_xticks(x, eras, rotation=30, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    axes[0].set_ylabel("population-weighted share")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    _source(fig)
    name = _save(fig, cfg, "fig6_transfer.png")
    plt.close(fig)
    return name


def fig_lexical(cfg: Config) -> str | None:
    p = cfg.stage1_dir / "lexical_jsd.csv"
    if not p.exists():
        return None
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    m = pd.read_csv(p, index_col=0)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(m.to_numpy(), cmap=cmap, vmin=0)
    ax.set_xticks(range(len(m)), m.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(m)), m.index)
    ax.grid(False)
    for i in range(len(m)):
        for j in range(len(m)):
            v = m.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if v > 0.6 * m.to_numpy().max() else INK,
            )
    ax.set_title("How far apart the Chair's vocabularies are\n(Jensen-Shannon divergence, bits)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _source(fig)
    name = _save(fig, cfg, "fig7_lexical_jsd.png")
    plt.close(fig)
    return name


def all_figures(cfg: Config) -> list[str]:
    out = []
    for f in (
        fig_naive_by_year,
        fig_naive_vs_corrected,
        fig_precision_recall,
        fig_decomposition,
        fig_sides,
        fig_transfer,
        fig_lexical,
    ):
        name = f(cfg)
        if name:
            out.append(name)
    return out
