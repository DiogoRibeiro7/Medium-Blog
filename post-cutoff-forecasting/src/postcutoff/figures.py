"""Figures for the article, drawn only from files in results/ and data/."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .data import load_snapshot, mask_hours  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WINDOW_WASH = "#f0efec"
# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode) for the
# foundation models; the classical models are drawn in ink greys.
COLOURS = {
    "chronos2": "#2a78d6",
    "timesfm25": "#eb6834",
    "tirex": "#1baf7a",
    "mstl_ets": INK2,
    "ets": MUTED,
    "seasonal_naive_week": AXIS,
    "seasonal_naive_day": AXIS,
}
# ETS with a daily season only is far behind the others on hourly load; it is reported in
# RESULTS.md and left out of the figures so that the scale shows the comparison that matters.
FIGURE_MODELS = ["chronos2", "timesfm25", "tirex", "mstl_ets"]
LOCAL = "Europe/Lisbon"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "legend.frameon": False,
            "legend.labelcolor": INK2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def _save(fig, cfg: Config, name: str):
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.figures_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _summary(cfg: Config) -> dict:
    return json.loads((cfg.results_dir / "summary.json").read_text(encoding="utf-8"))


def _models_present(summary: dict) -> list[str]:
    return [m for m in FIGURE_MODELS if m in summary["models"]]


def fig_series(cfg: Config):
    series = load_snapshot(cfg)
    hourly = series.copy()
    hourly[mask_hours(series.index, cfg)] = np.nan  # the blackout hours are not load
    daily = hourly.tz_convert(LOCAL).resample("D").mean()
    values = daily.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    top = np.nanmax(values) * 1.04
    base = np.nanmin(values)
    for w in cfg.windows:
        start = w.first_origin.tz_convert(LOCAL)
        end = (w.last_origin + pd.Timedelta(hours=cfg.horizon)).tz_convert(LOCAL)
        ax.axvspan(start, end, color=WINDOW_WASH, lw=0, zorder=0)
        ax.text(start + (end - start) / 2, top, w.name, ha="center", va="bottom", color=INK2, fontsize=9)
    ax.plot(daily.index, values, color="#2a78d6", lw=1.0)
    labels = {"tirex": "TiRex", "timesfm25": "TimesFM 2.5", "chronos2": "Chronos-2"}
    releases = sorted(cfg.releases.items(), key=lambda kv: kv[1])
    # Release labels sit in a band below the data, each joined to its line by a leader.
    text_x = releases[0][1].tz_convert(LOCAL) - pd.Timedelta(days=200)
    for k, (model, date) in enumerate(releases):
        d = date.tz_convert(LOCAL)
        y = base * (0.95 - 0.05 * k)
        ax.vlines(d, y, top, color=INK2, lw=0.8)
        ax.hlines(y, text_x + pd.Timedelta(days=8), d, color=MUTED, lw=0.6)
        ax.text(
            text_x,
            y,
            f"{labels.get(model, model)} released {date:%d %b %Y}",
            ha="right",
            va="center",
            color=INK2,
            fontsize=8,
        )
    ax.set_ylim(base * 0.82, top * 1.07)
    ax.set_ylabel("Daily mean load (MW)")
    ax.set_title("Portuguese electricity load and the four evaluation windows")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", visible=False)
    return _save(fig, cfg, "fig1_series.png")


def fig_relative(cfg: Config):
    s = _summary(cfg)
    models = _models_present(s)
    windows = s["windows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    offsets = np.linspace(-0.24, 0.24, len(models))
    for ax, horizon in zip(axes, ["day-ahead", "week-ahead"], strict=False):
        rel = pd.DataFrame(s["relative_mae"][horizon]["vs_reference"])
        for off, m in zip(offsets, models, strict=False):
            r = rel[(rel["model"] == m) & (rel["windows"].isin(windows))].set_index("windows").loc[windows]
            x = np.arange(len(windows)) + off
            lo = [c[0] for c in r["ci95"]]
            hi = [c[1] for c in r["ci95"]]
            ax.vlines(x, lo, hi, color=COLOURS[m], lw=1.5)
            ax.scatter(
                x,
                r["ratio"],
                s=36,
                color=COLOURS[m],
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
                label=s["models"][m],
            )
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.set_xticks(np.arange(len(windows)), windows)
        after_idx = [windows.index(w) for w in s["after"]]
        if after_idx:
            ax.axvline(min(after_idx) - 0.5, color=AXIS, lw=0.8)
            ax.text(
                min(after_idx) - 0.45,
                0.98,
                "after all\nreleases",
                transform=ax.get_xaxis_transform(),
                color=INK2,
                fontsize=8,
                va="top",
            )
        ax.set_title(f"{horizon.capitalize()} forecasts")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAE relative to seasonal naive (week)\n(below 1 = better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    return _save(fig, cfg, "fig2_relative_mae.png")


def fig_change(cfg: Config):
    s = _summary(cfg)
    models = [m for m in _models_present(s) if m != s["reference_model"]]
    ref_label = s["models"][s["reference_model"]]
    panels = [("mae_vs_reference", f"relative to {ref_label[0].lower() + ref_label[1:]}")]
    if "mae_vs_mstl_ets" in s["change"]["day-ahead"]:
        panels.append(("mae_vs_mstl_ets", "relative to MSTL + ETS"))
    fig, axes = plt.subplots(
        len(panels), 2, figsize=(10, 1.2 + 0.55 * len(models) * len(panels)), sharex=True, squeeze=False
    )
    for i, (key, title) in enumerate(panels):
        for j, horizon in enumerate(["day-ahead", "week-ahead"]):
            ax = axes[i, j]
            rows = {r["model"]: r for r in s["change"][horizon][key]}
            placebo = {r["model"]: r for r in s["change"][horizon].get("placebo_" + key, [])}
            shown = [m for m in models if m in rows]
            y = np.arange(len(shown))[::-1]
            for yy, m in zip(y, shown, strict=True):
                r = rows[m]
                ax.hlines(yy + 0.12, r["ci95"][0], r["ci95"][1], color=COLOURS[m], lw=1.5)
                ax.scatter(
                    r["change_pct"],
                    yy + 0.12,
                    s=36,
                    color=COLOURS[m],
                    edgecolor=SURFACE,
                    linewidth=1.5,
                    zorder=3,
                )
                if m in placebo:
                    p = placebo[m]
                    ax.hlines(yy - 0.18, p["ci95"][0], p["ci95"][1], color=AXIS, lw=1.5)
                    ax.scatter(
                        p["change_pct"],
                        yy - 0.18,
                        s=30,
                        facecolor=SURFACE,
                        edgecolor=MUTED,
                        linewidth=1.5,
                        zorder=3,
                    )
            ax.set_ylim(-0.6, len(shown) - 0.4)
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_yticks(y, [s["models"][m] for m in shown])
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{horizon.capitalize()}, MAE {title}", fontsize=10)
            if i == len(panels) - 1:
                ax.set_xlabel("Change in relative MAE (%)")
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            color=INK2,
            marker="o",
            ms=6,
            mec=SURFACE,
            lw=1.5,
            label="before to after every release (model colour)",
        ),
        Line2D(
            [], [], color=AXIS, marker="o", ms=6, mfc=SURFACE, mec=MUTED, lw=1.5, label="placebo: 2022-23 to 2023-24"
        ),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(0.01, 0.965), fontsize=9)
    fig.suptitle(
        "Change in relative MAE from the windows the models could have seen to the window after every release",
        x=0.01,
        ha="left",
        color=INK,
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, cfg, "fig3_change.png")


def fig_lead_day(cfg: Config):
    s = _summary(cfg)
    models = _models_present(s) + [s["reference_model"]]
    lead = pd.DataFrame(s["lead_days"])
    windows = list(s["after"])
    fig, ax = plt.subplots(figsize=(8, 4))
    sub = lead[lead["window"].isin(windows)]
    for m in models:
        r = sub[sub["model"] == m].groupby("lead_day")["mae"].mean()
        if r.empty:
            continue
        foundation_model = m in ("chronos2", "timesfm25", "tirex")
        ax.plot(
            r.index,
            r.to_numpy(),
            color=COLOURS[m],
            lw=2 if foundation_model else 1.5,
            marker="o",
            ms=4,
            mec=SURFACE,
            label=s["models"][m],
        )
    ax.legend(loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, 1.13))
    ax.set_xlim(0.7, 7.3)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Lead day")
    ax.set_ylabel("MAE (MW)")
    ax.set_title(f"Error by lead day, window {', '.join(windows)}", pad=28)
    ax.grid(axis="x", visible=False)
    return _save(fig, cfg, "fig4_lead_day.png")


def fig_example(cfg: Config):
    path = cfg.results_dir / "examples.csv"
    if not path.exists():
        return None
    ex = pd.read_csv(path)
    s = _summary(cfg)
    after_first = min(cfg.window(w).first_origin for w in s["after"])
    origins = [o for o in dict.fromkeys(ex["origin"]) if pd.Timestamp(o) >= after_first]
    if not origins:
        return None
    origin = origins[0]
    e = ex[ex["origin"] == origin].copy()
    e["t"] = pd.to_datetime(e["timestamp_utc"], utc=True).dt.tz_convert(LOCAL)
    panel_models = [m for m in ["chronos2", "timesfm25", "tirex", "mstl_ets"] if m in set(e["model"])]
    hist = e[e["model"] == "history"].tail(72)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), panel_models, strict=False):
        f = e[e["model"] == m]
        hol = f[f["holiday"] > 0]
        if not hol.empty:
            ax.axvspan(hol["t"].min(), hol["t"].max() + pd.Timedelta(hours=1), color=WINDOW_WASH, lw=0, zorder=0)
        ax.fill_between(f["t"], f["q10"], f["q90"], color=COLOURS[m], alpha=0.15, lw=0)
        ax.plot(hist["t"], hist["actual"], color=INK, lw=1.2)
        ax.plot(f["t"], f["actual"], color=INK, lw=1.2, label="Actual")
        ax.plot(f["t"], f["q50"], color=COLOURS[m], lw=1.6, label="Median forecast")
        ax.axvline(f["t"].min(), color=AXIS, lw=0.8)
        ax.set_title(s["models"][m], fontsize=10)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2, tz=LOCAL))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=LOCAL))
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Load (MW)")
    day = pd.Timestamp(origin).tz_convert(LOCAL).strftime("%d %B %Y")
    fig.suptitle(
        f"One week from {day}: actual load (black), median and 80% band; public holidays shaded",
        x=0.01,
        ha="left",
        color=INK,
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, cfg, "fig5_example_week.png")


def make_all(cfg: Config) -> list:
    _style()
    out = [fig_series(cfg), fig_relative(cfg), fig_change(cfg), fig_lead_day(cfg), fig_example(cfg)]
    return [p for p in out if p is not None]
