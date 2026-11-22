"""Figures for the article, drawn from results/summary.json and results/figure_data.json.

Colour carries the model's origin only (Europe, United States, China; the operators'
forecast in grey), in the same order everywhere. Every series is also named on the chart.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import BLOCS, Config  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLOC = {"Europe": "#2a78d6", "United States": "#eb6834", "China": "#1baf7a", "Reference": "#8a8984"}
STRATUM_LABELS = {
    "ordinary_workday": "Ordinary workday",
    "ordinary_weekend": "Ordinary weekend",
    "adjacent": "Next to a national holiday",
    "regional_holiday": "Regional holiday",
    "optional_holiday": "Optional or bank holiday",
    "national_holiday": "National holiday",
    "year_end": "24 Dec – 1 Jan",
    "informative": "Informative days",
}
SOURCE = "Data: energy-charts.info (Fraunhofer ISE), Open-Meteo.com, CC BY 4.0. Analysis: Diogo Ribeiro."


HATCHES = ("", "///", "...", "xxx")
LINESTYLES = ("-", "--", "-.", ":")


def _within_bloc(blocs: list[str]) -> list[int]:
    """Position of each item among the items of the same bloc, for a second encoding."""
    seen: dict[str, int] = {}
    out = []
    for b in blocs:
        out.append(seen.get(b, 0))
        seen[b] = seen.get(b, 0) + 1
    return out


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 9.5,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _finish(fig, path, top: float = 1.0):
    """Lay out, add the source line below everything, save and close."""
    fig.tight_layout(rect=(0, 0.05, 1, top))
    fig.text(0.01, 0.012, SOURCE, fontsize=7.5, color=INK_2, ha="left", va="bottom")
    fig.savefig(path)
    plt.close(fig)


def _pct(ax, axis="x"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%" if v >= 0.1 else f"{100 * v:.1f}%")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def fig_average_vs_tail(s: dict, path):
    fc = s["forecasting"]
    runs = [*fc["headline_runs"], *(["tso|reference"] if "tso|reference" in fc["runs"] else [])]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for run in runs:
        r = fc["runs"][run]
        color = BLOC[r["bloc"]]
        marker = "D" if run == "tso|reference" else "o"
        ax.scatter(
            r["wape"],
            r["cvar95_daily_wape"],
            s=70,
            color=color,
            marker=marker,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        label = "Operator day-ahead" if run == "tso|reference" else r["label"]
        ax.annotate(
            label,
            (r["wape"], r["cvar95_daily_wape"]),
            xytext=(7, 4),
            textcoords="offset points",
            color=INK,
            fontsize=8.5,
        )
    _pct(ax, "x")
    _pct(ax, "y")
    ax.set_xlabel("WAPE over all days (lower is better)")
    ax.set_ylabel("Mean daily WAPE on the worst 5% of days")
    ax.set_title("Average error and tail error do not rank the models the same way")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=b) for b, c in BLOC.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    _finish(fig, path)


def fig_text_step(s: dict, path):
    lab = s["labels"]
    llms = sorted(lab["llms"].items(), key=lambda kv: kv[1]["accuracy_informative"] or 0)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(llms) + 2.2))
    for i, (_, r) in enumerate(llms):
        color = BLOC[r["bloc"]]
        a, b = r["accuracy"], r["accuracy_informative"]
        lo, hi = r["accuracy_informative_ci"]
        if lo is not None:
            ax.plot([lo, hi], [i, i], color=color, lw=1, alpha=0.6, zorder=2)
        ax.plot([b, a], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([a], [i], s=60, facecolor=SURFACE, edgecolor=color, linewidth=2, zorder=3)
        ax.scatter([b], [i], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    for name, style in (("weekday_rule", ":"), ("flag_rule", "--")):
        v = lab["rules"][name]["accuracy_informative"]
        ax.axvline(v, color=INK_2, ls=style, lw=1, zorder=0)
        ax.text(v, -0.9, " " + name.replace("_", " "), color=INK_2, fontsize=8, ha="left", va="center")
    ax.set_ylim(-1.3, len(llms) - 0.5)
    ax.set_yticks(range(len(llms)), [r["label"] for _, r in llms])
    ax.set_xlim(-0.02, 1.02)
    _pct(ax, "x")
    ax.set_xlabel("Share of days labelled as demand turned out")
    ax.set_title("On the days that matter, the labels are much less reliable")
    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK_2, label="All days asked"),
        plt.Line2D([], [], marker="o", ls="", color=INK_2, label="Informative days (95% interval)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    _finish(fig, path)


MARKERS = ("o", "s", "^", "D")


def fig_strata(s: dict, path):
    """One row per kind of day, one dot per model: colour for the origin, shape within it."""
    fc = s["forecasting"]
    runs = sorted(fc["headline_runs"], key=lambda r: fc["runs"][r]["wape"])
    if "tso|reference" in fc["runs"]:
        runs.append("tso|reference")
    strata = [
        "ordinary_workday",
        "ordinary_weekend",
        "adjacent",
        "regional_holiday",
        "optional_holiday",
        "national_holiday",
        "year_end",
    ]
    strata = [st for st in strata if any(fc["by_stratum"][r][st].get("days") for r in runs)]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(strata) + 2.4))
    within = _within_bloc([fc["runs"][r]["bloc"] for r in runs])
    for y in range(len(strata)):
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
    for run, w in zip(runs, within, strict=True):
        r = fc["runs"][run]
        vals = [fc["by_stratum"][run][st].get("wape", np.nan) for st in strata]
        label = "Operator day-ahead" if run == "tso|reference" else r["label"]
        ax.scatter(
            vals,
            range(len(strata)),
            s=46,
            color=BLOC[r["bloc"]],
            marker=MARKERS[w % len(MARKERS)],
            edgecolor=SURFACE,
            linewidth=1.5,
            label=label,
            zorder=3,
        )
    ax.set_yticks(range(len(strata)), [STRATUM_LABELS[st] for st in strata])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _pct(ax, "x")
    ax.set_xlabel("WAPE")
    ax.set_title("Error by kind of day")
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    _finish(fig, path)


def fig_propagation(s: dict, cfg: Config, path):
    prop = s.get("propagation", {})
    forecasters = [f for f in prop if prop[f]]
    if not forecasters:
        return
    fig, axes = plt.subplots(1, len(forecasters), figsize=(3.4 * len(forecasters), 4.2), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], forecasters, strict=True):
        rows = prop[f]
        # Fixed order: no calendar, the flag, each language model by bloc, the ex-post truth.
        llm_order = sorted(cfg.llms, key=lambda k: (BLOCS.index(cfg.llms[k].bloc), k))
        order = ["none", "temp", "flag", *(f"llm:{k}" for k in llm_order), "oracle"]
        names = [v for v in order if v in rows]
        vals = [rows[v]["informative_wape"] for v in names]
        colors, hatches = [], []
        within = _within_bloc([cfg.llms[v[4:]].bloc if v.startswith("llm:") else "Reference" for v in names])
        for v, w in zip(names, within, strict=True):
            colors.append(BLOC[cfg.llms[v[4:]].bloc] if v.startswith("llm:") else BLOC["Reference"])
            hatches.append(HATCHES[w % len(HATCHES)] if v.startswith("llm:") else "")
        bars = ax.barh(range(len(names)), vals, color=colors, edgecolor=SURFACE, linewidth=0.5)
        for bar, h in zip(bars, hatches, strict=True):
            bar.set_hatch(h)
        labels = [
            "+ " + cfg.llms[v[4:]].label
            if v.startswith("llm:")
            else {
                "none": "load only",
                "temp": "+ temperature",
                "flag": "+ national holiday flag",
                "oracle": "+ true day type (ex post)",
            }.get(v, v)
            for v in names
        ]
        ax.set_yticks(range(len(names)), labels)
        ax.invert_yaxis()
        _pct(ax, "x")
        ax.set_title(cfg.forecasters[f].label, fontsize=10)
        ax.set_xlabel("WAPE on informative days")
    fig.suptitle(
        "Where the day type comes from decides the error on the days that matter",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    _finish(fig, path, top=0.93)


def fig_examples(data: dict, cfg: Config, path):
    examples = [e for e in data.get("examples", []) if e["timestamps"]]
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 1, figsize=(7.4, 2.6 * len(examples)), squeeze=False)
    for ax, e in zip(axes[:, 0], examples, strict=True):
        t = pd.to_datetime(e["timestamps"], utc=True).tz_convert(cfg.countries[e["country"]].tz)
        for name, values in e["series"].items():
            v = np.array([np.nan if x is None else x for x in values], dtype=float) / 1000
            if name == "actual":
                ax.plot(t, v, color=INK, lw=2, label="Actual load", zorder=4)
            elif name.startswith("Operator"):
                ax.plot(t, v, color=BLOC["Reference"], lw=1.6, ls="--", label=name, zorder=3)
            else:
                f = next((k for k, sp in cfg.forecasters.items() if name.startswith(sp.label)), None)
                color = BLOC[cfg.forecasters[f].bloc] if f else INK_2
                ax.plot(t, v, color=color, lw=1.6, label=name, zorder=2)
        ax.set_ylabel("GW")
        ax.set_title(f"{cfg.countries[e['country']].name}, week from {e['start']}", fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d %b", tz=t.tz))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8, bbox_to_anchor=(0.5, 1.0))
    _finish(fig, path, top=0.95)


def fig_daily_tail(data: dict, s: dict, path):
    daily = data.get("daily_wape_headline", {})
    if not daily:
        return
    fc = s["forecasting"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    within = _within_bloc([fc["runs"][r]["bloc"] for r in daily])
    for (run, values), w in zip(daily.items(), within, strict=True):
        v = np.sort(np.array(values, dtype=float))
        share = 1 - np.arange(1, v.size + 1) / v.size
        r = fc["runs"][run]
        ls = LINESTYLES[w % len(LINESTYLES)]
        label = "Operator day-ahead" if run == "tso|reference" else r["label"]
        ax.plot(v, share, color=BLOC[r["bloc"]], lw=1.6, ls=ls, label=label)
    ax.set_yscale("log")
    ax.set_ylim(1 / max(len(v) for v in daily.values()), 1)
    _pct(ax, "x")
    ax.set_xlabel("Daily WAPE")
    ax.set_ylabel("Share of country-days with a larger error")
    ax.set_title("The upper tail of daily errors")
    ax.legend(fontsize=7.5)
    _finish(fig, path)


def draw(cfg: Config) -> list[str]:
    _style()
    s = json.loads((cfg.results_dir / "summary.json").read_text(encoding="utf-8"))
    data_path = cfg.results_dir / "figure_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8")) if data_path.exists() else {}
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    out = []
    if s.get("forecasting"):
        fig_average_vs_tail(s, cfg.figures_dir / "fig1_average_vs_tail.png")
        fig_strata(s, cfg.figures_dir / "fig3_error_by_kind_of_day.png")
        fig_propagation(s, cfg, cfg.figures_dir / "fig4_where_the_day_type_comes_from.png")
        fig_examples(data, cfg, cfg.figures_dir / "fig5_example_weeks.png")
        fig_daily_tail(data, s, cfg.figures_dir / "fig6_daily_error_tail.png")
    if s.get("labels", {}).get("llms"):
        fig_text_step(s, cfg.figures_dir / "fig2_text_step.png")
    out = sorted(p.name for p in cfg.figures_dir.glob("*.png"))
    return out
