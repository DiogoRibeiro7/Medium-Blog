"""Figures for the article. Each is drawn only if its data are present."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analyse, nhanes
from .paths import DATA, FIGURES

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CREDIT = "Diogo Ribeiro · data: {src}"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _style(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def _save(fig, name, src):
    fig.text(0.01, 0.01, CREDIT.format(src=src), fontsize=7.5, color=MUTED)
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def _plain_log(axis):
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=9, **kw)
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


def fig_sbp_by_age(df: pd.DataFrame):
    d = analyse.nhanes_frame(df)
    base = (d["w_osc"] > 0) & ~d["pregnant"].fillna(False).astype(bool) & d["sbp"].notna()
    x = np.linspace(80, 200, 481)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, (name, _, _) in enumerate(analyse.AGE_BANDS):
        g = d[base & (d["band"] == name)]
        w = g["w_osc"].to_numpy() / g["w_osc"].sum()
        bw = 4.0
        dens = (w[None, :] * np.exp(-0.5 * ((x[:, None] - g["sbp"].to_numpy()[None, :]) / bw) ** 2)).sum(1)
        dens /= bw * np.sqrt(2 * np.pi)
        ax.plot(x, dens, color=SERIES[i], lw=2, label=f"{name}")
    for thr in (130, 140):
        ax.axvline(thr, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(thr + 1, ax.get_ylim()[1] * 0.97, f"{thr}", fontsize=8.5, color=INK2, va="top")
    _style(ax)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(80, 200)
    ax.set_xlabel("systolic blood pressure, mean of up to three readings (mmHg)")
    _legend(ax, title="age", title_fontsize=8.5, loc="upper right")
    ax.set_title("Systolic blood pressure by age, US adults 2017–2023", loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.17)
    _save(fig, "fig1_sbp_by_age.png", "NHANES 2017–March 2020 and 2021–2023, survey-weighted")


def fig_crossing(cross: pd.DataFrame, params: dict):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(len(cross))
    w = 0.36
    ax.bar(xs - w / 2 - 0.01, cross["average_only"], width=w, color=SERIES[0],
           label=f"everyone responds by the average ({params['mu']:+.1f} mmHg)")
    ax.bar(xs + w / 2 + 0.01, cross["hetero"], width=w, color=SERIES[1],
           label=f"responses vary between people (SD {params['tau']:.1f} mmHg)")
    for i, r in cross.iterrows():
        for off, v in ((-w / 2 - 0.01, r["average_only"]), (w / 2 + 0.01, r["hetero"])):
            ax.text(i + off, v + 0.002, f"{100 * v:.1f}%", ha="center", va="bottom", fontsize=8, color=INK2)
    _style(ax)
    ax.set_xticks(xs, cross["band"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("age")
    ax.set_ylabel("share of the age group moved to 130 mmHg or above")
    _legend(ax, loc="upper left")
    ax.set_title("Who a stimulant would push into the hypertensive range", loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.17)
    _save(fig, "fig2_threshold_crossing.png", "NHANES 2017–2023 baselines; response from pooled trials")


def fig_forest(vr: pd.DataFrame, summary: pd.DataFrame):
    kinds = [k for k in ("sbp", "hr") if (vr["kind"] == k).any()]
    if not kinds:
        return
    labels = {"sbp": "systolic blood pressure", "hr": "heart rate"}
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.2, 0.28 * max((vr["kind"] == k).sum() for k in kinds) + 2.2),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, kinds):
        g = vr[vr["kind"] == k].sort_values("ln_vr").reset_index(drop=True)
        y = np.arange(len(g))
        se = np.sqrt(g["ln_vr_var"])
        ax.errorbar(np.exp(g["ln_vr"]), y, xerr=[np.exp(g["ln_vr"]) - np.exp(g["ln_vr"] - 1.96 * se),
                                               np.exp(g["ln_vr"] + 1.96 * se) - np.exp(g["ln_vr"])],
                    fmt="o", color=SERIES[0], ms=4, lw=1, capsize=0)
        s = summary[summary["kind"] == k].iloc[0]
        ax.errorbar([s["vr"]], [-1.5], xerr=[[s["vr"] - s["vr_lo"]], [s["vr_hi"] - s["vr"]]],
                    fmt="D", color=SERIES[1], ms=7, lw=2, capsize=0)
        ax.axvline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xscale("log")
        _plain_log(ax.xaxis)
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(0.5, 0.7, 1.0, 1.4, 2.0)))
        ax.set_yticks(list(y) + [-1.5], list(g["nct"]) + ["pooled"], fontsize=7)
        ax.set_ylim(-2.5, len(g) - 0.3)
        ax.set_xlabel("SD on drug / SD on placebo")
        ax.set_title(labels[k], loc="left", fontsize=10)
        _style(ax, grid="x")
    fig.suptitle("Is the response to stimulants more variable than the response to placebo?",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.12, wspace=0.55)
    _save(fig, "fig3_variance_ratios.png", "ClinicalTrials.gov posted results, randomised adult trials")


def fig_faers(rors: pd.DataFrame):
    events = ["blood pressure", "heart rate and rhythm", "ischaemic and serious cardiac", "insomnia"]
    groups = list(dict.fromkeys(rors["drug_group"]))
    bands = list(dict.fromkeys(rors["age_band"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(7.2, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        for i, e in enumerate(events):
            x = rors[(rors["drug_group"] == g) & (rors["event_group"] == e)].set_index("age_band").reindex(bands)
            pos = np.arange(len(bands)) + (i - 1.5) * 0.17
            ok = np.isfinite(x["ror"].to_numpy(dtype=float))
            ax.errorbar(pos[ok], x["ror"][ok], yerr=[(x["ror"] - x["ror_lo"])[ok], (x["ror_hi"] - x["ror"])[ok]],
                        fmt="o", ms=4, lw=1, color=SERIES[i], capsize=0, label=e if g == groups[0] else None)
        ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_yscale("log")
        _plain_log(ax.yaxis)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.set_xticks(np.arange(len(bands)), bands, fontsize=8.5)
        ax.set_title(g, loc="left", fontsize=10)
        ax.set_xlabel("age at onset")
        _style(ax)
    axes[0].set_ylabel("reporting odds ratio")
    fig.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Adverse-event reports for stimulants, by age", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.3, wspace=0.12)
    _save(fig, "fig4_faers_by_age.png", "FDA Adverse Event Reporting System via openFDA")


def fig_medicare(trend: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    classes = ["amphetamine class", "methylphenidate class", "atomoxetine (non-stimulant)", "modafinil and armodafinil"]
    for i, c in enumerate(classes):
        g = trend[trend["drug_class"] == c].sort_values("year")
        if g.empty:
            continue
        ax.plot(g["year"], g["claims65"] / 1000, color=SERIES[i], lw=2, marker="o", ms=4, label=c)
    _style(ax)
    ax.set_xlabel("year")
    ax.set_ylabel("Part D claims, beneficiaries 65 and over (thousands)")
    _legend(ax, loc="upper left")
    ax.set_title("Stimulant prescriptions for Medicare beneficiaries aged 65+", loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.17)
    _save(fig, "fig5_medicare_65plus.png", "CMS Medicare Part D Prescribers by Geography and Drug, national")


def fig_readings(rep: pd.DataFrame):
    from .stats import readings_needed
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rises = np.arange(2, 21)
    for i, r in rep.iterrows():
        ax.plot(rises, [readings_needed(s, r["within_sd"]) for s in rises], color=SERIES[i], lw=2,
                label=f"{r['band']} (SD {r['within_sd']:.1f})")
    _style(ax)
    ax.set_yscale("log")
    _plain_log(ax.yaxis)
    ax.set_xlabel("true rise in systolic pressure to detect (mmHg)")
    ax.set_ylabel("readings needed on and off medication (each)")
    _legend(ax, title="age (within-visit SD, mmHg)", title_fontsize=8.5, loc="upper right")
    ax.set_title("How many readings it takes to see a rise", loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.17)
    _save(fig, "fig6_readings_needed.png", "NHANES 2017–2023 replicate readings; 80% power, one-sided 5%")


def fig_n_of_1(t: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, (d, lab) in enumerate([(0.3, "small benefit (0.3)"), (0.5, "moderate benefit (0.5)"),
                                  (1.0, "large benefit (1.0)")]):
        g = t[t["d"] == d]
        ax.plot(g["k"], g["p_wrong"], color=SERIES[i], lw=2, marker="o", ms=4, label=lab)
    _style(ax)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("number of on/off pairs of weeks compared")
    ax.set_ylabel("chance the comparison points the wrong way")
    _legend(ax, loc="upper right")
    ax.set_title("A single drug holiday is a noisy experiment", loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.17)
    _save(fig, "fig7_drug_holiday.png", "calculation; benefit in units of week-to-week noise")


def draw(out: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if (DATA / "nhanes_adults.csv.gz").exists():
        df = nhanes.load()
        fig_sbp_by_age(df)
        fig_crossing(out["crossing"], out["crossing_params"])
        fig_readings(out["replicates"])
    if "vr_table" in out and len(out["vr_table"]):
        fig_forest(out["vr_table"], out["vr_summary"])
    if "faers_rors" in out:
        fig_faers(out["faers_rors"])
    if "medicare" in out:
        fig_medicare(out["medicare"])
    fig_n_of_1(out["n_of_1"])
