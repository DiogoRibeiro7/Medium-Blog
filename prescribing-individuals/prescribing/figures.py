"""Figures for the article. Each is drawn only if its data are present."""

from __future__ import annotations

import re

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
        bw = 5.0
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
    bars = [("hetero", "hetero_se", "crosses 130/80 (not hypertensive before)", SERIES[0]),
            ("hetero_140", "hetero_140_se", "crosses 140/90 (below it and untreated before)", SERIES[1])]
    top = max((cross[c] + 1.96 * cross[se]).max() for c, se, _, _ in bars)
    for j, (c, se, lab, col) in enumerate(bars):
        off = (j - 0.5) * (w + 0.02)
        ax.bar(xs + off, cross[c], width=w, color=col, label=lab)
        ax.errorbar(xs + off, cross[c], yerr=1.96 * cross[se], fmt="none", ecolor=INK2, lw=0.8)
        for x0, v, e in zip(xs + off, cross[c], cross[se]):
            ax.text(x0, v + 1.96 * e + top * 0.015, f"{100 * v:.1f}%", ha="center", va="bottom",
                    fontsize=8, color=INK2)
    _style(ax)
    ax.set_xticks(xs, [f"{b}\n{100 * r:.0f}% below\n130/80" for b, r in zip(cross["band"], cross["share_not_htn"])],
                  fontsize=8.5)
    ax.set_ylim(0, top * 1.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("age, and share of the age group below 130/80 and untreated")
    ax.set_ylabel("share of the whole age group")
    _legend(ax, loc="upper left")
    ax.set_title(f"Share of each age group moved across a threshold by a {params['mu']:.1f} mmHg average rise",
                 loc="left", fontsize=11, pad=12)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.24)
    _save(fig, "fig2_threshold_crossing.png",
          f"NHANES 2017–2023; response N({params['mu']:.1f}, {params['tau']:.1f}²) mmHg from pooled trials; 95% CI")


def _trial_label(r) -> str:
    c = str(r["conditions"])
    for pat, short in (("attention|adhd", "ADHD"), ("depress", "depression"), ("schizo", "schizophrenia"),
                       ("cocaine", "cocaine dependence"), ("sclerosis", "multiple sclerosis"),
                       ("anxiety", "anxiety"), ("dementia|alzheimer", "dementia"), ("parkinson", "Parkinson's")):
        if re.search(pat, c, flags=re.I):
            c = short
            break
    lo = "" if pd.isna(r["min_age"]) else f"{r['min_age']:.0f}"
    hi = "" if pd.isna(r["max_age"]) else f"{r['max_age']:.0f}"
    return f"{c}, ages {lo}–{hi}\n{r['nct']}"


def fig_forest(vr: pd.DataFrame, summary: pd.DataFrame):
    kinds = [k for k in ("sbp", "hr") if (vr["kind"] == k).any()]
    if not kinds:
        return
    labels = {"sbp": "systolic blood pressure", "hr": "heart rate"}
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.2, 0.36 * max((vr["kind"] == k).sum() for k in kinds) + 2.2),
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
        ticks = [_trial_label(r) for _, r in g.iterrows()] if "conditions" in g.columns else list(g["nct"])
        ax.set_yticks(list(y) + [-1.5], ticks + ["pooled"], fontsize=6.5)
        ax.set_ylim(-2.5, len(g) - 0.3)
        ax.set_xlabel("SD on drug / SD on placebo")
        ax.set_title(labels[k], loc="left", fontsize=10)
        _style(ax, grid="x")
    fig.suptitle("Is the response to stimulants more variable than the response to placebo?",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.87, bottom=0.12, wspace=0.95)
    _save(fig, "fig3_variance_ratios.png",
          "ClinicalTrials.gov posted results; randomised placebo-controlled trials, all ages and indications")


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
    _save(fig, "fig7_drug_holiday.png", "calculation; benefit in units of the SD of one on-minus-off difference")


def fig_absolute(ar: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(len(ar))
    cols = [("extra_2", "+2 mmHg"), ("extra_5", "+5 mmHg"), ("extra_10", "+10 mmHg")]
    w = 0.26
    for i, (c, lab) in enumerate(cols):
        vals = 1000 * ar[c]
        ax.bar(xs + (i - 1) * (w + 0.02), vals, width=w, color=SERIES[i], label=lab)
        for x0, v in zip(xs + (i - 1) * (w + 0.02), vals):
            ax.text(x0, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    _style(ax)
    ax.set_xticks(xs, [f"{b}\n(baseline {100 * r:.0f}%)" for b, r in zip(ar["band"], ar["baseline_risk"])])
    ax.set_ylabel("extra cardiovascular events per 1,000 over 10 years")
    ax.set_xlabel("age (average 10-year risk without the drug)")
    _legend(ax, title="sustained rise in systolic pressure", title_fontsize=8.5, loc="upper left")
    ax.set_title("The same rise in pressure costs more at older ages", loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.2)
    _save(fig, "fig8_absolute_risk.png", "NHANES 2017–2023, Pooled Cohort Equations, adults 40–79 without CVD")


def fig_risk_spread(rs: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ys = np.arange(len(rs))[::-1]
    for y, (_, r) in zip(ys, rs.iterrows()):
        ax.plot([r["risk_p10"], r["risk_p90"]], [y, y], color=SERIES[0], lw=1.5, solid_capstyle="butt")
        ax.plot([r["risk_p25"], r["risk_p75"]], [y, y], color=SERIES[0], lw=7, solid_capstyle="butt")
        ax.plot([r["risk_p50"]], [y], "o", color=SURFACE, mec=INK, ms=6)
    for thr in (0.075, 0.20):
        ax.axvline(thr, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(thr * 1.04, len(rs) - 0.45, f"{100 * thr:g}%", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.002, 1.0)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    def _share(v):
        return f"{100 * v:.0f}%" if v >= 0.01 else f"{100 * v:.1f}%"
    ax.set_yticks(ys, [f"{b}\n{_share(v)} at 20%+" for b, v in zip(rs["band"], rs["share_20_plus"])], fontsize=8.5)
    ax.set_ylim(-0.7, len(rs) - 0.2)
    _style(ax, grid="x")
    ax.set_xlabel("10-year risk of a first heart attack or stroke (Pooled Cohort Equations)")
    ax.set_title("Cardiovascular risk overlaps across age groups", loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.2)
    _save(fig, "fig9_risk_spread.png",
          "NHANES 2017–2023, adults 40–79 without CVD; thin line 10th–90th percentile, thick 25th–75th, dot median")


def draw(out: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if (DATA / "nhanes_adults.csv.gz").exists():
        df = nhanes.load()
        fig_sbp_by_age(df)
        fig_crossing(out["crossing"], out["crossing_params"])
        fig_readings(out["replicates"])
        if len(out.get("absolute_risk", [])):
            fig_absolute(out["absolute_risk"])
        if "risk_spread" in out:
            fig_risk_spread(out["risk_spread"])
    if "vr_table" in out and len(out["vr_table"]):
        pop = "all"
        fig_forest(out["vr_table"][out["vr_table"]["population"] == pop],
                   out["vr_summary"][out["vr_summary"]["population"] == pop])
    if "faers_rors" in out:
        fig_faers(out["faers_rors"])
    if "medicare" in out:
        fig_medicare(out["medicare"])
    fig_n_of_1(out["n_of_1"])
