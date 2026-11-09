"""Figures for the drift-detection article.

fig1_stream_nulls.png   river's stream detectors on simulated in-control data: scale and autocorrelation
fig2_batch_nulls.png    batch scores on simulated in-control windows, and NannyML's threshold
fig3_bike.png           Bike Sharing: the daily mean residual, two versions of sigma, and the signals
fig4_elec2.png          Elec2: the daily error rate, and stream detectors with the order inside days shuffled
fig5_price.png          what the budget costs in detection delay on Bike Sharing

Run after drift_numbers.py (which writes figure_data.json):  python figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "figure_data.json").read_text())

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
GRID = "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
})

CODE = "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/drift-detectors-as-control-charts"
SIM = "Simulation; detectors from river 0.26.1."
BIKE = "Data: Bike Sharing, UCI Machine Learning Repository (Fanaee-T and Gama 2014), CC BY 4.0."
ELEC = "Data: Elec2, NSW electricity market 1996-1998 (Harries 1999), normalized version distributed with MOA."


def credit(fig, data: str) -> None:
    fig.text(0.01, 0.005, f"Figure: Diogo Ribeiro. {data}\n{CODE}", fontsize=7.2, color=MUTED, ha="left",
             va="bottom", linespacing=1.4)


def title(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)


def label_end(ax, x, y, text, color, dy=0, dx=4):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK2,
                va="center", ha="left")
    ax.plot([x], [y], "o", ms=4, color=color)


# ---------------------------------------------------------------------------------------------------
def fig1():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.42})
    s = D["ph_scale"]
    a1.plot(s["sigma"], s["ph"], color=ORANGE, lw=2, marker="o", ms=4)
    a1.set_xscale("log")
    a1.axhline(10000 / 30, color=INK, lw=0.9, alpha=0.6)
    a1.annotate("signals at every chance after its\n30-observation warm-up (333)", (0.012, 10000 / 30),
                xytext=(0, -8), textcoords="offset points", fontsize=8.3, color=INK2, va="top")
    a1.axvline(0.36, color=MUTED, lw=0.8, ls=":")
    a1.axvline(40.7, color=MUTED, lw=0.8, ls=":")
    a1.annotate("dotted: the Bike Sharing residual on the\nlog scale (sd 0.36) and in rentals (sd 41)",
                (0.012, 215), fontsize=8.3, color=INK2, va="top",
                bbox={"fc": "white", "ec": "none", "pad": 1.5})
    a1.set_xlabel("standard deviation of the data (log scale)")
    a1.set_ylabel("signals per 10,000 in-control observations")
    a1.set_title("Page-Hinkley's threshold of 50 is in the data's units")

    a = D["ar"]
    phi = a["phi"]
    for key, col in [("KSWIN", AQUA), ("Page-Hinkley", ORANGE), ("ADWIN", BLUE)]:
        a2.plot(phi, a[key], color=col, lw=2, marker="o", ms=4, label=key)
        label_end(a2, phi[-1], a[key][-1], key, col, dy={"KSWIN": 0, "Page-Hinkley": 5, "ADWIN": -7}[key])
    dd = D["ddm"]
    a2.plot(dd["rho"], dd["ddm"], color=VIOLET, lw=2, ls="--", marker="s", ms=4,
            label="DDM, error rate 0.3, errors in runs")
    a2.legend(fontsize=8.3, loc="upper left")
    a2.set_xlim(-0.03, 1.12)
    a2.set_xlabel("lag-1 autocorrelation of the monitored stream")
    a2.set_ylabel("signals per 10,000 in-control observations")
    a2.set_title("Each default assumes independent observations")
    title(fig, "Stream drift detectors on data with no drift")
    credit(fig, SIM + " Normal data (left); AR(1) with unit variance, and Markov-chain errors for DDM (right).")
    fig.subplots_adjust(left=0.07, right=0.95, top=0.84, bottom=0.2)
    fig.savefig(OUT / "fig1_stream_nulls.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
def fig2():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.38})
    p, w = D["psi"], D["wass"]
    n = p["n"]
    series = [("PSI >= 0.1, decile bins", p["deciles_01"], BLUE, "-"),
              ("PSI >= 0.1, Evidently's bins", p["evidently_01"], ORANGE, "-"),
              ("normed Wasserstein >= 0.1", w["iid"], AQUA, "-"),
              ("normed Wasserstein >= 0.1, AR(1) 0.5", w["ar05"], VIOLET, "--")]
    for lab, ys, col, ls in series:
        a1.plot(n, ys, color=col, lw=2, ls=ls, marker="o", ms=4, label=lab)
    a1.set_xscale("log")
    a1.set_xlabel("rows in the current window (reference: 5,000 rows; log scale)")
    a1.set_ylabel("share of windows flagged as drift")
    a1.set_ylim(-0.03, 1.05)
    a1.axvline(24, color=MUTED, lw=0.8, ls=":")
    a1.annotate("one day of\nhourly data", (24, 0.5), xytext=(4, 0), textcoords="offset points", fontsize=8.3,
                color=INK2)
    a1.legend(fontsize=8.3, loc="upper right")
    a1.set_title("Fixed score thresholds: the rate depends on n")

    r = np.asarray(D["nanny"]["rates"])
    bins = np.arange(0, 0.42, 0.02) - 0.01
    a2.hist(r, bins=bins, color=BLUE, edgecolor="white", linewidth=1.5)
    a2.axvline(r.mean(), color=ORANGE, lw=2)
    a2.annotate(f"mean {r.mean():.3f}", (r.mean(), a2.get_ylim()[1] * 0.92), xytext=(5, 0),
                textcoords="offset points", fontsize=8.5, color=INK2)
    a2.axvline(0.00135, color=INK, lw=1.0)
    a2.annotate("black line: a one-sided 3-sigma limit\non normal data, 0.00135", (0.00135, 100),
                xytext=(0.12, 95), textcoords="data", fontsize=8.5, color=INK2,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    a2.set_xlabel("false-alert rate per analysis chunk, given the reference")
    a2.set_ylabel("references (of 400)")
    a2.set_title("NannyML's mean + 3 sd over 10 reference chunks")
    title(fig, "Batch drift scores on windows drawn from the reference distribution")
    credit(fig, "Simulation; scores and default thresholds replicated from Evidently's and NannyML's source code. "
                "Right: KS statistic, chunks of 500.")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.2)
    fig.savefig(OUT / "fig2_batch_nulls.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
def fig3():
    b = D["bike"]
    dates = np.array(b["dates"], dtype="datetime64[D]")
    y = np.asarray(b["daily"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(dates, y, color=BLUE, lw=1.0)
    mu, s_iid, s_day = b["mu"], b["sd_hourly_iid"], b["sd_daily"]
    x0, x1 = dates[0], dates[364]
    ax.fill_between([x0, x1], mu - 3 * s_iid, mu + 3 * s_iid, color=ORANGE, alpha=0.18, lw=0)
    ax.plot([x0, x1], [mu - 3 * s_day] * 2, color=INK, lw=0.9)
    ax.plot([x0, x1], [mu + 3 * s_day] * 2, color=INK, lw=0.9)
    ax.annotate("3 sigma if the 24 hours of a day were independent", (x0, mu + 3 * s_iid), xytext=(4, 3),
                textcoords="offset points", fontsize=8.3, color=INK2, bbox={"fc": "white", "ec": "none", "pad": 1, "alpha": 0.85})
    ax.annotate("3 sigma of the day-to-day spread", (x0, mu + 3 * s_day), xytext=(4, 3),
                textcoords="offset points", fontsize=8.3, color=INK2, bbox={"fc": "white", "ec": "none", "pad": 1, "alpha": 0.85})
    top = 1.08
    ad = np.array(b["adwin11"], dtype="datetime64[D]")
    ax.plot(ad, [top] * len(ad), "v", color=ORANGE, ms=6)
    ax.annotate(f"ADWIN, default: {len(ad)} signals in 2011", (ad[0], top), xytext=(0, 9),
                textcoords="offset points", fontsize=8.3, color=INK2)
    cs = np.array(b["cusum11"] + ([b["first12_cusum"]] if b["first12_cusum"] else []), dtype="datetime64[D]")
    ax.plot(cs, [top - 0.16] * len(cs), "v", color=AQUA, ms=6)
    ax.annotate("green: budgeted CUSUM (one signal in 2011, then 6 Jan 2012)",
                (np.datetime64("2011-01-20"), top - 0.29), fontsize=8.3, color=INK2, va="center")
    ax.axvline(np.datetime64("2012-01-01"), color=MUTED, lw=0.8, ls=":")
    ax.annotate("model fitted to 2011, residuals out of fold", (np.datetime64("2011-02-01"), -1.08), fontsize=8.3,
                color=INK2)
    ax.annotate("2012: the same model, demand has grown", (np.datetime64("2012-02-01"), -1.08), fontsize=8.3,
                color=INK2)
    ax.set_ylim(-1.15, 1.25)
    ax.set_ylabel("daily mean residual, log(1 + rentals)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    title(fig, "A demand model's residual, day by day: which sigma the detector believes")
    credit(fig, BIKE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.16)
    fig.savefig(OUT / "fig3_bike.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
def fig4():
    e = D["elec"]
    y = np.asarray(e["daily"])
    days = np.arange(1, len(y) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.25})
    a1.plot(days, y, color=BLUE, lw=0.7)
    t, r = e["train_days"], e["ref_days"]
    lo, hi = t + 1, t + r
    pb = e["pb"]
    a1.axvspan(lo, hi, color=MUTED, alpha=0.15, lw=0)
    a1.annotate("Phase I", (lo, 0.86), fontsize=8.3, color=INK2)
    a1.plot([lo, len(y)], [pb + 3 * e["sd_bin"]] * 2, color=ORANGE, lw=1.2)
    a1.annotate("binomial 3 sigma", (len(y), pb + 3 * e["sd_bin"]), xytext=(-4, -9), textcoords="offset points",
                fontsize=8.3, color=INK2, ha="right", bbox={"fc": "white", "ec": "none", "pad": 1})
    a1.plot([lo, len(y)], [pb + e["k"] * e["sd_day"]] * 2, color=INK, lw=0.9)
    a1.annotate("between-day sigma, guaranteed k", (len(y), pb + e["k"] * e["sd_day"]), xytext=(-4, 7),
                textcoords="offset points", fontsize=8.3, color=INK2, ha="right",
                bbox={"fc": "white", "ec": "none", "pad": 1})
    dd = np.asarray(e["ddm_days"]) + 1
    cu = np.asarray(e["cusum_days"]) + 1
    a1.plot(dd, [1.06] * len(dd), "|", color=ORANGE, ms=8, mew=1.2)
    a1.annotate(f"DDM, default ({len(dd)} days with a signal)", (30, 1.06), xytext=(0, 7),
                textcoords="offset points", fontsize=8.3, color=INK2)
    a1.plot(cu, [0.95] * len(cu), "|", color=AQUA, ms=8, mew=1.2)
    a1.annotate(f"budgeted CUSUM ({len(cu)} signals)", (cu.max(), 0.95), xytext=(8, 0), textcoords="offset points",
                fontsize=8.3, color=INK2, va="center")
    a1.set_ylim(0, 1.16)
    a1.set_xlabel("day (from 7 May 1996)")
    a1.set_ylabel("daily error rate of a fixed classifier")
    a1.set_title("The error rate drifts, and varies 3x more than binomial")

    p = D["elec_perm"]
    names = list(p.keys())
    x = np.arange(len(names))
    wdt = 0.26
    for j, (lab, col) in enumerate([("real order", BLUE), ("shuffled within each day", ORANGE),
                                    ("fully shuffled", AQUA)]):
        vals = [p[n][j] for n in names]
        a2.bar(x + (j - 1) * wdt, vals, width=wdt - 0.03, color=col, label=lab)
        for xi, v in zip(x + (j - 1) * wdt, vals):
            a2.annotate(f"{v:.0f}", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.8,
                        color=INK2)
    a2.set_xticks(x, names)
    a2.set_ylabel("signals per year")
    a2.legend(fontsize=8.3, loc="upper right")
    a2.grid(axis="x", visible=False)
    a2.set_title("Same errors, different order")
    title(fig, "Elec2: monitoring a classifier's errors")
    credit(fig, ELEC)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.18)
    fig.savefig(OUT / "fig4_elec2.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
def fig5():
    dl, rates = D["delays"], D["rates"]
    shifts = sorted(float(s) for s in next(iter(dl.values())).keys())
    fig, ax = plt.subplots(figsize=(12, 5.0))
    style = {"ADWIN, default": (BLUE, ":"), "ADWIN, calibrated": (BLUE, "-"),
             "Page-Hinkley, default": (ORANGE, ":"), "Page-Hinkley, calibrated": (ORANGE, "-"),
             "CUSUM": (AQUA, "-"), "Shewhart": (VIOLET, "-")}
    names = {"CUSUM": "daily CUSUM, budgeted", "Shewhart": "daily Shewhart, budgeted"}
    never = 70
    for m, (col, ls) in style.items():
        med = []
        for s in shifts:
            v = np.asarray([math.inf if z is None else z for z in dl[m][str(s)]], float)
            q = np.quantile(np.sort(v), 0.5, method="inverted_cdf")
            med.append(never if math.isinf(q) else q)
        ax.plot(shifts, med, color=col, ls=ls, lw=2, marker="o", ms=5,
                label=f"{names.get(m, m)}  ({rates[m]:.2f} in-control signals per 30 days)")
    ax.axhline(never, color=MUTED, lw=0.8)
    ax.annotate("not detected within the quarter (median)", (shifts[0], never), xytext=(0, 4),
                textcoords="offset points", fontsize=8.3, color=INK2)
    ax.set_xticks(shifts, [f"+{s:g}\n({(math.exp(s) - 1) * 100:.0f}% more rentals)" for s in shifts])
    ax.set_xlabel("step added to the log residual")
    ax.set_ylabel("median days from the step to the first signal")
    ax.set_ylim(0, never + 8)
    ax.set_xlim(shifts[0] - 0.025, shifts[-1] + 0.025)
    ax.legend(fontsize=8.3, loc="center left", bbox_to_anchor=(1.01, 0.5))
    title(fig, "The price of the budget: detection delay when the target is one false alarm a year\n(0.08 signals per 30 days)")
    credit(fig, BIKE + " Each quarter of 2011 held out in turn; 28 steps per size.")
    fig.subplots_adjust(left=0.07, right=0.6, top=0.88, bottom=0.2)
    fig.savefig(OUT / "fig5_price.png")
    plt.close(fig)


for fn in (fig1, fig2, fig3, fig4, fig5):
    fn()
print("figures written to", OUT)
