"""Draw the article's figures from the tables forecast_numbers.py writes to results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)
SIM = "Simulations of Lorenz's 1963 and 1996 models; code: Medium-Blog/forecast-horizon. Figure: Diogo Ribeiro."


def frame(fig, title, subtitle, credit, note=None, top_in=0.75, bottom_in=0.7):
    """Title and subtitle above the plot, a note and the credit below."""
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(0.01, 1 - 0.08 / h, title, fontsize=11, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 1 - 0.34 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top")
    y = 0.06 / h
    fig.text(0.01, y, credit, fontsize=7.2, color=MUTED, ha="left", va="bottom")
    if note:
        fig.text(0.01, y + 0.2 / h, note, fontsize=8, color=INK2, ha="left", va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig1_rounding():
    r = pd.read_csv(RES / "rounding.csv")
    r = r[r.t <= 25]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    a1.plot(r.t, r.x_full, color=BLUE, lw=1.6, label="Start kept to six decimals")
    a1.plot(r.t, r.x_rounded, color=ORANGE, lw=1.4, ls=(0, (4, 2)), label="Start rounded to three decimals")
    a1.set_ylabel("First variable of the model")
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5, ncols=2)
    grid(a1)
    a2.semilogy(r.t, r.distance, color=INK2, lw=1.4)
    a2.set_ylabel("Distance between the runs")
    a2.set_xlabel("Model time")
    grid(a2)
    a2.text(
        14, 3e-3, "rising as a straight line on this scale:\nthe gap doubles at a steady rate", fontsize=8.5, color=INK2
    )
    frame(
        fig,
        "Figure 1. Two runs of Lorenz's 1963 model that start 0.0002 apart",
        "The second run starts from the same numbers rounded to three decimals, as in Lorenz's 1961 printout.",
        SIM,
        note="Top: one of the three variables in each run. Bottom: distance between the two runs, on a log scale.",
        bottom_in=0.85,
    )
    fig.savefig(FIG / "fig1_rounding.png")
    plt.close(fig)


def fig2_error_growth():
    g = pd.read_csv(RES / "one_scale_growth.csv")
    cols = [c for c in g.columns if c.startswith("eps_")]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for c, col in zip(cols, RAMP, strict=True):
        ax.semilogy(g.days, g[c], color=col, lw=1.5)
        exp = int(c.split("e")[-1])
        ax.text(
            -0.6,
            g[c].iloc[0],
            "10" + str(exp).translate(SUP),
            fontsize=8,
            color=INK2,
            ha="right",
            va="center",
        )
    sat = g.sat.iloc[0]
    ax.axhline(sat, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.axhline(sat / 2, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(-9, 62)
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Forecast error (root mean square)")
    grid(ax)
    frame(
        fig,
        "Figure 2. Forecast error in a model atmosphere, from starting errors of every size",
        "Lorenz's 1996 model with 40 variables; each line pools 300 forecasts with the starting error shown at left.",
        SIM,
        note="Left: starting error as a share of the natural variability. Dashed: error of a guess from the climate;\n"
        "dotted: half of it, the horizon used here. One model time unit = 5 days (Lorenz).",
        bottom_in=0.85,
    )
    fig.savefig(FIG / "fig2_error_growth.png")
    plt.close(fig)


def fig3_precision():
    p = pd.read_csv(RES / "precision.csv")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    styles = [
        ("one scale", "all", BLUE, "-", "One scale: every tenfold gain in precision buys the same days"),
        ("two scales", "slow", ORANGE, "-", "Two scales, starting error in the large-scale variables"),
        ("two scales", "fast", AQUA, (0, (4, 2)), "Two scales, starting error in the small-scale variables only"),
    ]
    for model, start, col, ls, lab in styles:
        s = p[(p.model == model) & (p.start == start) & (p.days > 0)]
        ax.semilogx(s.eps, s.days, color=col, lw=1.8, ls=ls, marker="o", ms=4, label=lab)
    ax.invert_xaxis()
    ax.set_xlabel("Starting error, as a share of natural variability (more precise to the right)")
    ax.set_ylabel("Horizon in days")
    ax.legend(loc="upper left", fontsize=8.5)
    grid(ax)
    frame(
        fig,
        "Figure 3. What more precise starting data buys, with one scale of motion and with two",
        "Days until the large-scale forecast error reaches half the error of a guess from the climate.",
        SIM,
        note="Two scales: 36 large-scale variables, each coupled to 10 small-scale ones that change ten times faster.",
        bottom_in=0.85,
    )
    fig.savefig(FIG / "fig3_precision.png")
    plt.close(fig)


def fig4_ensemble():
    pl = pd.read_csv(RES / "ensemble_plume.csv")
    sk = pd.read_csv(RES / "ensemble_skill.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"wspace": 0.28})
    for k in range(50):
        a1.plot(pl.days, pl[f"m{k}"], color="#9ec5f4", lw=0.6, alpha=0.8)
    a1.plot(pl.days, pl[[f"m{k}" for k in range(50)]].mean(axis=1), color=BLUE, lw=1.8, label="Ensemble mean")
    a1.plot(pl.days, pl.single, color=ORANGE, lw=1.4, label="Single forecast")
    a1.plot(pl.days, pl.truth, color=INK, lw=1.4, ls=(0, (3, 2)), label="What happened")
    a1.set_xlabel("Days ahead")
    a1.set_ylabel("One variable of the model")
    a1.set_title("One forecast: 50 members fan out", fontsize=9.5, color=INK2, loc="left")
    a1.legend(loc="lower left", fontsize=8)
    grid(a1)
    a2.plot(sk.days, sk.acc_single, color=ORANGE, lw=1.6, label="Single forecast (anomaly correlation)")
    a2.plot(sk.days, sk.acc_mean, color=BLUE, lw=1.6, label="Ensemble mean (anomaly correlation)")
    a2.plot(sk.days, sk.crpss, color=AQUA, lw=1.6, ls=(0, (4, 2)), label="Ensemble probabilities (skill score)")
    a2.axhline(0.6, color=MUTED, lw=0.8, ls=":")
    a2.text(29.5, 0.62, "0.6: usual limit of a useful map", fontsize=7.5, color=INK2, ha="right")
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylim(-0.25, 1.02)
    a2.set_xlabel("Days ahead")
    a2.set_ylabel("Skill (1 = perfect, 0 = no better than the climate)")
    a2.set_title("200 forecasts: how long each kind of skill lasts", fontsize=9.5, color=INK2, loc="left")
    a2.legend(loc="upper right", fontsize=7.8)
    grid(a2)
    frame(
        fig,
        "Figure 4. A single forecast against an ensemble of 50, in a model with no model error",
        "Lorenz's 1996 model; starting error 5% of the natural variability, the same size in every member.",
        SIM,
        bottom_in=0.75,
    )
    fig.savefig(FIG / "fig4_ensemble.png")
    plt.close(fig)


def fig5_analogues():
    a = pd.read_csv(RES / "analogues.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"wspace": 0.25}, sharey=True)
    a1.plot(a.day, a.real, color=BLUE, lw=1.8, marker="o", ms=3, label="Each day and its closest match")
    a1.plot(a.day, a.best_tenth, color=ORANGE, lw=1.6, marker="o", ms=3, label="The closest tenth of matches")
    a1.axhline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    a1.text(20, 1.02, "two unrelated days", fontsize=8, color=INK2, ha="right", va="bottom")
    a1.set_xlabel("Days after the match")
    a1.set_ylabel("Distance between the two sequences\n(share of the distance between unrelated days)")
    a1.set_title("Real days and their closest matches", fontsize=9.5, color=INK2, loc="left")
    a1.legend(loc="lower right", fontsize=8)
    a1.set_ylim(0, 1.12)
    grid(a1)
    a2.plot(a.day, a.pc_real, color=BLUE, lw=1.8, marker="o", ms=3, label="Real atmosphere")
    a2.plot(a.day, a.pc_surrogate, color=AQUA, lw=1.6, ls=(0, (4, 2)), label="Linear copy with random forcing")
    a2.axhline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    a2.set_xlabel("Days after the match")
    a2.set_title("The same test on a copy with no chaos", fontsize=9.5, color=INK2, loc="left")
    a2.legend(loc="lower right", fontsize=8)
    grid(a2)
    frame(
        fig,
        "Figure 5. How long two similar days in the real atmosphere stay similar",
        "Daily near-surface air temperature over North America, 2013 and 2014, seasonal cycle removed; matches "
        "from the same season, at least 30 days apart.",
        "Data: NCEP/NCAR Reanalysis 1 sample (NOAA PSL), as distributed with xarray. Figure: Diogo Ribeiro.",
        note="Right: both series reduced to their 20 leading patterns; the copy has the same memory and noise but "
        "no chaos.",
        bottom_in=0.85,
    )
    fig.savefig(FIG / "fig5_analogues.png")
    plt.close(fig)


def fig6_doubling():
    lit = pd.read_csv(HERE / "literature" / "doubling_times.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    hi = lit.doubling_days_high.fillna(lit.doubling_days)
    ax.vlines(lit.year, lit.doubling_days, hi, color=BLUE, lw=2.5)
    ax.plot(lit.year, lit.doubling_days, "o", color=BLUE, ms=6)
    ax.plot(lit.year[hi > lit.doubling_days], hi[hi > lit.doubling_days], "o", color=BLUE, ms=6, mfc="white")
    labels = {
        1966: "Early circulation models\n(Charney et al. 1966)",
        1969: "Primitive-equation model\n(Smagorinsky 1969)",
        1981: "ECMWF forecasts, winter 1980/81\n(Lorenz 1982)",
        1997: "ECMWF, winter 1996/97",
        2001: "ECMWF, winter 2000/01\n(Simmons and Hollingsworth)",
    }
    offsets = {1966: (1.0, 0.15), 1969: (1.0, 0.1), 1981: (1.0, 0.1), 1997: (-1.0, 0.05), 2001: (1.0, 0.1)}
    for row, top in zip(lit.itertuples(), hi, strict=True):
        dx, dy = offsets[row.year]
        ax.text(
            row.year + dx,
            top + dy,
            labels[row.year],
            fontsize=7.8,
            color=INK2,
            ha="left" if dx > 0 else "right",
            va="bottom",
        )
    ax.set_xlim(1960, 2012)
    ax.set_ylim(0, 6.2)
    ax.set_ylabel("Days for a small error to double")
    ax.set_xlabel("Year of the forecasts studied")
    grid(ax)
    frame(
        fig,
        "Figure 6. The better the models, the faster small errors are found to grow",
        "Published estimates of the doubling time of small forecast errors; for ECMWF in 1996/97 and 2000/01, "
        "days 1 and 2 (filled and open).",
        "Sources: Bengtsson and Hodges (2006) for 1966 and 1969; Lorenz (1982); Simmons and Hollingsworth (2002). "
        "Figure: Diogo Ribeiro.",
        bottom_in=0.75,
    )
    fig.savefig(FIG / "fig6_doubling.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig1_rounding()
    fig2_error_growth()
    fig3_precision()
    fig4_ensemble()
    fig5_analogues()
    fig6_doubling()


if __name__ == "__main__":
    main()
