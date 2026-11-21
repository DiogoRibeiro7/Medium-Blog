"""Draw the article's figures from data/ and results/ (run hours_numbers.py first).

python figures.py
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import hours_core as hc  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, RES, FIG = HERE / "data", HERE / "results", HERE / "figures"
NA_C, LFS_C, TR_C = "#2a78d6", "#eb6834", "#8a8985"
PERSONS_C, PER_WORKER_C = "#1baf7a", "#6250d6"
ABSENCE_C, AT_WORK_C = "#008300", "#e87ba4"
INK, INK2, GRID, SURF, SHADE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#ecebe7"
CREDIT = "Calculations: Diogo Ribeiro."
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#bdbcb7",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
    }
)

GVA, _ = hc.load(DATA / "gva.csv")
H_NA, _ = hc.load(DATA / "hours_na.csv")
H_LFS, _ = hc.load(DATA / "hours_lfs.csv", age="Y15-74")
END = hc.q(hc.LAST)


def rows(name):
    with open(RES / name, newline="") as f:
        return list(csv.DictReader(f))


def t(k):
    return k / 4


def label(c):
    return hc.NAMES[c] if c == hc.EU else c


def series(ax, s, color, lw=2.0, first=None):
    """Draw a quarterly series and leave gaps where quarters are missing."""
    ks = sorted(k for k in s if (first is None or k >= first) and k <= END)
    seg_x, seg_y = [], []
    for i, k in enumerate(ks):
        if i and k != ks[i - 1] + 1:
            ax.plot(seg_x, seg_y, color=color, lw=lw)
            seg_x, seg_y = [], []
        seg_x.append(t(k))
        seg_y.append(s[k])
    ax.plot(seg_x, seg_y, color=color, lw=lw)


def productivity_panel(ax, c, small=False):
    na = hc.index_to(hc.ratio(GVA[c], H_NA[c]))
    lfs = hc.index_to(hc.ratio(GVA[c], H_LFS[c]))
    first = hc.q("1997-Q1")
    _, trend = hc.log_trend(na, hc.q(hc.TREND[0]), hc.q(hc.TREND[1]))
    ax.axvspan(t(hc.q("2008-Q2")), t(hc.q("2009-Q3")), color=SHADE, lw=0, zorder=0)
    ks = [k for k in range(first, END + 1)]
    ax.plot([t(k) for k in ks], [trend(k) for k in ks], ls=(0, (4, 3)), color=TR_C, lw=1.2)
    series(ax, na, NA_C, lw=1.6 if small else 2.0, first=first)
    series(ax, lfs, LFS_C, lw=1.6 if small else 2.0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(1996.5, 2027.4)
    return na, lfs, trend


def fig1():
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
    na, lfs, trend = productivity_panel(ax, hc.EU)
    ax.set_ylim(75, 150)
    ax.set_xticks([1997, 2000, 2003, 2006, 2009, 2012, 2015, 2018, 2021, 2024])
    ax.annotate(f"National accounts hours: {na[END]:.0f}", (t(END), na[END]), xytext=(8, -10),
                textcoords="offset points", color=INK, fontsize=9.5, va="center")  # fmt: skip
    ax.annotate(f"LFS hours: {lfs[END]:.0f}", (t(END), lfs[END]), xytext=(8, 8),
                textcoords="offset points", color=INK, fontsize=9.5, va="center")  # fmt: skip
    ax.annotate(f"1997–2007 trend: {trend(END):.0f}", (t(END), trend(END)), xytext=(-6, 10),
                textcoords="offset points", ha="right", color=INK2, fontsize=9)  # fmt: skip
    ax.text(t(hc.q("2008-Q4")), 143, "Financial\ncrisis", ha="center", va="top", color=INK2, fontsize=9)
    ax.set_title("EU-27 output per hour with two measures of hours worked", loc="left", fontsize=13, color=INK, pad=26)
    ax.text(0, 1.02, "Real gross value added per hour worked, index 2009 = 100, seasonally adjusted, 1997Q1–2026Q2",
            transform=ax.transAxes, color=INK2, fontsize=10)  # fmt: skip
    handles = [
        Line2D([], [], color=NA_C, lw=2, label="National accounts hours"),
        Line2D([], [], color=LFS_C, lw=2, label="Labour Force Survey hours (from 2009)"),
        Line2D([], [], color=TR_C, lw=1.2, ls=(0, (4, 3)), label="Exponential trend 1997–2007 (national accounts)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "Source: Eurostat namq_10_a10, namq_10_a10_e, lfsi_ahw_q (ages 15–74). The LFS series has a "
             f"methodological break in 2021Q1. {CREDIT}", fontsize=7.5, color=INK2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig1_eu27_two_denominators.png")
    plt.close(fig)


def fig2():
    p = rows("eu27_paths.csv")
    ks = [hc.q(r["quarter"]) for r in p]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.4), dpi=200, sharex=True, height_ratios=(1.25, 1))
    for ax in (a1, a2):
        ax.axvspan(t(hc.q(hc.COVID[0])), t(hc.q(hc.COVID[1])) + 0.25, color=SHADE, lw=0, zorder=0)
        ax.axvline(t(hc.q(hc.BREAK)), color=INK2, lw=0.8, ls=":")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    sel = [i for i, k in enumerate(ks) if k >= hc.q("2009-Q1")]
    x = [t(ks[i]) for i in sel]
    a1.plot(x, [float(p[i]["hours_na"]) for i in sel], color=NA_C, lw=2)
    a1.plot(x, [float(p[i]["hours_lfs"]) for i in sel], color=LFS_C, lw=2)
    a1.text(x[-1] + 0.2, float(p[sel[-1]]["hours_na"]), "National accounts", color=INK, fontsize=9, va="center")
    a1.text(x[-1] + 0.2, float(p[sel[-1]]["hours_lfs"]), "LFS", color=INK, fontsize=9, va="center")
    a1.set_title("Hours worked, index 2009 = 100", loc="left", fontsize=11, color=INK)
    a1.text(t(hc.q(hc.BREAK)) + 0.15, 90.5, "LFS break 2021Q1", fontsize=8.5, color=INK2, va="center")
    a1.text(t(hc.q(hc.COVID[0])) - 0.1, 108.6, "Pandemic", fontsize=8.5, color=INK2, ha="right", va="center")
    gap = [100 * math.log(float(p[i]["hours_lfs"]) / float(p[i]["hours_na"])) for i in sel]
    a2.plot(x, gap, color=INK, lw=1.6)
    d = {r["country"]: r for r in rows("drift.csv")}[hc.EU]
    for (a, b), slope in (
        (("2009-Q1", "2019-Q4"), float(d["pre_slope"])),
        (("2021-Q2", hc.LAST), float(d["post_slope"])),
    ):
        ka, kb = hc.q(a), hc.q(b)
        idx = [i for i, k in enumerate(ks[j] for j in sel) if ka <= k <= kb]
        xs = [x[i] for i in idx]
        mean_x, mean_y = sum(xs) / len(xs), sum(gap[i] for i in idx) / len(idx)
        a2.plot([xs[0], xs[-1]], [mean_y + slope * (xs[0] - mean_x), mean_y + slope * (xs[-1] - mean_x)],
                color=LFS_C, lw=2.2, alpha=0.9)  # fmt: skip
        seg = [gap[i] for i in idx]
        if ka < hc.q(hc.COVID[0]):
            ax_x, ax_y, va = 2012.6, min(seg) - 0.35, "top"
        else:
            ax_x, ax_y, va = xs[0] + 0.6, max(seg) + 0.35, "bottom"
        a2.text(ax_x, ax_y, f"drift {slope:+.2f}% a year", fontsize=8.5, color=INK2, va=va)
    a2.axhline(0, color="#bdbcb7", lw=0.8)
    a2.set_title("LFS hours relative to national-accounts hours, log points (0 = 2009)", loc="left", fontsize=11,
                 color=INK)  # fmt: skip
    a2.set_xlim(2008.7, 2027.6)
    a2.set_xticks(range(2010, 2027, 2))
    a2.set_xticklabels([str(y) for y in range(2010, 2027, 2)])
    fig.suptitle("EU-27: the two hours series drift apart, before and after the 2021 survey break", x=0.01,
                 ha="left", fontsize=13, color=INK)  # fmt: skip
    fig.text(0.01, 0.01, "Seasonally adjusted. Drift lines: least-squares slopes, 2009Q1–2019Q4 and 2021Q2–2026Q2. "
             f"Source: Eurostat namq_10_a10_e, lfsi_ahw_q. {CREDIT}", fontsize=7.5, color=INK2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(FIG / "fig2_eu27_hours_drift.png")
    plt.close(fig)


def fig3():
    c = {r["country"]: r for r in rows("countries_2019_2025.csv")}
    adj = {r["country"]: float(r["prod_lfs_adjusted"]) for r in rows("break_adjusted.csv")}
    order = sorted(c, key=lambda k: float(c[k]["gap"]))
    fig, ax = plt.subplots(figsize=(8.8, 9.8), dpi=200)
    for i, k in enumerate(order):
        a, b = float(c[k]["prod_na"]), float(c[k]["prod_lfs"])
        ax.plot([a, b], [i, i], color="#cfcec9", lw=2, zorder=1)
        if k == hc.EU:
            ax.axhspan(i - 0.45, i + 0.45, color="#efeeea", zorder=0)
    y = range(len(order))
    ax.scatter([adj[k] for k in order], y, s=40, facecolor="none", edgecolor=LFS_C, linewidth=1.4, zorder=2,
               label="LFS hours, 2021 break step removed")  # fmt: skip
    ax.scatter([float(c[k]["prod_na"]) for k in order], y, s=46, color=NA_C, edgecolor=SURF, linewidth=1.2,
               zorder=3, label="National accounts hours")  # fmt: skip
    ax.scatter([float(c[k]["prod_lfs"]) for k in order], y, s=46, color=LFS_C, edgecolor=SURF, linewidth=1.2,
               zorder=3, label="LFS hours")  # fmt: skip
    ax.set_yticks(list(y))
    ax.set_yticklabels([label(k) for k in order], fontsize=9)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "EU-27":
            lab.set_fontweight("bold")
    ax.axvline(0, color="#bdbcb7", lw=0.8)
    ax.axvline(hc.STAGNATION, color=INK2, lw=0.8, ls=":")
    ax.text(hc.STAGNATION + 0.05, len(order) - 0.3, f"{hc.STAGNATION}% a year", fontsize=8, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Growth of real GVA per hour, 2019 to 2025 (% a year)")
    ax.set_title("Productivity growth 2019–2025 by hours source", loc="left", fontsize=12.5, color=INK, pad=24)
    ax.text(0, 1.015, "Sorted by the gap, LFS minus national accounts; largest gaps at the top.",
            transform=ax.transAxes, fontsize=9, color=INK2)  # fmt: skip
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "Growth between 2019 and 2025 annual totals. Luxembourg: cross-border workers are in GVA "
             "and national-accounts hours\nbut not in the resident LFS. Ireland: GVA includes large multinational "
             f"profits. Source: Eurostat. {CREDIT}", fontsize=7.3, color=INK2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig3_countries_2019_2025.png")
    plt.close(fig)


def stacked(ax, order, parts, colors, totals):
    """Diverging stacked bars: positive parts to the right of zero, negative parts to the left, 2px gaps."""
    for i, k in enumerate(order):
        pos, neg = 0.0, 0.0
        for vals, col in zip(parts, colors, strict=True):
            v = vals[k]
            left = pos if v >= 0 else neg + v
            ax.barh(i, abs(v), left=left, height=0.66, color=col, edgecolor=SURF, linewidth=1.0)
            if v >= 0:
                pos += v
            else:
                neg += v
        ax.scatter([totals[k]], [i], s=22, color=INK, zorder=4)
    ax.axvline(0, color="#bdbcb7", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label(k) for k in order], fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig4():
    d = {r["country"]: r for r in rows("decomposition.csv")}
    order = sorted(d, key=lambda k: -float(d[k]["total"]))
    persons = {k: float(d[k]["persons"]) for k in order}
    worker = {k: float(d[k]["hours_per_worker"]) for k in order}
    total = {k: float(d[k]["total"]) for k in order}
    fig, ax = plt.subplots(figsize=(8.8, 9.6), dpi=200)
    stacked(ax, order, (persons, worker), (PERSONS_C, PER_WORKER_C), total)
    ax.set_xlabel("Change in ln(LFS hours / national-accounts hours), 2019 to 2025, log points")
    ax.set_title("Where the two hours series disagree: workers or hours per worker", loc="left", fontsize=12.5,
                 color=INK, pad=24)  # fmt: skip
    ax.text(0, 1.015, "Negative: LFS hours grew less than national-accounts hours. Dot = total.",
            transform=ax.transAxes, fontsize=9, color=INK2)  # fmt: skip
    ax.legend(handles=[Patch(color=PERSONS_C, label="Employed persons (LFS vs national accounts)"),
                       Patch(color=PER_WORKER_C, label="Hours per worker (LFS vs national accounts)"),
                       Line2D([], [], marker="o", color=INK, lw=0, label="Total")],
              loc="lower left", frameon=False, fontsize=9)  # fmt: skip
    fig.text(0.01, 0.005, "Annual means. Persons: LFS employment aged 15–74 against national-accounts employment "
             "(domestic concept).\nSource: Eurostat namq_10_a10_e, namq_10_pe, lfsi_ahw_q, lfsi_emp_q. "
             f"{CREDIT}", fontsize=7.3, color=INK2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig4_workers_or_hours.png")
    plt.close(fig)


def fig5():
    d = {r["country"]: r for r in rows("absences.csv")}
    order = sorted(d, key=lambda k: -float(d[k]["per_worker"]))
    absence = {k: float(d[k]["absence_part"]) for k in order}
    at_work = {k: float(d[k]["at_work_part"]) for k in order}
    total = {k: float(d[k]["per_worker"]) for k in order}
    fig, ax = plt.subplots(figsize=(8.8, 9.4), dpi=200)
    stacked(ax, order, (absence, at_work), (ABSENCE_C, AT_WORK_C), total)
    ax.set_xlabel("Change in LFS hours per worker, 2019 to 2025, log points")
    ax.set_title("Inside the survey: more absences or shorter hours at work", loc="left", fontsize=12.5, color=INK,
                 pad=24)  # fmt: skip
    ax.text(0, 1.015, "LFS hours per worker = (1 − share absent in the reference week) × hours of those at work.",
            transform=ax.transAxes, fontsize=9, color=INK2)  # fmt: skip
    ax.legend(handles=[Patch(color=ABSENCE_C, label="Share absent from work"),
                       Patch(color=AT_WORK_C, label="Hours of those at work"),
                       Line2D([], [], marker="o", color=INK, lw=0, label="Total")],
              loc="lower left", frameon=False, fontsize=9)  # fmt: skip
    fig.text(0.01, 0.005, "Annual means, ages 15–74, seasonally adjusted. Malta publishes no absence series. "
             f"Source: Eurostat lfsi_ahw_q, lfsi_emp_q, lfsi_abt_q. {CREDIT}", fontsize=7.3, color=INK2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig5_absence_or_hours_at_work.png")
    plt.close(fig)


def fig6():
    c = {r["country"]: r for r in rows("countries_2019_2025.csv")}
    panels = ("DE", "FR", "IT", "ES", "NL", "PL", "PT", "CZ")
    fig, axs = plt.subplots(2, 4, figsize=(14, 7), dpi=200)
    for ax, k in zip(axs.flat, panels, strict=True):
        productivity_panel(ax, k, small=True)
        ax.set_title(hc.NAMES[k], loc="left", fontsize=11.5, color=INK)
        ax.text(0.02, 0.97, f"2019–25, % a year\nNA {float(c[k]['prod_na']):+.1f}   LFS {float(c[k]['prod_lfs']):+.1f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)  # fmt: skip
        ax.set_xticks([2000, 2010, 2020])
        ax.tick_params(labelsize=8.5)
    fig.suptitle("Output per hour by country with the two hours measures (index 2009 = 100)", x=0.01, ha="left",
                 fontsize=13, color=INK)  # fmt: skip
    handles = [
        Line2D([], [], color=NA_C, lw=2, label="National accounts hours"),
        Line2D([], [], color=LFS_C, lw=2, label="LFS hours (from 2009)"),
        Line2D([], [], color=TR_C, lw=1.2, ls=(0, (4, 3)), label="1997–2007 trend"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.005, 0.965))
    fig.text(0.01, 0.005, "Real GVA per hour, seasonally adjusted. Shaded: 2008Q2–2009Q3. Y-axes differ by panel. "
             "Germany has no LFS quarters in 2020. Source: Eurostat. " + CREDIT, fontsize=7.5, color=INK2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(FIG / "fig6_country_panels.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print("figures written to", FIG)
