"""Figures for the article, from ``profiles/`` and ``results/``.

    python figures.py

fig1  the feed: trips per monthly file, and three shares a table check would test against a fixed limit
fig2  every check on every real delivery from 2020-01: which fired
fig3  the fourteen planted faults: share of deliveries on which table checks and change checks caught each
fig4  the threshold of the change checks: alarms on real deliveries against faults caught
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from feed.bundle import read  # noqa: E402
from feed.checks import CHANGE, TABLE  # noqa: E402
from feed.faults import FAULTS  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG, PROF = HERE / "results", HERE / "figures", HERE / "profiles"
SURFACE, INK, INK2, GRID, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog · data: NYC TLC yellow-taxi trip records"

# dated events with a published source (see the article's references)
EVENTS = [("2020-03-22", "New York on PAUSE"), ("2022-12-19", "fare increase"),
          ("2025-01-05", "congestion fee")]
FAULT_LABEL = {
    "late_days": "last days of the month missing", "vendor_missing": "one vendor's trips missing",
    "duplicate_day": "one day loaded twice", "partial_rows": "15% of rows missing at random",
    "km_distance": "distance in kilometres", "payment_swap": "card and cash codes swapped",
    "utc_shift": "timestamps in UTC", "tips_dropped": "one vendor's card tips set to 0",
    "null_passengers": "one vendor's passenger counts empty", "zone_shift": "one vendor's pickup zone off by one",
    "cents": "fares in cents", "stale_copy": "last month's file again",
    "airport_fee_int": "airport fee as an integer column", "renamed_column": "tip_amount renamed",
}
KIND_LABEL = {"silent": "silent faults", "loud": "loud faults", "schema": "schema faults"}

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


def month_ts(m: str) -> pd.Timestamp:
    return pd.Timestamp(f"{m}-15")


def events(ax, labels=True, y=1.0):
    for d, label in EVENTS:
        t = pd.Timestamp(d)
        ax.axvline(t, color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=1)
        if labels:
            ax.text(t, y, " " + label, transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8,
                    color=INK2)


def fig1(real: dict):
    ms = sorted(real)
    t = [month_ts(m) for m in ms]
    n = np.array([real[m]["n_rows"] for m in ms], float)
    share = {
        "payment type 0, five columns empty": [real[m]["categories"]["payment_type"].get("0", 0) / real[m]["n_rows"]
                                               for m in ms],
        "negative fare": [real[m]["numeric"]["fare_amount"]["neg"] / real[m]["n_rows"] for m in ms],
        "total does not add up": [real[m]["recon_mismatch"] / real[m]["n_rows"] for m in ms],
    }
    fig, (a, b) = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.15]))
    a.plot(t, n / 1e6, color=INK, lw=2)
    a.set_yscale("log")
    a.set_yticks([0.25, 0.5, 1, 2, 4, 8])
    a.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.axhline(1.0, color=ORANGE, lw=1.2, ls="--")
    a.text(pd.Timestamp("2023-02-15"), 0.96, "table check's volume floor: 1M rows", ha="left", va="top", fontsize=8,
           color=INK2)
    a.set_ylabel("trips in the file (millions)")
    a.set_title("One monthly file per delivery: trips, January 2019 to September 2026", loc="left")
    grid(a)
    events(a)
    colors = (BLUE, ORANGE, AQUA)
    nudge = {"total does not add up": 1.12, "payment type 0, five columns empty": 0.85}
    for (label, v), c in zip(share.items(), colors, strict=True):
        v = np.clip(np.array(v), 1e-4, None)
        b.plot(t, 100 * v, color=c, lw=2)
        b.text(t[-1] + pd.Timedelta(days=25), 100 * v[-1] * nudge.get(label, 1.0), label, color=INK, fontsize=8.5,
               va="center")
    b.axhline(1.0, color=INK2, lw=1.2, ls="--")
    b.text(pd.Timestamp("2020-07-01"), 1.04, "limit of the tolerant table checks: 1% of rows", ha="left",
           va="bottom", fontsize=8, color=INK2)
    b.set_yscale("log")
    b.set_yticks([0.1, 1, 10, 50])
    b.set_yticklabels(["0.1%", "1%", "10%", "50%"])
    b.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_ylim(0.06, 70)
    b.set_ylabel("share of the file's rows")
    b.set_title("Three shares a table check compares with a fixed limit", loc="left")
    grid(b)
    events(b, labels=False)
    b.xaxis.set_major_locator(mdates.YearLocator())
    b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    b.set_xlim(t[0] - pd.Timedelta(days=20), t[-1] + pd.Timedelta(days=20))
    fig.subplots_adjust(left=0.08, right=0.75, top=0.95, bottom=0.07, hspace=0.25)
    credit(fig)
    save(fig, "fig1_feed.png")


def fig2(R: pd.DataFrame):
    ms = sorted(R.month.unique())
    fam = {"table": list(TABLE), "change": list(CHANGE)}
    fig, axes = plt.subplots(2, 1, figsize=(12, 9.2), gridspec_kw=dict(height_ratios=[len(TABLE), len(CHANGE)]))
    for ax, (family, checks), color in zip(axes, fam.items(), (INK2, BLUE), strict=True):
        x = R[R.family == family].pivot(index="check", columns="month", values="fired").reindex(checks)[ms]
        M = x.to_numpy(dtype=float)
        ax.pcolormesh(np.arange(len(ms) + 1), np.arange(len(checks) + 1), M, cmap=ListedColormap([EMPTY, color]),
                      vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1.2)
        ax.set_ylim(len(checks), 0)
        ax.set_yticks(np.arange(len(checks)) + 0.5)
        rate = M.mean(axis=1)
        ax.set_yticklabels([f"{c}  ({100 * r:.0f}%)" for c, r in zip(checks, rate, strict=True)], fontsize=8.5)
        jan = [i for i, m in enumerate(ms) if m.endswith("-01")]
        ax.set_xticks([i + 0.5 for i in jan])
        ax.set_xticklabels([ms[i][:4] for i in jan])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for d, label in EVENTS:
            m = d[:7]
            if m in ms:
                i = ms.index(m)
                ax.axvline(i, color=INK, lw=0.8, ls=(0, (2, 2)))
                if family == "table":
                    ax.text(i, -0.3, label, ha="left", va="bottom", fontsize=8, color=INK2)
    k = R[R.family == "table"].groupby("month").fired.sum().reindex(ms)
    c = R[R.family == "change"].groupby("month").fired.sum().reindex(ms)
    axes[0].set_title(f"Table checks: rules on one delivery, written from the data dictionary "
                      f"(mean {k.mean():.1f} alarms per delivery)", loc="left", pad=16)
    axes[1].set_title(f"Change checks: the same delivery against the 12 before it, robust |z| > 5 "
                      f"(mean {c.mean():.1f} alarms per delivery)", loc="left")
    fig.text(0.01, 0.985, "Real deliveries, January 2020 to September 2026: a filled cell is an alarm. "
             "In brackets, the share of deliveries on which the check fired.", fontsize=9, color=INK2, va="top")
    fig.subplots_adjust(left=0.2, right=0.99, top=0.92, bottom=0.04, hspace=0.18)
    credit(fig)
    save(fig, "fig2_alarms.png")


def fig3(F: pd.DataFrame):
    det = F.groupby(["fault", "family", "month"]).detected.any().groupby(["fault", "family"]).mean().unstack()
    order = [f for kind in ("silent", "loud", "schema") for f in FAULTS if FAULTS[f].kind == kind]
    fig, ax = plt.subplots(figsize=(10, 7.2))
    y, ticks, labels, prev = 0, [], [], None
    for f in order:
        kind = FAULTS[f].kind
        if kind != prev:
            if prev is not None:
                y += 0.8
            ax.text(-16, y, KIND_LABEL[kind], ha="right", va="center", fontsize=9.5, fontweight="bold")
            y += 1
            prev = kind
        tv, cv = 100 * det.loc[f, "table"], 100 * det.loc[f, "change"]
        ax.plot([tv, cv], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter([tv], [y], s=70, color=SURFACE, edgecolor=INK2, linewidth=2, zorder=3)
        ax.scatter([cv], [y], s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        if abs(tv - cv) < 0.5:
            ax.text(max(tv, cv) + 3, y, f"{cv:.0f}% (both)", va="center", fontsize=8, color=INK)
        else:
            ax.text(cv + (3 if cv > tv else -3), y, f"{cv:.0f}%", va="center", ha="left" if cv > tv else "right",
                    fontsize=8, color=INK)
            ax.text(tv + (-3 if cv > tv else 3), y, f"{tv:.0f}%", va="center", ha="right" if cv > tv else "left",
                    fontsize=8, color=INK2)
        ticks.append(y)
        labels.append(FAULT_LABEL[f])
        y += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(y, -0.2)
    ax.set_xlim(-14, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("faulted deliveries caught: the check fired on the fault and was quiet on the clean month "
                  "(45 per fault)")
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=70, color=SURFACE, edgecolor=INK2, linewidth=2, label="any table check")
    ax.scatter([], [], s=70, color=BLUE, edgecolor=SURFACE, label="any change check")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9)
    ax.set_title("Fourteen planted faults: what each family of checks caught", loc="left", pad=28)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.9, bottom=0.09)
    credit(fig)
    save(fig, "fig3_faults.png")


def fig4(S: pd.DataFrame, F: pd.DataFrame, R: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = 100 * S.real_months_with_alarm.to_numpy()
    for col, color, label in (("all_faults_detected", BLUE, "all 14 faults"),
                              ("silent_detected", ORANGE, "the 10 silent faults")):
        yv = 100 * S[col].to_numpy()
        ax.plot(x, yv, color=color, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, zorder=3)
        ax.text(x[0] + 1.5, yv[0], label, color=INK, fontsize=9, va="center", ha="left")
    for xi, yi, k in zip(x, 100 * S.all_faults_detected, S.k, strict=True):
        ax.annotate(f"k = {k:g}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8,
                    color=INK2)
    tbl = F[F.family == "table"].groupby(["fault", "month"]).detected.any().mean()
    tr = R[R.family == "table"].groupby("month").fired.any().mean()
    ax.scatter([100 * tr], [100 * tbl], s=80, color=SURFACE, edgecolor=INK2, linewidth=2, zorder=3)
    ax.annotate("table checks\n(every delivery alarms)", (100 * tr, 100 * tbl), textcoords="offset points",
                xytext=(-8, 0), ha="right", va="center", fontsize=8.5, color=INK2)
    ax.axvline(100 * S.loc[S.k == 5, "real_months_with_alarm"].iloc[0], color=GRID, lw=6, zorder=0)
    ax.set_xlim(10, 104)
    ax.set_ylim(0, 100)
    ax.set_xlabel("real deliveries with at least one change alarm, 2020-01 to 2026-09")
    ax.set_ylabel("faulted deliveries caught")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    grid(ax, "both")
    ax.set_title("The threshold of the change checks: alarms bought against faults caught", loc="left")
    fig.text(0.01, 0.045, "The band marks k = 5, fixed before the real profiles were computed. "
             "A fault counts as caught only if the same check was quiet on the clean month.",
             fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.17)
    credit(fig)
    save(fig, "fig4_threshold.png")


def main():
    real, _ = read(PROF)
    R = pd.read_csv(RES / "real_alarms.csv")
    F = pd.read_csv(RES / "fault_detection.csv.gz")
    S = pd.read_csv(RES / "k_sweep.csv")
    fig1(real)
    fig2(R)
    fig3(F)
    fig4(S, F, R)


if __name__ == "__main__":
    main()
