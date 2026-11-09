"""Draws the three figures from results/figure_data.json and
results/performance.json. Run after analysis.py and performance.py."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#d9d8d4"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GREY = "#b8b7b2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

LABELS = {
    "daily (time zone only)": "Date taken from a UTC timestamp\nin the session time zone",
    "daily (coalesce only)": "Missing arrival delays\ncoalesced to zero",
    "route_summary": "Inner join to a dimension\nwith missing airports",
    "plane_age_delays": "Join on a column that means\nsomething else (year)",
    "weather_delays": "Hourly weather joined\nby day (fan-out)",
    "monthly_long_delays": "Delays compared as text\n(dep_delay > '60')",
    "departure_slot_delays": "Delays from subtracting\nHHMM clock readings",
    "idle (not in, on staging)": "NOT IN over a column\nthat contains NULLs",
}


def fig_detection(data: dict) -> None:
    rows = [r for r in data["detection"] if r[0] in LABELS]
    cols = [("build", "Builds\nwithout error"), ("schema", "Schema\ntests"),
            ("reconciliation", "Reconciliation\ntests"), ("unit", "Unit tests\non fixtures")]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for i, (label, engine, caught) in enumerate(rows):
        y = len(rows) - 1 - i
        for j, (key, _) in enumerate(cols):
            if key == "build":
                hit = engine != "ok"
            else:
                hit = caught[key]
            face = BLUE if hit else SURFACE
            ax.add_patch(Rectangle((j + 0.06, y + 0.08), 0.88, 0.84, facecolor=face,
                                   edgecolor=BLUE if hit else MUTED, linewidth=1.2))
            ax.text(j + 0.5, y + 0.5, "caught" if hit else "passed", ha="center", va="center",
                    color="white" if hit else INK_2, fontsize=9)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks([j + 0.5 for j in range(len(cols))], [c[1] for c in cols])
    ax.xaxis.tick_top()
    ax.set_yticks([len(rows) - 1 - i + 0.5 for i in range(len(rows))], [LABELS[r[0]] for r in rows], fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Which test layer fails on each defect\n(real data, session time zone UTC)",
                 loc="left", fontsize=11, pad=34)
    fig.text(0.01, 0.01, "Every defective model ran without error. Linting (SQLFluff, all rules) flagged none of the eight "
             "defects.\nSource: nycflights13; code at github.com/DiogoRibeiro7/Medium-Blog/analytics-sql-quality",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / "fig1_detection.png", dpi=200)
    plt.close(fig)


FAMILY = {
    "relational operator": "Comparison operators\nand boundaries",
    "boundary literal": "Comparison operators\nand boundaries",
    "join type": "Joins",
    "join key dropped": "Joins",
    "not exists -> not in": "Joins",
    "predicate dropped": "Filters, negations\nand CASE branches",
    "filter dropped": "Filters, negations\nand CASE branches",
    "negation dropped": "Filters, negations\nand CASE branches",
    "logical connector": "Filters, negations\nand CASE branches",
    "case branch dropped": "Filters, negations\nand CASE branches",
    "null check": "NULL handling",
    "null handling": "NULL handling",
    "aggregate": "Aggregates, casts\nand arithmetic",
    "count argument": "Aggregates, casts\nand arithmetic",
    "arithmetic": "Aggregates, casts\nand arithmetic",
    "distinct dropped": "Aggregates, casts\nand arithmetic",
    "cast dropped": "Aggregates, casts\nand arithmetic",
    "column swapped": "Column swapped\n(dep/arr, origin/dest)",
}


def fig_mutation(data: dict) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    scores = data["scores"]
    for i, (label, k, n, lo, hi) in enumerate(scores):
        y = len(scores) - 1 - i
        a1.plot([100 * lo, 100 * hi], [y, y], color=BLUE, linewidth=2, solid_capstyle="round")
        a1.plot([100 * k / n], [y], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        a1.text(100 * hi + 2, y, f"{k}/{n}", va="center", fontsize=9, color=INK_2)
    a1.set_yticks(range(len(scores)), [s[0] for s in reversed(scores)])
    a1.set_xlim(0, 105)
    a1.set_xlabel("mutants killed, % (95% Wilson interval)")
    a1.grid(axis="x", color=MUTED, linewidth=0.6)
    a1.set_axisbelow(True)
    a1.set_title("Mutation score as test layers are added", loc="left", fontsize=11)
    for s in ("top", "right", "left"):
        a1.spines[s].set_visible(False)

    fam = defaultdict(lambda: [0, 0, 0, 0])
    for operator, killed_first, killed_all, unit_only in data["operators"]:
        f = FAMILY[operator]
        if killed_first and not unit_only:
            fam[f][0] += 1
        elif unit_only:
            fam[f][1] += 1
        elif killed_all:
            fam[f][2] += 1
        else:
            fam[f][3] += 1
    order = sorted(fam, key=lambda f: -sum(fam[f]))
    parts = [("data tests or build", GREY), ("first unit tests", BLUE), ("boundary fixtures", AQUA),
             ("survived", ORANGE)]
    for i, f in enumerate(order):
        y = len(order) - 1 - i
        left = 0
        for (lab, colr), v in zip(parts, fam[f]):
            if v:
                a2.barh(y, v, left=left, color=colr, height=0.62, edgecolor=SURFACE, linewidth=2)
            left += v
        a2.text(left + 0.4, y, str(left), va="center", fontsize=9, color=INK_2)
    a2.set_yticks(range(len(order)), list(reversed(order)), fontsize=9)
    a2.set_xlabel("mutants")
    a2.set_title("What killed each mutant, by operator family", loc="left", fontsize=11)
    handles = [Rectangle((0, 0), 1, 1, color=c) for _, c in parts]
    a2.legend(handles, [p[0] for p in parts], frameon=False, fontsize=8.5, loc="lower right")
    for s in ("top", "right", "left"):
        a2.spines[s].set_visible(False)
    a2.tick_params(axis="y", length=0)
    fig.text(0.01, 0.01, "112 single-point mutants of the reviewed project, generated with sqlglot. "
             "Source: nycflights13; code at github.com/DiogoRibeiro7/Medium-Blog/analytics-sql-quality",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "fig2_mutation.png", dpi=200)
    plt.close(fig)


PERF = [
    ("star_fetch", "SELECT * vs three columns,\nfetched into pandas (DuckDB)"),
    ("star_inner", "SELECT * subquery vs named\ncolumns, inside one query (DuckDB)"),
    ("duck_wrapped", "CAST(time_hour AS DATE) = day\nvs a range (DuckDB, Parquet)"),
    ("duck_corr", "Correlated subquery vs\npre-aggregated join (DuckDB)"),
    ("lite_corr", "Correlated subquery vs\npre-aggregated join (SQLite)"),
    ("lite_win", "Window function vs\npre-aggregated join (SQLite)"),
    ("lite_wrapped", "date(time_hour) = day vs a range\non an indexed column (SQLite)"),
    ("distinct_fanout", "DISTINCT over a fanned-out join\nvs the join at the right grain"),
]


def fig_performance(perf: list[dict]) -> None:
    by_key = {r["key"]: r for r in perf}
    rows = [(k, lab) for k, lab in PERF if k in by_key]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for i, (k, lab) in enumerate(rows):
        r = by_key[k]
        y = len(rows) - 1 - i
        ax.plot([r["lo"], r["hi"]], [y, y], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.plot([r["ratio"]], [y], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ratio = r["ratio"]
        label = f"{ratio:.0f}×" if ratio >= 100 else f"{ratio:.1f}×" if ratio >= 10 else f"{ratio:.2f}×"
        ax.text(max(r["hi"], ratio) * 1.15, y, label, va="center", fontsize=9, color=INK_2)
    ax.axvline(1, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), [lab for _, lab in reversed(rows)], fontsize=9)
    ax.set_xlabel("time of the first form ÷ time of the second (median, 95% bootstrap interval; log scale)")
    ax.grid(axis="x", color=MUTED, linewidth=0.6, which="major")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    lo = min(r["lo"] for r in perf) / 1.5
    hi = max(r["hi"] for r in perf if r["key"] in dict(PERF)) * 3
    ax.set_xlim(lo, hi)
    ax.set_title("How much each rewrite saves, measured", loc="left", fontsize=11)
    fig.text(0.01, 0.01, "DuckDB rows: 10.1 million (2013 flights repeated 30 times). SQLite rows: one month or one year "
             "of flights.\nTimings from one machine; see results/performance.txt. "
             "Code at github.com/DiogoRibeiro7/Medium-Blog/analytics-sql-quality",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "fig3_performance.png", dpi=200)
    plt.close(fig)


def main() -> None:
    data = json.loads((RESULTS / "figure_data.json").read_text())
    fig_detection(data)
    fig_mutation(data)
    perf_file = RESULTS / "performance.json"
    if perf_file.exists():
        fig_performance(json.loads(perf_file.read_text()))


if __name__ == "__main__":
    main()
