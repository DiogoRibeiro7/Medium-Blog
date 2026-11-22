"""History, scenarios, Monte Carlo and figures for the Portugal 2035 debt article.

    python fetch_data.py && python build_data.py && python run_analysis.py

Writes results/results.json (every number quoted in the article) and figures/*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import debt_model as M

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
RES = HERE / "results"
SEED, N_SIM, BLOCK = 20261122, 10_000, 3

# Palette (light surface): reference categorical slots, in fixed order.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#2a78d6"]

plt.rcParams.update(
    {
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 11,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "legend.labelcolor": INK,
    }
)


def source_note(fig, text="Source: Eurostat (gov_10dd_edpt1, nama_10_gdp, irt_lt_mcby_a); author's calculations."):
    fig.text(0.01, 0.01, text, fontsize=8.5, color=INK2, ha="left", va="bottom")


# --------------------------------------------------------------------------- history
def hp_trend(y: np.ndarray, lam: float = 100.0) -> np.ndarray:
    n = len(y)
    d = np.zeros((n - 2, n))
    for k in range(n - 2):
        d[k, k : k + 3] = [1.0, -2.0, 1.0]
    return np.linalg.solve(np.eye(n) + lam * d.T @ d, y)


def history(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.loc[1995:2025].copy()
    debt_meur, gdp_meur, int_meur = p["GD_meur"], p["gdp_meur"], p["D41PAY_meur"]
    pb_meur = p["B9_meur"] + int_meur
    p["debt"] = 100 * debt_meur / gdp_meur
    p["g"] = gdp_meur.pct_change()
    p["c_growth"] = -p["debt"].shift(1) * p["g"] / (1 + p["g"])
    p["c_interest"] = 100 * int_meur / gdp_meur
    p["c_primary"] = -100 * pb_meur / gdp_meur
    p["c_sfa"] = 100 * (debt_meur.diff() + p["B9_meur"]) / gdp_meur
    p["d_debt"] = p["debt"].diff()
    p["residual"] = p["d_debt"] - p[["c_growth", "c_interest", "c_primary", "c_sfa"]].sum(axis=1)
    p["primary_balance"] = 100 * pb_meur / gdp_meur
    p["effective_rate"] = 100 * int_meur / debt_meur.shift(1)
    lvl = np.log(np.cumprod(1 + p["real_growth"].fillna(0) / 100))
    ok = p.index >= 1996
    p.loc[ok, "gap"] = 100 * (lvl[ok].values - hp_trend(lvl[ok].values))
    return p


def ols_slope(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    X = np.column_stack([np.ones(len(x)), x.values])
    beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    r = y.values - X @ beta
    se = np.sqrt(np.diag(r @ r / (len(x) - 2) * np.linalg.inv(X.T @ X)))
    return float(beta[1]), float(se[1]), len(x)


# ------------------------------------------------------------------------ Monte Carlo
def historical_shocks(h: pd.DataFrame, scarring: float, drop_years=()):
    hh = h.loc[1997:2025]
    gap = hh["gap"]
    x, y = gap.shift(1).dropna(), gap.iloc[1:]
    rho = float(np.dot(x, y) / np.dot(x, x))
    u = (gap - rho * gap.shift(1)).loc[1998:2025]
    X = np.column_stack([np.ones(len(hh)), hh["gap"].values])
    b, *_ = np.linalg.lstsq(X, hh["deflator_growth"].values, rcond=None)
    v = pd.Series(hh["deflator_growth"].values - X @ b, index=hh.index).loc[1998:2025]
    dm = hh["yield_10y"].diff().loc[1998:2025]
    sfa = hh["c_sfa"].loc[1998:2025]
    frame = pd.DataFrame({"u": u / (1 - scarring), "v": v, "dm": dm, "sfa": sfa})
    frame = frame.drop(index=[yr for yr in drop_years if yr in frame.index])
    frame = frame - frame.mean()  # zero-mean shocks: the draw adds risk, not drift
    return rho, float(b[1]), frame


def simulate(sc: M.Scenario, shocks: pd.DataFrame, n: int, seed: int):
    rng = np.random.default_rng(seed)
    arr = shocks.values
    T, H = len(arr), len(M.YEARS)
    debt, bal = np.empty((n, H)), np.empty((n, H))
    for k in range(n):
        idx = []
        while len(idx) < H:
            st = int(rng.integers(0, T - BLOCK + 1))
            idx.extend(range(st, st + BLOCK))
        draws = {y: tuple(arr[j]) for y, j in zip(M.YEARS, idx[:H], strict=True)}
        df = M.run(sc, draws=draws)
        debt[k], bal[k] = df["debt"].values, df["balance"].values
    return debt, bal


def summarise(debt: np.ndarray, bal: np.ndarray) -> dict:
    yrs = np.array(M.YEARS)
    j = {y: k for k, y in enumerate(M.YEARS)}
    first = np.array([yrs[np.argmax(r <= 60)] if (r <= 60).any() else 9999 for r in debt])
    pct = np.percentile(debt, [5, 10, 25, 50, 75, 90, 95], axis=0)
    return {
        "p_at_or_below_60_in_2035": float((debt[:, j[2035]] <= 60).mean()),
        "p_at_or_below_70_in_2035": float((debt[:, j[2035]] <= 70).mean()),
        "p_above_2027_level_in_2035": float((debt[:, j[2035]] > debt[:, j[2027]]).mean()),
        "p_deficit_above_3_by_2035": float((bal[:, : j[2035] + 1] < -3).any(axis=1).mean()),
        "median_2035": float(pct[3, j[2035]]),
        "p10_2035": float(pct[1, j[2035]]),
        "p90_2035": float(pct[5, j[2035]]),
        "p5_2035": float(pct[0, j[2035]]),
        "p95_2035": float(pct[6, j[2035]]),
        "median_first_year_at_or_below_60": float(np.median(first)),
        "p_not_at_60_by_2045": float((first == 9999).mean()),
        "p_at_or_below_60_by_year": {int(y): float((debt[:, k] <= 60).mean()) for k, y in enumerate(M.YEARS)},
        "fan": {
            q: {int(y): float(pct[i, k]) for k, y in enumerate(M.YEARS)}
            for i, q in enumerate(["p5", "p10", "p25", "p50", "p75", "p90", "p95"])
        },
    }


# ---------------------------------------------------------------------------- figures
def fig_decomposition(h: pd.DataFrame):
    d = h.loc[2009:2025]
    comps = [
        ("c_growth", "Nominal GDP growth", C[0]),
        ("c_interest", "Interest", C[1]),
        ("c_primary", "Primary balance", C[2]),
        ("c_sfa", "Other flows (stock-flow)", C[3]),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(d))
    pos, neg = np.zeros(len(d)), np.zeros(len(d))
    for col, lab, color in comps:
        v = d[col].values
        up, dn = np.where(v > 0, v, 0), np.where(v < 0, v, 0)
        ax.bar(x, up, bottom=pos, color=color, width=0.72, label=lab, edgecolor=SURFACE, linewidth=1.5)
        ax.bar(x, dn, bottom=neg, color=color, width=0.72, edgecolor=SURFACE, linewidth=1.5)
        pos += up
        neg += dn
    ax.scatter(
        x, d["d_debt"].values, s=46, color=INK, zorder=5, label="Change in debt ratio", edgecolor=SURFACE, linewidth=1.5
    )
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x, [str(y) if y % 2 == 1 else "" for y in d.index])
    ax.set_ylabel("Contribution to the yearly change, pp of GDP")
    ax.set_title("What moved Portugal's debt ratio each year, 2009–2025")
    ax.legend(loc="lower left", ncol=3, fontsize=9.5)
    source_note(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig1_what_moved_debt.png", dpi=200)
    plt.close(fig)


def fig_scenarios(paths: dict, labels: dict):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    years = [2026] + M.YEARS[:14]
    for k, key in enumerate(["A", "B", "R1", "R2", "R3"]):
        vals = [M.D0] + [paths[key][y] for y in M.YEARS[:14]]
        ax.plot(years, vals, color=C[k], linewidth=2.2, label=labels[key])
    ax.axhline(60, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(2026.1, 61.0, "60% reference value", color=INK2, fontsize=9.5)
    ax.axvline(2035, color=GRID, linewidth=1.5, zorder=0)
    ax.set_ylim(50, 101)
    ax.set_xlim(2026, 2040.3)
    ax.set_ylabel("Gross public debt, % of GDP")
    ax.set_title("Debt paths from the 2027 budget, with and without a recession")
    ax.legend(loc="upper right", fontsize=9.3)
    source_note(fig, "Source: Eurostat; Orçamento do Estado 2027; author's model. Recession scenarios hit in 2027.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig2_scenarios.png", dpi=200)
    plt.close(fig)


def fig_fan(fans: dict, titles: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    years = np.array(M.YEARS)
    for ax, key in zip(axes, ["A", "B"], strict=True):
        f = fans[key]

        def g(q, f=f):
            return np.array([f[q][int(y)] for y in years])

        ax.fill_between(years, g("p5"), g("p95"), color=BLUES[0], linewidth=0, label="5–95th percentile")
        ax.fill_between(years, g("p10"), g("p90"), color=BLUES[1], linewidth=0, label="10–90th")
        ax.fill_between(years, g("p25"), g("p75"), color=BLUES[2], linewidth=0, label="25–75th")
        ax.plot(years, g("p50"), color=BLUES[3], linewidth=2.2, label="Median")
        ax.axhline(60, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.axvline(2035, color=INK2, linewidth=0.8, alpha=0.5)
        ax.set_title(titles[key], fontsize=11.5)
        ax.set_xlim(2027, 2045)
    axes[0].set_ylabel("Gross public debt, % of GDP")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        "10,000 futures built from Portugal's own 1998–2025 shocks",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color=INK,
    )
    source_note(
        fig,
        "Source: Eurostat; author's model. "
        "Block bootstrap (3-year blocks) of growth, inflation, yield and stock-flow shocks.",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIG / "fig3_fan_charts.png", dpi=200)
    plt.close(fig)


def fig_probability_by_year(probs: dict, labels: dict):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for k, key in enumerate(["A", "B"]):
        yrs = sorted(probs[key])
        vals = [100 * probs[key][y] for y in yrs]
        ax.plot(yrs, vals, color=C[k], linewidth=2.2, marker="o", markersize=5, label=labels[key])
        y35 = 100 * probs[key][2035]
        ax.annotate(
            f"{y35:.0f}% in 2035",
            (2035, y35),
            xytext=(-10, 10) if k == 0 else (10, -18),
            textcoords="offset points",
            color=INK,
            fontsize=9.5,
            ha="right" if k == 0 else "left",
        )
    ax.axvline(2035, color=GRID, linewidth=1.5, zorder=0)
    ax.set_ylim(0, 100)
    ax.set_xlim(2027, 2045)
    ax.set_ylabel("Share of simulated futures at or below 60%")
    ax.set_title("The probability of being at or below 60% of GDP, year by year")
    ax.legend(loc="upper left", fontsize=9.5)
    source_note(fig, "Source: Eurostat; author's model, 10,000 block-bootstrap paths per policy.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig4_probability_by_year.png", dpi=200)
    plt.close(fig)


def fig_policy_dial(dial: dict, record: dict):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    lv = sorted(dial)
    ax.plot(lv, [100 * dial[x] for x in lv], color=C[0], linewidth=2.4, marker="o", markersize=6)
    marks = [
        (record["best_8yr_average"], "Portugal's best\n8-year average"),
        (1.8, "Commission's\nassumption"),
        (2.5, "2027\nbudget"),
        (record["max_single_year"], "Best single\nyear (2023)"),
    ]
    for n, (x, lab) in enumerate(marks):
        ax.axvline(x, color=INK2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        left = n == 0
        ax.text(
            x - 0.03 if left else x + 0.03, 96, lab, color=INK2, fontsize=8.8, va="top", ha="right" if left else "left"
        )
    ax.axhline(50, color=GRID, linewidth=1.5, zorder=0)
    ax.set_xlim(min(lv) - 0.05, max(lv) + 0.05)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Primary surplus held every year from 2028, % of GDP")
    ax.set_ylabel("Probability debt is at or below 60% in 2035")
    ax.set_title("How much primary surplus buys how much probability")
    source_note(fig, "Source: Eurostat; author's model, 4,000 block-bootstrap paths per level.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig5_policy_dial.png", dpi=200)
    plt.close(fig)


def write_summary(out: dict) -> None:
    """Rounded headline numbers, committed and compared in CI (results/summary.txt)."""
    h, det, mc = out["history"], out["deterministic"], out["monte_carlo"]["results"]
    lines = [f"identity residual below 1e-9 pp: {h['max_abs_identity_residual'] < 1e-9}"]
    for k, v in h["episodes"].items():
        lines.append(
            f"episode {k}: "
            + ", ".join(
                f"{c}={v[c]:.2f}" for c in ("d_debt", "c_growth", "c_interest", "c_primary", "c_sfa", "start", "end")
            )
        )
    pbr = h["primary_balance_record"]
    lines.append(
        f"primary balance: max {pbr['max_single_year']:.2f} ({pbr['max_year']}), "
        f"years >= 3 {pbr['years_at_or_above_3']}, best 8-yr avg {pbr['best_8yr_average']:.2f} "
        f"(to {pbr['best_8yr_end']}), mean 2023-25 {pbr['mean_2023_2025']:.2f}"
    )
    be = h["budget_elasticity"]
    lines.append(
        f"budget elasticity: all {be['all_years']['beta']:.2f} (se {be['all_years']['se']:.2f}), "
        f"excl. {be['excluding_2010_14_and_2020_21']['beta']:.2f} "
        f"(se {be['excluding_2010_14_and_2020_21']['se']:.2f})"
    )
    for k, v in det.items():
        d = v["debt"]
        lines.append(
            f"{k}: debt 2027 {d[2027]:.1f} 2028 {d[2028]:.1f} 2030 {d[2030]:.1f} "
            f"2035 {d[2035]:.1f} 2040 {d[2040]:.1f}; first<=60 {v['first_year_at_or_below_60']}; "
            f"peak deficit {v['peak_deficit']:.2f} ({v['peak_deficit_year']}); "
            f"required pb {v['required_primary_surplus_2028_2035']:.2f}"
        )
    for k, v in out["sensitivity"].items():
        for name, c in v.items():
            lines.append(
                f"sensitivity {k} {name}: debt 2035 {c['debt_2035']:.1f}, "
                f"peak deficit {c['peak_deficit']:.2f}, edp {c['edp']}"
            )
    lines.append(
        "recession timing (2035 debt): "
        + ", ".join(
            f"{k} {v['debt_2035']:.1f}" if isinstance(v, dict) else f"{k} {v:.1f}"
            for k, v in out["recession_timing"].items()
        )
    )
    for k, v in mc.items():
        keys = (
            "p_at_or_below_60_in_2035",
            "p_at_or_below_70_in_2035",
            "p_above_2027_level_in_2035",
            "p_deficit_above_3_by_2035",
            "median_2035",
            "p10_2035",
            "p90_2035",
            "median_first_year_at_or_below_60",
            "p_not_at_60_by_2045",
        )
        lines.append(f"monte carlo {k}: " + ", ".join(f"{q}={v[q]:.3f}" for q in keys))
        lines.append(
            f"monte carlo {k} without 2020: " + ", ".join(f"{q}={v['without_2020_in_pool'][q]:.3f}" for q in keys)
        )
        lines.append(
            f"monte carlo {k} p(<=60) by year: "
            + ", ".join(f"{y}={p:.3f}" for y, p in v["p_at_or_below_60_by_year"].items())
        )
    lines.append("policy dial p(<=60 in 2035): " + ", ".join(f"{x}={p:.3f}" for x, p in out["policy_dial"].items()))
    lines.append(
        "policy dial p(2035 > 2027): "
        + ", ".join(f"{x}={p:.3f}" for x, p in out["policy_dial_p_2035_above_2027"].items())
    )
    (RES / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------- main
def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    panel = pd.read_csv(HERE / "data" / "panel.csv", index_col="year")
    h = history(panel)
    out: dict = {
        "meta": {"seed": SEED, "n_sim": N_SIM, "block": BLOCK, "panel_years": [int(h.index.min()), int(h.index.max())]}
    }

    # History
    def ep(a, b):
        e = h.loc[a:b, ["d_debt", "c_growth", "c_interest", "c_primary", "c_sfa"]].sum()
        return {k: round(float(v), 2) for k, v in e.items()} | {
            "start": round(float(h.loc[a - 1, "debt"]), 2),
            "end": round(float(h.loc[b, "debt"]), 2),
        }

    pb = h["primary_balance"]
    roll8 = pb.rolling(8).mean()
    out["history"] = {
        "max_abs_identity_residual": float(h["residual"].abs().max()),
        "debt_by_year": {int(y): round(float(v), 2) for y, v in h["debt"].dropna().items()},
        "episodes": {
            "2009-2010": ep(2009, 2010),
            "2009-2013": ep(2009, 2013),
            "2020": ep(2020, 2020),
            "2021-2025": ep(2021, 2025),
            "2016-2019": ep(2016, 2019),
        },
        "primary_balance_record": {
            "max_single_year": round(float(pb.max()), 2),
            "max_year": int(pb.idxmax()),
            "years_at_or_above_3": [int(y) for y in pb[pb >= 2.95].index],
            "best_8yr_average": round(float(roll8.max()), 2),
            "best_8yr_end": int(roll8.idxmax()),
            "mean_2023_2025": round(float(pb.loc[2023:2025].mean()), 2),
            "by_year": {int(y): round(float(v), 2) for y, v in pb.dropna().items()},
        },
        "deflator_2022_2024": {int(y): float(h.loc[y, "deflator_growth"]) for y in (2022, 2023, 2024)},
        "balance_2008_2010": {int(y): float(h.loc[y, "B9_pcgdp"]) for y in (2008, 2009, 2010)},
        "effective_rate_2025": round(float(h.loc[2025, "effective_rate"]), 2),
    }
    dgap = h["gap"].diff()
    dbal = h["B9_pcgdp"].diff()
    df = pd.DataFrame({"x": dgap, "y": dbal}).dropna().loc[1997:2025]
    b_all = ols_slope(df["x"], df["y"])
    ex = df.drop(index=[y for y in (2010, 2011, 2012, 2013, 2014, 2020, 2021) if y in df.index])
    b_ex = ols_slope(ex["x"], ex["y"])
    out["history"]["budget_elasticity"] = {
        "all_years": {"beta": round(b_all[0], 2), "se": round(b_all[1], 2), "n": b_all[2]},
        "excluding_2010_14_and_2020_21": {"beta": round(b_ex[0], 2), "se": round(b_ex[1], 2), "n": b_ex[2]},
    }

    # Deterministic scenarios
    sc = M.scenarios()
    paths, labels, det = {}, {}, {}
    for key, s in sc.items():
        r = M.run(s)
        paths[key] = r["debt"].to_dict()
        labels[key] = s.label
        hit = r.index[r["debt"] <= 60]
        det[key] = {
            "label": s.label,
            "debt": {int(y): round(float(v), 2) for y, v in r["debt"].items()},
            "balance": {int(y): round(float(v), 2) for y, v in r["balance"].items()},
            "peak_deficit": round(float(-r["balance"].min()), 2),
            "peak_deficit_year": int(r["balance"].idxmin()),
            "edp_triggered": bool((r["balance"] < -3).any()),
            "first_year_at_or_below_60": int(hit[0]) if len(hit) else None,
            "required_primary_surplus_2028_2035": round(M.required_primary_surplus(s), 2),
            "effective_rate": {int(y): round(float(v), 2) for y, v in r["effective_rate"].items()},
            "nominal_growth": {int(y): round(float(v), 2) for y, v in r["nominal_growth"].items()},
            "snowball": {int(y): round(float(v), 2) for y, v in r["snowball"].items()},
            "real_growth": {int(y): round(float(v), 2) for y, v in r["real_growth"].items()},
        }
    out["deterministic"] = det

    # Sensitivities
    def cell(s, **kw):
        r = M.run(s, **kw)
        return {
            "debt_2035": round(float(r.loc[2035, "debt"]), 2),
            "peak_deficit": round(float(-r["balance"].min()), 2),
            "edp": bool((r["balance"] < -3).any()),
        }

    sens = {}
    for key in ("A", "R2"):
        s = sc[key]
        sens[key] = {
            "as_specified": cell(s),
            "elasticity_0.84": cell(s.copy(eps=0.84)),
            "elasticity_0.84_and_2009_size_support": cell(s.copy(eps=0.84, disc={2027: 2.0, 2028: 1.0})),
            "start_from_commission_86.5": cell(s, d0=86.5),
            "marginal_rate_plus_50bp": cell(s.copy(spread={y: s.spread.get(y, 0) + 0.5 for y in M.YEARS})),
            "more_scarring_h0.5": cell(s.copy(h=0.5)),
            "slower_repricing_0.10": cell(s.copy(s_reprice=0.10)),
        }
    out["sensitivity"] = sens
    out["recession_timing"] = {
        str(start): {"debt_2035": round(float(M.run(M.shift_scenario(sc["R2"], start)).loc[2035, "debt"]), 2)}
        for start in (2027, 2030, 2033)
    }
    out["recession_timing"]["baseline_2035"] = det["A"]["debt"][2035]

    # Monte Carlo
    rho, kappa, shocks = historical_shocks(h, sc["A"].h)
    rho_x, _, shocks_x = historical_shocks(h, sc["A"].h, drop_years=(2020,))
    mc, fans, probs = {}, {}, {}
    for key in ("A", "B"):
        s = sc[key].copy(rho=rho)
        d, b = simulate(s, shocks, N_SIM, SEED)
        summ = summarise(d, b)
        dx, bx = simulate(sc[key].copy(rho=rho_x), shocks_x, N_SIM, SEED)
        summ_x = summarise(dx, bx)
        mc[key] = {k: v for k, v in summ.items() if k not in ("fan",)}
        mc[key]["without_2020_in_pool"] = {
            k: v for k, v in summ_x.items() if k not in ("fan", "p_at_or_below_60_by_year")
        }
        fans[key], probs[key] = summ["fan"], summ["p_at_or_below_60_by_year"]
        mc[key]["fan"] = summ["fan"]
    out["monte_carlo"] = {
        "rho_estimated": round(rho, 2),
        "kappa_estimated": round(kappa, 2),
        "shock_years": [int(shocks.index.min()), int(shocks.index.max())],
        "shock_sd": {k: round(float(v), 2) for k, v in shocks.std().items()},
        "results": mc,
    }

    # Policy dial
    dial, dial_rise = {}, {}
    j27, j35 = M.YEARS.index(2027), M.YEARS.index(2035)
    for lvl in np.round(np.arange(1.0, 4.01, 0.25), 2):
        s = sc["A"].copy(rho=rho, pbs={y: (2.5 if y == 2027 else float(lvl)) for y in M.YEARS})
        d, _ = simulate(s, shocks, 4000, SEED + 1)
        dial[float(lvl)] = float((d[:, j35] <= 60).mean())
        dial_rise[float(lvl)] = float((d[:, j35] > d[:, j27]).mean())
    out["policy_dial"] = dial
    out["policy_dial_p_2035_above_2027"] = dial_rise
    lv = sorted(dial)
    for target in (0.5, 0.75):
        hit = [x for x in lv if dial[x] >= target]
        out[f"policy_dial_first_level_reaching_{int(target * 100)}pct"] = hit[0] if hit else None

    # Figures
    fig_decomposition(h)
    fig_scenarios(paths, {k: v for k, v in labels.items()})
    fig_fan(fans, {"A": "Primary surplus held at 2.5%", "B": "Drift to 1.8% structural primary surplus"})
    fig_probability_by_year(
        probs, {"A": "Primary surplus held at 2.5%", "B": "Drift to 1.8% structural primary surplus"}
    )
    fig_policy_dial(dial, out["history"]["primary_balance_record"])

    (RES / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_summary(out)
    a, bb = mc["A"], mc["B"]
    print(f"identity residual max: {out['history']['max_abs_identity_residual']:.2e}")
    print("episodes:", json.dumps(out["history"]["episodes"], indent=0))
    print("PB record:", {k: v for k, v in out["history"]["primary_balance_record"].items() if k != "by_year"})
    print("elasticity:", out["history"]["budget_elasticity"])
    for k, v in det.items():
        print(
            f"{k}: 2027 {v['debt'][2027]:.1f} 2035 {v['debt'][2035]:.1f} 2040 {v['debt'][2040]:.1f} "
            f"first<=60 {v['first_year_at_or_below_60']} peak def {v['peak_deficit']:.1f} "
            f"req PB {v['required_primary_surplus_2028_2035']:.2f}"
        )
    print("sensitivity:", json.dumps(sens, indent=0))
    print("timing:", out["recession_timing"])
    for key, r in (("A", a), ("B", bb)):
        print(
            key,
            {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in r.items()
                if k not in ("fan", "p_at_or_below_60_by_year", "without_2020_in_pool")
            },
        )
        print("   without 2020:", {k: round(v, 3) for k, v in r["without_2020_in_pool"].items()})
    print("dial:", {k: round(v, 3) for k, v in dial.items()})
    print("dial p(2035 > 2027):", {k: round(v, 3) for k, v in dial_rise.items()})
    print(
        "MC params:",
        out["monte_carlo"]["rho_estimated"],
        out["monte_carlo"]["kappa_estimated"],
        out["monte_carlo"]["shock_sd"],
    )


if __name__ == "__main__":
    main()
