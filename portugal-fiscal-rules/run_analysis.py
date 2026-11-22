"""Six fiscal rules for Portugal, 2028-2035, compared at the same average effort.

    python fetch_data.py && python build_data.py && python run_analysis.py

Writes results/results.json, results/summary.txt and figures/*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import figures
import rules as R

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
SEED, N_SIM, BLOCK = 20261123, 10_000, 3
TARGET = 2.5  # average primary balance 2028-2035 every rule is calibrated to
EFFORTS = (2.0, 2.5, 3.0)
MULTIPLIERS = (0.0, 0.5, 1.0)
GAP_MISS = (0.0, 0.5, 1.0)
KAPPAS = (0.0, 0.02, 0.04, 0.078, 0.12, 0.2, 0.3, 0.4)
PT_KAPPA = 0.078  # Portugal's long-run response to debt, 1996-2025 (portugal-fiscal-reaction)

MAIN = [
    R.Rule("fixed", "Fixed primary surplus", "fixed"),
    R.Rule("rigid", "Fixed surplus, no room for the cycle", "fixed", eps=0.0),
    R.Rule("debt_pt", "Debt rule at Portugal's strength", "debt", kappa=PT_KAPPA),
    R.Rule("debt_strong", "Strong debt rule", "debt", kappa=0.2),
    R.Rule("spending", "Spending path in 4-year nominal plans", "spending"),
    R.Rule("structural", "Structural balance, surprises repaid in 6 years", "structural", level=0.0, amortise=6),
]
VARIANTS = [
    R.Rule("structural_plain", "Structural balance, surprises not repaid", "structural", level=0.0),
    R.Rule("fixed_repaid", "Fixed surplus, surprises repaid in 6 years", "fixed", amortise=6),
    R.Rule("spending_unanchored", "Spending path never re-anchored", "spending", level=0.0, anchor=False),
    R.Rule("spending_indexed", "Spending path indexed to inflation", "spending", level=0.0, indexed=True),
    R.Rule("spending_1y", "Spending path reset every year", "spending", level=0.0, plan_years=1),
    R.Rule("spending_6y", "Spending path in 6-year plans", "spending", level=0.0, plan_years=6),
]
AXES = ("p90_2035", "p_breach", "p_forced_tightening")


# ----------------------------------------------------------------------------- data
def portugal() -> pd.DataFrame:
    p = pd.read_csv(HERE / "data" / "portugal.csv").set_index("year").sort_index()
    p["g"] = 100 * p["gdp_nom"].pct_change()
    p["deflator_growth"] = 100 * p["deflator"].pct_change()
    p["sfa"] = p["debt"] - p["debt"].shift(1) / (1 + p["g"] / 100) - p["interest"] + p["pb"]
    return p


def fiscal_surprises(p: pd.DataFrame) -> pd.Series:
    """Residuals of Portugal's fiscal reaction function, 1996-2025 (same regression as portugal-fiscal-reaction)."""
    s = p.loc[1996:2025]
    X = np.column_stack(
        [
            np.ones(len(s)),
            p["debt"].shift(1).loc[1996:2025],
            s["gap"],
            p["pb"].shift(1).loc[1996:2025],
            (s.index == 2020).astype(float),
            (s.index == 2021).astype(float),
        ]
    )
    b = np.linalg.lstsq(X, s["pb"].values, rcond=None)[0]
    return pd.Series(s["pb"].values - X @ b, index=s.index)


def historical_shocks(p: pd.DataFrame, e: pd.Series, h: float):
    """Joint yearly shocks (gap innovation, deflator, yield, stock-flow, fiscal surprise), 1998-2025, demeaned."""
    hh = p.loc[1996:2025]
    gap = hh["gap"]
    x, y = gap.shift(1).dropna(), gap.iloc[1:]
    rho_gap = float(np.dot(x, y) / np.dot(x, x))
    u = (gap - rho_gap * gap.shift(1)).loc[1998:2025]
    X = np.column_stack([np.ones(len(hh)), hh["gap"].values])
    b = np.linalg.lstsq(X, hh["deflator_growth"].values, rcond=None)[0]
    v = pd.Series(hh["deflator_growth"].values - X @ b, index=hh.index).loc[1998:2025]
    frame = pd.DataFrame(
        {
            "u": u / (1 - h),
            "v": v,
            "dm": hh["ltr"].diff().loc[1998:2025],
            "sfa": hh["sfa"].loc[1998:2025],
            "e": e.reindex(range(1998, 2026)),
        }
    ).dropna()
    return rho_gap, frame - frame.mean()


def block_draws(shocks: pd.DataFrame, n: int, seed: int, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Common random numbers: one (n, H, 5) array of resampled history, shared by every rule."""
    rng = np.random.default_rng(seed)
    arr = shocks[["u", "v", "dm", "sfa", "e"]].values
    T = len(arr)
    starts = rng.integers(0, T - block + 1, size=(n, -(-R.H // block)))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, : R.H]
    return arr[idx], np.asarray(shocks.index)[idx]


# ----------------------------------------------------------------------- comparison
def compare(rules: list[R.Rule], macro: R.Macro, draws: np.ndarray, target: float) -> dict:
    out = {}
    for r in rules:
        c = R.calibrate(r, macro, draws, target)
        sim = R.simulate(c, macro, draws)
        out[r.key] = {"label": r.label, "level": c.level, **R.metrics(sim)}
    return out


def dominated(res: dict, keys: list[str]) -> dict[str, list[str]]:
    """For each rule, the rules that are at least as good on all three axes and better on one."""
    out = {}
    for a in keys:
        out[a] = [
            b
            for b in keys
            if b != a and all(res[b][m] <= res[a][m] for m in AXES) and any(res[b][m] < res[a][m] for m in AXES)
        ]
    return out


def weight_map(res: dict, keys: list[str], step: float = 0.02, axes: tuple[str, ...] = AXES) -> dict:
    """Best rule for each weighting of the three axes, each axis rescaled to 0-1 across the rules.

    Every axis is a loss (lower is better); "p_not_60" is one minus the chance of 60% in 2035."""
    vals = np.array([[_loss(res[k], m) for m in axes] for k in keys])
    lo, hi = vals.min(axis=0), vals.max(axis=0)
    z = (vals - lo) / np.where(hi > lo, hi - lo, 1.0)
    grid, share = [], {k: 0 for k in keys}
    m = int(round(1 / step))
    for a in range(m + 1):
        for b in range(m + 1 - a):
            w = np.array([a, b, m - a - b]) / m
            best = keys[int(np.argmin(z @ w))]
            grid.append({"w": [float(x) for x in w], "best": best})
            share[best] += 1
    total = sum(share.values())
    return {"grid": grid, "share": {k: v / total for k, v in share.items()}, "scaled": z.tolist()}


def _loss(r: dict, m: str) -> float:
    return 1 - r["p_at_or_below_60_2035"] if m == "p_not_60" else r[m]


def one_future(draws: np.ndarray, years: np.ndarray) -> int:
    """A future whose first resampled block is the 2009-2011 crisis: the same bad draw for every rule."""
    hit = np.where(years[:, 1] == 2009)[0]
    return int(hit[0])


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    p = portugal()
    e = fiscal_surprises(p)
    macro = R.Macro()
    rho_gap, shocks = historical_shocks(p, e, macro.h)
    macro = macro.copy(rho_gap=rho_gap)
    draws, years = block_draws(shocks, N_SIM, SEED)
    keys = [r.key for r in MAIN]

    det = {}
    for r in MAIN:
        c = R.calibrate(r, macro, draws, TARGET)
        s = R.simulate(c, macro, R.zero_draws())
        det[r.key] = {
            "debt": {y: float(s["debt"][0, t]) for t, y in enumerate(R.YEARS)},
            "pb": {y: float(s["pb"][0, t]) for t, y in enumerate(R.YEARS)},
        }

    main_res = compare(MAIN, macro, draws, TARGET)
    variants = compare(VARIANTS, macro, draws, TARGET)

    by_effort = {str(t): compare(MAIN, macro, draws, t) for t in EFFORTS}
    by_mu = {str(m): compare(MAIN, macro.copy(mu=m), draws, TARGET) for m in MULTIPLIERS}
    by_miss = {str(m): compare(MAIN, macro.copy(gap_miss=m), draws, TARGET) for m in GAP_MISS}
    by_block = {}
    for blk in (1, 5):
        dr, _ = block_draws(shocks, N_SIM, SEED + blk, block=blk)
        by_block[str(blk)] = compare(MAIN, macro, dr, TARGET)
    by_seed = compare(MAIN, macro, block_draws(shocks, N_SIM, SEED + 99)[0], TARGET)
    sweep = {
        f"{k:.3f}": compare([R.Rule("k", "debt rule", "debt", kappa=k)], macro, draws, TARGET)["k"] for k in KAPPAS
    }

    k = one_future(draws, years)
    story = {"years_drawn": [int(x) for x in years[k]], "rules": {}}
    for r in MAIN:
        c = R.calibrate(r, macro, draws, TARGET)
        s = R.simulate(c, macro, draws[k : k + 1])
        story["rules"][r.key] = {
            m: {y: float(s[m][0, t]) for t, y in enumerate(R.YEARS)} for m in ("debt", "pb", "bal", "gap", "sp")
        }

    res = {
        "meta": {
            "seed": SEED,
            "n_sim": N_SIM,
            "block": BLOCK,
            "target_effort": TARGET,
            "window": list(R.WINDOW),
            "rho_gap": rho_gap,
            "mu": macro.mu,
            "gap_miss": macro.gap_miss,
            "surprise_sd": float(shocks["e"].std()),
            "surprise_years": {int(y): float(v) for y, v in e.items()},
        },
        "deterministic": det,
        "main": main_res,
        "dominated_by": dominated(main_res, keys),
        "weights": weight_map(main_res, keys),
        "weights_target": weight_map(main_res, keys, axes=("p_not_60", "p_breach", "p_forced_tightening")),
        "variants": variants,
        "by_effort": by_effort,
        "by_multiplier": by_mu,
        "by_gap_miss": by_miss,
        "by_block": by_block,
        "by_seed": by_seed,
        "kappa_sweep": sweep,
        "one_future": story,
    }
    (RES / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    write_summary(res, keys)
    figures.draw_all(res, MAIN, FIG)
    print((RES / "summary.txt").read_text())


def write_summary(res: dict, keys: list[str]) -> None:
    L = []
    m = res["meta"]
    L.append(f"target effort {m['target_effort']:.1f} over {m['window'][0]}-{m['window'][1]}; n={m['n_sim']}, "
             f"block {m['block']}, mu {m['mu']:.1f}, gap miss {m['gap_miss']:.1f}, rho_gap {m['rho_gap']:.3f}, "
             f"surprise sd {m['surprise_sd']:.2f}")  # fmt: skip

    def row(name, r):
        return (
            f"{name:<17} level={r['level']:7.3f} effort={r['effort']:.3f} p60={r['p_at_or_below_60_2035']:.3f} "
            f"med={r['median_2035']:.1f} p10={r['p10_2035']:.1f} p90={r['p90_2035']:.1f} "
            f"breach={r['p_breach']:.3f} byrs={r['breach_years']:.2f} forced={r['p_forced_tightening']:.3f} "
            f"absdsp={r['mean_abs_discretionary_change']:.2f} gap={r['mean_gap']:.2f} worst={r['mean_worst_gap']:.2f}"
        )

    for k in keys:
        L.append(row(k, res["main"][k]))
    for k, r in res["variants"].items():
        L.append(row(k, r))
    for k in keys:
        d = res["deterministic"][k]
        L.append(f"deterministic {k}: debt 2030 {d['debt'][2030]:.1f} 2035 {d['debt'][2035]:.1f} "
                 f"2045 {d['debt'][2045]:.1f}; pb 2028 {d['pb'][2028]:.2f} 2035 {d['pb'][2035]:.2f}")  # fmt: skip
    L.append("dominated by: " + "; ".join(f"{a}={','.join(b) or '-'}" for a, b in res["dominated_by"].items()))
    L.append("weight share (bad tail): " + ", ".join(f"{a}={v:.3f}" for a, v in res["weights"]["share"].items()))
    L.append("weight share (chance of 60%): "
             + ", ".join(f"{a}={v:.3f}" for a, v in res["weights_target"]["share"].items()))  # fmt: skip
    for t, block in res["by_effort"].items():
        L.append(f"effort {t}: " + ", ".join(
            f"{k}=({block[k]['p90_2035']:.1f},{block[k]['p_breach']:.3f},{block[k]['p_forced_tightening']:.3f})"
            for k in keys))  # fmt: skip
    for t, block in res["by_multiplier"].items():
        L.append(f"mu {t}: " + ", ".join(
            f"{k}=({block[k]['p90_2035']:.1f},{block[k]['p_breach']:.3f},{block[k]['p_forced_tightening']:.3f})"
            for k in keys))  # fmt: skip
    for t, block in res["by_gap_miss"].items():
        L.append(f"gap miss {t}: " + ", ".join(
            f"{k}=({block[k]['p90_2035']:.1f},{block[k]['p_breach']:.3f},{block[k]['p_forced_tightening']:.3f})"
            for k in keys))  # fmt: skip
    for t, block in res["by_block"].items():
        L.append(f"block {t}: " + ", ".join(
            f"{k}=({block[k]['p90_2035']:.1f},{block[k]['p_breach']:.3f},{block[k]['p_forced_tightening']:.3f})"
            for k in keys))  # fmt: skip
    L.append("other seed: " + ", ".join(
        f"{k}=({res['by_seed'][k]['p90_2035']:.1f},{res['by_seed'][k]['p_breach']:.3f},"
        f"{res['by_seed'][k]['p_forced_tightening']:.3f})" for k in keys))  # fmt: skip
    L.append("kappa sweep: " + ", ".join(
        f"{k}=(lvl {v['level']:.2f}, p90 {v['p90_2035']:.1f}, breach {v['p_breach']:.3f}, "
        f"forced {v['p_forced_tightening']:.3f}, p60 {v['p_at_or_below_60_2035']:.3f})"
        for k, v in res["kappa_sweep"].items()))  # fmt: skip
    st = res["one_future"]
    L.append("one future draws: " + ",".join(str(y) for y in st["years_drawn"][:9]))
    for k in keys:
        s = st["rules"][k]
        w = range(2028, 2036)
        L.append(f"one future {k}: debt 2029 {s['debt'][2029]:.1f} 2035 {s['debt'][2035]:.1f}; "
                 f"pb 2028 {s['pb'][2028]:.2f}; min bal {min(s['bal'][y] for y in w):.2f}; "
                 f"max dsp {max(s['sp'][y] - s['sp'][y - 1] for y in w):.2f}")  # fmt: skip
    (RES / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
