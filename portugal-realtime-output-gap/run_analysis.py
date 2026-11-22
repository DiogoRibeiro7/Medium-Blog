"""How wrong were the Commission's real-time output gaps, and what does that do to fiscal rules?

    python fetch_data.py && python build_data.py && python run_analysis.py

Part 1 compares seventeen archived AMECO releases (2011-2021) with the current one (autumn 2026):
errors by age, the share of each year's cyclical surprise missed, compression, the split between
growth and potential-growth revisions, and how often the verdict on structural effort changes.
Part 2 replays the six fiscal rules of portugal-fiscal-rules with the real-time errors estimated in
part 1. Writes results/results.json, results/summary.txt and figures/*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import figures
import revisions as V
import rules as R

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
SEED, N_SIM, BLOCK, BOOT = 20261124, 10_000, 3, 2_000
TARGET = 2.5
AGES = (-1, 0, 1, 2, 3, 4, 5)
PT_KAPPA = 0.078

RULES = [
    R.Rule("fixed", "Fixed primary surplus", "fixed"),
    R.Rule("rigid", "Fixed surplus, no room for the cycle", "fixed", eps=0.0),
    R.Rule("debt_pt", "Debt rule at Portugal's strength", "debt", kappa=PT_KAPPA),
    R.Rule("debt_strong", "Strong debt rule", "debt", kappa=0.2),
    R.Rule("spending", "Spending path in 4-year nominal plans", "spending"),
    R.Rule("structural", "Structural balance, surprises repaid in 6 years", "structural", level=0.0, amortise=6),
]


# ------------------------------------------------------------------------ Portugal's shocks
def portugal() -> pd.DataFrame:
    p = pd.read_csv(HERE / "data" / "portugal.csv").set_index("year").sort_index()
    p["g"] = 100 * p["gdp_nom"].pct_change()
    p["deflator_growth"] = 100 * p["deflator"].pct_change()
    p["sfa"] = p["debt"] - p["debt"].shift(1) / (1 + p["g"] / 100) - p["interest"] + p["pb"]
    return p


def fiscal_surprises(p: pd.DataFrame) -> pd.Series:
    """Residuals of Portugal's fiscal reaction function, 1996-2025 (as in portugal-fiscal-reaction)."""
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


def block_draws(shocks: pd.DataFrame, n: int, seed: int, block: int = BLOCK) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = shocks[["u", "v", "dm", "sfa", "e"]].values
    T = len(arr)
    starts = rng.integers(0, T - block + 1, size=(n, -(-R.H // block)))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, : R.H]
    return arr[idx]


# --------------------------------------------------------------------------- part 1
def pooled_rho(v: pd.DataFrame) -> float:
    """AR(1) coefficient of the latest output gap, pooled over the 27 countries, 1996-2025."""
    lt = V.latest(v, "gap").reset_index()
    lt = lt[(lt.year >= 1995) & (lt.year <= 2025)].sort_values(["country", "year"])
    lt["prev"] = lt.groupby("country")["value"].shift(1)
    lt = lt.dropna()
    return float(np.dot(lt["prev"], lt["value"]) / np.dot(lt["prev"], lt["prev"]))


def own_rho(lt_gap: pd.Series) -> dict[str, float]:
    """Each country's AR(1) coefficient of the latest output gap, 1996-2025."""
    lt = lt_gap.reset_index()
    lt = lt[(lt.year >= 1995) & (lt.year <= 2025)].sort_values(["country", "year"])
    lt["prev"] = lt.groupby("country")["value"].shift(1)
    lt = lt.dropna()
    return {c: float(np.dot(g["prev"], g["value"]) / np.dot(g["prev"], g["prev"])) for c, g in lt.groupby("country")}


def own_innovation(df: pd.DataFrame, lt_gap: pd.Series) -> pd.DataFrame:
    """The year's innovation in each country's latest gap, with the country's own persistence."""
    rho = own_rho(lt_gap)
    out = df.copy()
    prev = np.array([lt_gap.get((c, y - 1), np.nan) for c, y in zip(out.country, out.year, strict=True)])
    out["innovation"] = out["latest"].to_numpy() - np.array([rho[c] for c in out.country]) * prev
    return out.dropna(subset=["innovation"]).reset_index(drop=True)


def error_process(df: pd.DataFrame, boot_seed: int | None) -> dict:
    """Bias, share missed, residual noise and its persistence, for one sample of estimates."""
    m = V.miss_share(df)
    bias = -m["const"]
    out = {
        "n": m["n"],
        "bias": bias,
        "c": m["c"],
        "c_se": m["c_se"],
        "noise_sd": m["resid_sd"],
        "noise_rho": max(V.residual_persistence(df, m["c"], bias)["rho"], 0.0),
        "lam": V.attenuation(df)["lam"],
    }
    if boot_seed is not None:
        out["c_ci90"] = V.cluster_bootstrap_c(df, BOOT, boot_seed)
    return out


def part_one(v: pd.DataFrame, rho: float) -> dict:
    lt_gap = V.latest(v, "gap")
    ages, by_country = {}, {}
    for a in AGES:
        e = V.estimates(v, "gap", a)
        ages[a] = {"eu": V.summary(e["error"]), "pt": V.summary(e.loc[e.country == "PRT", "error"])}
        if a == 0:
            by_country = {c: float(g["error"].abs().mean()) for c, g in e.groupby("country")}
    process, robust = {}, {}
    for a in (-1, 0):
        e = own_innovation(V.estimates(v, "gap", a), lt_gap)
        process[f"eu_age{a}"] = error_process(e, SEED + a + 10)
        process[f"pt_age{a}"] = error_process(e[e.country == "PRT"], None)
        robust[f"age{a}"] = {
            "without_2021_release": V.miss_share(e[e.vintage_year != 2021]),
            "without_ireland": V.miss_share(e[e.country != "IRL"]),
            "pooled_rho": V.miss_share(V.with_innovation(V.estimates(v, "gap", a), lt_gap, rho)),
        }
    d = V.decomposition(v)
    cov = np.cov(d["dgap_rev"], d["growth_rev"])[0, 1] / d["dgap_rev"].var()
    covp = -np.cov(d["dgap_rev"], d["potential_rev"])[0, 1] / d["dgap_rev"].var()
    X = np.column_stack([np.ones(len(d)), d["growth_rev"], d["potential_rev"]])
    lv, _, _ = V.ols(d["level_rev"].to_numpy(), X)
    effort = {}
    for var in ("sb", "cab"):
        s = V.structural_effort(v, var)
        effort[var] = {
            "n": len(s),
            "error": V.summary(s["error"]),
            "flip_share": float(s["flip"].mean()),
            "pt": s[s.country == "PRT"][["year", "realtime", "now", "flip"]].to_dict("records"),
            "pt_flip_share": float(s.loc[s.country == "PRT", "flip"].mean()),
            "points": s[["country", "year", "realtime", "now"]].to_dict("records"),
        }
    pt_paths = {}
    for (vy, season), g in v[(v["var"] == "gap") & (v.country == "PRT")].groupby(["vintage_year", "season"]):
        if season == "autumn":
            pt_paths[int(vy)] = {int(y): float(x) for y, x in zip(g.year, g.value, strict=True) if 2003 <= y <= vy + 1}
    scatter = {}
    for a in (-1, 0):
        e = V.estimates(v, "gap", a)
        scatter[a] = e[["country", "year", "estimate", "latest"]].to_dict("records")
    rank = sorted(by_country, key=by_country.get)
    return {
        "rho_pooled": rho,
        "rho_pt": own_rho(lt_gap)["PRT"],
        "ages": ages,
        "process": process,
        "robust": robust,
        "mae_age0_by_country": by_country,
        "pt_rank_age0": rank.index("PRT") + 1,
        "n_countries": len(rank),
        "decomposition": {
            "n": len(d),
            "share_growth": float(cov),
            "share_potential": float(covp),
            "level_on_growth": float(lv[1]),
            "level_on_potential": float(lv[2]),
            "growth_rev": V.summary(d["growth_rev"]),
            "potential_rev": V.summary(d["potential_rev"]),
            "pt": d[d.country == "PRT"].round(4).to_dict("records"),
        },
        "effort": effort,
        "pt_paths": pt_paths,
        "scatter": scatter,
    }


# --------------------------------------------------------------------------- part 2
def scenarios(proc: dict) -> dict[str, dict]:
    eu, pt, now = proc["eu_age-1"], proc["pt_age-1"], proc["eu_age0"]

    def from_(p, label, scale=1.0):
        return {
            "label": label,
            "gap_miss": scale * p["c"],
            "gap_bias": p["bias"],
            "gap_noise_sd": scale * p["noise_sd"],
            "gap_noise_rho": p["noise_rho"],
        }

    return {
        "perfect": {
            "label": "Cycle measured exactly",
            "gap_miss": 0.0,
            "gap_bias": 0.0,
            "gap_noise_sd": 0.0,
            "gap_noise_rho": 0.0,
        },  # fmt: skip
        "assumed": {
            "label": "Half of each surprise missed (earlier assumption)",
            "gap_miss": 0.5,
            "gap_bias": 0.0,
            "gap_noise_sd": 0.0,
            "gap_noise_rho": 0.0,
        },  # fmt: skip
        "eu_now": from_(now, "EU errors, current-year estimate"),
        "eu_budget": from_(eu, "EU errors, budget-time forecast"),
        "pt_budget": from_(pt, "Portugal's errors, budget-time forecast"),
        "eu_half": from_(eu, "EU budget-time errors halved", 0.5),
    }


def compare(macro: R.Macro, draws: np.ndarray, noise: np.ndarray) -> dict:
    out = {}
    for r in RULES:
        c = R.calibrate(r, macro, draws, TARGET, noise=noise)
        sim = R.simulate(c, macro, draws, noise)
        out[r.key] = {"label": r.label, "level": c.level, **R.metrics(sim)}
    return out


def part_two(proc: dict) -> dict:
    p = portugal()
    e = fiscal_surprises(p)
    base = R.Macro()
    rho_gap, shocks = historical_shocks(p, e, base.h)
    base = base.copy(rho_gap=rho_gap)
    draws = block_draws(shocks, N_SIM, SEED)
    noise = np.random.default_rng(SEED + 1).standard_normal((N_SIM, R.H))
    sc = scenarios(proc)
    res = {}
    for k, s in sc.items():
        m = base.copy(**{x: s[x] for x in ("gap_miss", "gap_bias", "gap_noise_sd", "gap_noise_rho")})
        res[k] = {"scenario": s, "rules": compare(m, draws, noise)}
    # a bias the government does not know about: rules keep the level set under exact measurement
    unknown = {}
    m_pt = base.copy(**{x: sc["pt_budget"][x] for x in ("gap_miss", "gap_bias", "gap_noise_sd", "gap_noise_rho")})
    for r in RULES:
        lvl = res["perfect"]["rules"][r.key]["level"]
        sim = R.simulate(r.copy(level=lvl), m_pt, draws, noise)
        unknown[r.key] = R.metrics(sim)
    # the share missed, swept with the EU noise and bias held at their estimates
    sweep = {}
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        m = base.copy(gap_miss=c, gap_bias=sc["eu_budget"]["gap_bias"], gap_noise_sd=sc["eu_budget"]["gap_noise_sd"],
                      gap_noise_rho=sc["eu_budget"]["gap_noise_rho"])  # fmt: skip
        sweep[f"{c:.2f}"] = compare(m, draws, noise)
    return {"rho_gap": rho_gap, "scenarios": res, "unknown_bias": unknown, "miss_sweep": sweep}


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    v = pd.read_csv(HERE / "data" / "vintages.csv")
    rho = pooled_rho(v)
    one = part_one(v, rho)
    two = part_two(one["process"])
    res = {"meta": {"seed": SEED, "n_sim": N_SIM, "block": BLOCK, "boot": BOOT, "target": TARGET}, "part1": one,
           "part2": two}  # fmt: skip
    (RES / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    write_summary(res)
    figures.draw_all(res, RULES, FIG)
    print((RES / "summary.txt").read_text())


def _triple(r: dict) -> str:
    return f"({r['p90_2035']:.1f},{r['p_breach']:.3f},{r['p_forced_tightening']:.3f})"


def _rule_line(rk: str, r: dict) -> str:
    return (
        f"  {rk:<12} level {r['level']:7.3f} p60 {r['p_at_or_below_60_2035']:.3f} p90 {r['p90_2035']:.1f} "
        f"breach {r['p_breach']:.3f} forced {r['p_forced_tightening']:.3f} "
        f"absdsp {r['mean_abs_discretionary_change']:.2f} worst gap {r['mean_worst_gap']:.2f}"
    )


def write_summary(res: dict) -> None:
    one, two = res["part1"], res["part2"]
    L = [
        f"pooled rho {one['rho_pooled']:.3f}; Portugal rho (shocks) {two['rho_gap']:.3f}; "
        f"Portugal own rho (vintage latest) {one['rho_pt']:.3f}"
    ]
    for a, s in one["ages"].items():
        eu, pt = s["eu"], s["pt"]
        L.append(
            f"age {a}: EU n={eu['n']} bias {eu['bias']:.2f} sd {eu['sd']:.2f} mae {eu['mae']:.2f} "
            f"iqr {eu['p25']:.2f}..{eu['p75']:.2f} | PT n={pt['n']} bias {pt['bias']:.2f} sd {pt['sd']:.2f} "
            f"mae {pt['mae']:.2f}"
        )
    for k, p in one["process"].items():
        ci = f" ci90 {p['c_ci90'][0]:.2f}..{p['c_ci90'][1]:.2f}" if "c_ci90" in p else ""
        L.append(
            f"process {k}: n={p['n']} bias {p['bias']:.2f} c {p['c']:.2f} (se {p['c_se']:.2f}){ci} "
            f"noise sd {p['noise_sd']:.2f} rho {p['noise_rho']:.2f} lam {p['lam']:.2f}"
        )
    for a, r in one["robust"].items():
        parts = [f"{k} c {x['c']:.2f} (se {x['c_se']:.2f}, n={x['n']})" for k, x in r.items()]
        L.append(f"robust {a}: " + ", ".join(parts))
    mae = one["mae_age0_by_country"]
    L.append(
        f"Portugal age-0 MAE rank {one['pt_rank_age0']} of {one['n_countries']} (1 = smallest); "
        f"PT mae {mae['PRT']:.2f}, median country {np.median(list(mae.values())):.2f}"
    )
    d = one["decomposition"]
    g = d["growth_rev"]
    L.append(
        f"decomposition n={d['n']}: share growth {d['share_growth']:.2f}, potential {d['share_potential']:.2f}; "
        f"level rev on growth {d['level_on_growth']:.2f}, on potential {d['level_on_potential']:.2f}; "
        f"growth rev median {g['median']:.2f} iqr {g['p25']:.2f}..{g['p75']:.2f}"
    )
    for r in d["pt"]:
        L.append(
            f"  PT {r['year']}: growth rt {r['growth_rt']:.2f} now {r['growth_latest']:.2f}; "
            f"potential rt {r['potential_rt']:.2f} now {r['potential_latest']:.2f}; "
            f"gap rt {r['gap_rt']:.2f} now {r['gap_latest']:.2f}"
        )
    for var, ef in one["effort"].items():
        L.append(
            f"effort {var}: n={ef['n']} error sd {ef['error']['sd']:.2f} mae {ef['error']['mae']:.2f} "
            f"flip share {ef['flip_share']:.3f}; PT flip share {ef['pt_flip_share']:.3f}"
        )
        for r in ef["pt"]:
            L.append(f"  PT {r['year']}: real time {r['realtime']:.2f}, now {r['now']:.2f}, flip {r['flip']}")
    for k, s in two["scenarios"].items():
        sc = s["scenario"]
        L.append(
            f"scenario {k}: miss {sc['gap_miss']:.2f} bias {sc['gap_bias']:.2f} "
            f"noise {sc['gap_noise_sd']:.2f} rho {sc['gap_noise_rho']:.2f}"
        )
        L += [_rule_line(rk, r) for rk, r in s["rules"].items()]
    for rk, r in two["unknown_bias"].items():
        L.append(
            f"unknown PT bias {rk:<12} effort {r['effort']:.3f} p60 {r['p_at_or_below_60_2035']:.3f} "
            f"p90 {r['p90_2035']:.1f} breach {r['p_breach']:.3f} forced {r['p_forced_tightening']:.3f}"
        )
    for c, block in two["miss_sweep"].items():
        L.append(f"miss {c}: " + ", ".join(f"{k}={_triple(r)}" for k, r in block.items()))
    (RES / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
