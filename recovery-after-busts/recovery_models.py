"""Never-recover shares by tradability and integration stage (mixture cure models).

    python recovery_models.py [--boot 300]

Two samples, both restricted to the main episodes (true peak observed, no CEE transition):

1. A*64 industries, EU-27, 1995-2025 (``results/episodes_all.csv``). Almost every trough
   here falls in the single-market or euro stage, so this sample compares tradable and
   non-tradable industries *within* an open market, with many episodes.
2. AMECO sectors, 1960-2025 (``results/long_episodes.csv``). Fewer, coarser episodes, but
   they span the stages, so the tradable gap can be compared *across* stages and with
   advanced economies outside the EU.

For each sample, capacity measure (employment, value added) and version (absolute,
relative to peers), the model gives the share of busts that never regain their peak and the
median years to regain it among those that do. Intervals come from a bootstrap over
countries. Writes ``results/cure_*.csv`` and ``results/cure_summary.txt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cure import bootstrap, fit, fit_lognormal, km, not_recovered

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

GROUPS = ["before single market", "single market", "euro", "outside"]
DEPTH_BANDS = [0.0, 0.3, 0.5, 0.8, np.inf]
DEPTH_LABELS = ["<0.30", "0.30-0.50", "0.50-0.80", ">0.80"]
EU15 = {"AT", "BE", "DE", "DK", "EL", "ES", "FI", "FR", "IE", "IT", "LU", "NL", "PT", "SE"}


def prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.main].copy()
    d["log_depth"] = np.log(d.depth)
    d["log_depth_c"] = d.log_depth - np.log(d.depth.median())
    return d


def a64_spec(d: pd.DataFrame, controls: bool) -> tuple[pd.DataFrame, list[str], list[str]]:
    d = d.copy()
    d["euro"] = (d.stage == "euro").astype(int)
    d = d[d.stage.isin(["single market", "euro"])]
    z = ["tradable", "euro", "log_depth_c"]
    x = ["tradable", "log_depth_c"]
    if controls:
        d = d.dropna(subset=["gdp_growth_after"])
        d["gdp_after_c"] = d.gdp_growth_after - d.gdp_growth_after.median()
        z += ["gdp_after_c"]
        x += ["gdp_after_c"]
    return d, z, x


def long_spec(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    d = d.copy()
    d["group"] = d.stage.replace({"pre-accession": "before single market", "customs union": "before single market"})
    z = []
    for g in GROUPS:
        for t in (0, 1):
            c = f"cell_{g.replace(' ', '_')}_{t}"
            d[c] = ((d.group == g) & (d.tradable == t)).astype(int)
            z.append(c)
    # the cells span the intercept; drop one so the design is full rank
    ref = z.pop(0)
    z.append("log_depth_c")
    x = ["tradable", "log_depth_c"]
    d.attrs["ref"] = ref
    return d, z, x


def never_share(gamma: pd.Series | dict, cell: str | None, ref: str | None, extra: dict | None = None) -> float:
    v = gamma["const"]
    if cell is not None and cell != ref:
        v += gamma[cell]
    for k, val in (extra or {}).items():
        v += gamma[k] * val
    return float(1 / (1 + np.exp(-v)))


def summarise_a64(d: pd.DataFrame, z: list[str], x: list[str], B: int, label: str) -> dict:
    f = fit(d, "t_full", "full_recovered", z, x)
    bs = bootstrap(d, "t_full", "full_recovered", z, x, B=B)

    def share(g, trad, euro):
        v = g["const"] + g["tradable"] * trad + g["euro"] * euro
        return 1 / (1 + np.exp(-v))

    est = {}
    for euro in (0, 1):
        for trad in (0, 1):
            est[(euro, trad)] = share(f.gamma, trad, euro)
    bsh = {k: share({c.replace("gamma_", ""): bs[c] for c in bs.columns if c.startswith("gamma_")}, k[1], k[0]) for k in est}
    gap = {euro: est[(euro, 1)] - est[(euro, 0)] for euro in (0, 1)}
    bgap = {euro: bsh[(euro, 1)] - bsh[(euro, 0)] for euro in (0, 1)}
    med_non = float(np.exp(f.beta["const"]))
    med_trad = float(np.exp(f.beta["const"] + f.beta["tradable"]))
    row = {
        "sample": label,
        "n": f.n,
        "events": f.events,
        "never_nontradable_sm": est[(0, 0)],
        "never_tradable_sm": est[(0, 1)],
        "never_nontradable_euro": est[(1, 0)],
        "never_tradable_euro": est[(1, 1)],
        "gap_sm": gap[0],
        "gap_sm_lo": float(np.nanpercentile(bgap[0], 5)),
        "gap_sm_hi": float(np.nanpercentile(bgap[0], 95)),
        "gap_euro": gap[1],
        "gap_euro_lo": float(np.nanpercentile(bgap[1], 5)),
        "gap_euro_hi": float(np.nanpercentile(bgap[1], 95)),
        "gamma_tradable": float(f.gamma["tradable"]),
        "gamma_tradable_lo": float(np.nanpercentile(bs["gamma_tradable"], 5)),
        "gamma_tradable_hi": float(np.nanpercentile(bs["gamma_tradable"], 95)),
        "gamma_depth": float(f.gamma["log_depth_c"]),
        "gamma_depth_lo": float(np.nanpercentile(bs["gamma_log_depth_c"], 5)),
        "gamma_depth_hi": float(np.nanpercentile(bs["gamma_log_depth_c"], 95)),
        "median_years_nontradable": med_non,
        "median_years_tradable": med_trad,
        "beta_depth": float(f.beta["log_depth_c"]),
        "beta_depth_lo": float(np.nanpercentile(bs["beta_log_depth_c"], 5)),
        "beta_depth_hi": float(np.nanpercentile(bs["beta_log_depth_c"], 95)),
        "boot_reps": len(bs),
    }
    return row


def summarise_long(d: pd.DataFrame, z: list[str], x: list[str], B: int, label: str) -> list[dict]:
    ref = d.attrs["ref"]
    f = fit(d, "t_full", "full_recovered", z, x)
    bs = bootstrap(d, "t_full", "full_recovered", z, x, B=B)
    bg = [{c.replace("gamma_", ""): r[c] for c in bs.columns if c.startswith("gamma_")} for _, r in bs.iterrows()]
    rows = []
    for g in GROUPS:
        cells = [f"cell_{g.replace(' ', '_')}_{t}" for t in (0, 1)]
        n = [int(d[c].sum()) for c in cells]
        p = [never_share(f.gamma, c, ref) for c in cells]
        bp = np.array([[never_share(b, c, ref) for c in cells] for b in bg])
        gap = bp[:, 1] - bp[:, 0]
        rows.append(
            {
                "sample": label,
                "group": g,
                "n_nontradable": n[0],
                "n_tradable": n[1],
                "never_nontradable": p[0],
                "never_tradable": p[1],
                "gap": p[1] - p[0],
                "gap_lo": float(np.nanpercentile(gap, 5)),
                "gap_hi": float(np.nanpercentile(gap, 95)),
                "boot_reps": len(bs),
            }
        )
    # difference in differences: tradable gap in the open stages minus before the single market
    open_cells = [("single market", 1), ("euro", 1)]
    for og, _ in open_cells:
        a1, a0 = f"cell_{og.replace(' ', '_')}_1", f"cell_{og.replace(' ', '_')}_0"
        b1, b0 = "cell_before_single_market_1", "cell_before_single_market_0"
        did = (never_share(f.gamma, a1, ref) - never_share(f.gamma, a0, ref)) - (never_share(f.gamma, b1, ref) - never_share(f.gamma, b0, ref))
        bdid = np.array([(never_share(b, a1, ref) - never_share(b, a0, ref)) - (never_share(b, b1, ref) - never_share(b, b0, ref)) for b in bg])
        rows.append({"sample": label, "group": f"DiD {og} vs before", "gap": did, "gap_lo": float(np.nanpercentile(bdid, 5)), "gap_hi": float(np.nanpercentile(bdid, 95)), "boot_reps": len(bs)})
    return rows


def depth_bands(d: pd.DataFrame, B: int, label: str) -> list[dict]:
    """Never-recover share and median years to regain by depth band (open-market stages)."""
    d = d[d.stage.isin(["single market", "euro"])].copy()
    d["band"] = pd.cut(d.depth, DEPTH_BANDS, labels=DEPTH_LABELS, right=False)
    dummies = []
    for i, lab in enumerate(DEPTH_LABELS[1:], start=1):
        d[f"band{i}"] = (d.band == lab).astype(int)
        dummies.append(f"band{i}")
    d["euro"] = (d.stage == "euro").astype(int)
    z, x = dummies + ["tradable", "euro"], dummies + ["tradable"]
    f = fit(d, "t_full", "full_recovered", z, x)
    bs = bootstrap(d, "t_full", "full_recovered", z, x, B=B)
    rows = []
    for i, lab in enumerate(DEPTH_LABELS):
        add = f"band{i}" if i else None
        v = f.gamma["const"] + (f.gamma[add] if add else 0.0)
        bv = bs["gamma_const"] + (bs[f"gamma_{add}"] if add else 0.0)
        bp = 1 / (1 + np.exp(-bv))
        rec = d[(d.band == lab) & (d.full_recovered == 1)]
        rows.append(
            {
                "sample": label,
                "band": lab,
                "n": int((d.band == lab).sum()),
                "never": float(1 / (1 + np.exp(-v))),
                "never_lo": float(np.nanpercentile(bp, 5)),
                "never_hi": float(np.nanpercentile(bp, 95)),
                "median_years_recoverers": float(np.exp(f.beta["const"] + (f.beta[add] if add else 0.0))),
                "observed_regain_speed": float((rec.depth / rec.t_full).median()) if len(rec) else np.nan,
            }
        )
    return rows


def with_bands(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = d[d.stage.isin(["single market", "euro"])].copy()
    d["band"] = pd.cut(d.depth, DEPTH_BANDS, labels=DEPTH_LABELS, right=False)
    names = []
    for i, lab in enumerate(DEPTH_LABELS[1:], start=1):
        d[f"band{i}"] = (d.band == lab).astype(int)
        names.append(f"band{i}")
    d["euro"] = (d.stage == "euro").astype(int)
    return d, names


def cure_vs_lognormal(g: pd.DataFrame, label: str, horizons=(10, 20, 40)) -> list[dict]:
    """Same timing covariates with and without a never-recover fraction; fit and extrapolation by band."""
    d, bands = with_bands(g)
    c = fit(d, "t_full", "full_recovered", bands + ["tradable", "euro"], bands + ["tradable"])
    l = fit_lognormal(d, "t_full", "full_recovered", bands + ["tradable", "euro"])
    k_c = len(c.gamma) + len(c.beta) + 1
    k_l = len(l.beta) + 1
    rows = []
    for lab in DEPTH_LABELS:
        sub = d[d.band == lab]
        obs = km(sub.t_full.to_numpy(), sub.full_recovered.to_numpy(), np.array([10.0, 20.0]))
        row = {
            "sample": label,
            "band": lab,
            "n": len(sub),
            "aic_cure": 2 * k_c - 2 * c.loglik,
            "aic_lognormal": 2 * k_l - 2 * l.loglik,
            "km_10": obs[0],
            "km_20": obs[1],
            "at_risk_10": int((sub.t_full >= 10).sum()),
            "at_risk_20": int((sub.t_full >= 20).sum()),
            "max_follow_up": int(sub.follow_up.max()),
        }
        for h in horizons:
            row[f"cure_{h}"] = float(not_recovered(c, sub, h).mean())
            row[f"lognormal_{h}"] = float(not_recovered(l, sub, h).mean())
        row["lognormal_median_years"] = float(np.median(np.exp(np.column_stack([np.ones(len(sub))] + [sub[k].to_numpy(float) for k in bands + ["tradable", "euro"]]) @ l.beta.to_numpy())))
        rows.append(row)
    return rows


def long_tail(g: pd.DataFrame, label: str) -> list[dict]:
    """How much follow-up the long series carry: Kaplan-Meier and numbers still at risk, EU members."""
    g = g[g.stage != "outside"].copy()
    edges, labels = [0.0, 0.3, 0.5, np.inf], ["<0.30", "0.30-0.50", ">0.50"]
    g["band"] = pd.cut(g.depth, edges, labels=labels, right=False)
    rows = []
    for lab in labels:
        s = g[g.band == lab]
        k = km(s.t_full.to_numpy(), s.full_recovered.to_numpy(), np.array([10.0, 20.0, 30.0]))
        rows.append({"sample": label, "band": lab, "n": len(s), "km_10": k[0], "km_20": k[1], "km_30": k[2], "at_risk_10": int((s.t_full >= 10).sum()), "at_risk_20": int((s.t_full >= 20).sum()), "at_risk_30": int((s.t_full >= 30).sum())})
    return rows


def km_boot(d: pd.DataFrame, horizons, B: int, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier share still below peak at the horizons, and country-bootstrap replicates."""
    h = np.array(horizons, float)
    est = km(d.t_full.to_numpy(), d.full_recovered.to_numpy(), h)
    rng = np.random.default_rng(seed)
    geos = d.geo.unique()
    idx = {g: np.flatnonzero(d.geo.to_numpy() == g) for g in geos}
    reps = []
    for _ in range(B):
        rows = np.concatenate([idx[g] for g in rng.choice(geos, len(geos), replace=True)])
        b = d.iloc[rows]
        reps.append(km(b.t_full.to_numpy(), b.full_recovered.to_numpy(), h))
    return est, np.array(reps)


def km_depth(g: pd.DataFrame, B: int, label: str) -> list[dict]:
    """Model-free: share still below the peak 5 and 10 years after the trough, by depth band."""
    d, _ = with_bands(g)
    rows = []
    for lab in DEPTH_LABELS:
        sub = d[d.band == lab]
        est, reps = km_boot(sub, (5, 10), B)
        rows.append(
            {
                "sample": label,
                "band": lab,
                "n": len(sub),
                "below_5": est[0],
                "below_5_lo": float(np.percentile(reps[:, 0], 5)),
                "below_5_hi": float(np.percentile(reps[:, 0], 95)),
                "below_10": est[1],
                "below_10_lo": float(np.percentile(reps[:, 1], 5)),
                "below_10_hi": float(np.percentile(reps[:, 1], 95)),
                "at_risk_10": int((sub.t_full >= 10).sum()),
            }
        )
    return rows


def km_tradable_gap(g: pd.DataFrame, B: int, label: str) -> list[dict]:
    """Model-free tradable gap: share still below peak at 10 years, tradable minus non-tradable."""
    rows = []
    for st in ("single market", "euro"):
        d = g[g.stage == st]
        a, ra = km_boot(d[d.tradable == 1], (5, 10), B, seed=7)
        b, rb = km_boot(d[d.tradable == 0], (5, 10), B, seed=7)
        # resample tradable and non-tradable with the same country draws
        rng = np.random.default_rng(9)
        geos = d.geo.unique()
        reps = []
        for _ in range(B):
            pick = rng.choice(geos, len(geos), replace=True)
            bt = pd.concat([d[(d.geo == x)] for x in pick])
            t1 = km(bt[bt.tradable == 1].t_full.to_numpy(), bt[bt.tradable == 1].full_recovered.to_numpy(), np.array([5.0, 10.0]))
            t0 = km(bt[bt.tradable == 0].t_full.to_numpy(), bt[bt.tradable == 0].full_recovered.to_numpy(), np.array([5.0, 10.0]))
            reps.append(t1 - t0)
        reps = np.array(reps)
        rows.append(
            {
                "sample": label,
                "stage": st,
                "n_tradable": int((d.tradable == 1).sum()),
                "n_nontradable": int((d.tradable == 0).sum()),
                "tradable_below_10": a[1],
                "nontradable_below_10": b[1],
                "gap_5": a[0] - b[0],
                "gap_5_lo": float(np.percentile(reps[:, 0], 5)),
                "gap_5_hi": float(np.percentile(reps[:, 0], 95)),
                "gap_10": a[1] - b[1],
                "gap_10_lo": float(np.percentile(reps[:, 1], 5)),
                "gap_10_hi": float(np.percentile(reps[:, 1], 95)),
            }
        )
    return rows


def robustness(g: pd.DataFrame, B: int, label: str) -> list[dict]:
    checks = {
        "trough by 2015 (10+ years of follow-up)": g[g.trough_year <= 2015],
        "tradable services counted as tradable": g.assign(tradable=g.tradable_wide),
        "pandemic busts dropped": g[~g.covid.astype(bool)],
        "EU-15 only": g[g.geo.isin(EU15)],
    }
    rows = []
    for name, sub in checks.items():
        d, z, x = a64_spec(sub, False)
        rows.append(summarise_a64(d, z, x, B, f"{label}: {name}"))
    return rows


def km_table(d: pd.DataFrame, by: list[str], horizons=(5, 10, 15, 20)) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(by):
        s = km(g.t_full.to_numpy(), g.full_recovered.to_numpy(), np.array(horizons, float))
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,))) | {"n": len(g)} | {f"not_recovered_{h}y": v for h, v in zip(horizons, s)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)

    a64 = prep(pd.read_csv(RES / "episodes_all.csv"))
    longe = prep(pd.read_csv(RES / "long_episodes.csv"))

    a_rows, l_rows, km_rows, band_rows, rob_rows, cmp_rows, kmd_rows, kmg_rows = [], [], [], [], [], [], [], []
    for (m, v), g in a64.groupby(["measure", "version"]):
        for controls in (False, True):
            d, z, x = a64_spec(g, controls)
            a_rows.append(summarise_a64(d, z, x, args.boot, f"A*64 {m} {v}" + (" + GDP after trough" if controls else "")))
        band_rows += depth_bands(g, args.boot, f"A*64 {m} {v}")
        cmp_rows += cure_vs_lognormal(g, f"A*64 {m} {v}")
        kmd_rows += km_depth(g, args.boot, f"A*64 {m} {v}")
        kmg_rows += km_tradable_gap(g, args.boot, f"A*64 {m} {v}")
        rob_rows += robustness(g, args.boot, f"A*64 {m} {v}")
        kt = km_table(g[g.stage.isin(["single market", "euro"])], ["stage", "tradable"])
        kt.insert(0, "sample", f"A*64 {m} {v}")
        km_rows.append(kt)
    tail_rows = []
    for (m, v), g in longe.groupby(["measure", "version"]):
        tail_rows += long_tail(g, f"AMECO {m} {v}")
        d, z, x = long_spec(g)
        l_rows += summarise_long(d, z, x, args.boot, f"AMECO {m} {v}")
        kt = km_table(d, ["group", "tradable"])
        kt.insert(0, "sample", f"AMECO {m} {v}")
        km_rows.append(kt)

    A = pd.DataFrame(a_rows)
    R = pd.DataFrame(rob_rows)
    D = pd.DataFrame(band_rows)
    Lg = pd.DataFrame(l_rows)
    R.to_csv(RES / "cure_a64_robustness.csv", index=False)
    D.to_csv(RES / "cure_depth_bands.csv", index=False)
    C = pd.DataFrame(cmp_rows)
    C.to_csv(RES / "cure_vs_lognormal.csv", index=False)
    KD = pd.DataFrame(kmd_rows)
    KG = pd.DataFrame(kmg_rows)
    KD.to_csv(RES / "km_depth_bands.csv", index=False)
    KG.to_csv(RES / "km_tradable_gap.csv", index=False)
    T = pd.DataFrame(tail_rows)
    T.to_csv(RES / "long_follow_up.csv", index=False)
    K = pd.concat(km_rows, ignore_index=True)
    A.to_csv(RES / "cure_a64.csv", index=False)
    Lg.to_csv(RES / "cure_long.csv", index=False)
    K.to_csv(RES / "km_not_recovered.csv", index=False)

    L = []
    say = L.append
    say("0. Model-free (Kaplan-Meier): share still below the peak 5 and 10 years after the trough, single market and euro stages, 90% country-bootstrap intervals")
    for s_, g in KD.groupby("sample", sort=False):
        say(f"  {s_}: " + "; ".join(f"{r.band} n={r.n} 5 y {r.below_5:.2f} [{r.below_5_lo:.2f}, {r.below_5_hi:.2f}], 10 y {r.below_10:.2f} [{r.below_10_lo:.2f}, {r.below_10_hi:.2f}] (at risk {r.at_risk_10})" for _, r in g.iterrows()))
    say("  Tradable minus non-tradable, share still below peak")
    for _, r in KG.iterrows():
        say(f"    {r['sample']} {r.stage}: n={r.n_tradable}/{r.n_nontradable}; 10 y tradable {r.tradable_below_10:.2f}, non-tradable {r.nontradable_below_10:.2f}; gap 5 y {r.gap_5:+.2f} [{r.gap_5_lo:+.2f}, {r.gap_5_hi:+.2f}], 10 y {r.gap_10:+.2f} [{r.gap_10_lo:+.2f}, {r.gap_10_hi:+.2f}]")
    say("")
    say("Share of busts that never regain their peak (mixture cure model), at the median depth.")
    say("Gap = tradable minus non-tradable; 90% intervals from a bootstrap over countries.\n")
    say("1. Within the open market: A*64 industries, EU-27, troughs 1995-2024")
    for _, r in A.iterrows():
        say(
            f"  {r['sample']}: n={r.n}, regained={r.events}; never (single market) non-tradable {r.never_nontradable_sm:.2f}, tradable {r.never_tradable_sm:.2f}, "
            f"gap {r.gap_sm:+.2f} [{r.gap_sm_lo:+.2f}, {r.gap_sm_hi:+.2f}]; never (euro) non-tradable {r.never_nontradable_euro:.2f}, tradable {r.never_tradable_euro:.2f}, "
            f"gap {r.gap_euro:+.2f} [{r.gap_euro_lo:+.2f}, {r.gap_euro_hi:+.2f}]; depth effect on never-recover (logit) {r.gamma_depth:+.2f} [{r.gamma_depth_lo:+.2f}, {r.gamma_depth_hi:+.2f}]; "
            f"median years to regain (recoverers) non-tradable {r.median_years_nontradable:.1f}, tradable {r.median_years_tradable:.1f}; depth effect on log time {r.beta_depth:+.2f} [{r.beta_depth_lo:+.2f}, {r.beta_depth_hi:+.2f}]"
        )
    say("\n   Robustness (same model, no demand control)")
    for _, r in R.iterrows():
        say(f"    {r['sample']}: n={r.n}; gap single market {r.gap_sm:+.2f} [{r.gap_sm_lo:+.2f}, {r.gap_sm_hi:+.2f}], gap euro {r.gap_euro:+.2f} [{r.gap_euro_lo:+.2f}, {r.gap_euro_hi:+.2f}]; depth (logit) {r.gamma_depth:+.2f} [{r.gamma_depth_lo:+.2f}, {r.gamma_depth_hi:+.2f}]")
    say("\n   By depth band (log points), single market and euro stages")
    for s_, g in D.groupby("sample", sort=False):
        say(f"    {s_}: " + "; ".join(f"{r.band} n={r.n} never {r.never:.2f} [{r.never_lo:.2f}, {r.never_hi:.2f}], median {r.median_years_recoverers:.1f} y, observed speed {r.observed_regain_speed:.3f}/y" for _, r in g.iterrows()))
    say("\n   Cure model against a lognormal model with no never-recover fraction (same covariates, by depth band)")
    for s_, g in C.groupby("sample", sort=False):
        r0 = g.iloc[0]
        say(f"    {s_}: AIC cure {r0.aic_cure:.1f}, lognormal {r0.aic_lognormal:.1f}")
        for _, r in g.iterrows():
            say(f"      {r.band}: n={r.n}; still below peak, observed (Kaplan-Meier) 10 y {r.km_10:.2f} (at risk {r.at_risk_10}), 20 y {r.km_20:.2f} (at risk {r.at_risk_20}); cure model 10/20/40 y {r.cure_10:.2f}/{r.cure_20:.2f}/{r.cure_40:.2f}; lognormal 10/20/40 y {r.lognormal_10:.2f}/{r.lognormal_20:.2f}/{r.lognormal_40:.2f}, median {r.lognormal_median_years:.1f} y")
    say("\n2. Across stages: AMECO sectors, 1960-2025")
    for s, g in Lg.groupby("sample", sort=False):
        say(f"  {s}")
        for _, r in g.iterrows():
            if r.group.startswith("DiD"):
                say(f"    {r.group}: {r.gap:+.2f} [{r.gap_lo:+.2f}, {r.gap_hi:+.2f}]")
            else:
                say(f"    {r.group:22s} n={int(r.n_nontradable)}/{int(r.n_tradable)}  never non-tradable {r.never_nontradable:.2f}, tradable {r.never_tradable:.2f}, gap {r.gap:+.2f} [{r.gap_lo:+.2f}, {r.gap_hi:+.2f}]")
    say("\n   Follow-up in the long series (EU members): share still below peak and episodes still observed")
    for _, r in T.iterrows():
        say(f"    {r['sample']} {r.band}: n={r.n}; 10 y {r.km_10:.2f} (at risk {r.at_risk_10}), 20 y {r.km_20:.2f} (at risk {r.at_risk_20}), 30 y {r.km_30:.2f} (at risk {r.at_risk_30})")
    text = "\n".join(L) + "\n"
    (RES / "cure_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
