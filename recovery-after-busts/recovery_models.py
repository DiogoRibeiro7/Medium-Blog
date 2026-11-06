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

from cure import bootstrap, fit, km

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

    a_rows, l_rows, km_rows, band_rows, rob_rows = [], [], [], [], []
    for (m, v), g in a64.groupby(["measure", "version"]):
        for controls in (False, True):
            d, z, x = a64_spec(g, controls)
            a_rows.append(summarise_a64(d, z, x, args.boot, f"A*64 {m} {v}" + (" + GDP after trough" if controls else "")))
        band_rows += depth_bands(g, args.boot, f"A*64 {m} {v}")
        rob_rows += robustness(g, args.boot, f"A*64 {m} {v}")
        kt = km_table(g[g.stage.isin(["single market", "euro"])], ["stage", "tradable"])
        kt.insert(0, "sample", f"A*64 {m} {v}")
        km_rows.append(kt)
    for (m, v), g in longe.groupby(["measure", "version"]):
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
    K = pd.concat(km_rows, ignore_index=True)
    A.to_csv(RES / "cure_a64.csv", index=False)
    Lg.to_csv(RES / "cure_long.csv", index=False)
    K.to_csv(RES / "km_not_recovered.csv", index=False)

    L = []
    say = L.append
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
    say("\n2. Across stages: AMECO sectors, 1960-2025")
    for s, g in Lg.groupby("sample", sort=False):
        say(f"  {s}")
        for _, r in g.iterrows():
            if r.group.startswith("DiD"):
                say(f"    {r.group}: {r.gap:+.2f} [{r.gap_lo:+.2f}, {r.gap_hi:+.2f}]")
            else:
                say(f"    {r.group:22s} n={int(r.n_nontradable)}/{int(r.n_tradable)}  never non-tradable {r.never_nontradable:.2f}, tradable {r.never_tradable:.2f}, gap {r.gap:+.2f} [{r.gap_lo:+.2f}, {r.gap_hi:+.2f}]")
    text = "\n".join(L) + "\n"
    (RES / "cure_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
