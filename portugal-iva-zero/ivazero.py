"""Pass-through of Portugal's zero VAT on 46 food products (IVA Zero) in official prices.

From 18 April 2023 to 4 January 2024 Portugal applied a 0% VAT rate to a basket
of 46 food products, previously taxed at 6% (vegetable oils at 23%). The measure
was announced on 24 March 2023. Its end was announced on 10 October 2023 with the
2024 budget proposal (first for the end of December, later moved to 4 January);
VAT returned on 5 January 2024.

The analysis uses Eurostat's monthly HICP by ECOICOP subclass (5 digits), the
same indices at constant tax rates (HICP-CT), and the item weights, for all EU
member states.

1. Treatment from the data. INE's constant-tax index assumes that every tax
   change is passed on in full, so the tax wedge
       w(c,s,t) = log HICP(c,s,t) - log HICP-CT(c,s,t)
   moves only when a tax rate changes. Its change between March and May 2023
   measures, subclass by subclass, how much of each Portuguese subclass the VAT
   cut covered. The same wedge screens the other countries: a country is dropped
   from the comparison set if the wedge of any treated or untreated subclass
   moved by half a log point or more from 2022 on; Hungary is also dropped
   (administrative price caps on basic foods until August 2023).
2. Within Portugal: an event study of treated against untreated food subclasses,
   each series in logs relative to February 2023 (the last month before the
   announcement). With subclass and month fixed effects on a balanced panel, the
   event coefficient is the difference between the two groups' mean log change.
3. Across countries: the same difference after subtracting, subclass by
   subclass, the mean log change in the comparison countries. This removes
   price shocks that hit a product everywhere in Europe (olive oil, sugar).
4. Inference: standard errors clustered by subclass; placebo jumps (every
   two-month and one-month change in the gap in 2019-2022); placebo dates; placebo
   countries (each comparison country treated as if it were Portugal, with
   Portugal's treated set).
5. Synthetic control on the relative price of the treated basket.
6. Pass-through = estimated price change / change in the tax wedge.
7. The reversal in January 2024 and whether prices returned to the comparison path.
8. Effect on headline inflation and incidence by income fifth.

Run from this folder::

    python ivazero.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

ANNOUNCED = "2023-03"        # 24 March 2023
STARTED = "2023-04"          # 18 April 2023: April is partly treated
FIRST_FULL = "2023-05"
END_ANNOUNCED = "2023-10"    # 10 October 2023, with the 2024 budget proposal
REVERSED = "2024-01"         # VAT back on 5 January 2024
BASE = "2023-02"             # last month before the announcement
EXIT_BASE = "2023-09"        # last month before the end was announced
PRE_START = "2018-01"
POLICY = [f"2023-{m:02d}" for m in range(5, 13)]           # May-December 2023, fully treated months
VAT_6 = np.log(1.06)
VAT_23 = np.log(1.23)
SCREEN_FROM = "2022-01"
SCREEN_TOL = 0.005            # a wedge move of half a percent or more counts as a tax change


# ---------------------------------------------------------------- loading ---

def load(name: str) -> pd.DataFrame:
    """Long Eurostat file -> wide frame, index (geo, coicop), columns = periods (YYYY-MM)."""
    d = pd.read_csv(DATA / f"{name}.csv")
    d = d[d["value"].notna()]
    return d.pivot_table(index=["geo", "coicop"], columns="period", values="value").sort_index(axis=1)


def subclasses(codes) -> list[str]:
    """ECOICOP 5-digit food subclasses (CP011.. and CP012..)."""
    return sorted(c for c in codes if c.startswith("CP01") and len(c) == 7)


def log_prices() -> pd.DataFrame:
    p = load("prc_hicp_midx")
    return np.log(p.loc[:, PRE_START:])


def tax_wedge() -> pd.DataFrame:
    p = load("prc_hicp_midx")
    ct = load("prc_hicp_cind")
    idx = p.index.intersection(ct.index)
    cols = p.columns.intersection(ct.columns)
    return np.log(p.loc[idx, cols]) - np.log(ct.loc[idx, cols])


def weights() -> pd.DataFrame:
    """HICP item weights, per mille of the national HICP, index (geo, coicop), columns = years."""
    w = pd.read_csv(DATA / "prc_hicp_inw.csv")
    w["period"] = w["period"].astype(str).str[:4].astype(int)
    return w.pivot_table(index=["geo", "coicop"], columns="period", values="value")


# --------------------------------------------------------------- treatment ---

def classify(wedge: pd.DataFrame) -> pd.DataFrame:
    """Treatment of each Portuguese food subclass from the change in its tax wedge."""
    pt = wedge.xs("PT", level="geo")
    rows = []
    for s in subclasses(pt.index):
        w = pt.loc[s]
        if w[[ANNOUNCED, FIRST_FULL, EXIT_BASE, REVERSED]].isna().any():
            continue
        entry = w[FIRST_FULL] - w[ANNOUNCED]
        exit_ = w[REVERSED] - w[EXIT_BASE]
        other = w.loc[SCREEN_FROM:].diff().drop(index=[STARTED, FIRST_FULL, REVERSED], errors="ignore").abs().max()
        rows.append({"coicop": s, "entry": entry, "exit": exit_, "other_moves": other})
    t = pd.DataFrame(rows).set_index("coicop")
    t["intensity"] = -t["entry"] / VAT_6
    t["group"] = np.select(
        [t["entry"] < -0.10, t["intensity"] >= 0.85, t["intensity"] >= 0.05, t["entry"].abs() < 1e-3],
        ["oils_23", "treated", "partial", "control"], default="unclear")
    moved = (t["other_moves"] > SCREEN_TOL) | (t["exit"].abs() >= 1e-3)
    t.loc[(t["group"] == "control") & moved, "group"] = "control_other_tax"
    return t


# Countries dropped for reasons the tax wedge cannot show.
PRICE_CONTROLS = {"HU": "administrative price caps on basic foods until 1 August 2023"}


def donors(wedge: pd.DataFrame, codes: list[str], start: str = SCREEN_FROM,
           end: str | None = None) -> tuple[list[str], pd.DataFrame]:
    """Countries whose tax wedges did not move between ``start`` and ``end`` in any of ``codes``."""
    rows = []
    geos = sorted(set(load("prc_hicp_midx").index.get_level_values("geo")))
    for geo in geos:
        if geo == "PT":
            continue
        have = []
        if geo in wedge.index.get_level_values("geo"):
            w = wedge.xs(geo, level="geo")
            have = [s for s in codes if s in w.index and w.loc[s].notna().any()]
        if not have:
            rows.append({"geo": geo, "subclasses": 0, "moved": 0, "largest_move": np.nan,
                         "other_reason": "no food subclasses at constant taxes"})
            continue
        mv = w.loc[have, start:end].diff(axis=1).abs().max(axis=1)
        rows.append({"geo": geo, "subclasses": len(have), "moved": int((mv >= SCREEN_TOL).sum()),
                     "largest_move": float(mv.max()), "other_reason": PRICE_CONTROLS.get(geo, "")})
    screen = pd.DataFrame(rows).set_index("geo")
    keep = screen[(screen["moved"] == 0) & (screen["other_reason"] == "")]
    return list(keep.index), screen


def moved_pairs(wedge: pd.DataFrame, start: str = SCREEN_FROM) -> pd.Series:
    """(geo, coicop) pairs whose tax wedge moved by SCREEN_TOL or more from ``start`` on."""
    mv = wedge.loc[:, start:].diff(axis=1).abs().max(axis=1)
    return mv >= SCREEN_TOL


# ------------------------------------------------------------ estimators ---

def rebased(lp: pd.DataFrame, base: str) -> pd.DataFrame:
    return lp.sub(lp[base], axis=0)


def group_gap(r: pd.DataFrame, treated: list[str], control: list[str],
              w: pd.Series | None = None) -> pd.DataFrame:
    """Difference between treated and control mean log change, with a cluster (subclass) standard error.

    ``r`` is indexed by subclass with periods as columns. ``w`` optionally weights the subclasses.
    """
    out = {}
    for col in r.columns:
        a, b = r.loc[treated, col].dropna(), r.loc[control, col].dropna()
        if w is None:
            ma, mb = a.mean(), b.mean()
            va = a.var(ddof=1) / len(a) if len(a) > 1 else np.nan
            vb = b.var(ddof=1) / len(b) if len(b) > 1 else np.nan
        else:
            wa, wb = w.reindex(a.index).fillna(0), w.reindex(b.index).fillna(0)
            wa, wb = wa / wa.sum(), wb / wb.sum()
            ma, mb = float((wa * a).sum()), float((wb * b).sum())
            va = float((wa ** 2 * (a - ma) ** 2).sum()) * len(a) / max(len(a) - 1, 1)
            vb = float((wb ** 2 * (b - mb) ** 2).sum()) * len(b) / max(len(b) - 1, 1)
        out[col] = {"gap": ma - mb, "se": np.sqrt(va + vb), "treated": ma, "control": mb}
    return pd.DataFrame(out).T


def within_country(lp: pd.DataFrame, geo: str, treated: list[str], control: list[str],
                   base: str, w: pd.Series | None = None) -> pd.DataFrame:
    r = rebased(lp.xs(geo, level="geo"), base)
    return group_gap(r, treated, control, w)


def across_countries(lp: pd.DataFrame, geo: str, comparison: list[str], treated: list[str],
                     control: list[str], base: str, w: pd.Series | None = None) -> pd.DataFrame:
    """Triple difference: each subclass's log change minus its mean log change in the comparison countries."""
    r = rebased(lp, base)
    own = r.xs(geo, level="geo")
    comp = r.loc[r.index.get_level_values("geo").isin(comparison)].groupby(level="coicop").mean()
    keep = [s for s in own.index if s in comp.index]
    diff = own.loc[keep] - comp.loc[keep]
    return group_gap(diff, [s for s in treated if s in keep], [s for s in control if s in keep], w)


def within_diff(lp: pd.DataFrame, geo: str, base) -> pd.DataFrame:
    own = lp.xs(geo, level="geo")
    ref = own[base].mean(axis=1) if isinstance(base, list) else own[base]
    return own.sub(ref, axis=0)


def across_diff(lp: pd.DataFrame, geo: str, comparison: list[str], base,
                drop: pd.Series | None = None) -> pd.DataFrame:
    """Each subclass's log change since ``base`` minus its mean change in the comparison countries.

    ``base`` is a month or a list of months (their mean is the base). ``drop`` marks (geo, coicop)
    pairs to leave out of the comparison means (subclass-specific tax changes).
    """
    ref = lp[base].mean(axis=1) if isinstance(base, list) else lp[base]
    r = lp.sub(ref, axis=0)
    own = r.xs(geo, level="geo")
    c = r.loc[r.index.get_level_values("geo").isin(comparison)]
    if drop is not None:
        c = c.loc[~drop.reindex(c.index).fillna(False).astype(bool)]
    comp = c.groupby(level="coicop").mean()
    keep = [s for s in own.index if s in comp.index]
    return own.loc[keep] - comp.loc[keep]


def window_gap(diff: pd.DataFrame, treated: list[str], control: list[str], months: list[str],
               w: pd.Series | None = None) -> tuple[float, float]:
    """Treated-minus-control gap averaged over ``months``, with a standard error clustered by subclass."""
    avg = diff[[m for m in months if m in diff.columns]].mean(axis=1).to_frame("window")
    g = group_gap(avg, [s for s in treated if s in avg.index], [s for s in control if s in avg.index], w)
    return float(g.loc["window", "gap"]), float(g.loc["window", "se"])


def pass_through(gap: float, se: float, tax: float) -> str:
    lo, hi = sorted([100 * (gap + 1.96 * se) / tax, 100 * (gap - 1.96 * se) / tax])
    return f"{100 * gap / tax:.0f}% (95% CI {lo:.0f} to {hi:.0f})"


def window_mean(est: pd.DataFrame, months: list[str]) -> float:
    return float(est.loc[[m for m in months if m in est.index], "gap"].mean())


def placebo_dates(lp: pd.DataFrame, treated: list[str], control: list[str],
                  comparison: list[str] | None = None) -> pd.Series:
    """The estimator centred on fake announcement months, 2019-01 to 2022-06 (within, or across if ``comparison``)."""
    cols = list(lp.columns)
    out = {}
    for i, m in enumerate(cols):
        if not ("2019-01" <= m <= "2022-06"):
            continue
        base = cols[i - 1]
        window = cols[i + 2:i + 10]           # same layout as the real event: base, announcement, partial month, 8 months
        sub = lp[[base] + window]
        diff = within_diff(sub, "PT", base) if comparison is None else across_diff(sub, "PT", comparison, base)
        out[m] = window_gap(diff, treated, control, window)[0]
    return pd.Series(out)


def placebo_countries(lp: pd.DataFrame, comparison: list[str], treated: list[str],
                      control: list[str]) -> pd.Series:
    out = {}
    for geo in comparison:
        others = [c for c in comparison if c != geo]
        est = across_countries(lp, geo, others, treated, control, BASE)
        out[geo] = window_mean(est, POLICY)
    return pd.Series(out).sort_values()


# -------------------------------------------------------- synthetic control ---

def basket(lp: pd.DataFrame, geo: str, codes: list[str], w: pd.Series) -> pd.Series:
    """Log of a fixed-weight geometric basket of subclasses (weighted mean of log indices), rebased to BASE."""
    r = rebased(lp.xs(geo, level="geo"), BASE)
    have = [s for s in codes if s in r.index and r.loc[s].notna().all()]
    ww = w.reindex(have).fillna(0)
    ww = ww / ww.sum()
    return (r.loc[have].mul(ww, axis=0)).sum()


def synthetic_control(y: pd.DataFrame, treated_unit: str, pre: list[str]) -> tuple[pd.Series, pd.Series]:
    """Convex weights on the comparison units that best reproduce the treated unit before the event."""
    donors_ = [c for c in y.index if c != treated_unit]
    X = y.loc[donors_, pre].to_numpy().T
    target = y.loc[treated_unit, pre].to_numpy()
    n = len(donors_)

    def loss(v):
        return float(((X @ v - target) ** 2).sum())

    res = minimize(loss, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1}],
                   options={"maxiter": 1000, "ftol": 1e-14})
    v = pd.Series(np.clip(res.x, 0, None), index=donors_)
    v = v / v.sum()
    synth = (y.loc[donors_].mul(v, axis=0)).sum()
    return v, y.loc[treated_unit] - synth


# ------------------------------------------------------------- extensions ---

LABELS = {
    "CP01111": "Rice", "CP01112": "Flours and other cereals", "CP01113": "Bread", "CP01114": "Other bakery products",
    "CP01115": "Pizza and quiche", "CP01116": "Pasta and couscous", "CP01117": "Breakfast cereals",
    "CP01121": "Beef and veal", "CP01122": "Pork", "CP01123": "Lamb and goat", "CP01124": "Poultry",
    "CP01125": "Other meats", "CP01127": "Dried, salted or smoked meat", "CP01128": "Other meat preparations",
    "CP01131": "Fresh or chilled fish", "CP01132": "Frozen fish", "CP01133": "Fresh or chilled seafood",
    "CP01134": "Frozen seafood", "CP01135": "Dried, smoked or salted fish", "CP01136": "Other preserved fish",
    "CP01141": "Whole milk", "CP01142": "Low-fat milk", "CP01143": "Preserved milk", "CP01144": "Yoghurt",
    "CP01145": "Cheese and curd", "CP01146": "Other milk products", "CP01147": "Eggs", "CP01151": "Butter",
    "CP01152": "Margarine", "CP01153": "Olive oil", "CP01154": "Other edible oils", "CP01161": "Fresh fruit",
    "CP01163": "Dried fruit and nuts", "CP01164": "Preserved fruit", "CP01171": "Fresh vegetables",
    "CP01172": "Frozen vegetables", "CP01173": "Dried and preserved vegetables", "CP01174": "Potatoes",
    "CP01175": "Crisps", "CP01176": "Other tuber products", "CP01181": "Sugar", "CP01182": "Jams and honey",
    "CP01183": "Chocolate", "CP01184": "Confectionery", "CP01185": "Edible ices", "CP01191": "Sauces and condiments",
    "CP01192": "Salt and spices", "CP01193": "Baby food", "CP01194": "Ready-made food", "CP01199": "Other food",
    "CP01211": "Coffee", "CP01212": "Tea", "CP01213": "Cocoa", "CP01221": "Mineral water",
    "CP01222": "Soft drinks", "CP01223": "Juices",
}


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope, intercept and heteroskedasticity-robust (HC1) standard error of the slope."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = (X * e[:, None]).T @ (X * e[:, None])
    cov = xtx_inv @ meat @ xtx_inv * n / (n - k)
    return float(beta[1]), float(beta[0]), float(np.sqrt(cov[1, 1]))


def subclass_effects(lp: pd.DataFrame, comp: list[str], codes: list[str], base: str,
                     drop: pd.Series | None = None) -> pd.DataFrame:
    """Each Portuguese subclass's log change since ``base`` minus its mean in the comparison countries."""
    d = across_diff(lp, "PT", comp, base, drop)
    return d.loc[[c for c in codes if c in d.index]]


def extensions(w, lp: pd.DataFrame, wedge: pd.DataFrame, t: pd.DataFrame, comp: list[str],
               w23: pd.Series) -> None:
    groups = {g: list(t.index[t["group"] == g]) for g in t["group"].unique()}
    treated, control, partial = groups["treated"], groups["control"], groups.get("partial", [])
    wpt = wedge.xs("PT", level="geo")
    drop = moved_pairs(wedge)

    # Short windows around each change: least exposed to slow trends.
    w("Short windows (treated minus untreated, log points x100; pass-through with 95% CI clustered by subclass)")
    windows = [("entry", ANNOUNCED, FIRST_FULL), ("entry", ANNOUNCED, "2023-06"), ("exit", "2023-12", REVERSED),
               ("exit", "2023-12", "2024-03"), ("exit", EXIT_BASE, REVERSED), ("exit", EXIT_BASE, "2024-03"),
               ("exit", "2023-10", REVERSED), ("exit", "2023-10", "2024-03")]
    rows = []
    for kind, a, b in windows:
        tax = group_gap(rebased(wpt, a)[[b]], treated, control)["gap"].iloc[0]
        wi, wi_se = window_gap(within_diff(lp[[a, b]], "PT", a), treated, control, [b])
        ac, ac_se = window_gap(across_diff(lp[[a, b]], "PT", comp, a), treated, control, [b])
        rows.append({"kind": kind, "from": a, "to": b, "tax": tax, "within": wi, "within_se": wi_se,
                     "across": ac, "across_se": ac_se})
        aw, aw_se = window_gap(across_diff(lp[[a, b]], "PT", comp, a), treated, control, [b], w23)
        rows[-1].update({"across_weighted": aw, "across_weighted_se": aw_se})
        w(f"  {kind} {a} to {b}: within {fmt(wi)}, {pass_through(wi, wi_se, tax)}; across {fmt(ac)}, "
          f"{pass_through(ac, ac_se, tax)}; across HICP-weighted {fmt(aw)}, {pass_through(aw, aw_se, tax)}; "
          f"tax wedge {fmt(tax)}")
    pd.DataFrame(rows).round(6).to_csv(RESULTS / "short_windows.csv", index=False)

    # Dose-response: every Portuguese subclass, effect against the change in its tax wedge.
    codes = treated + partial + control
    eff = subclass_effects(lp, comp, codes, BASE, drop)
    jump = subclass_effects(lp[[ANNOUNCED, FIRST_FULL]], comp, codes, ANNOUNCED, drop)[FIRST_FULL]
    dose = pd.DataFrame({"wedge_entry": t.loc[eff.index, "entry"], "group": t.loc[eff.index, "group"],
                         "effect_may_dec_2023": eff[POLICY].mean(axis=1), "effect_march_to_may": jump,
                         "effect_2024": eff[[c for c in eff.columns if c.startswith("2024")]].mean(axis=1),
                         "weight_2023": w23.reindex(eff.index) / 10,
                         "label": [LABELS.get(c, c) for c in eff.index]})
    dose.round(6).to_csv(RESULTS / "dose_response.csv")
    w(f"Dose-response across {len(dose)} Portuguese subclasses (treated, partial, control), effect vs tax-wedge change at "
      "entry; comparison means leave out country-subclass pairs with their own tax changes")
    for col in ("effect_march_to_may", "effect_may_dec_2023"):
        b, a, se = ols(dose["wedge_entry"].to_numpy(), dose[col].to_numpy())
        w(f"  {col}: slope (pass-through) {100 * b:.0f}% (95% CI {100 * (b - 1.96 * se):.0f} to "
          f"{100 * (b + 1.96 * se):.0f}), intercept {fmt(a)}")
    b, a, se = ols(dose["wedge_entry"].to_numpy(), dose["effect_2024"].to_numpy())
    w(f"  effect_2024 (after the reversal): slope {100 * b:.0f}% (95% CI {100 * (b - 1.96 * se):.0f} to "
      f"{100 * (b + 1.96 * se):.0f})")
    part = dose[dose["group"] == "partial"]
    b, a, se = ols(part["wedge_entry"].to_numpy(), part["effect_march_to_may"].to_numpy())
    w(f"  partial subclasses only, March to May: slope {100 * b:.0f}% (se {100 * se:.0f}); n = {len(part)}")
    tr = dose[dose["group"] == "treated"].sort_values("effect_may_dec_2023")
    w("  treated subclasses, May-December 2023 / March-May / 2024 mean: " + "; ".join(
        f"{r['label']} {fmt(r['effect_may_dec_2023'])} / {fmt(r['effect_march_to_may'])} / {fmt(r['effect_2024'])}"
        for _, r in tr.iterrows()))

    # Robustness of the across-country May-December estimate.
    w("Robustness, across countries, May-December 2023 mean, pass-through (95% CI)")
    tax = group_gap(rebased(wpt, BASE)[POLICY], treated, control)["gap"].mean()
    beverages = [c for c in control if c.startswith("CP012")]
    window_comp, _ = donors(wedge, treated + control, "2022-12", "2024-03")
    variants = {
        "baseline": (treated, control, comp, BASE, None),
        "HICP-weighted": (treated, control, comp, BASE, w23),
        "base December 2022": (treated, control, comp, "2022-12", None),
        "base January 2023": (treated, control, comp, "2023-01", None),
        "base March 2023": (treated, control, comp, ANNOUNCED, None),
        "base: mean of December 2022 - February 2023": (treated, control, comp, ["2022-12", "2023-01", BASE], None),
        "food-only controls (no beverages)": (treated, [c for c in control if c not in beverages], comp, BASE, None),
        "controls without preserved milk and dried vegetables":
            (treated, [c for c in control if c not in ("CP01143", "CP01173")], comp, BASE, None),
        "treated without olive oil": ([c for c in treated if c != "CP01153"], control, comp, BASE, None),
        "comparison without EL and IT (voluntary price-restraint schemes)":
            (treated, control, [c for c in comp if c not in ("EL", "IT")], BASE, None),
        f"comparison screened only December 2022 - March 2024 ({len(window_comp)} countries)":
            (treated, control, window_comp, BASE, None),
    }
    rob = []
    for name, (tt, cc, cm, base, ww) in variants.items():
        g, se = window_gap(across_diff(lp, "PT", cm, base), tt, cc, POLICY, ww)
        rob.append({"variant": name, "gap": g, "se": se, "pass_through": g / tax})
        w(f"  {name}: {fmt(g)}, {pass_through(g, se, tax)}")
    pd.DataFrame(rob).round(6).to_csv(RESULTS / "robustness.csv", index=False)
    w(f"  (screened only December 2022 - March 2024: {', '.join(window_comp)})")
    # Bracketing: counterfactual = mean of the plateau before (Dec 2022 - Apr 2023) and the level after (Feb - Jun 2024).
    diff = across_diff(lp, "PT", comp, BASE)
    gap = group_gap(diff, treated, control)["gap"]
    before = gap["2022-12":"2023-04"].mean()
    after = gap["2024-02":"2024-06"].mean()
    br = gap[POLICY].mean() - 0.5 * (before + after)
    w(f"  bracketing (counterfactual = mean of December 2022 - April 2023 and February - June 2024): {fmt(br)}, "
      f"{100 * br / tax:.0f}%; before {fmt(before)}, after {fmt(after)}")
    loo = {}
    for c in treated:
        loo[c] = window_gap(diff, [x for x in treated if x != c], control, POLICY)[0]
    loo = pd.Series(loo)
    w(f"  leave one treated subclass out: from {fmt(loo.min())} (without {LABELS[loo.idxmin()]}) "
      f"to {fmt(loo.max())} (without {LABELS[loo.idxmax()]})")
    loc = {}
    for g_ in comp:
        loc[g_] = window_gap(across_diff(lp, "PT", [x for x in comp if x != g_], BASE), treated, control, POLICY)[0]
    loc = pd.Series(loc)
    w(f"  leave one comparison country out: from {fmt(loc.min())} (without {loc.idxmin()}) "
      f"to {fmt(loc.max())} (without {loc.idxmax()})")
    # Spillovers onto untreated items: Portugal's control basket against the comparison countries' control baskets.
    ctrl = float(diff.loc[control, POLICY].mean(axis=1).mean())
    avg = diff[POLICY].mean(axis=1)
    wt_t = w23.reindex(treated) / w23.reindex(treated).sum()
    w(f"  treated subclasses alone, Portugal minus comparison countries (no subtraction of untreated): equal weights "
      f"{fmt(avg[treated].mean())} ({100 * avg[treated].mean() / tax:.0f}%), HICP weights {fmt((avg[treated] * wt_t).sum())} "
      f"({100 * (avg[treated] * wt_t).sum() / tax:.0f}%)")
    jump_c = float(across_diff(lp[[ANNOUNCED, FIRST_FULL]], "PT", comp, ANNOUNCED).loc[control, FIRST_FULL].mean())
    w(f"  untreated subclasses, Portugal minus comparison countries: March to May {fmt(jump_c)}, "
      f"May-December mean since February {fmt(ctrl)} (no sign of higher margins on untreated food)")

    # Vegetable oils: 23% to 0%, and back at the 13% intermediate rate.
    for c in groups.get("oils_23", []):
        r_in = subclass_effects(lp[[ANNOUNCED, FIRST_FULL, "2023-06"]], comp, [c], ANNOUNCED, drop)
        r_out = subclass_effects(lp[["2023-12", REVERSED, "2024-03"]], comp, [c], "2023-12", drop)
        tin = wpt.loc[c, FIRST_FULL] - wpt.loc[c, ANNOUNCED]
        tout = wpt.loc[c, REVERSED] - wpt.loc[c, "2023-12"]
        w(f"{LABELS[c]} ({c}): tax wedge {fmt(tin)} at entry (log 1.23), {fmt(tout)} at exit (log 1.13, the intermediate "
          f"rate); price relative to comparison countries March to May {fmt(r_in.loc[c, FIRST_FULL])}, March to June "
          f"{fmt(r_in.loc[c, '2023-06'])}; December to January {fmt(r_out.loc[c, REVERSED])}, December to March "
          f"{fmt(r_out.loc[c, '2024-03'])}")
    w("")


# -------------------------------------------------------------- incidence ---

POLICY_DAYS = 262                 # 18 April 2023 to 4 January 2024, inclusive
HOUSEHOLDS = 4_149_096            # private households, Census 2021 (INE)
GOVERNMENT_COST_MEUR = 550        # 410 (to October) + 140 (extension), government figures as reported by DN


def incidence(w, t: pd.DataFrame, w23: pd.Series, wedge: pd.DataFrame, theta: float) -> None:
    """Effect on headline inflation and the saving by income fifth, at pass-through ``theta``.

    Headline: the food aggregate's tax wedge (HICP food minus HICP food at constant taxes) times its
    weight, so that tax changes outside food do not enter. Saving: the fall in each food subclass's
    price, 1 - exp(theta * entry), summed with HICP weights, gives the share of food spending saved.
    """
    food = [s for s in t.index if s.startswith("CP01")]
    cut = -t["entry"].clip(upper=0)
    wf = w23.reindex(food).fillna(0)
    pt_wedge = wedge.xs("PT", level="geo")
    w01 = w23["CP01"] / 1000
    f01 = pt_wedge.loc["CP01"]
    entry01 = f01[FIRST_FULL] - f01[ANNOUNCED]
    w("Headline HICP")
    w(f"  food (CP01) tax wedge, March to May 2023: {fmt(entry01)} log points; weight of CP01 in 2023: {100 * w01:.1f}%")
    w(f"  effect on the all-items index: {fmt(w01 * entry01)} at full pass-through, {fmt(theta * w01 * entry01)} at "
      f"{100 * theta:.0f}% (HICP-weighted across-country estimate)")
    p00 = np.log(load("prc_hicp_midx").loc[("PT", "CP00")])
    eff = pd.Series(0.0, index=p00.index)
    months_on = [m for m in p00.index if STARTED <= m <= "2023-12"]
    eff[months_on] = theta * w01 * (f01[months_on] - f01[ANNOUNCED])
    cf = p00 - eff
    yoy = pd.DataFrame({"actual": 100 * (np.exp(p00 - p00.shift(12)) - 1),
                        "without_measure": 100 * (np.exp(cf - cf.shift(12)) - 1), "effect_log": eff})
    yoy.loc["2022-01":].round(4).to_csv(RESULTS / "headline_counterfactual.csv")
    ann = pd.DataFrame({"actual": np.exp(p00), "without_measure": np.exp(cf)})
    ann["year"] = [m[:4] for m in ann.index]
    ann = ann.groupby("year").mean()
    infl = 100 * (ann / ann.shift(1) - 1)
    w("  annual average HICP inflation, actual / without the measure: " + ", ".join(
        f"{y} {infl.loc[y, 'actual']:.2f} / {infl.loc[y, 'without_measure']:.2f}" for y in ("2023", "2024")))
    w("  year-on-year inflation, actual / without the measure: " + ", ".join(
        f"{m} {yoy.loc[m, 'actual']:.1f} / {yoy.loc[m, 'without_measure']:.1f}"
        for m in ("2023-05", "2023-12", "2024-01", "2024-05", "2024-12")))
    w("  (the 2024 difference is a base effect: prices in 2024 are compared with 2023 prices lowered by the measure)")
    share_full = float((wf * (1 - np.exp(-cut.reindex(food).fillna(0)))).sum() / wf.sum())
    share_food = float((wf * (1 - np.exp(-theta * cut.reindex(food).fillna(0)))).sum() / wf.sum())
    w(f"  share of food and beverage spending saved: {100 * share_full:.2f}% at full pass-through, "
      f"{100 * share_food:.2f}% at {100 * theta:.0f}%")

    h = pd.read_csv(DATA / "hbs_str_t223.csv")
    cp01 = h[(h["year"] == 2020) & (h["coicop"] == "CP01")].set_index("quant_inc")["value"] / 1000
    e = pd.read_csv(DATA / "hbs_exp_t133.csv")
    pps = e[(e["unit"] == "PPS_HH") & (e["year"] == 2020)].set_index("quant_inc")["value"]
    pps_ae = e[(e["unit"] == "PPS_AE") & (e["year"] == 2020)].set_index("quant_inc")["value"]
    ine = pd.read_csv(DATA / "ine_idef_totals.csv").set_index("wave")["mean_expenditure_eur_per_household"]
    factor = ine[2020] / pps["TOTAL"]
    months = POLICY_DAYS / 365.25 * 12
    rows = []
    for q in ["QU1", "QU2", "QU3", "QU4", "QU5"]:
        cons = factor * pps[q] / 12
        food_m = cons * cp01[q]
        save_m = food_m * share_food
        save_ae = factor * pps_ae[q] / 12 * cp01[q] * share_food
        rows.append({"quintile": q, "consumption_month": cons, "food_month": food_m, "food_share": cp01[q],
                     "saving_month": save_m, "saving_month_adult_eq": save_ae, "saving_period": save_m * months,
                     "saving_pct_consumption": save_m / cons})
    inc = pd.DataFrame(rows).set_index("quintile")
    inc["share_of_total"] = inc["saving_period"] / inc["saving_period"].sum()
    inc.round(6).to_csv(RESULTS / "incidence_by_quintile.csv")
    w("Saving by income fifth, budget survey 2020 wave (fieldwork 2022/23) in euros via INE's mean; same within-food "
      "mix for every fifth (the national HICP weights)")
    for q, r in inc.iterrows():
        w(f"  {q}: food {100 * r['food_share']:.1f}% of consumption, {r['food_month']:.0f} a month per household; saving "
          f"{r['saving_month']:.2f} a month per household ({r['saving_month_adult_eq']:.2f} per adult equivalent), "
          f"{r['saving_period']:.0f} over the {POLICY_DAYS} days, {100 * r['saving_pct_consumption']:.2f}% of consumption, "
          f"{100 * r['share_of_total']:.1f}% of the total")
    total = inc["saving_period"].sum() / 5 * HOUSEHOLDS / 1e6
    g = pd.read_csv(DATA / "nama_10_gdp_p31.csv")
    na23 = g[(g["na_item"] == "P31_S14") & (g["year"] == 2023)]["value"].iloc[0]
    survey = ine[2020] * HOUSEHOLDS / 1e6
    scale = na23 / survey
    w(f"  resident households in total over the {POLICY_DAYS} days: about {total:.0f} million euros on the survey basis "
      f"({total * share_full / share_food:.0f} at full pass-through); survey spending is {100 / scale:.0f}% of resident household "
      f"consumption in the 2023 national accounts, and scaled to them the saving is about {total * scale:.0f} million "
      f"({total * scale * share_full / share_food:.0f} at full pass-through)")
    w(f"  government ex-ante estimate of forgone VAT: {GOVERNMENT_COST_MEUR} million (it does not depend on pass-through "
      "and includes purchases by non-residents)")
    w(f"  the households' total saving, if paid only to the poorest fifth: {1 / inc.loc['QU1', 'share_of_total']:.1f} "
      f"times what that fifth saved ({total * 1e6 / (HOUSEHOLDS / 5) / months:.2f} euros a month per household on the "
      f"survey basis, {total * scale * 1e6 / (HOUSEHOLDS / 5) / months:.2f} scaled to the national accounts)")
    w("")


# ------------------------------------------------------------------- main ---

def fmt(x: float, scale: float = 100, nd: int = 2) -> str:
    v = scale * x
    return f"{0.0 if abs(v) < 0.5 * 10 ** -nd else v:+.{nd}f}"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    lines: list[str] = []
    w = lines.append

    lp = log_prices()
    wedge = tax_wedge()
    wt = weights()

    # 1. Treatment and comparison set.
    t = classify(wedge)
    t.round(6).to_csv(RESULTS / "treatment_by_subclass.csv")
    groups = {g: list(t.index[t["group"] == g]) for g in t["group"].unique()}
    treated, control, partial = groups.get("treated", []), groups.get("control", []), groups.get("partial", [])
    w("Treatment from the tax wedge (log HICP - log HICP at constant taxes), Portuguese food subclasses")
    w(f"  full 6% cut: log(1.06) = {100 * VAT_6:.2f} log points; 23% cut: log(1.23) = {100 * VAT_23:.2f}")
    for g in ("treated", "partial", "oils_23", "control", "control_other_tax", "unclear"):
        codes = groups.get(g, [])
        w(f"  {g} ({len(codes)}): " + ", ".join(
            f"{s} {100 * t.loc[s, 'entry']:+.2f}/{100 * t.loc[s, 'exit']:+.2f}" for s in codes))
    w("  (entry: wedge change March to May 2023; exit: September 2023 to January 2024; log points)")
    pt_w = wt.xs("PT", level="geo")
    w23 = pt_w[2023]
    for g in ("treated", "partial", "oils_23", "control"):
        codes = groups.get(g, [])
        w(f"  weight in the 2023 HICP, {g}: {w23.reindex(codes).sum() / 10:.2f}%")
    comp, screen = donors(wedge, treated + control)
    screen.round(5).to_csv(RESULTS / "donor_screen.csv")
    w(f"  comparison countries, no tax change in the treated or untreated subclasses since {SCREEN_FROM} "
      f"({len(comp)}): {', '.join(comp)}")
    w("  excluded: " + "; ".join(
        f"{g} ({r['moved']} of {r['subclasses']} subclasses moved, largest {100 * r['largest_move']:.1f})"
        if r["moved"] else f"{g} ({r['other_reason']})"
        for g, r in screen.iterrows() if g not in comp))
    w("")

    # 2-3. Event studies.
    months = [c for c in lp.columns if c >= "2021-01"]
    lp_w = lp.loc[:, PRE_START:]
    d_within = within_diff(lp_w, "PT", BASE)
    d_across = across_diff(lp_w, "PT", comp, BASE)
    wc = group_gap(d_within, treated, control)
    ac = group_gap(d_across, treated, control)
    wcw = group_gap(d_within, treated, control, w23)
    acw = group_gap(d_across, treated, control, w23)
    wedge_pt = rebased(wedge.xs("PT", level="geo"), BASE)
    tax = group_gap(wedge_pt, treated, control)["gap"]
    es = pd.DataFrame({"within": wc["gap"], "within_se": wc["se"], "across": ac["gap"], "across_se": ac["se"],
                       "within_weighted": wcw["gap"], "across_weighted": acw["gap"], "tax_wedge": tax})
    es.loc[months].round(6).to_csv(RESULTS / "event_study.csv")
    w("Event study, treated minus untreated food subclasses, log change since February 2023 (x100)")
    w("  month: within Portugal (se) | across countries (se) | HICP-weighted within / across | tax wedge")
    for m in [c for c in months if "2022-09" <= c <= "2024-06"] + ["2024-12", "2025-06", lp.columns[-1]]:
        r = es.loc[m]
        w(f"  {m}: {fmt(r['within'])} ({100 * r['within_se']:.2f}) | {fmt(r['across'])} ({100 * r['across_se']:.2f}) | "
          f"{fmt(r['within_weighted'])} / {fmt(r['across_weighted'])} | {fmt(r['tax_wedge'])}")
    pre = [c for c in es.index if "2022-03" <= c <= "2023-02"]
    w(f"  pre-period, March 2022 - February 2023: mean gap within {fmt(es.loc[pre, 'within'].mean())}, "
      f"across {fmt(es.loc[pre, 'across'].mean())}; December 2022 - April 2023 plateau: within "
      f"{fmt(es.loc['2022-12':'2023-04', 'within'].mean())}, across {fmt(es.loc['2022-12':'2023-04', 'across'].mean())}")
    taxm = es.loc[POLICY, "tax_wedge"].mean()
    w("  May-December 2023 mean (pass-through, 95% CI clustered by subclass):")
    for name, diff, ww in (("within Portugal (includes Europe-wide relative price changes)", d_within, None),
                           ("across countries", d_across, None), ("across countries, HICP-weighted", d_across, w23)):
        g, se = window_gap(diff, treated, control, POLICY, ww)
        w(f"    {name}: {fmt(g)}; {pass_through(g, se, taxm)}")
    w(f"  tax wedge May-December 2023: {fmt(taxm)}")
    w(f"  announcement month (March 2023): within {fmt(es.loc[ANNOUNCED, 'within'])}, across {fmt(es.loc[ANNOUNCED, 'across'])}; "
      f"April 2023: within {fmt(es.loc[STARTED, 'within'])}, across {fmt(es.loc[STARTED, 'across'])}")
    w(f"  comparison countries' own treated-minus-untreated change, May-December 2023 mean: "
      f"{fmt(es.loc[POLICY, 'within'].mean() - es.loc[POLICY, 'across'].mean())} (the difference between the two estimators)")
    w("")

    # 4. Inference.
    w("Placebo jumps: changes in the treated-minus-untreated gap, March 2019 - December 2022")
    jumps = {}
    for name in ("within", "across"):
        g = es[name]
        d2 = (g - g.shift(2)).loc["2019-03":"2022-12"]
        d1 = (g - g.shift(1)).loc["2019-02":"2022-12"]
        a2 = g[FIRST_FULL] - g[ANNOUNCED]
        a1 = g[REVERSED] - g["2023-12"]
        jumps[f"{name}_2m"], jumps[f"{name}_1m"] = d2, d1
        w(f"  {name}: two-month changes sd {100 * d2.std():.2f}, range {fmt(d2.min())} ({d2.idxmin()}) to {fmt(d2.max())} "
          f"({d2.idxmax()}); March-May 2023 {fmt(a2)}; as large in absolute value: {int((d2.abs() >= abs(a2)).sum())} of {len(d2)}")
        w(f"  {name}: one-month changes sd {100 * d1.std():.2f}, range {fmt(d1.min())} to {fmt(d1.max())} ({d1.idxmax()}); "
          f"December 2023 - January 2024 {fmt(a1)}; as large in absolute value: {int((d1.abs() >= abs(a1)).sum())} of {len(d1)}")
    pd.DataFrame(jumps).round(6).to_csv(RESULTS / "placebo_jumps.csv")
    w("Placebo dates: the estimator on fake events, January 2019 - June 2022, mean gap over the 8 months after")
    pdates = pd.DataFrame({"within": placebo_dates(lp_w, treated, control),
                           "across": placebo_dates(lp_w, treated, control, comp)})
    pdates.round(6).to_csv(RESULTS / "placebo_dates.csv")
    for name in ("within", "across"):
        v = pdates[name]
        actual = window_gap(d_within if name == "within" else d_across, treated, control, POLICY)[0]
        w(f"  {name}: {len(v)} dates, sd {100 * v.std():.2f}, range {fmt(v.min())} to {fmt(v.max())}; actual {fmt(actual)}; "
          f"as large in absolute value: {int((v.abs() >= abs(actual)).sum())}")
    pc = placebo_countries(lp_w, comp, treated, control)
    pc.round(6).to_csv(RESULTS / "placebo_countries.csv", header=["mean_gap_may_dec_2023"])
    actual_a = es.loc[POLICY, "across"].mean()
    w("Placebo countries: each comparison country treated as if it were Portugal (Portugal's treated set), May-December 2023")
    w("  " + ", ".join(f"{g} {fmt(v)}" for g, v in pc.items()))
    w(f"  sd {100 * pc.std():.2f}; actual Portugal {fmt(actual_a)}; as large in absolute value: "
      f"{int((pc.abs() >= abs(actual_a)).sum())} of {len(pc)}")
    w("")

    # 5. Synthetic control on the relative price of the treated basket.
    rel = pd.DataFrame({geo: basket(lp_w, geo, treated, w23) - basket(lp_w, geo, control, w23)
                        for geo in ["PT"] + comp}).T
    pre_months = [c for c in rel.columns if PRE_START <= c <= BASE]
    v, gap = synthetic_control(rel, "PT", pre_months)
    v.round(4).to_csv(RESULTS / "synthetic_weights.csv", header=["weight"])
    pd.DataFrame({"portugal": rel.loc["PT"], "synthetic": rel.loc["PT"] - gap, "gap": gap}).round(6).to_csv(
        RESULTS / "synthetic_gap.csv")
    bk = pd.DataFrame({
        "pt_treated": basket(lp_w, "PT", treated, w23), "pt_control": basket(lp_w, "PT", control, w23),
        "comparison_treated": pd.DataFrame({g: basket(lp_w, g, treated, w23) for g in comp}).mean(axis=1),
        "comparison_control": pd.DataFrame({g: basket(lp_w, g, control, w23) for g in comp}).mean(axis=1)})
    bk.round(6).to_csv(RESULTS / "baskets.csv")
    rebased(wedge.xs("PT", level="geo"), BASE).loc[treated + partial + groups.get("oils_23", []), "2022-01":].round(6).to_csv(
        RESULTS / "tax_wedge_pt.csv")
    rmspe_pre = float(np.sqrt((gap[pre_months] ** 2).mean()))
    post = [c for c in POLICY if c in gap.index]
    w("Synthetic control: relative price of the treated basket (treated minus untreated, 2023 Portuguese weights)")
    w("  weights: " + ", ".join(f"{g} {x:.2f}" for g, x in v.sort_values(ascending=False).items() if x >= 0.005))
    w(f"  pre-period RMSPE {100 * rmspe_pre:.2f} log points; mean gap May-December 2023 {fmt(gap[post].mean())} "
      f"({100 * gap[post].mean() / taxm:.0f}%); January-June 2024 {fmt(gap['2024-01':'2024-06'].mean())}; "
      f"2025 {fmt(gap['2025-01':'2025-12'].mean())}")
    v2, gap2 = synthetic_control(rel.drop(index="EL"), "PT", pre_months)
    w(f"  without Greece: weights " + ", ".join(f"{g} {x:.2f}" for g, x in v2.sort_values(ascending=False).items() if x >= 0.005)
      + f"; pre-period RMSPE {100 * float(np.sqrt((gap2[pre_months] ** 2).mean())):.2f}; mean gap May-December 2023 "
      f"{fmt(gap2[post].mean())} ({100 * gap2[post].mean() / taxm:.0f}%)")
    ratios = {}
    for geo in rel.index:
        _vv, gg = synthetic_control(rel, geo, pre_months)
        ratios[geo] = float(np.sqrt((gg[post] ** 2).mean())) / float(np.sqrt((gg[pre_months] ** 2).mean()))
    ratios = pd.Series(ratios).sort_values(ascending=False)
    ratios.round(4).to_csv(RESULTS / "synthetic_rmspe_ratios.csv", header=["post_over_pre_rmspe"])
    rank = int((ratios >= ratios["PT"]).sum())
    w(f"  post/pre RMSPE ratio: Portugal {ratios['PT']:.1f}, rank {rank} of {len(ratios)} "
      f"(permutation p-value {rank / len(ratios):.3f}, the smallest attainable with {len(ratios)} units)")
    w("")

    # 6-7. Reversal and persistence.
    ex_w = group_gap(within_diff(lp_w, "PT", EXIT_BASE), treated, control)
    ex_a = group_gap(across_diff(lp_w, "PT", comp, EXIT_BASE), treated, control)
    tax_ex = group_gap(rebased(wedge.xs("PT", level="geo"), EXIT_BASE), treated, control)["gap"]
    ex = pd.DataFrame({"within": ex_w["gap"], "across": ex_a["gap"], "across_se": ex_a["se"], "tax_wedge": tax_ex})
    ex.loc[[c for c in ex.index if c >= "2023-06"]].round(6).to_csv(RESULTS / "event_study_reversal.csv")
    w("Reversal: treated minus untreated, log change since September 2023 (x100)")
    for m in [c for c in ex.index if "2023-08" <= c <= "2024-06"]:
        r = ex.loc[m]
        w(f"  {m}: within {fmt(r['within'])}, across {fmt(r['across'])}, tax wedge {fmt(r['tax_wedge'])}")
    jm = ["2024-01", "2024-02", "2024-03"]
    for name, diff in (("within", within_diff(lp_w, "PT", EXIT_BASE)), ("across", across_diff(lp_w, "PT", comp, EXIT_BASE))):
        g, se = window_gap(diff, treated, control, jm)
        w(f"  January-March 2024 mean, {name}: {fmt(g)}; {pass_through(g, se, ex.loc[jm, 'tax_wedge'].mean())}")
    w(f"  October-December 2023, after the end was announced and before VAT returned: across "
      f"{fmt(ex.loc[['2023-10', '2023-11', '2023-12'], 'across'].mean())}")
    w("Persistence: gap relative to February 2023 after the reversal (0 = back on the comparison path)")
    for yr in ("2024", "2025"):
        ms = [c for c in es.index if c.startswith(yr)]
        w(f"  {yr} ({ms[0]} to {ms[-1]}): within {fmt(es.loc[ms, 'within'].mean())}, across {fmt(es.loc[ms, 'across'].mean())}")
    w("")

    # Short windows, dose-response, robustness, oils.
    extensions(w, lp_w, wedge, t, comp, w23)

    # 8. Headline inflation and incidence, at the HICP-weighted across-country pass-through.
    theta = window_gap(d_across, treated, control, POLICY, w23)[0] / taxm
    incidence(w, t, w23, wedge, theta)

    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
