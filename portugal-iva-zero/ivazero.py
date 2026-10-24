"""Pass-through of Portugal's zero VAT on 46 food products (IVA Zero) in official prices.

From 18 April 2023 to 4 January 2024 Portugal applied a 0% VAT rate to a basket
of 46 food products, previously taxed at 6% (vegetable oils at 23%). The measure
was announced on 24 March 2023 and its end on 27 October 2023; VAT returned on
5 January 2024.

The analysis uses Eurostat's monthly HICP by ECOICOP subclass (5 digits), the
same indices at constant tax rates (HICP-CT), and the item weights, for all EU
member states.

1. Treatment from the data. INE's constant-tax index assumes that every tax
   change is passed on in full, so the tax wedge
       w(c,s,t) = log HICP(c,s,t) - log HICP-CT(c,s,t)
   moves only when a tax rate changes. Its change between March and May 2023
   measures, subclass by subclass, how much of each Portuguese subclass the VAT
   cut covered. The same wedge screens the other countries: any country whose
   food wedges moved in 2022-2025 is dropped from the comparison set.
2. Within Portugal: an event study of treated against untreated food subclasses,
   each series in logs relative to February 2023 (the last month before the
   announcement). With subclass and month fixed effects on a balanced panel, the
   event coefficient is the difference between the two groups' mean log change.
3. Across countries: the same difference after subtracting, subclass by
   subclass, the mean log change in the comparison countries. This removes
   price shocks that hit a product everywhere in Europe (olive oil, sugar).
4. Inference: placebo countries (each comparison country treated as if it were
   Portugal, with Portugal's treated set) and placebo dates (the same estimator
   centred on every month from 2019 to 2022).
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
END_ANNOUNCED = "2023-10"    # 27 October 2023
REVERSED = "2024-01"         # VAT back on 5 January 2024
BASE = "2023-02"             # last month before the announcement
EXIT_BASE = "2023-10"        # last month before the reversal took effect
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
    t.loc[(t["group"] == "control") & (t["other_moves"] > SCREEN_TOL), "group"] = "control_other_tax"
    return t


# Countries dropped for reasons the tax wedge cannot show.
PRICE_CONTROLS = {"HU": "administrative price caps on basic foods until 1 August 2023"}


def donors(wedge: pd.DataFrame, codes: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Countries whose tax wedges did not move from 2022 on in any of ``codes``, and the screen itself."""
    rows = []
    for geo in sorted(wedge.index.get_level_values("geo").unique()):
        if geo == "PT":
            continue
        w = wedge.xs(geo, level="geo")
        have = [s for s in codes if s in w.index]
        if not have:
            continue
        mv = w.loc[have, SCREEN_FROM:].diff(axis=1).abs().max(axis=1)
        rows.append({"geo": geo, "subclasses": len(have), "moved": int((mv >= SCREEN_TOL).sum()),
                     "largest_move": float(mv.max()), "other_reason": PRICE_CONTROLS.get(geo, "")})
    screen = pd.DataFrame(rows).set_index("geo")
    keep = screen[(screen["moved"] == 0) & (screen["other_reason"] == "")]
    return list(keep.index), screen


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


def window_mean(est: pd.DataFrame, months: list[str]) -> float:
    return float(est.loc[[m for m in months if m in est.index], "gap"].mean())


def placebo_dates(lp: pd.DataFrame, treated: list[str], control: list[str]) -> pd.Series:
    """Within-Portugal estimator centred on fake announcement months, 2019-01 to 2022-06."""
    pt = lp.xs("PT", level="geo")
    cols = list(pt.columns)
    out = {}
    for i, m in enumerate(cols):
        if not ("2019-01" <= m <= "2022-06"):
            continue
        base = cols[i - 1]
        window = cols[i + 2:i + 10]           # same layout as the real event: base, announcement, partial month, 8 months
        r = rebased(pt, base)
        g = group_gap(r[window], treated, control)
        out[m] = float(g["gap"].mean())
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
    """Log of a fixed-weight (Laspeyres in logs) basket of subclasses, rebased to BASE."""
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


def subclass_effects(lp: pd.DataFrame, comp: list[str], codes: list[str], base: str) -> pd.DataFrame:
    """Each Portuguese subclass's log change since ``base`` minus its mean in the comparison countries."""
    r = rebased(lp, base)
    own = r.xs("PT", level="geo")
    cmp_ = r.loc[r.index.get_level_values("geo").isin(comp)].groupby(level="coicop").mean()
    keep = [c for c in codes if c in own.index and c in cmp_.index]
    return own.loc[keep] - cmp_.loc[keep]


def extensions(w, lp: pd.DataFrame, wedge: pd.DataFrame, t: pd.DataFrame, comp: list[str],
               w23: pd.Series) -> None:
    groups = {g: list(t.index[t["group"] == g]) for g in t["group"].unique()}
    treated, control, partial = groups["treated"], groups["control"], groups.get("partial", [])
    wpt = wedge.xs("PT", level="geo")

    # Short windows around each change: least exposed to slow trends.
    w("Short windows (treated minus untreated, log points x100; pass-through = change / tax-wedge change)")
    windows = [("entry", ANNOUNCED, FIRST_FULL), ("entry", ANNOUNCED, "2023-06"), ("exit", "2023-12", REVERSED),
               ("exit", "2023-12", "2024-02"), ("exit", "2023-12", "2024-03"), ("exit", EXIT_BASE, REVERSED),
               ("exit", EXIT_BASE, "2024-03")]
    rows = []
    for kind, a, b in windows:
        tax = group_gap(rebased(wpt, a)[[b]], treated, control)["gap"].iloc[0]
        wi = within_country(lp[[a, b]], "PT", treated, control, a)["gap"].iloc[-1]
        ac = across_countries(lp[[a, b]], "PT", comp, treated, control, a)["gap"].iloc[-1]
        rows.append({"kind": kind, "from": a, "to": b, "tax": tax, "within": wi, "across": ac})
        w(f"  {kind} {a} to {b}: within {fmt(wi)} ({100 * wi / tax:.0f}%), across {fmt(ac)} ({100 * ac / tax:.0f}%), "
          f"tax wedge {fmt(tax)}")
    pd.DataFrame(rows).round(6).to_csv(RESULTS / "short_windows.csv", index=False)

    # Dose-response: every Portuguese subclass, effect against the change in its tax wedge.
    codes = treated + partial + control
    eff = subclass_effects(lp, comp, codes, BASE)
    jump = subclass_effects(lp[[ANNOUNCED, FIRST_FULL]], comp, codes, ANNOUNCED)[FIRST_FULL]
    dose = pd.DataFrame({"wedge_entry": t.loc[eff.index, "entry"], "group": t.loc[eff.index, "group"],
                         "effect_may_dec_2023": eff[POLICY].mean(axis=1), "effect_march_to_may": jump,
                         "effect_2024": eff[[c for c in eff.columns if c.startswith("2024")]].mean(axis=1),
                         "weight_2023": w23.reindex(eff.index) / 10,
                         "label": [LABELS.get(c, c) for c in eff.index]})
    dose.round(6).to_csv(RESULTS / "dose_response.csv")
    w("Dose-response across Portuguese subclasses (treated, partial and control), effect vs tax-wedge change at entry")
    for col in ("effect_march_to_may", "effect_may_dec_2023"):
        b, a, se = ols(dose["wedge_entry"].to_numpy(), dose[col].to_numpy())
        w(f"  {col}: slope (pass-through) {100 * b:.0f}% (se {100 * se:.0f}), intercept {fmt(a)}; n = {len(dose)}")
    b, a, se = ols(dose["wedge_entry"].to_numpy(), dose["effect_2024"].to_numpy())
    w(f"  effect_2024 (after the reversal): slope {100 * b:.0f}% (se {100 * se:.0f}), intercept {fmt(a)}")
    part = dose[dose["group"] == "partial"]
    b, a, se = ols(part["wedge_entry"].to_numpy(), part["effect_march_to_may"].to_numpy())
    w(f"  partial subclasses only, March to May: slope {100 * b:.0f}% (se {100 * se:.0f}); n = {len(part)}")
    tr = dose[dose["group"] == "treated"].sort_values("effect_may_dec_2023")
    w("  treated subclasses, May-December 2023 effect / March-May jump: " + "; ".join(
        f"{r['label']} {fmt(r['effect_may_dec_2023'])} / {fmt(r['effect_march_to_may'])}" for _, r in tr.iterrows()))

    # Robustness of the across-country May-December estimate.
    w("Robustness, across countries, May-December 2023 mean (pass-through against -5.76)")
    tax = group_gap(rebased(wpt, BASE)[POLICY], treated, control)["gap"].mean()
    beverages = [c for c in control if c.startswith("CP012")]
    variants = {
        "baseline": (treated, control, comp),
        "food-only controls (no beverages)": (treated, [c for c in control if c not in beverages], comp),
        "controls without preserved milk and dried vegetables": (treated, [c for c in control if c not in ("CP01143", "CP01173")], comp),
        "treated without olive oil": ([c for c in treated if c != "CP01153"], control, comp),
        "comparison adds HU, FR, IE, NL, SI": (treated, control, comp + ["HU", "FR", "IE", "NL", "SI"]),
    }
    for name, (tt, cc, cm) in variants.items():
        est = across_countries(lp, "PT", cm, tt, cc, BASE)
        v = window_mean(est, POLICY)
        w(f"  {name}: {fmt(v)} ({100 * v / tax:.0f}%)")
    loo = {}
    for c in treated:
        est = across_countries(lp, "PT", comp, [x for x in treated if x != c], control, BASE)
        loo[c] = window_mean(est, POLICY)
    loo = pd.Series(loo)
    w(f"  leave one treated subclass out: from {fmt(loo.min())} (without {LABELS[loo.idxmin()]}) "
      f"to {fmt(loo.max())} (without {LABELS[loo.idxmax()]})")
    loc = {}
    for g in comp:
        est = across_countries(lp, "PT", [x for x in comp if x != g], treated, control, BASE)
        loc[g] = window_mean(est, POLICY)
    loc = pd.Series(loc)
    w(f"  leave one comparison country out: from {fmt(loc.min())} (without {loc.idxmin()}) "
      f"to {fmt(loc.max())} (without {loc.idxmax()})")

    # Vegetable oils: 23% to 0%, and back at a different rate.
    oils = groups.get("oils_23", [])
    for c in oils:
        r_in = subclass_effects(lp[[ANNOUNCED, FIRST_FULL, "2023-06"]], comp, [c], ANNOUNCED)
        r_out = subclass_effects(lp[["2023-12", REVERSED, "2024-03"]], comp, [c], "2023-12")
        tin = wpt.loc[c, FIRST_FULL] - wpt.loc[c, ANNOUNCED]
        tout = wpt.loc[c, REVERSED] - wpt.loc[c, "2023-12"]
        w(f"{LABELS[c]} ({c}): tax wedge {fmt(tin)} at entry, {fmt(tout)} at exit (exp: {100 * (np.exp(tout) - 1):.1f}% "
          f"VAT on the January 2024 price); price relative to comparison countries March to May {fmt(r_in.loc[c, FIRST_FULL])}, "
          f"March to June {fmt(r_in.loc[c, '2023-06'])}; December to January {fmt(r_out.loc[c, REVERSED])}, "
          f"December to March {fmt(r_out.loc[c, '2024-03'])}")
    w("")


# -------------------------------------------------------------- incidence ---

POLICY_DAYS = 262                 # 18 April 2023 to 4 January 2024, inclusive
HOUSEHOLDS = 4_149_096            # private households, Census 2021 (INE)
GOVERNMENT_COST_MEUR = 550        # 410 (to October) + 140 (extension), government figures as reported by DN


def incidence(w, t: pd.DataFrame, w23: pd.Series, wedge: pd.DataFrame, theta: float) -> None:
    """Effect on headline inflation and the saving by income fifth.

    The saving on a subclass is the fall in its price, 1 - exp(theta * entry), with entry the
    change in its tax wedge; summed with HICP weights it gives the share of spending saved.
    """
    food = [s for s in t.index if s.startswith("CP01")]
    cut = -t["entry"].clip(upper=0)
    full = (w23.reindex(food).fillna(0) * cut.reindex(food).fillna(0)).sum() / 1000
    est = (w23.reindex(food).fillna(0) * (1 - np.exp(-theta * cut.reindex(food).fillna(0)))).sum() / 1000
    ct = wedge.xs("PT", level="geo").loc["CP00"]
    entry_ct = ct[FIRST_FULL] - ct[ANNOUNCED]
    w("Headline HICP")
    w(f"  all-items tax wedge, March to May 2023: {fmt(entry_ct)} log points; "
      f"October 2023 to January 2024: {fmt(ct[REVERSED] - ct[EXIT_BASE])} (January 2024 also has other tax changes)")
    w(f"  estimated effect on the all-items index: {fmt(theta * entry_ct)} log points ({100 * theta:.0f}% of the wedge)")
    w(f"  check, weighted sum of food subclass wedge changes with 2023 weights: {100 * full:.2f} points at full "
      f"pass-through, {100 * est:.2f} at the estimated pass-through")
    # Measured inflation with and without the measure: the effect is removed from April to December 2023.
    p00 = np.log(load("prc_hicp_midx").loc[("PT", "CP00")])
    eff = pd.Series(0.0, index=p00.index)
    months_on = [m for m in p00.index if STARTED <= m <= "2023-12"]
    eff[months_on] = theta * (ct[months_on] - ct[ANNOUNCED])
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
    share_food = (w23.reindex(food).fillna(0) * (1 - np.exp(-theta * cut.reindex(food).fillna(0)))).sum() \
        / w23.reindex(food).fillna(0).sum()
    w(f"  food and non-alcoholic beverages: weight {w23.reindex(food).sum() / 10:.1f}% of the HICP; "
      f"share of food spending saved at the estimated pass-through {100 * share_food:.2f}%")

    h = pd.read_csv(DATA / "hbs_str_t223.csv")
    cp01 = h[(h["year"] == 2020) & (h["coicop"] == "CP01")].set_index("quant_inc")["value"] / 1000
    e = pd.read_csv(DATA / "hbs_exp_t133.csv")
    pps = e[(e["unit"] == "PPS_HH") & (e["year"] == 2020)].set_index("quant_inc")["value"]
    ine = pd.read_csv(DATA / "ine_idef_totals.csv").set_index("wave")["mean_expenditure_eur_per_household"]
    factor = ine[2020] / pps["TOTAL"]
    months = POLICY_DAYS / 365.25 * 12
    rows = []
    for q in ["QU1", "QU2", "QU3", "QU4", "QU5"]:
        cons = factor * pps[q] / 12
        food_m = cons * cp01[q]
        save_m = food_m * share_food
        rows.append({"quintile": q, "consumption_month": cons, "food_month": food_m, "food_share": cp01[q],
                     "saving_month": save_m, "saving_period": save_m * months,
                     "saving_pct_consumption": save_m / cons})
    inc = pd.DataFrame(rows).set_index("quintile")
    inc["share_of_total"] = inc["saving_period"] / inc["saving_period"].sum()
    inc.round(6).to_csv(RESULTS / "incidence_by_quintile.csv")
    w("Saving by income fifth, per household, budget survey 2020 wave (fieldwork 2022/23) in euros via INE's mean")
    for q, r in inc.iterrows():
        w(f"  {q}: food {100 * r['food_share']:.1f}% of consumption, {r['food_month']:.0f} a month; saving "
          f"{r['saving_month']:.2f} a month, {r['saving_period']:.0f} over the {POLICY_DAYS} days, "
          f"{100 * r['saving_pct_consumption']:.2f}% of consumption, {100 * r['share_of_total']:.1f}% of the total")
    total = inc["saving_period"].sum() / 5 * HOUSEHOLDS / 1e6
    g = pd.read_csv(DATA / "nama_10_gdp_p31.csv")
    na22 = g[(g["na_item"] == "P31_S14") & (g["year"] == 2022)]["value"].iloc[0]
    survey22 = ine[2020] * HOUSEHOLDS / 1e6
    w(f"  resident households in total, survey basis: about {total:.0f} million euros over the period "
      f"({HOUSEHOLDS:,} households); government cost estimate {GOVERNMENT_COST_MEUR} million")
    w(f"  survey coverage: mean expenditure x households = {survey22:,.0f} million euros a year, against resident household "
      f"consumption in the 2022 national accounts of {na22:,.0f} million ({100 * survey22 / na22:.0f}%); scaled to the "
      f"national accounts the saving would be about {total * na22 / survey22:.0f} million")
    w(f"  the same money paid only to the poorest fifth: {1 / inc.loc['QU1', 'share_of_total']:.1f} times what that "
      f"fifth saved ({total * 1e6 / (HOUSEHOLDS / 5) / months:.2f} euros a month per household on the survey basis)")
    w(f"  the government cost estimate paid only to the poorest fifth: "
      f"{GOVERNMENT_COST_MEUR * 1e6 / (HOUSEHOLDS / 5) / months:.2f} euros a month per household")
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
    w("  (entry: wedge change March to May 2023; exit: October 2023 to January 2024; log points)")
    pt_w = wt.xs("PT", level="geo")
    w23 = pt_w[2023] if 2023 in pt_w.columns else pt_w.iloc[:, -1]
    for g in ("treated", "partial", "oils_23", "control"):
        codes = groups.get(g, [])
        w(f"  weight in the 2023 HICP, {g}: {w23.reindex(codes).sum() / 10:.2f}%")
    comp, screen = donors(wedge, treated + control)
    screen.round(5).to_csv(RESULTS / "donor_screen.csv")
    w(f"  comparison countries, no tax change in the treated or control subclasses since {SCREEN_FROM} "
      f"({len(comp)}): {', '.join(comp)}")
    w("  excluded: " + "; ".join(
        f"{g} ({r['moved']} of {r['subclasses']} subclasses moved, largest {100 * r['largest_move']:.1f})"
        if r["moved"] else f"{g} ({r['other_reason']})"
        for g, r in screen.iterrows() if g not in comp))
    w("")

    # 2-3. Event studies.
    months = [c for c in lp.columns if c >= "2021-01"]
    lp_w = lp.loc[:, PRE_START:]
    wc = within_country(lp_w, "PT", treated, control, BASE)
    ac = across_countries(lp_w, "PT", comp, treated, control, BASE)
    wcw = within_country(lp_w, "PT", treated, control, BASE, w23)
    acw = across_countries(lp_w, "PT", comp, treated, control, BASE, w23)
    wedge_pt = rebased(wedge.xs("PT", level="geo"), BASE)
    tax = group_gap(wedge_pt, treated, control)["gap"]
    es = pd.DataFrame({"within": wc["gap"], "within_se": wc["se"], "across": ac["gap"], "across_se": ac["se"],
                       "within_weighted": wcw["gap"], "across_weighted": acw["gap"], "tax_wedge": tax})
    es.loc[months].round(6).to_csv(RESULTS / "event_study.csv")
    w("Event study, treated minus untreated food subclasses, log change since February 2023 (x100)")
    w("  month: within Portugal (se) | across countries (se) | HICP-weighted within / across | tax wedge")
    for m in [c for c in months if "2022-09" <= c <= "2024-06"] + ["2024-12", "2025-06", lp.columns[-1]]:
        if m not in es.index:
            continue
        r = es.loc[m]
        w(f"  {m}: {fmt(r['within'])} ({100 * r['within_se']:.2f}) | {fmt(r['across'])} ({100 * r['across_se']:.2f}) | "
          f"{fmt(r['within_weighted'])} / {fmt(r['across_weighted'])} | {fmt(r['tax_wedge'])}")
    pre = [c for c in es.index if "2022-03" <= c <= "2023-02"]
    w(f"  pre-period check, mean gap March 2022 - February 2023: within {fmt(es.loc[pre, 'within'].mean())}, "
      f"across {fmt(es.loc[pre, 'across'].mean())}; largest absolute: within {100 * es.loc[pre, 'within'].abs().max():.2f}, "
      f"across {100 * es.loc[pre, 'across'].abs().max():.2f}")
    for name in ("within", "across", "within_weighted", "across_weighted"):
        eff = es.loc[POLICY, name].mean()
        taxm = es.loc[POLICY, "tax_wedge"].mean()
        w(f"  May-December 2023 mean, {name}: {fmt(eff)} log points; tax wedge {fmt(taxm)}; pass-through {100 * eff / taxm:.0f}%")
    w(f"  announcement month (March 2023): within {fmt(es.loc[ANNOUNCED, 'within'])}, across {fmt(es.loc[ANNOUNCED, 'across'])}; "
      f"April 2023: within {fmt(es.loc[STARTED, 'within'])}, across {fmt(es.loc[STARTED, 'across'])}")
    w("")

    # 4. Inference.
    pd_ = placebo_dates(lp_w, treated, control)
    pd_.round(6).to_csv(RESULTS / "placebo_dates.csv", header=["mean_gap"])
    actual_w = es.loc[POLICY, "within"].mean()
    actual_a = es.loc[POLICY, "across"].mean()
    w("Placebo dates: within-Portugal estimator on fake events, January 2019 - June 2022, mean gap over the 8 months after")
    w(f"  {len(pd_)} placebo dates; mean {fmt(pd_.mean())}, sd {100 * pd_.std():.2f}, min {fmt(pd_.min())}, max {fmt(pd_.max())}; "
      f"actual {fmt(actual_w)}; placebo dates with an effect at least as negative: {(pd_ <= actual_w).sum()}")
    # Sharp changes: how often did the gap move as much in two months (entry) or one month (exit) before 2023?
    w("Placebo jumps: changes in the treated-minus-untreated gap over two months and one month, March 2019 - December 2022")
    jumps = {}
    for name in ("within", "across"):
        g = es[name]
        d2 = (g - g.shift(2)).loc["2019-03":"2022-12"]
        d1 = (g - g.shift(1)).loc["2019-02":"2022-12"]
        a2 = g[FIRST_FULL] - g[ANNOUNCED]
        a1 = g[REVERSED] - g["2023-12"]
        jumps[f"{name}_2m"], jumps[f"{name}_1m"] = d2, d1
        w(f"  {name}: two-month changes sd {100 * d2.std():.2f}, most negative {fmt(d2.min())} ({d2.idxmin()}); "
          f"March-May 2023 {fmt(a2)}, {int((d2 <= a2).sum())} of {len(d2)} as negative. One-month changes sd "
          f"{100 * d1.std():.2f}, largest {fmt(d1.max())} ({d1.idxmax()}); December 2023 - January 2024 {fmt(a1)}, "
          f"{int((d1 >= a1).sum())} of {len(d1)} as large")
    pd.DataFrame(jumps).round(6).to_csv(RESULTS / "placebo_jumps.csv")
    pc = placebo_countries(lp_w, comp, treated, control)
    pc.round(6).to_csv(RESULTS / "placebo_countries.csv", header=["mean_gap_may_dec_2023"])
    w("Placebo countries: each comparison country treated as if it were Portugal (Portugal's treated set), May-December 2023")
    w("  " + ", ".join(f"{g} {fmt(v)}" for g, v in pc.items()))
    w(f"  actual Portugal {fmt(actual_a)}; placebo countries with an effect at least as negative: {(pc <= actual_a).sum()} of {len(pc)}")
    w("")

    # 5. Synthetic control on the relative price of the treated basket.
    countries = ["PT"] + comp
    rel = {}
    for geo in countries:
        try:
            rel[geo] = basket(lp_w, geo, treated, w23) - basket(lp_w, geo, control, w23)
        except (KeyError, ZeroDivisionError):
            continue
    rel = pd.DataFrame(rel).T
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
    w(f"  pre-period RMSPE {100 * rmspe_pre:.2f} log points; mean gap May-December 2023 {fmt(gap[post].mean())}; "
      f"January-June 2024 {fmt(gap[[c for c in gap.index if '2024-01' <= c <= '2024-06']].mean())}; "
      f"latest ({gap.index[-1]}) {fmt(gap.iloc[-1])}")
    ratios = {}
    for geo in rel.index:
        vv, gg = synthetic_control(rel, geo, pre_months)
        pre_r = float(np.sqrt((gg[pre_months] ** 2).mean()))
        post_r = float(np.sqrt((gg[post] ** 2).mean()))
        ratios[geo] = post_r / pre_r if pre_r > 0 else np.nan
    ratios = pd.Series(ratios).sort_values(ascending=False)
    ratios.round(4).to_csv(RESULTS / "synthetic_rmspe_ratios.csv", header=["post_over_pre_rmspe"])
    w(f"  post/pre RMSPE ratio: Portugal {ratios['PT']:.1f}, rank {int((ratios >= ratios['PT']).sum())} of {len(ratios)}")
    w("")

    # 6-7. Reversal and persistence.
    ex_w = within_country(lp_w, "PT", treated, control, EXIT_BASE)
    ex_a = across_countries(lp_w, "PT", comp, treated, control, EXIT_BASE)
    tax_ex = group_gap(rebased(wedge.xs("PT", level="geo"), EXIT_BASE), treated, control)["gap"]
    ex = pd.DataFrame({"within": ex_w["gap"], "across": ex_a["gap"], "tax_wedge": tax_ex})
    ex.loc[[c for c in ex.index if c >= "2023-06"]].round(6).to_csv(RESULTS / "event_study_reversal.csv")
    w("Reversal: treated minus untreated, log change since October 2023 (x100)")
    for m in [c for c in ex.index if "2023-09" <= c <= "2024-06"]:
        r = ex.loc[m]
        w(f"  {m}: within {fmt(r['within'])}, across {fmt(r['across'])}, tax wedge {fmt(r['tax_wedge'])}")
    for name in ("within", "across"):
        jan_mar = ex.loc[["2024-01", "2024-02", "2024-03"], name].mean()
        w(f"  January-March 2024 mean, {name}: {fmt(jan_mar)}; pass-through of the reversal "
          f"{100 * jan_mar / ex.loc[['2024-01', '2024-02', '2024-03'], 'tax_wedge'].mean():.0f}%")
    later = [c for c in es.index if c >= "2024-01"]
    w("Persistence: gap relative to February 2023 after the reversal (0 = back on the comparison path)")
    for yr in ("2024", "2025"):
        ms = [c for c in later if c.startswith(yr)]
        if ms:
            w(f"  {yr} ({ms[0]} to {ms[-1]}): within {fmt(es.loc[ms, 'within'].mean())}, "
              f"across {fmt(es.loc[ms, 'across'].mean())}")
    w("")

    # Short windows, dose-response, robustness, oils.
    extensions(w, lp_w, wedge, t, comp, w23)

    # 8. Headline inflation and incidence.
    theta = es.loc[POLICY, "across"].mean() / es.loc[POLICY, "tax_wedge"].mean()
    incidence(w, t, w23, wedge, theta)

    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
