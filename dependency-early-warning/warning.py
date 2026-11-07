"""Forecast which products will enter the single-supplier screen, and how early.

    python warning.py

Reads ``results/panel.csv.gz`` (``panel.py``) and the partner parts in ``data/parts``.

For every product and origin year t that is not flagged in t-2, t-1 and t, the target is a
durable entry within the horizon: a first pair of consecutive flagged years (E, E+1) with
t < E <= t+h. Features use data up to t only. Four scores are compared:

- ``near_miss``: a rule with no fitting. Products meeting two of the three criteria come
  first, ranked by how far the missing criterion is from its threshold; then the others.
- ``snapshot``: logistic regression on the year-t values only (levels).
- ``trajectory``: logistic regression on levels and on one- and three-year changes, the
  fastest-rising outside supplier, relative unit values and past flags.
- ``boosting``: gradient-boosted trees on the same features as ``trajectory``.
- ``expected_value``: the boosting probability times the largest outside supplier's imports
  in year t, for a list ranked by value at stake rather than by probability.

Evaluation is out of time and rolling: the model scored at origin T is fitted on origins
t <= T-h-1 only, whose outcomes were known by T. Writes ``results/warning_*.csv`` and
``results/warning_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PARTS = HERE / "data" / "parts"
LABELS = HERE.parent / "europe-deindustrialisation" / "data" / "hs6_labels.csv"
H = 3
LOOKBACK = 3
FIRST_TEST = 2010
WATCHLISTS = (100, 200, 400)
K = 200
SEED = 20261107
BOOT = 300
MIN_ENTRIES = 20
MATERIAL = 1e7
MODELS = ("near_miss", "snapshot", "trajectory", "boosting", "expected_value")

LEVELS = ["top_share", "second_share", "hhi", "extra_share", "log_ratio_c", "log_value", "log_suppliers", "cn_share", "n_criteria", "c_top", "c_extra", "c_ratio"]
CHANGES = [
    "d1_top_share", "d3_top_share", "d3_extra_share", "d1_log_ratio", "d3_log_ratio", "d3_log_value", "d3_log_intra", "d3_log_exports", "d3_cn_share",
    "challenger_gain", "challenger_share", "challenger_is_top", "rel_unit_value", "d3_rel_unit_value", "past_flags", "years_since_flag", "chapter_rate",
]
FEATURES = LEVELS + CHANGES


def challengers() -> pd.DataFrame:
    """For each product and year t: the outside supplier whose share rose most since t-3."""
    files = sorted(PARTS.glob("partners_*.csv.gz"))
    p = pd.concat([pd.read_csv(f, dtype={"product": str, "partner": str}, keep_default_na=False, na_values=[""]) for f in files], ignore_index=True).drop_duplicates()
    p["share"] = p["value"] / p.groupby(["product", "year"])["value"].transform("sum")
    p = p[p["partner"] != "ZZ"][["product", "year", "partner", "share"]]
    then = p.assign(year=p["year"] + LOOKBACK).rename(columns={"share": "share_then"})
    m = p.merge(then, on=["product", "year", "partner"], how="left")
    # a partner absent from the top ten three years earlier had less than the tenth supplier's share; 0 is used
    m["gain"] = m["share"] - m["share_then"].fillna(0.0)
    top = m.loc[m.groupby(["product", "year"])["share"].idxmax(), ["product", "year", "partner"]].rename(columns={"partner": "top"})
    best = m.loc[m.groupby(["product", "year"])["gain"].idxmax(), ["product", "year", "partner", "gain", "share"]]
    best = best.merge(top, on=["product", "year"], how="left")
    best["challenger_is_top"] = (best["partner"] == best["top"]).astype(float)
    best = best.rename(columns={"gain": "challenger_gain", "share": "challenger_share", "partner": "challenger"})
    return best[["product", "year", "challenger", "challenger_gain", "challenger_share", "challenger_is_top"]]


def features(d: pd.DataFrame, chall: pd.DataFrame | None = None) -> pd.DataFrame:
    d = d.sort_values(["product", "year"]).copy()
    d["log_ratio_c"] = d["log_ratio"].clip(-5, 5)
    d["log_value"] = np.log(d["imports_extra"])
    d["log_suppliers"] = np.log(d["suppliers"].where(d["suppliers"] > 0))
    d["log_intra"] = np.log1p(d["imports_intra"])
    d["log_exports"] = np.log1p(d["exports_extra"] + d["exports_intra"])
    d["n_criteria"] = d[["c_top", "c_extra", "c_ratio"]].sum(axis=1)
    g = d.groupby("product")
    for c in ("top_share",):
        d[f"d1_{c}"] = d[c] - g[c].shift(1)
    d["d1_log_ratio"] = d["log_ratio_c"] - g["log_ratio_c"].shift(1)
    for c, name in (("top_share", "top_share"), ("extra_share", "extra_share"), ("log_ratio_c", "log_ratio"), ("log_value", "log_value"), ("log_intra", "log_intra"), ("log_exports", "log_exports"), ("cn_share", "cn_share"), ("rel_unit_value", "rel_unit_value")):
        d[f"d3_{name}"] = d[c] - g[c].shift(LOOKBACK)
    # flag history up to t
    f = d["flagged"].astype(int)
    d["past_flags"] = g["flagged"].transform(lambda s: s.astype(int).cumsum()) - f
    last = d["year"].where(d["flagged"])
    d["last_flag"] = last.groupby(d["product"]).ffill()
    d["years_since_flag"] = (d["year"] - d["last_flag"]).fillna(99).clip(upper=25)
    d = d.merge(challengers() if chall is None else chall, on=["product", "year"], how="left")
    # eligibility and target
    g = d.groupby("product")
    clean = (~d["flagged"]) & (g["flagged"].shift(1) == False) & (g["flagged"].shift(2) == False)  # noqa: E712
    has_history = g["year"].shift(LOOKBACK) == d["year"] - LOOKBACK
    fl = {k: g["flagged"].shift(-k) for k in range(1, max(H, 5) + 2)}
    for h in (H, 5):
        hit = pd.Series(False, index=d.index)
        observed = fl[h + 1].notna()
        for e in range(1, h + 1):
            hit |= (fl[e] == True) & (fl[e + 1] == True)  # noqa: E712
        d[f"y{h}"] = np.where(observed, hit.astype(float), np.nan)
        # first entry year within the horizon, for lead times
        first = pd.Series(np.nan, index=d.index)
        for e in range(h, 0, -1):
            first = first.mask((fl[e] == True) & (fl[e + 1] == True), d["year"] + e)  # noqa: E712
        d[f"entry{h}"] = first
    d["eligible"] = clean & has_history
    d["chapter"] = d["product"].str[:2]
    return d


def chapter_rates(train: pd.DataFrame, y: str, prior: float = 20.0) -> pd.Series:
    base = train[y].mean()
    s = train.groupby("chapter")[y].agg(["sum", "count"])
    return (s["sum"] + prior * base) / (s["count"] + prior)


def near_miss(x: pd.DataFrame) -> np.ndarray:
    """Two of three criteria met first, then by the relative shortfall on the missing ones."""
    short = (
        np.clip((0.4 - x["top_share"]) / 0.4, 0, None).fillna(1)
        + np.clip((0.5 - x["extra_share"]) / 0.5, 0, None).fillna(1)
        + np.clip(-x["log_ratio_c"] / 5, 0, None).fillna(1)
    )
    return (x["n_criteria"].to_numpy() * 10 - short.to_numpy()).astype(float)


def logistic():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))


def boosting():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=0)


def fit_score(train: pd.DataFrame, test: pd.DataFrame, y: str) -> dict[str, np.ndarray]:
    rates = chapter_rates(train, y)
    base = train[y].mean()
    tr = train.assign(chapter_rate=train["chapter"].map(rates).fillna(base))
    te = test.assign(chapter_rate=test["chapter"].map(rates).fillna(base))
    out = {"near_miss": near_miss(te)}
    lv = [c for c in LEVELS]
    m = logistic().fit(tr[lv].astype(float), tr[y])
    out["snapshot"] = m.predict_proba(te[lv].astype(float))[:, 1]
    m = logistic().fit(tr[FEATURES].astype(float), tr[y])
    out["trajectory"] = m.predict_proba(te[FEATURES].astype(float))[:, 1]
    m = boosting().fit(tr[FEATURES].astype(float), tr[y])
    out["boosting"] = m.predict_proba(te[FEATURES].astype(float))[:, 1]
    # probability times what is at stake now: the largest outside supplier's imports in year t
    out["expected_value"] = out["boosting"] * te["top_value"].fillna(0.0).to_numpy()
    return out


def rolling(d: pd.DataFrame, h: int) -> pd.DataFrame:
    y = f"y{h}"
    e = d[d["eligible"]]
    preds = []
    last_origin = int(e.loc[e[y].notna(), "year"].max())
    top_value = d.set_index(["product", "year"])["top_value"]
    for T in range(FIRST_TEST, int(e["year"].max()) + 1):
        train = e[(e["year"] <= T - h - 1) & e[y].notna()]
        test = e[e["year"] == T]
        if train[y].sum() < MIN_ENTRIES or not len(test):
            continue
        s = fit_score(train, test, y)
        out = test[["product", "year", y, f"entry{h}", "imports_extra", "top_value", "top_partner", "challenger"]].rename(columns={y: "y", f"entry{h}": "entry"})
        for k, v in s.items():
            out[k] = v
        out["scored_with_outcome"] = T <= last_origin
        out["entry_top_value"] = [top_value.get((p_, int(y_)), np.nan) if y_ == y_ else np.nan for p_, y_ in zip(out["product"], out["entry"])]
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def ranks(p: pd.DataFrame) -> pd.DataFrame:
    for m in MODELS:
        p[f"rank_{m}"] = p.groupby("year")[m].rank(ascending=False, method="first")
    return p


def metrics(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q = p[p["y"].notna()]
    for (T, m), g in [((T, m), g) for T, g in q.groupby("year") for m in MODELS]:
        if g["y"].sum() == 0:
            continue
        r = {"origin": T, "model": m, "n": len(g), "entries": int(g["y"].sum()), "auc": roc_auc_score(g["y"], g[m]), "ap": average_precision_score(g["y"], g[m])}
        for k in WATCHLISTS:
            top = g[f"rank_{m}"] <= k
            r[f"recall_{k}"] = g.loc[top, "y"].sum() / g["y"].sum()
            r[f"precision_{k}"] = g.loc[top, "y"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def pooled(p: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    q = p[p["y"].notna()].reset_index(drop=True)
    prods = q["product"].unique()
    idx = {k: v for k, v in q.groupby("product").indices.items()}

    def stats(g: pd.DataFrame) -> dict[str, float]:
        out = {}
        for m in MODELS:
            out[f"auc_{m}"] = roc_auc_score(g["y"], g[m])
            out[f"ap_{m}"] = average_precision_score(g["y"], g[m])
            top = g[f"rank_{m}"] <= K
            out[f"recall{K}_{m}"] = g.loc[top, "y"].sum() / g["y"].sum()
        return out

    point = stats(q)
    boots = []
    for _ in range(BOOT):
        pick = rng.choice(prods, size=len(prods), replace=True)
        rows = np.concatenate([idx[p_] for p_ in pick])
        boots.append(stats(q.iloc[rows]))
    b = pd.DataFrame(boots)
    rows = []
    for c, v in point.items():
        rows.append({"stat": c, "value": v, "lo": b[c].quantile(0.025), "hi": b[c].quantile(0.975)})
    for a, z in (("trajectory", "snapshot"), ("boosting", "snapshot"), ("snapshot", "near_miss"), ("boosting", "trajectory")):
        for s in ("auc", "ap", f"recall{K}"):
            diff = b[f"{s}_{a}"] - b[f"{s}_{z}"]
            rows.append({"stat": f"{s}_{a}_minus_{z}", "value": point[f"{s}_{a}"] - point[f"{s}_{z}"], "lo": diff.quantile(0.025), "hi": diff.quantile(0.975)})
    return pd.DataFrame(rows)


def leads(p: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """For each durable entry: the earliest of the three origins before it at which a score had it on the watchlist."""
    rows = []
    by = p.set_index(["product", "year"]).sort_index()
    first_test, last_test = int(p["year"].min()), int(p["year"].max())
    for r in events.itertuples(index=False):
        if r.entry - H < first_test or r.entry - 1 > last_test:
            continue
        rec = {"product": r.product, "entry": r.entry, "supplier": r.top_partner, "top_value": r.top_value}
        for m in MODELS:
            lead = 0
            for k in range(H, 0, -1):
                key = (r.product, r.entry - k)
                if key in by.index and by.loc[key, f"rank_{m}"] <= K:
                    lead = k
                    break
            rec[f"lead_{m}"] = lead
        rec["scored_origins"] = sum((r.product, r.entry - k) in by.index for k in range(1, H + 1))
        rows.append(rec)
    return pd.DataFrame(rows)


def importance(d: pd.DataFrame, h: int) -> pd.DataFrame:
    """Standardised coefficients of the trajectory model fitted on every origin with a known outcome."""
    y = f"y{h}"
    e = d[d["eligible"] & d[y].notna()]
    e = e.assign(chapter_rate=e["chapter"].map(chapter_rates(e, y)))
    m = logistic().fit(e[FEATURES].astype(float), e[y])
    lr = m[-1]
    names = list(m[0].get_feature_names_out(FEATURES))
    coef = pd.DataFrame({"feature": names, "coef": lr.coef_[0]})
    coef = coef[~coef["feature"].str.startswith("missingindicator")]
    # ablation: pooled out-of-time AUC with feature groups removed is in warning_ablation.csv
    return coef.sort_values("coef", key=np.abs, ascending=False)


def ablation(d: pd.DataFrame, h: int) -> pd.DataFrame:
    groups = {
        "levels only": LEVELS,
        "+ own changes": LEVELS + ["d1_top_share", "d3_top_share", "d3_extra_share", "d1_log_ratio", "d3_log_ratio", "d3_log_value", "d3_log_intra", "d3_log_exports"],
        "+ challenger and China": LEVELS + ["d1_top_share", "d3_top_share", "d3_extra_share", "d1_log_ratio", "d3_log_ratio", "d3_log_value", "d3_log_intra", "d3_log_exports", "d3_cn_share", "challenger_gain", "challenger_share", "challenger_is_top"],
        "+ unit values": LEVELS + ["d1_top_share", "d3_top_share", "d3_extra_share", "d1_log_ratio", "d3_log_ratio", "d3_log_value", "d3_log_intra", "d3_log_exports", "d3_cn_share", "challenger_gain", "challenger_share", "challenger_is_top", "rel_unit_value", "d3_rel_unit_value"],
        "+ flag history and chapter (all)": FEATURES,
    }
    y = f"y{h}"
    e = d[d["eligible"]]
    rows = []
    for name, cols in groups.items():
        ys, ss = [], []
        for T in range(FIRST_TEST, int(e.loc[e[y].notna(), "year"].max()) + 1):
            train = e[(e["year"] <= T - h - 1) & e[y].notna()]
            test = e[(e["year"] == T) & e[y].notna()]
            if train[y].sum() < MIN_ENTRIES or test[y].sum() == 0:
                continue
            rates = chapter_rates(train, y)
            tr = train.assign(chapter_rate=train["chapter"].map(rates).fillna(train[y].mean()))
            te = test.assign(chapter_rate=test["chapter"].map(rates).fillna(train[y].mean()))
            m = logistic().fit(tr[cols].astype(float), tr[y])
            ys.append(te[y].to_numpy())
            ss.append(m.predict_proba(te[cols].astype(float))[:, 1])
        yy, sc = np.concatenate(ys), np.concatenate(ss)
        rows.append({"features": name, "n_features": len(cols), "auc": roc_auc_score(yy, sc), "ap": average_precision_score(yy, sc)})
    return pd.DataFrame(rows)


GROUPS = {
    "Largest supplier's share and concentration": ["top_share", "second_share", "hhi", "log_suppliers", "c_top"],
    "Extra-EU share of imports": ["extra_share", "c_extra"],
    "Imports relative to EU exports": ["log_ratio_c", "c_ratio"],
    "Criteria met": ["n_criteria"],
    "Size and growth of imports": ["log_value", "d3_log_value"],
    "Changes in the three criteria": ["d1_top_share", "d3_top_share", "d3_extra_share", "d1_log_ratio", "d3_log_ratio", "d3_log_intra", "d3_log_exports"],
    "China's share and the fastest riser": ["cn_share", "d3_cn_share", "challenger_gain", "challenger_share", "challenger_is_top"],
    "Relative unit values": ["rel_unit_value", "d3_rel_unit_value"],
    "Flag history": ["past_flags", "years_since_flag"],
    "Chapter base rate": ["chapter_rate"],
}
PERM_TEST = (2020, 2021)
PERM_REPEATS = 20


def permutation(d: pd.DataFrame, h: int, rng: np.random.Generator) -> pd.DataFrame:
    """Drop in out-of-time average precision and AUC when a group of features is shuffled (boosting)."""
    y = f"y{h}"
    e = d[d["eligible"]]
    train = e[(e["year"] <= min(PERM_TEST) - h - 1) & e[y].notna()]
    test = e[e["year"].isin(PERM_TEST) & e[y].notna()]
    rates = chapter_rates(train, y)
    tr = train.assign(chapter_rate=train["chapter"].map(rates).fillna(train[y].mean()))
    te = test.assign(chapter_rate=test["chapter"].map(rates).fillna(train[y].mean()))
    m = boosting().fit(tr[FEATURES].astype(float), tr[y])
    x = te[FEATURES].astype(float).reset_index(drop=True)
    yy = te[y].to_numpy()
    base = m.predict_proba(x)[:, 1]
    ap0, auc0 = average_precision_score(yy, base), roc_auc_score(yy, base)
    rows = [{"group": "none", "ap": ap0, "auc": auc0, "ap_drop": 0.0, "auc_drop": 0.0}]
    for g, cols in GROUPS.items():
        aps, aucs = [], []
        for _ in range(PERM_REPEATS):
            xp = x.copy()
            perm = rng.permutation(len(xp))
            xp[cols] = xp[cols].to_numpy()[perm]  # shuffle the group jointly, keeping its internal structure
            s = m.predict_proba(xp)[:, 1]
            aps.append(average_precision_score(yy, s))
            aucs.append(roc_auc_score(yy, s))
        rows.append({"group": g, "ap": np.mean(aps), "auc": np.mean(aucs), "ap_drop": ap0 - np.mean(aps), "auc_drop": auc0 - np.mean(aucs)})
    out = pd.DataFrame(rows)
    out.attrs["n_test"], out.attrs["entries"], out.attrs["train_max"] = len(te), int(yy.sum()), int(train["year"].max())
    return out


def misses(p: pd.DataFrame, d: pd.DataFrame) -> dict[str, float]:
    """What became of watchlist products that did not enter within the horizon."""
    q = p[(p["y"] == 0) & (p["rank_boosting"] <= K)]
    later = d.set_index(["product", "year"])
    out = {"n": len(q)}
    top_then = [later["top_share"].get((r.product, r.year + H), np.nan) for r in q.itertuples(index=False)]
    any_flag = []
    for r in q.itertuples(index=False):
        any_flag.append(any(bool(later["flagged"].get((r.product, r.year + k), False)) for k in range(1, H + 2)))
    top_then = np.array(top_then, dtype=float)
    out["top_share_above_40_at_h"] = float(np.nanmean(top_then > 0.4))
    out["top_share_above_60_at_h"] = float(np.nanmean(top_then > 0.6))
    out["flagged_some_year"] = float(np.mean(any_flag))
    return out


def calibration(p: pd.DataFrame, m: str = "trajectory") -> pd.DataFrame:
    q = p[p["y"].notna()].copy()
    q["bin"] = pd.qcut(q[m].rank(method="first"), 10, labels=False)
    return q.groupby("bin").agg(predicted=(m, "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def label_map() -> dict[str, str]:
    if not LABELS.exists():
        return {}
    lab = pd.read_csv(LABELS, dtype=str, keep_default_na=False)
    return dict(zip(lab["product"], lab["label"]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(RESULTS / "panel.csv.gz", dtype={"product": str, "top_partner": str, "second_partner": str}, keep_default_na=False, na_values=[""])
    for c in ("flagged", "c_top", "c_extra", "c_ratio"):
        d[c] = d[c].astype(str).str.lower().eq("true")
    events = pd.read_csv(RESULTS / "panel_events.csv", dtype={"product": str, "supplier": str}, keep_default_na=False, na_values=[""]).rename(columns={"supplier": "top_partner"})
    ev_values = d.set_index(["product", "year"])["top_value"]
    events["top_value"] = [ev_values.get((p_, y_), np.nan) for p_, y_ in zip(events["product"], events["entry"])]
    f = features(d)
    labels = label_map()

    p = ranks(rolling(f, H))
    p.to_csv(RESULTS / "warning_predictions.csv.gz", index=False, float_format="%.6g", compression={"method": "gzip", "mtime": 0})
    met = metrics(p)
    met.to_csv(RESULTS / "warning_metrics.csv", index=False, float_format="%.4f")
    pool = pooled(p, rng)
    pool.to_csv(RESULTS / "warning_pooled.csv", index=False, float_format="%.4f")
    ld = leads(p, events)
    ld.to_csv(RESULTS / "warning_leads.csv", index=False, float_format="%.6g")
    imp = importance(f, H)
    imp.to_csv(RESULTS / "warning_coefficients.csv", index=False, float_format="%.4f")
    abl = ablation(f, H)
    abl.to_csv(RESULTS / "warning_ablation.csv", index=False, float_format="%.4f")
    cal = calibration(p)
    cal.to_csv(RESULTS / "warning_calibration.csv", index=False, float_format="%.4f")
    perm = permutation(f, H, rng)
    perm.to_csv(RESULTS / "warning_permutation.csv", index=False, float_format="%.4f")
    miss = misses(p, d)
    mat = p[p["y"].notna() & (p["imports_extra"] >= MATERIAL)].copy()
    mat_rows = []
    for m_ in MODELS:
        mat_rows.append(f"  {m_}: AUC {roc_auc_score(mat['y'], mat[m_]):.3f}, AP {average_precision_score(mat['y'], mat[m_]):.3f}")
    p5 = ranks(rolling(f, 5))
    met5 = metrics(p5)
    met5.to_csv(RESULTS / "warning_metrics_h5.csv", index=False, float_format="%.4f")

    e = f[f["eligible"]]
    lines = [
        f"Horizon {H} years, features from t-{LOOKBACK} to t, watchlist {K} products, rolling origins {int(p['year'].min())}-{int(p['year'].max())}",
        f"Eligible product-origins (unflagged in t-2..t, history from t-{LOOKBACK}): {len(e)}; with known outcome: {int(e[f'y{H}'].notna().sum())}; entry rate {e[f'y{H}'].mean():.4f}",
        f"Scored out of time with known outcome: {int(p['y'].notna().sum())} product-origins, {int(p['y'].sum())} entries within {H} years",
        "",
        "Pooled out-of-time performance (product-cluster bootstrap 95% interval):",
    ]
    for r in pool.itertuples(index=False):
        lines.append(f"  {r.stat}: {r.value:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
    lines += ["", "By origin (AUC, average precision, recall in the top 200):"]
    for T, g in met.groupby("origin"):
        g = g.set_index("model")
        lines.append(f"  {T}: entries {int(g['entries'].iloc[0])} of {int(g['n'].iloc[0])}; " + "; ".join(f"{m} {g.loc[m, 'auc']:.3f}/{g.loc[m, 'ap']:.3f}/{g.loc[m, f'recall_{K}']:.2f}" for m in MODELS))
    lines += ["", "Mean over origins of recall and precision by watchlist size:"]
    for m in MODELS:
        g = met[met["model"] == m]
        lines.append(f"  {m}: " + "; ".join(f"top {k} recall {g[f'recall_{k}'].mean():.3f} precision {g[f'precision_{k}'].mean():.3f}" for k in WATCHLISTS))
    q = p[p["y"].notna()]
    vw = []
    for m_ in MODELS:
        r = []
        for _, g in q.groupby("year"):
            w = g["entry_top_value"].fillna(0)
            pos = g["y"] == 1
            r.append(w[pos & (g[f"rank_{m_}"] <= K)].sum() / w[pos].sum())
        vw.append(f"{m_} {np.mean(r):.3f}")
    lines += ["", f"Value-weighted recall in the top {K} (entering supplier's imports in the entry year), mean over origins: " + "; ".join(vw)]
    lines.append(f"Average number of products ranked per origin: {q.groupby('year').size().mean():.0f}; entries per origin {q.groupby('year')['y'].sum().mean():.1f}")
    lines += ["", f"Horizon 5 years, mean over origins (AUC, AP, recall top {K}):"]
    for m in MODELS:
        g = met5[met5["model"] == m]
        lines.append(f"  {m}: {g['auc'].mean():.3f}, {g['ap'].mean():.3f}, {g[f'recall_{K}'].mean():.3f} ({len(g)} origins)")
    lines += ["", f"Lead time before durable entries {int(ld['entry'].min())}-{int(ld['entry'].max())} ({len(ld)} entries; on the top-{K} list at t = E-3, E-2 or E-1):"]
    for m in MODELS:
        c = ld[f"lead_{m}"]
        w = ld["top_value"].fillna(0)
        lines.append(f"  {m}: warned {(c > 0).mean():.3f} (value-weighted {(w * (c > 0)).sum() / w.sum():.3f}); 3 years ahead {(c == 3).mean():.3f}; 2+ years {(c >= 2).mean():.3f}; mean lead {c.mean():.2f}")
    warned = {m_: (ld[f"lead_{m_}"] > 0).to_numpy() for m_ in MODELS}
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(ld), len(ld))
        bs.append({m_: warned[m_][i].mean() for m_ in MODELS})
    bs = pd.DataFrame(bs)
    for a_, z_ in (("boosting", "near_miss"), ("boosting", "snapshot"), ("snapshot", "near_miss"), ("trajectory", "snapshot")):
        dd = bs[a_] - bs[z_]
        lines.append(f"  warned share {a_} minus {z_}: {warned[a_].mean() - warned[z_].mean():+.3f} [{dd.quantile(0.025):+.3f}, {dd.quantile(0.975):+.3f}]")
    big = ld[ld["top_value"] >= MATERIAL]
    lines.append(f"  entries with the entering supplier's imports >= EUR {MATERIAL / 1e6:.0f} m at entry ({len(big)}): " + "; ".join(f"{m_} warned {(big[f'lead_{m_}'] > 0).mean():.3f}, 3 years ahead {(big[f'lead_{m_}'] == 3).mean():.3f}" for m_ in MODELS))
    lines += ["", "Feature groups, trajectory logistic, pooled out-of-time AUC / AP:"]
    for r in abl.itertuples(index=False):
        lines.append(f"  {r.features} ({r.n_features}): {r.auc:.3f} / {r.ap:.3f}")
    lines += ["", "Largest standardised coefficients, trajectory model on all known outcomes:"]
    for r in imp.head(14).itertuples(index=False):
        lines.append(f"  {r.feature}: {r.coef:+.3f}")
    lines += ["", f"Grouped permutation importance, boosting, fitted on origins <= {perm.attrs['train_max']}, tested on {PERM_TEST[0]}-{PERM_TEST[1]} ({perm.attrs['n_test']} product-origins, {perm.attrs['entries']} entries), {PERM_REPEATS} shuffles:"]
    for r in perm.itertuples(index=False):
        lines.append(f"  {r.group}: AP {r.ap:.3f} (drop {r.ap_drop:+.3f}), AUC {r.auc:.3f} (drop {r.auc_drop:+.3f})")
    lines += ["", f"Watchlist products (boosting, top {K}) that did not enter within {H} years: {miss['n']} product-origins; largest supplier above 40% {H} years later {miss['top_share_above_40_at_h']:.3f}, above 60% {miss['top_share_above_60_at_h']:.3f}; flagged in at least one year t+1..t+{H + 1} {miss['flagged_some_year']:.3f}"]
    lines += ["", f"Products with extra-EU imports >= EUR {MATERIAL / 1e6:.0f} m at the origin ({len(mat)} product-origins, {int(mat['y'].sum())} entries), same scores:"] + mat_rows
    lines += ["", "Calibration, trajectory model (deciles of the score, predicted vs observed):"]
    for r in cal.itertuples(index=False):
        lines.append(f"  {int(r.bin) + 1}: {r.predicted:.4f} vs {r.observed:.4f} (n {r.n})")
    text = "\n".join(lines) + "\n"
    (RESULTS / "warning_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    # watchlists go to their own file: a near-tie can swap two products across machines
    lines = []
    for T in (2015, 2019, 2021):
        g = p[(p["year"] == T) & (p["imports_extra"] >= MATERIAL)].sort_values("rank_boosting").head(15)
        if not len(g):
            continue
        lines += ["", f"Top 15 of the {T} watchlist (boosting) among products with extra-EU imports >= EUR {MATERIAL / 1e6:.0f} m, outcome by {T + H + 1}:"]
        for r in g.itertuples(index=False):
            out = "entered " + str(int(r.entry)) if r.y == 1 else ("no entry" if r.y == 0 else "outcome not yet known")
            lines.append(f"  {int(r.rank_boosting)} {r.product} {labels.get(r.product, '')[:60]} | largest {r.top_partner}, rising {r.challenger} | EUR {r.imports_extra / 1e6:.0f} m | {out}")
    g = p[(p["year"] == p["year"].max()) & (p["imports_extra"] >= MATERIAL)].sort_values("rank_boosting").head(25)
    lines += ["", f"Top 25 of the {int(p['year'].max())} watchlist (boosting) among products with extra-EU imports >= EUR {MATERIAL / 1e6:.0f} m:"]
    for r in g.itertuples(index=False):
        lines.append(f"  {int(r.rank_boosting)} {r.product} {labels.get(r.product, '')[:60]} | largest {r.top_partner}, rising {r.challenger} | EUR {r.imports_extra / 1e6:.0f} m, p {r.boosting:.2f}")
    text = "\n".join(lines).lstrip("\n") + "\n"
    (RESULTS / "warning_lists.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
