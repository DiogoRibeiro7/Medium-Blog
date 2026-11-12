"""Part 6: does a fortune's own past volatility predict whether it leaves the Forbes list?

Part 5 conjectured that volatility is a stable property of individual fortunes,
and that a fortune's past volatility predicts its departure from the list
better than its distance from $1 billion. This script tests both claims on the
linked Forbes lists 2010-2025, using counts by cell from
``fetch_forbes_persistence.py`` (no person-level record is stored):

* persistence: among fortunes of $3 billion or more, which almost never fall
  off the list, does the volatility of the earlier changes (c2..c4, excluding
  the latest change c1) predict the size of the next change, within the same
  year and size band, and does it still do so once c1 is known?
* departures: below $3 billion, logistic models of leaving the list on
  distance from the line alone, on past volatility alone and on both, compared
  out of sample by leaving one pair of lists out at a time;
* one index: distance from the line divided by the fortune's own volatility
  (the logic of the distance to default in credit risk), with a test of whether
  distance and volatility enter only through that ratio;
* composition: whether the extra volatility Part 5 found near the line comes
  from volatile fortunes sitting there;
* sector: whether past volatility still predicts departures within Forbes's
  industry categories.

Every number quoted in the Part 6 article is computed here and written to
``results/part6/numbers.json``; ``--check`` recomputes everything and fails if
any number differs. Figures go to ``figures/part6/``.

Usage::

    python part6_past_volatility.py           # print, write results/part6 and figures/part6
    python part6_past_volatility.py --check   # fail if any number differs from numbers.json

All results are descriptive, not causal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from scipy.stats import chi2, t as student_t  # noqa: E402

from part1_poverty_lines import (AQUA, BASELINE, BLUE, INK, MUTED, ORANGE, SECONDARY,  # noqa: E402
                                 _compare, _frame, _style, fmt)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results" / "part6"
FIGURES = HERE / "figures" / "part6"

FIRST, LAST = 2010, 2024             # pairs of lists t -> t+1
VOL_YEARS = (2012, 2024)             # a volatility class needs two earlier changes' worth of lists before t
FULL_YEARS = (2014, 2024)            # a full window of three earlier changes
FAR = 3000
NEAR = 1500
NEAR_BANDS = [1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 2500]
BAND_MID = {1000: 1050, 1100: 1150, 1200: 1250, 1300: 1350, 1400: 1450, 1500: 1625, 1750: 1875, 2000: 2250,
            2500: 2750}
VOL_GROUPS = (("low", "below 0.06", lambda v: v < 60), ("mid", "0.06 to 0.13", lambda v: (v >= 60) & (v < 130)),
              ("high", "0.13 or more", lambda v: v >= 130))
S0 = 0.02                            # rounding noise added to measured volatility in the one-index model
RIDGE_CV = 1.0                       # weak prior (sd 1 on the logit scale) for held-out predictions only
N_BOOT = 400
INDUSTRY_MAP = {  # older labels onto Forbes's current 18 categories
    "Fashion and Retail": "Fashion & Retail", "Retail": "Fashion & Retail", "Retail & Ecommerce": "Fashion & Retail",
    "Service & Retail": "Fashion & Retail", "Finance": "Finance & Investments",
    "Finance and Investments": "Finance & Investments", "Investments": "Finance & Investments",
    "Food and Beverage": "Food & Beverage", "Food & Manufacturing": "Food & Beverage", "Gaming": "Gambling & Casinos",
    "Games": "Gambling & Casinos", "Health care": "Healthcare", "Medicine": "Healthcare", "Media": "Media & Entertainment",
    "Oil": "Energy", "Technology & Medicine": "Technology", "Technology / Business": "Technology",
    "All Star Alumni": "Other", "Billionaire": "Other", "Business": "Other", "Celebrity": "Other", "Politics": "Other",
    "Social Entrepreneurs": "Other", "unknown": "Other"}
SOURCE = ("Data: Forbes annual billionaire lists 2010-2025, linked year to year; only aggregate counts are stored. "
          "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/billionaires-poverty-followups")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype={"key": str, "prev": str, "tenure": str, "age": str,
                                                           "trend": str})


def load_inputs(data: Path = DATA) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for man, name, key in (("forbes_persistence_manifest.json", "forbes_persistence.csv.gz", "persistence_sha256"),
                           ("forbes_tenure_manifest.json", "forbes_tenure.csv.gz", "tenure_sha256")):
        m = json.loads((data / man).read_text(encoding="utf-8"))
        if _sha(data / name) != m[key]:
            raise ValueError(f"{name} does not match {man}")
        out[name.split(".")[0]] = _read(data / name)
        out[man.split(".")[0]] = m
    return out


def check_against_part5(pers: pd.DataFrame, tenure: pd.DataFrame) -> dict[str, Any]:
    """Summed over the new dimensions, the tables must equal Part 5's tenure tables exactly."""
    out = {}
    keys = ["year", "band_lo_musd", "tenure", "prev", "key"]
    for table in ("stock", "exit", "logchange"):
        a = pers[(pers.table == table) & pers.year.between(FIRST, LAST)].groupby(keys).n.sum()
        b = tenure[(tenure.table == table) & tenure.year.between(FIRST, LAST)].groupby(keys).n.sum()
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0)
        bad = int((j.a != j.b).sum())
        if bad:
            raise ValueError(f"persistence table {table} disagrees with Part 5 in {bad} cells")
        out[table] = {"cells": int(len(j)), "mismatches": bad, "people": int(j.a.sum())}
    return out


def _years(lo_hi: tuple[int, int]) -> list[int]:
    return list(range(lo_hi[0], lo_hi[1] + 1))


# ---------------------------------------------------------------------------
# Grouped logistic regression (cells with n people and k departures)
# ---------------------------------------------------------------------------

def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def fit_logit(X: np.ndarray, k: np.ndarray, n: np.ndarray, ridge: float = 1e-6) -> dict[str, Any]:
    """Newton-Raphson with step halving for a binomial logit on grouped data; a ridge on every coefficient but the
    intercept keeps empty categories finite."""
    p_ = X.shape[1]
    pen = np.full(p_, ridge)
    pen[0] = 0.0
    beta = np.zeros(p_)
    beta[0] = math.log(max(k.sum(), 0.5) / max(n.sum() - k.sum(), 0.5))

    def objective(b: np.ndarray) -> float:
        return loglik(_expit(X @ b), k, n) - 0.5 * float(np.sum(pen * b * b))

    current = objective(beta)
    for _ in range(200):
        p = _expit(X @ beta)
        w = n * p * (1 - p)
        g = X.T @ (k - n * p) - pen * beta
        H = X.T @ (X * w[:, None]) + np.diag(pen) + 1e-12 * np.eye(p_)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-6:
            trial = objective(beta + t * step)
            if trial >= current - 1e-12:
                break
            t /= 2
        beta = beta + t * step
        improved, current = trial - current, trial
        if np.max(np.abs(t * step)) < 1e-9 or abs(improved) < 1e-10:
            break
    p = _expit(X @ beta)
    w = n * p * (1 - p)
    H = X.T @ (X * w[:, None]) + np.diag(pen) + 1e-12 * np.eye(p_)
    return {"beta": beta, "cov": np.linalg.inv(H), "ll": loglik(p, k, n)}


def loglik(p: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))


def _concordance(p: np.ndarray, k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    d = pd.DataFrame({"p": p, "pos": k, "neg": n - k}).groupby("p", sort=True)[["pos", "neg"]].sum()
    neg_below = d.neg.cumsum() - d.neg
    return float((d.pos * neg_below).sum() + 0.5 * (d.pos * d.neg).sum()), float(d.pos.sum() * d.neg.sum())


def auc(p: np.ndarray, k: np.ndarray, n: np.ndarray, year: np.ndarray | None = None) -> float:
    """Area under the ROC curve from cells: departures ranked against stays, ties counted as half.

    With ``year``, only people on the same list are compared (pairs pooled over years), so a model is not
    rewarded or penalised for the level of a year it never saw."""
    if year is None:
        c, t = _concordance(p, k, n)
        return c / t
    c = t = 0.0
    for y in np.unique(year):
        s = year == y
        cy, ty = _concordance(p[s], k[s], n[s])
        c, t = c + cy, t + ty
    return c / t


class Design:
    """Dummy and numeric terms with levels fixed on the full sample, so held-out years get the same columns."""

    def __init__(self, frame: pd.DataFrame, terms: Sequence[str]):
        self.terms = list(terms)
        self.levels: dict[str, list[Any]] = {}
        for t in self.terms:
            if not t.startswith("num:"):
                counts = frame.groupby(t).n.sum().sort_values(ascending=False)
                self.levels[t] = sorted(counts.index[1:].tolist(), key=str)   # most common level is the reference
                self.levels[t + "@ref"] = [counts.index[0]]

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        cols = [np.ones(len(frame))]
        for t in self.terms:
            if t.startswith("num:"):
                cols.append(frame[t[4:]].to_numpy(float))
            else:
                v = frame[t].to_numpy()
                cols += [(v == lev).astype(float) for lev in self.levels[t]]
        return np.column_stack(cols)

    def names(self) -> list[str]:
        out = ["const"]
        for t in self.terms:
            out += [t[4:]] if t.startswith("num:") else [f"{t}={lev}" for lev in self.levels[t]]
        return out


def fit_terms(frame: pd.DataFrame, terms: Sequence[str]) -> dict[str, Any]:
    d = Design(frame, terms)
    X = d.matrix(frame)
    f = fit_logit(X, frame.k.to_numpy(float), frame.n.to_numpy(float))
    f.update(design=d, names=d.names(), df=X.shape[1])
    return f


def loyo(frame: pd.DataFrame, terms: Sequence[str]) -> dict[str, Any]:
    """Leave one pair of lists out at a time: fit on the other years, predict the held-out year."""
    d = Design(frame, terms)
    preds = np.empty(len(frame))
    for y in sorted(frame.year.unique()):
        tr, te = (frame.year != y).to_numpy(), (frame.year == y).to_numpy()
        f = fit_logit(d.matrix(frame[tr]), frame.k[tr].to_numpy(float), frame.n[tr].to_numpy(float), RIDGE_CV)
        preds[te] = _expit(d.matrix(frame[te]) @ f["beta"])
    k, n = frame.k.to_numpy(float), frame.n.to_numpy(float)
    by_year = {}
    for y in sorted(frame.year.unique()):
        s = (frame.year == y).to_numpy()
        by_year[str(int(y))] = -loglik(preds[s], k[s], n[s]) / n[s].sum()
    brier = float(np.sum(k * (1 - preds) ** 2 + (n - k) * preds**2) / n.sum())
    return {"logloss": -loglik(preds, k, n) / n.sum(), "brier": brier,
            "auc": auc(preds, k, n, frame.year.to_numpy()),
            "logloss_by_year": by_year, "pred": preds}


# ---------------------------------------------------------------------------
# 1. Who has a history
# ---------------------------------------------------------------------------

def _near(b: pd.Series) -> pd.Series:
    return (b >= 1000) & (b < FAR)


def coverage(pers: pd.DataFrame) -> dict[str, Any]:
    p = pers[pers.year.between(*VOL_YEARS) & _near(pers.band_lo_musd)]
    out: dict[str, Any] = {}
    for name, sel in (("hist0", p["hist"] == 0), ("hist1", p["hist"] == 1), ("hist2plus", p["hist"] >= 2),
                      ("hist4", p["hist"] == 4)):
        st = int(p[sel & (p.table == "stock")].n.sum())
        ex = int(p[sel & (p.table == "exit")].n.sum())
        out[name] = {"stock": st, "exit": ex, "exit_rate": ex / st if st else float("nan")}
    total = sum(out[k]["stock"] for k in ("hist0", "hist1", "hist2plus"))
    out["total_stock"] = total
    out["share_hist2plus"] = out["hist2plus"]["stock"] / total
    q = p[(p.table == "stock") & (p["hist"] >= 2)]
    nb = p[p.table == "stock"].groupby("band_lo_musd").n.sum()
    out["share_hist2plus_by_band"] = {str(b): float(q[q.band_lo_musd == b].n.sum() / nb[b]) for b in NEAR_BANDS}
    f = pers[pers.year.between(*VOL_YEARS) & (pers.band_lo_musd >= FAR) & (pers.table == "stock")]
    out["far_share_hist2plus"] = float(f[f["hist"] >= 2].n.sum() / f.n.sum())
    return out


# ---------------------------------------------------------------------------
# 2. Persistence among large fortunes
# ---------------------------------------------------------------------------

def _changes(lc: pd.DataFrame) -> pd.DataFrame:
    lc = lc.assign(lo=lc.key.astype(int))
    big = (lc.lo >= 100) | (lc.lo < -100)
    mid = (lc.lo + 10) / 1000.0
    return lc.assign(big=np.where(big, lc.n, 0), c0sq=lc.n * mid**2, absc=lc.n * mid.abs())


def vol_levels(manifest: dict[str, Any]) -> list[int]:
    return [int(round(e * 1000)) for e in manifest["vol_edges"]]


def persistence(pers: pd.DataFrame, levels: Sequence[int], years: Sequence[int] = None,
                hist_sel: Callable[[pd.Series], pd.Series] = lambda h: h >= 2) -> dict[str, Any]:
    """Next year's large moves (|change| >= 0.1) among stayers from $3bn, by class of earlier volatility.

    Expected counts are indirectly standardised: each year x size-band cell contributes its own share of large
    moves among all stayers with a volatility class, so market-wide years and size are held fixed."""
    years = years or _years(VOL_YEARS)
    lc = pers[(pers.table == "logchange") & pers.year.isin(years) & (pers.band_lo_musd >= FAR)
              & hist_sel(pers["hist"])]
    lc = _changes(lc)
    cell = lc.groupby(["year", "band_lo_musd"])[["n", "big"]].sum()
    rate = (cell.big / cell.n).rename("rate")
    lc = lc.join(rate, on=["year", "band_lo_musd"])
    lc["exp"] = lc.n * lc.rate
    g = lc.groupby("vol")[["n", "big", "exp", "c0sq", "absc"]].sum()
    msum = lc.groupby("vol").ms_sum.sum()
    out: dict[str, Any] = {"stayers": int(lc.n.sum()), "big_share": float(lc.big.sum() / lc.n.sum()), "by_class": {}}
    for v in levels:
        if v not in g.index or g.loc[v, "n"] == 0:
            continue
        r = g.loc[v]
        o, e = float(r.big), float(r.exp)
        out["by_class"][str(v)] = {"stayers": int(r.n), "big_share": o / r.n, "expected_share": e / r.n,
                                   "ratio": o / e, "ci95": [o / e * math.exp(-1.96 / math.sqrt(max(o, 1))),
                                                            o / e * math.exp(1.96 / math.sqrt(max(o, 1)))],
                                   "mean_past_ms": float(msum[v] / r.n), "mean_next_sq": float(r.c0sq / r.n)}
    return out


def persistence_models(pers: pd.DataFrame, years: Sequence[int] = None) -> dict[str, Any]:
    """Large moves among stayers from $3bn: does earlier volatility predict once the latest change is known?

    Logits with year x band effects; likelihood-ratio tests of each block given the other."""
    years = years or _years(VOL_YEARS)
    lc = _changes(pers[(pers.table == "logchange") & pers.year.isin(years) & (pers.band_lo_musd >= FAR)
                       & (pers["hist"] >= 2)])
    lc["yb"] = lc.year.astype(str) + "|" + lc.band_lo_musd.astype(str)
    cells = lc.groupby(["yb", "vol", "prev"])[["n", "big"]].sum().reset_index().rename(columns={"big": "k"})
    base = fit_terms(cells, ["yb"])
    fits = {name: fit_terms(cells, ["yb"] + t) for name, t in
            (("prev", ["prev"]), ("vol", ["vol"]), ("both", ["prev", "vol"]))}
    out: dict[str, Any] = {}
    for name, a, b in (("vol_given_prev", "both", "prev"), ("prev_given_vol", "both", "vol"),
                       ("vol_alone", "vol", None), ("prev_alone", "prev", None)):
        fa = fits[a]
        fb = fits[b] if b else base
        lr = 2 * (fa["ll"] - fb["ll"])
        df = fa["df"] - fb["df"]
        out[name] = {"lr": lr, "df": df, "p": float(chi2.sf(lr, df)), "deviance_share": lr / (2 * (fits["both"]["ll"] - base["ll"]))}
    both = fits["both"]
    names = both["names"]
    hi = [i for i, nm in enumerate(names) if nm.startswith("vol=") and int(nm[4:]) >= 130]
    lo_ref = both["design"].levels["vol@ref"][0]
    out["reference_vol"] = int(lo_ref)
    out["odds_ratio_vol"] = {nm[4:]: float(math.exp(both["beta"][i])) for i, nm in enumerate(names) if nm.startswith("vol=")}
    out["odds_ratio_prev"] = {nm[5:]: float(math.exp(both["beta"][i])) for i, nm in enumerate(names) if nm.startswith("prev=")}
    out["reference_prev"] = both["design"].levels["prev@ref"][0]
    out["n_high_vol_terms"] = len(hi)
    return out


RISE_GROUPS = (("near", "$1bn to $1.5bn", 1000, NEAR), ("middle", "$1.5bn to $3bn", NEAR, FAR),
               ("far", "$3bn and above", FAR, 10**9))


def rises(pers: pd.DataFrame, levels: Sequence[int], years: Sequence[int] = None) -> dict[str, Any]:
    """Rises of at least 0.1 by the next list, by class of earlier volatility, in three wealth groups.

    A rise is observed whoever stays, at any distance from the line, so this measure of persistence is the same near
    the line and far from it. The denominator is everyone on the list (those who leave did not rise); expected rises
    use each year x band cell's own rate."""
    years = years or _years(VOL_YEARS)
    out: dict[str, Any] = {}
    for name, label, lo, hi in RISE_GROUPS:
        p = pers[pers.year.isin(years) & (pers.band_lo_musd >= lo) & (pers.band_lo_musd < hi) & (pers["hist"] >= 2)]
        st = p[p.table == "stock"].groupby(["year", "band_lo_musd", "vol"]).n.sum()
        lc = p[p.table == "logchange"]
        up = lc[lc.key.astype(int) >= 100].groupby(["year", "band_lo_musd", "vol"]).n.sum().reindex(st.index, fill_value=0)
        rate = up.groupby(["year", "band_lo_musd"]).sum() / st.groupby(["year", "band_lo_musd"]).sum()
        exp = st * st.index.droplevel("vol").map(rate).to_numpy()
        o, e, n = up.groupby("vol").sum(), exp.groupby("vol").sum(), st.groupby("vol").sum()
        out[name] = {"label": label, "people": int(st.sum()), "rise_share": float(up.sum() / st.sum()),
                     "zero_history_share": float(p[(p.table == "stock") & (p.ms_sum == 0)].n.sum() / st.sum()),
                     "by_class": {str(v): {"n": int(n[v]), "rises": int(o[v]), "expected": float(e[v]),
                                           "ratio": float(o[v] / e[v]),
                                           "ci95": [float(o[v] / e[v] * math.exp(-1.96 / math.sqrt(max(o[v], 1)))),
                                                    float(o[v] / e[v] * math.exp(1.96 / math.sqrt(max(o[v], 1))))]}
                                  for v in levels if v in n.index and e[v] > 0}}
        top, bottom = [str(v) for v in levels if v >= 130], [str(v) for v in levels if v < 40]
        agg = lambda ks, f: sum(out[name]["by_class"][k][f] for k in ks if k in out[name]["by_class"])  # noqa: E731
        ratio = lambda ks: agg(ks, "rises") / agg(ks, "expected")  # noqa: E731
        out[name]["ratio_top"] = float(ratio(top))
        out[name]["ratio_bottom"] = float(ratio(bottom))
    return out


def elasticity(per: dict[str, Any]) -> dict[str, Any]:
    """Regression to the mean across classes: log of next year's root mean square change on log of the earlier one."""
    bc = {int(k): v for k, v in per["by_class"].items() if int(k) >= 20}
    x = np.log(np.sqrt([v["mean_past_ms"] for v in bc.values()]))
    y = np.log(np.sqrt([v["mean_next_sq"] for v in bc.values()]))
    w = np.array([v["stayers"] for v in bc.values()], float)
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    b = float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))
    lo, hi = min(bc), max(bc)
    return {"elasticity": b, "past_rms_range": [math.sqrt(bc[lo]["mean_past_ms"]), math.sqrt(bc[hi]["mean_past_ms"])],
            "next_rms_range": [math.sqrt(bc[lo]["mean_next_sq"]), math.sqrt(bc[hi]["mean_next_sq"])],
            "doubling_effect": float(2**b - 1)}


# ---------------------------------------------------------------------------
# 3. Departures below $3bn: distance, volatility, both
# ---------------------------------------------------------------------------

def hazard_cells(pers: pd.DataFrame, years: Sequence[int], hist_sel: Callable[[pd.Series], pd.Series],
                 bands: Callable[[pd.Series], pd.Series] = _near) -> pd.DataFrame:
    p = pers[pers.year.isin(years) & bands(pers.band_lo_musd) & hist_sel(pers["hist"])
             & pers.table.isin(["stock", "exit"])]
    keys = ["year", "band_lo_musd", "vol", "prev", "age", "trend", "hist", "tenure"]
    st = p[p.table == "stock"].groupby(keys)[["n", "ms_sum"]].sum()
    ex = p[p.table == "exit"].groupby(keys).n.sum().rename("k")
    c = st.join(ex).fillna({"k": 0}).reset_index()
    c["band"] = c.band_lo_musd
    mid = c.band_lo_musd.map(lambda b: BAND_MID.get(b, b * 1.25))
    c["d"] = np.log10(mid / 1000.0)
    c["sig"] = np.sqrt(c.ms_sum / c.n + S0**2)
    c["logd"] = np.log(c.d)
    c["logsig"] = np.log(c.sig)
    c["logz"] = c.logd - c.logsig
    c["volu"] = np.where(c["hist"] >= 2, c.vol, -1)
    return c


MODELS = (("constant", []),
          ("distance", ["band"]),
          ("volatility", ["vol"]),
          ("distance_volatility", ["band", "vol"]),
          ("distance_logvol", ["band", "num:logsig"]),
          ("part5", ["band", "prev", "age"]),
          ("part5_volatility", ["band", "prev", "age", "vol"]),
          ("part5_trend_hist", ["band", "prev", "age", "trend", "hist"]),
          ("part5_all", ["band", "prev", "age", "vol", "trend", "hist"]),
          ("one_index", ["num:logz"]),
          ("two_index", ["num:logd", "num:logsig"]))


SHORT = ("constant", "distance", "volatility", "distance_volatility", "distance_logvol", "one_index")


def compare_models(c: pd.DataFrame, models: Sequence[tuple[str, list[str]]] = MODELS) -> dict[str, Any]:
    out: dict[str, Any] = {"cells": int(len(c)), "people": int(c.n.sum()), "exits": int(c.k.sum()),
                           "exit_rate": float(c.k.sum() / c.n.sum()), "years": [int(y) for y in sorted(c.year.unique())],
                           "models": {}}
    base = None
    for name, terms in models:
        cv = loyo(c, terms)
        fit = fit_terms(c, terms)
        r = {"terms": terms, "df": fit["df"], "ll_in": fit["ll"], "logloss": cv["logloss"], "brier": cv["brier"],
             "auc": cv["auc"], "logloss_by_year": cv["logloss_by_year"]}
        if base is None:
            base = r
        r["logloss_gain_pct"] = 100 * (1 - r["logloss"] / base["logloss"])
        out["models"][name] = r
    return out


def paired_years(cmp: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    """Held-out years: log loss of model b minus model a (positive = a better), mean and t interval over years."""
    ya, yb = cmp["models"][a]["logloss_by_year"], cmp["models"][b]["logloss_by_year"]
    base = cmp["models"]["constant"]["logloss_by_year"]
    diff = np.array([100 * (yb[y] - ya[y]) / base[y] for y in ya])
    m, s = float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(len(diff)))
    q = float(student_t.ppf(0.975, len(diff) - 1))
    return {"mean_pct_points": m, "ci95": [m - q * s, m + q * s], "years_a_better": int((diff > 0).sum()),
            "n_years": int(len(diff))}


def lr_tests(c: pd.DataFrame) -> dict[str, Any]:
    """In-sample likelihood-ratio tests with year effects."""
    out = {}
    for name, big, small in (("vol_given_distance", ["year", "band", "vol"], ["year", "band"]),
                             ("distance_given_vol", ["year", "band", "vol"], ["year", "vol"]),
                             ("vol_given_part5", ["year", "band", "prev", "age", "vol"], ["year", "band", "prev", "age"]),
                             ("trend_hist_given_rest", ["year", "band", "prev", "age", "vol", "trend", "hist"],
                              ["year", "band", "prev", "age", "vol"]),
                             ("ratio_restriction", ["year", "num:logd", "num:logsig"], ["year", "num:logz"]),
                             ("flexible_vs_one_index", ["year", "band", "vol"], ["year", "num:logz"])):
        fb, fs = fit_terms(c, big), fit_terms(c, small)
        lr = 2 * (fb["ll"] - fs["ll"])
        df = fb["df"] - fs["df"]
        out[name] = {"lr": float(lr), "df": int(df), "p": float(chi2.sf(lr, df))}
    two = fit_terms(c, ["year", "num:logd", "num:logsig"])
    i, j = two["names"].index("logd"), two["names"].index("logsig")
    out["two_index_coef"] = {"logd": float(two["beta"][i]), "logsig": float(two["beta"][j]),
                             "se_logd": float(math.sqrt(two["cov"][i, i])), "se_logsig": float(math.sqrt(two["cov"][j, j]))}
    one = fit_terms(c, ["num:logz"])
    out["one_index_coef"] = {"const": float(one["beta"][0]), "logz": float(one["beta"][1])}
    lv = fit_terms(c, ["year", "band", "num:logsig"])
    j = lv["names"].index("logsig")
    se = math.sqrt(lv["cov"][j, j])
    out["logvol_given_distance"] = {"coef": float(lv["beta"][j]), "se": se,
                                    "odds_per_doubling": float(2 ** lv["beta"][j]),
                                    "ci95_odds_per_doubling": [float(2 ** (lv["beta"][j] - 1.96 * se)),
                                                               float(2 ** (lv["beta"][j] + 1.96 * se))]}
    f0, fd, fv = fit_terms(c, []), fit_terms(c, ["band"]), fit_terms(c, ["vol"])
    out["fit_share_vol_over_distance"] = float((fv["ll"] - f0["ll"]) / (fd["ll"] - f0["ll"]))
    tr = fit_terms(c, ["year", "band", "prev", "age", "trend", "hist"])
    out["trend_hist_coef"] = {nm: float(b) for nm, b in zip(tr["names"], tr["beta"]) if nm.startswith(("trend=", "hist=", "prev=", "age="))}
    out["trend_hist_ref"] = {t: tr["design"].levels[t + "@ref"][0] for t in ("trend", "hist", "prev", "age")}
    for name, big, small in (("trend_given_part5", ["year", "band", "prev", "age", "trend"], ["year", "band", "prev", "age"]),
                             ("hist_given_part5", ["year", "band", "prev", "age", "hist"], ["year", "band", "prev", "age"])):
        fb, fs = fit_terms(c, big), fit_terms(c, small)
        lr = 2 * (fb["ll"] - fs["ll"])
        out[name] = {"lr": float(lr), "df": int(fb["df"] - fs["df"]), "p": float(chi2.sf(lr, fb["df"] - fs["df"]))}
    out["by_trend"] = {t: {"n": int(c[c.trend == t].n.sum()), "exit_rate": float(c[c.trend == t].k.sum() / c[c.trend == t].n.sum())}
                       for t in ("down", "flat", "up")}
    out["by_hist"] = {str(h): {"n": int(c[c["hist"] == h].n.sum()),
                               "exit_rate": float(c[c["hist"] == h].k.sum() / c[c["hist"] == h].n.sum())} for h in (2, 3, 4)}
    return out


def exit_grid(c: pd.DataFrame) -> dict[str, Any]:
    """Observed departure rates by distance band and volatility group, pooled over years."""
    out: dict[str, Any] = {}
    for g, label, sel in VOL_GROUPS:
        d = c[sel(c.vol)]
        b = d.groupby("band_lo_musd")[["n", "k"]].sum()
        out[g] = {"label": label, "bands": {str(int(x)): {"n": int(r.n), "exit_rate": float(r.k / r.n)}
                                            for x, r in b.iterrows()},
                  "n": int(d.n.sum()), "exit_rate": float(d.k.sum() / d.n.sum()),
                  "mean_ms": float(d.ms_sum.sum() / d.n.sum())}
    lo, hi = out["low"]["bands"], out["high"]["bands"]
    out["high_over_low_by_band"] = {b: hi[b]["exit_rate"] / lo[b]["exit_rate"] for b in lo if b in hi and lo[b]["exit_rate"] > 0}
    return out


def collapse(c: pd.DataFrame) -> dict[str, Any]:
    """Departure rate against distance in units of own volatility, cells pooled over years and other dimensions."""
    g = c.groupby(["band_lo_musd", "vol"]).agg(n=("n", "sum"), k=("k", "sum"), ms=("ms_sum", "sum")).reset_index()
    g = g[g.n >= 25]
    g["d"] = np.log10(g.band_lo_musd.map(BAND_MID) / 1000.0)
    g["z"] = g.d / np.sqrt(g.ms / g.n + S0**2)
    g["rate"] = g.k / g.n
    one = fit_terms(c, ["num:logz"])
    zs = np.exp(np.linspace(np.log(0.05), np.log(25), 60))
    curve = _expit(one["beta"][0] + one["beta"][1] * np.log(zs))
    # how much of the spread across cells the index absorbs, against band alone
    w = g.n.to_numpy(float)
    pz = _expit(one["beta"][0] + one["beta"][1] * np.log(g.z.to_numpy()))
    band_rate = g.groupby("band_lo_musd").apply(lambda z: z.k.sum() / z.n.sum(), include_groups=False)
    pb = g.band_lo_musd.map(band_rate).to_numpy()
    tot = np.average((g.rate - np.average(g.rate, weights=w)) ** 2, weights=w)
    return {"cells": [{"band": int(r.band_lo_musd), "vol": int(r.vol), "n": int(r.n), "z": float(r.z),
                       "rate": float(r.rate)} for r in g.itertuples()],
            "curve": {"z": zs.tolist(), "p": curve.tolist()},
            "weighted_r2_index": float(1 - np.average((g.rate - pz) ** 2, weights=w) / tot),
            "weighted_r2_band": float(1 - np.average((g.rate - pb) ** 2, weights=w) / tot)}


# ---------------------------------------------------------------------------
# 4. Composition: is the extra volatility near the line a matter of who sits there?
# ---------------------------------------------------------------------------

def composition(pers: pd.DataFrame, levels: Sequence[int], ref: dict[str, Any]) -> dict[str, Any]:
    """Rises of at least 0.1 below $1.5bn against the same year's fortunes from $3bn (all of them, as in Part 5).

    Split by history: people with at least two earlier lists (who have a volatility class) and the others. For the
    first group the expectation is also adjusted for its volatility mix, using each class's relative rate of large
    rises among large fortunes with a class (indirect standardisation)."""
    years = _years(VOL_YEARS)
    p = pers[pers.year.isin(years)]
    far = p[p.band_lo_musd >= FAR]
    near = p[(p.band_lo_musd >= 1000) & (p.band_lo_musd < NEAR)]

    def up(lc: pd.DataFrame, keys: list[str]) -> pd.Series:
        return lc.assign(up=np.where(lc.key.astype(int) >= 100, lc.n, 0)).groupby(keys).up.sum()

    far_rate = up(far[far.table == "logchange"], ["year"]) / far[far.table == "stock"].groupby("year").n.sum()
    out: dict[str, Any] = {}
    for name, sel in (("with_history", near["hist"] >= 2), ("without_history", near["hist"] < 2)):
        d = near[sel]
        st = d[d.table == "stock"].groupby("year").n.sum()
        o = float(up(d[d.table == "logchange"], ["year"]).sum())
        e = float((st * st.index.map(far_rate).to_numpy()).sum())
        out[name] = {"people": int(st.sum()), "observed": o, "expected": e, "ratio": o / e,
                     "ci95": [o / e * math.exp(-1.96 / math.sqrt(o)), o / e * math.exp(1.96 / math.sqrt(o))]}
    fh = far[far["hist"] >= 2]
    f_st = fh[fh.table == "stock"].groupby(["year", "vol"]).n.sum()
    f_up = up(fh[fh.table == "logchange"], ["year", "vol"]).reindex(f_st.index, fill_value=0)
    f_rate = f_up.groupby("year").sum() / f_st.groupby("year").sum()
    factor = f_up.groupby("vol").sum() / (f_st * f_st.index.get_level_values("year").map(f_rate).to_numpy()).groupby("vol").sum()
    nh = near[near["hist"] >= 2]
    n_st = nh[nh.table == "stock"].groupby(["year", "vol"]).n.sum()
    n_up = float(up(nh[nh.table == "logchange"], ["year"]).sum())
    yr = n_st.index.get_level_values("year").map(f_rate).to_numpy()
    fac = n_st.index.get_level_values("vol").map(factor).to_numpy()
    e0, e1 = float((n_st * yr).sum()), float((n_st * yr * fac).sum())
    near_ms = float(nh[nh.table == "stock"].ms_sum.sum() / nh[nh.table == "stock"].n.sum())
    far_ms = float(fh[fh.table == "stock"].ms_sum.sum() / fh[fh.table == "stock"].n.sum())
    mix = lambda st: {str(v): float(st.groupby("vol").sum().get(v, 0) / st.sum()) for v in levels}  # noqa: E731
    out["class_adjusted"] = {"observed": n_up, "expected_pooled": e0, "expected_by_class": e1, "ratio_pooled": n_up / e0,
                             "ratio_by_class": n_up / e1,
                             "ci95_by_class": [n_up / e1 * math.exp(-1.96 / math.sqrt(n_up)),
                                               n_up / e1 * math.exp(1.96 / math.sqrt(n_up))]}
    out["class_factor"] = {str(int(v)): float(f) for v, f in factor.items()}
    out["near_mix"], out["far_mix"] = mix(n_st), mix(f_st)
    out["near_rms"], out["far_rms"] = math.sqrt(near_ms), math.sqrt(far_ms)
    out["near_low_share"] = sum(out["near_mix"][str(v)] for v in levels if v < 40)
    out["far_low_share"] = sum(out["far_mix"][str(v)] for v in levels if v < 40)
    out["part5_ratio_0.1"] = ref.get("upper_tail_near_0.1")
    return out


# ---------------------------------------------------------------------------
# 5. Sector
# ---------------------------------------------------------------------------

def sector(pers: pd.DataFrame) -> dict[str, Any]:
    p = pers[pers.year.isin(_years(VOL_YEARS)) & _near(pers.band_lo_musd) & (pers["hist"] >= 2)
             & pers.table.isin(["industry_stock", "industry_exit"])]
    p = p.assign(key=p.key.map(lambda k: INDUSTRY_MAP.get(k, k)))
    keys = ["year", "band_lo_musd", "vol", "key"]
    st = p[p.table == "industry_stock"].groupby(keys).n.sum()
    ex = p[p.table == "industry_exit"].groupby(keys).n.sum().rename("k")
    c = pd.concat([st, ex], axis=1).fillna({"k": 0}).reset_index()
    c["band"], c["industry"] = c.band_lo_musd, c.key
    labels = c.groupby("industry").n.sum()
    out: dict[str, Any] = {"n_labels": int(len(labels)), "largest": {k: int(v) for k, v in labels.nlargest(5).items()}}
    for name, big, small in (("vol_given_band", ["year", "band", "vol"], ["year", "band"]),
                             ("vol_given_band_industry", ["year", "band", "industry", "vol"], ["year", "band", "industry"]),
                             ("industry_given_band_vol", ["year", "band", "industry", "vol"], ["year", "band", "vol"])):
        fb, fs = fit_terms(c, big), fit_terms(c, small)
        lr = 2 * (fb["ll"] - fs["ll"])
        out[name] = {"lr": float(lr), "df": int(fb["df"] - fs["df"]), "p": float(chi2.sf(lr, fb["df"] - fs["df"]))}
    out["vol_lr_kept_within_industry"] = out["vol_given_band_industry"]["lr"] / out["vol_given_band"]["lr"]
    cv = compare_models(c, (("constant", []), ("distance", ["band"]), ("distance_industry", ["band", "industry"]),
                            ("distance_volatility", ["band", "vol"]),
                            ("distance_industry_volatility", ["band", "industry", "vol"])))
    out["cv"] = {k: {kk: v[kk] for kk in ("logloss_gain_pct", "auc")} for k, v in cv["models"].items()}
    return out


# ---------------------------------------------------------------------------
# 6. Simulation: what would the tests show if volatility were a fixed trait?
# ---------------------------------------------------------------------------

SIM_TRAITS = (("none", 0.0), ("moderate", 0.5), ("strong", 0.8))
SIM_SEEDS = (11, 12, 13)


def simulate_lists(seed: int, trait_sd: float, n: int = 10000, sigma_med: float = 0.075, alpha: float = 1.15,
                   years: Sequence[int] = range(2010, 2026), burn: int = 6) -> dict[int, list[Any]]:
    """Synthetic Forbes lists with the same reporting rules: fortunes rounded down to $100m, listed from $1bn.

    Each fortune has its own scale sigma_i = sigma_med * exp(trait_sd * Z - trait_sd^2); yearly change in log10 is a
    common year shift plus sigma_i times Student-t (4 df) noise. Two per cent of fortunes a year (six per cent from
    age 85) are replaced by a new draw, standing in for deaths and splits."""
    from fetch_forbes_flows import Person
    rng = np.random.default_rng(seed)
    draw_w = lambda k: np.log10(300 * rng.uniform(size=k) ** (-1 / alpha))  # noqa: E731
    draw_s = lambda k: sigma_med * np.exp(trait_sd * rng.standard_normal(k) - trait_sd**2)  # noqa: E731
    logw, sig, age = draw_w(n), draw_s(n), rng.integers(35, 90, n)
    ids, nxt = np.arange(n), n
    lists: dict[int, list[Any]] = {}
    for t in range(min(years) - burn, max(years) + 1):
        logw = np.minimum(logw + rng.normal(0.01, 0.05) + sig * rng.standard_t(4, n) / math.sqrt(2), np.log10(2e5))
        gone = rng.uniform(size=n) < 0.02 + 0.04 * (age >= 85)
        k = int(gone.sum())
        logw[gone], sig[gone], age[gone] = draw_w(k), draw_s(k), rng.integers(35, 70, k)
        ids[gone] = np.arange(nxt, nxt + k)
        nxt += k
        age = age + 1
        if t in years:
            rep = np.floor(10**logw / 100) * 100
            lists[t] = [Person({"uri": f"s{ids[i]}", "finalWorth": float(rep[i]), "age": int(age[i])}, t)
                        for i in np.flatnonzero(rep >= 1000)]
    return lists


def simulation(levels: Sequence[int]) -> dict[str, Any]:
    from fetch_forbes_persistence import aggregate
    out: dict[str, Any] = {}
    keep = ("constant", "distance", "volatility", "distance_volatility", "distance_logvol")
    for name, sd in SIM_TRAITS:
        runs = []
        for seed in SIM_SEEDS:
            rows = pd.DataFrame(aggregate(simulate_lists(seed, sd)))
            for col in ("key", "prev", "tenure", "age", "trend"):
                rows[col] = rows[col].astype(str)
            rs = rises(rows, levels)
            c = hazard_cells(rows, _years(VOL_YEARS), lambda h: h >= 2)
            cmp = compare_models(c, [m for m in MODELS if m[0] in keep])
            runs.append({"far_top": rs["far"]["ratio_top"], "far_bottom": rs["far"]["ratio_bottom"],
                         "near_top": rs["near"]["ratio_top"], "near_bottom": rs["near"]["ratio_bottom"],
                         "high_over_low": standardised_high_low(c)["high_over_low"], "exit_rate": cmp["exit_rate"],
                         **{f"gain_{m}": cmp["models"][m]["logloss_gain_pct"] for m in keep[1:]}})
        out[name] = {"trait_sd": sd, **{k: float(np.mean([r[k] for r in runs])) for k in runs[0]},
                     "added_by_logvol": [r["gain_distance_logvol"] - r["gain_distance"] for r in runs]}
    return out


# ---------------------------------------------------------------------------
# 7. Bootstrap over years
# ---------------------------------------------------------------------------

def bootstrap(c: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 6) -> dict[str, Any]:
    """Resample pairs of lists with replacement; refit distance + volatility and the one-index model."""
    rng = np.random.default_rng(seed)
    years = sorted(c.year.unique())
    by_year = {y: c[c.year == y] for y in years}
    hi_lo, slope, lr_share, logsig = [], [], [], []
    for _ in range(n_boot):
        draw = rng.choice(years, size=len(years), replace=True)
        b = pd.concat([by_year[y] for y in draw], ignore_index=True)
        b = b.groupby(["band", "vol", "prev", "age", "trend", "hist"], as_index=False)[["n", "k", "ms_sum", "logd"]].agg(
            {"n": "sum", "k": "sum", "ms_sum": "sum", "logd": "first"})
        b["logz"] = b.logd - np.log(np.sqrt(b.ms_sum / b.n + S0**2))
        g = b.assign(grp=np.select([b.vol < 60, b.vol < 130], ["low", "mid"], "high"))
        # within-band comparison of high and low volatility (direct standardisation to the pooled band mix)
        w = g.groupby("band").n.sum() / g.n.sum()
        rate = g.groupby(["grp", "band"])[["n", "k"]].sum()
        std = {grp: sum(w[bd] * rate.loc[(grp, bd), "k"] / rate.loc[(grp, bd), "n"]
                        for bd in w.index if (grp, bd) in rate.index and rate.loc[(grp, bd), "n"] > 0)
               for grp in ("low", "high")}
        hi_lo.append(std["high"] / std["low"])
        one = fit_terms(b, ["num:logz"])
        slope.append(float(one["beta"][1]))
        d1, v1, c0 = fit_terms(b, ["band"]), fit_terms(b, ["vol"]), fit_terms(b, [])
        lr_share.append((v1["ll"] - c0["ll"]) / (d1["ll"] - c0["ll"]))
        b["logsig"] = b.logd - b.logz
        lv = fit_terms(b, ["band", "num:logsig"])
        logsig.append(float(2 ** lv["beta"][lv["names"].index("logsig")]))
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]  # noqa: E731
    return {"n_boot": n_boot, "high_over_low_std_ci95": q(hi_lo), "one_index_slope_ci95": q(slope),
            "vol_over_distance_fit_ci95": q(lr_share), "odds_per_doubling_ci95": q(logsig)}


def standardised_high_low(c: pd.DataFrame) -> dict[str, Any]:
    w = c.groupby("band").n.sum() / c.n.sum()
    out = {}
    for g, _, sel in VOL_GROUPS:
        d = c[sel(c.vol)].groupby("band")[["n", "k"]].sum()
        out[g] = float(sum(w[b] * d.loc[b, "k"] / d.loc[b, "n"] for b in w.index if b in d.index and d.loc[b, "n"] > 0))
    out["high_over_low"] = out["high"] / out["low"]
    out["crude"] = {g: float(c[sel(c.vol)].k.sum() / c[sel(c.vol)].n.sum()) for g, _, sel in VOL_GROUPS}
    return out


# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------

def _py(x: Any) -> Any:
    """Plain Python types for JSON (numpy scalars become ints and floats, keys become strings)."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x.item() if isinstance(x, np.generic) else x


def compute(inp: dict[str, Any], n_boot: int = N_BOOT) -> dict[str, Any]:
    pers, tenure = inp["forbes_persistence"], inp["forbes_tenure"]
    man = inp["forbes_persistence_manifest"]
    levels = vol_levels(man)
    ref: dict[str, Any] = {}
    p5 = HERE / "results" / "part5" / "numbers.json"
    if p5.exists():
        n5 = json.loads(p5.read_text(encoding="utf-8"))
        ref = {"upper_tail_near_0.1": n5["upper_tail_near"]["0.1"]["ratio"], "scale_near": n5["scale_near"],
               "far_move_after_big": n5["previous"]["far_move_0.1_after_big"],
               "far_move_after_flat": n5["previous"]["far_move_0.1_after_flat"],
               "near_incumbent_exit_rate": n5["tenure"]["near"]["incumbent"]["exit_rate"],
               "near_new_exit_rate": n5["tenure"]["near"]["new"]["exit_rate"]}
    main = hazard_cells(pers, _years(VOL_YEARS), lambda h: h >= 2)
    full = hazard_cells(pers, _years(FULL_YEARS), lambda h: h == 4)
    everyone = hazard_cells(pers, _years(VOL_YEARS), lambda h: h >= 0)
    allbands = hazard_cells(pers, _years(VOL_YEARS), lambda h: h >= 2, bands=lambda b: b >= 1000)
    cmp_main = compare_models(main)
    cmp_full = compare_models(full, [m for m in MODELS if m[0] in SHORT])
    cmp_every = compare_models(everyone.assign(vol=everyone.volu),
                               (("constant", []), ("distance", ["band"]), ("distance_tenure", ["band", "hist"]),
                                ("distance_volatility", ["band", "vol", "hist"])))
    per = persistence(pers, levels)
    out = {
        "fetch": {y: {"n": e.get("n_persons"), "same_as_part5": e.get("same_as_part5")} for y, e in man["years"].items()},
        "vol_levels": levels,
        "part5_refs": ref,
        "check_part5": check_against_part5(pers, tenure),
        "coverage": coverage(pers),
        "persistence": per,
        "elasticity": elasticity(per),
        "persistence_full_window": persistence(pers, levels, _years(FULL_YEARS), lambda h: h == 4),
        "persistence_models": persistence_models(pers),
        "rises": rises(pers, levels),
        "hazard": cmp_main,
        "paired": {f"{a}_vs_{b}": paired_years(cmp_main, a, b) for a, b in
                   (("distance", "volatility"), ("distance_volatility", "distance"), ("distance_logvol", "distance"),
                    ("part5_volatility", "part5"), ("part5_trend_hist", "part5"), ("part5_all", "part5_trend_hist"),
                    ("distance", "one_index"))},
        "lr": lr_tests(main),
        "exit_grid": exit_grid(main),
        "standardised": standardised_high_low(main),
        "collapse": collapse(main),
        "hazard_full_window": cmp_full,
        "paired_full_window": {f"{a}_vs_{b}": paired_years(cmp_full, a, b) for a, b in
                               (("distance", "volatility"), ("distance_logvol", "distance"), ("distance_volatility", "distance"))},
        "hazard_everyone": cmp_every,
        "paired_everyone": {f"{a}_vs_{b}": paired_years(cmp_every, a, b) for a, b in
                            (("distance_tenure", "distance"), ("distance_volatility", "distance_tenure"))},
        "hazard_all_bands": compare_models(allbands, [m for m in MODELS if m[0] in SHORT]),
        "composition": composition(pers, levels, ref),
        "sector": sector(pers),
        "simulation": simulation(levels),
        "bootstrap": bootstrap(main, n_boot),
    }
    return _py(out)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def frame(fig: plt.Figure, title: str, subtitle: str, **kw: Any) -> None:
    _frame(fig, title, subtitle, source=SOURCE, **kw)


def _vol_tick(v: float) -> str:
    return f"{v / 1000:.2f}"


def fig_persistence(n: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    levels = n["vol_levels"]
    for (key, col), off in zip((("near", BLUE), ("middle", AQUA), ("far", ORANGE)), (-0.16, 0.0, 0.16)):
        r = n["rises"][key]
        ks = [k for k in r["by_class"] if r["by_class"][k]["n"] >= 40]
        x = np.array([levels.index(int(k)) for k in ks], float) + off
        v = np.array([r["by_class"][k]["ratio"] for k in ks])
        lo = np.array([r["by_class"][k]["ci95"][0] for k in ks])
        hi = np.array([r["by_class"][k]["ci95"][1] for k in ks])
        ax.vlines(x, lo, hi, color=col, lw=1.6, alpha=0.75)
        ax.plot(x, v, "o-", color=col, ms=6, lw=1.3, label=f"Net worth {r['label']}")
    ax.axhline(1, color=BASELINE, lw=1)
    ax.set_xticks(range(len(levels)))
    edges = levels[1:] + [None]
    ax.set_xticklabels([f"{_vol_tick(a)}-{_vol_tick(b)}" if b else f"{_vol_tick(a)}+" for a, b in zip(levels, edges)],
                       fontsize=9.5)
    ax.set_xlabel("Typical size of the earlier yearly changes (root mean square, log10 units)")
    ax.set_ylabel("Rises of 26% or more: observed ÷ expected")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}×"))
    ax.set_ylim(0, 2.8)
    ax.legend(loc="upper left", fontsize=9.5)
    frame(fig, "Fortunes that moved a lot before move a lot again",
          "People with at least two earlier lists, lists 2012-2024: share whose net worth rose by at least 0.1 in log10\n"
          "by the next list, relative to the same year and size band. The latest change is left out. 95% intervals.",
          top=0.83, bottom=0.15, left=0.09)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_grid(n: dict[str, Any], out: Path) -> None:
    g = n["exit_grid"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for key, col in (("low", AQUA), ("mid", BLUE), ("high", ORANGE)):
        bands = [b for b in NEAR_BANDS if str(b) in g[key]["bands"] and g[key]["bands"][str(b)]["n"] >= 30]
        x = np.array([BAND_MID[b] for b in bands], float)
        r = np.array([g[key]["bands"][str(b)]["exit_rate"] for b in bands])
        nn = np.array([g[key]["bands"][str(b)]["n"] for b in bands])
        se = np.sqrt(r * (1 - r) / nn)
        ax.fill_between(x, 100 * np.clip(r - 1.96 * se, 0, 1), 100 * (r + 1.96 * se), color=col, alpha=0.13, lw=0)
        ax.plot(x, 100 * r, "o-", color=col, lw=2, ms=5.5, label=f"Earlier changes {g[key]['label']}")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1000, 1250, 1500, 2000, 3000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:g}bn"))
    ax.set_xlim(1000, 3000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, None)
    ax.set_xlabel("Net worth on this year's list (log scale)")
    ax.set_ylabel("Share not on the next list")
    ax.legend(loc="upper right", fontsize=9.5)
    frame(fig, "Distance does most of the sorting; past volatility adds a little",
          "People below $3bn with at least two earlier lists, lists 2012-2024 pooled. Groups by the root mean square\n"
          "of their earlier yearly changes in log10 net worth (the latest change left out). Shading: 95% intervals.",
          top=0.83, bottom=0.14, left=0.08)
    fig.savefig(out, dpi=250)
    plt.close(fig)


MODEL_LABELS = (("distance", "Distance from $1bn"), ("volatility", "Past volatility"),
                ("distance_volatility", "Distance + past volatility"),
                ("one_index", "Distance ÷ own volatility (one number)"),
                ("part5", "Distance + latest change + age"),
                ("part5_volatility", "  … + past volatility"),
                ("part5_trend_hist", "  … + direction and length of history"))


def fig_models(n: dict[str, Any], out: Path) -> None:
    m = n["hazard"]["models"]
    base = m["constant"]["logloss_by_year"]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for i, (k, lab) in enumerate(MODEL_LABELS):
        per = np.array([100 * (1 - m[k]["logloss_by_year"][y] / base[y]) for y in base])
        v = m[k]["logloss_gain_pct"]
        col = ORANGE if "volatility" in k or k == "one_index" else BLUE
        ax.barh(i, v, color=col, height=0.6)
        ax.plot([per.min(), per.max()], [i, i], color=INK, lw=1, alpha=0.55)
        ax.plot(per, np.full(len(per), i), "|", color=INK, ms=9, alpha=0.55)
        ax.text(max(v, per.max()) + 0.4, i, f"{fmt(v, 1)}%   AUC {m[k]['auc']:.2f}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(MODEL_LABELS)))
    ax.set_yticklabels([lab for _, lab in MODEL_LABELS])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    lo = min(0, min(100 * (1 - m[k]["logloss_by_year"][y] / base[y]) for k, _ in MODEL_LABELS for y in base))
    ax.set_xlim(lo - 1, max(m[k]["logloss_gain_pct"] for k, _ in MODEL_LABELS) * 1.0 + 14)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_xlabel("Reduction in out-of-sample log loss against a constant departure rate")
    ax.grid(axis="y", visible=False)
    frame(fig, "Out of sample, distance beats past volatility",
          "Each year's departures predicted from models fitted on the other years, people below $3bn with at least two\n"
          "earlier lists, lists 2012-2024. Bars: all years pooled; ticks: single held-out years.",
          top=0.83, bottom=0.14, left=0.33)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_collapse(n: dict[str, Any], out: Path) -> None:
    c = n["collapse"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    groups = ((lambda b: b < 1300, BLUE, "$1.0bn-$1.3bn"), (lambda b: (b >= 1300) & (b < 2000), AQUA, "$1.3bn-$2bn"),
              (lambda b: b >= 2000, ORANGE, "$2bn-$3bn"))
    cells = pd.DataFrame(c["cells"])
    for sel, col, lab in groups:
        d = cells[sel(cells.band)]
        ax.scatter(d.z, 100 * d.rate, s=np.clip(d.n / 6, 12, 160), color=col, alpha=0.7, label=lab, lw=0)
    ax.plot(c["curve"]["z"], 100 * np.array(c["curve"]["p"]), color=INK, lw=2, label="Logit on log(distance ÷ volatility)")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.08, 25)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, None)
    ax.set_xlabel("Distance from $1bn ÷ own past volatility, both in log10 units (log scale)")
    ax.set_ylabel("Share not on the next list")
    ax.legend(loc="upper right", fontsize=9.5)
    frame(fig, "Dividing distance by volatility does not line departures up",
          "Each dot is a net-worth band and volatility class (at least 25 people), lists 2012-2024 pooled; dot size shows\n"
          "the number of people. At the same ratio, fortunes nearer the line still leave more often.",
          top=0.83, bottom=0.14, left=0.08)
    fig.savefig(out, dpi=250)
    plt.close(fig)


FIGURE_FILES = ("fig1_persistence.png", "fig2_distance_volatility.png", "fig3_out_of_sample.png", "fig4_one_index.png")


def render(n: dict[str, Any], outdir: Path) -> None:
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    fig_persistence(n, outdir / FIGURE_FILES[0])
    fig_grid(n, outdir / FIGURE_FILES[1])
    fig_models(n, outdir / FIGURE_FILES[2])
    fig_collapse(n, outdir / FIGURE_FILES[3])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _r(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_r(v) for v in x] if len(x) <= 12 else f"[{len(x)} items]"
    return x


def summary_lines(n: dict[str, Any]) -> list[str]:
    out = [f"Fetch: {n['fetch']}", f"Check against Part 5: {n['check_part5']}", f"Part 5 refs: {_r(n['part5_refs'])}",
           f"Coverage: {_r(n['coverage'])}"]
    for k in ("persistence", "persistence_full_window"):
        p = n[k]
        out.append(f"{k}: stayers {p['stayers']}, big share {p['big_share']:.4f}")
        for v, r in p["by_class"].items():
            out.append(f"  vol {v}: n {r['stayers']} big {r['big_share']:.3f} exp {r['expected_share']:.3f} ratio "
                       f"{r['ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] past ms {r['mean_past_ms']:.5f} "
                       f"next sq {r['mean_next_sq']:.5f}")
    out.append(f"Elasticity: {_r(n['elasticity'])}")
    out.append(f"Persistence models: {_r(n['persistence_models'])}")
    for g, r in n["rises"].items():
        out.append(f"Rises {g}: people {r['people']} share {r['rise_share']:.4f} zero-history {r['zero_history_share']:.4f} "
                   f"top {r['ratio_top']:.3f} bottom {r['ratio_bottom']:.3f}; "
                   + "; ".join(f"{v} {x['ratio']:.2f} ({x['n']})" for v, x in r["by_class"].items()))
    for key in ("hazard", "hazard_full_window", "hazard_everyone", "hazard_all_bands"):
        h = n[key]
        out.append(f"{key}: cells {h['cells']} people {h['people']} exits {h['exits']} rate {h['exit_rate']:.4f}")
        for m, r in h["models"].items():
            out.append(f"  {m}: df {r['df']} logloss {r['logloss']:.5f} gain {r['logloss_gain_pct']:.2f}% brier "
                       f"{r['brier']:.5f} auc {r['auc']:.4f}")
    out.append(f"Paired years: {_r(n['paired'])}")
    out.append(f"Paired full window: {_r(n['paired_full_window'])}")
    out.append(f"Paired everyone: {_r(n['paired_everyone'])}")
    out.append(f"LR tests: {_r(n['lr'])}")
    g = n["exit_grid"]
    for k in ("low", "mid", "high"):
        out.append(f"Grid {k}: n {g[k]['n']} rate {g[k]['exit_rate']:.4f} rms {math.sqrt(g[k]['mean_ms']):.4f} "
                   + "; ".join(f"{b} {v['exit_rate']:.3f} ({v['n']})" for b, v in g[k]["bands"].items()))
    out.append(f"  high/low by band: {_r(g['high_over_low_by_band'])}")
    out.append(f"Standardised: {_r(n['standardised'])}")
    c = n["collapse"]
    out.append(f"Collapse: cells {len(c['cells'])}, weighted R2 index {c['weighted_r2_index']:.4f}, band {c['weighted_r2_band']:.4f}")
    out.append(f"Composition: {_r(n['composition'])}")
    out.append(f"Sector: {_r(n['sector'])}")
    for k, v in n["simulation"].items():
        out.append(f"Simulation {k}: {_r(v)}")
    out.append(f"Bootstrap: {_r(n['bootstrap'])}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    inp = load_inputs()
    numbers = json.loads(json.dumps(compute(inp)))
    lines_out = summary_lines(numbers)
    print("\n".join(lines_out))
    target = RESULTS / "numbers.json"
    if args.check:
        problems: list[str] = []
        _compare(numbers, json.loads(target.read_text(encoding="utf-8")), "numbers", problems)
        if problems:
            print("\nREPRODUCTION FAILED:\n" + "\n".join(problems[:50]), file=sys.stderr)
            return 1
        print("\nREPRODUCED: every number matches results/part6/numbers.json")
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (RESULTS / "summary.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    if not args.no_figures:
        render(numbers, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
