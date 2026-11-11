"""Part 3 of the actuarial-tariff-calibration series: a calibration audit of GLM and boosted frequency models on freMTPL2.

Every number quoted in the article is printed by this script. Run:

    python audit_numbers.py > output.txt      # a few minutes, single-threaded; downloads freMTPL2 once

Data: freMTPL2freq and freMTPL2sev from a pinned commit of CASdatasets (677,991 policies). Cleaning follows the
common practice of the actuarial case studies: exposure capped at one year and claim counts capped at four for the
frequency models; claim amounts are kept in full for the loss targets.

Sections
1. The data: exposure, claim counts, the linked high-count policies, the reconciliation of the two tables, the fixed
   IRSA-IDA amounts and the large claims.
2. Three frequency models on a 60/20/20 split: a GLM with bonus-malus entered linearly (the usual case-study
   specification), the same GLM with bonus-malus in bands, and Poisson gradient boosting.
3. The test-sample verdict: Gini, deviance and its Murphy split, balance, the premium's own Gini, and the part of
   the miscalibration that is only the level.
4. Balance by segment (relative to each model's own level), with Poisson intervals.
5. Balance by premium decile: auto-calibration.
6. Uncertainty: a paired bootstrap of the test sample, and ten repeated random splits with refitted models.
7. The target: the same rankings scored on claim counts, on losses capped at 100,000 and on total losses.
8. Recalibration on the calibration sample: a global factor, a Poisson GLM on a spline of the log score, and an
   isotonic step tariff with its minimum step exposure chosen by cross-validation.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import french_mtpl2  # noqa: E402
from tariffcal import StepRecalibration, gini, murphy, poisson_deviance, premium_gini  # noqa: E402

SEED = 19950101
N_BOOT = 500
N_SPLITS = 10
LARGE = 100_000.0
AGE_EDGES = [21, 26, 31, 41, 51, 71]
AGE_LABELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]
BM_EDGES = [51, 61, 71, 81, 101, 126]
BM_LABELS = ["50", "51-60", "61-70", "71-80", "81-100", "101-125", "126+"]
VAGE_EDGES = [1, 6, 11]
VAGE_LABELS = ["0", "1-5", "6-10", "11+"]


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------
def load() -> dict:
    freq, sev = french_mtpl2()
    ids = freq["IDpol"].astype(str).to_numpy()
    sev_ids = sev["IDpol"].astype(str)
    amount = sev["ClaimAmount"].to_numpy(dtype=float)
    per_policy = sev.assign(k=sev_ids).groupby("k", sort=True)["ClaimAmount"]
    total = per_policy.sum().reindex(ids).fillna(0.0).to_numpy()
    capped = sev.assign(k=sev_ids, c=np.minimum(amount, LARGE)).groupby("k", sort=True)["c"].sum()
    capped = capped.reindex(ids).fillna(0.0).to_numpy()
    n_sev = per_policy.size().reindex(ids).fillna(0).to_numpy()
    d = {
        "raw_exposure": freq["Exposure"].to_numpy(dtype=float),
        "raw_claims": freq["ClaimNb"].to_numpy(dtype=float),
        "n_sev": n_sev,
        "amounts": amount,
        "loss": total,
        "loss_capped": capped,
        "vpow": freq["VehPower"].to_numpy(dtype=float),
        "vage": freq["VehAge"].to_numpy(dtype=float),
        "dage": freq["DrivAge"].to_numpy(dtype=float),
        "bm": freq["BonusMalus"].to_numpy(dtype=float),
        "logdens": np.log(freq["Density"].to_numpy(dtype=float)),
        "area": freq["Area"].astype(str).map({c: i + 1 for i, c in enumerate("ABCDEF")}).to_numpy(dtype=float),
        "gas": (freq["VehGas"].astype(str) == "Regular").to_numpy(dtype=float),
        "brand": freq["VehBrand"].cat.codes.to_numpy(),
        "brand_labels": list(freq["VehBrand"].cat.categories),
        "region": freq["Region"].cat.codes.to_numpy(),
        "region_labels": list(freq["Region"].cat.categories),
        "density": freq["Density"].to_numpy(dtype=float),
    }
    d["e"] = np.minimum(d["raw_exposure"], 1.0)
    d["N"] = np.minimum(d["raw_claims"], 4.0)
    return d


# ---------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------
def glm_design(d: dict, idx: np.ndarray, bm_bands: bool) -> np.ndarray:
    cols = [np.ones(idx.size)]
    vp = np.minimum(d["vpow"][idx], 9)
    cols += [(vp == k).astype(float) for k in range(5, 10)]
    va = np.digitize(d["vage"][idx], [1, 11])
    cols += [(va == 1).astype(float), (va == 2).astype(float)]
    da = np.digitize(d["dage"][idx], AGE_EDGES)
    cols += [(da == k).astype(float) for k in range(7) if k != 4]          # reference: 41-50
    bm = np.minimum(d["bm"][idx], 150)
    if bm_bands:
        b = np.digitize(bm, [51, 56, 61, 71, 81, 91, 101, 126])
        cols += [(b == k).astype(float) for k in range(1, 9)]
        cols.append(np.log(bm / 50))
    else:
        cols.append(bm / 100)
    cols += [d["logdens"][idx], d["area"][idx], d["gas"][idx]]
    cols += [(d["brand"][idx] == k).astype(float) for k in range(1, d["brand"].max() + 1)]
    cols += [(d["region"][idx] == k).astype(float) for k in range(1, d["region"].max() + 1)]
    return np.column_stack(cols)


def irls(X: np.ndarray, y: np.ndarray, offset_scale: np.ndarray) -> tuple[np.ndarray, int]:
    """Poisson GLM with log link and offset log(offset_scale)."""
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.sum() / offset_scale.sum())
    for it in range(1, 101):
        m = offset_scale * np.exp(X @ b)
        z = X @ b + (y - m) / m
        b_new = np.linalg.solve(X.T @ (m[:, None] * X), X.T @ (m * z))
        done = np.max(np.abs(b_new - b)) < 1e-10
        b = b_new
        if done:
            return b, it
    return b, it


def gbm_features(d: dict, idx: np.ndarray) -> np.ndarray:
    return np.column_stack([d["vpow"][idx], d["vage"][idx], d["dage"][idx], d["bm"][idx], d["brand"][idx],
                            d["gas"][idx], d["area"][idx], d["logdens"][idx], d["region"][idx]])


def fit_all(d: dict, tr: np.ndarray) -> dict:
    e, N = d["e"][tr], d["N"][tr]
    bA, itA = irls(glm_design(d, tr, False), N, e)
    bB, itB = irls(glm_design(d, tr, True), N, e)
    gbm = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=2000, max_leaf_nodes=31,
                                        min_samples_leaf=200, early_stopping=True, validation_fraction=0.15,
                                        n_iter_no_change=30, random_state=7, categorical_features=[4, 5, 8])
    gbm.fit(gbm_features(d, tr), N / e, sample_weight=e)
    return {"A": bA, "B": bB, "gbm": gbm, "itA": itA, "itB": itB}


def predict(models: dict, d: dict, idx: np.ndarray) -> dict:
    return {"GLM-A": np.exp(glm_design(d, idx, False) @ models["A"]),
            "GLM-B": np.exp(glm_design(d, idx, True) @ models["B"]),
            "GBM": models["gbm"].predict(gbm_features(d, idx))}


def split(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return np.sort(perm[:a]), np.sort(perm[a:b]), np.sort(perm[b:])


# ---------------------------------------------------------------------------------------------------
# Audit tools
# ---------------------------------------------------------------------------------------------------
def poisson_interval(c: float, a: float = 0.05) -> tuple[float, float]:
    lo = stats.chi2.ppf(a / 2, 2 * c) / 2 if c > 0 else 0.0
    hi = stats.chi2.ppf(1 - a / 2, 2 * c + 2) / 2
    return lo, hi


def segment_balance(codes, labels, rate, e, N) -> list[dict]:
    """Premium over claims by segment, relative to the model's own overall level, with a Poisson interval."""
    level = (e * rate).sum() / N.sum()
    out = []
    for k, lab in enumerate(labels):
        m = codes == k
        if not m.any():
            continue
        prem = (e[m] * rate[m]).sum() / level
        c = N[m].sum()
        lo, hi = poisson_interval(c)
        out.append({"segment": lab, "exposure": float(e[m].sum()), "claims": float(c), "ratio": prem / c if c else np.inf,
                    "lo": prem / hi, "hi": prem / lo if lo > 0 else np.inf})
    return out


def decile_balance(rate, e, N, groups=10) -> np.ndarray:
    """Premium over claims within groups of equal exposure, policies sorted by rate; relative to the overall level."""
    level = (e * rate).sum() / N.sum()
    order = np.argsort(rate, kind="stable")
    cum = np.cumsum(e[order]) - 0.5 * e[order]
    g = np.minimum((groups * cum / e.sum()).astype(int), groups - 1)
    lab = np.empty_like(g)
    lab[order] = g
    return np.bincount(lab, weights=e * rate, minlength=groups) / np.bincount(lab, weights=N, minlength=groups) / level


def decile_detail(rate, e, N, groups=10) -> list[dict]:
    """Claims and a 95% Poisson interval for the premium-over-claims ratio of each premium decile."""
    level = (e * rate).sum() / N.sum()
    order = np.argsort(rate, kind="stable")
    cum = np.cumsum(e[order]) - 0.5 * e[order]
    g = np.minimum((groups * cum / e.sum()).astype(int), groups - 1)
    lab = np.empty_like(g)
    lab[order] = g
    prem = np.bincount(lab, weights=e * rate, minlength=groups) / level
    claims = np.bincount(lab, weights=N, minlength=groups)
    out = []
    for j in range(groups):
        lo, hi = poisson_interval(claims[j])
        out.append({"claims": float(claims[j]), "ratio": prem[j] / claims[j], "lo": prem[j] / hi, "hi": prem[j] / lo})
    return out


def hinge_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x] + [np.maximum(x - k, 0.0) for k in knots])


def cv_min_exposure(rate, e, N, grid, folds=5, seed=0) -> tuple[float, dict]:
    """Choose the minimum exposure per isotonic step by K-fold cross-validated Poisson deviance."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, rate.size)
    scores = {}
    for mx in grid:
        dev = 0.0
        for k in range(folds):
            fit, val = fold != k, fold == k
            st = StepRecalibration(mx).fit(rate[fit], e[fit], N[fit])
            dev += poisson_deviance(N[val], e[val] * st.predict(rate[val]))
        scores[mx] = dev / (e.sum() / 1000.0)
    return min(scores, key=scores.get), scores


MODELS = ("GLM-A", "GLM-B", "GBM")


def main() -> None:
    print("Part 3: a calibration audit of GLM and boosted frequency models on freMTPL2.")
    d = load()
    n = d["e"].size

    # -----------------------------------------------------------------------------------------------
    header("1. The data")
    print(f"Policies: {n:,}; claims: {d['raw_claims'].sum():,.0f}; claim amounts: {d['amounts'].size:,}"
          f" totalling {d['amounts'].sum():,.0f} euros.")
    rec = np.mean(d["n_sev"] == d["raw_claims"])
    print(f"Share of policies whose number of claim amounts equals their claim count: {rec:.6f}")
    print(f"Exposure above one year: {np.sum(d['raw_exposure'] > 1):,} policies (maximum {d['raw_exposure'].max():.2f});"
          f" total exposure {d['raw_exposure'].sum():,.1f} raw, {d['e'].sum():,.1f} capped at one year;"
          f" full-year policies {np.mean(d['e'] == 1):.4f}")
    counts = {int(k): int(v) for k, v in zip(*np.unique(d["raw_claims"], return_counts=True))}
    print(f"Claim counts per policy: {counts}")
    big = d["raw_claims"] >= 5
    prof = np.column_stack([d["dage"], d["bm"], d["vage"], d["area"], d["density"]])[big]
    same = np.all(np.abs(prof[:, [1, 3, 4]] - np.array([50, 4, 824])) < 1e-9, axis=1)
    print(f"Policies with five or more claims: {big.sum()}, of which {same.sum()} share bonus-malus 50, area D and"
          f" density 824 (driver ages {sorted({int(x) for x in prof[same, 0]})}, vehicle ages"
          f" {sorted({int(x) for x in prof[same, 2]})}); claims removed by capping counts at four:"
          f" {(d['raw_claims'] - d['N']).sum():.0f}")
    print(f"Claim frequency after cleaning: {d['N'].sum() / d['e'].sum():.4f} claims per policy-year"
          f" ({d['N'].sum():,.0f} claims, {d['e'].sum():,.1f} policy-years)")
    vals, cnts = np.unique(np.round(d["amounts"], 2), return_counts=True)
    top = np.argsort(-cnts, kind="stable")[:3]
    print("Most frequent claim amounts (fixed amounts of the IRSA-IDA convention): " +
          ", ".join(f"{vals[i]:,.2f} ({cnts[i]:,})" for i in top) +
          f"; together {cnts[top].sum() / d['amounts'].size:.4f} of all claims")
    q = np.quantile(d["amounts"], [0.5, 0.9, 0.99])
    print(f"Claim amounts: median {q[0]:,.0f}, 90th percentile {q[1]:,.0f}, 99th percentile {q[2]:,.0f},"
          f" largest {d['amounts'].max():,.0f}")
    large = d["amounts"] > LARGE
    print(f"Claims above {LARGE:,.0f}: {large.sum()}, {d['amounts'][large].sum() / d['amounts'].sum():.4f} of all payments;"
          f" the largest claim alone {d['amounts'].max() / d['amounts'].sum():.4f}")

    # -----------------------------------------------------------------------------------------------
    header("2. Three frequency models on a 60/20/20 split")
    rng = np.random.default_rng(SEED)
    tr, ca, te = split(n, rng)
    for name, idx in (("training", tr), ("calibration", ca), ("test", te)):
        print(f"  {name:<12} {idx.size:,} policies, {d['e'][idx].sum():,.1f} policy-years, {d['N'][idx].sum():,.0f}"
              f" claims, frequency {d['N'][idx].sum() / d['e'][idx].sum():.4f}")
    fr = {nm: d["N"][i].sum() / d["e"][i].sum() for nm, i in (("tr", tr), ("ca", ca), ("te", te))}
    print(f"  test frequency / training frequency {fr['te'] / fr['tr']:.4f};"
          f" calibration / training {fr['ca'] / fr['tr']:.4f}")
    models = fit_all(d, tr)
    print(f"GLM-A (bonus-malus linear): {glm_design(d, tr[:1], False).shape[1]} parameters, IRLS iterations {models['itA']}")
    print(f"GLM-B (bonus-malus in bands): {glm_design(d, tr[:1], True).shape[1]} parameters, IRLS iterations {models['itB']}")
    print(f"GBM (Poisson boosting): {models['gbm'].n_iter_} iterations at early stopping")
    p_tr = predict(models, d, tr)
    print("Training balance (premium / claims): " + ", ".join(
        f"{k} {(d['e'][tr] * p_tr[k]).sum() / d['N'][tr].sum():.6f}" for k in MODELS))

    # -----------------------------------------------------------------------------------------------
    header("3. The test-sample verdict (per 1,000 policy-years)")
    e, N = d["e"][te], d["N"][te]
    per = e.sum() / 1000.0
    p_te = predict(models, d, te)
    print(f"{'model':<7}{'Gini':>8}{'D':>9}{'UNC':>9}{'DSC':>8}{'MCB':>8}{'MCB shape':>11}{'balance':>9}{'prem Gini':>11}")
    verdict = {}
    for k in MODELS:
        r = p_te[k]
        m = murphy(r, e, N)
        shape = murphy(r * N.sum() / (e * r).sum(), e, N)["MCB"]
        verdict[k] = {"gini": gini(r, e, N), "D": m["D"] / per, "DSC": m["DSC"] / per, "MCB": m["MCB"] / per}
        print(f"{k:<7}{gini(r, e, N):8.4f}{m['D'] / per:9.2f}{m['UNC'] / per:9.2f}{m['DSC'] / per:8.2f}"
              f"{m['MCB'] / per:8.3f}{shape / per:11.3f}{(e * r).sum() / N.sum():9.4f}{premium_gini(r, e):11.4f}")
    print("(MCB shape: miscalibration left after rescaling the rate to the test total; the rest is the level)")
    print(f"Isotonic steps on the test sample: " + ", ".join(f"{k} {murphy(p_te[k], e, N)['blocks']}" for k in MODELS))

    # -----------------------------------------------------------------------------------------------
    header("4. Balance by segment, relative to each model's own level (test sample, 95% Poisson intervals)")
    seg_vars = {
        "driver age": (np.digitize(d["dage"][te], AGE_EDGES), AGE_LABELS),
        "bonus-malus": (np.digitize(d["bm"][te], BM_EDGES), BM_LABELS),
        "vehicle age": (np.digitize(d["vage"][te], VAGE_EDGES), VAGE_LABELS),
        "vehicle brand": (d["brand"][te], d["brand_labels"]),
        "area": ((d["area"][te] - 1).astype(int), list("ABCDEF")),
        "region": (d["region"][te], d["region_labels"]),
    }
    seg_out = {}
    for k in MODELS:
        flagged, total_segs = [], 0
        for var, (codes, labels) in seg_vars.items():
            sb = segment_balance(codes, labels, p_te[k], e, N)
            seg_out.setdefault(k, {})[var] = sb
            total_segs += len(sb)
            flagged += [(var, s["segment"], s["ratio"]) for s in sb if s["hi"] < 1 or s["lo"] > 1]
        if k == MODELS[0]:
            print(f"(With {total_segs} segments and 95% intervals, about {0.05 * total_segs:.1f} would exclude 1 by chance"
                  f" for a perfectly calibrated tariff.)")
        print(f"{k}: {len(flagged)} of {total_segs} segments with an interval excluding 1:")
        for var, seg, ratio in flagged:
            print(f"    {var:<14} {seg:<28} {ratio:.3f}")
    for var in ("bonus-malus", "driver age"):
        print(f"{var} ratios:")
        for k in MODELS:
            print(f"  {k:<6} " + "  ".join(f"{s['segment']} {s['ratio']:.3f}" for s in seg_out[k][var]))
    for var in ("bonus-malus",):
        print(f"{var}: claims per band in the test sample: " +
              ", ".join(f"{s['segment']} {s['claims']:.0f}" for s in seg_out['GBM'][var]))

    # -----------------------------------------------------------------------------------------------
    header("5. Balance by premium decile, relative to each model's own level (auto-calibration)")
    dec = {k: decile_balance(p_te[k], e, N) for k in MODELS}
    for k in MODELS:
        print(f"  {k:<6} {np.round(dec[k], 3)}")
    for k in MODELS:
        dd_ = decile_detail(p_te[k], e, N)
        print(f"  {k:<6} claims per decile: " + " ".join(f"{x['claims']:.0f}" for x in dd_) +
              f"; decile 1 interval [{dd_[0]['lo']:.3f}, {dd_[0]['hi']:.3f}]; decile 10 interval"
              f" [{dd_[-1]['lo']:.3f}, {dd_[-1]['hi']:.3f}]; deciles with an interval excluding 1:"
              f" {[j + 1 for j, x in enumerate(dd_) if x['hi'] < 1 or x['lo'] > 1]}")

    # -----------------------------------------------------------------------------------------------
    header(f"6. Uncertainty: a paired bootstrap of the test sample ({N_BOOT} resamples) and {N_SPLITS} repeated splits")
    rb = np.random.default_rng(SEED + 1)
    L, Lc = d["loss"][te], d["loss_capped"][te]
    boot = {f"{k}|{t}": np.empty(N_BOOT) for k in MODELS for t in ("N", "Lc", "L")}
    bdev = {k: np.empty(N_BOOT) for k in MODELS}
    for b in range(N_BOOT):
        i = rb.integers(0, e.size, e.size)
        for k in MODELS:
            r = p_te[k][i]
            boot[f"{k}|N"][b] = gini(r, e[i], N[i])
            boot[f"{k}|Lc"][b] = gini(r, e[i], Lc[i])
            boot[f"{k}|L"][b] = gini(r, e[i], L[i])
            bdev[k][b] = poisson_deviance(N[i], e[i] * r) / (e[i].sum() / 1000.0)
    for k in MODELS:
        g = boot[f"{k}|N"]
        print(f"  {k:<6} Gini {verdict[k]['gini']:.4f}, bootstrap sd {g.std(ddof=1):.4f}, 95% interval"
              f" [{np.quantile(g, 0.025):.4f}, {np.quantile(g, 0.975):.4f}]")
    for a, bm_ in (("GBM", "GLM-A"), ("GLM-B", "GLM-A"), ("GBM", "GLM-B")):
        dg = boot[f"{a}|N"] - boot[f"{bm_}|N"]
        dd = bdev[a] - bdev[bm_]
        print(f"  {a} - {bm_}: Gini {verdict[a]['gini'] - verdict[bm_]['gini']:+.4f} (sd {dg.std(ddof=1):.4f},"
              f" positive in {np.mean(dg > 0):.3f}); deviance {verdict[a]['D'] - verdict[bm_]['D']:+.2f}"
              f" (sd {dd.std(ddof=1):.2f}, negative in {np.mean(dd < 0):.3f})")
    print()
    print("Repeated random 60/20/20 splits, models refitted on each training sample (test-sample values):")
    rs = np.random.default_rng(SEED + 2)
    reps = []
    for s in range(N_SPLITS):
        tr_s, _, te_s = split(n, rs)
        mod = fit_all(d, tr_s)
        es, Ns = d["e"][te_s], d["N"][te_s]
        ps = predict(mod, d, te_s)
        row = {"freq_ratio": (Ns.sum() / es.sum()) / (d["N"][tr_s].sum() / d["e"][tr_s].sum()), "iters": mod["gbm"].n_iter_}
        for k in MODELS:
            r = ps[k]
            row[k] = {"gini": gini(r, es, Ns), "D": poisson_deviance(Ns, es * r) / (es.sum() / 1000.0),
                      "balance": (es * r).sum() / Ns.sum(), "gap": gini(r, es, Ns) - premium_gini(r, es),
                      "gini_L": gini(r, es, d["loss"][te_s]), "gini_Lc": gini(r, es, d["loss_capped"][te_s]),
                      "dec": decile_balance(r, es, Ns), "MCB": murphy(r, es, Ns)["MCB"] / (es.sum() / 1000.0),
                      "MCBs": murphy(r * Ns.sum() / (es * r).sum(), es, Ns)["MCB"] / (es.sum() / 1000.0)}
        reps.append(row)
        print(f"  split {s + 1:2d}: test/train frequency {row['freq_ratio']:.4f}; GBM iterations {row['iters']:3d}; " +
              "; ".join(f"{k} Gini {row[k]['gini']:.4f} D {row[k]['D']:.2f} bal {row[k]['balance']:.4f}" for k in MODELS))
    for k in MODELS:
        g = np.array([r[k]["gini"] for r in reps])
        dv = np.array([r[k]["D"] for r in reps])
        bl = np.array([r[k]["balance"] for r in reps])
        gp = np.array([r[k]["gap"] for r in reps])
        gl = np.array([r[k]["gini_L"] for r in reps])
        print(f"  {k:<6} Gini mean {g.mean():.4f} sd {g.std(ddof=1):.4f} range [{g.min():.4f}, {g.max():.4f}];"
              f" balance range [{bl.min():.4f}, {bl.max():.4f}]; premium Gini below concentration Gini in"
              f" {np.sum(gp > 0)} of {N_SPLITS} splits (gap mean {gp.mean():.4f});"
              f" Gini on total losses range [{gl.min():.4f}, {gl.max():.4f}]")
    for k in MODELS:
        d1 = np.array([r[k]["dec"][0] for r in reps])
        d10 = np.array([r[k]["dec"][-1] for r in reps])
        mc = np.array([r[k]["MCB"] for r in reps])
        mcs = np.array([r[k]["MCBs"] for r in reps])
        print(f"  {k:<6} cheapest decile ratio mean {d1.mean():.3f} range [{d1.min():.3f}, {d1.max():.3f}], below 1 in"
              f" {np.sum(d1 < 1)} of {N_SPLITS}; dearest decile mean {d10.mean():.3f} range [{d10.min():.3f}, {d10.max():.3f}];"
              f" MCB mean {mc.mean():.3f} range [{mc.min():.3f}, {mc.max():.3f}], after rescaling to the test total"
              f" mean {mcs.mean():.3f}")
    best_g = [max(MODELS, key=lambda k: r[k]["gini"]) for r in reps]
    best_d = [min(MODELS, key=lambda k: r[k]["D"]) for r in reps]
    tally = lambda xs: ", ".join(f"{m} {xs.count(m)}" for m in MODELS if m in xs)
    print(f"  best by Gini: {tally(best_g)} of {N_SPLITS}; best by deviance: {tally(best_d)} of {N_SPLITS}")
    fr_all = np.array([r["freq_ratio"] for r in reps])
    print(f"  test/train frequency ratio across splits: range [{fr_all.min():.4f}, {fr_all.max():.4f}], sd {fr_all.std(ddof=1):.4f}")
    dAB = np.array([r["GLM-B"]["D"] - r["GLM-A"]["D"] for r in reps])
    gAB = np.array([r["GLM-B"]["gini"] - r["GLM-A"]["gini"] for r in reps])
    print(f"  GLM-B - GLM-A across splits: Gini {gAB.mean():+.4f} (range {gAB.min():+.4f} to {gAB.max():+.4f});"
          f" deviance {dAB.mean():+.2f} (range {dAB.min():+.2f} to {dAB.max():+.2f})")

    # -----------------------------------------------------------------------------------------------
    header("7. The target: one ranking, three things to rank (test sample)")
    print(f"Test sample payments: {L.sum():,.0f} euros; largest policy total {L.max():,.0f}"
          f" ({L.max() / L.sum():.4f} of the test payments); policies with a claim above {LARGE:,.0f}: {(L > LARGE).sum()}")
    for k in MODELS:
        r = p_te[k]
        parts = []
        for t, y in (("counts", N), ("capped losses", Lc), ("total losses", L)):
            key = {"counts": "N", "capped losses": "Lc", "total losses": "L"}[t]
            g = boot[f"{k}|{key}"]
            parts.append(f"{t} {gini(r, e, y):.4f} (sd {g.std(ddof=1):.4f})")
        print(f"  {k:<6} " + "; ".join(parts))
    j = int(np.argmax(L))
    keep = np.arange(e.size) != j
    for k in MODELS:
        print(f"  {k:<6} Gini on total losses without the single largest policy total:"
              f" {gini(p_te[k][keep], e[keep], L[keep]):.4f} (with it {gini(p_te[k], e, L):.4f});"
              f" that policy's rate is at percentile {np.mean(p_te[k] <= p_te[k][j]):.3f} of the test sample")
    for k in MODELS:
        dg = boot[f"{k}|L"] - boot["GLM-A|L"] if k != "GLM-A" else None
        if dg is not None:
            print(f"  {k} - GLM-A on total losses: positive in {np.mean(dg > 0):.3f} of resamples (sd {dg.std(ddof=1):.4f});"
                  f" on counts: positive in {np.mean(boot[f'{k}|N'] - boot['GLM-A|N'] > 0):.3f}")

    # -----------------------------------------------------------------------------------------------
    header("8. Recalibration on the calibration sample, judged on the test sample (per 1,000 policy-years)")
    ec, Nc = d["e"][ca], d["N"][ca]
    p_ca = predict(models, d, ca)
    grid = (100, 250, 500, 1000, 2000, 4000)
    recal = {}
    print(f"{'model / method':<28}{'steps':>6}{'Gini':>8}{'balance':>9}{'D':>9}{'DSC':>8}{'MCB':>8}{'MCB shape':>11}")
    for k in ("GLM-A", "GBM"):
        rc_, rt_ = p_ca[k], p_te[k]
        x = np.log(rc_)
        knots = np.quantile(x, [0.1, 0.3, 0.5, 0.7, 0.9])
        bspl, _ = irls(hinge_basis(x, knots), Nc, ec * rc_)
        mx, scores = cv_min_exposure(rc_, ec, Nc, grid, seed=SEED)
        st = StepRecalibration(mx).fit(rc_, ec, Nc)
        methods = {
            "as fitted": (rt_, "-"),
            "global factor": (rt_ * Nc.sum() / (ec * rc_).sum(), "-"),
            "spline on log score": (rt_ * np.exp(hinge_basis(np.log(rt_), knots) @ bspl), "-"),
            f"isotonic, >= {mx:,} py": (st.predict(rt_), str(st.steps)),
        }
        for lab, (r, steps) in methods.items():
            m = murphy(r, e, N)
            shape = murphy(r * N.sum() / (e * r).sum(), e, N)["MCB"]
            recal[f"{k}|{lab}"] = dec_r = decile_balance(r, e, N)
            print(f"{k + ', ' + lab:<28}{steps:>6}{gini(r, e, N):8.4f}{(e * r).sum() / N.sum():9.4f}{m['D'] / per:9.2f}"
                  f"{m['DSC'] / per:8.2f}{m['MCB'] / per:8.3f}{shape / per:11.3f}")
        print(f"   cross-validated deviance by minimum step exposure: " +
              ", ".join(f"{g_:,}: {v:.2f}" for g_, v in scores.items()))

    out = {
        "amounts": [round(float(x), 2) for x in np.sort(d["amounts"])],
        "segments": {k: {v: [{kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in s.items()}
                             for s in seg_out[k][v]] for v in ("bonus-malus", "driver age", "vehicle brand", "area")}
                     for k in MODELS},
        "deciles": {k: [round(float(x), 5) for x in v] for k, v in dec.items()},
        "boot_gini": {k: [round(float(x), 6) for x in v] for k, v in boot.items()},
        "point_gini": {f"{k}|{t}": round(gini(p_te[k], e, y), 6) for k in MODELS for t, y in (("N", N), ("Lc", Lc), ("L", L))},
        "splits": [{"freq_ratio": round(float(r["freq_ratio"]), 6),
                    **{k: {m: round(float(v), 6) for m, v in r[k].items() if m in ("gini", "D", "balance", "gap", "gini_L", "gini_Lc")}
                       for k in MODELS}} for r in reps],
    }
    (Path(__file__).parent / "figure_data.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
