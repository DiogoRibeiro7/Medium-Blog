"""Companion to the actuarial-tariff-calibration series: testing the winner's-curse conjecture of Part 4.

Every number quoted in the article is printed by this script. Run:

    python curse_numbers.py > output.txt      # several minutes, single-threaded; writes figure_data.json

The conjecture (Part 4): if competitors in a market use high-variance learners with similar features, the winner's
curse among them should require larger loadings than in a market of competitors using stable models, even if the
high-variance learners rank better individually.

Design. Two portfolios: freMTPL2 (France, 677,991 policies; the split of Parts 3 and 4) and beMTPL97 (Belgium,
163,212 policies; its own 60/20/20 split). Each insurer fits its tariff on its own book: a random slice of the
training sample, disjoint from every other insurer's. Three model classes:
    GLM   the banded Poisson GLM of Part 3 (GLM-B for France; the analogous design for Belgium), small ridge penalty
    GBM   Poisson gradient boosting with the settings of Part 3
    BAG   the average of five such boosting models, each fitted on a random 70% of the insurer's book (subagging;
          resampling without replacement keeps early stopping honest)
Every insurer rescales its rate to collect exactly the claims of its own book. Policyholders of the test sample
choose by the rule of Part 4: probability proportional to (premium) ** -beta, with beta = infinity meaning the
cheapest quote wins. Each experiment is repeated on R random partitions of the training sample.

Sections, for each portfolio
1. Data and models: sizes, and the single-insurer Gini and deviance of each class by book size.
2. Homogeneous markets: K = 2, 3, 5, 10 insurers of one class, books of equal size. What the market collects, the
   dispersion of the quotes, and the approximation 1 - sigma * c_K (c_K the expected maximum of K standard normals).
3. Rate reviews in homogeneous markets: the average level policyholders pay once every insurer breaks even.
4. Mixed markets: half boosting, half GLM, K = 2, 4, 10.
5. Book size: one boosting insurer against one GLM insurer as the book grows (both assignments of the two books).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, stats
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "part3"))
import audit_numbers as p3  # noqa: E402
from data import belgian_mtpl97  # noqa: E402
from tariffcal import gini, poisson_deviance  # noqa: E402

SEED = 20261111
R = 6
KS = (2, 3, 5, 10)
K_MIXED = (2, 4, 10)
BAG = 5
SUBSAMPLE = 0.7
ROUNDS = 40
RIDGE = 1.0
INF = np.inf
CLASSES = ("GLM", "GBM", "BAG")
PORTFOLIOS = {
    "freMTPL2": {"slice": 40_000, "sizes": (20_000, 50_000, 100_000, 200_000)},
    "beMTPL97": {"slice": 9_000, "sizes": (5_000, 12_000, 24_000, 48_000)},
}


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def lognormal_collect(k: int, sigma: float) -> float:
    """What the market collects per unit of claims if each of k quotes is the cost times exp(sigma Z - sigma^2/2).

    The quotes are then unbiased in money (their arithmetic mean is the cost), and the cheapest of the k is the cost
    times exp(-sigma M_k - sigma^2/2), with M_k the maximum of k independent standard normals.
    """
    f = lambda x: np.exp(-sigma * x) * k * stats.norm.pdf(x) * stats.norm.cdf(x) ** (k - 1)  # noqa: E731
    return float(integrate.quad(f, -12.0, 12.0, limit=200)[0] * np.exp(-sigma ** 2 / 2))   # mass outside is below 1e-30


def c_k(k: int) -> float:
    """Expected maximum of k independent standard normal variables."""
    f = lambda x: x * k * stats.norm.pdf(x) * stats.norm.cdf(x) ** (k - 1)  # noqa: E731
    return float(integrate.quad(f, -np.inf, np.inf)[0])


# ---------------------------------------------------------------------------------------------------
# Portfolios
# ---------------------------------------------------------------------------------------------------
def french() -> dict:
    d = p3.load()
    rng = np.random.default_rng(p3.SEED)
    tr, ca, te = p3.split(d["e"].size, rng)
    return {"name": "freMTPL2", "e": d["e"], "N": d["N"], "tr": tr, "ca": ca, "te": te,
            "X_glm": lambda idx: p3.glm_design(d, idx, True), "X_gbm": lambda idx: p3.gbm_features(d, idx),
            "cat": [4, 5, 8]}


def belgian() -> dict:
    df = belgian_mtpl97()
    e = df["expo"].to_numpy(dtype=float)
    N = df["nclaims"].to_numpy(dtype=float)
    age = df["ageph"].to_numpy(dtype=float)
    bm = df["bm"].to_numpy(dtype=float)
    pw = df["power"].to_numpy(dtype=float)
    agec = df["agec"].to_numpy(dtype=float)
    cov = df["coverage"].astype(str).map({"TPL": 0, "TPL+": 1, "TPL++": 2}).to_numpy(dtype=float)
    diesel = (df["fuel"].astype(str) == "diesel").to_numpy(dtype=float)
    work = (df["use"].astype(str) == "work").to_numpy(dtype=float)
    fleet = (df["fleet"].astype(str) == "1").to_numpy(dtype=float)
    region = (df["postcode"].to_numpy(dtype=int) // 1000).astype(float)
    lon, lat = df["long"].to_numpy(dtype=float), df["lat"].to_numpy(dtype=float)

    def X_glm(idx):
        cols = [np.ones(idx.size)]
        a = np.digitize(age[idx], [26, 31, 41, 51, 61, 71])
        cols += [(a == k).astype(float) for k in range(7) if k != 3]          # reference 41-50
        b = np.digitize(bm[idx], [1, 4, 7, 10, 14])
        cols += [(b == k).astype(float) for k in range(1, 6)]
        cols.append(bm[idx] / 10)
        p_ = np.digitize(pw[idx], [50, 66, 92])
        cols += [(p_ == k).astype(float) for k in range(1, 4)]
        cols.append(np.log(pw[idx]))
        v = np.digitize(agec[idx], [3, 6, 10, 15])
        cols += [(v == k).astype(float) for k in range(1, 5)]
        cols += [(cov[idx] == 1).astype(float), (cov[idx] == 2).astype(float), diesel[idx], work[idx], fleet[idx]]
        cols += [(region[idx] == k).astype(float) for k in range(2, 10)]
        return np.column_stack(cols)

    def X_gbm(idx):
        return np.column_stack([age[idx], bm[idx], pw[idx], agec[idx], cov[idx], diesel[idx], work[idx], fleet[idx],
                                lon[idx], lat[idx]])

    rng = np.random.default_rng(SEED + 7)
    tr, ca, te = p3.split(e.size, rng)
    return {"name": "beMTPL97", "e": e, "N": N, "tr": tr, "ca": ca, "te": te, "X_glm": X_glm, "X_gbm": X_gbm,
            "cat": [4]}


# ---------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------
def irls_ridge(X, y, expo, lam=RIDGE):
    """Poisson GLM with log link, offset log(exposure) and a ridge penalty on every coefficient but the intercept."""
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.sum() / expo.sum())
    D = np.eye(X.shape[1]) * lam
    D[0, 0] = 0.0
    for _ in range(100):
        m = expo * np.exp(X @ b)
        z = X @ b + (y - m) / m
        b_new = np.linalg.solve(X.T @ (m[:, None] * X) + D, X.T @ (m * z))
        done = np.max(np.abs(b_new - b)) < 1e-9
        b = b_new
        if done:
            break
    return b


def gbm(P, X, y, w, seed=7):
    g = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=2000, max_leaf_nodes=31,
                                      min_samples_leaf=200, early_stopping=True, validation_fraction=0.15,
                                      n_iter_no_change=30, random_state=seed, categorical_features=P["cat"])
    return g.fit(X, y, sample_weight=w)


def fit_insurer(P, idx, cls, seed) -> dict:
    """Fit one insurer's tariff on its book idx; return its rates on the calibration and test samples.

    The rate is rescaled so that it collects exactly the claims of the insurer's own book.
    """
    e, N = P["e"], P["N"]
    if cls == "GLM":
        b = irls_ridge(P["X_glm"](idx), N[idx], e[idx])
        pred = lambda j: np.exp(P["X_glm"](j) @ b)  # noqa: E731
    elif cls == "GBM":
        m = gbm(P, P["X_gbm"](idx), N[idx] / e[idx], e[idx])
        pred = lambda j: m.predict(P["X_gbm"](j))  # noqa: E731
    else:
        rng = np.random.default_rng(seed)
        ms = []
        for k in range(BAG):
            bi = np.sort(rng.choice(idx, int(SUBSAMPLE * idx.size), replace=False))
            ms.append(gbm(P, P["X_gbm"](bi), N[bi] / e[bi], e[bi], seed=7 + k))
        pred = lambda j: np.mean([m_.predict(P["X_gbm"](j)) for m_ in ms], axis=0)  # noqa: E731
    own = pred(idx)
    c = N[idx].sum() / (e[idx] * own).sum()
    return {"ca": c * pred(P["ca"]), "te": c * pred(P["te"])}


# ---------------------------------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------------------------------
def market(rates: np.ndarray, e, N, beta, levels=None) -> dict:
    """K insurers (rows of rates); returns shares, loss ratios and what the market collects."""
    K = rates.shape[0]
    L = np.ones(K) if levels is None else np.asarray(levels)
    Pm = e * rates * L[:, None]
    if np.isinf(beta):
        w = (Pm == Pm.min(axis=0)).astype(float)
        w /= w.sum(axis=0)
    else:
        lw = -beta * np.log(Pm)
        lw -= lw.max(axis=0)
        w = np.exp(lw)
        w /= w.sum(axis=0)
    prem = (w * Pm).sum(axis=1)
    clm = (w * N).sum(axis=1)
    expo = (w * e).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(prem > 0, clm / prem, np.nan)
    return {"share": expo / e.sum(), "lr": lr, "collected": float(prem.sum() / N.sum()), "prem": prem, "clm": clm}


def sigma_hat(rates: np.ndarray, e) -> float:
    """Exposure-weighted dispersion of log quotes across insurers, per policy (unbiased in K)."""
    q = np.log(rates)
    K = q.shape[0]
    v = ((q - q.mean(axis=0)) ** 2).sum(axis=0) / (K - 1)
    return float(np.sqrt(np.average(v, weights=e)))


def review(rates_ca: np.ndarray, e, N, beta, rounds=ROUNDS) -> tuple[np.ndarray, bool]:
    """Each round every insurer with business multiplies its level by its own loss ratio.

    Returns the levels and whether the reviews have settled: every insurer still in the market within 1% of a loss
    ratio of 1 after the last round.
    """
    L = np.ones(rates_ca.shape[0])
    for _ in range(rounds):
        m = market(rates_ca, e, N, beta, L)
        active = m["share"] > 1e-4
        L = np.where(active, L * np.nan_to_num(m["lr"], nan=1.0), L)
    m = market(rates_ca, e, N, beta, L)
    active = m["share"] > 1e-4
    return L, bool(np.all(np.abs(m["lr"][active] - 1) < 0.01))


def summary(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.3f} [{x.min():.3f}, {x.max():.3f}]"


# ---------------------------------------------------------------------------------------------------
def run(P: dict, cfg: dict, CK: dict) -> dict:
    e, N = P["e"], P["N"]
    et, Nt = e[P["te"]], N[P["te"]]
    ec, Nc = e[P["ca"]], N[P["ca"]]
    per = et.sum() / 1000.0
    name = P["name"]
    header(f"{name}: data and design")
    print(f"Policies {e.size:,}; claims {N.sum():,.0f}; exposure {e.sum():,.1f} policy-years; frequency {N.sum() / e.sum():.4f}")
    print(f"Training pool {P['tr'].size:,}; calibration {P['ca'].size:,}; test {P['te'].size:,} policies"
          f" ({et.sum():,.1f} policy-years, {Nt.sum():,.0f} claims)")
    print(f"Ten disjoint books of {cfg['slice']:,} policies per partition; book sizes for the head-to-head:"
          f" {', '.join(f'{s:,}' for s in cfg['sizes'])}; {R} partitions")
    rng = np.random.default_rng(SEED + (0 if name == "freMTPL2" else 100))

    homo = {c: {K: {"coll": [], "coll10": [], "sig": [], "formula": [], "lognormal": [], "lrmax": [], "level_inf": [],
                    "level_10": [], "conv_inf": [], "conv_10": [], "sig_dec": [], "coll_dec": []}
                for K in KS} for c in CLASSES}
    single = {c: {"gini": [], "dev": []} for c in CLASSES}
    mixed = {K: {b: {"shareG": [], "lrG": [], "lrL": [], "lvG": [], "lvL": [], "endG": [], "conv": []}
                 for b in ("inf", "10")}
             for K in K_MIXED}
    sizes = {n: {"gini": {c: [] for c in ("GLM", "GBM")}, "dev": {c: [] for c in ("GLM", "GBM")},
                 "twin_sig": {c: [] for c in ("GLM", "GBM")}, "twin_coll": {c: [] for c in ("GLM", "GBM")},
                 "shareG": [], "lrG": [], "lrL": [], "endG": [], "lvG": [], "lvL": [], "coll": [], "conv": []}
             for n in cfg["sizes"]}

    for r in range(R):
        perm = rng.permutation(P["tr"])
        books = [np.sort(perm[j * cfg["slice"]:(j + 1) * cfg["slice"]]) for j in range(10)]
        fits = {c: [fit_insurer(P, b, c, seed=SEED + 1000 * r + j) for j, b in enumerate(books)] for c in CLASSES}
        for c in CLASSES:
            for f in fits[c]:
                single[c]["gini"].append(gini(f["te"], et, Nt))
                single[c]["dev"].append(poisson_deviance(Nt, et * f["te"]) / per)
            for K in KS:
                Rt = np.array([f["te"] for f in fits[c][:K]])
                Rc = np.array([f["ca"] for f in fits[c][:K]])
                m = market(Rt, et, Nt, INF)
                m10 = market(Rt, et, Nt, 10.0)
                s_ = sigma_hat(Rt, et)
                h = homo[c][K]
                h["coll"].append(m["collected"])
                h["coll10"].append(m10["collected"])
                h["sig"].append(s_)
                if K == 10:
                    q = np.log(Rt)
                    mq = q.mean(axis=0)
                    vq = ((q - mq) ** 2).sum(axis=0) / (K - 1)
                    order = np.argsort(mq, kind="stable")
                    cum = np.cumsum(et[order]) - 0.5 * et[order]
                    dec = np.empty(et.size, dtype=int)
                    dec[order] = np.minimum((10 * cum / et.sum()).astype(int), 9)
                    cheapest = (et * Rt).min(axis=0)
                    h["sig_dec"].append([float(np.sqrt(np.average(vq[dec == j], weights=et[dec == j]))) for j in range(10)])
                    h["coll_dec"].append([float(cheapest[dec == j].sum() / Nt[dec == j].sum()) for j in range(10)])
                h["formula"].append(1 - s_ * CK[K])
                h["lognormal"].append(lognormal_collect(K, s_))
                h["lrmax"].append(np.nanmax(m["lr"]))
                for beta, key in ((INF, "inf"), (10.0, "10")):
                    L, ok = review(Rc, ec, Nc, beta)
                    mt = market(Rt, et, Nt, beta, L)
                    h[f"level_{key}"].append(float((mt["share"] * L).sum()))
                    h[f"conv_{key}"].append(ok)
        for K in K_MIXED:
            types = ["GBM" if j % 2 == 0 else "GLM" for j in range(K)]
            Rt = np.array([fits[t][j]["te"] for j, t in enumerate(types)])
            Rc = np.array([fits[t][j]["ca"] for j, t in enumerate(types)])
            g = np.array([t == "GBM" for t in types])
            for beta, key in ((INF, "inf"), (10.0, "10")):
                m = market(Rt, et, Nt, beta)
                L, ok = review(Rc, ec, Nc, beta)
                mt = market(Rt, et, Nt, beta, L)
                mm = mixed[K][key]
                mm["conv"].append(ok)
                mm["shareG"].append(m["share"][g].sum())
                mm["lrG"].append(m["clm"][g].sum() / m["prem"][g].sum())
                mm["lrL"].append(m["clm"][~g].sum() / m["prem"][~g].sum())
                mm["endG"].append(mt["share"][g].sum())
                sg, sl = mt["share"][g].sum(), mt["share"][~g].sum()
                mm["lvG"].append(float((mt["share"][g] * L[g]).sum() / sg) if sg > 1e-6 else np.nan)
                mm["lvL"].append(float((mt["share"][~g] * L[~g]).sum() / sl) if sl > 1e-6 else np.nan)
        perm2 = rng.permutation(P["tr"])
        for n in cfg["sizes"]:
            A, B = np.sort(perm2[:n]), np.sort(perm2[n:2 * n])
            f = {(c, k): fit_insurer(P, idx, c, seed=SEED + 7 * r + n) for c in ("GLM", "GBM") for k, idx in (("A", A), ("B", B))}
            z = sizes[n]
            for c in ("GLM", "GBM"):
                for k in ("A", "B"):
                    z["gini"][c].append(gini(f[(c, k)]["te"], et, Nt))
                    z["dev"][c].append(poisson_deviance(Nt, et * f[(c, k)]["te"]) / per)
                Rt = np.array([f[(c, "A")]["te"], f[(c, "B")]["te"]])
                z["twin_sig"][c].append(sigma_hat(Rt, et))
                z["twin_coll"][c].append(market(Rt, et, Nt, INF)["collected"])
            for gk, lk in (("A", "B"), ("B", "A")):
                Rt = np.array([f[("GBM", gk)]["te"], f[("GLM", lk)]["te"]])
                Rc = np.array([f[("GBM", gk)]["ca"], f[("GLM", lk)]["ca"]])
                m = market(Rt, et, Nt, INF)
                L, ok = review(Rc, ec, Nc, INF)
                mt = market(Rt, et, Nt, INF, L)
                z["coll"].append(m["collected"])
                z["conv"].append(ok)
                z["shareG"].append(m["share"][0])
                z["lrG"].append(m["lr"][0])
                z["lrL"].append(m["lr"][1])
                z["endG"].append(mt["share"][0])
                z["lvG"].append(L[0])
                z["lvL"].append(L[1])

    # -----------------------------------------------------------------------------------------------
    header(f"{name}: one insurer alone (test sample; Gini on claim counts, Poisson deviance per 1,000 policy-years)")
    for c in CLASSES:
        print(f"  {c}, books of {cfg['slice']:,}: Gini {summary(single[c]['gini'])}; deviance {summary(single[c]['dev'])}")
    for n in cfg["sizes"]:
        z = sizes[n]
        print(f"  books of {n:>7,}: Gini GLM {summary(z['gini']['GLM'])}, GBM {summary(z['gini']['GBM'])};"
              f" deviance GLM {np.mean(z['dev']['GLM']):.2f}, GBM {np.mean(z['dev']['GBM']):.2f};"
              f" GBM above GLM in Gini in {np.sum(np.array(z['gini']['GBM']) > np.array(z['gini']['GLM']))}"
              f" of {len(z['gini']['GBM'])} fits, below in deviance in"
              f" {np.sum(np.array(z['dev']['GBM']) < np.array(z['dev']['GLM']))}")

    header(f"{name}: homogeneous markets, cheapest wins (test sample; mean [min, max] over {R} partitions)")
    print("  c_K, the expected maximum of K standard normals: " + ", ".join(f"K={K} {CK[K]:.3f}" for K in KS))
    for c in CLASSES:
        for K in KS:
            h = homo[c][K]
            err = np.array(h["formula"]) - np.array(h["coll"])
            err2 = np.array(h["lognormal"]) - np.array(h["coll"])
            print(f"  {c} K={K:>2}: collected / claims {summary(h['coll'])}; sigma {summary(h['sig'])};"
                  f" linear 1 - sigma c_K {np.mean(h['formula']):.3f} (minus actual {err.mean():+.3f});"
                  f" lognormal {np.mean(h['lognormal']):.3f} (minus actual {err2.mean():+.3f});"
                  f" with beta 10 {np.mean(h['coll10']):.3f}; worst insurer's loss ratio {np.mean(h['lrmax']):.3f}")
    for key, lab in (("formula", "linear 1 - sigma c_K"), ("lognormal", "lognormal")):
        for ks, kl in (((2, 3), "K = 2, 3"), ((5, 10), "K = 5, 10")):
            allf = np.concatenate([np.array(homo[c][K][key]) - np.array(homo[c][K]["coll"]) for c in CLASSES for K in ks])
            print(f"  {lab} minus actual, {kl} ({allf.size} markets): mean {allf.mean():+.4f},"
                  f" largest absolute {np.abs(allf).max():.4f}")

    print("  Where the disagreement lives (K=10, cheapest wins): by decile of the policy's average log rate across the ten"
          " insurers, cheapest first, mean over partitions")
    for c in CLASSES:
        sd = np.mean(homo[c][10]["sig_dec"], axis=0)
        cd = np.mean(homo[c][10]["coll_dec"], axis=0)
        print(f"    {c} sigma by decile: " + " ".join(f"{x:.3f}" for x in sd))
        print(f"    {c} collected / claims by decile: " + " ".join(f"{x:.3f}" for x in cd))

    header(f"{name}: rate reviews in homogeneous markets: average level paid once every insurer has reviewed {ROUNDS} times")
    for c in CLASSES:
        print(f"  {c}: " + "; ".join(f"K={K} cheapest wins {np.mean(homo[c][K]['level_inf']):.3f}"
                                     f" (settled {np.sum(homo[c][K]['conv_inf'])}/{R}), beta 10"
                                     f" {np.mean(homo[c][K]['level_10']):.3f} (settled {np.sum(homo[c][K]['conv_10'])}/{R})"
                                     for K in KS))
    for c in CLASSES:
        print(f"  {c}: break-even common loading 1 / (collected / claims), cheapest wins: " +
              ", ".join(f"K={K} {np.mean(1 / np.array(homo[c][K]['coll'])):.3f}" for K in KS))
    for K in KS:
        gl, gb, bg = (np.array(homo[c][K]["level_10"]) for c in CLASSES)
        print(f"  K={K}, beta 10: GBM level above GLM level in {np.sum(gb > gl)} of {R} partitions; BAG below GBM in"
              f" {np.sum(bg < gb)} of {R}")

    header(f"{name}: mixed markets, half boosting and half GLM (books of {cfg['slice']:,})")
    for K in K_MIXED:
        for key in ("inf", "10"):
            mm = mixed[K][key]
            print(f"  K={K:>2} beta {key:>3}: one period: boosting share {summary(mm['shareG'])}, LR boosting"
                  f" {summary(mm['lrG'])}, LR GLM {summary(mm['lrL'])}; boosting LR higher in"
                  f" {np.sum(np.array(mm['lrG']) > np.array(mm['lrL']))} of {R}; after reviews: boosting share"
                  f" {summary(mm['endG'])}, level boosting {np.nanmean(mm['lvG']):.3f}, GLM {np.nanmean(mm['lvL']):.3f}"
                  f" (settled {np.sum(mm['conv'])}/{R})")

    header(f"{name}: one boosting insurer against one GLM insurer as the books grow (cheapest wins; both assignments)")
    for n in cfg["sizes"]:
        z = sizes[n]
        lrg, lrl = np.array(z["lrG"]), np.array(z["lrL"])
        print(f"  books of {n:>7,}: twins sigma GLM {np.mean(z['twin_sig']['GLM']):.3f}, GBM {np.mean(z['twin_sig']['GBM']):.3f};"
              f" twins collect GLM {np.mean(z['twin_coll']['GLM']):.3f}, GBM {np.mean(z['twin_coll']['GBM']):.3f} |"
              f" boosting against GLM: share {summary(z['shareG'])}, LR {np.mean(lrg):.3f} against {np.mean(lrl):.3f},"
              f" boosting higher in {np.sum(lrg > lrl)} of {lrg.size}; after reviews boosting share {summary(z['endG'])},"
              f" below 0.5 in {np.sum(np.array(z['endG']) < 0.5)} of {lrg.size} (settled {np.sum(z['conv'])});"
              f" the head-to-head market collects {np.mean(z['coll']):.3f}")

    for c in ("GLM", "GBM"):
        ln_n = np.log(np.array(cfg["sizes"], dtype=float))
        ln_s = np.log([np.mean(sizes[n]["twin_sig"][c]) for n in cfg["sizes"]])
        slope = np.polyfit(ln_n, ln_s, 1)[0]
        print(f"  {c}: twins' sigma falls like n to the power {slope:.2f} as the book size n grows"
              f" (least-squares slope on the log scale over the four sizes; -0.5 is the parametric rate)")

    return {"homo": {c: {str(K): {k: [round(float(x), 6) for x in np.ravel(v)] for k, v in homo[c][K].items()} for K in KS}
                     for c in CLASSES},
            "sizes": {str(n): {"shareG": [round(float(x), 6) for x in sizes[n]["shareG"]],
                               "endG": [round(float(x), 6) for x in sizes[n]["endG"]],
                               "lrG": [round(float(x), 6) for x in sizes[n]["lrG"]],
                               "lrL": [round(float(x), 6) for x in sizes[n]["lrL"]],
                               "twin_coll": {c: [round(float(x), 6) for x in v] for c, v in sizes[n]["twin_coll"].items()},
                               "gini": {c: [round(float(x), 6) for x in v] for c, v in sizes[n]["gini"].items()}}
                      for n in cfg["sizes"]},
            "mixed": {str(K): {b: {k: [round(float(x), 6) for x in v] for k, v in mixed[K][b].items()}
                               for b in ("inf", "10")} for K in K_MIXED},
            "single": {c: {k: [round(float(x), 6) for x in v] for k, v in single[c].items()} for c in CLASSES}}


def main() -> None:
    print("Companion: testing the winner's-curse conjecture of Part 4 with many insurers and two portfolios.")
    CK = {K: c_k(K) for K in KS}
    header("The lognormal prediction: what a market of K unbiased insurers collects per unit of claims, by sigma")
    for sg in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        print(f"  sigma {sg:.2f}: " + ", ".join(f"K={K} {lognormal_collect(K, sg):.3f}" for K in KS))
    print("  For comparison, the expected Poisson deviance of a single tariff whose log rate errs by sigma rises by about"
          " sigma^2 times the expected claims (second order), while the shortfall of the cheapest of K such tariffs is"
          " about sigma times c_K of the premium (first order).")
    out = {}
    out["freMTPL2"] = run(french(), PORTFOLIOS["freMTPL2"], CK)
    out["beMTPL97"] = run(belgian(), PORTFOLIOS["beMTPL97"], CK)
    out["c_k"] = {str(K): CK[K] for K in KS}
    (HERE / "figure_data.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
