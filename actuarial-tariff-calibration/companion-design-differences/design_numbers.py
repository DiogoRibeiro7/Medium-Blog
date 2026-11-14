"""Fourth companion to the actuarial-tariff-calibration series: competing insurers whose tariffs differ in design.

Every number quoted in the article is printed by this script. Run:

    python design_numbers.py > output.txt     # about a quarter of an hour, single-threaded; writes figure_data.json

The first companion measured the disagreement between competing insurers' prices with twin fits: the same model
fitted to two disjoint books. It took as a working assumption, labelled as such, that this disagreement is closer to
a lower bound than to what real competitors face, because real competitors also differ in their rating factors and
in the design of their models. This script measures the difference and asks what it changes in the market.

Designs. Five ways of building a claim-frequency tariff; each insurer fits its design on its own book.
    GLM            the banded Poisson GLM of the earlier companions (small ridge penalty)
    linear BM      the same GLM with the bonus-malus level entered linearly instead of in bands
    no territory   the same GLM without its territorial factors (France: region, population density and area code;
                   Belgium: the region given by the first digit of the postcode)
    GBM            Poisson gradient boosting with the settings of the earlier companions
    other GBM      Poisson gradient boosting tuned differently: shallower trees, a faster learning rate, an L2
                   penalty and another validation split
Pairs of competitors: the GLM against each of the other designs, the other GBM against the GBM, and the twins (GLM
against GLM, GBM against GBM). Books, partitions and seeds are those of the second and third companions, so the GLM
and GBM fits are theirs and the plays between them reproduce their tables (checked at the end).

Decomposition. For two tariffs fitted on independent books, the squared dispersion of their log quotes is the sum of
a data part, the mean of the two designs' squared twin dispersions, and a design part, the remainder. The data part
falls as the books grow; the design part is the disagreement that more data would not remove. Dispersions are root
mean squares over plays, so that the two parts add up.

Sections, for each portfolio
0. The mechanism, with known costs: disagreement that is noise against disagreement that is information.
1. The designs alone: Gini, deviance and twin dispersion by book size.
2. Disagreement between designs: dispersion, its data and design parts, and how each changes with the book size.
3. Noise or information: the best blend of the two tariffs, fitted on the calibration sample, scored on the test.
4. One period, no price reaction: what the market collects, the lognormal prediction from the dispersion, and the
   loss ratio of each insurer.
5. Rate reviews (beta 10): the loading each insurer applies and its share once both have reviewed 40 times.
6. The price game with curse-aware pricing (beta 10): one level, and ten levels by market position.
7. Checks.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "companion-segment-pricing"))
import segment_numbers as g3  # noqa: E402

g2 = g3.g2
c1 = g2.c1
p3 = c1.p3
from data import belgian_mtpl97  # noqa: E402
from tariffcal import gini, poisson_deviance  # noqa: E402

header = c1.header
SEED = g2.SEED                      # same partitions and books as the second and third companions
R = g2.R
BETA = 10.0
INF = np.inf
DESIGNS = ("GLM", "linear BM", "no territory", "GBM", "other GBM")
PAIRS = (("GLM", "GLM"), ("GLM", "linear BM"), ("GLM", "no territory"), ("GLM", "GBM"), ("GBM", "other GBM"),
         ("GBM", "GBM"))
OTHER_GBM = {"learning_rate": 0.1, "max_leaf_nodes": 8, "min_samples_leaf": 100, "l2_regularization": 1.0,
             "random_state": 11}
SIGMAS = (0.1, 0.2, 0.3)
PORTFOLIOS = g2.PORTFOLIOS
GAMES = ("one level", "market position")


def label(a: str, b: str) -> str:
    return f"{a} against {b}"


def summary(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.3f} [{x.min():.3f}, {x.max():.3f}]"


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


# ---------------------------------------------------------------------------------------------------
# Portfolios and designs
# ---------------------------------------------------------------------------------------------------
def french() -> dict:
    P = c1.french()
    d = p3.load()

    def X(idx, territory=True):
        cols = [np.ones(idx.size)]
        vp = np.minimum(d["vpow"][idx], 9)
        cols += [(vp == k).astype(float) for k in range(5, 10)]
        va = np.digitize(d["vage"][idx], [1, 11])
        cols += [(va == 1).astype(float), (va == 2).astype(float)]
        da = np.digitize(d["dage"][idx], p3.AGE_EDGES)
        cols += [(da == k).astype(float) for k in range(7) if k != 4]
        bm = np.minimum(d["bm"][idx], 150)
        b = np.digitize(bm, [51, 56, 61, 71, 81, 91, 101, 126])
        cols += [(b == k).astype(float) for k in range(1, 9)]
        cols.append(np.log(bm / 50))
        if territory:
            cols += [d["logdens"][idx], d["area"][idx]]
        cols.append(d["gas"][idx])
        cols += [(d["brand"][idx] == k).astype(float) for k in range(1, d["brand"].max() + 1)]
        if territory:
            cols += [(d["region"][idx] == k).astype(float) for k in range(1, d["region"].max() + 1)]
        return np.column_stack(cols)

    P["X"] = {"GLM": P["X_glm"], "linear BM": lambda idx: p3.glm_design(d, idx, False),
              "no territory": lambda idx: X(idx, territory=False)}
    P["same_as_before"] = bool(np.array_equal(X(P["tr"]), P["X_glm"](P["tr"])))
    P["territory"] = "region, population density and area code"
    P["bm"] = np.minimum(d["bm"], 150)
    P["bm_edges"] = [51, 61, 71, 81, 101]
    return P


def belgian() -> dict:
    P = c1.belgian()
    df = belgian_mtpl97()
    age = df["ageph"].to_numpy(dtype=float)
    bm = df["bm"].to_numpy(dtype=float)
    pw = df["power"].to_numpy(dtype=float)
    agec = df["agec"].to_numpy(dtype=float)
    cov = df["coverage"].astype(str).map({"TPL": 0, "TPL+": 1, "TPL++": 2}).to_numpy(dtype=float)
    diesel = (df["fuel"].astype(str) == "diesel").to_numpy(dtype=float)
    work = (df["use"].astype(str) == "work").to_numpy(dtype=float)
    fleet = (df["fleet"].astype(str) == "1").to_numpy(dtype=float)
    region = (df["postcode"].to_numpy(dtype=int) // 1000).astype(float)

    def X(idx, linear=False, territory=True):
        cols = [np.ones(idx.size)]
        a = np.digitize(age[idx], [26, 31, 41, 51, 61, 71])
        cols += [(a == k).astype(float) for k in range(7) if k != 3]
        if not linear:
            b = np.digitize(bm[idx], [1, 4, 7, 10, 14])
            cols += [(b == k).astype(float) for k in range(1, 6)]
        cols.append(bm[idx] / 10)
        p_ = np.digitize(pw[idx], [50, 66, 92])
        cols += [(p_ == k).astype(float) for k in range(1, 4)]
        cols.append(np.log(pw[idx]))
        v = np.digitize(agec[idx], [3, 6, 10, 15])
        cols += [(v == k).astype(float) for k in range(1, 5)]
        cols += [(cov[idx] == 1).astype(float), (cov[idx] == 2).astype(float), diesel[idx], work[idx], fleet[idx]]
        if territory:
            cols += [(region[idx] == k).astype(float) for k in range(2, 10)]
        return np.column_stack(cols)

    P["X"] = {"GLM": P["X_glm"], "linear BM": lambda idx: X(idx, linear=True),
              "no territory": lambda idx: X(idx, territory=False)}
    P["same_as_before"] = bool(np.array_equal(X(P["tr"]), P["X_glm"](P["tr"])))
    P["territory"] = "region of the postcode"
    P["bm"] = bm
    P["bm_edges"] = [1, 4, 7, 10, 14]
    return P


def gbm_other(P, X, y, w):
    g = HistGradientBoostingRegressor(loss="poisson", max_iter=2000, early_stopping=True, validation_fraction=0.15,
                                      n_iter_no_change=30, categorical_features=P["cat"], **OTHER_GBM)
    return g.fit(X, y, sample_weight=w)


def fit_design(P: dict, idx: np.ndarray, design: str, seed: int) -> dict:
    """One insurer's tariff of the given design, fitted on its book idx and rescaled to collect the book's claims;
    returns its rates on the calibration and test samples."""
    if design in ("GLM", "GBM"):
        return c1.fit_insurer(P, idx, design, seed)        # exactly the fits of the earlier companions
    e, N = P["e"], P["N"]
    if design == "other GBM":
        m = gbm_other(P, P["X_gbm"](idx), N[idx] / e[idx], e[idx])
        pred = lambda j: m.predict(P["X_gbm"](j))  # noqa: E731
    else:
        X = P["X"][design]
        b = c1.irls_ridge(X(idx), N[idx], e[idx])
        pred = lambda j: np.exp(X(j) @ b)  # noqa: E731
    own = pred(idx)
    c = N[idx].sum() / (e[idx] * own).sum()
    return {"ca": c * pred(P["ca"]), "te": c * pred(P["te"])}


# ---------------------------------------------------------------------------------------------------
# Measurements on one pair of tariffs
# ---------------------------------------------------------------------------------------------------
def blend(fa: dict, fb: dict, P: dict) -> dict:
    """The best log-linear blend of two tariffs, rate = exp(c + wa log ra + wb log rb), fitted by Poisson maximum
    likelihood on the calibration sample's claims; deviances on the test sample per 1,000 policy-years."""
    ec, Nc = P["e"][P["ca"]], P["N"][P["ca"]]
    et, Nt = P["e"][P["te"]], P["N"][P["te"]]
    Xc = np.column_stack([np.ones(ec.size), np.log(fa["ca"]), np.log(fb["ca"])])
    w = c1.irls_ridge(Xc, Nc, ec, lam=0.0)
    Xt = np.column_stack([np.ones(et.size), np.log(fa["te"]), np.log(fb["te"])])
    k = Nc.sum() / (ec * np.sqrt(fa["ca"] * fb["ca"])).sum()
    per = et.sum() / 1000.0
    dev = lambda r: poisson_deviance(Nt, et * r) / per  # noqa: E731
    return {"wa": float(w[1]), "wb": float(w[2]), "da": dev(fa["te"]), "db": dev(fb["te"]),
            "davg": dev(k * np.sqrt(fa["te"] * fb["te"])), "dbl": dev(np.exp(Xt @ w))}


def one_period(Rt, et, Nt) -> dict:
    m = c1.market(Rt, et, Nt, INF)
    m10 = c1.market(Rt, et, Nt, BETA)
    s_ = c1.sigma_hat(Rt, et)
    alone = (et * Rt).sum(axis=1) / Nt.sum()            # what each tariff would collect alone at level 1
    return {"sig": s_, "coll": m["collected"], "logn": float(alone.mean()) * c1.lognormal_collect(2, s_),
            "lr": m["lr"], "share": m["share"], "coll10": m10["collected"], "lr10": m10["lr"]}


def reviews(Rc, Rt, ec, Nc, et, Nt) -> dict:
    out = {}
    for beta, key in ((BETA, "10"),):
        L, ok = c1.review(Rc, ec, Nc, beta)
        m = c1.market(Rt, et, Nt, beta, L)
        out[key] = {"level": m["collected"], "share": m["share"], "ok": ok, "L": L,
                    "load": float((m["share"] * L).sum())}
    return out


def games(Rc, Rt, P, check: bool) -> dict:
    o1 = g2.stage2(Rc, Rt, P, BETA, "aware", check=check)
    o2 = g3.outcome(Rc, Rt, P, "market position", "aware")
    return {"one level": {"profit": o1["profit"], "level": o1["level"], "share": o1["share"], "ok": o1["ok"],
                          "gain": o1["gain"]},
            "market position": {"profit": o2["profit"], "level": o2["level"], "share": o2["share"], "ok": o2["ok"],
                                "gain": o2["gain"]}}


# ---------------------------------------------------------------------------------------------------
# 0. The mechanism
# ---------------------------------------------------------------------------------------------------
def mech_play(T: np.ndarray, ec: np.ndarray, c: np.ndarray) -> dict:
    """Two tariffs T (2 x n, per unit of exposure) against known costs c (expected claims per policy)."""
    lp = np.log(ec * T)
    s_ = c1.sigma_hat(T, ec)
    m = c1.market(T, ec, c, INF)
    m10 = c1.market(T, ec, c, BETA)
    L, ok_r = c1.review(T, ec, c, BETA)
    mr = c1.market(T, ec, c, BETA, L)
    x, ok1 = g2.nash(lp, c, BETA)
    p1 = g2.profits(lp, x, c, BETA) / c.sum() * 100
    s1 = g2.shares(lp, x, BETA)
    lev1 = float((s1 * np.exp(x[:, None] + lp)).sum() / c.sum())
    sc, _ = g3.segment_index(T, T, "market position")
    xs, ok2, left = g3.seg_nash(lp, sc, c, BETA)
    p2 = g3.seg_profits(lp, sc, xs, c, BETA) / c.sum() * 100
    xe = xs[np.arange(2)[:, None], sc]
    s2 = g3._shares(lp, xe, BETA)
    lev2 = float((s2 * np.exp(xe + lp)).sum() / c.sum())
    return {"sig": s_, "coll": m["collected"], "logn": c1.lognormal_collect(2, s_), "lr": m["lr"].tolist(),
            "coll10": m10["collected"], "rL": L.tolist(), "rload": float((mr["share"] * L).sum()),
            "rshare": float(mr["share"][0]), "rok": ok_r,
            "p1": p1.tolist(), "lev1": lev1, "ok1": ok1, "p2": p2.tolist(), "lev2": lev2, "ok2": ok2,
            "left": left / c.sum() * 100}


def mechanism(P: dict) -> dict:
    """Known costs: the GLM rate fitted on the whole training pool plays the true expected frequency. Three kinds of
    competitor pairs with the same dispersion of quotes:
        noise        both tariffs are the cost times exp(sigma Z - sigma^2/2), with independent Z
        information  insurer A quotes the cost exactly; B quotes it times exp(sqrt(2) sigma Z - sigma^2), so that the
                     pair has the same dispersion but all the disagreement is B's error
        a design     insurer A quotes the cost; B quotes the rate of another design fitted on the same pool"""
    e, N = P["e"], P["N"]
    tr, ca = P["tr"], P["ca"]
    ec = e[ca]
    b = c1.irls_ridge(P["X_glm"](tr), N[tr], e[tr])
    rate = np.exp(P["X_glm"](ca) @ b)
    c = ec * rate
    rng = np.random.default_rng(SEED + 9)
    Z = rng.standard_normal((2, c.size))
    header(f"{P['name']}: the mechanism with known costs (cost = the GLM rate fitted on the whole training pool;"
           f" calibration sample, {c.size:,} policies; beta 10 unless cheapest wins)")
    res = {}

    def show(key, T):
        o = mech_play(T, ec, c)
        res[key] = o
        print(f"  {key}: sigma {o['sig']:.3f}; cheapest wins, collected / cost {o['coll']:.3f} (lognormal from sigma"
              f" {o['logn']:.3f}), loss ratio A {o['lr'][0]:.3f}, B {o['lr'][1]:.3f}; beta 10, collected / cost"
              f" {o['coll10']:.3f}")
        print(f"      after rate reviews (beta 10): loading A {o['rL'][0]:.3f}, B {o['rL'][1]:.3f}, share-weighted"
              f" {o['rload']:.3f}; share of A {o['rshare']:.3f}"
              + ("" if o["rok"] else " (not settled)"))
        print(f"      curse-aware game, one level: profit A {o['p1'][0]:+.2f}, B {o['p1'][1]:+.2f}, level"
              f" {o['lev1']:.3f}" + ("" if o["ok1"] else " (not settled)")
              + f"; market position: profit A {o['p2'][0]:+.2f}, B {o['p2'][1]:+.2f}, level {o['lev2']:.3f}"
              + ("" if o["ok2"] else f" (not settled; largest remaining gain {g3.bound(o['left'])} percent)"))

    show("exact tariffs", np.vstack([rate, rate]))
    for sg in SIGMAS:
        show(f"noise, sigma {sg:.1f}", rate * np.exp(sg * Z - sg ** 2 / 2))
        show(f"information, sigma {sg:.1f}", np.vstack([rate, rate * np.exp(np.sqrt(2) * sg * Z[1] - sg ** 2)]))
    for design in ("linear BM", "no territory"):
        X = P["X"][design]
        bd = c1.irls_ridge(X(tr), N[tr], e[tr])
        rd = np.exp(X(ca) @ bd)
        rd *= c.sum() / (ec * rd).sum()
        show(f"A the GLM, B {design}", np.vstack([rate, rd]))
        if design == "linear BM":
            # Where the linear design's errors lie: by band of the bonus-malus level.
            bm = P["bm"][ca]
            band = np.digitize(bm, P["bm_edges"])
            under = np.maximum(c - ec * rd, 0.0)             # what B undercharges where it quotes below cost
            q = np.log(rd / rate)
            ed = [int(x) for x in P["bm_edges"]]
            names = [f"below {ed[0]}"] + [f"{ed[k]}-{ed[k + 1] - 1}" for k in range(len(ed) - 1)] + [f"{ed[-1]} and above"]
            print("      where B's errors lie, by bonus-malus level: share of exposure; mean log of B's rate over A's;"
                  " share of B's undercharging (cost minus premium where B quotes below cost)")
            for k, nm in enumerate(names):
                m = band == k
                print(f"        {nm:>14}: {ec[m].sum() / ec.sum():.3f}; {np.average(q[m], weights=ec[m]):+.3f};"
                      f" {under[m].sum() / under.sum():.3f}")
    return res


# ---------------------------------------------------------------------------------------------------
def run(P: dict, cfg: dict) -> dict:
    e, N = P["e"], P["N"]
    et, Nt = e[P["te"]], N[P["te"]]
    ec, Nc = e[P["ca"]], N[P["ca"]]
    per = et.sum() / 1000.0
    name = P["name"]
    sizes = cfg["sizes"]
    rng = np.random.default_rng(SEED + (0 if name == "freMTPL2" else 100))   # the second companion's partitions

    single = {n: {d: {"gini": [], "dev": []} for d in DESIGNS} for n in sizes}
    twin = {n: {d: [] for d in DESIGNS} for n in sizes}
    pairs = {n: {p: {"sig": [], "same": [], "part": [], "bl": [], "op": [], "rv": [], "gm": []} for p in PAIRS}
             for n in sizes}
    conv = {g: [0, 0] for g in GAMES}
    gains = {g: [] for g in GAMES}
    unsettled = []

    for r in range(R):
        print(f"{name}: partition {r + 1} of {R}", file=sys.stderr, flush=True)
        perm = rng.permutation(P["tr"])
        for si, n in enumerate(sizes):
            books = [np.sort(perm[j * n:(j + 1) * n]) for j in range(2)]
            f = {(d, j): fit_design(P, b, d, seed=SEED + 1000 * r + 10 * si + j) for j, b in enumerate(books)
                 for d in DESIGNS}
            tw = {}
            for d in DESIGNS:
                for j in range(2):
                    single[n][d]["gini"].append(gini(f[(d, j)]["te"], et, Nt))
                    single[n][d]["dev"].append(poisson_deviance(Nt, et * f[(d, j)]["te"]) / per)
                tw[d] = c1.sigma_hat(np.array([f[(d, 0)]["te"], f[(d, 1)]["te"]]), et)
                twin[n][d].append(tw[d])
            for a, b in PAIRS:
                z = pairs[n][(a, b)]
                plays = [(a, b, 0)] if a == b else [(a, b, 0), (b, a, 1)]   # (book 0, book 1, row of design a)
                cross = []
                for d0, d1, ia in plays:
                    f0, f1 = f[(d0, 0)], f[(d1, 1)]
                    Rc = np.array([f0["ca"], f1["ca"]])
                    Rt = np.array([f0["te"], f1["te"]])
                    fa, fb = (f0, f1) if ia == 0 else (f1, f0)
                    op = one_period(Rt, et, Nt)
                    cross.append(op["sig"])
                    z["sig"].append(op["sig"])
                    z["bl"].append(blend(fa, fb, P))
                    z["op"].append((op, ia))
                    z["rv"].append((reviews(Rc, Rt, ec, Nc, et, Nt), ia))
                    gm = games(Rc, Rt, P, check=(r == 0))
                    for g in GAMES:
                        conv[g][0] += gm[g]["ok"]
                        conv[g][1] += 1
                        if not np.isnan(gm[g]["gain"]):
                            gains[g].append(gm[g]["gain"])
                        if not gm[g]["ok"]:
                            unsettled.append(f"{g}, books of {n:,}, {d0} on book 1 against {d1} on book 2,"
                                             f" partition {r + 1}")
                    z["gm"].append((gm, ia))
                if a != b:
                    for j in range(2):
                        z["same"].append(c1.sigma_hat(np.array([f[(a, j)]["te"], f[(b, j)]["te"]]), et))
                z["part"].append(np.mean(np.square(cross)) - (tw[a] ** 2 + tw[b] ** 2) / 2)

    res = {"single": {}, "decomp": {}, "blend": {}, "op": {}, "rv": {}, "gm": {}}
    # -----------------------------------------------------------------------------------------------
    header(f"{name}: the designs alone (test sample; Gini on claim counts, Poisson deviance per 1,000 policy-years;"
           f" mean over {2 * R} fits) and twin dispersion (root mean square over {R} pairs of books)")
    print(f"  Policies {e.size:,}; training pool {P['tr'].size:,}; calibration {P['ca'].size:,}; test {P['te'].size:,}"
          f" ({et.sum():,.1f} policy-years, {Nt.sum():,.0f} claims); two disjoint books per partition, {R} partitions")
    print(f"  Columns of the GLM designs: " + ", ".join(f"{d} {P['X'][d](P['tr'][:10]).shape[1]}" for d in P["X"])
          + f"; the territorial factors left out of 'no territory': {P['territory']}")
    for n in sizes:
        print(f"  books of {n:,}:")
        for d in DESIGNS:
            s = single[n][d]
            print(f"      {d:<12} Gini {summary(s['gini'])}, deviance {np.mean(s['dev']):.2f}; twin sigma"
                  f" {rms(twin[n][d]):.3f} (mean {np.mean(twin[n][d]):.3f})")
            res["single"][f"{d}|{n}"] = {"gini": float(np.mean(s["gini"])), "dev": float(np.mean(s["dev"])),
                                         "twin": rms(twin[n][d])}
    ln_n = np.log(np.array(sizes, dtype=float))
    for d in DESIGNS:
        sl = np.polyfit(ln_n, np.log([rms(twin[n][d]) for n in sizes]), 1)[0]
        print(f"  {d}: twin sigma falls like n to the power {sl:.2f}")

    header(f"{name}: disagreement between designs on independent books (test sample; sigma as root mean square over"
           f" plays; data part = root of the mean of the two designs' squared twin sigmas; design part = root of the"
           f" remainder; same book = the two designs fitted on one book)")
    for a, b in PAIRS:
        if a == b:
            continue
        print(f"  {label(a, b)}:")
        dec = []
        for n in sizes:
            z = pairs[n][(a, b)]
            s2 = np.mean(np.square(z["sig"]))
            data2 = (rms(twin[n][a]) ** 2 + rms(twin[n][b]) ** 2) / 2
            des2 = s2 - data2
            pos = int(np.sum(np.array(z["part"]) > 0))
            dec.append((np.sqrt(s2), np.sqrt(data2), np.sqrt(max(des2, 0.0)), des2 / s2))
            print(f"      books of {n:>7,}: sigma {np.sqrt(s2):.3f} (same book {rms(z['same']):.3f}); data part"
                  f" {np.sqrt(data2):.3f}, design part " + (f"{np.sqrt(des2):.3f}" if des2 > 0 else "none")
                  + f" ({100 * des2 / s2:.0f} percent of the squared dispersion); design part positive in {pos} of"
                  f" {R} partitions")
            res["decomp"][f"{a}|{b}|{n}"] = {"sig": float(np.sqrt(s2)), "data": float(np.sqrt(data2)),
                                             "design": float(np.sqrt(max(des2, 0.0))), "share": float(des2 / s2),
                                             "same": rms(z["same"])}
        sl_s = np.polyfit(ln_n, np.log([x[0] for x in dec]), 1)[0]
        sl_d = np.polyfit(ln_n, np.log([x[1] for x in dec]), 1)[0]
        msg = f"      as the books grow, sigma falls like n to the power {sl_s:.2f}, the data part like {sl_d:.2f}"
        if all(x[2] > 0 for x in dec):
            msg += f", the design part like {np.polyfit(ln_n, np.log([x[2] for x in dec]), 1)[0]:.2f}"
        print(msg)

    header(f"{name}: noise or information: the best blend exp(c + wA log rA + wB log rB) fitted on the calibration"
           f" sample; deviance on the test sample per 1,000 policy-years (mean over plays)")
    for a, b in PAIRS:
        print(f"  {label(a, b)}:")
        for n in sizes:
            bl = pairs[n][(a, b)]["bl"]
            wa, wb = np.array([x["wa"] for x in bl]), np.array([x["wb"] for x in bl])
            da, db = np.mean([x["da"] for x in bl]), np.mean([x["db"] for x in bl])
            dv, dbl = np.mean([x["davg"] for x in bl]), np.mean([x["dbl"] for x in bl])
            gain = np.mean([min(x["da"], x["db"]) - x["dbl"] for x in bl])
            print(f"      books of {n:>7,}: weight on {a} {summary(wa)}, on {b} {summary(wb)}; deviance {a} {da:.2f},"
                  f" {b} {db:.2f}, geometric mean {dv:.2f}, best blend {dbl:.2f}; the blend gains {gain:.2f} over the"
                  f" better of the two")
            res["blend"][f"{a}|{b}|{n}"] = {"wa": float(wa.mean()), "wb": float(wb.mean()), "da": float(da),
                                            "db": float(db), "davg": float(dv), "dbl": float(dbl), "gain": float(gain)}

    header(f"{name}: one period at a common level of 1, cheapest wins (test sample; mean over plays): what the market"
           f" collects per unit of claims; the lognormal prediction from the pair's sigma, scaled by what the two"
           f" tariffs would collect alone; each insurer's loss ratio; and the same market with beta 10")
    for a, b in PAIRS:
        print(f"  {label(a, b)}:")
        for n in sizes:
            ops = pairs[n][(a, b)]["op"]
            coll = np.array([o["coll"] for o, _ in ops])
            logn = np.array([o["logn"] for o, _ in ops])
            lra = np.array([o["lr"][ia] for o, ia in ops])
            lrb = np.array([o["lr"][1 - ia] for o, ia in ops])
            sha = np.array([o["share"][ia] for o, ia in ops])
            c10 = np.array([o["coll10"] for o, _ in ops])
            la10 = np.array([o["lr10"][ia] for o, ia in ops])
            lb10 = np.array([o["lr10"][1 - ia] for o, ia in ops])
            if a == b:
                lra = lrb = np.concatenate([lra, lrb])
                la10 = lb10 = np.concatenate([la10, lb10])
                sha = np.full(1, 0.5)
            print(f"      books of {n:>7,}: collected {summary(coll)} (lognormal {np.mean(logn):.3f}, minus actual"
                  f" {np.mean(logn - coll):+.3f}); loss ratio {a} {np.mean(lra):.3f}, {b} {np.mean(lrb):.3f}"
                  + ("" if a == b else f" ({b} higher in {int(np.sum(lrb > lra))} of {lra.size}); share of {a}"
                                       f" {np.mean(sha):.3f}")
                  + f" | beta 10: collected {np.mean(c10):.3f}, loss ratio {a} {np.mean(la10):.3f}, {b}"
                  f" {np.mean(lb10):.3f}")
            res["op"][f"{a}|{b}|{n}"] = {"coll": float(coll.mean()), "logn": float(logn.mean()),
                                         "lra": float(lra.mean()), "lrb": float(lrb.mean()), "sha": float(sha.mean()),
                                         "coll10": float(c10.mean()), "lra10": float(la10.mean()),
                                         "lrb10": float(lb10.mean())}
    for group, ps in (("twins", [p for p in PAIRS if p[0] == p[1]]), ("different designs", [p for p in PAIRS if p[0] != p[1]])):
        err = np.array([o["logn"] - o["coll"] for n in sizes for p in ps for o, _ in pairs[n][p]["op"]])
        print(f"  lognormal minus actual, {group} ({err.size} plays): mean {err.mean():+.4f}, largest absolute"
              f" {np.abs(err).max():.4f}")

    header(f"{name}: after 40 rate reviews on the calibration sample (beta 10), once every insurer with business breaks even:"
           f" the loading each insurer applies to its own tariff, the share-weighted loading, and the first design's"
           f" share (test sample; mean over plays)")
    lv_all = []
    for a, b in PAIRS:
        print(f"  {label(a, b)}:")
        for n in sizes:
            rvs = pairs[n][(a, b)]["rv"]
            row = []
            for key, lab in (("10", "beta 10"),):
                lv_all += [v[key]["level"] for v, _ in rvs]
                la = np.array([v[key]["L"][ia] for v, ia in rvs])
                lb = np.array([v[key]["L"][1 - ia] for v, ia in rvs])
                ld = np.array([v[key]["load"] for v, _ in rvs])
                sh = np.array([v[key]["share"][ia] for v, ia in rvs])
                ok = sum(v[key]["ok"] for v, _ in rvs)
                if a == b:
                    txt = f"{lab}: loading {summary(ld)}"
                else:
                    txt = (f"{lab}: loading {a} {np.mean(la):.3f}, {b} {np.mean(lb):.3f}, share-weighted {np.mean(ld):.3f};"
                           f" share of {a} {summary(sh)}, below 0.5 in {int(np.sum(sh < 0.5))} of {sh.size}")
                row.append(txt + f" (settled {ok} of {len(rvs)})")
                res["rv"][f"{a}|{b}|{n}|{key}"] = {"la": float(la.mean()), "lb": float(lb.mean()),
                                                   "load": float(ld.mean()), "sha": float(sh.mean())}
            print(f"      books of {n:>7,}: " + "; ".join(row))
    print(f"  premium / claims on the test sample after the reviews, all plays: between {min(lv_all):.3f} and"
          f" {max(lv_all):.3f} (1 on the calibration sample once both insurers break even)")

    header(f"{name}: the price game, curse-aware pricing, beta 10: each insurer's profit (percent of the test sample's"
           f" claims) and the price level, mean over the plays that settled")
    for a, b in PAIRS:
        print(f"  {label(a, b)}:")
        for n in sizes:
            gms = pairs[n][(a, b)]["gm"]
            row = []
            for g in GAMES:
                kept = [(x[g], ia) for x, ia in gms if x[g]["ok"]]
                pa = np.array([o["profit"][ia] for o, ia in kept])
                pb = np.array([o["profit"][1 - ia] for o, ia in kept])
                lv = np.array([o["level"] for o, _ in kept])
                if a == b:
                    txt = f"{g}: each {np.mean(np.concatenate([pa, pb])):+.2f}"
                else:
                    txt = (f"{g}: {a} {np.mean(pa):+.2f}, {b} {np.mean(pb):+.2f} ({a} ahead in"
                           f" {int(np.sum(pa > pb))} of {pa.size})")
                row.append(txt + f", level {np.mean(lv):.3f}" + ("" if len(kept) == len(gms)
                                                                 else f" ({len(gms) - len(kept)} left out)"))
                res["gm"][f"{a}|{b}|{n}|{g}"] = {"pa": float(pa.mean()), "pb": float(pb.mean()),
                                                 "level": float(lv.mean()), "plays": len(kept)}
            print(f"      books of {n:>7,}: " + "; ".join(row))

    header(f"{name}: checks")
    print(f"  the GLM design built here is the earlier companions' design: {P['same_as_before']}")
    try:
        ref2 = json.loads((HERE.parent / "companion-model-choice" / "figure_data.json").read_text())[name]
        ref3 = json.loads((HERE.parent / "companion-segment-pricing" / "figure_data.json").read_text())[name]
        diffs = {"one level": [], "market position": [], "twin": []}
        for n in sizes:
            for g in GAMES:
                A = np.zeros((2, 2))
                for i, s in enumerate(("GLM", "GBM")):
                    for j, t in enumerate(("GLM", "GBM")):
                        if s == t:
                            gms = pairs[n][(s, s)]["gm"]
                            A[i, j] = np.mean([x[g]["profit"][k] for x, _ in gms for k in range(2)])
                        else:
                            gms = pairs[n][("GLM", "GBM")]["gm"]
                            own = 0 if s == "GLM" else 1          # GLM is the pair's first design
                            A[i, j] = np.mean([x[g]["profit"][ia if own == 0 else 1 - ia] for x, ia in gms])
                ref = ref2[f"aware10"][str(n)]["A"] if g == "one level" else ref3["two"][f"market position|aware|{n}"]["A"]
                diffs[g].append(float(np.max(np.abs(A.round(4) - np.array(ref)[:2, :2]))))
            for d in ("GLM", "GBM"):
                diffs["twin"].append(abs(float(np.mean(twin[n][d])) - ref2["twin"][str(n)][d]))
        for k, v in diffs.items():
            lab = {"one level": "GLM and GBM payoffs, one level, against the second companion",
                   "market position": "GLM and GBM payoffs, market position, against the third companion",
                   "twin": "twin sigma of the GLM and the GBM against the second companion"}[k]
            print(f"  {lab}: largest difference " + ("below 1e-9" if max(v) < 1e-9 else f"{max(v):.2e}"))
    except FileNotFoundError:
        print("  earlier companions' figure_data.json not found; comparison skipped")
    for g in GAMES:
        print(f"  {g}: settled in {conv[g][0]} of {conv[g][1]} plays; largest gain from moving one level (first"
              f" partition for one level), percent of claims: {g3.bound(max(gains[g])) if gains[g] else 'n/a'}")
    print("  plays that did not settle, left out of the game payoffs: " + ("none" if not unsettled else ""))
    for u in unsettled:
        print(f"    {u}")
    return res


def main() -> None:
    print("Fourth companion: competing insurers whose tariffs differ in design, on freMTPL2 and beMTPL97.")
    print(f"Designs: {', '.join(DESIGNS)}. Other GBM: " + ", ".join(f"{k} {v}" for k, v in OTHER_GBM.items())
          + "; the GBM of the earlier companions: learning_rate 0.05, max_leaf_nodes 31, min_samples_leaf 200,"
          " random_state 7.")
    quick = os.environ.get("DESIGN_QUICK")
    global R
    if quick:
        R = 1
    out = {}
    for P, key in ((french(), "freMTPL2"), (belgian(), "beMTPL97")):
        cfg = dict(PORTFOLIOS[key])
        if quick:
            cfg["sizes"] = cfg["sizes"][:2]
        mech = mechanism(P)
        out[key] = run(P, cfg)
        out[key]["mechanism"] = mech
    if not quick:
        (HERE / "figure_data.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
