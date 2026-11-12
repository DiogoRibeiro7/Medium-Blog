"""Second companion to the actuarial-tariff-calibration series: model choice among competing insurers as a game.

Every number quoted in the article is printed by this script. Run:

    python game_numbers.py > output.txt      # about forty minutes, single-threaded; writes figure_data.json

The conjecture (first companion): in a head-to-head the better-ranking model wins whatever its variance, so model
choice in a competitive market can have the structure of a prisoner's dilemma: each insurer's best response is the
better-ranking, higher-variance model, and a market in which every insurer makes that choice ends up worse off than
one in which none does.

Design. A two-stage game, repeated on R random partitions of each training sample into books.
  Stage 1. Each insurer chooses a model class for its tariff: GLM, GBM (Poisson gradient boosting) or, with two
           insurers, BAG (the average of five boosting models on 70% subsamples). Models and portfolios are those of
           the first companion (curse_numbers.py): freMTPL2 and beMTPL97, each insurer fitting its own disjoint book,
           every tariff rescaled to collect its own book's claims.
  Stage 2. Given the tariffs, the insurers set price levels L_i (one multiplier each). Policyholders choose by the
           logit rule of Part 4: probability proportional to premium ** -beta. Three ways of setting the levels:
             aware   Nash equilibrium in levels; each insurer maximizes its expected profit, with the cost of each
                     policy it might win measured by the claims of the calibration sample, so it sees the adverse
                     selection its tariff suffers against the others.
             naive   Nash equilibrium in levels; each insurer maximizes the profit it expects if its own tariff were
                     the true cost (the textbook markup, blind to the winner's curse).
             review  each insurer multiplies its level by its own loss ratio, 40 times (Parts 4 and the first
                     companion); profits are then close to zero by construction.
  Payoff. Each insurer's expected profit on the test sample at the stage-2 levels, in percent of the claims of the
          whole test sample. The price level policyholders pay is total premium over total claims.

Sections, for each portfolio
0. The mechanism: tariffs that differ only in noise around a common rate, so that variance is separated from ranking.
1. The models alone: Gini and deviance by class and book size; dispersion of twin fits.
2. Two insurers, aware pricing, beta 10: payoff tables over {GLM, GBM, BAG}, equilibria, the price level.
3. Two insurers, aware pricing, beta 5 and 20.
4. Two insurers, naive pricing, beta 10.
5. Two insurers, rate reviews, beta 10.
6. Four insurers choosing between GLM and GBM, aware and naive pricing, beta 10 and 20: payoffs by the number of
   boosting insurers, equilibria, the path of one-at-a-time switches from an all-GLM market, and whether the game is
   a prisoner's dilemma, its reverse, or neither, overall and partition by partition.
7. Checks: convergence of the equilibria and the largest gain any insurer could make by deviating from its level.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "companion-winners-curse"))
import curse_numbers as c1  # noqa: E402
from tariffcal import gini, poisson_deviance  # noqa: E402

SEED = 20261112
R = 6
BETAS = (5.0, 10.0, 20.0)
CLASSES = ("GLM", "GBM", "BAG")
PAIR = ("GLM", "GBM")
K4 = 4
K4_BETAS = (10.0, 20.0)
PORTFOLIOS = {
    "freMTPL2": {"sizes": (20_000, 50_000, 100_000, 200_000)},
    "beMTPL97": {"sizes": (5_000, 12_000, 24_000, 48_000)},
}
header = c1.header


# ---------------------------------------------------------------------------------------------------
# Stage 2: price levels
# ---------------------------------------------------------------------------------------------------
def shares(lp: np.ndarray, x: np.ndarray, beta: float) -> np.ndarray:
    u = -beta * (x[:, None] + lp)
    u -= u.max(axis=0)
    w = np.exp(u)
    return w / w.sum(axis=0)


def profits(lp, x, cost, beta) -> np.ndarray:
    """Expected profit of each insurer: sum over policies of share times (premium - cost)."""
    s = shares(lp, x, beta)
    return (s * (np.exp(x[:, None] + lp) - cost)).sum(axis=1)


def nash(lp: np.ndarray, cost: np.ndarray, beta: float, iters: int = 400) -> tuple[np.ndarray, bool]:
    """Nash equilibrium in log levels x_i under logit demand.

    lp is the log of each insurer's base premium per policy (K x n); cost is the cost each insurer counts per policy
    (n, or K x n). Simultaneous damped Newton steps on every insurer's first-order condition; returns x and whether
    every step has fallen below 1e-10.
    """
    K = lp.shape[0]
    x = np.full(K, np.log(1 + 1 / (beta * (1 - 1 / K))))

    def newton_step(x, who):
        s = shares(lp, x, beta)
        P = np.exp(x[:, None] + lp)
        M = P - cost
        q = s * (1 - s)
        g = (s * P - beta * q * M).sum(axis=1)
        h = (s * P * (1 - beta * (1 - s)) + beta ** 2 * q * (1 - 2 * s) * M - beta * q * P).sum(axis=1)
        step = np.where(h < 0, -g / np.where(h < 0, h, -1.0), np.sign(g) * 0.05)
        step = np.clip(step, -0.1, 0.1)
        new = x.copy()
        new[who] = np.clip(x[who] + step[who], -1.0, 3.0)   # an insurer at the upper bound has left the market
        return new

    everyone = np.arange(K)
    for _ in range(iters):                                  # all insurers move at once
        new = newton_step(x, everyone)
        moved = np.max(np.abs(new - x))
        x = new
        if moved < 1e-10:
            return x, True
    for _ in range(200):                                    # fallback: each insurer replies in turn
        start = x.copy()
        for i in range(K):
            for _ in range(100):
                new = newton_step(x, np.array([i]))
                moved = abs(new[i] - x[i])
                x = new
                if moved < 1e-12:
                    break
        if np.max(np.abs(x - start)) < 1e-10:
            return x, True
    return x, False


def deviation_gain(lp, x, cost, beta, points=61) -> float:
    """Largest gain any insurer can make by moving its own log level on a grid of +-0.5 around x, others fixed."""
    base = profits(lp, x, cost, beta)
    best = 0.0
    for i in range(lp.shape[0]):
        for d in np.linspace(-0.5, 0.5, points):
            y = x.copy()
            y[i] += d
            best = max(best, float(profits(lp, y, cost, beta)[i] - base[i]))
    return best


def stage2(Rc, Rt, P, beta, how, check=False) -> dict:
    """Levels set on the calibration sample by rule `how`; outcome evaluated on the test sample."""
    ec, Nc = P["e"][P["ca"]], P["N"][P["ca"]]
    et, Nt = P["e"][P["te"]], P["N"][P["te"]]
    lpc = np.log(ec * Rc)
    gain = np.nan
    if how == "review":
        L, ok = c1.review(Rc, ec, Nc, beta)
        x = np.log(L)
    else:
        cost = Nc if how == "aware" else ec * Rc
        x, ok = nash(lpc, cost, beta)
        if check:
            gain = deviation_gain(lpc, x, cost, beta) / Nc.sum() * 100
    lpt = np.log(et * Rt)
    s = shares(lpt, x, beta)
    prem = (s * np.exp(x[:, None] + lpt)).sum(axis=1)
    clm = (s * Nt).sum(axis=1)
    return {"profit": (prem - clm) / Nt.sum() * 100, "level": float(prem.sum() / Nt.sum()),
            "share": (s * et).sum(axis=1) / et.sum(), "L": np.exp(x), "ok": ok, "gain": gain}


# ---------------------------------------------------------------------------------------------------
# Equilibria of the model-choice stage
# ---------------------------------------------------------------------------------------------------
def pure_equilibria(A: np.ndarray, names) -> list[str]:
    """Pure Nash equilibria of the symmetric two-player game with payoff A[own, other]."""
    out = []
    S = len(names)
    for s in range(S):
        for t in range(s, S):
            if A[s, t] >= A[:, t].max() - 1e-12 and A[t, s] >= A[:, s].max() - 1e-12:
                out.append(f"({names[s]}, {names[t]})")
    return out


def dominant(A: np.ndarray, names) -> str:
    for s in range(len(names)):
        if all(A[s, t] > A[u, t] for t in range(len(names)) for u in range(len(names)) if u != s):
            return names[s]
    return "none"


def k_equilibria(pg: np.ndarray, pl: np.ndarray, K: int) -> list[int]:
    """Equilibria by number j of boosting insurers; pg[j] payoff of a boosting insurer when j use boosting (j>=1),
    pl[j] payoff of a GLM insurer when j use boosting (j<=K-1)."""
    out = []
    for j in range(K + 1):
        stay_g = j == 0 or pg[j] >= pl[j - 1]
        stay_l = j == K or pl[j] >= pg[j + 1]
        if stay_g and stay_l:
            out.append(j)
    return out


def fmt(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{np.nanmean(x):+.2f}"


# ---------------------------------------------------------------------------------------------------
def run(P: dict, cfg: dict) -> dict:
    e, N = P["e"], P["N"]
    et, Nt = e[P["te"]], N[P["te"]]
    per = et.sum() / 1000.0
    name = P["name"]
    sizes = cfg["sizes"]
    rng = np.random.default_rng(SEED + (0 if name == "freMTPL2" else 100))
    n_tr = P["tr"].size

    single = {n: {c: {"gini": [], "dev": []} for c in CLASSES} for n in sizes}
    twin = {n: {c: [] for c in CLASSES} for n in sizes}
    # two[n][(how, beta)][(s, t)] -> list over partitions and both book assignments of (payoff s, payoff t, level)
    two = {n: {} for n in sizes}
    # four[n][(how, beta)][r][j] -> {"G": payoffs of boosting insurers, "L": of GLM insurers, "level": levels}
    four = {n: {(h, b): [{j: {"G": [], "L": [], "level": []} for j in range(K4 + 1)} for _ in range(R)]
                for h in ("aware", "naive") for b in K4_BETAS}
            for n in sizes if K4 * n <= n_tr}
    conv = {"aware": [0, 0], "naive": [0, 0], "review": [0, 0]}
    gains = {"aware": [], "naive": []}
    gains4 = {"aware": [], "naive": []}
    unsettled = []
    best_by_part = {n: {} for n in sizes}

    for r in range(R):
        print(f"{name}: partition {r + 1} of {R}", file=sys.stderr, flush=True)
        perm = rng.permutation(P["tr"])
        for si, n in enumerate(sizes):
            nb = K4 if K4 * n <= n_tr else 2
            books = [np.sort(perm[j * n:(j + 1) * n]) for j in range(nb)]
            fits = {}
            for j, b in enumerate(books):
                for c in CLASSES:
                    if c == "BAG" and j >= 2:
                        continue
                    fits[(c, j)] = c1.fit_insurer(P, b, c, seed=SEED + 1000 * r + 10 * si + j)
            for c in CLASSES:
                for j in range(2):
                    single[n][c]["gini"].append(gini(fits[(c, j)]["te"], et, Nt))
                    single[n][c]["dev"].append(poisson_deviance(Nt, et * fits[(c, j)]["te"]) / per)
                twin[n][c].append(c1.sigma_hat(np.array([fits[(c, 0)]["te"], fits[(c, 1)]["te"]]), et))

            # Two insurers on books 0 and 1, every ordered pair of classes (so both assignments of each pair).
            part = {}
            for s, t in itertools.product(CLASSES, CLASSES):
                Rc = np.array([fits[(s, 0)]["ca"], fits[(t, 1)]["ca"]])
                Rt = np.array([fits[(s, 0)]["te"], fits[(t, 1)]["te"]])
                runs = [(h, b) for h in ("aware", "naive") for b in BETAS] + [("review", 10.0)]
                for how, beta in runs:
                    if how == "naive" and beta != 10.0:
                        continue
                    out = stage2(Rc, Rt, P, beta, how, check=(r == 0 and beta == 10.0 and how != "review"))
                    conv[how][0] += out["ok"]
                    conv[how][1] += 1
                    if not out["ok"]:
                        unsettled.append(f"{how}, beta {beta:.0f}, two insurers, books of {n:,}, {s} on book 1 against"
                                         f" {t} on book 2, partition {r + 1}: levels {np.round(out['L'], 3).tolist()},"
                                         f" shares {np.round(out['share'], 3).tolist()}")
                    if how != "review" and not np.isnan(out["gain"]):
                        gains[how].append(out["gain"])
                    cell = two[n].setdefault((how, beta), {}).setdefault((s, t), [])
                    cell.append((out["profit"][0], out["profit"][1], out["level"], out["share"][0]))
                    part.setdefault((how, beta), {}).setdefault((s, t), []).append(out["profit"][0])
                    part[(how, beta)].setdefault((t, s), []).append(out["profit"][1])
            for key, d in part.items():
                A = np.array([[np.mean(d[(s, t)]) for t in PAIR] for s in PAIR])
                best_by_part[n].setdefault(key, []).append(A)

            # Four insurers choosing between GLM and GBM.
            if nb == K4:
                for G in itertools.product((0, 1), repeat=K4):
                    cls = ["GBM" if g else "GLM" for g in G]
                    Rc = np.array([fits[(c, j)]["ca"] for j, c in enumerate(cls)])
                    Rt = np.array([fits[(c, j)]["te"] for j, c in enumerate(cls)])
                    for how, beta in itertools.product(("aware", "naive"), K4_BETAS):
                        out = stage2(Rc, Rt, P, beta, how, check=(r == 0 and beta == 10.0))
                        conv[how][0] += out["ok"]
                        conv[how][1] += 1
                        if not out["ok"]:
                            unsettled.append(f"{how}, beta {beta:.0f}, four insurers, books of {n:,}, classes"
                                             f" {'/'.join(cls)}, partition {r + 1}: levels"
                                             f" {np.round(out['L'], 3).tolist()}, shares {np.round(out['share'], 3).tolist()}")
                        if not np.isnan(out["gain"]):
                            gains4[how].append(out["gain"])
                        z = four[n][(how, beta)][r][sum(G)]
                        z["G"] += [out["profit"][j] for j in range(K4) if G[j]]
                        z["L"] += [out["profit"][j] for j in range(K4) if not G[j]]
                        z["level"].append(out["level"])

    # -----------------------------------------------------------------------------------------------
    header(f"{name}: the models alone (test sample; books 1 and 2 of each partition, {2 * R} fits per cell)")
    print(f"Training pool {n_tr:,}; calibration {P['ca'].size:,}; test {P['te'].size:,} policies"
          f" ({et.sum():,.1f} policy-years, {Nt.sum():,.0f} claims); {R} partitions")
    for n in sizes:
        print(f"  books of {n:>7,}: " + "; ".join(
            f"{c} Gini {np.mean(single[n][c]['gini']):.3f} deviance {np.mean(single[n][c]['dev']):.2f}"
            f" twin sigma {np.mean(twin[n][c]):.3f}" for c in CLASSES))
        g = {c: np.array(single[n][c]["gini"]) for c in CLASSES}
        d = {c: np.array(single[n][c]["dev"]) for c in CLASSES}
        print(f"      GBM above GLM in Gini in {np.sum(g['GBM'] > g['GLM'])} of {2 * R}, below in deviance in"
              f" {np.sum(d['GBM'] < d['GLM'])}; BAG above GLM in Gini in {np.sum(g['BAG'] > g['GLM'])},"
              f" below in deviance in {np.sum(d['BAG'] < d['GLM'])}")

    def table(n, how, beta, names=CLASSES, verbose=True):
        cells = two[n][(how, beta)]
        A = np.full((len(names), len(names)), np.nan)
        lev = np.full_like(A, np.nan)
        for a, s in enumerate(names):
            for b, t in enumerate(names):
                pay = [x[0] for x in cells[(s, t)]] + [x[1] for x in cells[(t, s)]]
                A[a, b] = np.mean(pay)
                lev[a, b] = np.mean([x[2] for x in cells[(s, t)]] + [x[2] for x in cells[(t, s)]])
        if verbose:
            for a, s in enumerate(names):
                print(f"      {s} against " + ", ".join(f"{t} {A[a, b]:+.2f} (level {lev[a, b]:.3f})"
                                                  for b, t in enumerate(names)))
        return A, lev

    def game_report(how, beta, label):
        header(f"{name}: two insurers, {label}: payoff of the row insurer against the column insurer, in percent of"
               f" the test sample's claims (mean of {2 * R} plays), and the price level policyholders pay")
        res = {}
        for n in sizes:
            print(f"  books of {n:,}:")
            A, lev = table(n, how, beta)
            A2 = A[:2, :2]
            eq2 = pure_equilibria(A2, PAIR)
            if "(GBM, GBM)" in eq2 and A2[0, 0] > A2[1, 1]:
                kind2 = "prisoner's dilemma: (GBM, GBM) is an equilibrium, (GLM, GLM) would pay both more"
            elif "(GLM, GLM)" in eq2 and A2[1, 1] > A2[0, 0]:
                kind2 = "the reverse: (GLM, GLM) is an equilibrium, (GBM, GBM) would pay both more"
            else:
                kind2 = "no conflict"
            per_part = np.array(best_by_part[n][(how, beta)])
            br_glm = int(np.sum(per_part[:, 1, 0] > per_part[:, 0, 0]))
            br_gbm = int(np.sum(per_part[:, 1, 1] > per_part[:, 0, 1]))
            both = int(np.sum(per_part[:, 0, 0] > per_part[:, 1, 1]))
            print(f"    GLM or GBM only: dominant {dominant(A2, PAIR)}; equilibria {', '.join(eq2)}; {kind2};"
                  f" by partition: GBM the better reply to GLM in {br_glm} of {R}, to GBM in {br_gbm} of {R},"
                  f" (GLM, GLM) pays more than (GBM, GBM) in {both} of {R}")
            print(f"    all three classes: dominant {dominant(A, CLASSES)}; equilibria {', '.join(pure_equilibria(A, CLASSES))}")
            res[str(n)] = {"A": A.round(4).tolist(), "level": lev.round(4).tolist()}
        return res

    out = {"single": {str(n): {c: {k: float(np.mean(v)) for k, v in single[n][c].items()} for c in CLASSES}
                      for n in sizes},
           "twin": {str(n): {c: float(np.mean(twin[n][c])) for c in CLASSES} for n in sizes}}
    out["aware10"] = game_report("aware", 10.0, "curse-aware price competition, beta 10")
    for beta in (5.0, 20.0):
        out[f"aware{beta:.0f}"] = game_report("aware", beta, f"curse-aware price competition, beta {beta:.0f}")
    out["naive10"] = game_report("naive", 10.0, "naive price competition (own tariff taken as the cost), beta 10")

    header(f"{name}: two insurers, rate reviews to break-even (beta 10): share of the row insurer, and the price level")
    rv = {}
    for n in sizes:
        cells = two[n][("review", 10.0)]
        print(f"  books of {n:,}:")
        rows = []
        for s in CLASSES:
            row = []
            for t in CLASSES:
                sh = [x[3] for x in cells[(s, t)]] + [1 - x[3] for x in cells[(t, s)]]
                pr = [x[0] for x in cells[(s, t)]] + [x[1] for x in cells[(t, s)]]
                lv = [x[2] for x in cells[(s, t)]]
                row.append((np.mean(sh), np.mean(pr), np.mean(lv)))
            rows.append(row)
            print(f"      {s} against " + ", ".join(f"{t} share {v[0]:.3f} profit {v[1]:+.2f} level {v[2]:.3f}"
                                                for t, v in zip(CLASSES, row)))
        rv[str(n)] = [[list(map(float, v)) for v in row] for row in rows]
    out["review10"] = rv

    def kind(pg, pl, eq) -> str:
        """Name the conflict, if any, between what each insurer chooses and what would pay all of them more."""
        if K4 in eq and pl[0] > pg[K4]:
            return "prisoner's dilemma: all boosting is an equilibrium, all GLM would pay every insurer more"
        if 0 in eq and pg[K4] > pl[0]:
            return "the reverse: all GLM is an equilibrium, all boosting would pay every insurer more"
        return "no conflict: the equilibrium pays every insurer at least as much as the other uniform market"

    def curves(zz):
        pg = np.array([np.mean(zz[j]["G"]) if zz[j]["G"] else np.nan for j in range(K4 + 1)])
        pl = np.array([np.mean(zz[j]["L"]) if zz[j]["L"] else np.nan for j in range(K4 + 1)])
        lv = np.array([np.mean(zz[j]["level"]) for j in range(K4 + 1)])
        return pg, pl, lv

    header(f"{name}: four insurers choosing between GLM and GBM: payoff of a boosting and of a GLM insurer, in percent"
           f" of the test sample's claims, by the number j of boosting insurers (mean over {R} partitions)")
    fo = {}
    for n in four:
        for how, beta in itertools.product(("aware", "naive"), K4_BETAS):
            zs = four[n][(how, beta)]
            pooled = {j: {k: sum((zs[r][j][k] for r in range(R)), []) for k in ("G", "L", "level")}
                      for j in range(K4 + 1)}
            pg, pl, lv = curves(pooled)
            mkt = np.array([(j * pg[j] if j > 0 else 0.0) + ((K4 - j) * pl[j] if j < K4 else 0.0)
                            for j in range(K4 + 1)])
            eq = k_equilibria(pg, pl, K4)
            path, j = [0], 0
            while j < K4 and pg[j + 1] > pl[j]:
                j += 1
                path.append(j)
            parts = []
            for j in range(K4 + 1):
                who = ([f"boosting {pg[j]:+.2f}"] if j > 0 else []) + ([f"GLM {pl[j]:+.2f}"] if j < K4 else [])
                parts.append(f"j={j}: {', '.join(who)}, market {mkt[j]:+.2f}, level {lv[j]:.3f}")
            kinds = [kind(*curves(zs[r])[:2], k_equilibria(*curves(zs[r])[:2], K4)) for r in range(R)]
            counts = {k.split(":")[0]: kinds.count(k) for k in dict.fromkeys(kinds)}
            print(f"  books of {n:,}, {how}, beta {beta:.0f}: " + "; ".join(parts))
            print(f"      equilibria at j = {', '.join(map(str, eq))}; switching one insurer at a time from an all-GLM"
                  f" market: j = {' -> '.join(map(str, path))}; {kind(pg, pl, eq)}; by partition: "
                  + ", ".join(f"{k} in {v} of {R}" for k, v in counts.items()))
            fo[f"{n}_{how}_{beta:.0f}"] = {"pg": [None if np.isnan(v) else float(v) for v in pg],
                                           "pl": [None if np.isnan(v) else float(v) for v in pl],
                                           "level": lv.tolist(), "market": mkt.tolist(), "eq": eq, "path": path}
    out["four"] = fo

    header(f"{name}: checks")
    for how in ("aware", "naive", "review"):
        print(f"  {how}: settled in {conv[how][0]} of {conv[how][1]} plays")
    print("  plays that did not settle: " + ("none" if not unsettled else ""))
    for u in unsettled:
        print(f"    {u}")
    for how in ("aware", "naive"):
        for lab, gl in (("two", gains[how]), ("four", gains4[how])):
            if not gl:
                continue
            g = max(gl)
            print(f"  {how}, partition 1, beta 10, {lab} insurers ({len(gl)} plays): largest gain any insurer could make"
                  f" by moving its own level on a grid of +-0.5 around the equilibrium, in percent of the claims: "
                  + ("below 1e-6" if g < 1e-6 else f"{g:.6f}"))
    return out


def mechanism(P: dict) -> dict:
    """Tariffs that differ only in noise. The 'true' rate is a GLM fitted on the whole training pool; each insurer's
    tariff is that rate times exp(sigma Z - sigma^2 / 2), with independent standard normal Z per insurer and policy,
    so every tariff ranks the policies equally well in expectation and none is biased."""
    e, N = P["e"], P["N"]
    tr, ca = P["tr"], P["ca"]
    b = c1.irls_ridge(P["X_glm"](tr), N[tr], e[tr])
    c = e[ca] * np.exp(P["X_glm"](ca) @ b)
    rng = np.random.default_rng(SEED + 5)
    Z = rng.standard_normal((K4, c.size))
    header(f"{P['name']}: the mechanism, with tariffs that differ only in noise (beta 10; cost = the GLM rate on the"
           f" whole training pool; calibration sample, {c.size:,} policies)")
    res = {"sym": {}, "mat": {}}
    for K in (2, K4):
        for how in ("aware", "naive"):
            row = []
            for sg in (0.0, 0.1, 0.2, 0.3, 0.4):
                T = c * np.exp(sg * Z[:K] - sg ** 2 / 2)
                lp = np.log(T)
                x, ok = nash(lp, c if how == "aware" else T, 10.0)
                pr = profits(lp, x, c, 10.0) / c.sum() * 100
                s = shares(lp, x, 10.0)
                level = float((s * np.exp(x[:, None] + lp)).sum() / c.sum())
                row.append((sg, float(pr.mean()), level, ok))
            res["sym"][f"{K}_{how}"] = row
            print(f"  {K} insurers, {how}: " + "; ".join(f"sigma {sg:.1f}: profit per insurer {p:+.2f}, level {lv:.3f}"
                                                      + ("" if ok else " (not settled)") for sg, p, lv, ok in row))
    sig = (0.1, 0.2, 0.3)
    A = np.zeros((3, 3))
    for a, sa in enumerate(sig):
        for bb, sb in enumerate(sig):
            T = np.vstack([c * np.exp(sa * Z[0] - sa ** 2 / 2), c * np.exp(sb * Z[1] - sb ** 2 / 2)])
            lp = np.log(T)
            x, _ = nash(lp, c, 10.0)
            A[a, bb] = profits(lp, x, c, 10.0)[0] / c.sum() * 100
    print("  two insurers, aware: payoff of the row insurer, by the noise in its own tariff (rows) and in its"
          " competitor's (columns):")
    for a, sa in enumerate(sig):
        print(f"      sigma {sa:.1f} against " + ", ".join(f"{sb:.1f} {A[a, bb]:+.2f}" for bb, sb in enumerate(sig)))
    res["mat"] = A.round(4).tolist()
    return res


def main() -> None:
    print("Second companion: model choice among competing insurers as a game, on freMTPL2 and beMTPL97.")
    quick = os.environ.get("GAME_QUICK")
    global R
    if quick:
        R = 1
    out = {}
    for P, key in ((c1.french(), "freMTPL2"), (c1.belgian(), "beMTPL97")):
        cfg = dict(PORTFOLIOS[key])
        if quick:
            cfg["sizes"] = cfg["sizes"][:1] if key == "freMTPL2" else cfg["sizes"][:2]
        mech = mechanism(P)
        out[key] = run(P, cfg)
        out[key]["mechanism"] = mech
    if not quick:
        (HERE / "figure_data.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
