"""Third companion to the actuarial-tariff-calibration series: the model-choice game when insurers price by segment.

Every number quoted in the article is printed by this script. Run:

    python segment_numbers.py > output.txt     # about two hours, single-threaded; writes figure_data.json

The second companion (companion-model-choice) let each insurer choose one level for its whole tariff. Its main
finding was that noise in the tariffs works like product differentiation: it makes an insurer's demand less sensitive
to its level, so insurers that see their adverse selection earn more with noisier tariffs, and a prisoner's dilemma
appeared only for naive insurers. Real pricing teams optimize price by segment. This script repeats the game with
ten levels per insurer, under two segmentations:

  risk band        deciles of the insurer's own rate (price optimization by risk band)
  market position  deciles of the insurer's log rate minus the mean log rate of its competitors for the same policy
                   (the relative position an insurer sees on a price-comparison site; competitors' tariffs, not
                   their levels, so the segments do not move with the prices)

Cut points come from the calibration sample and are applied unchanged to the test sample. The books, partitions,
models and seeds are those of the second companion, so the one-level results reproduce its tables exactly (checked
below against its figure_data.json).

Sections, for each portfolio
0. The mechanism, with tariffs that differ only in noise: equilibrium price level, profit per insurer, and the
   dispersion of the final quotes a policyholder sees, for one level, risk bands and market position.
1. Two insurers, beta 10: payoff tables over {GLM, GBM, BAG}, equilibria and the price level, for one level and
   market position with curse-aware and naive pricing, and for risk bands with naive pricing.
2. Four insurers choosing between GLM and GBM, beta 10: payoffs by the number of boosting insurers, equilibria,
   one-at-a-time switching paths, and the kind of game, partition by partition.
3. How the insurers use the segments: levels by market-position decile in a GLM-against-boosting market.
4. Checks: agreement with the second companion, convergence, and the largest gain from moving one segment level.

Equilibria in segment levels are found by safeguarded Newton steps (damped when overlapping segments make the
insurers' levels oscillate) and then checked by a grid search over every segment level of every insurer; where a
level has a better optimum elsewhere on the grid, the best such move is made and the iterations resume. A play whose
levels never settle is not an equilibrium, and where its iterations stop depends on the last bits of the arithmetic,
so it differs from one machine to another: such plays are listed and left out of the payoffs.
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
sys.path.insert(0, str(HERE.parent / "companion-model-choice"))
import game_numbers as g2  # noqa: E402

c1 = g2.c1
header = c1.header
SEED = g2.SEED                      # same partitions, books and fits as the second companion
R = g2.R
BETA = 10.0
S = 10
K4 = g2.K4
CLASSES = g2.CLASSES
PAIR = g2.PAIR
SEGMENTS = ("one level", "risk band", "market position")
HOWS = ("aware", "naive")
# Games played with the real models. Curse-aware pricing by risk band is left out: with costs measured by the
# calibration sample's claims, an insurer's profit in a risk band can have more than one peak, the best replies can
# cycle, and a play that does not settle takes minutes; the diagnostic printed for the first partition shows how
# often that happens on the smallest books.
GAMES = [("one level", "aware"), ("one level", "naive"), ("market position", "aware"), ("market position", "naive"),
         ("risk band", "naive")]
PORTFOLIOS = g2.PORTFOLIOS


# ---------------------------------------------------------------------------------------------------
# Segments and the segment-level price game
# ---------------------------------------------------------------------------------------------------
def segment_index(Rc: np.ndarray, Rt: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Segment of each policy for each insurer (K x n), on the calibration and the test sample."""
    K = Rc.shape[0]
    if kind == "one level":
        return np.zeros(Rc.shape, dtype=np.int64), np.zeros(Rt.shape, dtype=np.int64)
    lc, lt = np.log(Rc), np.log(Rt)
    if kind == "market position":
        lc = lc - (lc.sum(axis=0) - lc) / (K - 1)
        lt = lt - (lt.sum(axis=0) - lt) / (K - 1)
    sc = np.empty(Rc.shape, dtype=np.int64)
    st = np.empty(Rt.shape, dtype=np.int64)
    for i in range(K):
        cuts = np.quantile(lc[i], np.linspace(0, 1, S + 1)[1:-1])
        sc[i] = np.searchsorted(cuts, lc[i], side="right")
        st[i] = np.searchsorted(cuts, lt[i], side="right")
    return sc, st


def seg_newton(lp, seg, cost, beta, x, who) -> np.ndarray:
    """One damped Newton step on the segment levels of the insurers in `who`. An insurer's profit is separable
    across its own segments, so the step is exact Newton for each insurer given the others."""
    K = lp.shape[0]
    rows = np.arange(K)[:, None]
    xe = x[rows, seg]
    s = _shares(lp, xe, beta)
    P = np.exp(xe + lp)
    M = P - cost
    q = s * (1 - s)
    gp = s * P - beta * q * M
    hp = s * P * (1 - beta * (1 - s)) + beta ** 2 * q * (1 - 2 * s) * M - beta * q * P
    new = x.copy()
    sp = s * P
    for i in who:
        g = np.bincount(seg[i], weights=gp[i], minlength=S)
        h = np.bincount(seg[i], weights=hp[i], minlength=S)
        # Safeguarded Newton: the curvature where it is clearly negative, otherwise a gradient step scaled by the
        # segment's premium, so that a flat or locally convex stretch of profit does not make the step bounce.
        scale = np.maximum(-h, np.bincount(seg[i], weights=sp[i], minlength=S))
        step = np.where(scale > 0, g / np.where(scale > 0, scale, 1.0), 0.0)
        new[i] = np.clip(x[i] + np.clip(step, -0.1, 0.1), -1.0, 3.0)
    return new


def _shares(lp, xe, beta):
    u = -beta * (xe + lp)
    u -= u.max(axis=0)
    w = np.exp(u)
    return w / w.sum(axis=0)


def seg_profits(lp, seg, x, cost, beta) -> np.ndarray:
    xe = x[np.arange(lp.shape[0])[:, None], seg]
    s = _shares(lp, xe, beta)
    return (s * (np.exp(xe + lp) - cost)).sum(axis=1)


def seg_iterate(lp, seg, cost, beta, x, fast=60, iters=600, damping=0.3) -> tuple[np.ndarray, bool]:
    """Simultaneous safeguarded Newton steps from x: undamped first; damped after `fast` steps, because the levels
    of insurers whose segments overlap can push each other back and forth instead of settling."""
    everyone = range(lp.shape[0])
    for it in range(iters):
        new = seg_newton(lp, seg, cost, beta, x, everyone)
        if it >= fast:
            new = x + damping * (new - x)
        moved = np.max(np.abs(new - x))
        x = new
        if moved < 1e-10:
            return x, True
    return x, False


def best_moves(lp, seg, x, cost, beta, points=21, width=0.3):
    """For every insurer and segment, the best move of that one level on a grid of +-width, others fixed:
    returns (gain, d) arrays of shape (K, S). Only the policies of the segment change, so each move is evaluated
    on them alone."""
    K = lp.shape[0]
    xe = x[np.arange(K)[:, None], seg]
    grid = np.linspace(-width, width, points)          # odd number of points, so the middle one is no move
    gain = np.zeros((K, S))
    move = np.zeros((K, S))
    for i in range(K):
        for k in range(S):
            idx = seg[i] == k
            if not idx.any():
                continue
            sub_lp, sub_xe = lp[:, idx], xe[:, idx]
            ci = cost[i, idx] if np.ndim(cost) == 2 else cost[idx]
            vals = []
            for d in grid:
                y = sub_xe.copy()
                y[i] += d
                sh = _shares(sub_lp, y, beta)
                vals.append(float((sh[i] * (np.exp(y[i] + sub_lp[i]) - ci)).sum()))
            vals = np.array(vals)
            gain[i, k] = vals.max() - vals[points // 2]
            move[i, k] = grid[int(vals.argmax())]
    return gain, move


def seg_nash(lp, seg, cost, beta, rounds=25, iters=600) -> tuple[np.ndarray, bool, float]:
    """Nash equilibrium in segment levels. Newton iterations find a point where every segment level is at a local
    optimum; a grid search over every segment level then checks it. Where some level has a better optimum elsewhere
    (an insurer's profit in a segment can have two peaks: undercut and win the segment, or price high and keep only
    the policies its competitor prices higher still), the best such move is made and the iterations resume.
    Returns the levels, whether they form an equilibrium on the grid, and the largest remaining gain."""
    K = lp.shape[0]
    x = np.full((K, S), np.log(1 + 1 / (beta * (1 - 1 / K))))
    tol = 1e-9 * float(np.exp(lp).sum())
    left = np.inf
    for _ in range(rounds):
        x, ok = seg_iterate(lp, seg, cost, beta, x, iters=iters)
        gain, move = best_moves(lp, seg, x, cost, beta)
        left = float(gain.max())
        if ok and left <= tol:
            return x, True, 0.0
        i, k = np.unravel_index(int(gain.argmax()), gain.shape)
        if gain[i, k] > tol:
            x = x.copy()
            x[i, k] += move[i, k]
    return x, False, left


def seg_deviation_gain(lp, seg, x, cost, beta) -> float:
    """Largest gain any insurer can make by moving one of its segment levels on the grid, others fixed."""
    return float(best_moves(lp, seg, x, cost, beta)[0].max())


def outcome(Rc, Rt, P, kind, how, beta=BETA, check=False) -> dict:
    """Levels set on the calibration sample; outcome on the test sample."""
    ec, Nc = P["e"][P["ca"]], P["N"][P["ca"]]
    et, Nt = P["e"][P["te"]], P["N"][P["te"]]
    if kind == "one level":                       # exactly the second companion's game
        out = g2.stage2(Rc, Rt, P, beta, how)
        x = np.log(out["L"])[:, None] * np.ones((1, S))
        sc, st = segment_index(Rc, Rt, kind)
        gain = np.nan
        ok = out["ok"]
    else:
        sc, st = segment_index(Rc, Rt, kind)
        lpc = np.log(ec * Rc)
        cost = Nc if how == "aware" else ec * Rc
        x, ok, left = seg_nash(lpc, sc, cost, beta)
        gain = left / Nc.sum() * 100
    lpt = np.log(et * Rt)
    xe = x[np.arange(Rt.shape[0])[:, None], st]
    s = _shares(lpt, xe, beta)
    prem_p = np.exp(xe + lpt)
    prem = (s * prem_p).sum(axis=1)
    clm = (s * Nt).sum(axis=1)
    disp = float(np.sqrt(np.average(np.log(prem_p / et).var(axis=0, ddof=1), weights=et)))
    return {"profit": (prem - clm) / Nt.sum() * 100, "level": float(prem.sum() / Nt.sum()),
            "share": (s * et).sum(axis=1) / et.sum(), "x": x, "ok": ok, "gain": gain, "disp": disp}


# ---------------------------------------------------------------------------------------------------
# Equilibria of the model-choice stage
# ---------------------------------------------------------------------------------------------------
def bound(g: float) -> str:
    """A gain that differs between machines in its last digits, reported as a coarse upper bound."""
    for b in (0.005, 0.05, 0.25, 1.0):
        if g < b:
            return f"below {b}"
    return f"about {g:.0f}"


def kind_two(A2: np.ndarray) -> str:
    eq = g2.pure_equilibria(A2, PAIR)
    if "(GBM, GBM)" in eq and A2[0, 0] > A2[1, 1]:
        return "prisoner's dilemma: (GBM, GBM) is an equilibrium, (GLM, GLM) would pay both more"
    if "(GLM, GLM)" in eq and A2[1, 1] > A2[0, 0]:
        return "the reverse: (GLM, GLM) is an equilibrium, (GBM, GBM) would pay both more"
    if eq == ["(GLM, GBM)"]:
        return "mixed: the only equilibrium has one insurer using each class"
    return "no conflict: the equilibrium pays both at least as much as the other uniform market"


def kind_four(pg, pl, eq) -> str:
    if not eq:
        return "none: no number of boosting insurers is an equilibrium"
    if K4 in eq and pl[0] > pg[K4]:
        return "prisoner's dilemma: all boosting is an equilibrium, all GLM would pay every insurer more"
    if 0 in eq and pg[K4] > pl[0]:
        return "the reverse: all GLM is an equilibrium, all boosting would pay every insurer more"
    if 0 not in eq and K4 not in eq:
        return "mixed: every equilibrium has insurers of both classes"
    return "no conflict: the equilibrium pays every insurer at least as much as the other uniform market"


def curves(z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pg = np.array([np.mean(z[j]["G"]) if z[j]["G"] else np.nan for j in range(K4 + 1)])
    pl = np.array([np.mean(z[j]["L"]) if z[j]["L"] else np.nan for j in range(K4 + 1)])
    lv = np.array([np.mean(z[j]["level"]) for j in range(K4 + 1)])
    return pg, pl, lv


# ---------------------------------------------------------------------------------------------------
def run(P: dict, cfg: dict) -> dict:
    name = P["name"]
    sizes = cfg["sizes"]
    rng = np.random.default_rng(SEED + (0 if name == "freMTPL2" else 100))
    n_tr = P["tr"].size
    keys = GAMES
    two = {n: {key: {} for key in keys} for n in sizes}
    part2 = {n: {key: [] for key in keys} for n in sizes}
    four = {n: {key: [{j: {"G": [], "L": [], "level": []} for j in range(K4 + 1)} for _ in range(R)] for key in keys}
            for n in sizes if K4 * n <= n_tr}
    prof = {n: {h: {"GLM": [], "GBM": []} for h in HOWS} for n in sizes}
    conv = {key: [0, 0] for key in keys}
    gains = {(k, h, K): [] for k, h in GAMES if k != "one level" for K in (2, K4)}
    diag = []
    unsettled = []
    unsettled_gain = []
    left_out = {n: {key: 0 for key in GAMES} for n in sizes}

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

            if r == 0 and si == 0:
                # Curse-aware pricing by risk band, three plays on the smallest books of the first partition.
                ec, Nc = P["e"][P["ca"]], P["N"][P["ca"]]
                for s, t in (("GLM", "GLM"), ("GLM", "GBM"), ("GBM", "GBM")):
                    Rc = np.array([fits[(s, 0)]["ca"], fits[(t, 1)]["ca"]])
                    sc, _ = segment_index(Rc, Rc, "risk band")
                    _, ok, left = seg_nash(np.log(ec * Rc), sc, Nc, BETA, rounds=10, iters=300)
                    diag.append((s, t, ok, left / Nc.sum() * 100))

            cell = {key: {} for key in keys}
            for s, t in itertools.product(CLASSES, CLASSES):
                Rc = np.array([fits[(s, 0)]["ca"], fits[(t, 1)]["ca"]])
                Rt = np.array([fits[(s, 0)]["te"], fits[(t, 1)]["te"]])
                for kind, how in keys:
                    out = outcome(Rc, Rt, P, kind, how, check=(r == 0 and kind != "one level"))
                    conv[(kind, how)][0] += out["ok"]
                    conv[(kind, how)][1] += 1
                    if not out["ok"]:
                        # Not an equilibrium, and where its iterations stop depends on the last bits of the
                        # arithmetic, so it differs between machines: reported, and left out of the payoffs.
                        unsettled.append(f"{kind}, {how}, two insurers, books of {n:,}, {s} against {t}, partition {r + 1}")
                        unsettled_gain.append(out["gain"])
                        continue
                    if not np.isnan(out["gain"]):
                        gains[(kind, how, 2)].append(out["gain"])
                    two[n][(kind, how)].setdefault((s, t), []).append(
                        (out["profit"][0], out["profit"][1], out["level"], out["share"][0], out["disp"]))
                    cell[(kind, how)].setdefault((s, t), []).append(out["profit"][0])
                    cell[(kind, how)].setdefault((t, s), []).append(out["profit"][1])
                    if kind == "market position" and {s, t} == {"GLM", "GBM"}:
                        for i, c in enumerate((s, t)):
                            prof[n][how][c].append(np.exp(out["x"][i]))
            for key in keys:
                A = np.array([[np.mean(cell[key][(s, t)]) for t in PAIR] for s in PAIR])
                part2[n][key].append(A)

            if nb == K4:
                for G in itertools.product((0, 1), repeat=K4):
                    cls = ["GBM" if g else "GLM" for g in G]
                    Rc = np.array([fits[(c, j)]["ca"] for j, c in enumerate(cls)])
                    Rt = np.array([fits[(c, j)]["te"] for j, c in enumerate(cls)])
                    for kind, how in keys:
                        out = outcome(Rc, Rt, P, kind, how, check=(r == 0 and kind != "one level"))
                        conv[(kind, how)][0] += out["ok"]
                        conv[(kind, how)][1] += 1
                        if not out["ok"]:
                            unsettled.append(f"{kind}, {how}, four insurers, books of {n:,}, {'/'.join(cls)},"
                                             f" partition {r + 1}")
                            unsettled_gain.append(out["gain"])
                            left_out[n][(kind, how)] += 1
                            continue
                        if not np.isnan(out["gain"]):
                            gains[(kind, how, K4)].append(out["gain"])
                        z = four[n][(kind, how)][r][sum(G)]
                        z["G"] += [out["profit"][j] for j in range(K4) if G[j]]
                        z["L"] += [out["profit"][j] for j in range(K4) if not G[j]]
                        z["level"].append(out["level"])

    res = {"two": {}, "four": {}, "profile": {}}
    # -----------------------------------------------------------------------------------------------
    for kind, how in keys:
        header(f"{name}: two insurers, {kind}, {'curse-aware' if how == 'aware' else 'naive'} pricing, beta 10: payoff"
               f" of the row insurer against the column insurer (percent of the test sample's claims, mean of {2 * R}"
               f" plays), the price level policyholders pay, and the dispersion of the two quotes")
        for n in sizes:
            cells = two[n][(kind, how)]
            A = np.zeros((3, 3))
            lev = np.zeros((3, 3))
            dsp = np.zeros((3, 3))
            for a, s in enumerate(CLASSES):
                for b, t in enumerate(CLASSES):
                    A[a, b] = np.mean([x[0] for x in cells[(s, t)]] + [x[1] for x in cells[(t, s)]])
                    lev[a, b] = np.mean([x[2] for x in cells[(s, t)]] + [x[2] for x in cells[(t, s)]])
                    dsp[a, b] = np.mean([x[4] for x in cells[(s, t)]] + [x[4] for x in cells[(t, s)]])
            print(f"  books of {n:,}:")
            for a, s in enumerate(CLASSES):
                print(f"      {s} against " + ", ".join(f"{t} {A[a, b]:+.2f} (level {lev[a, b]:.3f}, quotes {dsp[a, b]:.3f})"
                                                  for b, t in enumerate(CLASSES)))
            A2 = A[:2, :2]
            pp = np.array(part2[n][(kind, how)])
            print(f"    GLM or GBM only: dominant {g2.dominant(A2, PAIR)}; equilibria"
                  f" {', '.join(g2.pure_equilibria(A2, PAIR))}; {kind_two(A2)}; by partition: GBM the better reply to"
                  f" GLM in {int(np.sum(pp[:, 1, 0] > pp[:, 0, 0]))} of {R}, to GBM in"
                  f" {int(np.sum(pp[:, 1, 1] > pp[:, 0, 1]))} of {R}, (GLM, GLM) pays more than (GBM, GBM) in"
                  f" {int(np.sum(pp[:, 0, 0] > pp[:, 1, 1]))} of {R}")
            print(f"    all three classes: dominant {g2.dominant(A, CLASSES)}; equilibria"
                  f" {', '.join(g2.pure_equilibria(A, CLASSES))}")
            res["two"][f"{kind}|{how}|{n}"] = {"A": A.round(4).tolist(), "level": lev.round(4).tolist(),
                                               "disp": dsp.round(4).tolist()}

    header(f"{name}: four insurers choosing between GLM and GBM, beta 10: payoff of a boosting and of a GLM insurer"
           f" (percent of the test sample's claims), by the number j of boosting insurers")
    for n in four:
        for kind, how in keys:
            zs = four[n][(kind, how)]
            pooled = {j: {k: sum((zs[r][j][k] for r in range(R)), []) for k in ("G", "L", "level")}
                      for j in range(K4 + 1)}
            pg, pl, lv = curves(pooled)
            mkt = np.array([(j * pg[j] if j > 0 else 0.0) + ((K4 - j) * pl[j] if j < K4 else 0.0)
                            for j in range(K4 + 1)])
            eq = g2.k_equilibria(pg, pl, K4)
            path, j = [0], 0
            while j < K4 and pg[j + 1] > pl[j]:
                j += 1
                path.append(j)
            kinds = []
            for r in range(R):
                a, b, _ = curves(zs[r])
                kinds.append(kind_four(a, b, g2.k_equilibria(a, b, K4)))
            counts = {k.split(":")[0]: kinds.count(k) for k in dict.fromkeys(kinds)}
            parts = []
            for j in range(K4 + 1):
                who = ([f"boosting {pg[j]:+.2f}"] if j > 0 else []) + ([f"GLM {pl[j]:+.2f}"] if j < K4 else [])
                parts.append(f"j={j}: {', '.join(who)}, level {lv[j]:.3f}")
            print(f"  books of {n:,}, {kind}, {how}: " + "; ".join(parts))
            print(f"      equilibria at j = {', '.join(map(str, eq))}; path from all GLM: j = {' -> '.join(map(str, path))};"
                  f" {kind_four(pg, pl, eq)}; by partition: " + ", ".join(f"{k} in {v} of {R}" for k, v in counts.items())
                  + (f"; left out because they did not settle: {left_out[n][(kind, how)]} of {2 ** K4 * R} plays"
                     if left_out[n][(kind, how)] else ""))
            res["four"][f"{kind}|{how}|{n}"] = {"pg": [None if np.isnan(v) else float(v) for v in pg],
                                                "pl": [None if np.isnan(v) else float(v) for v in pl],
                                                "level": lv.tolist(), "market": mkt.tolist(), "eq": eq, "path": path}

    header(f"{name}: how the insurers use market-position segments, in a market of one GLM and one boosting insurer"
           f" (beta 10): level by decile of the insurer's rate relative to its competitor's, cheapest decile first"
           f" (mean of {2 * R} plays)")
    for n in sizes:
        for how in HOWS:
            for c in ("GLM", "GBM"):
                L = np.mean(prof[n][how][c], axis=0)
                print(f"  books of {n:,}, {how}, {c} insurer: " + " ".join(f"{v:.3f}" for v in L))
                res["profile"][f"{how}|{c}|{n}"] = L.tolist()

    header(f"{name}: checks")
    try:
        ref = json.loads((HERE.parent / "companion-model-choice" / "figure_data.json").read_text())[name]
        diffs = []
        for n in sizes:
            for how in HOWS:
                mine = np.array(res["two"][f"one level|{how}|{n}"]["A"])
                diffs.append(np.max(np.abs(mine - np.array(ref[f"{how}10"][str(n)]["A"]))))
        for n in four:
            for how in HOWS:
                f = res["four"][f"one level|{how}|{n}"]
                r_ = ref["four"][f"{n}_{how}_10"]
                for a, b in ((f["pg"], r_["pg"]), (f["pl"], r_["pl"])):
                    diffs += [abs(u - v) for u, v in zip(a, b) if u is not None and v is not None]
        d = max(diffs)
        print(f"  one level against the second companion's tables (two and four insurers, beta 10): largest difference "
              + ("below 1e-9" if d < 1e-9 else f"{d:.2e}"))
    except FileNotFoundError:
        print("  second companion's figure_data.json not found; comparison skipped")
    for key in keys:
        print(f"  {key[0]}, {key[1]}: settled in {conv[key][0]} of {conv[key][1]} plays")
    print(f"  risk band with curse-aware pricing, two insurers, books of {sizes[0]:,}, partition 1 (ten rounds of"
          f" iterations and grid moves): " + "; ".join(
              f"{s} against {t}: " + ("settled" if ok else f"did not settle, largest remaining gain from moving one"
                                       f" segment level {bound(g)} percent of claims") for s, t, ok, g in diag))
    print("  plays that did not settle, left out of the payoffs above: " + ("none" if not unsettled else ""))
    for u in unsettled:
        print(f"    {u}")
    if unsettled_gain:
        print(f"  in those plays, the largest gain an insurer could still make by moving one segment level where the"
              f" iterations stop: {bound(max(unsettled_gain))} percent of claims")
    for (kind, how, K), gl in gains.items():
        if gl:
            gmax = max(gl)
            print(f"  {kind}, {how}, {K} insurers ({len(gl)} plays): largest gain any insurer could make by moving one"
                  f" segment level on a grid of +-0.3, in percent of claims: " + ("below 1e-6" if gmax < 1e-6 else f"{gmax:.6f}"))
    return res


def mechanism(P: dict) -> dict:
    """Tariffs that differ only in noise around a common rate (as in the second companion), priced with one level,
    by risk band and by market position."""
    e, N = P["e"], P["N"]
    tr, ca = P["tr"], P["ca"]
    b = c1.irls_ridge(P["X_glm"](tr), N[tr], e[tr])
    rate = np.exp(P["X_glm"](ca) @ b)
    c = e[ca] * rate
    Z = np.random.default_rng(SEED + 5).standard_normal((K4, c.size))
    header(f"{P['name']}: the mechanism, with tariffs that differ only in noise (beta 10; cost = the GLM rate on the"
           f" whole training pool; calibration sample, {c.size:,} policies): profit per insurer in percent of the"
           f" expected claims, price level, and the dispersion of the final quotes")
    lr = np.log(rate)
    spread = float(np.sqrt(np.average((lr - np.average(lr, weights=e[ca])) ** 2, weights=e[ca])))
    print(f"  spread of the true rates: standard deviation of the log rate across policies {spread:.3f}")
    out = {}
    for K in (2, K4):
        for how in HOWS:
            for kind in SEGMENTS:
                row = []
                for sg in (0.0, 0.1, 0.2, 0.3, 0.4):
                    T = rate * np.exp(sg * Z[:K] - sg ** 2 / 2)
                    seg, _ = segment_index(T, T, kind)
                    lp = np.log(e[ca] * T)
                    cost = c if how == "aware" else e[ca] * T
                    x, ok, _ = seg_nash(lp, seg, cost, BETA)
                    xe = x[np.arange(K)[:, None], seg]
                    s = _shares(lp, xe, BETA)
                    pr = (s * (np.exp(xe + lp) - c)).sum(axis=1) / c.sum() * 100
                    level = float((s * np.exp(xe + lp)).sum() / c.sum())
                    disp = float(np.sqrt(np.average((xe + np.log(T)).var(axis=0, ddof=1), weights=e[ca])))
                    row.append((sg, float(pr.mean()), level, disp, ok))
                out[f"{K}|{how}|{kind}"] = row
                print(f"  {K} insurers, {how}, {kind}: " + "; ".join(
                    f"sigma {sg:.1f}: profit {p:+.2f}, level {lv:.3f}, quotes {d:.3f}" + ("" if ok else " (not settled)")
                    for sg, p, lv, d, ok in row))
    sig = (0.1, 0.2, 0.3)
    for kind in SEGMENTS[1:]:
        A = np.zeros((3, 3))
        for a, sa in enumerate(sig):
            for bb, sb in enumerate(sig):
                T = np.vstack([rate * np.exp(sa * Z[0] - sa ** 2 / 2), rate * np.exp(sb * Z[1] - sb ** 2 / 2)])
                seg, _ = segment_index(T, T, kind)
                lp = np.log(e[ca] * T)
                x, _, _ = seg_nash(lp, seg, c, BETA)
                A[a, bb] = seg_profits(lp, seg, x, c, BETA)[0] / c.sum() * 100
        print(f"  two insurers, aware, {kind}: payoff of the row insurer by the noise in its own tariff (rows) and in"
              f" its competitor's (columns):")
        for a, sa in enumerate(sig):
            print(f"      sigma {sa:.1f} against " + ", ".join(f"{sb:.1f} {A[a, bb]:+.2f}" for bb, sb in enumerate(sig)))
        out[f"mat|{kind}"] = A.round(4).tolist()
    return out


def main() -> None:
    print("Third companion: the model-choice game when insurers price by segment, on freMTPL2 and beMTPL97.")
    quick = os.environ.get("SEG_QUICK")
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
