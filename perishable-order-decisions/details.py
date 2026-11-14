"""Descriptive analyses added after the results (PLAN.md, "Descriptive analyses added after the results").

python details.py --orders data/orders.npz --results results --out results/details.txt

Not part of the registered test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from evaluate import BOOT, BOOT_SEED, cell, comparisons, shares
from feed import decide as K
from feed import forecast as F

STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
QUANTILE_LAYERS = [(t, r) for t in K.TARGETS for r in ("pooled", "series")]


def costs_by_store(orders: np.ndarray, cells: list[str], demand: np.ndarray, store_idx: np.ndarray, S: int):
    """Store totals of cost for every cell, and store totals of demand."""
    out = np.empty((S, len(cells)))
    for i, name in enumerate(cells):
        tau = float(name.split("|")[3])
        c = K.cost(orders[i].astype(np.float64), demand, tau).sum(axis=1)
        out[:, i] = np.bincount(store_idx, weights=c, minlength=S)
    return out, np.bincount(store_idx, weights=demand.sum(axis=1), minlength=S)


def as_result(A: np.ndarray, dem: np.ndarray, W: np.ndarray, index: dict[str, int]) -> dict:
    point = A.sum(axis=0) / dem.sum()
    boot = (W @ A) / (W @ dem)[:, None]
    return {"index": index, "point": {"all": point, "instock": point}, "boot": {"all": boot, "instock": boot}}


def sub_table(costs: np.ndarray, index: dict[str, int], tau: float, layers) -> np.ndarray:
    idx = np.array([[index[cell(a, t, r, tau)] for t, r in layers] for a in F.ALGOS])
    return costs[..., idx]


def three_way(costs: np.ndarray, index: dict[str, int], tau: float) -> dict[str, float]:
    """Additive decomposition of the 4 x 2 x 3 table (algorithm, target, rule)."""
    t = np.array([[[costs[index[cell(a, g, r, tau)]] for r in K.RULES] for g in K.TARGETS] for a in F.ALGOS])
    m = t.mean()
    total = ((t - m) ** 2).sum()
    n = t.size
    out = {}
    for name, axis in (("algorithm", (1, 2)), ("target", (0, 2)), ("rule", (0, 1))):
        eff = t.mean(axis=axis) - m
        out[name] = float((n / eff.size) * (eff**2).sum() / total)
    out["remainder"] = 1 - sum(out.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orders", type=Path, default=Path("data/orders.npz"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/details.txt"))
    a = ap.parse_args()
    z = np.load(a.orders)
    cells = [str(c) for c in z["cells"]]
    index = {c: i for i, c in enumerate(cells)}
    orders, demand, sales, store_idx = z["orders"], z["demand"], z["sales"], z["store_idx"]
    S = int(store_idx.max()) + 1
    rng = np.random.default_rng(BOOT_SEED)
    W = rng.multinomial(S, np.full(S, 1 / S), size=BOOT).astype(np.float64)
    with open(a.results / "cells.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    L = ["Descriptive analyses added after the results (not part of the registered test).", ""]
    js: dict = {"strength": []}

    L.append(
        "1. Strength of the recovery. Demand d(s) = sales + s x (recovered - sales) on the test days: s = 0 treats"
    )
    L.append("   recorded sales as demand (a lower bound on censored days), s = 1 is the registered evaluation.")
    for s in STRENGTHS:
        d = sales + s * (demand - sales)
        A, dem = costs_by_store(orders, cells, d, store_idx, S)
        res = as_result(A, dem, W, index)
        comp = [x for x in comparisons(res) if x["convention"] == "all"]
        L.append(f"   s = {s}: demand / sales {d.sum() / sales.sum():.4f}")
        entry = {"s": s, "comparisons": comp, "costs": {c: float(res["point"]["all"][i]) for c, i in index.items()}}
        js["strength"].append(entry)
        for h in ("H1", "H2", "H3"):
            sel = [x for x in comp if x["hypothesis"] == h]
            L.append(
                f"     {h}: {sum(x['established'] for x in sel)} of {len(sel)} established, "
                f"{sum(x['estimate'] > 0 for x in sel)} of {len(sel)} in the stated direction"
            )
        for x in comp:
            if x["hypothesis"] in ("H1", "H2") or (x["hypothesis"] == "H3" and "lgbm_l2" in x["comparison"]):
                L.append(
                    f"       tau {x['tau']}: {x['comparison']}: {x['estimate']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"
                )
    L.append("")

    A, dem = costs_by_store(orders, cells, demand, store_idx, S)
    res = as_result(A, dem, W, index)
    L.append("2. Variation over algorithms and layers with the point rule left out (4 algorithms x 4 quantile layers)")
    js["quantile_only"] = []
    for tau in K.TAUS:
        sh = shares(sub_table(res["point"]["all"], index, tau, QUANTILE_LAYERS))
        b = shares(sub_table(res["boot"]["all"], index, tau, QUANTILE_LAYERS))
        diff = b[:, 0] - b[:, 1]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        L.append(
            f"   tau {tau}: layer {sh[0]:.3f}, algorithm {sh[1]:.3f}, remainder {sh[2]:.3f}; "
            f"layer - algorithm {sh[0] - sh[1]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        )
        js["quantile_only"].append({"tau": tau, "layer": float(sh[0]), "algorithm": float(sh[1])})
    L.append("")

    L.append(
        "3. Three-way additive decomposition of the 4 x 2 x 3 table (algorithm, target, rule), shares of variation"
    )
    js["three_way"] = []
    for tau in K.TAUS:
        tw = three_way(res["point"]["all"], index, tau)
        L.append("   tau " + str(tau) + ": " + ", ".join(f"{k} {v:.3f}" for k, v in tw.items()))
        js["three_way"].append({"tau": tau, **tw})
    L.append("")

    L.append("4. Order of the algorithms by cost and by WAPE under each layer at tau 0.9")
    for t in K.TARGETS:
        for r in K.RULES:
            sel = [x for x in rows if x["target"] == t and x["rule"] == r and float(x["tau"]) == 0.9]
            by_cost = [x["algo"] for x in sorted(sel, key=lambda x: float(x["cost"]))]
            by_wape = [x["algo"] for x in sorted(sel, key=lambda x: float(x["wape"]))]
            same = "same" if by_cost == by_wape else "different"
            L.append(f"   {t}/{r}: by cost {', '.join(by_cost)}; by WAPE {', '.join(by_wape)} ({same})")
    L.append("")

    L.append("5. Coverage (share of test series-days with demand at or below the order), LightGBM l2 and ma4w")
    for x in rows:
        if x["algo"] in ("lgbm_l2", "ma4w") and x["rule"] != "point" and float(x["tau"]) in (0.7, 0.9):
            L.append(f"   {x['algo']} {x['target']}/{x['rule']} tau {x['tau']}: coverage {float(x['coverage']):.3f}")
    L.append("")

    L.append("6. Cost of ordering the forecast over the cost of the per-series quantile, models trained on recovered")
    L.append("   demand, against phi(0) / phi(z_tau), the same ratio for unbiased normal errors")
    nd = NormalDist()
    js["point_penalty"] = []
    for tau in K.TAUS:
        gauss = nd.pdf(0) / nd.pdf(nd.inv_cdf(tau))
        parts = []
        for alg in F.ALGOS:
            c = {
                r: float(
                    next(
                        x["cost"]
                        for x in rows
                        if (x["algo"], x["target"], x["rule"], float(x["tau"])) == (alg, "recovered", r, tau)
                    )
                )
                for r in ("point", "series")
            }
            parts.append(f"{alg} {c['point'] / c['series']:.3f}")
            js["point_penalty"].append({"tau": tau, "algo": alg, "ratio": c["point"] / c["series"], "gauss": gauss})
        L.append(f"   tau {tau}: normal {gauss:.3f}; " + "; ".join(parts))
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (a.out.parent / "details.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
