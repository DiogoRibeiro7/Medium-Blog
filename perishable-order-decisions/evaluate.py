"""The registered test (PLAN.md): 4 algorithms x 6 decision layers x 3 cost ratios on FreshRetailNet-50K.

python evaluate.py --data data --out results
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from feed import decide as K
from feed import forecast as F
from feed import recover as R
from feed.data import Panel, load

FIT_END, CAL_END = 76, 90  # fit: days 0..75, calibration: 76..89, test: 90..96
BOOT, BOOT_SEED = 2000, 11
ASYM = (0.7, 0.9)
LGBMS = ("lgbm_l2", "lgbm_tweedie")


def windows(T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    day = np.arange(T)
    return day < FIT_END, (day >= FIT_END) & (day < CAL_END), day >= CAL_END


def cell(algo: str, target: str, rule: str, tau: float) -> str:
    return f"{algo}|{target}|{rule}|{tau}"


def run(panel: Panel, settings: dict | None = None, boot: int = BOOT, keep_orders: bool = False) -> dict:
    fit, cal, test = windows(panel.days)
    if not np.array_equal(test, panel.test):
        raise ValueError("the test window is not the eval split")
    coding = R.stock_coding(panel)
    out = R.out_hours(panel, coding)
    prof = R.profiles(panel, fit)
    rec = R.recover(panel, out, prof)
    d = rec.demand[:, test]
    instock = panel.oos_6_22[:, test] == 0
    stores, store_idx = np.unique(panel.ids["store_id"], return_inverse=True)
    S = len(stores)

    def by_store(x: np.ndarray) -> np.ndarray:
        return np.bincount(store_idx, weights=x.sum(axis=1), minlength=S)

    demand_all = by_store(d)
    demand_in = by_store(np.where(instock, d, 0.0))
    names, sums_all, sums_in, rows, orders = [], [], [], [], []
    fit_days = list(range(F.MAX_LAG, FIT_END))
    predict_days = list(range(FIT_END, panel.days))
    for target in K.TARGETS:
        y = panel.sales if target == "sales" else rec.demand
        fc = F.forecasts(panel, y, fit_days, predict_days, settings)
        for algo in F.ALGOS:
            f = fc[algo]
            calib = K.calibrate(y, f, fit, cal)
            ft = f[:, test]
            wape = float(np.abs(d - ft).sum() / d.sum())
            bias = float((ft - d).sum() / d.sum())
            for rule, tau in product(K.RULES, K.TAUS):
                q = K.order(ft, rule, tau, calib)
                c = K.cost(q, d, tau)
                if keep_orders:
                    orders.append(q.astype(np.float32))
                names.append(cell(algo, target, rule, tau))
                sums_all.append(by_store(c))
                sums_in.append(by_store(np.where(instock, c, 0.0)))
                rows.append(
                    {
                        "algo": algo,
                        "target": target,
                        "rule": rule,
                        "tau": tau,
                        "cost": float(c.sum() / d.sum()),
                        "cost_instock": float(c[instock].sum() / d[instock].sum()),
                        "coverage": float(np.mean(d <= q)),
                        "order_over_demand": float(q.sum() / d.sum()),
                        "wape": wape,
                        "bias": bias,
                    }
                )
    A = np.array(sums_all).T  # stores x cells
    B = np.array(sums_in).T
    rng = np.random.default_rng(BOOT_SEED)
    W = rng.multinomial(S, np.full(S, 1 / S), size=boot).astype(np.float64)
    boot_all = (W @ A) / (W @ demand_all)[:, None]
    boot_in = (W @ B) / (W @ demand_in)[:, None]
    point_all = A.sum(axis=0) / demand_all.sum()
    point_in = B.sum(axis=0) / demand_in.sum()
    index = {n: i for i, n in enumerate(names)}
    for r, n in zip(rows, names, strict=True):
        i = index[n]
        r["lo"], r["hi"] = (float(x) for x in np.percentile(boot_all[:, i], [2.5, 97.5]))
    return {
        "coding": coding,
        "prof": prof,
        "rec": rec,
        "instock": instock,
        "rows": rows,
        "index": index,
        "point": {"all": point_all, "instock": point_in},
        "boot": {"all": boot_all, "instock": boot_in},
        "stores": S,
        "store_idx": store_idx,
        "orders": np.array(orders) if keep_orders else None,
    }


def comparisons(res: dict) -> list[dict]:
    """Every registered difference, as (larger cost) - (smaller cost) under the hypothesis: positive supports it."""
    ix = res["index"]
    out = []

    def c(conv: str, which: str, a: str, t: str, r: str, tau: float):
        key = cell(a, t, r, tau)
        return res[which][conv][..., ix[key]]

    def add(h: str, label: str, tau: float, fn) -> None:
        for conv in ("all", "instock"):
            est = float(fn(conv, "point"))
            b = fn(conv, "boot")
            lo, hi = (float(x) for x in np.percentile(b, [2.5, 97.5]))
            out.append(
                {
                    "hypothesis": h,
                    "comparison": label,
                    "tau": tau,
                    "convention": conv,
                    "estimate": est,
                    "lo": lo,
                    "hi": hi,
                    "established": lo > 0,
                }
            )

    for tau in ASYM:
        for m in LGBMS:
            add(
                "H1",
                f"{m} default - ma4w full",
                tau,
                lambda conv, w, m=m, tau=tau: c(conv, w, m, *K.DEFAULT, tau) - c(conv, w, "ma4w", *K.FULL, tau),
            )
        add(
            "H2",
            "layer share - algorithm share of the variation over the 4 x 6 table",
            tau,
            lambda conv, w, tau=tau: layer_minus_algorithm(table(res[w][conv], ix, tau)),
        )
        for a in F.ALGOS:
            for r in K.RULES:
                add(
                    "H3",
                    f"{a} {r}: sales - recovered",
                    tau,
                    lambda conv, w, a=a, r=r, tau=tau: (
                        c(conv, w, a, "sales", r, tau) - c(conv, w, a, "recovered", r, tau)
                    ),
                )
    return out


LAYERS = [(t, r) for t in K.TARGETS for r in K.RULES]


def table(costs: np.ndarray, ix: dict[str, int], tau: float) -> np.ndarray:
    """Costs (..., cells) arranged as (..., algorithms, layers) for one tau."""
    idx = np.array([[ix[cell(a, t, r, tau)] for t, r in LAYERS] for a in F.ALGOS])
    return costs[..., idx]


def shares(tab: np.ndarray) -> np.ndarray:
    """(..., 4, 6) -> (..., 3): layer, algorithm and remainder shares of the two-way additive decomposition."""
    m = tab.mean(axis=(-2, -1), keepdims=True)
    ss_total = ((tab - m) ** 2).sum(axis=(-2, -1))
    ss_algo = tab.shape[-1] * ((tab.mean(axis=-1, keepdims=True) - m) ** 2).sum(axis=(-2, -1))
    ss_layer = tab.shape[-2] * ((tab.mean(axis=-2, keepdims=True) - m) ** 2).sum(axis=(-2, -1))
    return np.stack([ss_layer, ss_algo, ss_total - ss_layer - ss_algo], axis=-1) / ss_total[..., None]


def layer_minus_algorithm(tab: np.ndarray) -> np.ndarray:
    s = shares(tab)
    return s[..., 0] - s[..., 1]


def decomposition(res: dict, tau: float) -> dict:
    layer, algo, rest = shares(table(res["point"]["all"], res["index"], tau))
    return {"tau": tau, "layer": float(layer), "algorithm": float(algo), "remainder": float(rest)}


def h4(rows: list[dict]) -> dict:
    full = [x for x in rows if (x["target"], x["rule"]) == K.FULL and x["tau"] == 0.9]
    by_cost = [x["algo"] for x in sorted(full, key=lambda x: x["cost"])]
    by_wape = [x["algo"] for x in sorted(full, key=lambda x: x["wape"])]
    return {"by_cost": by_cost, "by_wape": by_wape, "supported": by_cost != by_wape}


def verdicts(comp: list[dict]) -> dict[str, dict]:
    out = {}
    for h in ("H1", "H2", "H3"):
        sel = [x for x in comp if x["hypothesis"] == h and x["convention"] == "all"]
        rob = [x for x in comp if x["hypothesis"] == h and x["convention"] == "instock"]
        out[h] = {
            "n": len(sel),
            "established": sum(x["established"] for x in sel),
            "positive": sum(x["estimate"] > 0 for x in sel),
            "supported": all(x["established"] for x in sel),
            "instock_established": sum(x["established"] for x in rob),
        }
    return out


def recovery_lines(panel: Panel, res: dict) -> list[str]:
    fit, cal, test = windows(panel.days)
    rec, prof = res["rec"], res["prof"]
    oos = panel.oos_6_22 > 0
    lines = [
        f"hourly status coding: out of stock = {res['coding'].out_value}; stock_hour6_22_cnt counts hours "
        f"{res['coding'].first} to {res['coding'].last}; agreement {res['coding'].agreement:.4f}",
        f"days with out-of-stock hours between 6 and 22: fit {oos[:, fit].mean():.4f}, calibration "
        f"{oos[:, cal].mean():.4f}, test {oos[:, test].mean():.4f}",
        f"days with any out-of-stock hour (24 h): fit {rec.stockout[:, fit].mean():.4f}, test "
        f"{rec.stockout[:, test].mean():.4f}",
        f"days recovered with the fallback (observed share < {R.MIN_SHARE}): fit {rec.fallback[:, fit].mean():.4f}, "
        f"test {rec.fallback[:, test].mean():.4f}",
        f"profile source: own {np.mean(prof.source == 0):.4f}, third category {np.mean(prof.source == 1):.4f}, "
        f"all series {np.mean(prof.source == 2):.4f}",
    ]
    for name, w in (("fit", fit), ("calibration", cal), ("test", test)):
        s, dd = panel.sales[:, w].sum(), rec.demand[:, w].sum()
        lines.append(f"recovered demand / sales, {name}: {dd / s:.4f}")
    s, dd = panel.sales[:, test], rec.demand[:, test]
    lines.append(f"test: share of demand on days with out-of-stock hours 6-22: {dd[oos[:, test]].sum() / dd.sum():.4f}")
    return lines


def write(panel: Panel, res: dict, structure: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "structure.txt").write_text("\n".join(structure) + "\n", encoding="utf-8")
    rec_lines = recovery_lines(panel, res)
    (out / "recovery.txt").write_text("\n".join(rec_lines) + "\n", encoding="utf-8")
    rows = res["rows"]
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and k != "tau" else v) for k, v in r.items()})
    comp = comparisons(res)
    with open(out / "comparisons.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp[0]))
        w.writeheader()
        for r in comp:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and k != "tau" else v) for k, v in r.items()})
    ver = verdicts(comp)
    dec = [decomposition(res, tau) for tau in K.TAUS]
    four = h4(rows)
    fit, _, test = windows(panel.days)
    oos = panel.oos_6_22 > 0
    daily = [
        {
            "date": str(panel.dates[t]),
            "window": "fit" if fit[t] else ("test" if test[t] else "calibration"),
            "sales": float(panel.sales[:, t].sum()),
            "recovered": float(res["rec"].demand[:, t].sum()),
            "stockout_share": float(oos[:, t].mean()),
            "fallback_share": float(res["rec"].fallback[:, t].mean()),
        }
        for t in range(panel.days)
    ]
    with open(out / "days.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(daily[0]))
        w.writeheader()
        w.writerows(daily)
    out_h = R.out_hours(panel, res["coding"])
    clean = (panel.oos_6_22 == 0) & fit[None, :]
    hs = (panel.hours_sale * clean[:, :, None]).sum(axis=(0, 1), dtype=np.float64)
    hours = [
        {"hour": h, "sales_share_clean_days": float(hs[h] / hs.sum()), "out_share_fit": float(out_h[:, fit, h].mean())}
        for h in range(24)
    ]
    with open(out / "hours.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hours[0]))
        w.writeheader()
        w.writerows(hours)
    np.savez_compressed(
        out / "bootstrap.npz",
        cells=np.array(list(res["index"])),
        all=res["boot"]["all"].astype(np.float32),
        instock=res["boot"]["instock"].astype(np.float32),
    )

    L = ["Algorithms against decision layers for daily perishable orders, FreshRetailNet-50K. Settings: PLAN.md.", ""]
    L += structure + [""] + rec_lines + [""]
    for tau in K.TAUS:
        L.append(f"tau = {tau}: cost per unit of demand (95% store-bootstrap interval)")
        for a in F.ALGOS:
            parts = []
            for t, r in ((t, r) for t in K.TARGETS for r in K.RULES):
                x = next(x for x in rows if (x["algo"], x["target"], x["rule"], x["tau"]) == (a, t, r, tau))
                parts.append(f"{t}/{r} {x['cost']:.4f}")
            L.append(f"  {a}: " + "; ".join(parts))
        dd = next(x for x in dec if x["tau"] == tau)
        L.append(
            f"  variation over the 4 x 6 table: algorithm {dd['algorithm']:.3f}, layer {dd['layer']:.3f}, "
            f"remainder {dd['remainder']:.3f}"
        )
        L.append("")
    L.append("Point forecasts on the test days, against demand: WAPE and bias")
    for t in K.TARGETS:
        for a in F.ALGOS:
            x = next(x for x in rows if (x["algo"], x["target"]) == (a, t))
            L.append(f"  {a} trained on {t}: WAPE {x['wape']:.4f}, bias {x['bias']:+.4f}")
    L.append("")
    for h, txt in (
        ("H1", "ma4w with the full layer costs less than each LightGBM model with the default layer"),
        ("H2", "the decision layer explains more of the variation in cost than the algorithm"),
        ("H3", "training on recovered demand costs less than on sales, every algorithm and rule"),
    ):
        v = ver[h]
        L.append(
            f"{h} ({txt}, tau 0.7 and 0.9): {'supported' if v['supported'] else 'not supported'} "
            f"({v['established']} of {v['n']} established, {v['positive']} of {v['n']} in the stated direction; "
            f"in-stock days only: {v['instock_established']} of {v['n']} established)"
        )
        for x in (x for x in comp if x["hypothesis"] == h and x["convention"] == "all"):
            L.append(f"    tau {x['tau']}: {x['comparison']}: {x['estimate']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]")
    L.append(
        f"H4 (cost order differs from WAPE order, full layer, tau 0.9): "
        f"{'supported' if four['supported'] else 'not supported'}; by cost {', '.join(four['by_cost'])}; "
        f"by WAPE {', '.join(four['by_wape'])}"
    )
    (out / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "verdicts.json").write_text(
        json.dumps({"verdicts": ver, "h4": four, "decomposition": dec}, indent=1), encoding="utf-8"
    )
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--orders", type=Path, help="also save every test order here, for details.py (not committed)")
    a = ap.parse_args()
    panel, structure = load(a.data)
    res = run(panel, keep_orders=a.orders is not None)
    write(panel, res, structure, a.out)
    if a.orders is not None:
        _, _, test = windows(panel.days)
        np.savez(
            a.orders,
            cells=np.array(list(res["index"])),
            orders=res["orders"],
            demand=res["rec"].demand[:, test],
            sales=panel.sales[:, test],
            instock=res["instock"],
            store_idx=res["store_idx"],
        )


if __name__ == "__main__":
    main()
