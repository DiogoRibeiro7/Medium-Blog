"""The registered test: score the nine candidates under S0 to S5 and decide H1 to H3.

Reads `<data>/forecasts.npz` and the sales, writes `<out>/summary.txt` (every registered
number), `<out>/metrics.csv`, `<out>/robustness.csv`, `<out>/by_quarter.csv` and
`<out>/series_metrics.npz` (per-series totals used by the descriptive analyses).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import success_core as sc

TAU = 0.8
B = 1000


class Boot:
    """Product-level bootstrap: totals are summed per product, products are resampled."""

    def __init__(self, item: np.ndarray, seed: int = 11, reps: int = B):
        self.item = item
        self.n_items = int(item.max()) + 1
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.n_items, size=(reps, self.n_items))
        self.counts = np.zeros((reps, self.n_items))
        for r in range(reps):
            self.counts[r] = np.bincount(draws[r], minlength=self.n_items)

    def per_item(self, v: np.ndarray) -> np.ndarray:
        return np.bincount(self.item, weights=v, minlength=self.n_items)

    def ratio(self, num: np.ndarray, den: np.ndarray) -> tuple[float, np.ndarray]:
        """Point estimate and bootstrap draws of sum(num) / sum(den)."""
        ni, di = self.per_item(num), self.per_item(den)
        return float(num.sum() / den.sum()), (self.counts @ ni) / (self.counts @ di)


def ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    f = np.load(args.data / "forecasts.npz", allow_pickle=True)
    p = sc.load_panel(args.raw)
    assert np.array_equal(p.ids, f["ids"]), "series order differs from the forecast run"
    y = p.sales
    n = p.n
    origins = list(f["origins"])
    pos = {o: i for i, o in enumerate(origins)}
    test_idx = np.array([pos[o] for o in sc.TEST_ORIGINS])
    act = np.stack([np.stack(sc.week_totals(y, o)) for o in origins], axis=2)  # (2, n, no)
    act_a, act_b = act[0], act[1]
    wk = {m: f[f"wk__{m}"] for m in sc.ALL_MODELS}
    boot = Boot(p.item)
    lines: list[str] = []
    rows: list[dict] = []
    series: dict[str, np.ndarray] = {}

    def add(spec: str, model: str, num: np.ndarray, den: np.ndarray, store: bool = True) -> tuple[float, np.ndarray]:
        est, draws = boot.ratio(num, den)
        lo, hi = ci(draws)
        if store:
            rows.append({"spec": spec, "model": model, "value": est, "lo": lo, "hi": hi})
            series[f"{spec}__{model}__num"] = num
            series[f"{spec}__{model}__den"] = den
        return est, draws

    res: dict[str, dict[str, tuple[float, np.ndarray]]] = {s: {} for s in ["S0", "S1", "S2", "S3", "S4"]}

    # S0: daily WAPE, h = 1..28 at the 13 S0 origins
    s0_act = np.stack([y[:, o:o + 28] for o in sc.S0_ORIGINS], axis=1)  # (n, 13, 28)
    for m in sc.DAILY_MODELS:
        e = np.abs(f[f"s0__{m}"] - s0_act)
        res["S0"][m] = add("S0", m, e.sum(axis=(1, 2)), s0_act.sum(axis=(1, 2)))

    # S1: week B WAPE over the 52 test origins
    den_b = act_b[:, test_idx].sum(axis=1)
    for m in sc.ALL_MODELS:
        e = np.abs(wk[m][1][:, test_idx] - act_b[:, test_idx])
        res["S1"][m] = add("S1", m, e.sum(axis=1), den_b)

    # S2: cost of the week B order at tau = 0.8
    def orders_for(m: str, tau: float, adjust_q80: bool = False) -> np.ndarray:
        if m == "lgbw_q80" and not adjust_q80:
            return wk[m][1][:, test_idx]
        od = sc.residual_quantile_orders(wk[m][1], act_b, origins, tau)
        return np.stack([od[o] for o in sc.TEST_ORIGINS], axis=1)

    orders = {m: orders_for(m, TAU) for m in sc.ALL_MODELS}
    for m in sc.ALL_MODELS:
        c = sc.pinball(orders[m], act_b[:, test_idx], TAU)
        res["S2"][m] = add("S2", m, c.sum(axis=1), den_b)

    # S3: share of series-quarters with more than one bad week
    q_of = np.repeat(np.arange(4), 13)
    for m in sc.ALL_MODELS:
        bad = sc.bad_weeks(wk[m][1][:, test_idx], act_b[:, test_idx])
        per_q = np.stack([bad[:, q_of == q].sum(axis=1) for q in range(4)], axis=1)
        res["S3"][m] = add("S3", m, (per_q > 1).sum(axis=1).astype(float), np.full(n, 4.0))

    # S4: revision of the same week between consecutive origins
    pairs = [(pos[o], pos[o + 7]) for o in sc.TEST_ORIGINS if o + 7 in pos]
    i0 = np.array([a for a, _ in pairs])
    i1 = np.array([b for _, b in pairs])
    den_rev = act_b[:, i0].sum(axis=1)
    for m in sc.ALL_MODELS:
        rev = np.abs(wk[m][1][:, i0] - wk[m][0][:, i1])
        res["S4"][m] = add("S4", m, rev.sum(axis=1), den_rev)

    point = {s: {m: v[0] for m, v in d.items()} for s, d in res.items()}
    winner = {s: min(d, key=d.get) for s, d in point.items()}

    # S5: lowest S2 among candidates no worse than the incumbent on S3 and S4
    inc = sc.INCUMBENT
    admissible = [m for m in sc.ALL_MODELS
                  if point["S3"][m] <= point["S3"][inc] and point["S4"][m] <= point["S4"][inc]]
    s5 = min(admissible, key=lambda m: point["S2"][m]) if admissible else inc

    lines.append("Success-specification test on M5 FOODS, 10 stores")
    lines.append(f"series: {n}; products: {boot.n_items}; test origins: {len(sc.TEST_ORIGINS)}; bootstrap: {B}")
    lines.append(f"week B sales over the test origins: {den_b.sum():,.0f} units")
    lines.append("")
    names = {"S0": "S0 default: daily WAPE, h 1-28", "S1": "S1 unit and timing: week B WAPE",
             "S2": "S2 cost: pinball loss at 0.8 of the week B order, per unit sold",
             "S3": "S3 tolerance: share of series-quarters with more than one bad week",
             "S4": "S4 stability: revisions per unit sold"}
    for s in ["S0", "S1", "S2", "S3", "S4"]:
        lines.append(names[s])
        for m in sorted(point[s], key=point[s].get):
            lo, hi = ci(res[s][m][1])
            lines.append(f"  {m:13s} {point[s][m]:.4f}  [{lo:.4f}, {hi:.4f}]")
        lines.append(f"  winner: {winner[s]}")
        lines.append("")
    lines.append(f"S5 full specification: admissible (no worse than {inc} on S3 and S4): {', '.join(admissible)}")
    lines.append(f"  winner: {s5}")
    lines.append("")

    # H1
    w0 = winner["S0"]
    h1_specs = []
    lines.append(f"H1: S0 winner is {w0}")
    for s in ["S1", "S2", "S3", "S4"]:
        if winner[s] == w0:
            lines.append(f"  {s}: same winner")
            continue
        diff = res[s][w0][1] - res[s][winner[s]][1]
        est = point[s][w0] - point[s][winner[s]]
        lo, hi = ci(diff)
        ok = lo > 0
        h1_specs.append(s) if ok else None
        lines.append(f"  {s}: winner {winner[s]}; {w0} minus winner {est:.4f} [{lo:.4f}, {hi:.4f}] "
                     f"({est / point[s][winner[s]]:.1%} of the winner's value); established: {ok}")
    lines.append(f"  H1 supported: {len(h1_specs) >= 2} ({len(h1_specs)} specifications: {', '.join(h1_specs)})")
    lines.append("")

    # H2
    I_est, I_d = res["S2"][inc]
    D_est, D_d = res["S2"][w0]
    lines.append("H2: captured share of the S5 gain over the incumbent when choosing by S0")
    if s5 == inc:
        lines.append("  not testable: the S5 winner is the incumbent")
    else:
        W_est, W_d = res["S2"][s5]
        lines.append(f"  I (incumbent {inc}) {I_est:.4f}; W (S5 winner {s5}) {W_est:.4f}; D (S0 winner {w0}) {D_est:.4f}")
        lines.append(f"  S0 winner admissible under S5: {w0 in admissible}")
        if w0 in admissible:
            cap = (I_est - D_est) / (I_est - W_est)
            cap_d = (I_d - D_d) / (I_d - W_d)
            lo, hi = ci(cap_d)
            lines.append(f"  captured share {cap:.3f} [{lo:.3f}, {hi:.3f}]")
            lines.append(f"  H2 supported: {cap < 0.5 and hi < 0.5}")
        else:
            lines.append("  captured share 0 (the planner keeps the incumbent)")
            lines.append("  H2 supported: True")
    w2 = winner["S2"]
    W2_est, W2_d = res["S2"][w2]
    if w2 != inc:
        cap_u = (I_est - D_est) / (I_est - W2_est)
        lo, hi = ci((I_d - D_d) / (I_d - W2_d))
        lines.append(f"  unconstrained captured share (S2 winner {w2}): {cap_u:.3f} [{lo:.3f}, {hi:.3f}]")
    lines.append("")

    # H3
    models = sc.ALL_MODELS
    tau_est = sc.kendall_tau(np.array([point["S1"][m] for m in models]), np.array([point["S4"][m] for m in models]))
    m1 = np.stack([res["S1"][m][1] for m in models], axis=1)
    m4 = np.stack([res["S4"][m][1] for m in models], axis=1)
    tau_d = np.array([sc.kendall_tau(a, b) for a, b in zip(m1, m4)])
    lo, hi = ci(tau_d)
    lines.append(f"H3: Kendall tau between the S1 and S4 orderings: {tau_est:.3f} [{lo:.3f}, {hi:.3f}]")
    lines.append(f"  H3 supported: {tau_est < 0 and hi < 0}")
    lines.append("")

    # Also reported: bias and coverage
    lines.append("Week B bias (forecast minus sales, over sales) and S2 order coverage")
    for m in models:
        fb = wk[m][1][:, test_idx]
        ab = act_b[:, test_idx]
        bias_est, bias_d = add("bias", m, (fb - ab).sum(axis=1), den_b)
        fill = add("fill", m, np.minimum(orders[m], ab).sum(axis=1), den_b)[0]
        cov = add("covered_weeks", m, (ab <= orders[m]).sum(axis=1).astype(float), np.full(n, len(test_idx)))[0]
        lo, hi = ci(bias_d)
        lines.append(f"  {m:13s} bias {bias_est:+.3f} [{lo:+.3f}, {hi:+.3f}]; fill rate {fill:.3f}; "
                     f"weeks with sales at or below the order {cov:.3f}")
    lines.append("")

    # Robustness, not deciding
    rob: list[dict] = []
    lines.append("Robustness (not deciding)")
    for tau in (0.7, 0.9):
        vals = {}
        for m in models:
            o = orders_for(m, tau, adjust_q80=True)
            vals[m] = add(f"S2_tau{tau}", m, sc.pinball(o, act_b[:, test_idx], tau).sum(axis=1), den_b)[0]
        w = min(vals, key=vals.get)
        rob.append({"check": f"S2 at tau {tau}", "winner": w, "s0_winner_value": vals[w0], "winner_value": vals[w]})
        lines.append(f"  S2 at tau {tau}: winner {w} {vals[w]:.4f}; {w0} {vals[w0]:.4f}; {inc} {vals[inc]:.4f}")
    for units, share in ((3, 0.3), (10, 0.7)):
        vals = {}
        for m in models:
            bad = sc.bad_weeks(wk[m][1][:, test_idx], act_b[:, test_idx], units, share)
            per_q = np.stack([bad[:, q_of == q].sum(axis=1) for q in range(4)], axis=1)
            vals[m] = add(f"S3_{units}_{share}", m, (per_q > 1).sum(axis=1).astype(float), np.full(n, 4.0))[0]
        w = min(vals, key=vals.get)
        rob.append({"check": f"S3 at ({units}, {share})", "winner": w, "s0_winner_value": vals[w0], "winner_value": vals[w]})
        lines.append(f"  S3 at ({units} units, {share:.0%}): winner {w} {vals[w]:.4f}; {w0} {vals[w0]:.4f}; {inc} {vals[inc]:.4f}")
    vals = {}
    for m in sc.DAILY_MODELS:
        e2 = (f[f"s0__{m}"] - s0_act) ** 2
        vals[m] = np.sqrt(e2.sum() / e2.size)
        rows.append({"spec": "S0_rmse", "model": m, "value": vals[m], "lo": np.nan, "hi": np.nan})
    w = min(vals, key=vals.get)
    rob.append({"check": "S0 with RMSE", "winner": w, "s0_winner_value": vals[w0], "winner_value": vals[w]})
    lines.append(f"  S0 with RMSE: winner {w} {vals[w]:.4f}; {w0} {vals[w0]:.4f}")
    vals = {}
    for m in models:
        o = wk[m][1][:, test_idx]
        vals[m] = add("S2_as_delivered", m, sc.pinball(o, act_b[:, test_idx], TAU).sum(axis=1), den_b)[0]
    w = min(vals, key=vals.get)
    rob.append({"check": "S2 as delivered", "winner": w, "s0_winner_value": vals[w0], "winner_value": vals[w]})
    lines.append(f"  S2 as delivered (no adjustment): winner {w} {vals[w]:.4f}; {w0} {vals[w0]:.4f}; {inc} {vals[inc]:.4f}")
    lines.append("")

    # By quarter
    qrows = []
    lines.append("Winners by test quarter (S1 to S4)")
    for q in range(4):
        sel = test_idx[q_of == q]
        dq = act_b[:, sel].sum(axis=1)
        vals = {s: {} for s in ["S1", "S2", "S3", "S4"]}
        for m in models:
            vals["S1"][m] = np.abs(wk[m][1][:, sel] - act_b[:, sel]).sum() / dq.sum()
            vals["S2"][m] = sc.pinball(orders[m][:, q_of == q], act_b[:, sel], TAU).sum() / dq.sum()
            bad = sc.bad_weeks(wk[m][1][:, sel], act_b[:, sel])
            vals["S3"][m] = float((bad.sum(axis=1) > 1).mean())
            qp = [(pos[o], pos[o + 7]) for o in np.array(sc.TEST_ORIGINS)[q_of == q] if o + 7 in pos]
            a0 = np.array([a for a, _ in qp])
            a1 = np.array([b for _, b in qp])
            vals["S4"][m] = np.abs(wk[m][1][:, a0] - wk[m][0][:, a1]).sum() / act_b[:, a0].sum()
            for s in vals:
                qrows.append({"quarter": q + 1, "spec": s, "model": m, "value": vals[s][m]})
        lines.append("  Q%d: " % (q + 1) + "; ".join(f"{s} {min(v, key=v.get)}" for s, v in vals.items()))

    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(args.out / "metrics.csv", index=False)
    pd.DataFrame(rob).to_csv(args.out / "robustness.csv", index=False)
    pd.DataFrame(qrows).to_csv(args.out / "by_quarter.csv", index=False)
    np.savez_compressed(args.out / "series_metrics.npz", item=p.item, store=p.store, dept=p.dept, **series)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
