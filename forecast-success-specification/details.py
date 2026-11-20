"""Descriptive analyses added after the registered results (20 November 2026).

Not part of the registered test and labelled as such in the article. Reads the forecasts
and the sales, writes `<out>/details.txt` and the CSV files the figures use. Nothing
registered is recomputed differently here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import success_core as sc
from evaluate import TAU, Boot, ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    f = np.load(args.data / "forecasts.npz", allow_pickle=True)
    p = sc.load_panel(args.raw)
    y = p.sales
    origins = list(f["origins"])
    pos = {o: i for i, o in enumerate(origins)}
    test_idx = np.array([pos[o] for o in sc.TEST_ORIGINS])
    act = np.stack([np.stack(sc.week_totals(y, o)) for o in origins], axis=2)
    act_a, act_b = act[0], act[1]
    ab = act_b[:, test_idx]
    wk = {m: f[f"wk__{m}"].astype(np.float64) for m in sc.ALL_MODELS}
    pairs = [(pos[o], pos[o + 7]) for o in sc.TEST_ORIGINS if o + 7 in pos]
    i0 = np.array([a for a, _ in pairs])
    i1 = np.array([b for _, b in pairs])
    q_of = np.repeat(np.arange(4), 13)
    boot = Boot(p.item)
    out: list[str] = []

    def metrics(fa: np.ndarray, fb: np.ndarray, order_fb: np.ndarray | None = None, convert: bool = True) -> dict:
        """S1 to S4 for week totals fa, fb (series x all origins). The order is built from order_fb."""
        ob = fb if order_fb is None else order_fb
        if convert:
            od = sc.residual_quantile_orders(ob, act_b, origins, TAU)
            orders = np.stack([od[o] for o in sc.TEST_ORIGINS], axis=1)
        else:
            orders = ob[:, test_idx]
        bad = sc.bad_weeks(fb[:, test_idx], ab)
        per_q = np.stack([bad[:, q_of == q].sum(axis=1) for q in range(4)], axis=1)
        return {
            "S1": np.abs(fb[:, test_idx] - ab).sum() / ab.sum(),
            "S2": sc.pinball(orders, ab, TAU).sum() / ab.sum(),
            "S3": float((per_q > 1).mean()),
            "S4": np.abs(fb[:, i0] - fa[:, i1]).sum() / act_b[:, i0].sum(),
            "bias": (fb[:, test_idx] - ab).sum() / ab.sum(),
        }

    base = {m: metrics(wk[m][0], wk[m][1], convert=(m != "lgbw_q80")) for m in sc.ALL_MODELS}
    inc = base[sc.INCUMBENT]

    # (a) the stability allowance: S5 with S4 <= (1 + d) x incumbent
    out.append("(a) S5 with a stability allowance d: admissible if S3 <= incumbent and S4 <= (1 + d) x incumbent")
    frontier = []
    last = None
    for d in np.round(np.arange(0, 0.6001, 0.005), 3):
        adm = [m for m in sc.ALL_MODELS if base[m]["S3"] <= inc["S3"] and base[m]["S4"] <= (1 + d) * inc["S4"]]
        w = min(adm, key=lambda m: base[m]["S2"])
        frontier.append({"allowance": d, "winner": w, "S2": base[w]["S2"]})
        if w != last:
            out.append(f"  from d = {d:.3f}: {w}, S2 {base[w]['S2']:.4f}, S4 {base[w]['S4']:.4f}")
            last = w
    for m in sc.ALL_MODELS:
        out.append(f"  {m:13s} S4 / incumbent S4 = {base[m]['S4'] / inc['S4']:.3f}; "
                   f"S3 {'<=' if base[m]['S3'] <= inc['S3'] else '>'} incumbent")
    pd.DataFrame(frontier).to_csv(args.out / "stability_allowance.csv", index=False)
    out.append("")

    # (b) week B bias by intermittency
    zero_share = (y[:, sc.FIRST_ORIGIN - 364: sc.FIRST_ORIGIN] == 0).mean(axis=1)
    bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    out.append("(b) Week B bias and daily WAPE by share of zero days in the year before the first origin")
    s0_act = np.stack([y[:, o:o + 28] for o in sc.S0_ORIGINS], axis=1)
    rows = []
    for lo, hi in bins:
        sel = (zero_share >= lo) & (zero_share < hi)
        line = f"  zeros {lo:.0%}-{min(hi, 1):.0%}: {sel.sum()} series, {ab[sel].sum() / ab.sum():.1%} of week B sales;"
        for m in ["lgb_l1", "lgb_tweedie", "ma28", "lgbw_tweedie"]:
            bias = (wk[m][1][sel][:, test_idx] - ab[sel]).sum() / ab[sel].sum()
            row = {"zeros": f"{lo:.0%}-{min(hi, 1):.0%}", "model": m, "series": int(sel.sum()), "bias": bias}
            if m in sc.DAILY_MODELS:
                e = np.abs(f[f"s0__{m}"][sel] - s0_act[sel]).sum() / s0_act[sel].sum()
                row["daily_wape"] = e
            rows.append(row)
            line += f" {m} {bias:+.3f}"
        out.append(line)
    for lo, hi in bins:
        sel = (zero_share >= lo) & (zero_share < hi)
        med_l1 = np.median(f["s0__lgb_l1"][sel])
        share_zero = float((f["s0__lgb_l1"][sel] < 0.5).mean())
        out.append(f"  zeros {lo:.0%}-{min(hi, 1):.0%}: lgb_l1 daily forecasts below 0.5 units {share_zero:.1%} "
                   f"(median {med_l1:.2f}); daily sales of zero {(s0_act[sel] == 0).mean():.1%}")
    pd.DataFrame(rows).to_csv(args.out / "bias_by_intermittency.csv", index=False)
    out.append("")

    # (c) captured share for every possible default pick, against the S2 winner
    out.append("(c) Captured share of the S2 gain over the incumbent, (I - D) / (I - W2), for each daily model as the pick")
    s2_num = {m: sc.pinball(np.stack([sc.residual_quantile_orders(wk[m][1], act_b, origins, TAU)[o]
                                      for o in sc.TEST_ORIGINS], axis=1) if m != "lgbw_q80" else wk[m][1][:, test_idx],
                            ab, TAU).sum(axis=1) for m in sc.ALL_MODELS}
    den = ab.sum(axis=1)
    I_e, I_d = boot.ratio(s2_num[sc.INCUMBENT], den)
    W_e, W_d = boot.ratio(s2_num["lgbw_q80"], den)
    cap_rows = []
    for m in sc.DAILY_MODELS + ["lgbw_tweedie"]:
        D_e, D_d = boot.ratio(s2_num[m], den)
        cap = (I_e - D_e) / (I_e - W_e)
        lo, hi = ci((I_d - D_d) / (I_d - W_d))
        cap_rows.append({"model": m, "S2": D_e, "captured": cap, "lo": lo, "hi": hi})
        out.append(f"  {m:13s} S2 {D_e:.4f}; captured {cap:.3f} [{lo:.3f}, {hi:.3f}]")
    pd.DataFrame(cap_rows).to_csv(args.out / "captured_share.csv", index=False)
    out.append("")

    # (d) two numbers: the planner sees the Tweedie forecast, the order comes from the 0.8 quantile model
    two = metrics(wk["lgbw_tweedie"][0], wk["lgbw_tweedie"][1], order_fb=wk["lgbw_q80"][1], convert=False)
    out.append("(d) Two numbers: lgbw_tweedie shown and judged, lgbw_q80 ordered")
    out.append("  " + "; ".join(f"{k} {v:.4f}" for k, v in two.items()))
    out.append(f"  S4 / incumbent S4 = {two['S4'] / inc['S4']:.3f}")
    out.append("")

    # (e) blends with the incumbent
    out.append("(e) Blends w x lgbw_tweedie + (1 - w) x ma28 (week totals), each converted to its own 0.8 order")
    blend_rows = []
    for w in np.round(np.arange(0, 1.0001, 0.1), 1):
        fa = w * wk["lgbw_tweedie"][0] + (1 - w) * wk["ma28"][0]
        fb = w * wk["lgbw_tweedie"][1] + (1 - w) * wk["ma28"][1]
        mt = metrics(fa, fb)
        adm = mt["S3"] <= inc["S3"] and mt["S4"] <= inc["S4"]
        blend_rows.append({"w": w, **mt, "admissible": adm})
        out.append(f"  w {w:.1f}: " + "; ".join(f"{k} {v:.4f}" for k, v in mt.items()) + f"; admissible {adm}")
    bl = pd.DataFrame(blend_rows)
    bl.to_csv(args.out / "blends.csv", index=False)
    best = bl[bl["admissible"] & (bl["w"] > 0)]
    if len(best):
        b = best.loc[best["S2"].idxmin()]
        out.append(f"  best admissible blend: w {b['w']:.1f}, S2 {b['S2']:.4f} "
                   f"(captured share of the S2 gain over the incumbent {(inc['S2'] - b['S2']) / (inc['S2'] - base['lgbw_q80']['S2']):.3f})")
    out.append("")

    # (f) what the tolerance clause can ask for
    out.append("(f) Bad weeks per series-quarter, best S3 model (lgbw_tweedie) and the incumbent")
    for m in ["lgbw_tweedie", "ma28", "lgb_l1"]:
        bad = sc.bad_weeks(wk[m][1][:, test_idx], ab)
        per_q = np.stack([bad[:, q_of == q].sum(axis=1) for q in range(4)], axis=1).ravel()
        dist = [float((per_q == k).mean()) for k in range(4)] + [float((per_q >= 4).mean())]
        out.append(f"  {m:13s} 0: {dist[0]:.3f}, 1: {dist[1]:.3f}, 2: {dist[2]:.3f}, 3: {dist[3]:.3f}, 4+: {dist[4]:.3f}; "
                   f"mean {per_q.mean():.2f} bad weeks per quarter")
    vol = ab.mean(axis=1)
    bad = sc.bad_weeks(wk["lgbw_tweedie"][1][:, test_idx], ab)
    per_q = np.stack([bad[:, q_of == q].sum(axis=1) for q in range(4)], axis=1)
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]:
        sel = (vol >= lo) & (vol < hi)
        out.append(f"  mean weekly sales {lo}-{hi if hi < 1e9 else 'up'}: {sel.sum()} series, "
                   f"{(per_q[sel] > 1).mean():.3f} of series-quarters with more than one bad week, "
                   f"{bad[sel].mean():.3f} of weeks bad")
    ev_week = np.array([(p.event[o + 7: o + 14] > 0).any() for o in sc.TEST_ORIGINS])
    out.append(f"  weeks B with an event day: {ev_week.mean():.3f} of test weeks; "
               f"{bad[:, ev_week].sum() / bad.sum():.3f} of lgbw_tweedie's bad weeks; "
               f"bad-week rate {bad[:, ev_week].mean():.3f} in event weeks, {bad[:, ~ev_week].mean():.3f} otherwise")
    out.append("")

    # (h) the noise floor of the tolerance clause: bad weeks of a perfect mean forecast under Poisson sales
    rng = np.random.default_rng(5)
    mu = np.maximum(wk["lgbw_tweedie"][1][:, test_idx], 1e-6)
    draws = 20
    floor_bad = np.zeros_like(mu)
    for _ in range(draws):
        ys = rng.poisson(mu)
        floor_bad += sc.bad_weeks(mu, ys)
    floor_rate = floor_bad / draws
    floor_q = np.stack([floor_rate[:, q_of == q] for q in range(4)], axis=1)  # (n, 4, 13)
    # P(more than one bad week in a quarter), weeks independent: 1 - P(0) - P(1)
    p0 = np.prod(1 - floor_q, axis=2)
    p1 = sum(floor_q[:, :, k] * np.prod(np.delete(1 - floor_q, k, axis=2), axis=2) for k in range(13))
    out.append("(h) Poisson floor: sales drawn as Poisson around lgbw_tweedie's week B forecast, judged against it")
    out.append(f"  bad-week rate {floor_rate.mean():.3f} (model: {bad.mean():.3f}); "
               f"series-quarters with more than one bad week {(1 - p0 - p1).mean():.3f} (model: {(per_q > 1).mean():.3f})")
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]:
        sel = (vol >= lo) & (vol < hi)
        out.append(f"  mean weekly sales {lo}-{hi if hi < 1e9 else 'up'}: floor {floor_rate[sel].mean():.3f} of weeks bad, "
                   f"{(1 - p0 - p1)[sel].mean():.3f} of series-quarters with more than one")
    out.append("")

    # (g) features: what the weekly Tweedie model leans on (gain, summed over the five fits)
    imp = sum(f[f"imp__lgbw_tweedie_{r}"] for r in sc.REFITS)
    order = np.argsort(imp)[::-1]
    out.append("(g) lgbw_tweedie feature gain, share of total over the five fits")
    out.append("  " + ", ".join(f"{sc.WEEKLY_FEATURES[i]} {imp[i] / imp.sum():.2f}" for i in order[:8]))
    imp = sum(f[f"imp__lgb_l1_{r}"] for r in sc.REFITS)
    order = np.argsort(imp)[::-1]
    out.append("  lgb_l1: " + ", ".join(f"{sc.DAILY_FEATURES[i]} {imp[i] / imp.sum():.2f}" for i in order[:8]))
    out.append(f"  SES alpha chosen (share of series per value, test refits): "
               + ", ".join(f"{a}: {(f['ses_alpha'][1:] == a).mean():.2f}" for a in sc.SES_GRID))

    (args.out / "details.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
