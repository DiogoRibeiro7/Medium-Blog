"""Run the nine candidate forecasters at every warm-up and test origin.

Writes `<out>/forecasts.npz`: for each model, week A and week B totals at every origin
(series x origin), and the daily forecasts of the daily models at the 13 S0 origins
(series x origin x horizon). LightGBM models are refitted per block and cached in
`<out>/models/`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

import success_core as sc


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def block_origins(refit: int) -> list[int]:
    nxt = [r for r in sc.REFITS if r > refit]
    end = nxt[0] if nxt else 10**9
    return [o for o in sc.ALL_ORIGINS if refit <= o < end]


def train(X: np.ndarray, y: np.ndarray, features: list[str], extra: dict) -> lgb.Booster:
    cats = [f for f in sc.CATEGORICAL if f in features]
    ds = lgb.Dataset(X, label=y, feature_name=features, categorical_feature=cats, free_raw_data=True)
    params = {**sc.LGB_PARAMS, **extra}
    return lgb.train(params, ds, num_boost_round=sc.LGB_ROUNDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--category", default="FOODS")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "models").mkdir(exist_ok=True)

    log("loading panel")
    p = sc.load_panel(args.raw, args.category)
    log(f"{p.n} series")
    cs = sc.cumsum0(p.sales)
    origins = sc.ALL_ORIGINS
    pos = {o: i for i, o in enumerate(origins)}
    s0pos = {o: i for i, o in enumerate(sc.S0_ORIGINS)}
    n, no = p.n, len(origins)
    wk = {m: np.zeros((2, n, no), dtype=np.float32) for m in sc.ALL_MODELS}
    s0 = {m: np.zeros((n, len(sc.S0_ORIGINS), 28), dtype=np.float32) for m in sc.DAILY_MODELS}

    def store_daily(m: str, o: int, f: np.ndarray) -> None:
        f = np.maximum(f, 0)
        wk[m][0, :, pos[o]] = f[:, 0:7].sum(axis=1)
        wk[m][1, :, pos[o]] = f[:, 7:14].sum(axis=1)
        if o in s0pos:
            s0[m][:, s0pos[o], :] = f[:, :28]

    log("simple models")
    for o in origins:
        store_daily("snaive", o, sc.snaive(p.sales, o))
        store_daily("ma28", o, sc.ma28(p.sales, o))
        store_daily("dow4", o, sc.dow4(p.sales, o))

    log("ses")
    levels, sse = sc.ses_run(p.sales, origins, sc.REFITS)
    alpha_idx = {}
    for r in sc.REFITS:
        alpha_idx[r] = sc.ses_choose(sse[r])
        for o in block_origins(r):
            lev = levels[o][alpha_idx[r], np.arange(n)]
            store_daily("ses", o, np.repeat(lev[:, None], 28, axis=1))
    chosen = np.stack([np.asarray(sc.SES_GRID)[alpha_idx[r]] for r in sc.REFITS])

    rng = np.random.default_rng(7)
    importance = {}
    for r in sc.REFITS:
        log(f"refit {r}: daily training set")
        X, y = sc.daily_training_set(p, cs, r, 0.10, rng)
        log(f"  {len(y)} rows")
        boosters = {}
        for m in ["lgb_l2", "lgb_l1", "lgb_tweedie"]:
            path = args.out / "models" / f"{m}_{r}.txt"
            if path.exists():
                boosters[m] = lgb.Booster(model_file=str(path))
            else:
                boosters[m] = train(X, y, sc.DAILY_FEATURES, sc.OBJECTIVES[m])
                boosters[m].save_model(str(path))
            importance[f"{m}_{r}"] = boosters[m].feature_importance("gain")
            log(f"  trained {m}")
        del X, y
        log(f"refit {r}: weekly training set")
        Xw, yw = sc.weekly_training_set(p, cs, r, 0.50, rng)
        log(f"  {len(yw)} rows")
        for m in sc.WEEKLY_MODELS:
            path = args.out / "models" / f"{m}_{r}.txt"
            if path.exists():
                boosters[m] = lgb.Booster(model_file=str(path))
            else:
                boosters[m] = train(Xw, yw, sc.WEEKLY_FEATURES, sc.OBJECTIVES[m])
                boosters[m].save_model(str(path))
            importance[f"{m}_{r}"] = boosters[m].feature_importance("gain")
            log(f"  trained {m}")
        del Xw, yw
        for o in block_origins(r):
            hz = 28 if o in s0pos else 14
            X, _ = sc.daily_features(p, cs, o, horizons=hz)
            for m in ["lgb_l2", "lgb_l1", "lgb_tweedie"]:
                f = boosters[m].predict(X, num_threads=2).reshape(n, hz)
                if hz < 28:
                    f = np.concatenate([f, np.zeros((n, 28 - hz), dtype=f.dtype)], axis=1)
                store_daily(m, o, f)
            Xw = sc.weekly_features(p, cs, o)
            for m in sc.WEEKLY_MODELS:
                f = np.maximum(boosters[m].predict(Xw, num_threads=2).reshape(n, 2), 0)
                wk[m][0, :, pos[o]] = f[:, 0]
                wk[m][1, :, pos[o]] = f[:, 1]
        log(f"refit {r}: predicted {len(block_origins(r))} origins")

    np.savez_compressed(
        args.out / "forecasts.npz",
        origins=np.array(origins),
        s0_origins=np.array(sc.S0_ORIGINS),
        ids=p.ids, item=p.item, store=p.store, dept=p.dept,
        ses_alpha=chosen,
        **{f"wk__{m}": wk[m] for m in sc.ALL_MODELS},
        **{f"s0__{m}": s0[m] for m in sc.DAILY_MODELS},
        **{f"imp__{k}": v for k, v in importance.items()},
    )
    log("done")


if __name__ == "__main__":
    main()
