"""Prints every number in the article and writes the data behind the figures.

Run from this folder:  python analysis.py > results/output.txt
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from mlp import checks, data, model
from mlp import features as F

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

LABELS = {
    "reviewed": "reviewed pipeline",
    "leak_dep_delay": "own departure delay as a feature",
    "unlagged_congestion": "congestion window not lagged to T",
    "global_encoding": "route and tail rates from all labels",
    "random_split": "shuffled split instead of a split in time",
    "drop_incomplete": "cancelled and diverted flights dropped",
    "utc_hour": "hour taken from the UTC timestamp",
}
CUTOFFS = [dt.datetime(2013, m, d, h, mi) for m, d, h, mi in [
    (10, 3, 8, 0), (10, 11, 15, 30), (10, 22, 11, 0), (10, 30, 17, 0), (11, 6, 9, 30),
    (11, 15, 13, 0), (11, 27, 16, 30), (12, 4, 7, 30), (12, 12, 12, 0), (12, 21, 18, 0)]]


def header(t: str) -> None:
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def one(con, sql: str):
    return con.execute(sql).fetchall()[0][0]


def section_data(con) -> None:
    header("1. Data, label and periods")
    n = one(con, "select count(*) from base")
    canc = one(con, "select count(*) from base where dep_delay is null")
    div = one(con, "select count(*) from base where dep_delay is not null and arr_delay is null")
    late = one(con, "select count(*) from base where arr_delay >= 15")
    print(f"scheduled flights {n:,}; cancelled {canc:,}; diverted or no arrival {div:,}; arrived 15+ min late {late:,}")
    print(f"disrupted (any of the three): {one(con, 'select sum(disrupted) from base'):,} "
          f"({pct(one(con, 'select avg(disrupted) from base'))})")
    rows = con.execute("select month, count(*), avg(disrupted) from base group by 1 order by 1").fetchall()
    print("disruption rate by month: " + ", ".join(f"{m}: {pct(r)}" for m, _, r in rows))
    for label, span in (("train", model.TRAIN), ("validation", model.VALID), ("test", model.TEST), ("deployment", model.DEPLOY)):
        k, r = con.execute(f"select count(*), avg(disrupted) from base where month between {span[0]} and {span[1]}").fetchall()[0]
        print(f"{label:10s} months {span[0]}-{span[1]}: {k:,} flights, disruption rate {pct(r)}")
    w = con.execute("select origin, time_hour, wind_speed from raw_weather where wind_speed > 100").fetchall()
    print(f"weather rows with wind speed above 100 mph: {len(w)} ({w[0][0]}, {w[0][1]:%Y-%m-%d %H:%M} UTC, {w[0][2]:.1f} mph); "
          f"next highest {one(con, 'select max(wind_speed) from raw_weather where wind_speed <= 100'):.1f} mph")
    print(f"features: {', '.join(F.CATEGORICAL + F.NUMERIC)}")


def section_results(con, frames) -> dict:
    header("2. Reported against realized performance, one defect at a time")
    variants = [F.Variant()] + [F.Variant(**{d: True}) for d in F.DEFECTS] + [F.SHIPPED]
    out = {}
    for v in variants:
        r = model.run(con, frames, v)
        out[v.name] = r
        label = LABELS.get(v.name, "shipped pipeline (all six defects)")
        print(f"{label:42s} reported AUC {r.reported.auc:.3f} (n={r.reported.n:,})  realized AUC {r.realized.auc:.3f}  "
              f"gap {r.reported.auc - r.realized.auc:+.3f}  realized mean prediction {r.realized.mean_pred:.3f} "
              f"vs rate {r.realized.rate:.3f}")
    base = out["reviewed"]
    print(f"\nreviewed: reported AP {base.reported.ap:.3f}, F1 {base.reported.f1:.3f} at threshold {base.threshold:.2f}; "
          f"realized AP {base.realized.ap:.3f}, F1 {base.realized.f1:.3f}, Brier {base.realized.brier:.4f}")

    # day-level bootstrap intervals for the reviewed model
    df = frames[F.Variant()]
    cols = F.Variant().feature_columns()
    m = model.fit(model.months(df, model.TRAIN), cols)
    for label, span in (("test", model.TEST), ("deployment", model.DEPLOY)):
        part = model.months(df, span)
        p = model.scores(m, part, cols)
        lo, hi = model.day_bootstrap_auc(part, p)
        print(f"reviewed {label} AUC {roc_auc_score(part['disrupted'], p):.3f}, 95% day-bootstrap interval {lo:.3f} to {hi:.3f}")

    # the leaking model at serving time
    v = F.Variant(leak_dep_delay=True)
    lm = model.fit(model.months(frames[v], model.TRAIN), v.feature_columns())
    dec = model.serving_frame(con, frames[F.Variant()], v)
    p = model.scores(lm, dec, v.feature_columns())
    train = model.months(frames[v], model.TRAIN)
    miss = train["dep_delay"].isna()
    print(f"\nleaking model: in training, dep_delay is missing for {miss.sum():,} flights, "
          f"all disrupted: {bool(train.loc[miss, 'disrupted'].eq(1).all())}")
    print(f"leaking model in December: predictions >= 0.99 for {pct(np.mean(p >= 0.99))} of flights; "
          f"median prediction {np.median(p):.3f}")
    imp_auc = roc_auc_score(model.months(frames[v], model.TEST)["disrupted"],
                            model.months(frames[v], model.TEST)["dep_delay"].fillna(9999))
    print(f"dep_delay alone on the test months (missing ranked highest): AUC {imp_auc:.3f}")
    np.save(RESULTS / "leak_december_predictions.npy", p)
    pr = model.scores(m, model.months(frames[F.Variant()], model.DEPLOY), cols)
    np.save(RESULTS / "reviewed_december_predictions.npy", pr)
    return out


def section_split(con, frames) -> dict:
    header("3. Where a shuffled split's optimism comes from")
    df = frames[F.Variant()]
    cols = F.Variant().feature_columns()
    pool = df[df["month"] <= model.TEST[1]]
    rng = np.random.default_rng(0)
    res = {}
    # rows shuffled
    u = rng.random(len(pool))
    m = model.fit(pool[u < 0.8], cols)
    res["random rows"] = roc_auc_score(pool[u >= 0.8]["disrupted"], model.scores(m, pool[u >= 0.8], cols))
    # whole days shuffled
    days = pool["flight_date"].unique()
    test_days = set(np.random.default_rng(1).choice(days, int(0.2 * len(days)), replace=False))
    is_test = pool["flight_date"].isin(test_days)
    m = model.fit(pool[~is_test], cols)
    res["random days"] = roc_auc_score(pool[is_test]["disrupted"], model.scores(m, pool[is_test], cols))
    # forward in time
    m = model.fit(model.months(df, model.TRAIN), cols)
    t = model.months(df, model.TEST)
    res["months ahead"] = roc_auc_score(t["disrupted"], model.scores(m, t, cols))
    for k, a in res.items():
        print(f"{k:13s} AUC {a:.3f}")
    # the same for a model without the two day-level features
    reduced = [c for c in cols if c not in ("cong_mean_delay", "cong_departures", "route_rate")]
    m = model.fit(pool[u < 0.8], reduced)
    a_rows = roc_auc_score(pool[u >= 0.8]["disrupted"], model.scores(m, pool[u >= 0.8], reduced))
    m = model.fit(model.months(df, model.TRAIN), reduced)
    a_fwd = roc_auc_score(t["disrupted"], model.scores(m, t, reduced))
    print(f"without congestion and route-rate features: random rows {a_rows:.3f}, months ahead {a_fwd:.3f}")
    same_day = pool[u >= 0.8]["flight_date"].isin(set(pool[u < 0.8]["flight_date"])).mean()
    print(f"share of shuffled-test flights whose day also appears in training: {pct(same_day)}")
    return res


def section_detection(con, frames, results) -> list:
    header("4. Which test layer catches which defect")
    ddl = checks._raw_ddl(con)
    rows = []
    null_aucs = {}
    for v in [F.Variant()] + [F.Variant(**{d: True}) for d in F.DEFECTS]:
        res = [checks.check_data_validation(con, frames, v)]
        res += checks.check_unit_tests(ddl, v)
        res.append(checks.check_outcome_property(ddl, v))
        res.append(checks.check_future_perturbation(con, frames, v, CUTOFFS))
        nl, aucs = checks.check_null_label(con, v)
        null_aucs[v.name] = aucs
        res.append(nl)
        res += checks.check_split_audit(con, frames, v)
        res.append(checks.check_parity(con, frames, v))
        res.append(checks.check_metric_gate(results[v.name]))
        rows.append((v.name, res))
        print(f"\n{LABELS[v.name]}")
        for r in res:
            mark = "pass" if r.passed else "FAIL"
            print(f"  {mark}  {r.layer:20s} {r.name}" + (f"  [{r.detail}]" if r.detail else ""))
    print("\nnull-label AUCs: " + "; ".join(f"{k}: {', '.join(f'{a:.3f}' for a in v)}" for k, v in null_aucs.items()))
    return rows, null_aucs


def section_behaviour(con, frames) -> dict:
    header("5. Behavioural, slice and reproducibility tests on the reviewed model")
    df = frames[F.Variant()]
    cols = F.Variant().feature_columns()
    train, dec = model.months(df, model.TRAIN), model.months(df, model.DEPLOY)
    m = model.fit(train, cols)
    p = model.scores(m, dec, cols)
    shuffled = dec.sample(frac=1.0, random_state=3)
    ps = pd.Series(model.scores(m, shuffled, cols), index=shuffled.index).reindex(dec.index).to_numpy()
    print(f"row order: max |difference| in predictions after shuffling rows {np.max(np.abs(ps - p)):.2e}")
    bumped = dec.copy()
    has = bumped["cong_mean_delay"].notna()
    bumped.loc[has, "cong_mean_delay"] += 30
    pb = model.scores(m, bumped, cols)
    viol = np.mean(pb[has.to_numpy()] < p[has.to_numpy()] - 1e-12)
    print(f"directional: +30 min of mean delay at the airport lowers the predicted risk for {pct(viol)} "
          f"of the {has.sum():,} December flights with recent departures")
    cst = [1 if c == "cong_mean_delay" else 0 for c in cols]
    mc = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=50,
        categorical_features="from_dtype", early_stopping=False, random_state=0, monotonic_cst=cst)
    mc.fit(train[cols], train["disrupted"])
    pc, pcb = mc.predict_proba(dec[cols])[:, 1], mc.predict_proba(bumped[cols])[:, 1]
    print(f"with a monotonic constraint on that feature: violations {pct(np.mean(pcb[has.to_numpy()] < pc[has.to_numpy()] - 1e-12))}; "
          f"December AUC {roc_auc_score(dec['disrupted'], pc):.3f} against {roc_auc_score(dec['disrupted'], p):.3f} unconstrained")

    # slices
    slices = []
    for col in ("origin", "carrier"):
        for val, part in dec.groupby(col, observed=True):
            if len(part) < 1000 or part["disrupted"].nunique() < 2:
                continue
            pp = model.scores(m, part, cols)
            lo, hi = model.day_bootstrap_auc(part, pp, reps=200)
            slices.append((col, str(val), len(part), roc_auc_score(part["disrupted"], pp), lo, hi))
    overall = roc_auc_score(dec["disrupted"], p)
    print(f"December AUC overall {overall:.3f}; slices with at least 1,000 flights:")
    for col, val, n, a, lo, hi in sorted(slices, key=lambda s: s[3]):
        print(f"  {col:7s} {val:4s} n={n:6,} AUC {a:.3f} ({lo:.3f} to {hi:.3f})")

    # reproducibility
    m2 = model.fit(train, cols)
    with threadpool_limits(limits=1):
        m1 = model.fit(train, cols)
        p1 = model.scores(m1, dec, cols)
    print(f"refit with the same seed: max |difference| {np.max(np.abs(model.scores(m2, dec, cols) - p)):.2e}; "
          f"refit on one thread: {np.max(np.abs(p1 - p)):.2e}")
    m3 = model.fit(train, cols, seed=1)
    d3 = np.abs(model.scores(m3, dec, cols) - p)
    print(f"refit with another seed: max |difference| {d3.max():.3f}, median {np.median(d3):.4f}; "
          f"the training set has {len(train):,} rows and the binning step subsamples 200,000 of them")
    seed_aucs = []
    for seed in range(5):
        ms = model.fit(train, cols, seed=seed)
        seed_aucs.append(roc_auc_score(dec["disrupted"], model.scores(ms, dec, cols)))
    print("December AUC over seeds 0-4: " + ", ".join(f"{a:.4f}" for a in seed_aucs)
          + f" (range {max(seed_aucs) - min(seed_aucs):.4f})")
    return {"slices": slices, "overall": overall}


def section_thresholds(con, frames) -> None:
    header("6. Thresholds under a changing disruption rate")
    df = frames[F.Variant()]
    cols = F.Variant().feature_columns()
    m = model.fit(model.months(df, model.TRAIN), cols)
    parts = {k: model.months(df, s) for k, s in (("validation", model.VALID), ("test", model.TEST), ("deployment", model.DEPLOY))}
    preds = {k: model.scores(m, v, cols) for k, v in parts.items()}
    best = {k: model.best_threshold(parts[k]["disrupted"].to_numpy(), preds[k]) for k in parts}
    for src in parts:
        cells = []
        for tgt in parts:
            f1 = model.evaluate(parts[tgt]["disrupted"].to_numpy(), preds[tgt], best[src]).f1
            cells.append(f"{tgt} F1 {f1:.3f}")
        print(f"threshold tuned on {src:10s} = {best[src]:.2f}: " + ", ".join(cells))


def main() -> None:
    print(f"duckdb {duckdb.__version__}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}")
    con = data.connect()
    frames = F.FeatureCache(con)
    section_data(con)
    results = section_results(con, frames)
    split_res = section_split(con, frames)
    rows, null_aucs = section_detection(con, frames, results)
    beh = section_behaviour(con, frames)
    section_thresholds(con, frames)
    fig = {
        "results": {k: {"reported": r.reported.auc, "realized": r.realized.auc} for k, r in results.items()},
        "detection": [(name, [(c.layer, c.name, c.passed) for c in res]) for name, res in rows],
        "null_aucs": null_aucs,
        "split": split_res,
        "slices": beh["slices"],
        "overall": beh["overall"],
    }
    (RESULTS / "figure_data.json").write_text(json.dumps(fig, indent=1, default=str))


if __name__ == "__main__":
    main()
