"""Every comparison reported in the article.

The reference system is a careful build: point-in-time snapshots, columns known
at booking, gradient boosting refit monthly, a nightly batch 30 days before
arrival, probabilities handed to a decision that treats blocks of bookings as
one unit, and a walk cost of three nights' rate. Each variant changes one thing
and is replayed on the same 854 arrival nights (427 per hotel).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import features as F
from . import models as M
from . import snapshots as S
from .replay import HOTELS, LIVE_END, LIVE_START, Replayer, SystemSpec, nightly_log

WALK_MULTIPLE = 3.0
REFERENCE = SystemSpec()
REFERENCE_INTERFACE = "distribution_blocks"
BOOTSTRAP_DRAWS = 2000


@dataclass(frozen=True)
class Variant:
    key: str
    group: str        # model, engineering, architecture
    label: str
    spec: SystemSpec
    interface: str = REFERENCE_INTERFACE


VARIANTS = [
    Variant("reference", "reference", "Reference system", REFERENCE),
    # The model and its inputs.
    Variant("base_rate", "model", "No model: historical rate for every booking", REFERENCE.but(model="base_rate")),
    Variant("logistic", "model", "Logistic regression", REFERENCE.but(model="logistic")),
    Variant("boosting_tuned", "model", "Boosting, tuned on a time-ordered split", REFERENCE.but(model="boosting_tuned")),
    Variant("plus_deposit", "model", "Boosting, adding the deposit column", REFERENCE.but(features="plus_deposit")),
    # How the data reach the model.
    Variant("all_bookings", "engineering", "Trained on all bookings, label 'ever cancelled'", REFERENCE.but(training="all_bookings")),
    Variant("known_status", "engineering", "Trained on 'everything with a known status'", REFERENCE.but(training="known_status")),
    Variant("leak_recorded", "engineering", "Late columns, served as recorded (impossible)", REFERENCE.but(features="as_recorded")),
    Variant("leak_defaults", "engineering", "Late columns, served as an unarrived booking shows them", REFERENCE.but(features="as_recorded", serve="arrival_defaults")),
    Variant("cadence_once", "engineering", "Fitted once, never refit", REFERENCE.but(cadence="once")),
    Variant("cadence_weekly", "engineering", "Refit weekly", REFERENCE.but(cadence="weekly")),
    # Where the pieces meet.
    Variant("independent", "architecture", "Decision treats bookings as independent", REFERENCE, "distribution_independent"),
    Variant("mean_only", "architecture", "Decision receives only the expected count", REFERENCE, "mean"),
    Variant("labels", "architecture", "Decision receives yes/no labels", REFERENCE, "labels"),
    Variant("balanced", "architecture", "Model swapped for a class-weighted one", REFERENCE.but(model="boosting_balanced")),
    Variant("horizon_7", "architecture", "Batch runs 7 days before arrival", REFERENCE.but(horizon=7)),
    Variant("horizon_14", "architecture", "Batch runs 14 days before arrival", REFERENCE.but(horizon=14)),
    Variant("horizon_60", "architecture", "Batch runs 60 days before arrival", REFERENCE.but(horizon=60)),
]


def summarise_log(log: pd.DataFrame, interface: str) -> dict:
    nights = len(log)
    profit = log[f"profit_{interface}"].sum()
    oracle = log["oracle"].sum()
    return dict(
        nights=nights,
        share_of_oracle=profit / oracle,
        profit_per_night=profit / nights,
        oracle_per_night=oracle / nights,
        resold_per_night=log[f"q_{interface}"].mean(),
        walked_per_night=log[f"walked_{interface}"].mean(),
        released_per_night=log["released"].mean(),
        nights_with_walks=float((log[f"walked_{interface}"] > 0).mean()),
    )


def prediction_metrics(pred: pd.DataFrame) -> dict:
    y, p = pred["label"].to_numpy(), pred["p"].to_numpy()
    return dict(
        auc=roc_auc_score(y, p),
        log_loss=log_loss(y, p, labels=[0, 1]),
        brier=float(np.mean((p - y) ** 2)),
        predicted_over_realised=float(p.sum() / y.sum()),
        rows=len(y),
        base_rate=float(y.mean()),
    )


def week_index(log: pd.DataFrame) -> np.ndarray:
    return ((log["arrival"] - LIVE_START).dt.days // 7).to_numpy()


def bootstrap_difference(ref: pd.DataFrame, ref_if: str, var: pd.DataFrame, var_if: str,
                         seed: int = 20261026) -> tuple[float, float, float]:
    """Paired weekly-block bootstrap of the difference in share of oracle."""
    cols_a = [f"profit_{ref_if}", "oracle"]
    cols_b = [f"profit_{var_if}", "oracle"]
    a = ref.set_index(["hotel", "arrival"])[cols_a]
    b = var.set_index(["hotel", "arrival"])[cols_b]
    # A night with no visible booking has no row; it contributes nothing.
    index = a.index.union(b.index).sort_values()
    a = a.reindex(index).fillna(0.0)
    b = b.reindex(index).fillna(0.0)
    weeks = week_index(index.to_frame(index=False))
    unique = np.unique(weeks)
    by_week = {}
    for col, frame, key in (("ref", a, f"profit_{ref_if}"), ("var", b, f"profit_{var_if}"),
                            ("ref_oracle", a, "oracle"), ("var_oracle", b, "oracle")):
        by_week[col] = pd.Series(frame[key].to_numpy()).groupby(weeks).sum().reindex(unique).to_numpy()
    point = by_week["var"].sum() / by_week["var_oracle"].sum() - by_week["ref"].sum() / by_week["ref_oracle"].sum()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(BOOTSTRAP_DRAWS, len(unique)))
    diff = (by_week["var"][draws].sum(1) / by_week["var_oracle"][draws].sum(1)
            - by_week["ref"][draws].sum(1) / by_week["ref_oracle"][draws].sum(1))
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def run_variants(rp: Replayer, walk_multiple: float = WALK_MULTIPLE) -> tuple[pd.DataFrame, dict]:
    logs = {}
    rows = []
    for v in VARIANTS:
        pred = rp.predict(v.spec)
        log = nightly_log(pred, walk_multiple)
        logs[v.key] = log
        row = dict(key=v.key, group=v.group, label=v.label, interface=v.interface)
        row.update(summarise_log(log, v.interface))
        row.update(prediction_metrics(pred))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("key")
    ref_log = logs["reference"]
    cis = {}
    for v in VARIANTS:
        if v.key == "reference":
            cis[v.key] = (0.0, 0.0, 0.0)
            continue
        cis[v.key] = bootstrap_difference(ref_log, REFERENCE_INTERFACE, logs[v.key], v.interface)
    table["diff_vs_reference"] = [cis[k][0] for k in table.index]
    table["diff_lo"] = [cis[k][1] for k in table.index]
    table["diff_hi"] = [cis[k][2] for k in table.index]
    return table, logs


def by_hotel(logs: dict) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        for hotel in HOTELS:
            log = logs[v.key]
            row = dict(key=v.key, hotel=hotel)
            row.update(summarise_log(log.loc[log["hotel"] == hotel], v.interface))
            rows.append(row)
    return pd.DataFrame(rows)


def walk_cost_sensitivity(rp: Replayer, keys=("reference", "base_rate", "logistic", "boosting_tuned",
                                              "all_bookings", "leak_defaults", "independent",
                                              "mean_only", "labels", "balanced")) -> pd.DataFrame:
    rows = []
    for multiple in (2.0, 3.0, 5.0):
        for v in VARIANTS:
            if v.key not in keys:
                continue
            log = nightly_log(rp.predict(v.spec), multiple, interfaces=(v.interface,))
            rows.append(dict(walk_multiple=multiple, key=v.key, **summarise_log(log, v.interface)))
    return pd.DataFrame(rows)


# --- Descriptive analyses -----------------------------------------------------


def cancellation_timing(df: pd.DataFrame) -> dict:
    cancelled = df.loc[df["reservation_status"] == "Canceled"]
    days_before = (cancelled["arrival"] - cancelled["status_date"]).dt.days
    out = dict(
        bookings=len(df),
        cancelled_or_no_show=int(df["is_canceled"].sum()),
        no_shows=int((df["reservation_status"] == "No-Show").sum()),
        ever_cancel_rate=float(df["is_canceled"].mean()),
        median_days_before_arrival=float(days_before.median()),
        quartiles_days_before=[float(days_before.quantile(0.25)), float(days_before.quantile(0.75))],
    )
    out["by_horizon"] = {}
    for h in (1, 7, 14, 30, 60, 90):
        rows = S.snapshot_rows(df, h)
        releases = int(df["is_canceled"].sum())
        out["by_horizon"][h] = dict(
            visible_share_of_bookings=len(rows) / len(df),
            late_release_rate=float(rows["label"].mean()),
            share_of_all_releases_still_ahead=float(rows["label"].sum() / releases),
            visible_per_night=float(rows.groupby(["hotel", "arrival"]).size().mean()),
        )
    return out


def leak_smoke_test(df: pd.DataFrame) -> dict:
    """Values of a column that predict the label without a single exception."""
    out = {}
    for f in F.REGISTRY:
        col = df[f.name].astype(str)
        stats = df.groupby(col)["is_canceled"].agg(["mean", "size"])
        pure = stats.loc[(stats["size"] >= 500) & ((stats["mean"] == 0) | (stats["mean"] == 1))]
        if len(pure):
            out[f.name] = {str(k): dict(rows=int(r["size"]), cancel_rate=float(r["mean"])) for k, r in pure.iterrows()}
    prt = df.assign(prt=df["country"] == "PRT").groupby(["hotel", "prt"])["is_canceled"].mean()
    out["country_PRT_cancel_rate"] = {f"{h} | PRT={p}": float(v) for (h, p), v in prt.items()}
    deposit = df.groupby("deposit_type")["is_canceled"].agg(["mean", "size"])
    out["deposit_type"] = {k: dict(rows=int(r["size"]), cancel_rate=float(r["mean"])) for k, r in deposit.iterrows()}
    return out


def random_split_auc(df: pd.DataFrame, rp: Replayer) -> dict:
    """The number usually reported, against the number that matters.

    Five-fold stratified random cross-validation of boosting, per hotel, for
    three questions: 'ever cancel' with every column, 'ever cancel' with
    booking-time columns, and 'release after the 30-day batch' with
    booking-time columns. The last line is the out-of-time AUC of the
    reference system in the live period.
    """
    out = {}
    cases = {
        "ever_cancel_all_columns": (F.AS_RECORDED, None),
        "ever_cancel_booking_columns": (F.POINT_IN_TIME, None),
        "after_30_day_batch_booking_columns": (F.POINT_IN_TIME, 30),
    }
    for name, (feats, horizon) in cases.items():
        per_hotel = {}
        for hotel in HOTELS:
            dh = df.loc[df["hotel"] == hotel]
            rows = S.snapshot_rows(dh, horizon) if horizon else dh.assign(label=dh["is_canceled"])
            y = rows["label"].to_numpy()
            pred = np.zeros(len(rows))
            folds = StratifiedKFold(5, shuffle=True, random_state=0)
            for tr, te in folds.split(rows, y):
                m = M.Boosting(feats).fit(rows.iloc[tr], y[tr])
                pred[te] = m.predict_proba(rows.iloc[te])
            per_hotel[hotel] = float(roc_auc_score(y, pred))
        out[name] = per_hotel
    ref = rp.predict(REFERENCE)
    out["reference_out_of_time"] = {
        hotel: float(roc_auc_score(ref.loc[ref["hotel"] == hotel, "label"], ref.loc[ref["hotel"] == hotel, "p"]))
        for hotel in HOTELS
    }
    return out


def block_structure(rp: Replayer, horizon: int = 30) -> dict:
    df = rp.df
    sizes = df.groupby("block")["is_canceled"].transform("size")
    multi = df.loc[sizes >= 2]
    stats = multi.groupby("block")["is_canceled"].agg(["mean", "size"])
    cancelled_blocks = df.loc[(sizes >= 2) & (df["reservation_status"] == "Canceled")]
    one_date = cancelled_blocks.groupby("block")["release_date"].nunique().eq(1).mean()
    out = dict(
        bookings_in_blocks_of_two_or_more=float((sizes >= 2).mean()),
        blocks_of_two_or_more=int(len(stats)),
        largest_block=int(stats["size"].max()),
        blocks_all_same_outcome=float(((stats["mean"] == 0) | (stats["mean"] == 1)).mean()),
        cancelled_blocks_on_one_date=float(one_date),
    )
    rows = pd.concat([rp.snapshots(h, horizon) for h in HOTELS])
    rows = rows.assign(bsize=rows.groupby("block")["label"].transform("size"))
    late = rows["label"].sum()
    out["late_releases_from_blocks_2plus"] = float(rows.loc[rows["bsize"] >= 2, "label"].sum() / late)
    out["late_releases_from_blocks_10plus"] = float(rows.loc[rows["bsize"] >= 10, "label"].sum() / late)

    # How far the independent model understates the spread of nightly releases.
    pred = rp.predict(REFERENCE)
    g = pred.groupby(["hotel", "arrival"])
    nightly = pd.DataFrame(dict(N=g["label"].sum(), mu=g["p"].sum(),
                                v=g["p"].apply(lambda p: float((p * (1 - p)).sum()))))
    resid = nightly["N"] - nightly["mu"]
    out["variance_ratio_observed_to_independent"] = float(np.mean(resid ** 2) / nightly["v"].mean())
    # Discrimination narrows the independent spread; by how much, per night.
    n = g.size()
    pbar = nightly["mu"] / n
    out["sd_ratio_model_vs_constant_rate"] = float(
        np.mean(np.sqrt(nightly["v"]) / np.sqrt(n * pbar * (1 - pbar)))
    )
    return out
