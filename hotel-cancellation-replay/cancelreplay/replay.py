"""Replay a cancellation system night by night on the real outcomes.

A system is the whole chain: which bookings it learns from, which columns it
reads, which model it fits, how often it refits, when the nightly batch runs
relative to arrival, what the model hands to the decision, and how the
decision aggregates bookings. ``SystemSpec`` names one choice of each.
Predictions depend on the first five; the interface and the aggregation are
applied afterwards to the same predictions, so they are compared on identical
model output.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import decision as D
from . import features as F
from . import models as M
from . import snapshots as S

LIVE_START = pd.Timestamp("2016-07-01")
LIVE_END = pd.Timestamp("2017-08-31")
HOTELS = ("Resort Hotel", "City Hotel")
BLOCK_KEY = ["hotel", "arrival", "booked", "agent", "company", "market_segment"]

FEATURE_SETS = {
    "point_in_time": F.POINT_IN_TIME,
    "plus_deposit": F.POINT_IN_TIME_PLUS_DEPOSIT,
    "as_recorded": F.AS_RECORDED,
}


@dataclass(frozen=True)
class SystemSpec:
    model: str = "boosting"          # base_rate, logistic, boosting, boosting_tuned, boosting_balanced
    features: str = "point_in_time"  # point_in_time, plus_deposit, as_recorded
    training: str = "point_in_time"  # point_in_time, all_bookings, known_status
    cadence: str = "monthly"         # once, monthly, weekly
    horizon: int = 30                # days between the nightly batch and arrival
    serve: str = "recorded"          # recorded, arrival_defaults

    def but(self, **changes) -> "SystemSpec":
        return replace(self, **changes)


def retrain_times(cadence: str, horizon: int) -> list[pd.Timestamp]:
    """Refit dates, anchored at the first batch of the live period.

    The live period is the same set of arrival nights for every horizon, so
    the first batch runs h days before the first live arrival.
    """
    first = LIVE_START - pd.Timedelta(days=horizon)
    last = LIVE_END - pd.Timedelta(days=horizon)
    if cadence == "once":
        return [first]
    if cadence == "monthly":
        return [first + pd.DateOffset(months=k) for k in range(15) if first + pd.DateOffset(months=k) <= last]
    if cadence == "weekly":
        return list(pd.date_range(first, last, freq="7D"))
    raise ValueError(cadence)


def make_model(name: str, features, tuned_params: dict | None):
    if name == "base_rate":
        return M.BaseRate(features)
    if name == "logistic":
        return M.Logistic(features)
    if name == "boosting":
        return M.Boosting(features)
    if name == "boosting_tuned":
        return M.Boosting(features, **(tuned_params or {}))
    if name == "boosting_balanced":
        return M.Boosting(features, class_weight="balanced")
    raise ValueError(name)


def serve_view(rows: pd.DataFrame, how: str) -> pd.DataFrame:
    """What the hotel system would show for a booking that has not arrived yet.

    ``recorded`` uses the values in the published file. ``arrival_defaults``
    replaces the three columns whose values are set at or after check-in with
    what an unarrived booking carries: the default country, the room that was
    reserved, and no parking space. Amendments and special requests are left
    as recorded, which favours the model that reads them.
    """
    if how == "recorded":
        return rows
    if how == "arrival_defaults":
        out = rows.copy()
        out["country"] = "PRT"
        out["assigned_room_type"] = out["reserved_room_type"]
        out["required_car_parking_spaces"] = 0
        return out
    raise ValueError(how)


class Replayer:
    """Holds the data and caches snapshots, tuned parameters and predictions."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["block"] = self.df.groupby(BLOCK_KEY).ngroup()
        self._snapshots: dict = {}
        self._tuned: dict = {}
        self._predictions: dict = {}

    def snapshots(self, hotel: str, horizon: int) -> pd.DataFrame:
        key = (hotel, horizon)
        if key not in self._snapshots:
            dh = self.df.loc[self.df["hotel"] == hotel]
            self._snapshots[key] = S.snapshot_rows(dh, horizon)
        return self._snapshots[key]

    def tuned(self, hotel: str, spec: SystemSpec) -> dict:
        key = (hotel, spec.features, spec.training, spec.horizon)
        if key not in self._tuned:
            dh = self.df.loc[self.df["hotel"] == hotel]
            first = retrain_times("once", spec.horizon)[0]
            train = S.TRAINING_SETS[spec.training](dh, spec.horizon, first)
            self._tuned[key] = M.tune_boosting(
                FEATURE_SETS[spec.features], train, train["label"].to_numpy()
            )
        return self._tuned[key]

    def predict(self, spec: SystemSpec) -> pd.DataFrame:
        """Serving rows for every live arrival, each scored by the model in force."""
        if spec in self._predictions:
            return self._predictions[spec]
        features = FEATURE_SETS[spec.features]
        if spec.serve == "recorded" and spec.features == "point_in_time":
            F.assert_point_in_time(features)
        out = []
        for hotel in HOTELS:
            dh = self.df.loc[self.df["hotel"] == hotel]
            serving = self.snapshots(hotel, spec.horizon)
            serving = serving.loc[(serving["arrival"] >= LIVE_START) & (serving["arrival"] <= LIVE_END)]
            params = self.tuned(hotel, spec) if spec.model == "boosting_tuned" else None
            times = retrain_times(spec.cadence, spec.horizon)
            for i, T in enumerate(times):
                end = times[i + 1] if i + 1 < len(times) else LIVE_END + pd.Timedelta(days=1)
                # The model refit at T, from arrivals before T, serves the batches
                # run in [T, end). A batch at decision time t concerns arrival t + h.
                batch = serving.loc[
                    (serving["decision_time"] >= T) & (serving["decision_time"] < end)
                ]
                if batch.empty:
                    continue
                train = S.TRAINING_SETS[spec.training](dh, spec.horizon, T)
                model = make_model(spec.model, features, params)
                model.fit(train, train["label"].to_numpy())
                p = model.predict_proba(serve_view(batch, spec.serve))
                out.append(batch.assign(p=np.clip(p, 1e-6, 1 - 1e-6)))
        result = pd.concat(out).sort_values(["hotel", "arrival"], kind="stable")
        self._predictions[spec] = result
        return result


INTERFACES = ("distribution_blocks", "distribution_independent", "mean", "labels")


def nightly_log(pred: pd.DataFrame, walk_multiple: float, interfaces=INTERFACES) -> pd.DataFrame:
    """One row per hotel and arrival night: what was visible, decided and realised."""
    ratio = 1.0 / walk_multiple
    records = []
    for (hotel, arrival), g in pred.groupby(["hotel", "arrival"], sort=True):
        p = g["p"].to_numpy()
        released = int(g["label"].sum())
        price = float(np.median(np.clip(g["adr"].to_numpy(), 0, None)))
        rec = dict(hotel=hotel, arrival=arrival, visible=len(g), released=released,
                   expected=float(p.sum()), price=price)
        for name in interfaces:
            if name == "distribution_blocks":
                blocks = g.groupby("block")["p"].agg(["mean", "size"])
                pmf = D.block_pmf(blocks["mean"].to_numpy(), blocks["size"].to_numpy())
                q = D.optimal_release(pmf, ratio)
            elif name == "distribution_independent":
                q = D.optimal_release(D.poisson_binomial_pmf(p), ratio)
            elif name == "mean":
                q = D.release_from_mean(p)
            elif name == "labels":
                q = D.release_from_labels(p)
            else:
                raise ValueError(name)
            rec[f"q_{name}"] = q
        records.append(rec)
    log = pd.DataFrame(records)
    for name in interfaces:
        q = log[f"q_{name}"]
        log[f"profit_{name}"] = D.profit(q, log["released"], log["price"], walk_multiple)
        log[f"walked_{name}"] = np.maximum(q - log["released"], 0)
    log["oracle"] = log["price"] * log["released"]
    return log
