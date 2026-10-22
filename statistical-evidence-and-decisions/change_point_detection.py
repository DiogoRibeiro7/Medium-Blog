"""Prequential CUSUM detection and evidence-triggered refitting.

Bootstrap thresholds are descriptive, not formal sequential error guarantees.
Labels from j become usable at t >= j + delay + 1. Synthetic conditional means
are used only for final decision-regret evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np
import pandas as pd
from gradual_drift import generate_delayed_stream, available_label_count
from uncertainty_analysis import fixed_candidates

@dataclass(frozen=True)
class DetectorSettings:
    delay: int = 0
    scheduled_every: int = 15
    cooldown: int = 15
    train_window: int = 100
    bootstrap_draws: int = 50
    allowance: float = 0.2

    def __post_init__(self) -> None:
        for name in ("delay", "scheduled_every", "cooldown", "train_window", "bootstrap_draws"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"{name} must be an integer")
        if self.delay < 0 or self.scheduled_every < 1 or self.cooldown < 0:
            raise ValueError("Invalid delay, cadence or cooldown")
        if self.train_window < 64 or self.bootstrap_draws < 10:
            raise ValueError("Insufficient training window or calibration draws")
        if not np.isfinite(self.allowance) or self.allowance < 0:
            raise ValueError("Invalid CUSUM allowance")

def residual_disagreement(y: np.ndarray, yhat: np.ndarray, scale: float) -> np.ndarray:
    """Positive score favours a sign reversal in the predictor relationship."""
    actual = np.asarray(y, dtype=float).reshape(-1)
    predicted = np.asarray(yhat, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Invalid residual score inputs")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be strictly positive")
    return -((actual - predicted) * predicted) / scale

def bootstrap_threshold(scores: np.ndarray, config: DetectorSettings, seed: int) -> tuple[float,float,float]:
    """Estimate reference location, scale and CUSUM threshold with circular blocks."""
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or len(x) < 20 or not np.isfinite(x).all():
        raise ValueError("Need at least 20 historical scores")
    center = float(np.mean(x))
    scale = max(float(np.std(x, ddof=1)), 1e-8)
    standard = (x-center)/scale
    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(config.bootstrap_draws):
        simulated = []
        while len(simulated) < len(standard):
            start = int(rng.integers(len(standard)))
            simulated.extend(standard[np.arange(start, start+8) % len(standard)])
        stat = maximum = 0.0
        for value in simulated[:len(standard)]:
            stat = max(0.0, stat + float(value) - config.allowance)
            maximum = max(maximum, stat)
        maxima.append(maximum)
    return center, scale, max(float(np.quantile(maxima, 0.95)), 1e-6)

def _fit(x: np.ndarray, y: np.ndarray):
    model = fixed_candidates()["OLS"]
    model.fit(x, y)
    return model

def evaluate_detector(stream, config: DetectorSettings, seed: int = 17) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Process eligible labels once before making each step's predictions."""
    reference = _fit(stream.x_fit, stream.y_fit)
    training_pred = np.asarray(reference.predict(stream.x_fit)).reshape(-1)
    noise_scale = max(float(np.std(stream.y_fit - training_pred, ddof=1)), 1e-6)
    calibration_pred = np.asarray(reference.predict(stream.x_cal)).reshape(-1)
    center, score_scale, threshold = bootstrap_threshold(
        residual_disagreement(stream.y_cal, calibration_pred, noise_scale),
        config, seed
    )
    models = {"frozen": reference, "scheduled": _fit(stream.x_fit, stream.y_fit),
              "detected": _fit(stream.x_fit, stream.y_fit)}
    reference_forecasts = []
    rows, utilities = [], []
    stat = 0.0
    processed = 0
    cooldown_to = -1
    refits = {"frozen": 0, "scheduled": 0, "detected": 0}
    for t in range(len(stream.y_live)):
        available = available_label_count(t, config.delay)
        alarm = False
        source_index = -1
        for j in range(processed, available):
            score = float(residual_disagreement(
                np.array([stream.y_live[j]]), np.array([reference_forecasts[j]]),
                noise_scale
            )[0])
            stat = max(0.0, stat + (score-center)/score_scale-config.allowance)
            if stat > threshold and t >= cooldown_to:
                alarm = True
                source_index = j
                stat = 0.0
                cooldown_to = t + config.cooldown + 1
        processed = available
        history_x = history_y = None
        if (t and available and t % config.scheduled_every == 0) or alarm:
            history_x = np.concatenate([stream.x_fit, stream.x_cal, stream.x_live[:available]])
            history_y = np.concatenate([stream.y_fit, stream.y_cal, stream.y_live[:available]])
        if t and available and t % config.scheduled_every == 0:
            models["scheduled"] = _fit(history_x[-config.train_window:], history_y[-config.train_window:])
            refits["scheduled"] += 1
        if alarm:
            models["detected"] = _fit(history_x[-config.train_window:], history_y[-config.train_window:])
            refits["detected"] += 1
        for name, model in models.items():
            pred = float(np.asarray(model.predict(stream.x_live[t:t+1])).reshape(-1)[0])
            if name == "frozen":
                reference_forecasts.append(pred)
            actual = float(stream.y_live[t])
            true_mean = float(stream.mean_live[t])
            rows.append({"step": t, "policy": name, "available_labels": available,
                "prediction": pred, "squared_error": (actual-pred)**2,
                "alarm": int(alarm) if name=="detected" else 0,
                "alarm_source": source_index if name=="detected" else -1,
                "refits": refits[name], "statistic": stat, "threshold": threshold})
            for cost in (0.2, 0.6, 1.0, 1.4):
                action = pred > cost
                profit = (true_mean-cost)*action
                utilities.append({"step":t, "policy":name, "cost":cost,
                    "regret":max(true_mean-cost,0.0)-profit, "profit":profit})
    return pd.DataFrame(rows), pd.DataFrame(utilities)

def run_detector_study(repeats: int = 6, smoke: bool = False, seed: int = 2026) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Use identical streams for all policies and feedback delays."""
    if isinstance(repeats, bool) or not isinstance(repeats,int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    regimes = ("stable","gradual_drift") if smoke else ("stable","gradual_drift","ar1","gradual_ar1")
    delays = (0,3) if smoke else (0,12)
    scores, decisions = [], []
    for replicate in range(repeats):
        for regime in regimes:
            data = generate_delayed_stream(
                seed+replicate, regime=regime, n_fit=80 if smoke else 160,
                n_cal=35 if smoke else 80, n_live=45 if smoke else 120)
            for delay in delays:
                config = DetectorSettings(delay=delay,
                    scheduled_every=10 if smoke else 15,
                    cooldown=10 if smoke else 15,
                    train_window=70 if smoke else 100,
                    bootstrap_draws=10 if smoke else 50)
                a,b = evaluate_detector(data, config, seed=seed+replicate)
                for df in (a,b):
                    df.insert(0,"replicate",replicate)
                    df.insert(0,"regime",regime)
                    df.insert(0,"delay",delay)
                scores.append(a);decisions.append(b)
    return pd.concat(scores,ignore_index=True),pd.concat(decisions,ignore_index=True)

def detector_summary(scores: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Average per-stream metrics; do not treat time steps as independent runs."""
    keys=["regime","delay","replicate","policy"]
    per_stream=scores.groupby(keys,as_index=False).agg(
        rmse=("squared_error",lambda x:float(np.sqrt(np.mean(x)))),
        alarms=("alarm","sum"),refits=("refits","max"))
    loss=decisions.groupby(keys+["cost"],as_index=False).agg(regret=("regret","mean"))
    merged=per_stream.merge(loss,on=keys,validate="one_to_many")
    return merged.groupby(["regime","delay","policy","cost"],as_index=False).agg(
        streams=("replicate","nunique"),rmse=("rmse","mean"),
        alarms=("alarms","mean"),refits=("refits","mean"),regret=("regret","mean"))
