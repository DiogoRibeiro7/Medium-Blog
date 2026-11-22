"""The forecasting step: one interface, many models.

Every adapter takes a Batch (load contexts, plus the known covariates over context and
horizon) and returns quantile forecasts of shape (series, horizon, quantiles). Nothing is
fitted to the load series: the context window and the covariates are all a model sees.

Covariates, when a model takes them and the variant asks for them:
- temperature: the forecast made two days earlier, numeric;
- day type: an integer code per hour (0 workday, 1 saturday_like, 2 sunday_like; for the
  'flag' variant 0 ordinary, 1 national holiday). Passed as a categorical covariate where
  the model has one (Timestral, Chronos-2) and as one-hot indicators otherwise (t0).

Model packages are imported only inside each adapter, so each model can be installed and
run in its own environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC
from importlib import metadata

import numpy as np

from .config import DAY_TYPES, ForecasterSpec

WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".bin", ".pt", ".pth")


@dataclass
class Batch:
    contexts: np.ndarray  # (B, L) float
    horizon: int
    temperature: np.ndarray | None = None  # (B, L + H)
    calendar: np.ndarray | None = None  # (B, L + H) integer codes
    calendar_levels: tuple[str, ...] = DAY_TYPES

    @property
    def size(self) -> int:
        return self.contexts.shape[0]

    @property
    def length(self) -> int:
        return self.contexts.shape[1]


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _torch():
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return torch


def hub_record(repo: str, revision: str) -> dict:
    """The exact weights that will be used: commit, weights files with their SHA-256, the
    latest commit that changed any of them, and the licence the model card declares."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo, revision=revision or None)
    sha = info.sha
    files = [f for f in api.list_repo_files(repo, revision=sha) if f.endswith(WEIGHT_SUFFIXES)]
    paths = api.get_paths_info(repo, files, revision=sha, expand=True) if files else []
    weights, dates = [], []
    for p in paths:
        commit = getattr(p, "last_commit", None)
        if commit is not None:
            dates.append(commit.date.astimezone(UTC))
        weights.append({"file": p.path, "sha256": p.lfs.sha256 if getattr(p, "lfs", None) else None})
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None) or {}
    licence = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    return {
        "repo": repo,
        "revision": sha,
        "weights": weights,
        "weights_last_commit": max(dates).isoformat() if dates else None,
        "licence_declared": licence,
        "gated": getattr(info, "gated", None),
    }


def snapshot(repo: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision)


def select(levels, quantiles) -> list[int]:
    available = [round(float(q), 6) for q in levels]
    out = []
    for q in quantiles:
        q = round(float(q), 6)
        if q not in available:
            raise ValueError(f"quantile {q} not produced (model gives {available})")
        out.append(available.index(q))
    return out


def one_hot(codes: np.ndarray, levels: tuple[str, ...]) -> np.ndarray:
    """(B, T) codes -> (B, K - 1, T) indicators, the first level as baseline."""
    return np.stack([(codes == k).astype(np.float32) for k in range(1, len(levels))], axis=1)


def sorted_quantiles(q: np.ndarray) -> np.ndarray:
    """Quantiles sorted along the last axis, so the median is never above the 0.9 quantile."""
    return np.sort(np.asarray(q, dtype=np.float64), axis=-1)


# --------------------------------------------------------------------------- adapters


class SeasonalNaive:
    """The same hour one week earlier. Needs the context to reach a week back."""

    def __init__(self, spec: ForecasterSpec, period: int = 168):
        self.spec, self.period = spec, period

    @classmethod
    def load(cls, spec: ForecasterSpec):
        return cls(spec)

    def provenance(self) -> dict:
        return {"period_hours": self.period}

    def forecast(self, batch: Batch, quantiles) -> np.ndarray:
        L, H = batch.length, batch.horizon
        if H > self.period or L < self.period:
            raise ValueError("seasonal naive needs context >= period >= horizon")
        point = batch.contexts[:, L - self.period : L - self.period + H].astype(np.float64)
        return np.repeat(point[:, :, None], len(quantiles), axis=2)


class SimilarDay:
    """The classic operator's rule: copy the most recent day of the same type.

    For each forecast hour, the value at the most recent lag that is a whole number of weeks
    with the same day-type code; failing that, the most recent whole number of days with the
    same code; failing that, one week earlier. Without a calendar covariate it is the
    seasonal naive forecast. It shows what the day-type labels are worth to a model that
    does nothing else with them.
    """

    def __init__(self, spec: ForecasterSpec):
        self.spec = spec

    @classmethod
    def load(cls, spec: ForecasterSpec):
        return cls(spec)

    def provenance(self) -> dict:
        return {"rule": "most recent same-type day, weeks first, then days"}

    def forecast(self, batch: Batch, quantiles) -> np.ndarray:
        L, H = batch.length, batch.horizon
        out = np.empty((batch.size, H))
        for i in range(batch.size):
            y = batch.contexts[i]
            cal = batch.calendar[i] if batch.calendar is not None else None
            for h in range(H):
                t = L + h
                lag = 168
                if cal is not None:
                    lag = self._lag(cal, t, L) or 168
                while t - lag >= L:
                    lag += 168
                out[i, h] = y[t - lag]
        return np.repeat(out[:, :, None], len(quantiles), axis=2)

    @staticmethod
    def _lag(cal: np.ndarray, t: int, L: int) -> int | None:
        for step in (168, 24):
            lag = step
            while t - lag >= 0:
                if t - lag < L and cal[t - lag] == cal[t]:
                    return lag
                lag += step
        return None


class Timestral:
    package = "timestral"

    def __init__(self, spec, forecaster, record):
        self.spec, self.model, self.record = spec, forecaster, record

    @classmethod
    def load(cls, spec: ForecasterSpec):
        _torch()
        from timestral import TimestralForecaster

        record = hub_record(spec.repo, spec.revision)
        model = TimestralForecaster.from_pretrained(snapshot(spec.repo, record["revision"]))
        return cls(spec, model, record)

    def provenance(self) -> dict:
        return {"package": f"{self.package}=={_version(self.package)}", **self.record}

    def forecast(self, batch: Batch, quantiles) -> np.ndarray:
        kwargs = {}
        if batch.temperature is not None:
            kwargs["known_covariates"] = batch.temperature[:, None, :].astype(np.float32)
        if batch.calendar is not None:
            kwargs["categorical_known_covariates"] = batch.calendar[:, None, :].astype(np.float32)
        fc = self.model.forecast(
            batch.contexts[:, None, :].astype(np.float32), prediction_length=batch.horizon, **kwargs
        )
        out = np.stack([np.asarray(fc.quantile(q))[:, 0, :] for q in quantiles], axis=-1)
        return sorted_quantiles(out)


class T0:
    package = "tfc-t0"

    def __init__(self, spec, model, record):
        self.spec, self.model, self.record = spec, model, record
        self.torch = _torch()

    @classmethod
    def load(cls, spec: ForecasterSpec):
        _torch()
        from t0 import T0Forecaster

        record = hub_record(spec.repo, spec.revision)
        model = T0Forecaster.from_pretrained(snapshot(spec.repo, record["revision"])).eval()
        return cls(spec, model, record)

    def provenance(self) -> dict:
        return {"package": f"{self.package}=={_version(self.package)}", **self.record}

    def forecast(self, batch: Batch, quantiles) -> np.ndarray:
        covs = []
        if batch.temperature is not None:
            covs.append(batch.temperature[:, None, :].astype(np.float32))
        if batch.calendar is not None:
            covs.append(one_hot(batch.calendar, batch.calendar_levels))
        kwargs = {"future_covariates": np.concatenate(covs, axis=1)} if covs else {}
        with self.torch.no_grad():
            out = self.model.predict(
                batch.contexts.astype(np.float32), horizon=batch.horizon, quantile_levels=list(quantiles), **kwargs
            )
        q = out.quantiles
        q = q.numpy() if hasattr(q, "numpy") else np.asarray(q)
        return sorted_quantiles(q)


class TiRex:
    package = "tirex-ts"

    def __init__(self, spec, model, record):
        self.spec, self.model, self.record = spec, model, record

    @classmethod
    def load(cls, spec: ForecasterSpec):
        _torch()
        from tirex import load_model

        record = hub_record(spec.repo, spec.revision)
        model = load_model(spec.repo, device="cpu", backend="torch", hf_kwargs={"revision": record["revision"]})
        return cls(spec, model, record)

    def provenance(self) -> dict:
        return {"package": f"{self.package}=={_version(self.package)}", **self.record}

    def forecast(self, batch: Batch, quantiles) -> np.ndarray:
        q, _ = self.model.forecast(
            context=batch.contexts.astype(np.float32),
            prediction_length=batch.horizon,
            output_type="numpy",
            batch_size=self.spec.batch_size,
        )
        q = np.asarray(q, dtype=np.float64)[:, : batch.horizon, :]
        return sorted_quantiles(q[:, :, select(self.model.config.quantiles, quantiles)])


class Chronos2:
    package = "chronos-forecasting"

    def __init__(self, spec, pipeline, record):
        self.spec, self.pipeline, self.record = spec, pipeline, record
        self.torch = _torch()

    @classmethod
    def load(cls, spec: ForecasterSpec):
        _torch()
        from chronos import Chronos2Pipeline

        record = hub_record(spec.repo, spec.revision)
        pipeline = Chronos2Pipeline.from_pretrained(spec.repo, revision=record["revision"], device_map="cpu")
        return cls(spec, pipeline, record)

    def provenance(self) -> dict:
        return {"package": f"{self.package}=={_version(self.package)}", **self.record}

    def forecast(self, batch: Batch, quantiles) -> np.ndarray:
        L, H = batch.length, batch.horizon
        contexts = batch.contexts.astype(np.float32)
        if batch.temperature is None and batch.calendar is None:
            inputs = self.torch.from_numpy(contexts[:, None, :])
        else:
            inputs = []
            levels = np.array(batch.calendar_levels, dtype=object)
            for i in range(batch.size):
                past, future = {}, {}
                if batch.temperature is not None:
                    past["temperature"] = batch.temperature[i, :L].astype(np.float32)
                    future["temperature"] = batch.temperature[i, L:].astype(np.float32)
                if batch.calendar is not None:
                    # Categories, so Chronos-2 target-encodes them from the context.
                    past["day_type"] = levels[batch.calendar[i, :L]].astype(str)
                    future["day_type"] = levels[batch.calendar[i, L:]].astype(str)
                inputs.append({"target": contexts[i], "past_covariates": past, "future_covariates": future})
        q, _ = self.pipeline.predict_quantiles(
            inputs,
            prediction_length=H,
            quantile_levels=list(quantiles),
            batch_size=self.spec.batch_size,
            context_length=L,
        )
        return sorted_quantiles(np.stack([t.float().numpy()[0] for t in q]))


class TimesFM25:
    package = "timesfm"
    deciles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, spec, model, record, settings):
        self.spec, self.model, self.record, self.settings = spec, model, record, settings

    @classmethod
    def load(cls, spec: ForecasterSpec, context: int = 8760, horizon: int = 32):
        _torch()
        import timesfm

        record = hub_record(spec.repo, spec.revision)
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            spec.repo, revision=record["revision"], torch_compile=False
        )
        # The forecast settings shown in the TimesFM 2.5 README.
        settings = dict(
            max_context=int(np.ceil(context / 32) * 32),
            max_horizon=int(np.ceil(horizon / 128) * 128),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=spec.batch_size,
        )
        model.compile(timesfm.ForecastConfig(**settings))
        return cls(spec, model, record, settings)

    def provenance(self) -> dict:
        return {"package": f"{self.package}=={_version(self.package)}", **self.record, "forecast_config": self.settings}

    def forecast(self, batch: Batch, quantiles) -> np.ndarray:
        point, q = self.model.forecast(horizon=batch.horizon, inputs=[row for row in batch.contexts.astype(np.float32)])
        q = np.asarray(q, dtype=np.float64)
        # Documented layout: channel 0 is the mean, channels 1-9 the deciles.
        if q.shape[2] != 10:
            raise ValueError(f"unexpected TimesFM output shape {q.shape}")
        q = q[:, : batch.horizon, 1:]
        return sorted_quantiles(q[:, :, select(self.deciles, quantiles)])


class Sundial:
    package = "transformers"

    def __init__(self, spec, model, record):
        self.spec, self.model, self.record = spec, model, record
        self.torch = _torch()

    @classmethod
    def load(cls, spec: ForecasterSpec):
        _torch()
        from transformers import AutoModelForCausalLM

        record = hub_record(spec.repo, spec.revision)
        model = AutoModelForCausalLM.from_pretrained(snapshot(spec.repo, record["revision"]), trust_remote_code=True)
        return cls(spec, model, record)

    def provenance(self) -> dict:
        return {
            "package": f"{self.package}=={_version(self.package)}",
            **self.record,
            "num_samples": self.spec.num_samples,
            "quantiles_from": "empirical quantiles of the generated samples",
        }

    def forecast(self, batch: Batch, quantiles) -> np.ndarray:
        S, H = self.spec.num_samples, batch.horizon
        with self.torch.no_grad():
            out = self.model.generate(
                self.torch.from_numpy(batch.contexts.astype(np.float32)), max_new_tokens=H, num_samples=S
            )
        samples = np.asarray(out.float().numpy() if hasattr(out, "float") else out, dtype=np.float64)
        if samples.ndim == 2:
            samples = samples[:, None, :]
        if samples.shape[1] != S and samples.shape[2] == S:
            samples = samples.transpose(0, 2, 1)
        samples = samples[:, :, :H]
        q = np.quantile(samples, list(quantiles), axis=1)  # (Q, B, H)
        return sorted_quantiles(np.moveaxis(q, 0, -1))


ADAPTERS = {
    "seasonal_naive": SeasonalNaive,
    "similar_day": SimilarDay,
    "timestral": Timestral,
    "t0": T0,
    "tirex": TiRex,
    "chronos2": Chronos2,
    "timesfm25": TimesFM25,
    "sundial": Sundial,
}


def load(spec: ForecasterSpec, context: int, horizon: int):
    if spec.kind not in ADAPTERS:
        raise ValueError(f"{spec.key}: unknown kind {spec.kind!r}")
    if spec.kind == "timesfm25":
        return TimesFM25.load(spec, context=context, horizon=horizon)
    return ADAPTERS[spec.kind].load(spec)
