"""Zero-shot forecasts from the three foundation models.

Each adapter returns quantile forecasts with shape (origins, horizon, quantiles) for the
quantile levels the experiment asks for, and a provenance record: the package version,
the exact weights revision and the settings used. Nothing is fitted to the Portuguese
series; the context window is the only information about it that the model receives.
"""

from __future__ import annotations

import os
from datetime import UTC
from importlib import metadata

import numpy as np

from .config import ModelSpec


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def resolve_revision(repo: str, revision: str) -> str:
    """The commit hash of the weights that will be downloaded."""
    from huggingface_hub import HfApi

    return HfApi().model_info(repo, revision=revision or None).sha


WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".bin", ".pt", ".pth")


def weights_record(repo: str, revision: str, filename: str) -> dict:
    """The weights files at this revision and the latest commit that changed any of them.

    ``filename`` is the file the package loads; every other weights file in the repository
    (in a format the package could load) is included, so the date is an upper bound on
    when the weights last changed. Files the package does not read, such as an ONNX export,
    are left out.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    files = [f for f in api.list_repo_files(repo, revision=revision) if f.endswith(WEIGHT_SUFFIXES)]
    if filename not in files:
        raise ValueError(f"{repo}@{revision}: expected weights file {filename}, found {files}")
    info = api.get_paths_info(repo, files, revision=revision, expand=True)
    commits = []
    for f in info:
        if getattr(f, "last_commit", None) is None:
            raise ValueError(f"{repo}@{revision}: no commit information for {f.path}")
        commits.append((f.last_commit.date.astimezone(UTC), f))
    latest_date, latest = max(commits, key=lambda c: c[0])
    loaded = next(f for _, f in commits if f.path == filename)
    return {
        "weights_file": filename,
        "weights_sha256": loaded.lfs.sha256 if loaded.lfs is not None else None,
        "weights_files_checked": sorted(f.path for _, f in commits),
        "weights_last_commit": latest_date.isoformat(),
        "weights_commit_title": latest.last_commit.title,
    }


def _torch():
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return torch


def select_quantiles(levels_available, quantiles) -> list[int]:
    available = [round(float(q), 6) for q in levels_available]
    index = []
    for q in quantiles:
        q = round(float(q), 6)
        if q not in available:
            raise ValueError(f"quantile {q} is not produced by the model (it produces {available})")
        index.append(available.index(q))
    return index


def as_category(flags) -> np.ndarray:
    """A 0/1 calendar flag as a categorical covariate.

    Chronos-2 standardises a numeric covariate with its mean and spread in the context, so a
    flag that is 0 throughout the context would turn a 1 in the forecast period into an
    extreme value. As a category it is target-encoded from the context instead, and a
    category the context has not seen carries no information.
    """
    return np.where(np.asarray(flags) > 0, "holiday", "other")


class Chronos2:
    package = "chronos-forecasting"
    weights_file = "model.safetensors"

    def __init__(self, spec: ModelSpec, pipeline, context_length: int, revision: str = "", weights: dict | None = None):
        self.spec = spec
        self.pipeline = pipeline
        self.context_length = context_length
        self.revision = revision
        self.weights = weights or {}
        self.torch = _torch()

    @classmethod
    def from_hub(cls, spec: ModelSpec, context_length: int, **_):
        revision = resolve_revision(spec.repo, spec.revision)
        _torch()
        from chronos import Chronos2Pipeline

        weights = weights_record(spec.repo, revision, cls.weights_file)
        pipeline = Chronos2Pipeline.from_pretrained(spec.repo, revision=revision, device_map="cpu")
        return cls(spec, pipeline, context_length, revision, weights)

    def provenance(self) -> dict:
        return {
            "package": f"{self.package}=={_version(self.package)}",
            "repo": self.spec.repo,
            "revision": self.revision,
            **self.weights,
            "covariates": list(self.spec.covariates),
            "batch_size": self.spec.batch_size,
            "context_length_used": self.context_length,
            "model_context_length": int(self.pipeline.model_context_length),
            "model_quantiles": [float(q) for q in self.pipeline.quantiles],
            "cross_learning": False,
        }

    def forecast(self, contexts, horizon, quantiles, past_covariates=None, future_covariates=None):
        contexts = np.asarray(contexts, dtype=np.float32)
        if self.spec.covariates:
            if past_covariates is None or future_covariates is None:
                raise ValueError(f"{self.spec.name} needs covariates {self.spec.covariates}")
            inputs = [
                {
                    "target": contexts[i],
                    "past_covariates": {k: as_category(past_covariates[k][i]) for k in self.spec.covariates},
                    "future_covariates": {k: as_category(future_covariates[k][i]) for k in self.spec.covariates},
                }
                for i in range(contexts.shape[0])
            ]
        else:
            inputs = self.torch.from_numpy(contexts[:, None, :])
        q, _ = self.pipeline.predict_quantiles(
            inputs,
            prediction_length=horizon,
            quantile_levels=list(quantiles),
            batch_size=self.spec.batch_size,
            context_length=self.context_length,
        )
        # One tensor per series, shape (n_variates=1, horizon, quantiles).
        return np.stack([t.float().numpy()[0] for t in q]).astype(np.float64)


class TimesFM25:
    package = "timesfm"
    # Output channel 0 is the mean; channels 1-9 are the deciles 0.1 ... 0.9.
    deciles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    weights_file = "model.safetensors"

    def __init__(
        self, spec: ModelSpec, model, context_length: int, horizon: int, revision: str = "", weights: dict | None = None
    ):
        import timesfm

        _torch()
        self.spec = spec
        self.model = model
        self.revision = revision
        self.weights = weights or {}
        # The settings shown in the TimesFM 2.5 README.
        self.settings = dict(
            max_context=context_length,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=spec.batch_size,
        )
        self.model.compile(timesfm.ForecastConfig(**self.settings))

    @classmethod
    def from_hub(cls, spec: ModelSpec, context_length: int, horizon: int, **_):
        revision = resolve_revision(spec.repo, spec.revision)
        _torch()
        import timesfm

        weights = weights_record(spec.repo, revision, cls.weights_file)
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(spec.repo, revision=revision, torch_compile=False)
        return cls(spec, model, context_length, horizon, revision, weights)

    def provenance(self) -> dict:
        return {
            "package": f"{self.package}=={_version(self.package)}",
            "repo": self.spec.repo,
            "revision": self.revision,
            **self.weights,
            "forecast_config": self.settings,
        }

    def forecast(self, contexts, horizon, quantiles, **_):
        contexts = np.asarray(contexts, dtype=np.float32)
        # forecast() appends padding rows to the list it is given, so it gets a fresh list.
        point, q = self.model.forecast(horizon=horizon, inputs=[row for row in contexts])
        q = np.asarray(q, dtype=np.float64)
        # Guard the documented output layout: ten channels, the point forecast in channel 5.
        if q.shape[2] != 10 or not np.allclose(np.asarray(point, dtype=np.float64), q[..., 5]):
            raise ValueError("unexpected TimesFM output layout")
        q = q[:, :horizon, 1:]
        return q[:, :, select_quantiles(self.deciles, quantiles)]


class TiRex:
    package = "tirex-ts"
    weights_file = "model.ckpt"

    def __init__(self, spec: ModelSpec, model, context_length: int, revision: str = "", weights: dict | None = None):
        _torch()
        self.spec = spec
        self.model = model
        self.context_length = context_length
        self.revision = revision
        self.weights = weights or {}

    @classmethod
    def from_hub(cls, spec: ModelSpec, context_length: int, **_):
        revision = resolve_revision(spec.repo, spec.revision)
        _torch()
        from tirex import load_model

        weights = weights_record(spec.repo, revision, cls.weights_file)
        model = load_model(spec.repo, device="cpu", backend="torch", hf_kwargs={"revision": revision})
        return cls(spec, model, context_length, revision, weights)

    def provenance(self) -> dict:
        cfg = self.model.config
        return {
            "package": f"{self.package}=={_version(self.package)}",
            "repo": self.spec.repo,
            "revision": self.revision,
            **self.weights,
            "backend": "torch",
            "train_ctx_len": None if cfg.train_ctx_len is None else int(cfg.train_ctx_len),
            "context_length_used": self.context_length,
            "model_quantiles": [float(q) for q in cfg.quantiles],
            "batch_size": self.spec.batch_size,
        }

    def forecast(self, contexts, horizon, quantiles, **_):
        contexts = np.asarray(contexts, dtype=np.float32)
        q, _ = self.model.forecast(
            context=contexts, prediction_length=horizon, output_type="numpy", batch_size=self.spec.batch_size
        )
        q = np.asarray(q, dtype=np.float64)[:, :horizon, :]
        return q[:, :, select_quantiles(self.model.config.quantiles, quantiles)]


ADAPTERS = {"chronos2": Chronos2, "timesfm25": TimesFM25, "tirex": TiRex}


def load(spec: ModelSpec, context_length: int, horizon: int):
    if spec.kind not in ADAPTERS:
        raise ValueError(f"{spec.name} is not a foundation model")
    return ADAPTERS[spec.kind].from_hub(spec, context_length=context_length, horizon=horizon)
