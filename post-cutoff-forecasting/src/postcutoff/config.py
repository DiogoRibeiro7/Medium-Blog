"""Reading the experiment definition."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Mask:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    reason: str


@dataclass(frozen=True)
class Window:
    name: str
    first_origin: pd.Timestamp
    last_origin: pd.Timestamp


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    label: str
    spec: str = ""
    repo: str = ""
    revision: str = ""
    batch_size: int = 32
    covariates: tuple[str, ...] = ()

    @property
    def is_foundation(self) -> bool:
        return self.kind in {"chronos2", "timesfm25", "tirex"}


@dataclass(frozen=True)
class Config:
    path: Path
    root: Path
    raw: dict
    endpoint: str
    country: str
    series: str
    first_day: pd.Timestamp
    settle_days: int
    pause_seconds: float
    snapshot: Path
    manifest: Path
    attribution: str
    masks: tuple[Mask, ...]
    anomaly_low: float
    anomaly_high: float
    max_missing_hours: int
    context_length: int
    horizon: int
    origin_hour_utc: int
    quantiles: tuple[float, ...]
    seasonal_period: int
    reference_model: str
    releases: dict[str, pd.Timestamp]
    windows: tuple[Window, ...]
    before: tuple[str, ...]
    after: tuple[str, ...]
    placebo_before: tuple[str, ...]
    placebo_after: tuple[str, ...]
    models: dict[str, ModelSpec] = field(default_factory=dict)
    bootstrap_replications: int = 2000
    bootstrap_block_days: int = 14
    bootstrap_seed: int = 0
    example_origins: tuple[pd.Timestamp, ...] = ()

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def forecasts_dir(self) -> Path:
        return self.root / "forecasts"

    def window(self, name: str) -> Window:
        for w in self.windows:
            if w.name == name:
                return w
        raise KeyError(name)

    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def _utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_config(path: str | Path, root: str | Path | None = None) -> Config:
    """Read a TOML experiment definition.

    Relative paths in the file are resolved against ``root``, which defaults to the
    directory above the one holding the file (the project directory).
    """
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    root = Path(root).resolve() if root is not None else path.parent.parent

    data = raw["data"]
    design = raw["design"]
    hour = int(design["origin_hour_utc"])

    masks = tuple(Mask(m["name"], _utc(m["start"]), _utc(m["end"]), m.get("reason", "")) for m in data.get("masks", []))
    for m in masks:
        if not m.end > m.start:
            raise ValueError(f"mask {m.name} ends before it starts")

    windows = tuple(
        Window(
            w["name"],
            _utc(w["first_origin"]) + pd.Timedelta(hours=hour),
            _utc(w["last_origin"]) + pd.Timedelta(hours=hour),
        )
        for w in raw["windows"]
    )
    names = [w.name for w in windows]
    if len(set(names)) != len(names):
        raise ValueError("window names must be unique")
    contrast = raw.get("contrast", {})
    before = tuple(contrast.get("before", []))
    after = tuple(contrast.get("after", []))
    placebo_before = tuple(contrast.get("placebo_before", []))
    placebo_after = tuple(contrast.get("placebo_after", []))
    for n in before + after + placebo_before + placebo_after:
        if n not in names:
            raise ValueError(f"contrast refers to unknown window {n}")

    models = {}
    for name, m in raw["models"].items():
        models[name] = ModelSpec(
            name=name,
            kind=m["kind"],
            label=m.get("label", name),
            spec=m.get("spec", ""),
            repo=m.get("repo", ""),
            revision=m.get("revision", ""),
            batch_size=int(m.get("batch_size", 32)),
            covariates=tuple(m.get("covariates", [])),
        )
    reference = design["reference_model"]
    if reference not in models:
        raise ValueError(f"reference model {reference} is not defined")

    quantiles = tuple(float(q) for q in design["quantiles"])
    if 0.5 not in quantiles or 0.1 not in quantiles or 0.9 not in quantiles:
        raise ValueError("quantiles must include 0.1, 0.5 and 0.9")
    if list(quantiles) != sorted(quantiles):
        raise ValueError("quantiles must be increasing")

    boot = raw.get("bootstrap", {})
    examples = raw.get("examples", {})
    return Config(
        path=path,
        root=root,
        raw=raw,
        endpoint=data["endpoint"],
        country=data["country"],
        series=data["series"],
        first_day=_utc(data["first_day"]),
        settle_days=int(data.get("settle_days", 3)),
        pause_seconds=float(data.get("pause_seconds", 35)),
        snapshot=root / data["snapshot"],
        manifest=root / data["manifest"],
        attribution=data.get("attribution", ""),
        masks=masks,
        anomaly_low=float(data.get("anomaly", {}).get("low", 0.5)),
        anomaly_high=float(data.get("anomaly", {}).get("high", 2.0)),
        max_missing_hours=int(data.get("max_missing_hours", 168)),
        context_length=int(design["context_length"]),
        horizon=int(design["horizon"]),
        origin_hour_utc=hour,
        quantiles=quantiles,
        seasonal_period=int(design["seasonal_period"]),
        reference_model=reference,
        releases={k: _utc(v["date"]) for k, v in raw.get("releases", {}).items()},
        windows=windows,
        before=before,
        after=after,
        placebo_before=placebo_before,
        placebo_after=placebo_after,
        models=models,
        bootstrap_replications=int(boot.get("replications", 2000)),
        bootstrap_block_days=int(boot.get("block_days", 14)),
        bootstrap_seed=int(boot.get("seed", 0)),
        example_origins=tuple(_utc(o) + pd.Timedelta(hours=hour) for o in examples.get("origins", [])),
    )
