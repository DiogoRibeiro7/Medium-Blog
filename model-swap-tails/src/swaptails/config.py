"""The experiment definition, read from config/experiment.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAY_TYPES = ("workday", "saturday_like", "sunday_like")
BLOCS = ("Europe", "United States", "China", "Reference")


@dataclass(frozen=True)
class Mask:
    name: str
    countries: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    reason: str


@dataclass(frozen=True)
class Country:
    key: str
    name: str
    tz: str
    ec: str
    holidays: str
    exclude_subdivisions: tuple[str, ...]
    cities: tuple[tuple[str, float, float], ...]
    include_subdivisions: tuple[str, ...] = ()
    subdivision_term: str = "regions"


@dataclass(frozen=True)
class LLMSpec:
    key: str
    label: str
    bloc: str
    developer: str
    licence: str
    backend: str
    model: str
    think: bool | None = None
    base_url: str = ""
    api_key_env: str = ""


@dataclass(frozen=True)
class ForecasterSpec:
    key: str
    label: str
    bloc: str
    developer: str
    kind: str
    covariates: bool
    max_context: int
    repo: str = ""
    revision: str = ""
    batch_size: int = 32
    num_samples: int = 100


@dataclass(frozen=True)
class Config:
    path: Path
    root: Path
    load_endpoint: str
    load_series: str
    tso_endpoint: str
    weather_endpoint: str
    weather_variable: str
    first_day: date
    last_day: date
    pause_seconds: float
    cache_dir: Path
    manifest: Path
    attribution_load: str
    attribution_weather: str
    max_missing_hours: int
    masks: tuple[Mask, ...]
    forecast_first_day: date
    forecast_last_day: date
    label_years: tuple[int, ...]
    cutoff_local_hour: int
    context_hours: int
    quantiles: tuple[float, ...]
    seed: int
    truth_window_days: int
    candidate_margin_days: int
    year_end_from: tuple[int, int]
    year_end_to: tuple[int, int]
    control_days_per_year: int
    max_regions_named: int
    prompt_version: str
    tail_quantile: float
    extreme_temperature_quantiles: tuple[float, float]
    bootstrap_replications: int
    bootstrap_block_days: int
    headline_variant: str
    reference_min_coverage: float
    examples: tuple[tuple[str, date], ...]
    countries: dict[str, Country]
    llms: dict[str, LLMSpec]
    forecasters: dict[str, ForecasterSpec]
    variant_names: tuple[str, ...]
    results_dir: Path = field(default=Path("results"))
    figures_dir: Path = field(default=Path("figures"))
    labels_dir: Path = field(default=Path("labels"))
    forecasts_dir: Path = field(default=Path("forecasts"))

    def variants_for(self, forecaster: str) -> list[str]:
        """Covariate variants a forecaster is run with."""
        spec = self.forecasters[forecaster]
        if not spec.covariates:
            return ["none"]
        out = []
        for v in self.variant_names:
            if v == "llm":
                out.extend(f"llm:{k}" for k in self.llms)
            else:
                out.append(v)
        return out

    def headline(self, forecaster: str) -> str:
        """The run that stands for a forecaster in the headline comparison."""
        variant = self.headline_variant if self.forecasters[forecaster].covariates else "none"
        return f"{forecaster}|{variant}"

    def run_names(self) -> list[str]:
        """Every forecaster-variant pair, as written in the results."""
        return [f"{f}|{v}" for f in self.forecasters for v in self.variants_for(f)]

    def masks_for(self, country: str) -> list[Mask]:
        return [m for m in self.masks if country in m.countries]


def _md(text: str) -> tuple[int, int]:
    month, day = text.split("-")
    return int(month), int(day)


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config" / "experiment.toml"
    root = Path(root) if root else path.resolve().parents[1]
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    d, des, lab, ev = raw["data"], raw["design"], raw["labels"], raw["evaluation"]

    masks = tuple(
        Mask(
            name=m["name"],
            countries=tuple(m["countries"]),
            start=pd.Timestamp(m["start"]),
            end=pd.Timestamp(m["end"]),
            reason=m["reason"],
        )
        for m in d.get("masks", [])
    )
    countries = {
        k: Country(
            key=k,
            name=v["name"],
            tz=v["tz"],
            ec=v["ec"],
            holidays=v["holidays"],
            exclude_subdivisions=tuple(v.get("exclude_subdivisions", [])),
            cities=tuple((c[0], float(c[1]), float(c[2])) for c in v["cities"]),
            include_subdivisions=tuple(v.get("include_subdivisions", [])),
            subdivision_term=v.get("subdivision_term", "regions"),
        )
        for k, v in raw["countries"].items()
    }
    llms = {
        k: LLMSpec(
            key=k,
            label=v["label"],
            bloc=v["bloc"],
            developer=v["developer"],
            licence=v["licence"],
            backend=v["backend"],
            model=v["model"],
            think=v.get("think"),
            base_url=v.get("base_url", ""),
            api_key_env=v.get("api_key_env", ""),
        )
        for k, v in raw.get("llms", {}).items()
    }
    forecasters = {
        k: ForecasterSpec(
            key=k,
            label=v["label"],
            bloc=v["bloc"],
            developer=v["developer"],
            kind=v["kind"],
            covariates=bool(v.get("covariates", False)),
            max_context=int(v["max_context"]),
            repo=v.get("repo", ""),
            revision=v.get("revision", ""),
            batch_size=int(v.get("batch_size", 32)),
            num_samples=int(v.get("num_samples", 100)),
        )
        for k, v in raw["forecasters"].items()
    }
    for spec in [*llms.values(), *forecasters.values()]:
        if spec.bloc not in BLOCS:
            raise ValueError(f"{spec.key}: bloc must be one of {BLOCS}, not {spec.bloc!r}")
    for key in [*llms, *forecasters]:
        if "|" in key or ":" in key:
            raise ValueError(f"model key {key!r} may not contain '|' or ':'")

    cfg = Config(
        path=path,
        root=root,
        load_endpoint=d["load_endpoint"],
        load_series=d["load_series"],
        tso_endpoint=d["tso_endpoint"],
        weather_endpoint=d["weather_endpoint"],
        weather_variable=d["weather_variable"],
        first_day=date.fromisoformat(d["first_day"]),
        last_day=date.fromisoformat(d["last_day"]),
        pause_seconds=float(d["pause_seconds"]),
        cache_dir=root / d["cache_dir"],
        manifest=root / d["manifest"],
        attribution_load=d["attribution_load"],
        attribution_weather=d["attribution_weather"],
        max_missing_hours=int(d["max_missing_hours"]),
        masks=masks,
        forecast_first_day=date.fromisoformat(des["forecast_first_day"]),
        forecast_last_day=date.fromisoformat(des["forecast_last_day"]),
        label_years=tuple(int(y) for y in des["label_years"]),
        cutoff_local_hour=int(des["cutoff_local_hour"]),
        context_hours=int(des["context_hours"]),
        quantiles=tuple(float(q) for q in des["quantiles"]),
        seed=int(des["seed"]),
        truth_window_days=int(lab["truth_window_days"]),
        candidate_margin_days=int(lab["candidate_margin_days"]),
        year_end_from=_md(lab["year_end_from"]),
        year_end_to=_md(lab["year_end_to"]),
        control_days_per_year=int(lab["control_days_per_year"]),
        max_regions_named=int(lab["max_regions_named"]),
        prompt_version=lab["prompt_version"],
        tail_quantile=float(ev["tail_quantile"]),
        extreme_temperature_quantiles=tuple(float(q) for q in ev["extreme_temperature_quantiles"]),
        bootstrap_replications=int(ev["bootstrap_replications"]),
        bootstrap_block_days=int(ev["bootstrap_block_days"]),
        headline_variant=ev["headline_variant"],
        reference_min_coverage=float(ev["reference_min_coverage"]),
        examples=tuple((e[0], date.fromisoformat(e[1])) for e in ev.get("examples", [])),
        countries=countries,
        llms=llms,
        forecasters=forecasters,
        variant_names=tuple(raw["variants"]["names"]),
        results_dir=root / "results",
        figures_dir=root / "figures",
        labels_dir=root / "labels",
        forecasts_dir=root / "forecasts",
    )
    if 0.5 not in cfg.quantiles:
        raise ValueError("the quantiles must include 0.5, the point forecast")
    if cfg.first_day > cfg.forecast_first_day:
        raise ValueError("first_day must come before forecast_first_day")
    return cfg
