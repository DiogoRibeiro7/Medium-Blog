"""Expanding-window refits of the Bayesian specifications (``PREDICTIVE_PROTOCOL.md``).

Expensive: three specifications times 32 origins take about two CPU-hours. Run by
``eu-okun predictive-fit`` and the separate predictive workflow, never by the ordinary
pull-request checks. Each origin is written to its own directory, so an interrupted run
resumes where it stopped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.bayes_fit import (
    PROTOCOL_SETTINGS,
    FitResult,
    SamplerSettings,
    environment_versions,
    fit_specification,
)
from eu_okun.bayes_model import SPECS
from eu_okun.contract import EurostatDataError
from eu_okun.predictive import (
    BAYES_MODELS,
    EVALUATION_SETS,
    LAGS,
    bayes_predictions,
    build_predictive_data,
    write_predictions,
)
from eu_okun.review import read_flags

REPLICATES = 4               # innovation and factor draws per posterior draw (section 4)
RETRY_WARMUP = 3000          # the single retry of BAYES_PROTOCOL.md, section 6


def flat_draws(samples: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Merge chains: (chain, draw, ...) -> (chain * draw, ...)."""
    return {name: values.reshape(-1, *values.shape[2:]) for name, values in samples.items()}


def _attempt(fit: FitResult) -> dict[str, Any]:
    return {**asdict(fit.diagnostics), "accepted": fit.diagnostics.accepted,
            "failures": fit.diagnostics.failures(), "warmup": fit.settings.warmup,
            "runtime_seconds": fit.runtime_seconds}


def _segments(quarters: Sequence[str], starts: np.ndarray) -> list[list[str]]:
    edges = [*np.flatnonzero(starts).tolist(), len(quarters)]
    return [[quarters[a], quarters[b - 1]] for a, b in zip(edges[:-1], edges[1:], strict=True)]


def run_predictive_refits(
    data_dir: Path, output_dir: Path, sets: Sequence[str], models: Sequence[str],
    settings: SamplerSettings = PROTOCOL_SETTINGS, retry_warmup: int | None = RETRY_WARMUP,
    targets: Sequence[str] | None = None, resume: bool = True,
) -> list[dict[str, Any]]:
    """Refit each Bayesian model at each origin and save its scored predictions.

    Output: ``<output_dir>/<set>/<model>/<target quarter>/`` with ``predictions.csv`` and
    ``origin.json`` (diagnostics of every attempt). ``origin.json`` is written last, so
    its presence marks a finished origin; with ``resume`` finished origins are skipped.
    """
    unknown = [s for s in sets if s not in EVALUATION_SETS]
    unknown += [m for m in models if m not in BAYES_MODELS]
    if unknown:
        raise ValueError(f"Unknown evaluation sets or models: {', '.join(unknown)}.")
    audit = audit_snapshot(data_dir)
    csv_path = data_dir / "processed" / "eu27_quarterly.csv"
    panel, flags = _read_panel(csv_path), read_flags(csv_path)
    snapshot = {"manifest_sha256": audit["snapshot_manifest_sha256"],
                "processed_csv_sha256": audit["processed_csv_sha256"]}
    environment = environment_versions()
    records: list[dict[str, Any]] = []
    for set_name in sets:
        pdata = build_predictive_data(panel, flags, EVALUATION_SETS[set_name])
        for model in models:
            spec = SPECS[BAYES_MODELS[model]]
            if spec.lags != LAGS:
                raise ValueError(f"{model} does not use {LAGS} GDP lags.")
            for column in pdata.targets:
                quarter = pdata.data.quarters[column]
                if targets is not None and quarter not in targets:
                    continue
                directory = output_dir / set_name / model / quarter
                done = directory / "origin.json"
                if resume and done.exists():
                    records.append(json.loads(done.read_text(encoding="utf-8")))
                    continue
                train = pdata.training(column)
                fit = fit_specification(train, spec, settings)
                attempts = [_attempt(fit)]
                if not fit.diagnostics.accepted and retry_warmup is not None:
                    fit = fit_specification(train, spec, replace(settings, warmup=retry_warmup))
                    attempts.append(_attempt(fit))
                rows = bayes_predictions(set_name, model, pdata, column, flat_draws(fit.samples),
                                         spec, REPLICATES, settings.seed + column)
                write_predictions(directory / "predictions.csv", rows)
                record = {
                    "evaluation": set_name, "model": model, "specification": spec.name,
                    "target": quarter, "training_segments": _segments(train.quarters,
                                                                      train.segment_start),
                    "training_quarters": train.n_quarters,
                    "masked_training_terms": int((~train.mask).sum()),
                    "scored_cells": len(rows), "accepted": fit.diagnostics.accepted,
                    "attempts": attempts, "settings": asdict(fit.settings),
                    "predictive_seed": settings.seed + column, "replicates": REPLICATES,
                    "environment": environment, "snapshot": snapshot,
                }
                done.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n",
                                encoding="utf-8", newline="\n")
                records.append(record)
    if not records:
        raise EurostatDataError("No origin was fitted.")
    return records
