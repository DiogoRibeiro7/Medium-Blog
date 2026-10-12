"""Running every model on every scored origin and storing the scores."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata

import numpy as np
import pandas as pd

from . import __version__, baselines, foundation, metrics
from .calendar_pt import holiday_flags
from .config import Config, ModelSpec
from .data import anomalies, load_snapshot, prepare, sha256
from .origins import build_origins, contexts, targets

PACKAGES = [
    "numpy",
    "pandas",
    "statsforecast",
    "holidays",
    "torch",
    "transformers",
    "chronos-forecasting",
    "timesfm",
    "tirex-ts",
    "huggingface-hub",
]


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


class Inputs:
    """Everything the models and the scorer need, built once per run."""

    def __init__(self, cfg: Config, limit_per_window: int | None = None):
        self.cfg = cfg
        series = load_snapshot(cfg)
        flagged = anomalies(series, cfg)
        unmasked = flagged[~flagged["masked"]]
        if not unmasked.empty:
            raise ValueError(
                "unmasked anomalous hours in the snapshot; inspect them and add a mask to the config:\n"
                + unmasked.head(20).to_string(index=False)
            )
        self.series = series
        self.prep = prepare(series, cfg)
        self.origins = build_origins(self.prep, cfg)
        scored = self.origins[self.origins["status"] == "scored"]
        if limit_per_window:
            scored = scored.groupby("window", group_keys=False).tail(limit_per_window)
        self.scored = scored.reset_index(drop=True)
        if self.scored.empty:
            raise ValueError("no origin can be scored with this snapshot")
        pos = self.scored["pos"].to_numpy()
        C, H = cfg.context_length, cfg.horizon
        self.X = contexts(self.prep, pos, C)
        self.Y = targets(self.prep, pos, H)
        self.scale = metrics.seasonal_scale(self.X, cfg.seasonal_period)
        hol = holiday_flags(self.prep.index)
        self.holiday_context = hol[pos[:, None] + np.arange(-C, 0)[None, :]]
        self.holiday_target = hol[pos[:, None] + np.arange(H)[None, :]]
        self.data_sha = sha256(cfg.snapshot)

    def key(self, spec: ModelSpec) -> str:
        """What a stored forecast depends on; a mismatch means it is recomputed."""
        arrays = hashlib.sha256()
        for a in (self.X, self.holiday_context, self.holiday_target):
            arrays.update(np.ascontiguousarray(a).tobytes())
        payload = {
            "inputs_sha": arrays.hexdigest(),
            "horizon": self.cfg.horizon,
            "quantiles": list(self.cfg.quantiles),
            "model": asdict(spec),
            "origins": [o.isoformat() for o in self.scored["origin"]],
            "packages": package_versions(),
            "postcutoff": __version__,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def forecast_model(spec: ModelSpec, inputs: Inputs, chunk: int = 256, log=print) -> tuple[np.ndarray, dict]:
    cfg = inputs.cfg
    H, Q = cfg.horizon, cfg.quantiles
    t0 = time.time()
    if spec.kind == "statsforecast":
        f = baselines.forecast(spec.spec, inputs.X, H, Q)
        provenance = {"package": f"statsforecast=={metadata.version('statsforecast')}", "spec": spec.spec}
    elif spec.is_foundation:
        model = foundation.load(spec, cfg.context_length, H)
        provenance = model.provenance()
        check_weights_date(cfg, spec, provenance)
        parts = []
        n = inputs.X.shape[0]
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            kwargs = {}
            if spec.covariates:
                kwargs["past_covariates"] = {"holiday": inputs.holiday_context[start:stop]}
                kwargs["future_covariates"] = {"holiday": inputs.holiday_target[start:stop]}
            parts.append(model.forecast(inputs.X[start:stop], H, Q, **kwargs))
            log(f"  {spec.name}: {stop}/{n} origins, {time.time() - t0:.0f} s")
        f = np.concatenate(parts, axis=0)
    else:
        raise ValueError(f"unknown model kind {spec.kind}")
    expected = (inputs.X.shape[0], H, len(Q))
    if f.shape != expected:
        raise ValueError(f"{spec.name} returned shape {f.shape}, expected {expected}")
    if not np.isfinite(f).all():
        raise ValueError(f"{spec.name} returned non-finite forecasts")
    provenance["seconds"] = round(time.time() - t0, 1)
    provenance["quantile_crossing_rate"] = metrics.crossing_rate(f)
    return f, provenance


def check_weights_date(cfg: Config, spec: ModelSpec, provenance: dict) -> None:
    """Stop if the weights were changed after the first forecast period of the window the
    comparison treats as unseen."""
    if not cfg.after:
        return
    first = min(cfg.window(w).first_origin for w in cfg.after)
    date = provenance.get("weights_last_commit")
    if date and pd.Timestamp(date) >= first:
        raise ValueError(
            f"{spec.name}: weights last changed on {date}, after the start of the window "
            f"{', '.join(cfg.after)} ({first.isoformat()}); pin an earlier revision in the config"
        )


def alignment(Y: np.ndarray, median: np.ndarray, max_lag: int = 3) -> dict[int, float]:
    """Mean absolute error of the median against the target shifted by each lag in hours.

    A forecast aligned with its target has its smallest error at lag 0.
    """
    out = {}
    H = Y.shape[1]
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            out[lag] = float(np.mean(np.abs(median[:, : H - lag] - Y[:, lag:])))
        else:
            out[lag] = float(np.mean(np.abs(median[:, -lag:] - Y[:, : H + lag])))
    return out


def run(cfg: Config, model_names=None, force: bool = False, limit_per_window: int | None = None, log=print):
    started = _now()
    inputs = Inputs(cfg, limit_per_window)
    names = list(model_names or cfg.models)
    for n in names:
        if n not in cfg.models:
            raise KeyError(f"model {n} is not defined in the config")
    if cfg.reference_model not in names:
        raise ValueError(f"the reference model {cfg.reference_model} must be run")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.forecasts_dir.mkdir(parents=True, exist_ok=True)
    inputs.origins.assign(origin=inputs.origins["origin"].map(lambda t: t.isoformat())).to_csv(
        cfg.results_dir / "origins.csv", index=False, lineterminator="\n"
    )
    log(f"{len(inputs.scored)} scored origins; running {', '.join(names)}")

    frames, provenance, forecasts = [], {}, {}
    for name in names:
        spec = cfg.models[name]
        key = inputs.key(spec)
        path = cfg.forecasts_dir / f"{name}.npz"
        f = None
        if path.exists() and not force:
            stored = np.load(path, allow_pickle=False)
            if str(stored["key"]) == key:
                f = stored["forecasts"].astype(np.float64)
                provenance[name] = json.loads(str(stored["provenance"]))
                provenance[name]["reused"] = True
                log(f"{name}: reusing stored forecasts")
        if f is None:
            log(f"{name}: forecasting")
            f, prov = forecast_model(spec, inputs, log=log)
            provenance[name] = prov
            np.savez_compressed(
                path,
                key=np.array(key),
                origins=inputs.scored["origin"].map(lambda t: t.isoformat()).to_numpy(dtype=str),
                quantiles=np.asarray(cfg.quantiles),
                forecasts=f.astype(np.float32),
                provenance=np.array(json.dumps(prov)),
            )
            # Score what was stored, so a rerun from disk gives identical numbers.
            f = f.astype(np.float32).astype(np.float64)
        forecasts[name] = f
        s = metrics.score(inputs.Y, f, cfg.quantiles, inputs.scale, inputs.holiday_target)
        s.insert(0, "model", name)
        frames.append(s)

    scores = pd.concat(frames, ignore_index=True)
    meta = inputs.scored[["window", "origin", "context_imputed_hours"]].reset_index(drop=True)
    scores = scores.merge(meta, left_on="origin_idx", right_index=True).drop(columns="origin_idx")
    scores["origin"] = scores["origin"].map(lambda t: t.isoformat())
    cols = ["window", "origin", "model", "lead_day"] + [
        c for c in scores.columns if c not in ("window", "origin", "model", "lead_day")
    ]
    scores = scores[cols].sort_values(["window", "origin", "model", "lead_day"])
    scores.to_csv(cfg.results_dir / "scores.csv", index=False, float_format="%.6g", lineterminator="\n")

    write_examples(cfg, inputs, forecasts)

    excluded = inputs.origins[inputs.origins["status"] == "excluded"]
    manifest = {
        "started": started,
        "finished": _now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "config_sha256": cfg.digest(),
        "data_sha256": inputs.data_sha,
        "data_first_hour": inputs.series.index.min().isoformat(),
        "data_last_hour": inputs.series.index.max().isoformat(),
        "limit_per_window": limit_per_window,
        "origins": {
            w.name: {
                "candidates": int((inputs.origins["window"] == w.name).sum()),
                "scored": int((inputs.scored["window"] == w.name).sum()),
            }
            for w in cfg.windows
        },
        "excluded_reasons": excluded.groupby(["window", "reason"]).size().rename("n").reset_index().to_dict("records"),
        "imputed_hours_in_snapshot": int(inputs.prep.imputed.sum()),
        "models": provenance,
    }
    (cfg.results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {cfg.results_dir / 'scores.csv'} ({len(scores)} rows)")
    return scores, manifest


def smoke(cfg: Config, model_names=None, n: int = 3, log=print) -> list[str]:
    """Run each model on the last ``n`` origins of the last window and check the output
    before committing an hour of compute to the full run."""
    inputs = Inputs(cfg)
    last = cfg.windows[-1].name
    keep = inputs.scored.index[inputs.scored["window"] == last][-n:]
    if len(keep) == 0:
        keep = inputs.scored.index[-n:]
    X, Y = inputs.X[keep], inputs.Y[keep]
    problems = []
    for name in model_names or cfg.models:
        spec = cfg.models[name]
        t0 = time.time()
        sub = _Subset(inputs, keep)
        try:
            f, prov = forecast_model(spec, sub, log=lambda *_: None)
        except Exception as exc:  # report every model before failing
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        med = np.sort(f, axis=2)[:, :, list(cfg.quantiles).index(0.5)]
        lags = alignment(Y, med)
        best = min(lags, key=lags.get)
        if best != 0:
            log(f"{name}: smallest error at lag {best:+d} h (error by lag: {lags})")
        level = X[:, -168:].mean(axis=1, keepdims=True)
        ratio = med / level
        mae = float(np.mean(np.abs(Y - med)))
        log(
            f"{name}: {time.time() - t0:.1f} s for {len(keep)} origins; MAE {mae:.0f} MW; "
            f"median/level in [{ratio.min():.2f}, {ratio.max():.2f}]; "
            f"crossing rate {prov['quantile_crossing_rate']:.3f}; revision {prov.get('revision', '-')}"
        )
        if ratio.min() < 0.4 or ratio.max() > 2.0:
            problems.append(f"{name}: median forecast implausible relative to the last week's level")
    return problems


class _Subset:
    """The same interface as Inputs, restricted to some origins."""

    def __init__(self, inputs: Inputs, rows):
        self.cfg = inputs.cfg
        self.X = inputs.X[rows]
        self.Y = inputs.Y[rows]
        self.holiday_context = inputs.holiday_context[rows]
        self.holiday_target = inputs.holiday_target[rows]


def write_examples(cfg: Config, inputs: Inputs, forecasts: dict) -> None:
    rows = []
    origins = list(inputs.scored["origin"])
    for o in cfg.example_origins:
        if o not in origins:
            continue
        i = origins.index(o)
        times = pd.date_range(o, periods=cfg.horizon, freq="h", tz="UTC")
        for name, f in forecasts.items():
            frame = pd.DataFrame(
                {"origin": o.isoformat(), "model": name, "timestamp_utc": times.map(lambda t: t.isoformat())}
            )
            frame["actual"] = inputs.Y[i]
            frame["holiday"] = inputs.holiday_target[i]
            for k, q in enumerate(cfg.quantiles):
                frame[f"q{int(round(q * 100)):02d}"] = np.sort(f[i], axis=1)[:, k]
            rows.append(frame)
        # The four weeks before the origin, for context in the figure.
        hist = pd.DataFrame(
            {
                "origin": o.isoformat(),
                "model": "history",
                "timestamp_utc": pd.date_range(o - pd.Timedelta(hours=672), periods=672, freq="h", tz="UTC").map(
                    lambda t: t.isoformat()
                ),
                "actual": inputs.X[i][-672:],
            }
        )
        rows.append(hist)
    path = cfg.results_dir / "examples.csv"
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
    elif path.exists():
        path.unlink()
