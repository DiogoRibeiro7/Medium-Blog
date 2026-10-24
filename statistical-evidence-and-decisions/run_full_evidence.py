"""Execute the preregistered detector comparison and archive reproducible evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels

from comparative_change_detection import ComparisonDesign, run_comparison
from evidence_synthesis import PrimaryProtocol, descriptive_context, preregistered_verdict


def execute(output_dir: Path, streams: int = 100, null_streams: int = 199) -> None:
    """Run the unchanged primary protocol and save the complete provenance trail.

    Args:
        output_dir: Existing or new directory for all result tables.
        streams: Number of independent evaluation streams per regime.
        null_streams: Number of disjoint streams per null calibration.
    """
    if isinstance(streams, bool) or not isinstance(streams, int) or streams < 1:
        raise ValueError("streams must be a positive integer")
    if isinstance(null_streams, bool) or not isinstance(null_streams, int) or null_streams < 2:
        raise ValueError("null_streams must be an integer >= 2")

    protocol = PrimaryProtocol()
    design = ComparisonDesign(
        n_fit=160, n_cal=80, n_live=120, null_streams=null_streams,
        delays=(protocol.delay,), action_costs=(protocol.action_cost,),
        refit_costs=(protocol.refit_cost,),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration, models, economics = run_comparison(
        design, evaluation_repeats=streams,
        scenarios=(protocol.scenario,),
        regimes=(protocol.null_regime, protocol.drift_regime),
        seed=2026,
    )
    verdict = preregistered_verdict(models, economics, protocol)
    context = descriptive_context(models, economics)
    if streams < protocol.minimum_streams:
        assert set(verdict["status"]) == {"exploratory_insufficient_streams"}
    assert not verdict.empty
    assert np.isfinite(economics["net_regret"].to_numpy()).all()

    tables: dict[str, pd.DataFrame] = {
        "calibration": calibration, "models": models,
        "economics": economics, "verdict": verdict, "context": context
    }
    checksums: dict[str, str] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.17g")
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol.metadata(),
        "design": {
            "n_fit": design.n_fit, "n_cal": design.n_cal, "n_live": design.n_live,
            "calibration_streams_per_null": design.null_streams,
            "evaluation_streams_per_regime": streams,
            "evaluation_seed": 2026,
            "drift_regime": protocol.drift_regime,
            "null_regime": protocol.null_regime,
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
        },
        "sha256": checksums,
        "status": verdict[["candidate", "status"]].to_dict("records"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Preregistered detector study — execution report", "",
        f"Independent evaluation streams per regime: {streams}.",
        f"Independent null-calibration streams: {null_streams}.", "",
        "Conclusions apply only to the prespecified synthetic setting.", "",
        verdict.to_markdown(index=False), "",
        "Verdicts are conditional on the calibrated thresholds and simulation design.",
        "See manifest.json and complete CSV outputs for reproducibility.", "",
    ]
    # Do not require tabulate as an additional runtime dependency.
    lines[7] = verdict.to_csv(index=False)
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Parse CLI parameters and run an entirely local experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/full-evidence"))
    parser.add_argument("--streams", type=int, default=100)
    parser.add_argument("--null-streams", type=int, default=199)
    args = parser.parse_args()
    execute(args.output, streams=args.streams, null_streams=args.null_streams)


if __name__ == "__main__":
    main()
