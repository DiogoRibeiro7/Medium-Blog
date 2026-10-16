import pandas as pd
import pytest

from hansard_drift.build import build
from hansard_drift.check import check
from hansard_drift.estimate import estimate
from hansard_drift.figures import all_figures
from hansard_drift.report import write_report
from hansard_drift.sample import draw
from hansard_drift.synth import autolabel, write_synthetic
from hansard_drift.transfer import transfer


def test_pipeline_recovers_known_drift(cfg):
    write_synthetic(cfg, days_per_era=20)
    s = build(cfg)
    assert s["ids_parsed_share"] == 1.0
    truth = pd.read_parquet(cfg.cache_dir / "truth.parquet")
    assert s["chair_speeches"] == len(truth)

    draw(cfg)
    with pytest.raises(FileExistsError):
        draw(cfg)
    sample = pd.read_csv(cfg.labels_dir / "sample.csv")
    assert "d_tag" not in sample.columns and "era" not in sample.columns

    autolabel(cfg)
    estimate(cfg)
    transfer(cfg)
    assert check(cfg) == []

    chair = pd.read_parquet(cfg.build_dir / "chair.parquet").merge(truth, on="id")
    true_by_era = chair.groupby("era")["truth_discipline"].sum()
    ce = pd.read_csv(cfg.stage2_dir / "corrected_by_era.csv").set_index("era")
    covered = [(ce.loc[e, "interventions_lo"] <= true_by_era[e] <= ce.loc[e, "interventions_hi"]) for e in ce.index]
    assert sum(covered) >= len(covered) - 1

    # The built-in drift: recall in the Bercow era is lower than in the Weatherill era.
    assert ce.loc["Bercow", "recall_median"] < ce.loc["Weatherill", "recall_median"]

    assert len(all_figures(cfg)) == 7
    assert (cfg.root / "results" / "RESULTS.md").exists() or write_report(cfg)
    write_report(cfg)
    assert "Corrected estimates" in (cfg.root / "results" / "RESULTS.md").read_text()
