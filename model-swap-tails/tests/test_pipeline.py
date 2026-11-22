import json

from swaptails.cli import main


def test_end_to_end_on_synthetic_data(synthetic):
    cfg = synthetic
    config = str(cfg.path)
    main(["--config", config, "validate"])
    for llm in cfg.llms:
        main(["--config", config, "labels", "--llm", llm])
    for model in cfg.forecasters:
        main(["--config", config, "forecast", "--model", model])
    main(["--config", config, "summarize"])
    main(["--config", config, "figures"])
    main(["--config", config, "check"])
    s = json.loads((cfg.results_dir / "summary.json").read_text())
    fc = s["forecasting"]
    assert set(fc["headline_runs"]) == {"seasonal_naive|none", "similar_day|flag"}
    assert "tso|reference" in fc["runs"]
    # The rule stand-ins answer like the national flag, so their accuracy equals the flag rule's.
    lab = s["labels"]
    assert lab["llms"]["rule_eu"]["accuracy"] == lab["rules"]["flag_rule"]["accuracy"]
    # On synthetic data built from the calendar, the true day type helps the similar-day rule.
    prop = s["propagation"]["similar_day"]
    assert prop["oracle"]["informative_wape"] < prop["none"]["informative_wape"]
    assert (cfg.figures_dir / "fig1_average_vs_tail.png").exists()
    assert (cfg.results_dir / "RESULTS.md").read_text().startswith("# Results")
