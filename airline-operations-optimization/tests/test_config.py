from pathlib import Path

import pytest

from airline_ops.config import load_config, parse_clock
from airline_ops.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def test_both_configs_load() -> None:
    main = load_config(ROOT / "configs" / "experiment.toml")
    smoke = load_config(ROOT / "configs" / "smoke.toml")
    assert main.experiment.replications >= 100
    assert smoke.network.hub == main.network.hub == "MUC"


def test_parse_clock() -> None:
    assert parse_clock("06:15") == 375
    with pytest.raises(ConfigError):
        parse_clock("25:00")
    with pytest.raises(ConfigError):
        parse_clock("noon")


def test_unknown_key_rejected(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "smoke.toml").read_text().replace("[crew]", "[crew]\nsurprise = 1")
    bad = tmp_path / "bad.toml"
    bad.write_text(text)
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(bad)


def test_invalid_value_rejected(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "smoke.toml").read_text().replace("load_factor = 0.82", "load_factor = 1.5")
    bad = tmp_path / "bad.toml"
    bad.write_text(text)
    with pytest.raises(ConfigError, match="load_factor"):
        load_config(bad)


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(ROOT / "configs" / "nope.toml")


def test_digest_ignores_figure_code(tmp_path):
    import shutil
    from pathlib import Path

    from airline_ops.experiments.digest import compute_digest

    root = Path(__file__).resolve().parents[1]
    copy = tmp_path / "airline-operations-optimization"
    shutil.copytree(root / "src", copy / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(root / "configs", copy / "configs")
    shutil.copy(root / "requirements.txt", copy / "requirements.txt")
    before = compute_digest(copy)
    (copy / "src" / "airline_ops" / "viz" / "figures.py").write_text("# redrawn\n")
    assert compute_digest(copy) == before
    (copy / "configs" / "experiment.toml").write_text("changed\n")
    assert compute_digest(copy) != before
