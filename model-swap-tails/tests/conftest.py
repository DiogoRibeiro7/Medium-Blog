import shutil
from pathlib import Path

import pytest

from swaptails.config import load_config

FIXTURE = Path(__file__).parent / "fixtures" / "experiment.toml"


@pytest.fixture
def cfg(tmp_path):
    """The small test experiment (Portugal and Germany, January-February 2025), rooted in a
    temporary directory so nothing is written into the repository."""
    (tmp_path / "config").mkdir()
    target = tmp_path / "config" / "experiment.toml"
    shutil.copy(FIXTURE, target)
    return load_config(target, root=tmp_path)


@pytest.fixture
def synthetic(cfg):
    from swaptails.synth import synthesise

    synthesise(cfg, list(cfg.countries), seed=1)
    return cfg
