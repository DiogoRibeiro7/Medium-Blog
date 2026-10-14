from pathlib import Path

import pytest

from postcutoff.config import load_config

PROJECT = Path(__file__).resolve().parents[1]


def make_project(tmp_path: Path, edits: dict | None = None) -> Path:
    """A copy of the experiment config in a temporary project directory."""
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    text = (PROJECT / "config" / "experiment.toml").read_text(encoding="utf-8")
    for old, new in (edits or {}).items():
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "config" / "experiment.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def cfg(tmp_path):
    return load_config(make_project(tmp_path))


@pytest.fixture
def fast_cfg(tmp_path):
    """Fewer bootstrap draws, for end-to-end tests."""
    return load_config(make_project(tmp_path, {"replications = 2000": "replications = 200"}))
