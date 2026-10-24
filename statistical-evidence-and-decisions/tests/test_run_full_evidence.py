"""Check evidence runner prerequisites without running the expensive study."""
from pathlib import Path

import pytest

from run_full_evidence import execute


def test_runner_rejects_invalid_stream_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        execute(tmp_path, streams=0)
    with pytest.raises(ValueError):
        execute(tmp_path, streams=1, null_streams=1)
