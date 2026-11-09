import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import checks, data
from mlp import features as F


@pytest.fixture(scope="session")
def con():
    return data.connect()


@pytest.fixture(scope="session")
def frames(con):
    return F.FeatureCache(con)


@pytest.fixture(scope="session")
def ddl(con):
    return checks._raw_ddl(con)
