import pytest

from secureagent_iot.config import load_config
from secureagent_iot.simulator import build_registry, build_templates


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def registry():
    return build_registry()


@pytest.fixture(scope="session")
def templates():
    return build_templates()
