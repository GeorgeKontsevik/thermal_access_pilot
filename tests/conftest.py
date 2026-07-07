from pathlib import Path

import pytest

from thermal_access_pilot.config import load_config


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def real_config(repo_root: Path):
    return load_config(repo_root / "thermal_access_pilot/configs/kaliningrad.toml", repo_root=repo_root)


@pytest.fixture(scope="session")
def real_output_dir(real_config) -> Path:
    return real_config.output_dir

