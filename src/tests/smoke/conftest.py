"""
tests/smoke/conftest.py

Smoke 테스트 공용 Fixture.
Shared fixtures for smoke tests.

실제 데이터(data_set/labeled/)가 필요한 테스트.
Tests that require real data (data_set/labeled/).
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def cfg():
    """실제 config.json을 로드한 설정 dict."""
    from utils.utils_config import load_config
    return load_config()


@pytest.fixture
def mini_cfg(cfg):
    """학습 epoch/batch를 최소화한 smoke용 config."""
    c = copy.deepcopy(cfg)
    c["phase0"]["epochs"]     = 3
    c["phase0"]["batch_size"] = 4
    c["phase2"]["epochs"]     = 3
    c["phase2"]["batch_size"] = 4
    c["phase2"]["early_stopping"]["enabled"] = False
    return c


def data_exists(cfg: dict, channel: str) -> bool:
    """labeled/{channel}/ 디렉토리에 실제 데이터가 있는지 확인한다."""
    labeled = Path(cfg["storage"]["labeled_dir"]) / channel
    if not labeled.exists():
        return False
    return any(labeled.rglob("*.png"))
