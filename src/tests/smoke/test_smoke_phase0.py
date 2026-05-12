"""
tests/smoke/test_smoke_phase0.py

Phase 0 Contrastive Learning 전체 파이프라인 Smoke 테스트.
Smoke tests for Phase 0 Contrastive Learning pipeline.

실제 데이터가 있을 때만 의미 있는 테스트 — 데이터 없으면 자동 skip.
Only meaningful with real data — auto-skipped when data is absent.

실행 / Run:
    pytest src/tests/smoke/test_smoke_phase0.py -v -m smoke
    pytest src/tests/smoke/test_smoke_phase0.py -v -k "Y"
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from models.grayspot_model import GrayspotModel
from tests.smoke.conftest import data_exists
from training.trainer import ContrastiveDataset, Phase0Trainer

CHANNELS = ["Y", "M", "C", "K"]


@pytest.mark.smoke
def test_config_has_required_keys(cfg):
    required = ["data", "model", "phase0", "phase2", "storage", "train"]
    missing = [k for k in required if k not in cfg]
    assert not missing, f"누락된 키 / Missing keys: {missing}"


@pytest.mark.smoke
@pytest.mark.parametrize("channel", CHANNELS)
def test_contrastive_dataset_builds(cfg, channel):
    if not data_exists(cfg, channel):
        pytest.skip(f"[{channel}] 데이터 없음 — labeled/{channel}/ 확인 필요")

    ds = ContrastiveDataset(cfg, channel)
    assert len(ds) > 0

    v1, v2 = ds[0]
    assert v1.shape == v2.shape
    assert v1.shape == (3, cfg["data"]["image_size"], cfg["data"]["image_size"])


@pytest.mark.smoke
def test_phase0_model_init(cfg):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GrayspotModel(cfg, phase=0).to(device)
    size = cfg["data"]["image_size"]

    with torch.no_grad():
        out = model(torch.randn(2, 3, size, size).to(device))

    assert out.shape == (2, cfg["phase0"]["projection_dim"])


@pytest.mark.smoke
@pytest.mark.parametrize("channel", CHANNELS)
def test_phase0_mini_training(mini_cfg, channel):
    if not data_exists(mini_cfg, channel):
        pytest.skip(f"[{channel}] 데이터 없음 — 학습 skip")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds = ContrastiveDataset(mini_cfg, channel)
    loader = DataLoader(
        ds,
        batch_size=min(mini_cfg["phase0"]["batch_size"], len(ds)),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    model = GrayspotModel(mini_cfg, phase=0).to(device)
    trainer = Phase0Trainer(model, mini_cfg, channel, device)
    history = trainer.train(loader)

    assert len(history) > 0, "학습 이력이 비어 있음"
    assert all(
        torch.isfinite(torch.tensor(e["loss"])) for e in history
    ), "Loss에 NaN/Inf 포함"


@pytest.mark.smoke
@pytest.mark.parametrize("channel", CHANNELS)
def test_phase0_backbone_saved(mini_cfg, channel):
    if not data_exists(mini_cfg, channel):
        pytest.skip(f"[{channel}] 데이터 없음 — skip")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds = ContrastiveDataset(mini_cfg, channel)
    loader = DataLoader(
        ds,
        batch_size=min(mini_cfg["phase0"]["batch_size"], len(ds)),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    model = GrayspotModel(mini_cfg, phase=0).to(device)
    trainer = Phase0Trainer(model, mini_cfg, channel, device)
    trainer.train(loader)

    path = trainer.save_backbone()
    assert path is not None and path.exists(), f"Backbone 저장 실패: {path}"
    assert path.stat().st_size > 0
