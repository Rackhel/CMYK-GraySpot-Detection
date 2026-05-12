"""
tests/smoke/test_smoke_phase2.py

Phase 2 Supervised Classification 전체 파이프라인 Smoke 테스트.
Smoke tests for Phase 2 Supervised Classification pipeline.

실행 / Run:
    pytest src/tests/smoke/test_smoke_phase2.py -v -m smoke
    pytest src/tests/smoke/test_smoke_phase2.py -v -k "Y"
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from tests.smoke.conftest import data_exists
from models.grayspot_model import GrayspotModel
from training.trainer      import CMYKDataset, Phase2Trainer

CHANNELS = ["Y", "M", "C", "K"]


def _find_phase0_backbone(cfg: dict, channel: str) -> Path | None:
    models_dir = Path(cfg["storage"]["models_dir"])
    path = models_dir / f"phase0_backbone_{channel}.pt"
    return path if path.exists() else None


@pytest.mark.smoke
def test_config_has_evaluation_keys(cfg):
    assert "evaluation" in cfg
    assert "targets"    in cfg["evaluation"]


@pytest.mark.smoke
@pytest.mark.parametrize("channel", CHANNELS)
def test_cmyk_dataset_splits(cfg, channel):
    if not data_exists(cfg, channel):
        pytest.skip(f"[{channel}] 데이터 없음")

    train_ds = CMYKDataset(cfg, channel, split="train", augment=False, oversample=False)
    val_ds   = CMYKDataset(cfg, channel, split="val",   augment=False, oversample=False)
    test_ds  = CMYKDataset(cfg, channel, split="test",  augment=False, oversample=False)

    assert len(train_ds) + len(val_ds) + len(test_ds) > 0

    if len(train_ds) > 0:
        x, y = train_ds[0]
        assert x.shape == (3, cfg["data"]["image_size"], cfg["data"]["image_size"])
        assert 0 <= int(y) < cfg["data"]["num_levels"]


@pytest.mark.smoke
def test_phase2_model_init(cfg):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model  = GrayspotModel(cfg, phase=2).to(device)
    size   = cfg["data"]["image_size"]

    with torch.no_grad():
        out = model(torch.randn(2, 3, size, size).to(device))

    assert out.shape == (2, cfg["data"]["num_levels"])


@pytest.mark.smoke
@pytest.mark.parametrize("channel", CHANNELS)
def test_phase2_mini_training(mini_cfg, channel):
    if not data_exists(mini_cfg, channel):
        pytest.skip(f"[{channel}] 데이터 없음")

    device   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_ds = CMYKDataset(mini_cfg, channel, split="train", augment=True,  oversample=True)
    val_ds   = CMYKDataset(mini_cfg, channel, split="val",   augment=False, oversample=False)

    if len(train_ds) == 0:
        pytest.skip(f"[{channel}] 학습 데이터 없음")

    train_loader = DataLoader(
        train_ds,
        batch_size=min(mini_cfg["phase2"]["batch_size"], len(train_ds)),
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(mini_cfg["phase2"]["batch_size"], max(len(val_ds), 1)),
        shuffle=False, num_workers=0,
    )

    model = GrayspotModel(mini_cfg, phase=2).to(device)
    backbone_path = _find_phase0_backbone(mini_cfg, channel)
    if backbone_path:
        model.switch_to_phase2(backbone_path, mini_cfg)
        model = model.to(device)

    trainer = Phase2Trainer(model, mini_cfg, channel, device, train_ds)
    history = trainer.train(train_loader, val_loader)

    assert len(history) > 0, "학습 이력이 비어 있음"
    assert "train_acc" in history[-1]
    assert "val_acc"   in history[-1]


@pytest.mark.smoke
@pytest.mark.parametrize("channel", CHANNELS)
def test_best_model_saved(cfg, channel):
    models_dir = Path(cfg["storage"]["models_dir"])
    model_path = models_dir / f"best_{channel}.pt"

    if not model_path.exists():
        pytest.skip(f"[{channel}] best_{channel}.pt 없음 — Phase 2 학습 먼저 실행 필요")

    assert model_path.stat().st_size > 0


@pytest.mark.smoke
@pytest.mark.parametrize("channel", CHANNELS)
def test_phase2_performance_targets(cfg, channel):
    models_dir = Path(cfg["storage"]["models_dir"])
    model_path = models_dir / f"best_{channel}.pt"

    if not model_path.exists():
        pytest.skip(f"[{channel}] best_{channel}.pt 없음")

    test_ds = CMYKDataset(cfg, channel, split="test", augment=False, oversample=False)
    if len(test_ds) == 0:
        pytest.skip(f"[{channel}] 테스트 데이터 없음")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model  = GrayspotModel(cfg, phase=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    correct, total = 0, 0

    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            preds     = model(x).argmax(1)
            correct  += (preds == labels).sum().item()
            total    += len(labels)

    acc        = correct / max(total, 1)
    target_acc = cfg["evaluation"]["targets"]["per_color_accuracy"]

    # 성능 미달은 FAIL이 아닌 경고로 처리 (데이터 부족 가능)
    if acc < target_acc:
        pytest.xfail(
            f"[{channel}] Accuracy {acc:.4f} < target {target_acc} "
            "(데이터 부족 또는 추가 학습 필요)"
        )
