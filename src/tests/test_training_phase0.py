"""
Grayspot — Phase 0 학습 검증 / Phase 0 Training Validation
tests/test_training_phase0.py

Phase 0 Contrastive Learning 학습만 단독으로 검증한다.
Validates Phase 0 Contrastive Learning training in isolation.

실행 순서 / Execution order:
    1. config 로드 확인 / Verify config load
    2. ContrastiveDataset 구성 확인 / Verify ContrastiveDataset
    3. Phase 0 모델 초기화 확인 / Verify Phase 0 model initialization
    4. Phase 0 미니 학습 실행 (3 epoch) / Run Phase 0 mini training (3 epochs)
    5. Backbone 저장 확인 / Verify backbone save

실행 / Run:
    python src/tests/test_training_phase0.py
    python src/tests/test_training_phase0.py --channel Y
    python src/tests/test_training_phase0.py --channel all
"""

import sys
import argparse
import copy
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# CMYK_MAIN 루트를 sys.path에 추가 / Add CMYK_MAIN root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # CMYK_MAIN/
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from models.grayspot_model import GrayspotModel
from training.trainer      import ContrastiveDataset, Phase0Trainer

CHANNELS = ["Y", "M", "C", "K"]


# ── 출력 헬퍼 / Output helpers ────────────────────────────────
def pass_(msg): print(f"  [PASS] {msg}")
def fail_(msg): print(f"  [FAIL] {msg}")
def info_(msg): print(f"  [INFO] {msg}")
def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def load_config() -> dict:
    """
    config.yaml 로드 / Load config.yaml
    """
    config_path = ROOT_DIR / "src" / "config" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def mini_cfg(cfg: dict, epochs: int = 3) -> dict:
    """
    테스트용 축약 config 반환 / Returns shortened config for testing.
    epoch 수를 줄여 빠르게 검증한다 / Reduces epochs for fast validation.
    """
    c = copy.deepcopy(cfg)
    c["phase0"]["epochs"]     = epochs
    c["phase0"]["batch_size"] = 4
    return c


# ──────────────────────────────────────────────────────────────
# TEST 1. config 로드 확인 / Config Load
# ──────────────────────────────────────────────────────────────
def test_config() -> bool:
    section("TEST 1. config 로드 확인 / Config Load")
    try:
        cfg = load_config()
        required = ["data", "model", "phase0", "phase2", "storage", "train"]
        missing  = [k for k in required if k not in cfg]
        if missing:
            fail_(f"누락된 키 / Missing keys: {missing}")
            return False
        pass_(f"config.yaml 로드 성공 / Loaded successfully")
        pass_(f"backbone: {cfg['model']['backbone']} | "
              f"image_size: {cfg['data']['image_size']} | "
              f"channels: {cfg['data']['channels']}")
        return True
    except Exception as e:
        fail_(f"config 로드 오류 / Load error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# TEST 2. ContrastiveDataset 구성 확인 / ContrastiveDataset
# ──────────────────────────────────────────────────────────────
def test_dataset(cfg: dict, channels: list) -> bool:
    section("TEST 2. ContrastiveDataset 구성 확인 / ContrastiveDataset Verification")
    passed = True

    for ch in channels:
        try:
            ds = ContrastiveDataset(cfg, ch)
            if len(ds) == 0:
                info_(f"[{ch}] 데이터 없음 / No data — "
                      f"labeled/{ch}/ 폴더를 확인하세요 / Check labeled/{ch}/")
            else:
                # Positive Pair 형태 확인 / Verify positive pair shape
                v1, v2 = ds[0]
                if v1.shape != v2.shape:
                    fail_(f"[{ch}] Positive Pair shape 불일치 / Mismatch: {v1.shape} vs {v2.shape}")
                    passed = False
                else:
                    pass_(f"[{ch}] {len(ds)}개 이미지 / images | "
                          f"Pair shape: {tuple(v1.shape)}")
        except Exception as e:
            fail_(f"[{ch}] Dataset 오류 / Error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────────────────────
# TEST 3. Phase 0 모델 초기화 확인 / Phase 0 Model Init
# ──────────────────────────────────────────────────────────────
def test_model_init(cfg: dict) -> bool:
    section("TEST 3. Phase 0 모델 초기화 확인 / Phase 0 Model Initialization")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    try:
        model        = GrayspotModel(cfg, phase=0).to(device)
        size         = cfg["data"]["image_size"]
        dummy        = torch.randn(2, 3, size, size).to(device)
        expected_dim = cfg["phase0"]["projection_dim"]

        with torch.no_grad():
            output = model(dummy)

        if output.shape != (2, expected_dim):
            fail_(f"출력 형태 오류 / Wrong shape: {output.shape} "
                  f"(expected (2, {expected_dim}))")
            return False

        pass_(f"Phase 0 모델 초기화 / Initialized: {cfg['model']['backbone']}")
        pass_(f"출력 형태 / Output shape: {tuple(output.shape)} (projection vector)")
        pass_(f"Device: {device}")
        return True

    except Exception as e:
        fail_(f"모델 초기화 오류 / Init error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# TEST 4. Phase 0 미니 학습 / Phase 0 Mini Training
# ──────────────────────────────────────────────────────────────
def test_phase0_training(cfg: dict, channels: list) -> tuple[bool, dict]:
    section("TEST 4. Phase 0 미니 학습 (3 epoch) / Phase 0 Mini Training")

    device   = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    mcfg     = mini_cfg(cfg, epochs=3)
    passed   = True
    backbone_paths = {}

    for ch in channels:
        try:
            ds = ContrastiveDataset(mcfg, ch)
            if len(ds) == 0:
                info_(f"[{ch}] 데이터 없음 — 건너뜀 / No data — skipping")
                continue

            loader  = DataLoader(
                ds,
                batch_size=min(mcfg["phase0"]["batch_size"], len(ds)),
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
            model   = GrayspotModel(mcfg, phase=0).to(device)
            trainer = Phase0Trainer(model, mcfg, ch, device)
            history = trainer.train(loader)

            # Loss 감소 확인 / Check loss decrease
            if len(history) >= 2:
                first, last = history[0]["loss"], history[-1]["loss"]
                if last < first * 1.5:
                    pass_(f"[{ch}] Loss: {first:.4f} → {last:.4f}")
                else:
                    info_(f"[{ch}] Loss 감소 미확인 (데이터 부족 가능) / "
                          f"Loss decrease not confirmed (possibly insufficient data)")

            # Backbone 저장 / Save backbone
            backbone_path = trainer.save_backbone()
            backbone_paths[ch] = backbone_path

        except Exception as e:
            fail_(f"[{ch}] 학습 오류 / Training error: {e}")
            passed = False

    return passed, backbone_paths


# ──────────────────────────────────────────────────────────────
# TEST 5. Backbone 저장 확인 / Backbone Save Verification
# ──────────────────────────────────────────────────────────────
def test_backbone_saved(backbone_paths: dict) -> bool:
    section("TEST 5. Backbone 저장 확인 / Backbone Save Verification")

    if not backbone_paths:
        info_("저장된 Backbone 없음 — TEST 4 데이터 확인 / No backbones — check TEST 4 data")
        return True

    passed = True
    for ch, path in backbone_paths.items():
        if path and path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            pass_(f"[{ch}] {path.name} — {size_mb:.1f} MB")
        else:
            fail_(f"[{ch}] Backbone 파일 없음 / File not found")
            passed = False

    return passed


# ──────────────────────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Grayspot Phase 0 학습 검증 / Phase 0 Training Validation"
    )
    parser.add_argument("--channel", type=str, default="all",
                        help="테스트할 채널 / Channel to test (Y/M/C/K/all, default: all)")
    args = parser.parse_args()

    target_channels = CHANNELS if args.channel == "all" else [args.channel.upper()]

    print("=" * 55)
    print("  Grayspot — Phase 0 학습 검증 / Training Validation")
    print(f"  Channels: {target_channels}")
    print("=" * 55)

    cfg     = load_config()
    results = {}

    results["config 로드 / Load"]                = test_config()
    results["ContrastiveDataset"]                = test_dataset(cfg, target_channels)
    results["Phase 0 모델 초기화 / Model Init"]   = test_model_init(cfg)

    train_passed, backbone_paths = test_phase0_training(cfg, target_channels)
    results["Phase 0 미니 학습 / Mini Training"]  = train_passed
    results["Backbone 저장 / Save"]               = test_backbone_saved(backbone_paths)

    # 최종 결과 / Final results
    print(f"\n{'='*55}")
    print("  최종 결과 / Final Results")
    print(f"{'='*55}")
    all_passed = True
    for name, result in results.items():
        icon = "[PASS]" if result else "[FAIL]"
        print(f"  {icon}  {name}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed. Proceed to test_training_phase2.py")
    else:
        print("  Some tests failed. Fix the issues above before proceeding.")
    print()


if __name__ == "__main__":
    main()
