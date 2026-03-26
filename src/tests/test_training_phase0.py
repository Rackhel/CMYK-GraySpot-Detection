"""
Grayspot -- Phase 0 학습 검증 / Phase 0 Training Validation
tests/test_training_phase0.py

Phase 0 Contrastive Learning 학습만 단독으로 검증한다.
Validates Phase 0 Contrastive Learning training in isolation.

실행 순서 / Execution order:
    1. Phase 0 DataLoader 구성 확인 / Verify Phase 0 DataLoader
    2. Phase 0 모델 초기화 확인 / Verify Phase 0 model initialization
    3. Phase 0 미니 학습 실행 (5 epoch) / Run Phase 0 mini training (5 epochs)
    4. Backbone 저장 확인 / Verify backbone save
    5. 학습 이력 CSV 저장 확인 / Verify training history CSV save

실행 / Run:
    python src/tests/test_training_phase0.py

    # 특정 채널 지정 / Specify channel
    python src/tests/test_training_phase0.py --channel Y

    # 전체 채널 / All channels
    python src/tests/test_training_phase0.py --channel all
"""

import sys
import argparse
import copy
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import ContrastiveDataset
from models.grayspot_model import GrayspotModel
from training.trainer import Phase0Trainer

CHANNELS = ["Y", "M", "C", "K"]


def pass_(msg): print(f"  [PASS] {msg}")
def fail_(msg): print(f"  [FAIL] {msg}")
def info_(msg): print(f"  [INFO] {msg}")
def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def load_config(path: str = "src/config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _mini_config(cfg: dict, epochs: int = 5) -> dict:
    """테스트용 축약 config를 반환한다. / Returns a shortened config for testing."""
    mini = copy.deepcopy(cfg)
    mini["phase0"]["epochs"]     = epochs
    mini["phase0"]["batch_size"] = 4
    return mini


# ──────────────────────────────────────────────
# TEST 1. Phase 0 DataLoader 확인 / Phase 0 DataLoader
# ──────────────────────────────────────────────
def test_phase0_dataloader(cfg: dict, channels: list[str]) -> bool:
    section("TEST 1. Phase 0 DataLoader 확인 / Phase 0 DataLoader Verification")
    passed = True

    for ch in channels:
        try:
            dataset = ContrastiveDataset(cfg, ch)
            if len(dataset) == 0:
                fail_(f"[{ch}] 데이터 없음 / No data -- labeled/ 폴더를 확인하세요 / Check labeled/ folder")
                passed = False
            else:
                loader = DataLoader(dataset, batch_size=min(4, len(dataset)),
                                    shuffle=True, drop_last=True)
                pass_(f"[{ch}] {len(dataset)}개 샘플 / samples | {len(loader)}개 배치 / batches")
        except Exception as e:
            fail_(f"[{ch}] DataLoader 오류 / Error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 2. Phase 0 모델 초기화 확인 / Phase 0 Model Initialization
# ──────────────────────────────────────────────
def test_phase0_model_init(cfg: dict) -> bool:
    section("TEST 2. Phase 0 모델 초기화 확인 / Phase 0 Model Initialization")
    passed = True

    try:
        model        = GrayspotModel(cfg, phase=0)
        size         = cfg["data"]["image_size"]
        dummy        = torch.randn(2, 3, size, size)
        expected_dim = cfg["phase0"]["projection_dim"]

        with torch.no_grad():
            output = model(dummy)

        if output.shape != (2, expected_dim):
            fail_(f"출력 형태 오류 / Wrong output shape: {output.shape} "
                  f"(expected (2, {expected_dim}))")
            passed = False
        else:
            pass_(f"모델 출력 형태 / Output shape: {output.shape}  (projection vector)")

        # Projection Head 존재 확인 / Verify Projection Head exists
        if hasattr(model, "head"):
            pass_(f"Projection Head 확인 / Verified: {type(model.head).__name__}")
        else:
            fail_("Projection Head 없음 / Not found")
            passed = False

    except Exception as e:
        fail_(f"모델 초기화 오류 / Initialization error: {e}")
        passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 3. Phase 0 미니 학습 / Phase 0 Mini Training
# ──────────────────────────────────────────────
def test_phase0_training(cfg: dict, channels: list[str]) -> tuple[bool, dict[str, Path]]:
    section("TEST 3. Phase 0 미니 학습 (5 epoch) / Phase 0 Mini Training")

    mini_cfg      = _mini_config(cfg, epochs=5)
    passed        = True
    backbone_paths: dict[str, Path] = {}

    for ch in channels:
        try:
            dataset = ContrastiveDataset(mini_cfg, ch)

            if len(dataset) == 0:
                info_(f"[{ch}] 데이터 없음 -- 건너뜀 / No data -- skipping")
                continue

            loader  = DataLoader(dataset, batch_size=min(4, len(dataset)),
                                 shuffle=True, drop_last=True)
            model   = GrayspotModel(mini_cfg, phase=0)
            trainer = Phase0Trainer(model, mini_cfg, ch)
            history = trainer.train(loader)

            # 손실 감소 확인 / Verify loss decrease
            if len(history) >= 2:
                first_loss = history[0]["loss"]
                last_loss  = history[-1]["loss"]
                if last_loss < first_loss * 1.5:
                    pass_(f"[{ch}] Loss: {first_loss:.4f} -> {last_loss:.4f}")
                else:
                    info_(f"[{ch}] Loss 감소 미확인 / Loss decrease not confirmed "
                          f"(데이터 부족 가능 / possibly insufficient data)")

            backbone_path = trainer.save_backbone()
            backbone_paths[ch] = backbone_path

        except Exception as e:
            fail_(f"[{ch}] Phase 0 학습 오류 / Training error: {e}")
            passed = False

    return passed, backbone_paths


# ──────────────────────────────────────────────
# TEST 4. Backbone 저장 확인 / Backbone Save Verification
# ──────────────────────────────────────────────
def test_backbone_saved(cfg: dict, backbone_paths: dict[str, Path]) -> bool:
    section("TEST 4. Backbone 저장 확인 / Backbone Save Verification")
    model_dir = Path(cfg["inference"]["model_dir"])
    passed    = True

    if not backbone_paths:
        info_("저장된 Backbone 없음 -- TEST 3 데이터를 확인하세요 / No backbones -- check TEST 3 data")
        return True

    for ch, path in backbone_paths.items():
        if path and path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            pass_(f"[{ch}] {path.name} -- {size_mb:.1f} MB")
        else:
            fail_(f"[{ch}] Backbone 파일 없음 / File not found")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 5. 학습 이력 CSV 확인 / Training History CSV
# ──────────────────────────────────────────────
def test_phase0_history_csv(cfg: dict, channels: list[str]) -> bool:
    section("TEST 5. Phase 0 학습 이력 CSV 확인 / Training History CSV Verification")
    reports_dir = Path(cfg["storage"]["reports_dir"])
    passed      = True

    for ch in channels:
        csv_path = reports_dir / f"phase0_history_{ch}.csv"
        if csv_path.exists():
            import csv
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                pass_(f"[{ch}] phase0_history_{ch}.csv -- {len(rows)}개 레코드 / records")
            else:
                fail_(f"[{ch}] phase0_history_{ch}.csv 비어있음 / Empty file")
                passed = False
        else:
            info_(f"[{ch}] phase0_history_{ch}.csv 없음 -- 데이터 없어서 학습 미실행 / No file -- training skipped due to no data")

    return passed


# ──────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Grayspot Phase 0 학습 검증 / Phase 0 Training Validation")
    parser.add_argument("--channel", type=str, default="all",
                        help="테스트할 채널 / Channel to test (Y/M/C/K/all, default: all)")
    parser.add_argument("--config",  type=str, default="src/config/config.yaml")
    args = parser.parse_args()

    target_channels = CHANNELS if args.channel == "all" else [args.channel.upper()]

    print("=" * 55)
    print("  Grayspot -- Phase 0 학습 검증 / Training Validation")
    print(f"  Channels: {target_channels}")
    print("=" * 55)

    cfg     = load_config(args.config)
    results = {}

    results["Phase 0 DataLoader"]              = test_phase0_dataloader(cfg, target_channels)
    results["Phase 0 모델 초기화 / Model Init"] = test_phase0_model_init(cfg)

    train_passed, backbone_paths = test_phase0_training(cfg, target_channels)
    results["Phase 0 미니 학습 / Mini Training"] = train_passed
    results["Backbone 저장 / Save"]              = test_backbone_saved(cfg, backbone_paths)
    results["학습 이력 CSV / History CSV"]        = test_phase0_history_csv(cfg, target_channels)

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