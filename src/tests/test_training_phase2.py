"""
Grayspot -- Phase 2 학습 검증 / Phase 2 Training Validation
tests/test_training_phase2.py

Phase 2 Supervised Classification 학습만 단독으로 검증한다.
Validates Phase 2 Supervised Classification training in isolation.

Phase 0 backbone이 있으면 로드하고, 없으면 pretrained weights로 시작한다.
Loads Phase 0 backbone if available, otherwise starts from pretrained weights.

실행 순서 / Execution order:
    1. Phase 2 DataLoader 구성 확인 / Verify Phase 2 DataLoader
    2. Phase 2 모델 초기화 확인 / Verify Phase 2 model initialization
    3. Phase 2 미니 학습 실행 (6 epoch) / Run Phase 2 mini training (6 epochs)
    4. 모델 저장 확인 / Verify model save
    5. 학습 이력 CSV 저장 확인 / Verify training history CSV save

실행 / Run:
    python src/tests/test_training_phase2.py

    # 특정 채널 지정 / Specify channel
    python src/tests/test_training_phase2.py --channel Y

    # 전체 채널 / All channels
    python src/tests/test_training_phase2.py --channel all

    # Phase 0 backbone 없이 시작 / Start without Phase 0 backbone
    python src/tests/test_training_phase2.py --skip-phase0
"""

import sys
import argparse
import copy
import csv
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import GrayspotDataset, compute_class_weights
from models.grayspot_model import GrayspotModel
from training.trainer import Phase2Trainer

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


def _mini_config(cfg: dict, epochs: int = 6) -> dict:
    """테스트용 축약 config를 반환한다. / Returns a shortened config for testing."""
    mini = copy.deepcopy(cfg)
    mini["phase2"]["epochs"]        = epochs
    mini["phase2"]["batch_size"]    = 4
    mini["phase2"]["stage1_epochs"] = 3
    mini["phase2"]["stage2_epochs"] = 3
    return mini


def _find_backbone(cfg: dict, channel: str) -> Path | None:
    """
    Phase 0 학습된 Backbone 경로를 반환한다.
    없으면 None을 반환한다.
    Returns Phase 0 backbone path, or None if not found.
    """
    model_dir     = Path(cfg["inference"]["model_dir"])
    backbone_path = model_dir / f"phase0_backbone_{channel}.pt"
    return backbone_path if backbone_path.exists() else None


# ──────────────────────────────────────────────
# TEST 1. Phase 2 DataLoader 확인 / Phase 2 DataLoader
# ──────────────────────────────────────────────
def test_phase2_dataloader(cfg: dict, channels: list[str]) -> bool:
    section("TEST 1. Phase 2 DataLoader 확인 / Phase 2 DataLoader Verification")
    passed = True

    for ch in channels:
        try:
            train_ds = GrayspotDataset(cfg, ch, split="train", augment=False)
            val_ds   = GrayspotDataset(cfg, ch, split="val",   augment=False)
            test_ds  = GrayspotDataset(cfg, ch, split="test",  augment=False)
            total    = len(train_ds) + len(val_ds) + len(test_ds)

            if len(train_ds) == 0:
                fail_(f"[{ch}] 학습 데이터 없음 / No training data -- "
                      f"labeled/ 폴더를 확인하세요 / Check labeled/ folder")
                passed = False
            else:
                pass_(f"[{ch}] Train: {len(train_ds)} | Val: {len(val_ds)} | "
                      f"Test: {len(test_ds)} | Total: {total}")

        except Exception as e:
            fail_(f"[{ch}] DataLoader 오류 / Error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 2. Phase 2 모델 초기화 확인 / Phase 2 Model Initialization
# ──────────────────────────────────────────────
def test_phase2_model_init(cfg: dict, channels: list[str], skip_phase0: bool) -> bool:
    section("TEST 2. Phase 2 모델 초기화 확인 / Phase 2 Model Initialization")
    passed = True

    for ch in channels:
        try:
            model        = GrayspotModel(cfg, phase=2)
            size         = cfg["data"]["image_size"]
            dummy        = torch.randn(2, 3, size, size)
            expected_dim = cfg["data"]["num_levels"]

            # Phase 0 backbone 로드 시도 / Try loading Phase 0 backbone
            backbone_path = _find_backbone(cfg, ch)
            if backbone_path and not skip_phase0:
                model.switch_to_phase2(backbone_path)
                info_(f"[{ch}] Phase 0 backbone 로드 / Loaded: {backbone_path.name}")
            else:
                info_(f"[{ch}] Pretrained weights로 시작 / Starting from pretrained weights")

            with torch.no_grad():
                output = model(dummy)

            if output.shape != (2, expected_dim):
                fail_(f"[{ch}] 출력 형태 오류 / Wrong output shape: {output.shape} "
                      f"(expected (2, {expected_dim}))")
                passed = False
            else:
                pass_(f"[{ch}] 출력 형태 / Output shape: {output.shape}  (6-class logits)")

            # Classifier Head 존재 확인 / Verify Classifier Head exists
            if hasattr(model, "head"):
                pass_(f"[{ch}] Classifier Head 확인 / Verified: {type(model.head).__name__}")

        except Exception as e:
            fail_(f"[{ch}] 모델 초기화 오류 / Initialization error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 3. Phase 2 미니 학습 / Phase 2 Mini Training
# ──────────────────────────────────────────────
def test_phase2_training(
    cfg: dict,
    channels: list[str],
    skip_phase0: bool,
) -> bool:
    section("TEST 3. Phase 2 미니 학습 (6 epoch) / Phase 2 Mini Training")

    mini_cfg = _mini_config(cfg, epochs=6)
    passed   = True

    for ch in channels:
        try:
            train_ds = GrayspotDataset(mini_cfg, ch, split="train", augment=False)
            val_ds   = GrayspotDataset(mini_cfg, ch, split="val",   augment=False)

            if len(train_ds) == 0:
                info_(f"[{ch}] 학습 데이터 없음 -- 건너뜀 / No training data -- skipping")
                continue

            train_loader = DataLoader(
                train_ds,
                batch_size=min(4, len(train_ds)),
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=min(4, max(len(val_ds), 1)),
                shuffle=False,
            )

            # 클래스 가중치 / Class weights
            class_weights = compute_class_weights(train_ds) \
                if mini_cfg["phase2"]["class_weights"] == "balanced" else None

            # 모델 초기화 / Initialize model
            model         = GrayspotModel(mini_cfg, phase=2)
            backbone_path = _find_backbone(cfg, ch)

            if backbone_path and not skip_phase0:
                model.switch_to_phase2(backbone_path)
                info_(f"[{ch}] Phase 0 backbone 로드 / Loaded: {backbone_path.name}")
            else:
                info_(f"[{ch}] Pretrained weights로 시작 / Starting from pretrained weights (Option A)")

            trainer = Phase2Trainer(model, mini_cfg, ch, class_weights)
            history = trainer.train(train_loader, val_loader)

            # 학습 완료 확인 / Verify training completion
            if len(history) > 0:
                last = history[-1]
                pass_(f"[{ch}] Phase 2 완료 / Done -- "
                      f"Train Acc: {last['train_acc']:.3f} | "
                      f"Val Acc: {last['val_acc']:.3f}")
            else:
                fail_(f"[{ch}] 학습 이력 없음 / No training history")
                passed = False

        except Exception as e:
            fail_(f"[{ch}] Phase 2 학습 오류 / Training error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 4. 모델 저장 확인 / Model Save Verification
# ──────────────────────────────────────────────
def test_model_saved(cfg: dict, channels: list[str]) -> bool:
    section("TEST 4. 모델 저장 확인 / Model Save Verification")
    model_dir = Path(cfg["inference"]["model_dir"])
    passed    = True

    for ch in channels:
        model_path = model_dir / f"best_{ch}.pt"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            pass_(f"[{ch}] best_{ch}.pt -- {size_mb:.1f} MB")
        else:
            fail_(f"[{ch}] best_{ch}.pt 없음 / Not found -- "
                  f"데이터가 있는지 확인하세요 / Check if data exists")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 5. 학습 이력 CSV 확인 / Training History CSV
# ──────────────────────────────────────────────
def test_phase2_history_csv(cfg: dict) -> bool:
    section("TEST 5. Phase 2 학습 이력 CSV 확인 / Training History CSV Verification")
    reports_dir  = Path(cfg["storage"]["reports_dir"])
    history_path = reports_dir / cfg["reporting"]["csv_files"]["training_history"]

    if not history_path.exists():
        fail_(f"training_history.csv 없음 / Not found: {history_path}")
        return False

    with open(history_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Phase 2 레코드만 필터 / Filter Phase 2 records only
    phase2_rows = [r for r in rows if r.get("phase") == "2"]

    if not phase2_rows:
        fail_("Phase 2 학습 이력 없음 / No Phase 2 records in training_history.csv")
        return False

    pass_(f"training_history.csv -- Phase 2 레코드 / records: {len(phase2_rows)}개")

    # 필수 컬럼 확인 / Check required columns
    required_cols = {"phase", "channel", "epoch", "stage", "train_acc", "val_acc"}
    actual_cols   = set(phase2_rows[0].keys())
    missing       = required_cols - actual_cols
    if missing:
        fail_(f"누락된 컬럼 / Missing columns: {missing}")
        return False

    pass_(f"필수 컬럼 확인 / Required columns verified: {required_cols}")
    return True


# ──────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Grayspot Phase 2 학습 검증 / Phase 2 Training Validation"
    )
    parser.add_argument("--channel",     type=str, default="all",
                        help="테스트할 채널 / Channel to test (Y/M/C/K/all, default: all)")
    parser.add_argument("--skip-phase0", action="store_true",
                        help="Phase 0 backbone 없이 Pretrained weights로 시작 / "
                             "Start from pretrained weights without Phase 0 backbone")
    parser.add_argument("--config",      type=str, default="src/config/config.yaml")
    args = parser.parse_args()

    target_channels = CHANNELS if args.channel == "all" else [args.channel.upper()]

    print("=" * 55)
    print("  Grayspot -- Phase 2 학습 검증 / Training Validation")
    print(f"  Channels: {target_channels}")
    if args.skip_phase0:
        print("  Mode: Pretrained weights (Phase 0 backbone 미사용 / not used)")
    else:
        print("  Mode: Phase 0 backbone 자동 탐색 / Auto-detect Phase 0 backbone")
    print("=" * 55)

    cfg     = load_config(args.config)
    results = {}

    results["Phase 2 DataLoader"]               = test_phase2_dataloader(cfg, target_channels)
    results["Phase 2 모델 초기화 / Model Init"]  = test_phase2_model_init(cfg, target_channels, args.skip_phase0)
    results["Phase 2 미니 학습 / Mini Training"] = test_phase2_training(cfg, target_channels, args.skip_phase0)
    results["모델 저장 확인 / Model Save"]        = test_model_saved(cfg, target_channels)
    results["학습 이력 CSV / History CSV"]        = test_phase2_history_csv(cfg)

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
        print("  All tests passed. Proceed to test_evaluation.py")
    else:
        print("  Some tests failed. Fix the issues above before proceeding.")
    print()


if __name__ == "__main__":
    main()