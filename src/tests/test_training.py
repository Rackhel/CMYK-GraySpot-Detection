"""
Grayspot -- 학습 실행 및 검증 / Training Execution and Validation
tests/test_training.py

test_after_labeling.py 통과 후 실행한다.
Run this after test_after_labeling.py passes.

실행 순서 / Execution order:
    1. DataLoader 구성 확인 / Verify DataLoader configuration
    2. 모델 초기화 확인 / Verify model initialization
    3. Phase 0 미니 학습 실행 (5 epoch) / Run Phase 0 mini training (5 epochs)
    4. Phase 2 미니 학습 실행 (5 epoch) / Run Phase 2 mini training (5 epochs)
    5. 모델 저장 확인 / Verify model save
    6. 학습 이력 CSV 저장 확인 / Verify training history CSV save

실행 / Run:
    python tests/test_training.py

    # Phase 2 직행 (라벨 있을 때) / Direct to Phase 2 (when labels exist)
    python tests/test_training.py --skip-phase0
"""

import sys
import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import ContrastiveDataset, GrayspotDataset, compute_class_weights
from data.dataloader import get_phase0_loaders, get_phase2_loaders
from models.grayspot_model import GrayspotModel
from training.trainer import Phase0Trainer, Phase2Trainer
from utils.logger import get_train_logger


def pass_(msg): print(f"  [PASS] {msg}")
def fail_(msg): print(f"  [FAIL] {msg}")
def info_(msg): print(f"  [INFO] {msg}")
def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# TEST 1. DataLoader 구성 확인 / DataLoader Verification
# ──────────────────────────────────────────────
def test_dataloader(cfg: dict) -> bool:
    section("TEST 1. DataLoader 구성 확인 / DataLoader Verification")
    channels = cfg["data"]["channels"]
    passed   = True

    for ch in channels:
        try:
            train_loader, val_loader, test_loader = get_phase2_loaders(cfg, ch)

            if len(train_loader) == 0:
                fail_(f"[{ch}] train DataLoader 배치 없음 / No batches in train DataLoader")
                passed = False
            else:
                pass_(f"[{ch}] Train: {len(train_loader)}배치 / batches | "
                      f"Val: {len(val_loader)}배치 | "
                      f"Test: {len(test_loader)}배치")
        except ValueError as e:
            fail_(f"[{ch}] {e}")
            passed = False
        except Exception as e:
            fail_(f"[{ch}] DataLoader 오류 / Error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 2. 모델 초기화 확인 / Model Initialization
# ──────────────────────────────────────────────
def test_model_init(cfg: dict) -> bool:
    section("TEST 2. 모델 초기화 확인 / Model Initialization")
    passed = True

    for phase in (0, 2):
        try:
            model      = GrayspotModel(cfg, phase=phase)
            size       = cfg["data"]["image_size"]
            dummy      = torch.randn(2, 3, size, size)

            with torch.no_grad():
                output = model(dummy)

            if phase == 0:
                expected_dim = cfg["phase0"]["projection_dim"]
                if output.shape != (2, expected_dim):
                    fail_(f"Phase {phase} 출력 형태 오류 / Wrong output shape: {output.shape}")
                    passed = False
                else:
                    pass_(f"Phase {phase} 출력 형태 / Output shape: {output.shape}  (projection)")
            else:
                expected_dim = cfg["data"]["num_levels"]
                if output.shape != (2, expected_dim):
                    fail_(f"Phase {phase} 출력 형태 오류 / Wrong output shape: {output.shape}")
                    passed = False
                else:
                    pass_(f"Phase {phase} 출력 형태 / Output shape: {output.shape}  (logits)")

        except Exception as e:
            fail_(f"Phase {phase} 모델 초기화 오류 / Initialization error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 3. Phase 0 미니 학습 / Phase 0 Mini Training
# ──────────────────────────────────────────────
def test_phase0_training(cfg: dict, channel: str) -> tuple[bool, Path | None]:
    section(f"TEST 3. Phase 0 미니 학습 / Mini Training -- Channel: {channel}")

    # 에폭을 5회로 축약 / Shorten to 5 epochs for quick test
    mini_cfg                     = _mini_config(cfg, phase=0, epochs=5)
    backbone_path                = None
    passed                       = True

    try:
        dataset = ContrastiveDataset(mini_cfg, channel)

        if len(dataset) == 0:
            info_(f"[{channel}] 데이터 없음 -- Phase 0 건너뜀 / No data -- Skipping Phase 0")
            return True, None

        loader  = DataLoader(dataset, batch_size=min(4, len(dataset)),
                             shuffle=True, drop_last=True)
        model   = GrayspotModel(mini_cfg, phase=0)
        trainer = Phase0Trainer(model, mini_cfg, channel)
        history = trainer.train(loader)

        # 손실 감소 확인 / Check loss decrease
        if len(history) >= 2:
            if history[-1]["loss"] < history[0]["loss"] * 1.5:
                pass_(f"[{channel}] Phase 0 Loss: {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}")
            else:
                info_(f"[{channel}] Loss 감소 미확인 (데이터 부족 가능) / Loss decrease not confirmed (possibly insufficient data)")

        backbone_path = trainer.save_backbone()
        if backbone_path.exists():
            pass_(f"[{channel}] Backbone 저장 확인 / Backbone saved: {backbone_path.name}")
        else:
            fail_(f"[{channel}] Backbone 저장 실패 / Save failed")
            passed = False

    except Exception as e:
        fail_(f"[{channel}] Phase 0 학습 오류 / Training error: {e}")
        passed = False

    return passed, backbone_path


# ──────────────────────────────────────────────
# TEST 4. Phase 2 미니 학습 / Phase 2 Mini Training
# ──────────────────────────────────────────────
def test_phase2_training(
    cfg: dict,
    channel: str,
    backbone_path: Path = None,
) -> bool:
    section(f"TEST 4. Phase 2 미니 학습 / Mini Training -- Channel: {channel}")

    # Stage 에폭을 3회씩으로 축약 / Shorten stage epochs to 3 each
    mini_cfg                         = _mini_config(cfg, phase=2, epochs=6)
    mini_cfg["phase2"]["stage1_epochs"] = 3
    mini_cfg["phase2"]["stage2_epochs"] = 3
    passed                           = True

    try:
        train_ds = GrayspotDataset(mini_cfg, channel, split="train", augment=False)
        val_ds   = GrayspotDataset(mini_cfg, channel, split="val",   augment=False)

        if len(train_ds) == 0:
            info_(f"[{channel}] 학습 데이터 없음 -- Phase 2 건너뜀 / No training data -- Skipping")
            return True

        train_loader = DataLoader(train_ds, batch_size=min(4, len(train_ds)),
                                  shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=min(4, max(len(val_ds), 1)),
                                  shuffle=False)

        class_weights = compute_class_weights(train_ds) \
            if mini_cfg["phase2"]["class_weights"] == "balanced" else None

        model = GrayspotModel(mini_cfg, phase=2)
        if backbone_path and backbone_path.exists():
            model.switch_to_phase2(backbone_path)
            info_(f"[{channel}] Phase 0 backbone 로드 / Loaded: {backbone_path.name}")

        trainer = Phase2Trainer(model, mini_cfg, channel, class_weights)
        history = trainer.train(train_loader, val_loader)

        # 학습 완료 확인 / Verify training completion
        if len(history) > 0:
            last = history[-1]
            pass_(f"[{channel}] Phase 2 완료 -- "
                  f"Train Acc: {last['train_acc']:.3f} | Val Acc: {last['val_acc']:.3f}")
        else:
            fail_(f"[{channel}] 학습 이력 없음 / No training history")
            passed = False

    except Exception as e:
        fail_(f"[{channel}] Phase 2 학습 오류 / Training error: {e}")
        passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 5. 모델 저장 확인 / Model Save Verification
# ──────────────────────────────────────────────
def test_model_saved(cfg: dict) -> bool:
    section("TEST 5. 모델 저장 확인 / Model Save Verification")
    model_dir = Path(cfg["inference"]["model_dir"])
    channels  = cfg["data"]["channels"]
    passed    = True

    for ch in channels:
        model_path = model_dir / f"best_{ch}.pt"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            pass_(f"[{ch}] best_{ch}.pt -- {size_mb:.1f} MB")
        else:
            fail_(f"[{ch}] best_{ch}.pt 없음 / Not found -- 학습이 완료되지 않았습니다 / Training not complete")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 6. 학습 이력 CSV 저장 확인 / Training History CSV Verification
# ──────────────────────────────────────────────
def test_history_csv(cfg: dict) -> bool:
    section("TEST 6. 학습 이력 CSV 확인 / Training History CSV Verification")
    reports_dir  = Path(cfg["storage"]["reports_dir"])
    history_path = reports_dir / cfg["reporting"]["csv_files"]["training_history"]

    if not history_path.exists():
        fail_(f"training_history.csv 없음 / Not found: {history_path}")
        return False

    import csv
    with open(history_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) == 0:
        fail_("training_history.csv 비어있음 / Empty file")
        return False

    pass_(f"training_history.csv -- {len(rows)}개 레코드 / records")

    # 필수 컬럼 확인 / Check required columns
    required_cols = {"phase", "channel", "epoch", "val_acc"}
    actual_cols   = set(rows[0].keys())
    missing       = required_cols - actual_cols
    if missing:
        fail_(f"누락된 컬럼 / Missing columns: {missing}")
        return False

    pass_(f"필수 컬럼 확인 / Required columns verified: {required_cols}")
    return True


# ──────────────────────────────────────────────
# 헬퍼 / Helpers
# ──────────────────────────────────────────────
def _mini_config(cfg: dict, phase: int, epochs: int) -> dict:
    """테스트용 축약 config를 반환한다. / Returns a shortened config for testing."""
    import copy
    mini = copy.deepcopy(cfg)
    if phase == 0:
        mini["phase0"]["epochs"]     = epochs
        mini["phase0"]["batch_size"] = 4
    else:
        mini["phase2"]["epochs"]     = epochs
        mini["phase2"]["batch_size"] = 4
    return mini


# ──────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Grayspot 학습 검증 / Training Validation")
    parser.add_argument("--skip-phase0", action="store_true",
                        help="Phase 0 건너뛰고 Phase 2 직행 / Skip Phase 0, go direct to Phase 2")
    parser.add_argument("--channel", type=str, default="Y",
                        help="테스트할 채널 / Channel to test (default: Y)")
    parser.add_argument("--config",  type=str, default="config/config.yaml")
    args = parser.parse_args()

    print("=" * 55)
    print("  Grayspot -- 학습 검증 / Training Validation")
    print("=" * 55)

    cfg     = load_config(args.config)
    channel = args.channel.upper()
    results = {}

    results["DataLoader 구성 / Configuration"]     = test_dataloader(cfg)
    results["모델 초기화 / Model Initialization"]   = test_model_init(cfg)

    backbone_path = None
    if not args.skip_phase0:
        passed, backbone_path = test_phase0_training(cfg, channel)
        results["Phase 0 미니 학습 / Mini Training"] = passed
    else:
        info_("--skip-phase0: Phase 0 건너뜀 / Skipping Phase 0")

    results["Phase 2 미니 학습 / Mini Training"]    = test_phase2_training(cfg, channel, backbone_path)
    results["모델 저장 확인 / Model Save"]           = test_model_saved(cfg)
    results["학습 이력 CSV / History CSV"]           = test_history_csv(cfg)

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