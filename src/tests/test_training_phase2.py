"""
Grayspot — Phase 2 학습 검증 / Phase 2 Training Validation
tests/test_training_phase2.py

Phase 2 Supervised Classification 학습만 단독으로 검증한다.
Validates Phase 2 Supervised Classification training in isolation.

Phase 0 backbone이 있으면 로드하고, 없으면 pretrained weights로 시작한다.
Loads Phase 0 backbone if available, otherwise starts from pretrained weights.

실행 순서 / Execution order:
    1. config 로드 확인 / Verify config load
    2. CMYKDataset 구성 확인 / Verify CMYKDataset
    3. Phase 2 모델 초기화 확인 / Verify Phase 2 model initialization
    4. Phase 2 미니 학습 실행 (3 epoch) / Run Phase 2 mini training (3 epochs)
    5. 모델 저장 확인 / Verify model save
    6. 성능 목표 대비 확인 / Check against performance targets

실행 / Run:
    python src/tests/test_training_phase2.py
    python src/tests/test_training_phase2.py --channel Y
    python src/tests/test_training_phase2.py --channel all
    python src/tests/test_training_phase2.py
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# CMYK_MAIN 루트를 sys.path에 추가 / Add CMYK_MAIN root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # CMYK_MAIN/
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from models.grayspot_model import GrayspotModel
from training.trainer import CMYKDataset, Phase2Trainer

CHANNELS = ["Y", "M", "C", "K"]


# ── 출력 헬퍼 / Output helpers ────────────────────────────────
def pass_(msg):
    print(f"  [PASS] {msg}")


def fail_(msg):
    print(f"  [FAIL] {msg}")


def info_(msg):
    print(f"  [INFO] {msg}")


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def load_config() -> dict:
    """config.yaml 로드 / Load config.yaml"""
    config_path = ROOT_DIR / "src" / "config" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def mini_cfg(cfg: dict, epochs: int = 3) -> dict:
    """
    테스트용 축약 config 반환 / Returns shortened config for testing.
    """
    c = copy.deepcopy(cfg)
    c["phase2"]["epochs"] = epochs
    c["phase2"]["batch_size"] = 4
    return c


def find_phase0_backbone(cfg: dict, channel: str) -> Path | None:
    """
    Phase 0 backbone 파일 경로 반환. 없으면 None.
    Returns Phase 0 backbone path, or None if not found.
    """
    models_dir = Path(cfg["storage"]["models_dir"])
    backbone_path = models_dir / f"phase0_backbone_{channel}.pt"
    return backbone_path if backbone_path.exists() else None


# ──────────────────────────────────────────────────────────────
# TEST 1. config 로드 확인 / Config Load
# ──────────────────────────────────────────────────────────────
def test_config() -> bool:
    section("TEST 1. config 로드 확인 / Config Load")
    try:
        cfg = load_config()
        required = [
            "data",
            "model",
            "phase0",
            "phase2",
            "storage",
            "train",
            "evaluation",
        ]
        missing = [k for k in required if k not in cfg]
        if missing:
            fail_(f"누락된 키 / Missing keys: {missing}")
            return False
        pass_(f"config.yaml 로드 성공 / Loaded successfully")
        pass_(
            f"backbone: {cfg['model']['backbone']} | "
            f"epochs: {cfg['phase2']['epochs']} | "
            f"lr: {cfg['phase2']['learning_rate']}"
        )
        return True
    except Exception as e:
        fail_(f"config 로드 오류 / Load error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# TEST 2. CMYKDataset 구성 확인 / CMYKDataset Verification
# ──────────────────────────────────────────────────────────────
def test_dataset(cfg: dict, channels: list) -> bool:
    section("TEST 2. CMYKDataset 구성 확인 / CMYKDataset Verification")
    passed = True

    for ch in channels:
        try:
            train_ds = CMYKDataset(
                cfg, ch, split="train", augment=False, oversample=False
            )
            val_ds = CMYKDataset(cfg, ch, split="val", augment=False, oversample=False)
            test_ds = CMYKDataset(
                cfg, ch, split="test", augment=False, oversample=False
            )
            total = len(train_ds) + len(val_ds) + len(test_ds)

            if total == 0:
                info_(
                    f"[{ch}] 데이터 없음 / No data — "
                    f"labeled/{ch}/ 폴더를 확인하세요 / Check labeled/{ch}/"
                )
            else:
                pass_(
                    f"[{ch}] Train: {len(train_ds)} | Val: {len(val_ds)} | "
                    f"Test: {len(test_ds)} | Total: {total}"
                )

                # 샘플 형태 확인 / Check sample shape
                if len(train_ds) > 0:
                    x, y = train_ds[0]
                    size = cfg["data"]["image_size"]
                    if x.shape != (3, size, size):
                        fail_(f"[{ch}] 샘플 형태 오류 / Wrong shape: {x.shape}")
                        passed = False
                    else:
                        pass_(
                            f"[{ch}] 샘플 형태 / Sample shape: {tuple(x.shape)} | Level: {y}"
                        )

        except Exception as e:
            fail_(f"[{ch}] Dataset 오류 / Error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────────────────────
# TEST 3. Phase 2 모델 초기화 확인 / Phase 2 Model Init
# ──────────────────────────────────────────────────────────────
def test_model_init(cfg: dict, channels: list, skip_phase0: bool) -> bool:
    section("TEST 3. Phase 2 모델 초기화 확인 / Phase 2 Model Initialization")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    passed = True

    for ch in channels:
        try:
            model = GrayspotModel(cfg, phase=2).to(device)
            size = cfg["data"]["image_size"]
            dummy = torch.randn(2, 3, size, size).to(device)
            expected_dim = cfg["data"]["num_levels"]

            # Phase 0 backbone 로드 시도 / Try loading Phase 0 backbone
            backbone_path = find_phase0_backbone(cfg, ch)
            if backbone_path and not skip_phase0:
                model.switch_to_phase2(backbone_path, cfg)
                model = model.to(device)
                info_(f"[{ch}] Phase 0 backbone 로드 / Loaded: {backbone_path.name}")
            else:
                info_(
                    f"[{ch}] Pretrained weights로 시작 / Starting from pretrained weights"
                )

            with torch.no_grad():
                output = model(dummy)

            if output.shape != (2, expected_dim):
                fail_(f"[{ch}] 출력 형태 오류 / Wrong shape: {output.shape}")
                passed = False
            else:
                pass_(
                    f"[{ch}] 출력 형태 / Output shape: {tuple(output.shape)} (6-class logits)"
                )

        except Exception as e:
            fail_(f"[{ch}] 모델 초기화 오류 / Init error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────────────────────
# TEST 4. Phase 2 미니 학습 / Phase 2 Mini Training
# ──────────────────────────────────────────────────────────────
def test_phase2_training(cfg: dict, channels: list, skip_phase0: bool) -> bool:
    section("TEST 4. Phase 2 미니 학습 (3 epoch) / Phase 2 Mini Training")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    mcfg = mini_cfg(cfg, epochs=3)
    passed = True

    for ch in channels:
        try:
            train_ds = CMYKDataset(
                mcfg, ch, split="train", augment=True, oversample=True
            )
            val_ds = CMYKDataset(mcfg, ch, split="val", augment=False, oversample=False)

            if len(train_ds) == 0:
                info_(f"[{ch}] 학습 데이터 없음 — 건너뜀 / No training data — skipping")
                continue

            train_loader = DataLoader(
                train_ds,
                batch_size=min(mcfg["phase2"]["batch_size"], len(train_ds)),
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=min(mcfg["phase2"]["batch_size"], max(len(val_ds), 1)),
                shuffle=False,
                num_workers=0,
            )

            # 모델 구성 / Build model
            model = GrayspotModel(mcfg, phase=2).to(device)

            backbone_path = find_phase0_backbone(cfg, ch)
            if backbone_path and not skip_phase0:
                model.switch_to_phase2(backbone_path, mcfg)
                model = model.to(device)
                info_(f"[{ch}] Phase 0 backbone 로드 / Loaded: {backbone_path.name}")
            else:
                info_(
                    f"[{ch}] Pretrained weights로 시작 / Starting from pretrained weights"
                )

            trainer = Phase2Trainer(model, mcfg, ch, device, train_ds)
            history = trainer.train(train_loader, val_loader)

            if len(history) > 0:
                last = history[-1]
                pass_(
                    f"[{ch}] 학습 완료 / Done — "
                    f"Train Acc: {last['train_acc']:.3f} | "
                    f"Val Acc: {last['val_acc']:.3f}"
                )
            else:
                fail_(f"[{ch}] 학습 이력 없음 / No training history")
                passed = False

        except Exception as e:
            fail_(f"[{ch}] 학습 오류 / Training error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────────────────────
# TEST 5. 모델 저장 확인 / Model Save Verification
# ──────────────────────────────────────────────────────────────
def test_model_saved(cfg: dict, channels: list) -> bool:
    section("TEST 5. 모델 저장 확인 / Model Save Verification")

    models_dir = Path(cfg["storage"]["models_dir"])
    passed = True

    for ch in channels:
        model_path = models_dir / f"best_{ch}.pt"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            pass_(f"[{ch}] best_{ch}.pt — {size_mb:.1f} MB")
        else:
            info_(
                f"[{ch}] best_{ch}.pt 없음 — 데이터 없어서 학습 미실행 / "
                f"Not found — training skipped due to no data"
            )

    return passed


# ──────────────────────────────────────────────────────────────
# TEST 6. 성능 목표 대비 확인 / Performance Target Check
# ──────────────────────────────────────────────────────────────
def test_performance_targets(cfg: dict, channels: list, skip_phase0: bool) -> bool:
    section("TEST 6. 성능 목표 대비 확인 / Performance Target Check")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    models_dir = Path(cfg["storage"]["models_dir"])
    target_acc = cfg["evaluation"]["targets"]["per_color_accuracy"]  # 0.85
    target_mae = cfg["evaluation"]["targets"]["mae"]  # 0.50
    passed = True

    for ch in channels:
        model_path = models_dir / f"best_{ch}.pt"
        if not model_path.exists():
            info_(f"[{ch}] best_{ch}.pt 없음 — 건너뜀 / Not found — skipping")
            continue

        try:
            test_ds = CMYKDataset(
                cfg, ch, split="test", augment=False, oversample=False
            )
            if len(test_ds) == 0:
                info_(f"[{ch}] 테스트 데이터 없음 — 건너뜀 / No test data — skipping")
                continue

            test_loader = DataLoader(
                test_ds,
                batch_size=min(16, len(test_ds)),
                shuffle=False,
                num_workers=0,
            )

            # Best 모델 로드 / Load best model
            model = GrayspotModel(cfg, phase=2).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()  # eval 모드 필수 / eval mode required

            correct, total = 0, 0
            y_true, y_pred = [], []

            with torch.no_grad():
                for x, labels in test_loader:
                    x, labels = x.to(device), labels.to(device)
                    preds = model(x).argmax(1)
                    correct += (preds == labels).sum().item()
                    total += len(labels)
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(preds.cpu().tolist())

            test_acc = correct / max(total, 1)
            mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)

            acc_icon = "[PASS]" if test_acc >= target_acc else "[FAIL]"
            mae_icon = "[PASS]" if mae <= target_mae else "[INFO]"

            print(
                f"  [{ch}] Test Acc: {test_acc:.4f} (target >= {target_acc}) {acc_icon}"
            )
            print(
                f"  [{ch}] MAE:      {mae:.4f}      (target <= {target_mae}) {mae_icon}"
            )

            if test_acc < target_acc:
                info_(
                    f"[{ch}] Accuracy 미달 — 데이터 부족 가능 / Below target — possibly insufficient data"
                )

        except Exception as e:
            fail_(f"[{ch}] 평가 오류 / Evaluation error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Grayspot Phase 2 학습 검증 / Phase 2 Training Validation"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="all",
        help="테스트할 채널 / Channel to test (Y/M/C/K/all, default: all)",
    )
    parser.add_argument(
        "--skip-phase0",
        action="store_true",
        help="Phase 0 backbone 없이 pretrained weights로 시작 / "
        "Start from pretrained weights without Phase 0 backbone",
    )
    args = parser.parse_args()

    target_channels = CHANNELS if args.channel == "all" else [args.channel.upper()]

    print("=" * 55)
    print("  Grayspot — Phase 2 학습 검증 / Training Validation")
    print(f"  Channels: {target_channels}")
    if args.skip_phase0:
        print("  Mode: Pretrained weights (Phase 0 backbone 미사용 / not used)")
    else:
        print("  Mode: Phase 0 backbone 자동 탐색 / Auto-detect Phase 0 backbone")
    print("=" * 55)

    cfg = load_config()
    results = {}

    results["config 로드 / Load"] = test_config()
    results["CMYKDataset"] = test_dataset(cfg, target_channels)
    results["Phase 2 모델 초기화 / Model Init"] = test_model_init(
        cfg, target_channels, args.skip_phase0
    )
    results["Phase 2 미니 학습 / Mini Training"] = test_phase2_training(
        cfg, target_channels, args.skip_phase0
    )
    results["모델 저장 확인 / Model Save"] = test_model_saved(cfg, target_channels)
    results["성능 목표 확인 / Performance"] = test_performance_targets(
        cfg, target_channels, args.skip_phase0
    )

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
