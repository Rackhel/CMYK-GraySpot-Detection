"""
Grayspot — 학습 실행 스크립트 / Training Execution Script
scripts/train.py

사용법 / Usage:
    # Phase 0 (Contrastive Learning)
    python scripts/train.py --phase 0 --channel Y

    # Phase 2 (Supervised) — 특정 채널 / Specific channel
    python scripts/train.py --phase 2 --channel Y

    # Phase 2 — 전체 채널 / All channels
    python scripts/train.py --phase 2 --channel all

    # Phase 2 — 이미 라벨링된 데이터가 있을 때 직행 / Direct to Phase 2 when labels already exist
    python scripts/train.py --phase 2 --channel all --skip-phase0
"""

import argparse
import yaml
import sys
from pathlib import Path

# 루트를 sys.path에 추가 / Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader

from data.dataset import ContrastiveDataset, GrayspotDataset, compute_class_weights
from models.grayspot_model import GrayspotModel
from training.trainer import Phase0Trainer, Phase2Trainer
from evaluation.metrics import evaluate_all_channels, print_evaluation_report, save_evaluation_results

CHANNELS = ["Y", "M", "C", "K"]


def load_config(path: str = "src/config/config.yaml") -> dict:
    """config.yaml을 로드한다. / Loads config.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def train_phase0(cfg: dict, channel: str) -> Path:
    """Phase 0 Contrastive Learning 학습을 실행한다. / Runs Phase 0 Contrastive Learning."""
    p       = cfg["phase0"]
    dataset = ContrastiveDataset(cfg, channel)
    loader  = DataLoader(dataset, batch_size=p["batch_size"], shuffle=True, drop_last=True)

    model   = GrayspotModel(cfg, phase=0)
    trainer = Phase0Trainer(model, cfg, channel)
    trainer.train(loader)
    return trainer.save_backbone()  # Backbone weights 저장 후 경로 반환 / Save and return backbone path


def train_phase2(cfg: dict, channel: str, backbone_path: Path = None) -> None:
    """Phase 2 Supervised Classification 학습을 실행한다. / Runs Phase 2 Supervised Classification."""
    p = cfg["phase2"]

    # 데이터셋 구성 / Build datasets
    train_ds = GrayspotDataset(cfg, channel, split="train", augment=True)
    val_ds   = GrayspotDataset(cfg, channel, split="val",   augment=False)

    if len(train_ds) == 0:
        print(f"    [{channel}] 학습 데이터 없음 / No training data. labeled/ 폴더를 확인하세요 / Check the labeled/ folder.")
        return

    train_loader = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=p["batch_size"], shuffle=False)

    # 클래스 불균형 보정 가중치 / Class imbalance correction weights
    class_weights = compute_class_weights(train_ds) if p["class_weights"] == "balanced" else None

    # 모델 생성 — Phase 0 backbone이 있으면 로드, 없으면 pretrained으로 시작
    # Build model — load Phase 0 backbone if available, otherwise start from pretrained weights
    model = GrayspotModel(cfg, phase=2)
    if backbone_path and backbone_path.exists():
        model.switch_to_phase2(backbone_path)
        print(f"    Phase 0 → Phase 2 전환 / Switching (backbone: {backbone_path})")
    else:
        print(f"     Phase 0 backbone 없음 → Pretrained weights로 시작 (Option A) / No Phase 0 backbone → starting from pretrained weights")

    trainer = Phase2Trainer(model, cfg, channel, class_weights)
    trainer.train(train_loader, val_loader)


def run_evaluation(cfg: dict) -> dict:
    """Phase 3 — 테스트셋 평가를 실행한다. / Runs Phase 3 test set evaluation."""
    import torch

    print("\n🔍  Phase 3 — 테스트셋 평가 시작 / Test set evaluation started")
    results = {}

    for ch in CHANNELS:
        test_ds    = GrayspotDataset(cfg, ch, split="test", augment=False)
        if len(test_ds) == 0:
            continue

        model_path = Path(cfg["inference"]["model_dir"]) / f"best_{ch}.pt"
        if not model_path.exists():
            continue

        # 저장된 best 모델 로드 / Load saved best model
        model = GrayspotModel(cfg, phase=2)
        model.load(model_path)
        model.eval()

        loader         = torch.utils.data.DataLoader(
            test_ds, batch_size=cfg["phase2"]["batch_size"], shuffle=False
        )
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, labels, _ in loader:
                logits = model(x)
                y_pred.extend(logits.argmax(1).tolist())  # 예측 레벨 / Predicted levels
                y_true.extend(labels.tolist())            # 실제 레벨 / True levels

        results[ch] = {"y_true": y_true, "y_pred": y_pred}

    if not results:
        print("    평가 가능한 채널 없음 / No channels available for evaluation")
        return {}

    eval_result = evaluate_all_channels(results, cfg)
    print_evaluation_report(eval_result)
    save_evaluation_results(eval_result, cfg)
    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Grayspot 학습 스크립트 / Training Script")
    parser.add_argument("--phase",       type=int, default=2, choices=[0, 2],
                        help="학습 Phase / Training phase (0: Contrastive, 2: Supervised)")
    parser.add_argument("--channel",     type=str, default="all",
                        help="학습 채널 / Training channel (Y/M/C/K/all)")
    parser.add_argument("--skip-phase0", action="store_true",
                        help="Phase 0 없이 Phase 2 직행 / Skip Phase 0 and go directly to Phase 2 (use when labels already exist)")
    parser.add_argument("--config",      type=str, default="src/config/config.yaml",
                        help="config.yaml 경로 / Path to config.yaml")
    args = parser.parse_args()

    cfg             = load_config(args.config)
    target_channels = CHANNELS if args.channel == "all" else [args.channel.upper()]

    print("=" * 55)
    print(f"  Grayspot Training — Phase {args.phase}")
    print(f"  Channels: {target_channels}")
    print("=" * 55)

    if args.phase == 0:
        # Phase 0: Contrastive Learning (라벨 불필요 / No labels required)
        for ch in target_channels:
            train_phase0(cfg, ch)

    elif args.phase == 2:
        if args.skip_phase0:
            # Option A: 라벨이 이미 있으므로 Phase 2 직행
            # Option A: Labels already exist, go directly to Phase 2
            print("\n     --skip-phase0: Phase 2 직행 모드 / Direct Phase 2 mode (Option A)")
            for ch in target_channels:
                train_phase2(cfg, ch, backbone_path=None)
        else:
            # Phase 0 → Phase 2 순차 실행 / Sequential Phase 0 → Phase 2
            for ch in target_channels:
                backbone_path = train_phase0(cfg, ch)
                train_phase2(cfg, ch, backbone_path)

        # Phase 3 평가 실행 / Run Phase 3 evaluation
        eval_result = run_evaluation(cfg)
        if eval_result:
            decision = eval_result.get("swing_decision", {})
            if any(v != "pass" for v in decision.values()):
                print("\n    일부 채널이 Swing 복귀 조건에 해당합니다 / Some channels require Swing feedback:")
                for ch, action in decision.items():
                    if action != "pass":
                        print(f"      [{ch}] → {action}")
            else:
                print("\n    모든 채널 목표 달성 / All channel targets met!")


if __name__ == "__main__":
    main()