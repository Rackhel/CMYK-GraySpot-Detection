"""
scripts/run_baseline.py

Naive Baseline: Supervised-only (Phase 2만) 학습 → 성능 기록
Naive Baseline: Supervised-only (Phase 2 only) training → performance logging

Phase 0 없이 pretrained weights 로 바로 Phase 2 학습을 실행한다.
Runs Phase 2 training directly from pretrained weights without Phase 0.

목적 / Purpose:
    Swing Architecture 도입 효과를 비교하기 위한 기준선 성능 측정.
    Measures baseline performance to compare with Swing Architecture improvements.

출력 / Outputs:
    data_set/baseline/
    ├── best_{channel}.pt              ← 채널별 최적 모델 / Best model per channel
    ├── baseline_history_{channel}.csv ← 학습 이력 / Training history
    └── baseline_summary.json         ← 전체 채널 성능 요약 / Performance summary

실행 / Run:
    python src/scripts/run_baseline.py
    python src/scripts/run_baseline.py --channel C
    python src/scripts/run_baseline.py --channel all
"""

import sys
import json
import yaml
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# CMYK_MAIN 루트와 src/ 를 sys.path에 추가
# Add CMYK_MAIN root and src/ to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # CMYK_MAIN/
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from models.grayspot_model import GrayspotModel
from training.trainer      import CMYKDataset, Phase2Trainer

CHANNELS = ["Y", "M", "C", "K"]


def load_config() -> dict:
    config_path = SRC_DIR / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_baseline(cfg: dict, channel: str, device: torch.device) -> dict:
    """
    단일 채널 Naive Baseline 학습 실행.
    Runs Naive Baseline training for a single channel.

    Returns:
        result dict: {channel, test_acc, mae, best_val_acc, n_train, n_test}
    """
    # baseline 저장 폴더 / Baseline save folder
    baseline_dir = ROOT_DIR / "data_set" / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # config 에서 모델 저장 경로를 baseline/ 으로 임시 변경
    # Temporarily redirect model save path to baseline/
    import copy
    bcfg = copy.deepcopy(cfg)
    bcfg["storage"]["models_dir"]  = str(baseline_dir)
    bcfg["storage"]["reports_dir"] = str(baseline_dir)

    print(f"\n{'='*60}")
    print(f"  Baseline Training — Channel: [{channel}]")
    print(f"  Mode: Supervised-only (Phase 2, no Phase 0)")
    print(f"  Backbone: {cfg['model']['backbone']}")
    print(f"  Epochs: {cfg['phase2']['epochs']} | LR: {cfg['phase2']['learning_rate']}")
    print(f"{'='*60}")

    # 데이터셋 구성 / Build datasets
    train_ds = CMYKDataset(bcfg, channel, split="train", augment=True,  oversample=True)
    val_ds   = CMYKDataset(bcfg, channel, split="val",   augment=False, oversample=False)
    test_ds  = CMYKDataset(bcfg, channel, split="test",  augment=False, oversample=False)

    print(f"  [{channel}] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    if len(train_ds) == 0:
        print(f"  [WARN] 학습 데이터 없음 — 건너뜀 / No training data — skipping [{channel}]")
        return {"channel": channel, "skipped": True}

    train_loader = DataLoader(
        train_ds,
        batch_size=min(bcfg["phase2"]["batch_size"], len(train_ds)),
        shuffle=True,
        drop_last=True,
        num_workers=bcfg["train"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(bcfg["phase2"]["batch_size"], max(len(val_ds), 1)),
        shuffle=False,
        num_workers=bcfg["train"]["num_workers"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=min(bcfg["phase2"]["batch_size"], max(len(test_ds), 1)),
        shuffle=False,
        num_workers=bcfg["train"]["num_workers"],
    )

    # Phase 0 없이 pretrained weights 로 바로 Phase 2 시작
    # Start Phase 2 directly from pretrained weights without Phase 0
    model   = GrayspotModel(bcfg, phase=2).to(device)
    trainer = Phase2Trainer(model, bcfg, channel, device, train_ds)
    history = trainer.train(train_loader, val_loader)
    trainer.save_history(history)

    # 최적 모델 로드 후 테스트셋 평가 / Load best model and evaluate on test set
    best_path = baseline_dir / f"best_{channel}.pt"
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    correct, total = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            preds     = model(x).argmax(1)
            correct  += (preds == labels).sum().item()
            total    += len(labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    test_acc     = correct / max(total, 1)
    mae          = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    best_val_acc = max(r["val_acc"] for r in history)

    # 성능 목표 대비 출력 / Print vs performance targets
    target_acc = cfg["evaluation"]["targets"]["per_color_accuracy"]
    target_mae = cfg["evaluation"]["targets"]["mae"]

    print(f"\n  [{channel}] 테스트셋 결과 / Test Set Results")
    print(f"  {'─'*40}")
    print(f"  Test Accuracy : {test_acc:.4f}  (target >= {target_acc}) "
          f"{'[PASS]' if test_acc >= target_acc else '[FAIL]'}")
    print(f"  MAE           : {mae:.4f}  (target <= {target_mae}) "
          f"{'[PASS]' if mae <= target_mae else '[FAIL]'}")
    print(f"  Best Val Acc  : {best_val_acc:.4f}")
    print(f"  Test Samples  : {total}개")

    return {
        "channel":      channel,
        "skipped":      False,
        "test_acc":     round(test_acc, 4),
        "mae":          round(mae, 4),
        "best_val_acc": round(best_val_acc, 4),
        "n_train":      len(train_ds),
        "n_val":        len(val_ds),
        "n_test":       len(test_ds),
        "epochs":       cfg["phase2"]["epochs"],
        "backbone":     cfg["model"]["backbone"],
        "pass_acc":     test_acc >= target_acc,
        "pass_mae":     mae <= target_mae,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Naive Baseline 학습 / Naive Baseline Training"
    )
    parser.add_argument("--channel", type=str, default="all",
                        help="학습할 채널 / Channel to train (Y/M/C/K/all, default: all)")
    args = parser.parse_args()

    target_channels = CHANNELS if args.channel == "all" else [args.channel.upper()]

    print("=" * 60)
    print("  Grayspot — Naive Baseline (Supervised-only)")
    print(f"  Channels: {target_channels}")
    print("=" * 60)

    cfg = load_config()
    set_seed(cfg["train"]["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"  Device: {device}")

    # 채널별 학습 실행 / Run training per channel
    results = []
    for ch in target_channels:
        result = run_baseline(cfg, ch, device)
        results.append(result)

    # 전체 요약 출력 / Print overall summary
    print(f"\n{'='*60}")
    print("  Baseline 성능 요약 / Baseline Performance Summary")
    print(f"{'='*60}")
    print(f"  {'Channel':<10} {'Test Acc':<12} {'MAE':<10} {'Val Acc':<10} Acc Pass")
    print(f"  {'─'*50}")

    for r in results:
        if r.get("skipped"):
            print(f"  {r['channel']:<10} {'SKIPPED':<12}")
            continue
        acc_mark = "[PASS]" if r["pass_acc"] else "[FAIL]"
        print(f"  {r['channel']:<10} {r['test_acc']:<12.4f} {r['mae']:<10.4f} "
              f"{r['best_val_acc']:<10.4f} {acc_mark}")

    # 요약 JSON 저장 / Save summary JSON
    baseline_dir  = ROOT_DIR / "data_set" / "baseline"
    summary_path  = baseline_dir / "baseline_summary.json"
    summary       = {
        "mode":     "Naive Baseline (Supervised-only, no Phase 0)",
        "backbone": cfg["model"]["backbone"],
        "epochs":   cfg["phase2"]["epochs"],
        "results":  results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  Summary saved / 요약 저장: {summary_path}")
    print(f"  Baseline outputs / 산출물: {baseline_dir}")
    print()


if __name__ == "__main__":
    main()