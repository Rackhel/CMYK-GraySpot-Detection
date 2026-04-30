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
    python -m src.scripts.run_baseline
    python -m src.scripts.run_baseline --channel C
    python -m src.scripts.run_baseline --channel all
"""

import copy
import json
import os
import random
import argparse
import warnings
import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# CMYK_MAIN 루트와 src/ 를 sys.path에 추가
# Add CMYK_MAIN root and src/ to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from src.config import get_config
from src.utils import setup_logging, get_logger, log_training_config
from models.grayspot_model import GrayspotModel
from training.trainer import CMYKDataset, Phase2Trainer

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

CHANNELS = ["Y", "M", "C", "K"]


def load_config() -> object:
    return get_config(config_path=SRC_DIR / "config" / "config.yaml", root_dir=ROOT_DIR)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dataloader(dataset, cfg: dict, shuffle: bool = False) -> DataLoader:
    num_workers = min(int(cfg["train"].get("num_workers", 0)), os.cpu_count() or 1)
    persistent_workers = bool(cfg["train"].get("persistent_workers", False) and num_workers > 0)
    batch_size = min(cfg["phase2"]["batch_size"], max(len(dataset), 1))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=cfg["train"].get("drop_last", False) if shuffle else False,
        num_workers=num_workers,
        pin_memory=bool(cfg["train"].get("pin_memory", False)),
        persistent_workers=persistent_workers,
        prefetch_factor=cfg["train"].get("prefetch_factor", 2) if num_workers > 0 else 2,
    )


def run_baseline(cfg: dict, channel: str, device: torch.device) -> dict:
    """
    단일 채널 Naive Baseline 학습 실행.
    Runs Naive Baseline training for a single channel.

    Returns:
        result dict: {channel, test_acc, mae, best_val_acc, n_train, n_test}
    """
    baseline_dir = Path(cfg["storage"]["data_root"]) / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    bcfg = copy.deepcopy(cfg)
    bcfg["storage"]["models_dir"] = str(baseline_dir)
    bcfg["storage"]["reports_dir"] = str(baseline_dir)

    train_ds = CMYKDataset(bcfg, channel, split="train", augment=True, oversample=True)
    val_ds = CMYKDataset(bcfg, channel, split="val", augment=False, oversample=False)
    test_ds = CMYKDataset(bcfg, channel, split="test", augment=False, oversample=False)

    logger.info("=" * 60)
    logger.info(f"  Baseline Training — Channel: [{channel}]")
    logger.info("  Mode: Supervised-only (Phase 2, no Phase 0)")
    logger.info(f"  Backbone: {cfg['model']['backbone']}")
    logger.info(f"  Epochs: {cfg['phase2']['epochs']} | LR: {cfg['phase2']['learning_rate']}")
    logger.info("=" * 60)
    logger.info(f"  [{channel}] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    if len(train_ds) == 0:
        logger.warning(f"  [WARN] 학습 데이터 없음 — 건너뜀 / No training data — skipping [{channel}]")
        return {
            "channel": channel,
            "skipped": True,
            "test_acc": 0.0,
            "mae": 0.0,
            "best_val_acc": 0.0,
            "n_train": 0,
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "epochs": cfg["phase2"]["epochs"],
            "backbone": cfg["model"]["backbone"],
            "pass_acc": False,
            "pass_mae": False,
        }

    train_loader = create_dataloader(train_ds, bcfg, shuffle=True)
    val_loader = create_dataloader(val_ds, bcfg, shuffle=False)
    test_loader = create_dataloader(test_ds, bcfg, shuffle=False)

    model = GrayspotModel(bcfg, phase=2).to(device)
    trainer = Phase2Trainer(model, bcfg, channel, device, train_ds)
    history = trainer.train(train_loader, val_loader)
    trainer.save_history(history)

    best_path = baseline_dir / f"best_{channel}.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

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
    best_val_acc = max(r["val_acc"] for r in history)

    logger.info(f"[{channel}] 테스트셋 결과 / Test Set Results")
    logger.info(f"  {'─' * 40}")
    logger.info(
        f"  Test Accuracy : {test_acc:.4f}  (target >= {cfg['evaluation']['targets']['per_color_accuracy']}) "
        f"{'[PASS]' if test_acc >= cfg['evaluation']['targets']['per_color_accuracy'] else '[FAIL]'}"
    )
    logger.info(
        f"  MAE           : {mae:.4f}  (target <= {cfg['evaluation']['targets']['mae']}) "
        f"{'[PASS]' if mae <= cfg['evaluation']['targets']['mae'] else '[FAIL]'}"
    )
    logger.info(f"  Best Val Acc  : {best_val_acc:.4f}")
    logger.info(f"  Test Samples  : {total}개")

    return {
        "channel": channel,
        "skipped": False,
        "test_acc": round(test_acc, 4),
        "mae": round(mae, 4),
        "best_val_acc": round(best_val_acc, 4),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "epochs": cfg["phase2"]["epochs"],
        "backbone": cfg["model"]["backbone"],
        "pass_acc": test_acc >= cfg["evaluation"]["targets"]["per_color_accuracy"],
        "pass_mae": mae <= cfg["evaluation"]["targets"]["mae"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Naive Baseline 학습 / Naive Baseline Training"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="all",
        help="학습할 채널 / Channel to train (Y/M/C/K/all, default: all)",
    )
    args = parser.parse_args()

    target_channels = CHANNELS if args.channel == "all" else [args.channel.upper()]

    config = load_config()
    if not config.validate():
        logger.error("Configuration validation failed. Fix config.yaml and retry.")
        raise SystemExit(1)

    config.create_necessary_directories()
    setup_logging(
        log_dir=Path(config.get("storage.logs_dir")),
        level=config.get("logging.level") or "INFO",
        format_style=config.get("logging.format") or "detailed",
        console=config.get("logging.console_output"),
        file=config.get("logging.file_output"),
    )
    log_training_config(config.config, logger=logger)

    logger.info("=" * 60)
    logger.info("  Grayspot — Naive Baseline (Supervised-only)")
    logger.info(f"  Channels: {target_channels}")
    logger.info("=" * 60)

    set_seed(config.get("train.seed") or 42)

    device = torch.device(config.get("system.device"))
    logger.info(f"  Device: {device}")

    results = []
    for ch in target_channels:
        results.append(run_baseline(config.config, ch, device))

    logger.info("=" * 60)
    logger.info("  Baseline 성능 요약 / Baseline Performance Summary")
    logger.info("=" * 60)
    logger.info(f"  {'Channel':<10} {'Test Acc':<12} {'MAE':<10} {'Val Acc':<10} Acc Pass")
    logger.info(f"  {'─' * 50}")

    for r in results:
        if r.get("skipped"):
            logger.info(f"  {r['channel']:<10} {'SKIPPED':<12}")
            continue
        acc_mark = "[PASS]" if r["pass_acc"] else "[FAIL]"
        logger.info(
            f"  {r['channel']:<10} {r['test_acc']:<12.4f} {r['mae']:<10.4f} "
            f"{r['best_val_acc']:<10.4f} {acc_mark}"
        )

    baseline_dir = Path(config.get("storage.data_root")) / "baseline"
    summary_path = baseline_dir / "baseline_summary.json"
    summary = {
        "mode": "Naive Baseline (Supervised-only, no Phase 0)",
        "backbone": config.get("model.backbone"),
        "epochs": config.get("phase2.epochs"),
        "results": results,
    }
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Summary saved / 요약 저장: {summary_path}")
    logger.info(f"  Baseline outputs / 산출물: {baseline_dir}")
    logger.info("")


if __name__ == "__main__":
    main()
