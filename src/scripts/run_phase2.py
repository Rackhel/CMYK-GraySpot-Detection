"""
scripts/run_phase2.py

Phase 2 Supervised Classification 학습 스크립트 (Swing Cycle 1).
Phase 2 Supervised Classification training script (Swing Cycle 1).

Phase 0 backbone weights를 로드한 뒤 ClassifierHead를 부착하여
정제 라벨(labels_v1) 기반 Supervised 학습을 실행한다.
Loads Phase 0 backbone weights, attaches ClassifierHead,
and runs Supervised training on refined labels (labels_v1).

학습 흐름 / Training flow:
    Phase 0 backbone (.pt) → switch_to_phase2() → ClassifierHead 학습
    Phase 0 backbone (.pt) → switch_to_phase2() → Train ClassifierHead

주의 / Note:
    - Phase 0 checkpoint(phase0_backbone_{channel}.pt)가 반드시 존재해야 함
    - Phase 0 checkpoint (phase0_backbone_{channel}.pt) must exist
    - 라벨 폴더(data_set/labeled/{channel}/{level}/)의 데이터를 직접 로드함
    - Loads data directly from label folder (data_set/labeled/{channel}/{level}/)
    - labels_v1.csv가 없으면 현재 폴더 상태(v0)로 학습 — Phase 1 완료 후 재실행 권장
    - If labels_v1.csv is absent, trains on current folder state (v0) — re-run after Phase 1

실행 / Run:
    # 전체 채널 / All channels
    python src/scripts/run_phase2.py

    # 단일 채널 / Single channel
    python src/scripts/run_phase2.py --channel C

    # Phase 0 checkpoint 경로 지정 / Specify Phase 0 checkpoint directory
    python src/scripts/run_phase2.py --phase0-dir data_set/models

    # 브라우저로 리포트 열기 / Open report in browser
    python src/scripts/run_phase2.py --open-browser

출력 / Outputs:
    outputs/checkpoints/
    ├── phase2_{channel}_v1.pt          ← 채널별 최적 모델 / Best model per channel
    └── phase2_history_{channel}.csv    ← 학습 이력 / Training history

    outputs/reports/
    └── phase2_v1.html                  ← Phase 2 평가 리포트 / Evaluation report
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── 경로 설정 / Path setup ─────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent   # CMYK_MAIN/
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

# ── utils shim (팀 코드 호환) / utils shim for team code compatibility ──────────
# evaluator.py 등이 `from utils import ...`를 사용하므로
# top-level utils 모듈을 등록한다.
# Register top-level utils so that evaluator.py etc.
# can resolve `from utils import ...` without modification.
import importlib, types as _types
_logger_mod = importlib.import_module("src.utils.logger")
_utils_shim = _types.ModuleType("utils")
_utils_shim.LoggerMixin        = _logger_mod.LoggerMixin
_utils_shim.get_logger         = _logger_mod.get_logger
_utils_shim.setup_logging      = _logger_mod.setup_logging
_utils_shim.log_training_config = _logger_mod.log_training_config
_utils_shim.log_epoch_summary  = _logger_mod.log_epoch_summary
sys.modules["utils"] = _utils_shim

from src.utils  import setup_logging, get_logger, log_training_config, log_snapshot, set_seed, load_config, backbone_tag, validate_config, create_directories, get_nested
from models.grayspot_model  import GrayspotModel
from data.dataset       import CMYKDataset
from training.trainer   import Phase2Trainer

warnings.filterwarnings("ignore")

# 전역 상수 / Global constants
CHANNELS     = ["Y", "M", "C", "K"]
CYCLE_TAG    = "v1"            # Swing Cycle 1 식별자 / Swing Cycle 1 identifier
CKPT_SUBDIR  = "outputs/checkpoints"


def _make_dataloader(dataset: CMYKDataset, cfg: dict, shuffle: bool) -> DataLoader:
    """
    DataLoader를 생성한다.
    Creates a DataLoader.

    Args:
        dataset: CMYKDataset 인스턴스 / CMYKDataset instance
        cfg:     config dict
        shuffle: 셔플 여부 — 학습 시 True, 검증/테스트 시 False
                 Whether to shuffle — True for training, False for val/test
    """
    num_workers        = min(int(cfg["train"].get("num_workers", 0)), os.cpu_count() or 1)
    persistent_workers = bool(cfg["train"].get("persistent_workers", False) and num_workers > 0)
    batch_size         = min(cfg["phase2"]["batch_size"], max(len(dataset), 1))

    return DataLoader(
        dataset,
        batch_size        = batch_size,
        shuffle           = shuffle,
        drop_last         = cfg["train"].get("drop_last", False) if shuffle else False,
        num_workers       = num_workers,
        pin_memory        = bool(cfg["train"].get("pin_memory", False)),
        persistent_workers= persistent_workers,
        prefetch_factor   = cfg["train"].get("prefetch_factor", 2) if num_workers > 0 else 2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 핵심 실행 / Phase 2 Core Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_phase2(
    cfg:          dict,
    channel:      str,
    device:       torch.device,
    phase0_dir:   Path,
    ckpt_dir:     Path,
) -> dict:
    """
    단일 채널 Phase 2 학습을 실행한다.
    Runs Phase 2 training for a single channel.

    Phase 0 backbone을 로드하여 ClassifierHead로 교체한 뒤,
    CMYKDataset 폴더 구조 기반 Supervised 학습을 수행한다.
    Loads Phase 0 backbone, replaces with ClassifierHead,
    then runs Supervised training on folder-structured CMYKDataset.

    Args:
        cfg:        config.json dict
        channel:    "Y" | "M" | "C" | "K"
        device:     torch.device
        phase0_dir: phase0_backbone_{channel}.pt 가 저장된 디렉토리
                    Directory containing phase0_backbone_{channel}.pt
        ckpt_dir:   Phase 2 checkpoint 저장 디렉토리
                    Directory to save Phase 2 checkpoints

    Returns:
        result dict:
            channel, test_acc, mae, best_val_acc,
            n_train, n_val, n_test, epochs, backbone,
            pass_acc, pass_mae, checkpoint_path
    """
    logger = get_logger(__name__)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # config에서 모델 저장 경로를 ckpt_dir로 오버라이드 / Override model save path in config
    p2cfg                        = copy.deepcopy(cfg)
    p2cfg["storage"]["models_dir"]  = str(ckpt_dir)
    p2cfg["storage"]["reports_dir"] = str(ckpt_dir)

    # ── 데이터셋 구성 / Dataset construction ──────────────────────────────────
    train_ds = CMYKDataset(p2cfg, channel, split="train", augment=True,  oversample=True)
    val_ds   = CMYKDataset(p2cfg, channel, split="val",   augment=False, oversample=False)
    test_ds  = CMYKDataset(p2cfg, channel, split="test",  augment=False, oversample=False)

    logger.info("=" * 65)
    logger.info(f"  Phase 2 Training (Swing Cycle 1) — Channel: [{channel}]")
    logger.info(f"  Mode   : Supervised + Phase 0 Backbone")
    logger.info(f"  Backbone: {cfg['model']['backbone']}")
    logger.info(f"  Epochs : {cfg['phase2']['epochs']}  |  LR: {cfg['phase2']['learning_rate']}")
    logger.info("=" * 65)
    logger.info(f"  [{channel}] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # 학습 데이터 없을 때 조기 반환 / Early return when no training data
    if len(train_ds) == 0:
        logger.warning(f"  [WARN] 학습 데이터 없음 — 건너뜀 / No training data — skipping [{channel}]")
        return {
            "channel": channel, "skipped": True,
            "test_acc": 0.0, "mae": 0.0, "best_val_acc": 0.0,
            "n_train": 0, "n_val": len(val_ds), "n_test": len(test_ds),
            "epochs": cfg["phase2"]["epochs"], "backbone": cfg["model"]["backbone"],
            "pass_acc": False, "pass_mae": False, "checkpoint_path": "",
        }

    # ── Phase 0 backbone 로드 → Phase 2로 전환 / Load Phase 0 backbone → switch to Phase 2 ──
    tag           = backbone_tag(p2cfg["model"]["backbone"])
    backbone_path = phase0_dir / f"phase0_backbone_{channel}_{tag}.pt"

    # Phase 0 모드로 모델 초기화 후 switch_to_phase2() 호출
    # Initialize model in Phase 0 mode, then call switch_to_phase2()
    model = GrayspotModel(p2cfg, phase=0)
    model.switch_to_phase2(backbone_path=backbone_path, cfg=p2cfg)
    model = model.to(device)

    # ── DataLoader 생성 / Create DataLoaders ──────────────────────────────────
    train_loader = _make_dataloader(train_ds, p2cfg, shuffle=True)
    val_loader   = _make_dataloader(val_ds,   p2cfg, shuffle=False)
    test_loader  = _make_dataloader(test_ds,  p2cfg, shuffle=False)

    # ── Phase 2 학습 루프 / Phase 2 training loop ─────────────────────────────
    trainer = Phase2Trainer(model, p2cfg, channel, device, train_ds)
    history = trainer.train(train_loader, val_loader)

    # 학습 이력 CSV 저장 / Save training history CSV
    history_path = ckpt_dir / f"phase2_history_{channel}.csv"
    _save_history_csv(history, history_path)
    logger.info(f"  History saved / 이력 저장: {history_path}")

    # ── Best checkpoint 로드 후 Test셋 평가 / Load best checkpoint → evaluate on test set ──
    best_path = ckpt_dir / f"best_{channel}.pt"

    # Trainer가 best_{channel}.pt에 저장하므로 phase2_{channel}_{tag}_{CYCLE_TAG}.pt로 복사
    # Trainer saves to best_{channel}.pt — copy to phase2_{channel}_{tag}_{CYCLE_TAG}.pt for cycle versioning
    versioned_path = ckpt_dir / f"phase2_{channel}_{tag}_{CYCLE_TAG}.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        torch.save(torch.load(best_path, map_location="cpu"), versioned_path)
        logger.info(f"  Checkpoint saved / 체크포인트 저장: {versioned_path}")

    model.eval()

    correct, total = 0, 0
    y_true, y_pred  = [], []

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

    # 평가 목표 달성 여부 / Check evaluation target pass/fail
    target_acc = cfg["evaluation"]["targets"]["per_color_accuracy"]
    target_mae = cfg["evaluation"]["targets"]["mae"]

    logger.info(f"  [{channel}] 테스트셋 결과 / Test Set Results")
    logger.info(f"  {'─' * 50}")
    logger.info(
        f"  Test Accuracy : {test_acc:.4f}  (target >= {target_acc}) "
        f"{'[PASS]' if test_acc >= target_acc else '[FAIL]'}"
    )
    logger.info(
        f"  MAE           : {mae:.4f}  (target <= {target_mae}) "
        f"{'[PASS]' if mae <= target_mae else '[FAIL]'}"
    )
    logger.info(f"  Best Val Acc  : {best_val_acc:.4f}")
    logger.info(f"  Test Samples  : {total}개")

    return {
        "channel":         channel,
        "skipped":         False,
        "test_acc":        round(test_acc, 4),
        "mae":             round(mae, 4),
        "best_val_acc":    round(best_val_acc, 4),
        "n_train":         len(train_ds),
        "n_val":           len(val_ds),
        "n_test":          len(test_ds),
        "epochs":          cfg["phase2"]["epochs"],
        "backbone":        cfg["model"]["backbone"],
        "pass_acc":        test_acc >= target_acc,
        "pass_mae":        mae <= target_mae,
        "checkpoint_path": str(versioned_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 저장 헬퍼 / Save Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_history_csv(history: list[dict], path: Path) -> None:
    """
    학습 이력을 CSV로 저장한다.
    Saves training history to CSV.
    """
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def _save_summary(results: list[dict], ckpt_dir: Path, cfg: dict) -> Path:
    """
    전체 채널 학습 결과를 JSON으로 저장한다.
    Saves full-channel training results to JSON.

    출력 / Output:
        outputs/checkpoints/phase2_summary_v1.json
    """
    summary = {
        "mode":        f"Phase 2 Supervised (Swing Cycle 1, Phase 0 backbone)",
        "cycle_tag":   CYCLE_TAG,
        "backbone":    cfg["model"]["backbone"],
        "epochs":      cfg["phase2"]["epochs"],
        "results":     results,
    }
    summary_path = ckpt_dir / f"phase2_summary_{CYCLE_TAG}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_path


# ──────────────────────────────────────────────────────────────────────────────
# 진입점 / Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 Supervised 학습 (Swing Cycle 1) / Phase 2 Supervised Training (Swing Cycle 1)"
    )
    parser.add_argument(
        "--channel", type=str, default="all",
        help="학습할 채널 / Channel to train  (Y / M / C / K / all,  default: all)",
    )
    parser.add_argument(
        "--phase0-dir", type=str, default=None,
        help=(
            "Phase 0 backbone .pt 파일이 있는 디렉토리 / "
            "Directory containing phase0_backbone_*.pt  "
            "(default: data_set/models)"
        ),
    )
    parser.add_argument(
        "--open-browser", action="store_true",
        help="완료 후 리포트를 브라우저로 열기 / Open report in browser after completion",
    )
    args = parser.parse_args()

    # 채널 결정 / Resolve target channels
    target_channels = CHANNELS if args.channel.lower() == "all" else [args.channel.upper()]

    # config 로드 / Load config
    cfg = load_config()
    if not validate_config(cfg):
        print("[ERROR] Configuration validation failed — fix config.json and retry.")
        sys.exit(1)
    create_directories(cfg)

    # 로깅 설정 / Setup logging
    setup_logging(
        log_dir      = Path(cfg["storage"]["logs_dir"]),
        level        = get_nested(cfg, "logging.level") or "INFO",
        format_style = get_nested(cfg, "logging.format") or "detailed",
        console      = get_nested(cfg, "logging.console_output"),
        file         = get_nested(cfg, "logging.file_output"),
    )
    logger = get_logger(__name__)
    log_training_config(cfg, logger=logger)

    # 디바이스 및 경로 결정 / Resolve device and paths
    device    = torch.device(cfg["system"]["device"])
    ckpt_dir  = ROOT_DIR / CKPT_SUBDIR
    phase0_dir = (
        Path(args.phase0_dir) if args.phase0_dir
        else ROOT_DIR / cfg["storage"]["models_dir"]
    )

    set_seed(cfg["train"].get("seed") or 42, cfg)

    # ── 스냅샷 저장 / Save config snapshot ───────────────────────────────────
    log_snapshot(
        config       = cfg,
        snapshot_dir = ROOT_DIR / "outputs" / "snapshots",
        tag          = "phase2",
        logger       = logger,
    )

    # ── 시작 배너 / Start banner ───────────────────────────────────────────────
    import time
    from datetime import datetime as _dt
    print()
    print("=" * 65)
    print("  Grayspot — Phase 2 Supervised Training (Swing Cycle 1)")
    print(f"  {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"  Channels     : {target_channels}")
    print(f"  Device       : {device}")
    print(f"  Backbone     : {cfg['model']['backbone']}")
    print(f"  Phase 0 dir  : {phase0_dir}")
    print(f"  Checkpoint   : {ckpt_dir}")
    print("=" * 65)
    print()

    # Phase 0 체크포인트 존재 여부 사전 확인 / Pre-check Phase 0 checkpoints
    _tag = backbone_tag(cfg["model"]["backbone"])
    missing = [
        ch for ch in target_channels
        if not (phase0_dir / f"phase0_backbone_{ch}_{_tag}.pt").exists()
    ]
    if missing:
        logger.error(
            f"[SSOT-PH01] Phase 0 backbone 없음 / Phase 0 backbone not found: {missing}\n"
            f"            Phase 0 완료 후 실행 / Run Phase 0 first: python -m src.scripts.run_phase0\n"
            f"            경로 확인 / Check path: {phase0_dir}"
        )
        sys.exit(1)

    # ── 채널별 Phase 2 학습 / Per-channel Phase 2 training ────────────────────
    t_start = time.time()
    results = []

    for ch in target_channels:
        result = run_phase2(
            cfg        = cfg,
            channel    = ch,
            device     = device,
            phase0_dir = phase0_dir,
            ckpt_dir   = ckpt_dir,
        )
        results.append(result)

    # ── 요약 출력 / Print summary ──────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  Phase 2 성능 요약 / Phase 2 Performance Summary")
    print("=" * 65)
    print(f"  {'Channel':<10} {'Test Acc':<12} {'MAE':<10} {'Val Acc':<10} Acc Pass")
    print(f"  {'─' * 55}")

    for r in results:
        if r.get("skipped"):
            print(f"  {r['channel']:<10} SKIPPED")
            continue
        mark = "[PASS]" if r["pass_acc"] else "[FAIL]"
        print(
            f"  {r['channel']:<10} {r['test_acc']:<12.4f} "
            f"{r['mae']:<10.4f} {r['best_val_acc']:<10.4f} {mark}"
        )

    # ── 요약 JSON 저장 / Save summary JSON ────────────────────────────────────
    summary_path = _save_summary(results, ckpt_dir, cfg)
    elapsed      = time.time() - t_start

    print()
    print("=" * 65)
    print("  Pipeline Complete")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 65)
    print(f"  Checkpoints → {ckpt_dir}/phase2_{{channel}}_{CYCLE_TAG}.pt")
    print(f"  Summary     → {summary_path}")
    print("=" * 65)
    print()

    # ── (선택) 브라우저 열기 / (Optional) Open browser ────────────────────────
    if args.open_browser:
        report_path = ROOT_DIR / "outputs" / "reports" / f"phase2_{CYCLE_TAG}.html"
        if report_path.exists():
            import webbrowser
            webbrowser.open(report_path.resolve().as_uri())
        else:
            logger.info(
                f"  리포트 파일 없음 — R3 evaluate.py 실행 후 열기 가능 / "
                f"Report not found — run R3 evaluate.py first: {report_path}"
            )


if __name__ == "__main__":
    main()
