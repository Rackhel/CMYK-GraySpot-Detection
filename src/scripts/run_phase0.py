"""
# scripts/run_phase0.py
# R2 W9~W10 산출물 / R2 W9~W10 Deliverable

## 개요 / Overview

Phase 0 SimCLR Contrastive Learning 실행 스크립트.
Phase 0 SimCLR Contrastive Learning execution script.

Swing Architecture의 첫 번째 단계로, 라벨 없이 전체 데이터로
SimCLR Contrastive Learning을 수행하여 Grayspot 패턴에 특화된 backbone을 사전 학습한다.

The first stage of the Swing Architecture. Performs SimCLR Contrastive Learning
on the full dataset without labels to pretrain a Grayspot-specialized backbone.

---

## 실행 방법 / How to Run

```bash
# 전체 채널 / All channels (default)
python src/scripts/run_phase0.py

# 특정 채널 지정 / Specific channel
python src/scripts/run_phase0.py --channel C
python src/scripts/run_phase0.py --channel all
```

---

## 산출물 / Outputs

| 경로 / Path | 설명 / Description |
|---|---|
| `outputs/checkpoints/phase0_v1.pt` | Execution Plan 공식 산출물 (전 채널 묶음) / Official deliverable |
| `data_set/models/phase0_backbone_{ch}.pt` | Phase 2에서 로드할 채널별 backbone / Per-channel backbone for Phase 2 |
| `outputs/checkpoints/phase0_history_{ch}.csv` | 채널별 epoch별 loss 기록 / Per-channel training history |
| `outputs/checkpoints/phase0_summary.json` | 전체 실행 요약 / Overall run summary |

---

## 전제 조건 / Prerequisites

- `data_set/labeled/{channel}/{level}/*.png` 이미지 존재 / Images must exist
- `src/config/config.json` 설정 완료 / Config must be set up

---

## 참조 / Reference

- Execution Plan: S3 W9~W10, R2
- PRD: Section 3 (Swing Architecture), Section 5.3 (Phase 0)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── 경로 설정 / Path setup ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # CMYK_MAIN/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ── Utils / Logger ────────────────────────────────────────────────────────────
from src.utils import (
    setup_logging,
    get_logger,
    log_snapshot,
    set_seed,
    backbone_tag,
    load_config,
    validate_config,
    create_directories,
    get_nested,
)

# ── 모델 / 학습 모듈 / Model & Training modules ──────────────────────────────
from models.grayspot_model import GrayspotModel
from data.dataset import ContrastiveDataset
from training.trainer import Phase0Trainer

CHANNELS = ["Y", "M", "C", "K"]


# ─────────────────────────────────────────────────────────────────────────────
# 채널별 Phase 0 실행 / Run Phase 0 per channel
# ─────────────────────────────────────────────────────────────────────────────


def run_phase0(cfg: dict, channel: str, device: torch.device) -> dict:
    """
    단일 채널에 대해 Phase 0 Contrastive Learning을 실행한다.
    Runs Phase 0 Contrastive Learning for a single channel.

    ### 처리 순서 / Processing Order

    1. `ContrastiveDataset` 구성 — 라벨 없이 전체 이미지 로드 / Build dataset without labels
    2. `GrayspotModel(phase=0)` 초기화 — Projection Head 포함 / Init model with projection head
    3. `Phase0Trainer.train()` 실행 — InfoNCE loss 학습 루프 / Run InfoNCE training loop
    4. `Phase0Trainer.save_backbone()` — backbone 저장 / Save backbone

    ### Args

    | 인자 / Arg | 타입 / Type | 설명 / Description |
    |---|---|---|
    | `cfg` | `dict` | config dict from load_config() |
    | `channel` | `str` | CMYK 채널 문자 / Channel character (Y/M/C/K) |
    | `device` | `torch.device` | 연산 디바이스 / Compute device |

    ### Returns

    ```python
    {
        'channel'      : str,    # 채널 / Channel
        'n_images'     : int,    # 학습 이미지 수 / Number of training images
        'epochs'       : int,    # 학습 에폭 수 / Training epochs
        'final_loss'   : float,  # 마지막 epoch loss / Final epoch loss
        'history'      : list,   # epoch별 기록 / Per-epoch records
        'backbone_path': str,    # 저장 경로 / Save path
        'skipped'      : bool,   # 건너뜀 여부 / Whether skipped
    }
    ```

    > **Note**: 이미지가 0개인 채널은 `skipped: True`로 반환된다.
    > Channels with 0 images are returned with `skipped: True`.
    """
    logger = get_logger(__name__)

    # ── 데이터셋 구성 / Build dataset ─────────────────────────────────────────
    dataset = ContrastiveDataset(cfg, channel)
    n_images = len(dataset)

    if n_images == 0:
        logger.warning(f"[{channel}] 이미지 없음 — 건너뜀 / No images — skipping")
        return {"channel": channel, "n_images": 0, "skipped": True}

    logger.info(
        f"[{channel}] ContrastiveDataset: {n_images}개 이미지 로드 / images loaded"
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["phase0"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
        drop_last=True,  # InfoNCE: 배치 크기 일정하게 / Keep batch size consistent
    )

    # ── 모델 초기화 / Initialize model ────────────────────────────────────────
    # Phase 0 모드: Contrastive (projection head 사용)
    # Phase 0 mode: Contrastive (uses projection head)
    model = GrayspotModel(cfg, phase=0).to(device)
    logger.info(f"[{channel}] GrayspotModel(phase=0) 초기화 완료 / initialized")

    # ── Phase0Trainer 실행 / Run Phase0Trainer ────────────────────────────────
    trainer = Phase0Trainer(
        model=model,
        cfg=cfg,
        channel=channel,
        device=device,
    )

    history = trainer.train(loader)

    # ── Backbone 저장 (data_set/models/) / Save backbone ─────────────────────
    backbone_path = trainer.save_backbone()

    final_loss = history[-1]["loss"] if history else float("nan")
    logger.info(f"[{channel}] Phase 0 완료 / done — Final Loss: {final_loss:.4f}")

    return {
        "channel": channel,
        "n_images": n_images,
        "epochs": cfg["phase0"]["epochs"],
        "final_loss": round(final_loss, 6),
        "history": history,
        "backbone_path": str(backbone_path),
        "skipped": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 공식 산출물 저장 / Save official deliverables
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoint(cfg: dict, results: list[dict], device: torch.device) -> Path:
    """
    Execution Plan 공식 산출물 `outputs/checkpoints/phase0_v1.pt` 를 저장한다.
    Saves the official deliverable `outputs/checkpoints/phase0_v1.pt`.

    모든 채널의 backbone `state_dict`를 메타 정보와 함께 하나의 `.pt` 파일로 묶어 저장한다.
    Bundles all channel backbone `state_dict`s with metadata into a single `.pt` file.

    ### 저장 구조 / Saved Structure

    ```python
    {
        "version"       : "phase0_v1",
        "generated_at"  : "YYYY-MM-DD HH:MM:SS",
        "backbone"      : "efficientnet_b0",
        "epochs"        : int,
        "temperature"   : float,   # InfoNCE τ
        "projection_dim": int,
        "channels"      : ["Y", "M", "C", "K"],
        "state_dicts"   : {"Y": {...}, "M": {...}, ...},
    }
    ```
    """
    logger = get_logger(__name__)

    checkpoint_dir = ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "phase0_v1.pt"

    # 채널별 backbone state_dict 수집 / Collect per-channel backbone state_dicts
    state_dicts: dict[str, dict] = {}
    for r in results:
        if r.get("skipped"):
            continue
        ch = r["channel"]
        backbone_pt = Path(r["backbone_path"])
        if backbone_pt.exists():
            state_dicts[ch] = torch.load(str(backbone_pt), map_location="cpu")
            logger.info(f"  [{ch}] backbone 로드 / loaded: {backbone_pt.name}")

    # 메타 정보와 함께 저장 / Save with metadata
    torch.save(
        {
            "version": "phase0_v1",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": cfg["model"]["backbone"],
            "epochs": cfg["phase0"]["epochs"],
            "temperature": cfg["phase0"]["temperature"],
            "projection_dim": cfg["phase0"]["projection_dim"],
            "channels": [r["channel"] for r in results if not r.get("skipped")],
            "state_dicts": state_dicts,  # {'Y': {...}, 'M': {...}, ...}
        },
        str(checkpoint_path),
    )
    logger.info(
        f"  공식 체크포인트 저장 / Official checkpoint saved: {checkpoint_path}"
    )
    return checkpoint_path


def save_history_csv(results: list[dict]) -> None:
    """
    채널별 학습 이력을 CSV로 저장한다.
    Saves per-channel training history as CSV.

    - 저장 경로 / Save path: `outputs/checkpoints/phase0_history_{channel}.csv`
    - 컬럼 / Columns: `epoch`, `loss`, `lr`, `elapsed`
    """
    logger = get_logger(__name__)
    checkpoint_dir = ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        if r.get("skipped") or not r.get("history"):
            continue
        ch = r["channel"]
        csv_path = checkpoint_dir / f"phase0_history_{ch}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "lr", "elapsed"])
            writer.writeheader()
            writer.writerows(r["history"])
        logger.info(f"  [{ch}] 학습 이력 저장 / History saved: {csv_path.name}")


def save_summary_json(results: list[dict], checkpoint_path: Path) -> Path:
    """
    Phase 0 전체 실행 요약을 JSON으로 저장한다.
    Saves the Phase 0 overall run summary as JSON.

    - 저장 경로 / Save path: `outputs/checkpoints/phase0_summary.json`

    ### 저장 구조 / Saved Structure

    ```json
    {
        "version": "phase0_v1",
        "generated_at": "YYYY-MM-DD HH:MM:SS",
        "checkpoint_path": "outputs/checkpoints/phase0_v1.pt",
        "results": [
            {
                "channel": "Y",
                "n_images": 21,
                "epochs": 10,
                "final_loss": 1.2345,
                "skipped": false
            }
        ]
    }
    ```
    """
    logger = get_logger(__name__)
    checkpoint_dir = ROOT / "outputs" / "checkpoints"
    summary_path = checkpoint_dir / "phase0_summary.json"

    summary = {
        "version": "phase0_v1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_path": str(checkpoint_path),
        "results": [
            {
                "channel": r["channel"],
                "n_images": r.get("n_images", 0),
                "epochs": r.get("epochs", 0),
                "final_loss": r.get("final_loss", None),
                "skipped": r.get("skipped", False),
            }
            for r in results
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"  요약 저장 / Summary saved: {summary_path}")
    return summary_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 SimCLR Contrastive Learning — R2 W9~W10"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="all",
        help="채널 지정 / Channel (Y/M/C/K/all, default: all)",
    )
    args = parser.parse_args()

    # ── 채널 결정 / Resolve channels ─────────────────────────────────────────
    target_channels = (
        CHANNELS if args.channel.lower() == "all" else [args.channel.upper()]
    )

    # ── config 로드 / Load config ─────────────────────────────────────────────
    cfg = load_config()
    if not validate_config(cfg):
        print("[ERROR] Configuration validation failed.")
        sys.exit(1)

    create_directories(cfg)

    # ── 로깅 설정 / Setup logging ─────────────────────────────────────────────
    setup_logging(
        log_dir=Path(cfg["storage"]["logs_dir"]),
        level=get_nested(cfg, "logging.level") or "INFO",
        format_style=get_nested(cfg, "logging.format") or "detailed",
        console=get_nested(cfg, "logging.console_output"),
        file=get_nested(cfg, "logging.file_output"),
    )
    logger = get_logger(__name__)

    # ── 디바이스 설정 / Device setup ─────────────────────────────────────────
    device = torch.device(cfg["system"]["device"])

    # ── 시드 설정 / Seed setup ────────────────────────────────────────────────
    seed = cfg["train"].get("seed") or 42
    set_seed(seed, cfg)

    # ── 스냅샷 저장 / Save config snapshot ───────────────────────────────────
    log_snapshot(
        config=cfg,
        snapshot_dir=ROOT / "outputs" / "snapshots",
        tag="phase0",
        logger=logger,
    )

    # ── 시작 배너 / Start banner ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Phase 0 — SimCLR Contrastive Learning")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  Channels    : {target_channels}")
    print(f"  Device      : {device}")
    print(f"  Backbone    : {cfg['model']['backbone']}")
    print(f"  Epochs      : {cfg['phase0']['epochs']}")
    print(f"  Batch size  : {cfg['phase0']['batch_size']}")
    print(f"  Temperature : {cfg['phase0']['temperature']}")
    print(f"  Proj dim    : {cfg['phase0']['projection_dim']}")
    print(f"  Seed        : {seed}")
    print("=" * 60)
    print()

    # ── 채널별 학습 실행 / Run training per channel ───────────────────────────
    start_time = time.time()
    results = []

    for ch in target_channels:
        result = run_phase0(cfg, ch, device)
        results.append(result)

    # ── 산출물 저장 / Save deliverables ──────────────────────────────────────
    print()
    print("=" * 60)
    print("  산출물 저장 / Saving deliverables...")
    print("=" * 60)

    checkpoint_path = save_checkpoint(cfg, results, device)
    save_history_csv(results)
    summary_path = save_summary_json(results, checkpoint_path)

    # ── 완료 요약 / Completion summary ────────────────────────────────────────
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("  Phase 0 완료 / Phase 0 Complete")
    print(f"  총 소요 시간 / Total elapsed: {elapsed:.1f}s")
    print("=" * 60)
    print(f"  {'Channel':<10} {'Images':<10} {'Epochs':<10} {'Final Loss':<14} Status")
    print(f"  {'-' * 54}")
    for r in results:
        if r.get("skipped"):
            print(f"  {r['channel']:<10} {'—':<10} {'—':<10} {'—':<14} SKIPPED")
        else:
            print(
                f"  {r['channel']:<10} {r['n_images']:<10} {r['epochs']:<10} "
                f"{r['final_loss']:<14.4f} OK"
            )
    print()
    print(f"  공식 산출물 / Official deliverable:")
    print(f"    → {checkpoint_path}")
    print(f"  Backbone 저장 위치 / Backbone save path:")
    tag_str = backbone_tag(cfg["model"]["backbone"])
    print(f"    → data_set/models/phase0_backbone_{{channel}}_{tag_str}.pt")
    print(f"  요약 / Summary:")
    print(f"    → {summary_path}")
    print("=" * 60)
    print()
    print("  다음 단계 / Next step:")
    print("    python src/scripts/run_phase0.py  ← 이미 완료 / Done")
    print("    python src/scripts/run_optuna.py --phase 0  ← R4: Phase 0 HPO")
    print("    python src/scripts/run_phase2.py  ← R2 W11~W12: Phase 2 학습")
    print()


if __name__ == "__main__":
    main()
