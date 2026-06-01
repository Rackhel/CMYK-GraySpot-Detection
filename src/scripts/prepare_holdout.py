"""
scripts/prepare_holdout.py

학습 전 한 번만 실행 — holdout 테스트 세트 분리 스크립트.
Run exactly ONCE before any training — splits holdout test set from labeled data.

동작 / Behavior:
    - labeled/{channel}/{level}/ 에서 레벨별 stratified로 holdout_ratio(기본 15%) 추출
    - holdout/{channel}/{level}/ 로 이동 (복사 아님 — 원본 삭제)
    - 한 번만 실행: holdout/ 폴더가 이미 있으면 실행 거부
    - 소량 클래스도 최소 1장 holdout 포함 보장

경고 / Warning:
    이 스크립트는 파일을 이동합니다 (삭제 후 복사).
    실행 전 데이터 백업을 권장합니다.
    This script MOVES files (not copies). Back up data before running.

실행 방법 / Usage:
    python -m src.scripts.prepare_holdout
    python -m src.scripts.prepare_holdout --ratio 0.15 --dry-run
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CHANNELS = ["Y", "M", "C", "K"]
_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def prepare_holdout(
    labeled_dir: Path,
    holdout_dir: Path,
    num_levels: int = 6,
    holdout_ratio: float = 0.15,
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    """
    Stratified holdout 분리 실행.
    Executes stratified holdout split.

    Returns:
        stats dict: {channel: {level: {"total": N, "moved": M}}}
    """
    if holdout_dir.exists() and any(holdout_dir.iterdir()):
        raise RuntimeError(
            f"Holdout directory already exists and is not empty: {holdout_dir}\n"
            "Delete it manually if you want to re-split."
        )

    _rng = random.Random(seed)
    stats: dict = {}

    for channel in _CHANNELS:
        stats[channel] = {}
        ch_labeled = labeled_dir / channel
        ch_holdout = holdout_dir / channel

        if not ch_labeled.exists():
            print(f"  [SKIP] {channel}: labeled dir not found")
            continue

        for level in range(num_levels):
            lv_labeled = ch_labeled / str(level)
            lv_holdout = ch_holdout / str(level)

            if not lv_labeled.exists():
                continue

            files = sorted(p for p in lv_labeled.glob("*") if p.suffix.lower() in _EXTS)
            if not files:
                continue

            _rng.shuffle(files)
            n_holdout = max(1, int(len(files) * holdout_ratio))
            holdout_files = files[:n_holdout]

            stats[channel][level] = {
                "total": len(files),
                "moved": len(holdout_files),
            }

            if not dry_run:
                lv_holdout.mkdir(parents=True, exist_ok=True)
                for f in holdout_files:
                    shutil.move(str(f), str(lv_holdout / f.name))

            status = "[DRY-RUN]" if dry_run else "[MOVED]"
            print(
                f"  {status} {channel}/Level{level}: "
                f"{len(holdout_files)}/{len(files)} → holdout/"
            )

    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Split holdout test set from labeled data (run ONCE before training)"
    )
    parser.add_argument(
        "--ratio", type=float, default=0.15, help="Holdout ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.json (optional)"
    )
    args = parser.parse_args(argv)

    try:
        from utils.utils_config import load_config

        cfg = load_config(args.config)
    except Exception:
        cfg = {}

    labeled_dir = Path(cfg.get("storage", {}).get("labeled_dir", "data_set/labeled"))
    holdout_dir = Path(cfg.get("storage", {}).get("holdout_dir", "data_set/holdout"))
    num_levels = cfg.get("data", {}).get("num_levels", 6)
    seed = cfg.get("train", {}).get("seed", args.seed)

    # 경로를 project root 기준으로 / Resolve relative to project root
    if not labeled_dir.is_absolute():
        labeled_dir = _ROOT / labeled_dir
    if not holdout_dir.is_absolute():
        holdout_dir = _ROOT / holdout_dir

    print(f"\n{'='*60}")
    print(f"  Holdout Split")
    print(f"  labeled_dir : {labeled_dir}")
    print(f"  holdout_dir : {holdout_dir}")
    print(f"  ratio       : {args.ratio}")
    print(f"  seed        : {seed}")
    if args.dry_run:
        print("  *** DRY RUN — no files will be moved ***")
    print(f"{'='*60}\n")

    if not args.dry_run:
        confirm = input("Continue? Files will be MOVED (not copied). [y/N]: ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

    stats = prepare_holdout(
        labeled_dir=labeled_dir,
        holdout_dir=holdout_dir,
        num_levels=num_levels,
        holdout_ratio=args.ratio,
        seed=seed,
        dry_run=args.dry_run,
    )

    total_moved = sum(v["moved"] for ch in stats.values() for v in ch.values())
    total_all = sum(v["total"] for ch in stats.values() for v in ch.values())
    print(f"\n{'='*60}")
    print(f"  Done — moved {total_moved}/{total_all} files to holdout/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
