"""
scripts/augment_dataset.py

PRD 목표 수량 미달 레벨에 대해 제어된 증강을 수행하고 labels_master.csv 를 갱신한다.
Performs controlled augmentation for levels below PRD targets and updates labels_master.csv.

증강 정책 / Augmentation policy:
    - PRD 최소 목표에 미달하는 (channel, level) 에만 적용한다.
    - 허용 변환: 수평 뒤집기, 90°/180°/270° 회전.
    - 금지 변환: 색상 변형·블러·노이즈 등 결함 의미를 훼손하는 변환.

PRD 목표 / PRD targets (PRD Section 6.3, v2):
    Level 0: 330,  Level 1: 330,  Level 2: 330,
    Level 3: 265,  Level 4: 165,  Level 5: 100
    채널당 총 / Total per channel: 1,520장

사용법 / Usage:
    python -m src.scripts.augment_dataset

SSOT 근거 / SSOT Reference:
    - SSOT_Data_Pipeline.md §4 — 증강 정책 (allowed / forbidden transforms)
    - SSOT_Artifacts.md §3.7 — labels_master.csv canonical 라벨 파일
    - doc/Guideline/augmentation_policy.md — 공식 증강 정책 문서

Python 3.11.5
"""

from __future__ import annotations

import csv
import random
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

# ── 경로 설정 / Path setup ────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent
_ROOT_DIR = _SRC_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

from utils.utils_config import load_config

# ── PRD 목표 수량 / PRD minimum targets (PRD Section 6.3, v2) ─────────────────
PRD_TARGETS: dict[int, int] = {
    0: 330,
    1: 330,
    2: 330,
    3: 265,
    4: 165,
    5: 100,
}

_FIELDNAMES = ["filepath", "channel", "level"]


def _augment_image(img: Image.Image) -> Image.Image:
    """
    허용된 변환 중 하나를 무작위 적용한다.
    Applies one of the allowed transforms at random.

    허용 / Allowed : 원본, 수평 뒤집기, 90° / 180° / 270° 회전.
    금지 / Forbidden: 색상 변형, 블러, 노이즈 (결함 의미 훼손 방지).
    """
    transforms = [
        img,
        ImageOps.mirror(img),
        img.rotate(90),
        img.rotate(180),
        img.rotate(270),
    ]
    return random.choice(transforms)


def _read_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """
    labels_master.csv 를 읽어 미달 레벨을 증강하고 CSV를 갱신한다.
    Reads labels_master.csv, augments under-target levels, and updates the CSV.
    """
    cfg = load_config()
    storage = cfg.get("storage", {})
    data_root = Path(storage.get("data_root", "data_set"))
    labeled_dir = Path(storage.get("labeled_dir", "data_set/labeled"))
    csv_path = data_root / "labels_master.csv"

    if not csv_path.exists():
        print(f"[ERROR] labels_master.csv not found: {csv_path}")
        sys.exit(1)

    rows = _read_csv(csv_path)
    if not rows:
        print("[ERROR] labels_master.csv is empty.")
        sys.exit(1)

    # ── 채널×레벨별 그룹 구성 / Group by channel × level ──────────────────────
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["channel"], int(row["level"]))
        groups[key].append(row)

    new_rows: list[dict] = []

    for channel in sorted({"Y", "M", "C", "K"}):
        for level in range(6):
            group = groups.get((channel, level), [])
            current_count = len(group)
            target_count = PRD_TARGETS.get(level, 0)

            if current_count >= target_count:
                print(f"  [{channel}] Level {level}: {current_count:4d} >= {target_count} → skip")
                continue

            shortage = target_count - current_count
            print(f"  [{channel}] Level {level}: {current_count:4d} < {target_count} → augment {shortage}")

            if not group:
                print(f"    [WARN] No source images for [{channel}] Level {level} — skipping")
                continue

            dst_dir = labeled_dir / channel / str(level)
            dst_dir.mkdir(parents=True, exist_ok=True)

            samples = random.choices(group, k=shortage)

            for src_row in samples:
                src_rel = src_row["filepath"]
                src_path = Path(src_rel)
                if not src_path.is_absolute():
                    src_path = _ROOT_DIR / src_path

                if not src_path.exists():
                    print(f"    [WARN] Source not found: {src_path}")
                    continue

                try:
                    img = Image.open(str(src_path))
                    aug_img = _augment_image(img)

                    fname = f"aug_{uuid.uuid4().hex[:8]}.png"
                    dst_path = dst_dir / fname
                    aug_img.save(str(dst_path))

                    try:
                        rel_path = str(dst_path.relative_to(_ROOT_DIR))
                    except ValueError:
                        rel_path = str(dst_path)

                    new_rows.append({
                        "filepath": rel_path,
                        "channel": channel,
                        "level": level,
                    })

                except Exception as exc:
                    print(f"    [ERROR] {src_path}: {exc}")

    if new_rows:
        _write_csv(csv_path, rows + new_rows)
        print(f"\n[DONE] Added {len(new_rows)} augmented samples → {csv_path}")
    else:
        print("\n[DONE] No augmentation needed — all levels meet PRD targets.")


if __name__ == "__main__":
    main()
