"""
scripts/prepare_dataset.py

data_set/roi/ 의 lvlX_..._CH.png 파일에서 128×128 패치를 추출하고
data_set/labeled/{channel}/{level}/ 에 배치한 뒤 labels_master.csv 를 갱신한다.
갱신 완료 후 augment_dataset.py 를 실행하여 PRD v2 목표 달성.

Extracts 128×128 patches from data_set/roi/lvlX_..._CH.png files,
places them under data_set/labeled/{channel}/{level}/,
updates labels_master.csv, then runs augment_dataset.py for PRD v2 targets.

파일명 규칙 / Filename convention:
    roi/ : lvl{level}_{name}_{suffix}_{CHANNEL}.png
    labeled/ : {uuid8}.png  (UUID 기반 고유 파일명)

패치 추출 전략 / Patch extraction strategy:
    - 가로(width): 128px 중앙 크롭 (더 좁으면 reflect 패딩)
    - 세로(height): stride=128 비겹침 슬라이딩 윈도우
    - 저분산 패치 제거: std < MIN_STD (비인쇄 영역 제거)

Usage:
    python -m src.scripts.prepare_dataset

SSOT Reference:
    - SSOT_Data_Pipeline.md §0 — 데이터 생산 파이프라인 흐름
    - SSOT_ROI_Pipeline.md §2 — CMYK 채널 분리 및 채널별 독립 라벨링
    - doc/Guideline/augmentation_policy.md §3 — PRD v2 목표

Python 3.11.5
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# ── 경로 설정 / Path setup ────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent
_ROOT_DIR = _SRC_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

ROI_DIR = _ROOT_DIR / "data_set" / "roi"
LABELED_DIR = _ROOT_DIR / "data_set" / "labeled"
LABELS_CSV = _ROOT_DIR / "data_set" / "labels_master.csv"

# ── 설정 / Settings ───────────────────────────────────────────────────────────
PATCH_SIZE = 128
STRIDE = 128
MIN_STD = 5.0   # 이 값 미만이면 비인쇄 영역으로 판단하여 제거
CHANNELS = {"Y", "M", "C", "K"}

# PRD v2 레벨별 추출 상한선 (augment_dataset.py 의 목표와 동일)
# 이 수까지만 원본 패치를 저장하고 나머지는 augment_dataset.py 가 채운다.
EXTRACT_CAP: dict[int, int] = {
    0: 330,
    1: 330,
    2: 330,
    3: 265,
    4: 165,
    5: 100,
}

# lvlX_..._CH.png 파일명 파싱 패턴
_FNAME_RE = re.compile(r"^lvl(\d+)_.+_([YMCK])$")


def _parse_filename(stem: str) -> tuple[int, str] | None:
    """파일명 stem에서 (level, channel) 추출. 매칭 실패 시 None."""
    m = _FNAME_RE.match(stem)
    if not m:
        return None
    level, channel = int(m.group(1)), m.group(2)
    if channel not in CHANNELS:
        return None
    return level, channel


def _extract_patches(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    BGR uint8 이미지에서 128×128 패치를 추출한다.

    전략:
      - 가로: 128px 중앙 크롭 (더 좁으면 reflect 패딩)
      - 세로: stride=128 비겹침 슬라이딩 윈도우
      - 저분산 패치 제거 (MIN_STD 미만)
    """
    h, w = img_bgr.shape[:2]

    # ── 가로 정규화 / Width normalization ─────────────────────────────────────
    if w >= PATCH_SIZE:
        x0 = (w - PATCH_SIZE) // 2
        strip = img_bgr[:, x0 : x0 + PATCH_SIZE]
    else:
        pad_total = PATCH_SIZE - w
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        strip = cv2.copyMakeBorder(
            img_bgr, 0, 0, pad_l, pad_r, cv2.BORDER_REFLECT
        )

    # ── 세로 슬라이딩 / Vertical sliding ──────────────────────────────────────
    patches: list[np.ndarray] = []
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        patch = strip[y : y + PATCH_SIZE, :PATCH_SIZE]
        if patch.std() >= MIN_STD:
            patches.append(patch.copy())

    return patches


def _ensure_dirs() -> None:
    for ch in CHANNELS:
        for lv in range(6):
            (LABELED_DIR / ch / str(lv)).mkdir(parents=True, exist_ok=True)


def _load_existing_rows() -> list[dict]:
    if not LABELS_CSV.exists():
        return []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict]) -> None:
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "channel", "level"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 60)
    print("prepare_dataset.py — ROI 패치 추출 + labeled/ 배치")
    print("=" * 60)

    _ensure_dirs()
    roi_files = sorted(ROI_DIR.glob("lvl*.png"))

    if not roi_files:
        print(f"[WARN] ROI 파일 없음: {ROI_DIR}")
        return

    print(f"[INFO] ROI 파일 수: {len(roi_files)}")

    existing_rows = _load_existing_rows()
    existing_paths = {r["filepath"] for r in existing_rows}
    new_rows: list[dict] = []

    # ── 채널×레벨 카운터 (상한선 추적 + 출력용)
    counts: dict[tuple[str, int], int] = {}
    # 이미 existing_rows 에 있는 수를 상한선 계산에 반영
    for r in existing_rows:
        key = (r["channel"], int(r["level"]))
        counts[key] = counts.get(key, 0) + 1

    for roi_path in roi_files:
        parsed = _parse_filename(roi_path.stem)
        if parsed is None:
            print(f"  [SKIP] 파싱 실패: {roi_path.name}")
            continue
        level, channel = parsed

        # ── PRD v2 상한선 도달 시 이 (channel, level) 건너뜀 ──────────────────
        cap = EXTRACT_CAP.get(level, 0)
        if counts.get((channel, level), 0) >= cap:
            continue

        img = cv2.imread(str(roi_path))
        if img is None:
            print(f"  [SKIP] 이미지 로드 실패: {roi_path.name}")
            continue

        patches = _extract_patches(img)
        if not patches:
            continue

        dst_dir = LABELED_DIR / channel / str(level)
        # ROI stem → 파일명 안전 문자열 (공백 → '_')
        safe_stem = roi_path.stem.replace(" ", "_")
        for patch_idx, patch in enumerate(patches, start=1):
            if counts.get((channel, level), 0) >= cap:
                break  # 이 (channel, level) 의 상한선 도달
            fname = f"{safe_stem}_{patch_idx:04d}.png"
            dst_path = dst_dir / fname
            cv2.imwrite(str(dst_path), patch)
            rel_path = str(dst_path.relative_to(_ROOT_DIR))
            if rel_path not in existing_paths:
                new_rows.append(
                    {"filepath": rel_path, "channel": channel, "level": level}
                )
                existing_paths.add(rel_path)
            counts[(channel, level)] = counts.get((channel, level), 0) + 1

    # ── CSV 갱신 / Update CSV ──────────────────────────────────────────────────
    all_rows = existing_rows + new_rows
    _write_csv(all_rows)

    total_patches = sum(counts.values())
    print(f"\n[DONE] 추출 완료: {total_patches}장 (신규 {len(new_rows)}행 추가)")
    print("\n채널×레벨 분포 / Channel×Level distribution:")
    print(f"  {'':>4}  " + "  ".join(f"L{lv}" for lv in range(6)))
    for ch in sorted(CHANNELS):
        row_str = "  ".join(
            f"{counts.get((ch, lv), 0):>3}" for lv in range(6)
        )
        total = sum(counts.get((ch, lv), 0) for lv in range(6))
        print(f"  {ch:>4}: {row_str}  (합계: {total})")

    # ── augment_dataset.py 실행 / Run augmentation ────────────────────────────
    print("\n" + "=" * 60)
    print("augment_dataset.py 실행 — PRD v2 목표 달성")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "src.scripts.augment_dataset"],
        cwd=str(_ROOT_DIR),
    )
    if result.returncode != 0:
        print(f"[ERROR] augment_dataset.py 실패 (returncode={result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
