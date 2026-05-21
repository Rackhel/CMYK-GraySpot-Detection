"""
scripts/prepare_dataset.py

data_set/roi/ 의 lvlX_..._CH.png 파일에서 128×128 패치를 추출하고
data_set/labeled/{channel}/{level}/ 에 배치한 뒤 labels_master.csv 를 갱신한다.
갱신 완료 후 augment_dataset.py 를 실행하여 PRD v2 목표 달성.

Extracts 128×128 patches from data_set/roi/lvlX_..._CH.png files,
places them under data_set/labeled/{channel}/{level}/,
updates labels_master.csv, then runs augment_dataset.py for PRD v2 targets.

파일명 규칙 / Filename convention:
    roi/    : lvl{level}_{name}_{CHANNEL}.png
    labeled : {roi_stem}_{patch_idx:04d}.png  (원본)
              aug_{uuid8}.png                 (증강, augment_dataset.py 생성)

채널별 독립 라벨링 / Per-channel independent labeling:
    기본값: ROI 파일명의 lvlX 를 사용 (스캔 단위 레벨).
    오버라이드: data_set/roi_labels.csv 가 존재하면 채널별 시각 검사 레벨을 우선 적용.
    roi_labels.csv 형식: roi_filename,level  (헤더 포함, roi_filename = 확장자 없는 스템)

    Default: use lvlX from ROI filename (scan-level label).
    Override: if data_set/roi_labels.csv exists, per-channel visual-inspection levels
    take priority. Format: roi_filename,level  (with header; roi_filename = stem without ext)

패치 추출 / Patch extraction:
    ROIExtractor.extract_patches_from_roi() 위임
    (가로 중앙 크롭, 세로 stride=128 슬라이딩, 저분산 패치 제거)

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

# ── 경로 설정 / Path setup ────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent
_ROOT_DIR = _SRC_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

from data.roi_extractor import ROIExtractor

ROI_DIR = _ROOT_DIR / "data_set" / "roi"
LABELED_DIR = _ROOT_DIR / "data_set" / "labeled"
LABELS_CSV = _ROOT_DIR / "data_set" / "labels_master.csv"
ROI_LABELS_CSV = _ROOT_DIR / "data_set" / "roi_labels.csv"

# ── 설정 / Settings ───────────────────────────────────────────────────────────
CHANNELS = {"Y", "M", "C", "K"}

# PRD v2 레벨별 추출 상한선 (augment_dataset.py 목표와 동일)
# 이 수까지만 원본 패치를 저장하고 나머지는 augment_dataset.py 가 채운다.
EXTRACT_CAP: dict[int, int] = {
    0: 330,
    1: 330,
    2: 330,
    3: 265,
    4: 165,
    5: 100,
}

# ROIExtractor 설정 (lvlX_..._CH.png 는 이미 채널 분리된 이미지)
_ROI_CFG = {
    "roi": {"mode": "auto"},
    "data": {"image_size": 128},
}

_extractor = ROIExtractor(cfg=_ROI_CFG)

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


def _load_roi_labels() -> dict[str, int]:
    """data_set/roi_labels.csv 에서 roi_filename → level 매핑을 로드한다.

    파일이 없으면 빈 dict 반환 → 파일명 기반 레벨 fallback 사용.
    파일이 있으면 채널별 시각 검사 레벨(per-channel visual-inspection level)이 우선 적용된다.

    CSV 형식 / CSV format:
        roi_filename,level
        lvl3_Scanned Documents (113)_3_1_C,1
        lvl3_Scanned Documents (113)_3_1_M,3
    """
    if not ROI_LABELS_CSV.exists():
        return {}
    mapping: dict[str, int] = {}
    with open(ROI_LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["roi_filename"]] = int(row["level"])
    print(f"[INFO] roi_labels.csv 로드: {len(mapping)}개 채널별 레벨 오버라이드")
    return mapping


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

    roi_label_map = _load_roi_labels()
    if not roi_label_map:
        print("[INFO] roi_labels.csv 없음 — 파일명 기반 레벨(스캔 단위) 사용")
    else:
        print("[INFO] roi_labels.csv 적용 — 채널별 독립 라벨링 모드")

    existing_rows = _load_existing_rows()
    existing_paths = {r["filepath"] for r in existing_rows}
    new_rows: list[dict] = []

    # 상한선 추적 카운터 (기존 rows 반영)
    counts: dict[tuple[str, int], int] = {}
    for r in existing_rows:
        key = (r["channel"], int(r["level"]))
        counts[key] = counts.get(key, 0) + 1

    for roi_path in roi_files:
        parsed = _parse_filename(roi_path.stem)
        if parsed is None:
            print(f"  [SKIP] 파싱 실패: {roi_path.name}")
            continue
        filename_level, channel = parsed
        # roi_labels.csv 에 해당 파일이 있으면 채널별 시각 검사 레벨 우선 적용
        level = roi_label_map.get(roi_path.stem, filename_level)

        cap = EXTRACT_CAP.get(level, 0)
        if counts.get((channel, level), 0) >= cap:
            continue  # 이 (channel, level) 상한선 도달

        try:
            patches = _extractor.extract_patches_from_roi(roi_path)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        if not patches:
            continue

        dst_dir = LABELED_DIR / channel / str(level)
        safe_stem = roi_path.stem.replace(" ", "_")

        for patch_idx, patch in enumerate(patches, start=1):
            if counts.get((channel, level), 0) >= cap:
                break
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

    all_rows = existing_rows + new_rows
    _write_csv(all_rows)

    total_patches = sum(counts.values())
    print(f"\n[DONE] 추출 완료: {total_patches}장 (신규 {len(new_rows)}행 추가)")
    print("\n채널×레벨 분포 / Channel×Level distribution:")
    print(f"  {'':>4}  " + "  ".join(f"L{lv}" for lv in range(6)))
    for ch in sorted(CHANNELS):
        row_str = "  ".join(f"{counts.get((ch, lv), 0):>3}" for lv in range(6))
        total = sum(counts.get((ch, lv), 0) for lv in range(6))
        print(f"  {ch:>4}: {row_str}  (합계: {total})")

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
