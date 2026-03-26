"""
Grayspot — 라벨링 후 검증 테스트 / Post-labeling Validation Tests
tests/test_after_labeling.py

라벨링 작업 완료 후 학습 시작 전에 실행한다.
Run this after labeling is complete and before starting training.

검증 항목 / Validation checklist:
    1. 라벨링 폴더에 이미지가 존재하는지 / Images exist in labeled folders
    2. 채널·레벨별 샘플 수가 충분한지 / Sufficient samples per channel and level
    3. 이미지 파일이 손상되지 않았는지 / Image files are not corrupted
    4. Dataset 클래스가 정상 로드되는지 / Dataset class loads correctly
    5. DataLoader가 정상 동작하는지 / DataLoader works correctly
    6. 클래스 불균형 수준 확인 / Check class imbalance level
    7. 학습 준비 완료 여부 최종 확인 / Final readiness check for training

실행 / Run:
    python tests/test_after_labeling.py
"""

import sys
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

# 루트를 sys.path에 추가 / Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 색상 출력 헬퍼 / Color output helpers ──
def ok(msg):   print(f"    {msg}")
def fail(msg): print(f"    {msg}")
def warn(msg): print(f"     {msg}")
def info(msg): print(f"     {msg}")
def section(title):
    print(f"\n{'─'*52}")
    print(f"  {title}")
    print(f"{'─'*52}")


def load_config() -> dict:
    path = Path("config/config.yaml")
    if not path.exists():
        fail(f"config.yaml 없음 / Not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# TEST 1. 라벨링 폴더 이미지 존재 / Labeled Images Exist
# ──────────────────────────────────────────────
def test_labeled_images_exist(cfg: dict) -> tuple[bool, dict]:
    """
    labeled/{channel}/level_{N}/ 폴더에 이미지가 있는지 확인한다.
    Checks if images exist in labeled/{channel}/level_{N}/ folders.
    """
    section("TEST 1. 라벨링 이미지 존재 / Labeled Images Exist")
    labeled_dir = Path(cfg["storage"]["labeled_dir"])
    channels    = cfg["data"]["channels"]
    n_levels    = cfg["data"]["num_levels"]
    exts        = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    # 채널·레벨별 이미지 수 집계 / Count images per channel and level
    counts: dict[str, dict[int, int]] = {ch: {lv: 0 for lv in range(n_levels)} for ch in channels}

    for ch in channels:
        for lv in range(n_levels):
            folder = labeled_dir / ch / f"level_{lv}"
            if folder.exists():
                count = sum(1 for p in folder.iterdir() if p.suffix.lower() in exts)
                counts[ch][lv] = count

    # 결과 출력 / Print results
    header = f"  {'Level':<8}" + "".join(f"{ch:>8}" for ch in channels)
    print(header)
    print("  " + "─" * (8 + 8 * len(channels)))

    passed = True
    for lv in range(n_levels):
        row = f"  Level {lv:<3}"
        for ch in channels:
            c = counts[ch][lv]
            row += f"{c:>8}"
            if c == 0:
                passed = False
        print(row)

    print("  " + "─" * (8 + 8 * len(channels)))
    total_row = f"  {'Total':<8}"
    for ch in channels:
        total_row += f"{sum(counts[ch].values()):>8}"
    print(total_row)

    if passed:
        ok("모든 채널·레벨에 이미지 존재 / Images exist in all channel-level folders")
    else:
        fail("이미지가 없는 채널·레벨 존재 / Some channel-level folders have no images")

    return passed, counts


# ──────────────────────────────────────────────
# TEST 2. 샘플 수 충분 여부 / Sufficient Samples
# ──────────────────────────────────────────────
def test_sufficient_samples(cfg: dict, counts: dict) -> bool:
    """
    PRD 권장 샘플 수 기준으로 충분 여부를 확인한다.
    Checks if sample counts meet PRD recommended minimums.
    """
    section("TEST 2. 샘플 수 충분 여부 / Sufficient Samples")

    # PRD 권장 최소 샘플 수 / PRD recommended minimum samples per level
    min_samples = {0: 100, 1: 100, 2: 100, 3: 80, 4: 50, 5: 30}
    channels    = cfg["data"]["channels"]
    passed      = True

    for ch in channels:
        for lv, min_count in min_samples.items():
            actual = counts[ch][lv]
            if actual == 0:
                fail(f"[{ch}] Level {lv}: 0장 — 라벨링 필요 / Labeling required")
                passed = False
            elif actual < min_count:
                warn(f"[{ch}] Level {lv}: {actual}장 (권장 / recommended: {min_count}+) — 부족 / Insufficient")
            else:
                ok(f"[{ch}] Level {lv}: {actual}장 ✓")

    return passed


# ──────────────────────────────────────────────
# TEST 3. 이미지 파일 무결성 / Image Integrity
# ──────────────────────────────────────────────
def test_image_integrity(cfg: dict) -> bool:
    """
    라벨링된 이미지 파일이 손상되지 않았는지 확인한다.
    Checks that labeled image files are not corrupted.
    """
    section("TEST 3. 이미지 무결성 / Image Integrity")
    import cv2

    labeled_dir = Path(cfg["storage"]["labeled_dir"])
    channels    = cfg["data"]["channels"]
    n_levels    = cfg["data"]["num_levels"]
    exts        = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    corrupted = []
    total     = 0

    for ch in channels:
        for lv in range(n_levels):
            folder = labeled_dir / ch / f"level_{lv}"
            if not folder.exists():
                continue
            for img_path in folder.iterdir():
                if img_path.suffix.lower() not in exts:
                    continue
                total += 1
                img    = cv2.imread(str(img_path))
                if img is None:
                    corrupted.append(img_path)

    if not corrupted:
        ok(f"전체 {total}장 무결성 확인 완료 / All {total} images passed integrity check")
        return True
    else:
        fail(f"{len(corrupted)}장 손상 / {len(corrupted)} corrupted images found:")
        for p in corrupted[:5]:  # 최대 5개만 출력 / Show up to 5
            print(f"       - {p}")
        return False


# ──────────────────────────────────────────────
# TEST 4. Dataset 클래스 로드 / Dataset Loading
# ──────────────────────────────────────────────
def test_dataset_loading(cfg: dict) -> bool:
    """
    GrayspotDataset 클래스가 정상적으로 샘플을 로드하는지 확인한다.
    Checks that GrayspotDataset correctly loads samples.
    """
    section("TEST 4. Dataset 로드 / Dataset Loading")
    from data.dataset import GrayspotDataset

    channels = cfg["data"]["channels"]
    passed   = True

    for ch in channels:
        try:
            train_ds = GrayspotDataset(cfg, ch, split="train", augment=False)
            val_ds   = GrayspotDataset(cfg, ch, split="val",   augment=False)
            test_ds  = GrayspotDataset(cfg, ch, split="test",  augment=False)
            total    = len(train_ds) + len(val_ds) + len(test_ds)

            if total == 0:
                fail(f"[{ch}] 전체 샘플 0개 / No samples loaded")
                passed = False
            else:
                ok(f"[{ch}] Train:{len(train_ds)} | Val:{len(val_ds)} | Test:{len(test_ds)} | Total:{total}")
        except Exception as e:
            fail(f"[{ch}] Dataset 로드 오류 / Load error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 5. DataLoader 동작 / DataLoader Operation
# ──────────────────────────────────────────────
def test_dataloader(cfg: dict) -> bool:
    """
    DataLoader가 정상적으로 배치를 생성하는지 확인한다.
    Checks that DataLoader correctly generates batches.
    """
    section("TEST 5. DataLoader 동작 / DataLoader Operation")
    import torch
    from torch.utils.data import DataLoader
    from data.dataset import GrayspotDataset

    channels = cfg["data"]["channels"]
    size     = cfg["data"]["image_size"]
    passed   = True

    for ch in channels:
        try:
            ds = GrayspotDataset(cfg, ch, split="train", augment=False)
            if len(ds) == 0:
                warn(f"[{ch}] 샘플 없음 — 건너뜀 / No samples — skipping")
                continue

            loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=False)
            batch  = next(iter(loader))
            x, labels, meta = batch

            # 텐서 형태 검증 / Validate tensor shape
            expected_shape = (min(4, len(ds)), 3, size, size)
            if x.shape != torch.Size(expected_shape):
                fail(f"[{ch}] 배치 형태 오류 / Wrong batch shape: {x.shape} (expected {expected_shape})")
                passed = False
            else:
                ok(f"[{ch}] 배치 형태 / Batch shape: {x.shape}  ✓")

            # 레이블 범위 검증 / Validate label range
            if labels.min() < 0 or labels.max() >= cfg["data"]["num_levels"]:
                fail(f"[{ch}] 레이블 범위 오류 / Label out of range: [{labels.min()}, {labels.max()}]")
                passed = False
            else:
                ok(f"[{ch}] 레이블 범위 / Label range: [{labels.min().item()}, {labels.max().item()}]  ✓")

        except Exception as e:
            fail(f"[{ch}] DataLoader 오류 / Error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 6. 클래스 불균형 확인 / Class Imbalance Check
# ──────────────────────────────────────────────
def test_class_imbalance(cfg: dict, counts: dict) -> bool:
    """
    클래스 불균형 수준을 확인하고 심각한 경우 경고를 출력한다.
    Checks class imbalance and warns if severe.
    """
    section("TEST 6. 클래스 불균형 / Class Imbalance")
    channels = cfg["data"]["channels"]
    n_levels = cfg["data"]["num_levels"]
    passed   = True

    for ch in channels:
        ch_counts = [counts[ch][lv] for lv in range(n_levels)]
        total     = sum(ch_counts)
        if total == 0:
            warn(f"[{ch}] 샘플 없음 / No samples")
            continue

        max_c  = max(ch_counts)
        min_c  = min(c for c in ch_counts if c > 0) if any(ch_counts) else 0
        ratio  = max_c / max(min_c, 1)

        if ratio > 10:
            warn(f"[{ch}] 심각한 불균형 / Severe imbalance — 최대/최소 비율 / Max/min ratio: {ratio:.1f}x")
            info(f"       class_weights='balanced' 설정이 자동 보정합니다 / class_weights='balanced' will auto-correct")
        elif ratio > 5:
            warn(f"[{ch}] 중간 불균형 / Moderate imbalance — 비율 / Ratio: {ratio:.1f}x")
        else:
            ok(f"[{ch}] 균형 양호 / Good balance — 비율 / Ratio: {ratio:.1f}x")

        dist = "  ".join(f"L{lv}:{c}" for lv, c in enumerate(ch_counts))
        info(f"       분포 / Distribution: {dist}")

    return passed


# ──────────────────────────────────────────────
# TEST 7. 학습 준비 최종 확인 / Training Readiness
# ──────────────────────────────────────────────
def test_training_readiness(cfg: dict, counts: dict) -> bool:
    """
    학습을 시작하기 위한 최소 조건을 만족하는지 확인한다.
    Checks if minimum conditions for training are met.
    """
    section("TEST 7. 학습 준비 최종 확인 / Training Readiness")
    channels = cfg["data"]["channels"]
    n_levels = cfg["data"]["num_levels"]
    passed   = True

    for ch in channels:
        total          = sum(counts[ch].values())
        empty_levels   = [lv for lv in range(n_levels) if counts[ch][lv] == 0]
        nonempty_levels = [lv for lv in range(n_levels) if counts[ch][lv] > 0]

        if total < 10:
            fail(f"[{ch}] 총 샘플 {total}장 — 최소 10장 필요 / Min 10 samples required")
            passed = False
        elif empty_levels:
            warn(f"[{ch}] Level {empty_levels} 샘플 없음 / No samples — 해당 레벨 예측 불가 / Cannot predict these levels")
        else:
            ok(f"[{ch}] 총 {total}장, 전 레벨 커버 / {total} samples, all levels covered")

    # 권장 실행 명령어 출력 / Print recommended command
    if passed:
        print()
        info("권장 학습 명령어 / Recommended training command:")
        print()
        print("    # 라벨 있음 → Phase 2 직행 / Labels exist → direct to Phase 2")
        print("    python src/scripts/train.py --phase 2 --channel all --skip-phase0")
        print()
        print("    # Swing Architecture 전체 실행 / Full Swing Architecture")
        print("    python src/scripts/train.py --phase 2 --channel all")

    return passed


# ──────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  Grayspot — 라벨링 후 검증 / Post-labeling Tests")
    print("=" * 52)

    cfg = load_config()

    # TEST 1 — 이미지 존재 + 집계 / Image existence + counts
    t1_passed, counts = test_labeled_images_exist(cfg)

    results = {
        "라벨링 이미지 존재 / Labeled Images":  t1_passed,
        "샘플 수 충분 여부 / Sample Count":     test_sufficient_samples(cfg, counts),
        "이미지 무결성 / Image Integrity":       test_image_integrity(cfg),
        "Dataset 로드 / Dataset Loading":       test_dataset_loading(cfg),
        "DataLoader 동작 / DataLoader":         test_dataloader(cfg),
        "클래스 불균형 / Class Imbalance":       test_class_imbalance(cfg, counts),
        "학습 준비 확인 / Training Readiness":   test_training_readiness(cfg, counts),
    }

    # 최종 결과 요약 / Final summary
    print(f"\n{'='*52}")
    print("  최종 결과 / Final Results")
    print(f"{'='*52}")
    all_passed = True
    for name, result in results.items():
        bool = "success" if result else "failed"
        print(f"  {bool}  {name}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("    모든 테스트 통과! 학습을 시작할 수 있습니다.")
        print("       All tests passed! You can start training.")
    else:
        print("     일부 테스트 실패. 위 항목을 먼저 해결하세요.")
        print("       Some tests failed. Please fix the issues above first.")
    print()


if __name__ == "__main__":
    main()