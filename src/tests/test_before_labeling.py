"""
Grayspot — 라벨링 전 검증 테스트 / Pre-labeling Validation Tests
tests/test_before_labeling.py

라벨링 작업 시작 전에 실행한다. / Run this before starting the labeling process.

검증 항목 / Validation checklist:
    1. 폴더 구조가 올바르게 생성됐는지 / Folder structure is correctly created
    2. 다운로드된 이미지가 존재하는지 / Downloaded images exist
    3. 이미지가 정상적으로 로드되는지 / Images can be loaded correctly
    4. RGB → CMYK 변환이 정상 동작하는지 / RGB to CMYK conversion works correctly
    5. ROI 자동 검출이 동작하는지 / ROI auto-detection works
    6. 전처리 파이프라인이 정상 동작하는지 / Preprocessing pipeline works correctly
    7. config.yaml 경로 설정이 올바른지 / config.yaml path settings are correct

실행 / Run:
    python tests/test_before_labeling.py
"""

import sys
import yaml
import numpy as np
from pathlib import Path

# 루트를 sys.path에 추가 / Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 색상 출력 헬퍼 / Color output helpers ──
def ok(msg):   print(f"    {msg}")
def fail(msg): print(f"    {msg}")
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
# TEST 1. 폴더 구조 검증 / Folder Structure
# ──────────────────────────────────────────────
def test_folder_structure(cfg: dict) -> bool:
    section("TEST 1. 폴더 구조 / Folder Structure")
    st       = cfg["storage"]
    channels = cfg["data"]["channels"]
    n_levels = cfg["data"]["num_levels"]
    passed   = True

    # 필수 루트 폴더 / Required root folders
    required = [
        Path(st["base_dir"]),
        Path(st["raw_dir"]),
        Path(st["labeled_dir"]),
        Path(st["training_dir"]),
        Path(st["analyzed_dir"]),
        Path(st["reports_dir"]),
        Path(cfg["inference"]["model_dir"]),
    ]
    for folder in required:
        if folder.exists():
            ok(f"{folder}")
        else:
            fail(f"{folder} — 없음 / Missing (setup_storage.py 실행 필요 / Run setup_storage.py)")
            passed = False

    # labeled/ 채널·레벨 폴더 / labeled/ channel-level folders
    missing_count = 0
    for ch in channels:
        for lv in range(n_levels):
            p = Path(st["labeled_dir"]) / ch / f"level_{lv}"
            if not p.exists():
                missing_count += 1
    if missing_count == 0:
        ok(f"labeled/ 채널·레벨 폴더 전체 존재 / All channel-level folders exist ({len(channels) * n_levels}개)")
    else:
        fail(f"labeled/ 폴더 {missing_count}개 누락 / {missing_count} folders missing")
        passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 2. 이미지 존재 여부 / Image Existence
# ──────────────────────────────────────────────
def test_images_exist(cfg: dict) -> bool:
    section("TEST 2. 다운로드 이미지 / Downloaded Images")
    raw_dir = Path(cfg["storage"]["raw_dir"])
    exts    = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    images  = [p for p in raw_dir.rglob("*") if p.suffix.lower() in exts]

    if len(images) == 0:
        fail(f"이미지 없음 / No images found in {raw_dir}")
        info("download_dataset.py 를 먼저 실행하세요 / Run download_dataset.py first")
        return False

    ok(f"{len(images)}장 발견 / images found: {raw_dir}")
    info(f"첫 번째 이미지 / First image: {images[0].name}")
    return True


# ──────────────────────────────────────────────
# TEST 3. 이미지 로드 / Image Loading
# ──────────────────────────────────────────────
def test_image_loading(cfg: dict) -> bool:
    section("TEST 3. 이미지 로드 / Image Loading")
    import cv2
    raw_dir = Path(cfg["storage"]["raw_dir"])
    exts    = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    images  = [p for p in raw_dir.rglob("*") if p.suffix.lower() in exts]

    if not images:
        fail("이미지 없음 — TEST 2 먼저 해결하세요 / No images — fix TEST 2 first")
        return False

    # 첫 3장 로드 테스트 / Test loading first 3 images
    passed = True
    for img_path in images[:3]:
        img = cv2.imread(str(img_path))
        if img is None:
            fail(f"로드 실패 / Load failed: {img_path.name}")
            passed = False
        else:
            h, w, c = img.shape
            ok(f"{img_path.name} — {w}×{h} px, {c}ch")

    return passed


# ──────────────────────────────────────────────
# TEST 4. RGB → CMYK 변환 / Color Conversion
# ──────────────────────────────────────────────
def test_rgb_to_cmyk() -> bool:
    section("TEST 4. RGB → CMYK 변환 / Color Conversion")
    from data.preprocessing import rgb_to_cmyk

    # 테스트 케이스 / Test cases
    test_cases = [
        ("순수 빨강 / Pure Red",   np.array([[[255, 0, 0]]], dtype=np.uint8)),
        ("순수 초록 / Pure Green", np.array([[[0, 255, 0]]], dtype=np.uint8)),
        ("순수 파랑 / Pure Blue",  np.array([[[0, 0, 255]]], dtype=np.uint8)),
        ("흰색 / White",           np.array([[[255, 255, 255]]], dtype=np.uint8)),
        ("검정 / Black",           np.array([[[0, 0, 0]]], dtype=np.uint8)),
    ]

    passed = True
    for name, rgb in test_cases:
        try:
            cmyk = rgb_to_cmyk(rgb)
            # 범위 검증 / Range validation [0, 1]
            for ch, arr in cmyk.items():
                if arr.min() < -0.01 or arr.max() > 1.01:
                    fail(f"{name} — [{ch}] 범위 초과 / Out of range: [{arr.min():.3f}, {arr.max():.3f}]")
                    passed = False
                    break
            else:
                vals = {ch: f"{v[0,0]:.2f}" for ch, v in cmyk.items()}
                ok(f"{name} → {vals}")
        except Exception as e:
            fail(f"{name} — 오류 / Error: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 5. ROI 자동 검출 / ROI Auto-detection
# ──────────────────────────────────────────────
def test_roi_extraction(cfg: dict) -> bool:
    section("TEST 5. ROI 추출 / ROI Extraction")
    import cv2
    from data.preprocessing import rgb_to_cmyk, extract_roi_auto, extract_roi_fixed

    raw_dir = Path(cfg["storage"]["raw_dir"])
    exts    = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    images  = [p for p in raw_dir.rglob("*") if p.suffix.lower() in exts]

    if not images:
        fail("이미지 없음 — TEST 2 먼저 해결하세요 / No images — fix TEST 2 first")
        return False

    img_path = images[0]
    img      = cv2.imread(str(img_path))
    rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cmyk     = rgb_to_cmyk(rgb)

    passed = True

    # 자동 검출 테스트 / Auto detection test
    try:
        rois, used_fallback = extract_roi_auto(
            cmyk,
            min_confidence=cfg["preprocessing"]["roi_auto_min_confidence"],
            fallback_coords=cfg["preprocessing"].get("roi_fixed_coords"),
        )
        if used_fallback:
            info("자동 검출 신뢰도 미달 → fixed fallback 사용 / Low confidence → using fixed fallback")
        else:
            ok("자동 경계 검출 성공 / Auto boundary detection succeeded")

        for ch, roi in rois.items():
            if roi.size == 0:
                fail(f"[{ch}] ROI 크기 0 / ROI size is 0")
                passed = False
            else:
                ok(f"[{ch}] ROI 크기 / Size: {roi.shape[1]}×{roi.shape[0]} px")
    except Exception as e:
        fail(f"ROI 추출 오류 / Extraction error: {e}")
        passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 6. 전처리 파이프라인 / Preprocessing Pipeline
# ──────────────────────────────────────────────
def test_preprocessing_pipeline(cfg: dict) -> bool:
    section("TEST 6. 전처리 파이프라인 / Preprocessing Pipeline")
    from data.preprocessing import preprocess

    raw_dir = Path(cfg["storage"]["raw_dir"])
    exts    = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    images  = [p for p in raw_dir.rglob("*") if p.suffix.lower() in exts]

    if not images:
        fail("이미지 없음 — TEST 2 먼저 해결하세요 / No images — fix TEST 2 first")
        return False

    passed   = True
    img_path = images[0]
    size     = cfg["data"]["image_size"]

    try:
        result = preprocess(img_path, cfg, return_feature=True)

        for ch in ["Y", "M", "C", "K"]:
            # 채널 ROI 검증 / Validate channel ROI
            arr = result[ch]
            if arr.shape != (size, size, 3):
                fail(f"[{ch}] 형태 오류 / Wrong shape: {arr.shape} (expected ({size},{size},3))")
                passed = False
            else:
                ok(f"[{ch}] ROI — {arr.shape}  ✓")

            # Feature Enhancement 검증 / Validate feature enhancement
            feat = result.get(f"{ch}_feature")
            if feat is None or feat.shape != (size, size, 3):
                fail(f"[{ch}_feature] 형태 오류 / Wrong shape")
                passed = False
            else:
                ok(f"[{ch}_feature] — {feat.shape}  ✓")

    except Exception as e:
        fail(f"전처리 오류 / Preprocessing error: {e}")
        passed = False

    return passed


# ──────────────────────────────────────────────
# TEST 7. config.yaml 경로 설정 / Config Paths
# ──────────────────────────────────────────────
def test_config_paths(cfg: dict) -> bool:
    section("TEST 7. config.yaml 경로 설정 / Config Path Settings")
    passed = True

    checks = [
        ("storage.base_dir",    cfg["storage"]["base_dir"]),
        ("storage.raw_dir",     cfg["storage"]["raw_dir"]),
        ("storage.labeled_dir", cfg["storage"]["labeled_dir"]),
        ("inference.model_dir", cfg["inference"]["model_dir"]),
        ("storage.reports_dir", cfg["storage"]["reports_dir"]),
    ]
    for key, val in checks:
        ok(f"{key}: \"{val}\"")

    # data 설정 검증 / Validate data settings
    size = cfg["data"]["image_size"]
    lvls = cfg["data"]["num_levels"]
    chs  = cfg["data"]["channels"]

    if size not in [224, 256, 299, 384]:
        info(f"image_size={size} — 일반적이지 않은 값 / Uncommon value (권장 / recommended: 224)")
    else:
        ok(f"image_size: {size}")

    ok(f"num_levels: {lvls} (Level 0~{lvls-1})")
    ok(f"channels: {chs}")

    return passed


# ──────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  Grayspot — 라벨링 전 검증 / Pre-labeling Tests")
    print("=" * 52)

    cfg     = load_config()
    results = {}

    results["폴더 구조 / Folder Structure"]          = test_folder_structure(cfg)
    results["이미지 존재 / Images Exist"]             = test_images_exist(cfg)
    results["이미지 로드 / Image Loading"]            = test_image_loading(cfg)
    results["RGB→CMYK 변환 / Conversion"]            = test_rgb_to_cmyk()
    results["ROI 추출 / ROI Extraction"]             = test_roi_extraction(cfg)
    results["전처리 파이프라인 / Preprocessing"]      = test_preprocessing_pipeline(cfg)
    results["config 경로 / Config Paths"]            = test_config_paths(cfg)

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
        print("    모든 테스트 통과! 라벨링을 시작하세요.")
        print("       All tests passed! You can start labeling.")
    else:
        print("     일부 테스트 실패. 위 항목을 먼저 해결하세요.")
        print("       Some tests failed. Please fix the issues above first.")
    print()


if __name__ == "__main__":
    main()