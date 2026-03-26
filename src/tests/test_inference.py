"""
Grayspot -- 추론 실행 및 검증 / Inference Execution and Validation
tests/test_inference.py

test_evaluation.py 통과 후 실행한다.
Run this after test_evaluation.py passes.

실행 순서 / Execution order:
    1. 저장된 모델 로드 확인 / Verify model loading
    2. 단일 이미지 추론 확인 / Verify single image inference
    3. 출력 형식 확인 / Verify output format
    4. Fallback 전략 동작 확인 / Verify Fallback strategy
    5. 결과 JSON 저장 확인 / Verify result JSON save
    6. 일괄 추론 확인 / Verify batch inference

실행 / Run:
    python tests/test_inference.py

    # 특정 이미지로 테스트 / Test with specific image
    python tests/test_inference.py --image data/images/scan_001.png
"""

import sys
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predictor import GrayspotPredictor
from utils.logger import get_inference_logger


def pass_(msg): print(f"  [PASS] {msg}")
def fail_(msg): print(f"  [FAIL] {msg}")
def info_(msg): print(f"  [INFO] {msg}")
def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_sample_image(cfg: dict) -> Path | None:
    """
    테스트용 샘플 이미지 경로를 반환한다.
    Returns a sample image path for testing.
    """
    raw_dir = Path(cfg["storage"]["raw_dir"])
    exts    = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    images  = [p for p in raw_dir.rglob("*") if p.suffix.lower() in exts]
    return images[0] if images else None


# ──────────────────────────────────────────────
# TEST 1. 모델 로드 확인 / Model Load Verification
# ──────────────────────────────────────────────
def test_model_loading(cfg: dict) -> tuple[bool, GrayspotPredictor | None]:
    section("TEST 1. 모델 로드 확인 / Model Load Verification")
    passed    = True
    predictor = None

    try:
        predictor = GrayspotPredictor(cfg)

        loaded   = list(predictor.models.keys())
        channels = cfg["data"]["channels"]

        if not loaded:
            fail_("로드된 모델 없음 / No models loaded")
            fail_("test_training.py 를 먼저 실행하세요 / Run test_training.py first")
            return False, None

        for ch in channels:
            if ch in predictor.models:
                pass_(f"[{ch}] 모델 로드 성공 / Model loaded")
            else:
                info_(f"[{ch}] 모델 없음 -- 건너뜀 / No model -- skipping")

    except Exception as e:
        fail_(f"Predictor 초기화 오류 / Initialization error: {e}")
        passed = False

    return passed, predictor


# ──────────────────────────────────────────────
# TEST 2. 단일 이미지 추론 / Single Image Inference
# ──────────────────────────────────────────────
def test_single_inference(
    cfg: dict,
    predictor: GrayspotPredictor,
    image_path: Path = None,
) -> tuple[bool, dict]:
    section("TEST 2. 단일 이미지 추론 / Single Image Inference")

    if predictor is None:
        fail_("Predictor 없음 -- TEST 1 먼저 해결하세요 / No predictor -- fix TEST 1 first")
        return False, {}

    if image_path is None:
        image_path = get_sample_image(cfg)

    if image_path is None:
        fail_("테스트 이미지 없음 / No test image found")
        info_("data/images/ 폴더에 이미지가 있어야 합니다 / Images must exist in data/images/")
        return False, {}

    try:
        result = predictor.predict(image_path)
        pass_(f"추론 완료 / Inference done: {image_path.name} ({result['elapsed_ms']}ms)")
        return True, result

    except Exception as e:
        fail_(f"추론 오류 / Inference error: {e}")
        return False, {}


# ──────────────────────────────────────────────
# TEST 3. 출력 형식 확인 / Output Format Verification
# ──────────────────────────────────────────────
def test_output_format(cfg: dict, result: dict) -> bool:
    section("TEST 3. 출력 형식 확인 / Output Format Verification")

    if not result:
        fail_("추론 결과 없음 -- TEST 2 먼저 해결하세요 / No result -- fix TEST 2 first")
        return False

    channels   = cfg["data"]["channels"]
    num_levels = cfg["data"]["num_levels"]
    passed     = True

    # 필수 키 확인 / Check required keys
    required_keys = {"image", "timestamp", "elapsed_ms", "confidence", "status", "probabilities"}
    missing       = required_keys - set(result.keys())
    if missing:
        fail_(f"누락된 키 / Missing keys: {missing}")
        passed = False
    else:
        pass_(f"필수 키 확인 / Required keys verified: {required_keys}")

    for ch in channels:
        level_key = f"{ch}_Level"

        # Level 키 존재 확인 / Check Level key existence
        if level_key not in result:
            fail_(f"[{ch}] {level_key} 키 없음 / Key missing")
            passed = False
            continue

        level = result[level_key]
        conf  = result["confidence"].get(ch, -1)
        probs = result["probabilities"].get(ch, [])

        # Level 범위 확인 (-1은 수동검수, 0~5는 정상) / Check Level range
        if level != -1 and not (0 <= level < num_levels):
            fail_(f"[{ch}] Level 범위 오류 / Out of range: {level}")
            passed = False

        # 신뢰도 범위 확인 / Check confidence range
        elif not (0.0 <= conf <= 1.0):
            fail_(f"[{ch}] Confidence 범위 오류 / Out of range: {conf}")
            passed = False

        # 확률 분포 합계 확인 / Check probability sum
        elif probs and abs(sum(probs) - 1.0) > 0.01:
            fail_(f"[{ch}] 확률 합계 오류 / Probability sum != 1: {sum(probs):.4f}")
            passed = False

        else:
            status = result["status"].get(ch, "")
            pass_(f"[{ch}] Level: {level} | Conf: {conf:.3f} | Status: {status}")

    return passed


# ──────────────────────────────────────────────
# TEST 4. Fallback 전략 동작 확인 / Fallback Strategy Verification
# ──────────────────────────────────────────────
def test_fallback_strategy(cfg: dict, result: dict) -> bool:
    section("TEST 4. Fallback 전략 확인 / Fallback Strategy Verification")

    if not result:
        fail_("추론 결과 없음 -- TEST 2 먼저 해결하세요 / No result -- fix TEST 2 first")
        return False

    inf_cfg  = cfg["inference"]
    auto_thr = inf_cfg["confidence_auto"]
    warn_thr = inf_cfg["confidence_warn"]
    channels = cfg["data"]["channels"]
    passed   = True

    for ch in channels:
        if ch not in result["confidence"]:
            continue

        conf   = result["confidence"][ch]
        status = result["status"][ch]
        level  = result.get(f"{ch}_Level", -1)

        # Fallback 로직 검증 / Verify Fallback logic
        if conf >= auto_thr:
            expected = "confirmed"
        elif conf >= warn_thr:
            expected = "warning"
        else:
            expected = "manual_review"

        if status == expected:
            pass_(f"[{ch}] Conf: {conf:.3f} --> Status: {status}  (correct)")
        else:
            fail_(f"[{ch}] Conf: {conf:.3f} --> Status: {status} (expected: {expected})")
            passed = False

        # manual_review 시 Level -1 확인 / Check Level -1 for manual_review
        if status == "manual_review" and level != -1:
            fail_(f"[{ch}] manual_review 인데 Level이 -1이 아님 / Level should be -1: {level}")
            passed = False

    pass_(f"Fallback 임계값 / Thresholds -- Auto: {auto_thr} | Warn: {warn_thr}")
    return passed


# ──────────────────────────────────────────────
# TEST 5. 결과 JSON 저장 확인 / Result JSON Save
# ──────────────────────────────────────────────
def test_result_save(
    cfg: dict,
    predictor: GrayspotPredictor,
    image_path: Path,
) -> bool:
    section("TEST 5. 결과 JSON 저장 확인 / Result JSON Save Verification")

    if predictor is None or image_path is None:
        fail_("Predictor 또는 이미지 없음 / No predictor or image")
        return False

    try:
        result       = predictor.predict_and_save(image_path)
        analyzed_dir = Path(cfg["storage"]["analyzed_dir"])

        json_files = sorted(analyzed_dir.glob("result_*.json"))
        if not json_files:
            fail_("저장된 JSON 없음 / No saved JSON files")
            return False

        latest = json_files[-1]
        with open(latest) as f:
            saved = json.load(f)

        # 저장된 내용 검증 / Validate saved content
        if "timestamp" in saved and "elapsed_ms" in saved:
            size_kb = latest.stat().st_size / 1024
            pass_(f"JSON 저장 확인 / Saved: {latest.name} ({size_kb:.1f} KB)")
            return True
        else:
            fail_("저장된 JSON 형식 오류 / Invalid JSON format")
            return False

    except Exception as e:
        fail_(f"저장 오류 / Save error: {e}")
        return False


# ──────────────────────────────────────────────
# TEST 6. 일괄 추론 확인 / Batch Inference Verification
# ──────────────────────────────────────────────
def test_batch_inference(cfg: dict, predictor: GrayspotPredictor) -> bool:
    section("TEST 6. 일괄 추론 확인 / Batch Inference Verification")

    if predictor is None:
        fail_("Predictor 없음 -- TEST 1 먼저 해결하세요 / No predictor -- fix TEST 1 first")
        return False

    raw_dir = Path(cfg["storage"]["raw_dir"])
    exts    = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    images  = [p for p in raw_dir.rglob("*") if p.suffix.lower() in exts]

    if len(images) < 2:
        info_(f"이미지 {len(images)}장 -- 일괄 추론 테스트 생략 / Skipping batch test (need 2+)")
        return True

    # 최대 3장으로 제한 / Limit to 3 images
    test_images = images[:3]
    passed      = True
    success     = 0

    for img_path in test_images:
        try:
            result = predictor.predict(img_path)
            if result and "elapsed_ms" in result:
                success += 1
        except Exception as e:
            fail_(f"{img_path.name} 오류 / Error: {e}")
            passed = False

    pass_(f"일괄 추론 완료 / Batch done: {success}/{len(test_images)}장 성공 / succeeded")
    return passed


# ──────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Grayspot 추론 검증 / Inference Validation")
    parser.add_argument("--image",  type=str, default=None,
                        help="테스트 이미지 경로 / Test image path")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    print("=" * 55)
    print("  Grayspot -- 추론 검증 / Inference Validation")
    print("=" * 55)

    cfg        = load_config(args.config)
    image_path = Path(args.image) if args.image else get_sample_image(cfg)
    result     = {}
    predictor  = None

    test_results = {}

    load_passed, predictor = test_model_loading(cfg)
    test_results["모델 로드 / Model Load"] = load_passed

    inf_passed, result = test_single_inference(cfg, predictor, image_path)
    test_results["단일 추론 / Single Inference"] = inf_passed

    test_results["출력 형식 / Output Format"]      = test_output_format(cfg, result)
    test_results["Fallback 전략 / Fallback"]       = test_fallback_strategy(cfg, result)
    test_results["JSON 저장 / JSON Save"]          = test_result_save(cfg, predictor, image_path)
    test_results["일괄 추론 / Batch Inference"]    = test_batch_inference(cfg, predictor)

    # 최종 결과 / Final results
    print(f"\n{'='*55}")
    print("  최종 결과 / Final Results")
    print(f"{'='*55}")
    all_passed = True
    for name, res in test_results.items():
        icon = "[PASS]" if res else "[FAIL]"
        print(f"  {icon}  {name}")
        if not res:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed.")
        print("  Pipeline complete -- Ready for model conversion and deployment.")
    else:
        print("  Some tests failed. Fix the issues above before proceeding.")
    print()


if __name__ == "__main__":
    main()