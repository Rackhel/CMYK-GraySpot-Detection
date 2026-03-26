"""
Grayspot — 단독 추론 스크립트 / Standalone Inference Script
scripts/predict.py

사용법 / Usage:
    # 이미지 1장 추론 / Single image inference
    python scripts/predict.py --image data/images/scan_001.png

    # 폴더 내 전체 이미지 일괄 추론 / Batch inference for all images in a folder
    python scripts/predict.py --folder data/images/session_001

    # 결과 저장 없이 출력만 / Print results without saving
    python scripts/predict.py --image scan_001.png --no-save
"""

import argparse
import json
import sys
import yaml
from pathlib import Path

# 루트를 sys.path에 추가 / Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predictor import GrayspotPredictor

CHANNELS = ["Y", "M", "C", "K"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

# 레벨별 설명 / Level descriptions
LEVEL_DESC = {
    0:  "정상 · Normal",
    1:  "매우 미세 · Very subtle",
    2:  "약간 인지 · Slight",
    3:  "명확 · Clear defect",
    4:  "심각 · Severe",
    5:  "극심 · Critical",
    -1: "수동 검수 필요 · Manual review",
}


def load_config(path: str = "config/config.yaml") -> dict:
    """config.yaml을 로드한다. / Loads config.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def print_result(result: dict) -> None:
    """추론 결과를 터미널에 보기 좋게 출력한다. / Prints inference results in a readable format."""
    print("\n" + "─" * 52)
    print(f"  🖼   {result['image']}")
    print(f"  🕐  {result['timestamp']}  ({result['elapsed_ms']}ms)")
    print("─" * 52)

    # 상태별 아이콘 / Status icons
    status_icon = {
        "confirmed":     "",
        "warning":       " ",
        "manual_review": "🔴",
        "no_model":      "⛔",
    }

    for ch in CHANNELS:
        lvl    = result.get(f"{ch}_Level", -1)
        conf   = result["confidence"].get(ch, 0.0)
        status = result["status"].get(ch, "")
        icon   = status_icon.get(status, "  ")
        desc   = LEVEL_DESC.get(lvl, "")
        bar    = _conf_bar(conf)

        print(f"  {icon} [{ch}]  Level {lvl:>2}  {desc:<28}  {bar}  {conf:.2f}")

    print("─" * 52)

    # 수동 검수 필요 채널 안내 / List channels requiring manual review
    manual = [ch for ch in CHANNELS if result["status"].get(ch) == "manual_review"]
    if manual:
        print(f"\n  🔴  수동 검수 필요 / Manual review required: {', '.join(manual)}")
        print(f"       → data/analyzed/ 에서 해당 JSON 확인 / Check JSON in data/analyzed/")

    print()


def _conf_bar(conf: float, width: int = 10) -> str:
    """신뢰도를 막대 그래프로 표현한다. / Represents confidence as a bar graph."""
    filled = int(conf * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def predict_single(predictor: GrayspotPredictor, image_path: Path, save: bool) -> dict:
    """이미지 1장 추론. / Runs inference on a single image."""
    if not image_path.exists():
        print(f"    파일 없음 / File not found: {image_path}")
        return {}

    if save:
        result = predictor.predict_and_save(image_path)  # 결과 저장 / Save result
    else:
        result = predictor.predict(image_path)            # 결과 저장 없음 / No save

    print_result(result)
    return result


def predict_folder(predictor: GrayspotPredictor, folder: Path, save: bool) -> list[dict]:
    """폴더 내 전체 이미지를 일괄 추론한다. / Runs batch inference on all images in a folder."""
    images = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    )

    if not images:
        print(f"    이미지 없음 / No images found: {folder}")
        return []

    print(f"\n  📂  폴더 일괄 추론 / Batch inference: {folder}")
    print(f"      총 / Total: {len(images)}장\n")

    results = []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}")
        result = predict_single(predictor, img_path, save)
        if result:
            results.append(result)

    # 일괄 추론 요약 출력 / Print batch inference summary
    _print_batch_summary(results)
    return results


def _print_batch_summary(results: list[dict]) -> None:
    """일괄 추론 결과 요약을 출력한다. / Prints a summary of batch inference results."""
    if not results:
        return

    print("\n" + "=" * 52)
    print(f"  📊  일괄 추론 요약 / Batch Summary — 총 / Total: {len(results)}장")
    print("=" * 52)

    for ch in CHANNELS:
        levels       = [r.get(f"{ch}_Level", -1) for r in results]
        valid        = [l for l in levels if l >= 0]
        if not valid:
            continue

        avg          = sum(valid) / len(valid)
        manual_count = levels.count(-1)

        # 레벨 분포 계산 / Compute level distribution
        dist     = {lv: valid.count(lv) for lv in range(6) if valid.count(lv) > 0}
        dist_str = "  ".join(f"L{k}:{v}" for k, v in sorted(dist.items()))

        print(f"\n  [{ch}]  평균 레벨 / Avg Level: {avg:.2f}  |  수동검수 / Manual: {manual_count}장")
        print(f"       분포 / Distribution: {dist_str}")

    # 수동 검수 필요 이미지 목록 / List images requiring manual review
    manual_images = [
        r["image"] for r in results
        if any(r["status"].get(ch) == "manual_review" for ch in CHANNELS)
    ]
    if manual_images:
        print(f"\n  🔴  수동 검수 필요 / Manual review required ({len(manual_images)}장):")
        for name in manual_images:
            print(f"       - {name}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Grayspot 단독 추론 스크립트 / Standalone Inference Script")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",    type=str, help="추론할 이미지 파일 경로 / Path to image file")
    group.add_argument("--folder",   type=str, help="일괄 추론할 폴더 경로 / Path to folder for batch inference")

    parser.add_argument("--no-save", action="store_true",
                        help="결과를 analyzed/ 폴더에 저장하지 않음 / Do not save results to analyzed/ folder")
    parser.add_argument("--config",  type=str, default="config/config.yaml",
                        help="config.yaml 경로 / Path to config.yaml")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    predictor = GrayspotPredictor(cfg)
    save      = not args.no_save

    if args.image:
        predict_single(predictor, Path(args.image), save)
    else:
        predict_folder(predictor, Path(args.folder), save)


if __name__ == "__main__":
    main()