"""
scripts/generate_synthetic.py

희귀 클래스를 위한 합성 데이터 생성 스크립트.
Synthetic data generation for rare-class augmentation.

두 가지 방법 / Two methods:
    1. interpolation (기본 / default):
       Level 0 (clean)과 Level N (defect)을 alpha 비율로 보간하여 중간 레벨 합성.
       Interpolates between Level 0 (clean) and Level N (defect) at alpha ratio
       to synthesize intermediate defect severity.

    2. img2img (옵션 / optional, requires diffusers):
       Stable Diffusion img2img 방식으로 Level 0 패치에 결함 강도 조건 부여.
       Uses Stable Diffusion img2img conditioned on defect severity.

출력 / Output:
    labeled/{channel}/{level}/synthetic_{N:04d}.png

주의 / Note:
    - prepare_holdout.py 실행 AFTER 에 실행할 것 (labeled/ 만 보강)
    - Run AFTER prepare_holdout.py (augments labeled/ only, never holdout/)
    - 생성된 파일은 CMYKDataset에서 exclude_synthetic=True로 제외 가능

실행 방법 / Usage:
    python -m src.scripts.generate_synthetic --method interpolation
    python -m src.scripts.generate_synthetic --channel K --level 2 --count 100
    python -m src.scripts.generate_synthetic --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CHANNELS = ["Y", "M", "C", "K"]
_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


# ──────────────────────────────────────────────────────────────
# Method 1: Level Interpolation
# ──────────────────────────────────────────────────────────────


def _load_images(level_dir: Path, limit: int | None = None) -> list[np.ndarray]:
    """폴더에서 이미지 로드 (BGR, uint8) / Load images from folder (BGR uint8)."""
    imgs = []
    for p in sorted(level_dir.glob("*")):
        if p.suffix.lower() not in _EXTS:
            continue
        if p.stem.startswith("synthetic_"):
            continue  # 기존 합성 이미지는 소스로 사용하지 않음
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)
        if limit and len(imgs) >= limit:
            break
    return imgs


def generate_interpolation(
    channel_dir: Path,
    target_level: int,
    source_level_low: int,
    source_level_high: int,
    count: int,
    image_size: int = 128,
    seed: int = 42,
    dry_run: bool = False,
) -> int:
    """
    Level 보간으로 합성 이미지 생성.
    Generate synthetic images via level interpolation.

    target_level 이미지 = alpha * high_defect + (1-alpha) * clean

    Args:
        channel_dir:      labeled/{channel}/ 경로
        target_level:     생성할 레벨 (예: 2)
        source_level_low: 낮은 결함 소스 레벨 (보통 0)
        source_level_high: 높은 결함 소스 레벨 (보통 5)
        count:            생성할 이미지 수
        seed:             재현성 시드
    Returns:
        생성된 이미지 수 / number of images generated
    """
    rng = np.random.default_rng(seed)

    low_dir  = channel_dir / str(source_level_low)
    high_dir = channel_dir / str(source_level_high)
    out_dir  = channel_dir / str(target_level)

    low_imgs  = _load_images(low_dir)
    high_imgs = _load_images(high_dir)

    if not low_imgs or not high_imgs:
        print(f"  [SKIP] Not enough source images for level {target_level} interpolation")
        return 0

    # alpha: 타겟 레벨 기준으로 결함 강도 결정
    # alpha = target_level / (num_levels - 1) → 선형 보간
    num_levels = max(source_level_high + 1, target_level + 1)
    alpha_center = target_level / (num_levels - 1)
    alpha_std = 0.05  # 약간의 다양성 추가

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("synthetic_*.png")))
    generated = 0

    for i in range(count):
        alpha = float(np.clip(rng.normal(alpha_center, alpha_std), 0.05, 0.95))
        low_img  = cv2.resize(rng.choice(low_imgs),  (image_size, image_size))
        high_img = cv2.resize(rng.choice(high_imgs), (image_size, image_size))

        synthetic = cv2.addWeighted(low_img, 1.0 - alpha, high_img, alpha, 0)

        # 미세 노이즈 추가로 다양성 확보 / Add slight noise for diversity
        noise = rng.integers(-8, 8, synthetic.shape, dtype=np.int16)
        synthetic = np.clip(synthetic.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        fname = out_dir / f"synthetic_{existing + i:04d}.png"
        if not dry_run:
            cv2.imwrite(str(fname), synthetic)
        generated += 1

    return generated


# ──────────────────────────────────────────────────────────────
# Method 2: Stable Diffusion img2img (optional)
# ──────────────────────────────────────────────────────────────


def generate_img2img(
    channel_dir: Path,
    target_level: int,
    count: int,
    image_size: int = 128,
    strength: float = 0.4,
    seed: int = 42,
    dry_run: bool = False,
) -> int:
    """
    Stable Diffusion img2img 방식으로 합성 이미지 생성.
    Generate synthetic images using Stable Diffusion img2img.

    Level 0 (clean) 패치를 입력으로, 결함 강도에 맞는 텍스처를 추가합니다.
    Takes Level 0 (clean) patches and applies defect-conditioned texture.

    requires: pip install diffusers transformers accelerate
    """
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        import torch
        from PIL import Image
    except ImportError:
        print(
            "  [ERROR] diffusers not installed. Run:\n"
            "  pip install diffusers transformers accelerate"
        )
        return 0

    rng = np.random.default_rng(seed)
    low_dir = channel_dir / "0"
    out_dir = channel_dir / str(target_level)
    low_imgs = _load_images(low_dir, limit=50)

    if not low_imgs:
        print("  [SKIP] No Level-0 source images found")
        return 0

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    pipe.safety_checker = None

    severity_prompts = {
        1: "slight ink bleeding, minor print defect, cmyk patch",
        2: "moderate ink bleeding, visible print defect, cmyk patch",
        3: "heavy ink bleeding, significant print defect, cmyk patch",
        4: "severe ink bleeding, major print defect, cmyk patch",
        5: "extreme ink overflow, critical print defect, cmyk patch",
    }
    prompt = severity_prompts.get(target_level, f"level {target_level} print defect, cmyk patch")

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("synthetic_*.png")))
    generated = 0

    for i in range(count):
        src = cv2.resize(rng.choice(low_imgs), (512, 512))
        src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(src_rgb)

        result = pipe(
            prompt=prompt,
            image=pil_img,
            strength=strength,
            guidance_scale=7.5,
            generator=torch.Generator(device).manual_seed(int(seed) + i),
        ).images[0]

        out_img = np.array(result.resize((image_size, image_size)))
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        fname = out_dir / f"synthetic_{existing + i:04d}.png"
        if not dry_run:
            cv2.imwrite(str(fname), out_bgr)
        generated += 1

    return generated


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic images for rare defect levels"
    )
    parser.add_argument("--method", choices=["interpolation", "img2img"],
                        default="interpolation")
    parser.add_argument("--channel", choices=_CHANNELS + ["all"], default="all",
                        help="Target channel (default: all)")
    parser.add_argument("--level", type=int, default=None,
                        help="Target level to generate (default: auto-detect rare levels)")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of images to generate per level (default: 100)")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Only generate if existing count < this threshold (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)

    try:
        from utils.utils_config import load_config
        cfg = load_config(args.config)
    except Exception:
        cfg = {}

    labeled_dir = _ROOT / Path(cfg.get("storage", {}).get("labeled_dir", "data_set/labeled"))
    image_size  = cfg.get("data", {}).get("image_size", 128)
    num_levels  = cfg.get("data", {}).get("num_levels", 6)

    channels = _CHANNELS if args.channel == "all" else [args.channel]

    print(f"\n{'='*60}")
    print(f"  Synthetic Data Generation")
    print(f"  method     : {args.method}")
    print(f"  channels   : {channels}")
    print(f"  count/level: {args.count}")
    if args.dry_run:
        print("  *** DRY RUN ***")
    print(f"{'='*60}\n")

    total_generated = 0

    for channel in channels:
        ch_dir = labeled_dir / channel
        if not ch_dir.exists():
            print(f"  [SKIP] {channel}: directory not found")
            continue

        # 생성할 레벨 결정 / Determine target levels
        if args.level is not None:
            target_levels = [args.level]
        else:
            # 샘플이 min_samples 미만인 레벨 자동 탐지
            target_levels = []
            for lv in range(num_levels):
                lv_dir = ch_dir / str(lv)
                if not lv_dir.exists():
                    target_levels.append(lv)
                    continue
                real_count = sum(
                    1 for p in lv_dir.glob("*")
                    if p.suffix.lower() in _EXTS and not p.stem.startswith("synthetic_")
                )
                if real_count < args.min_samples:
                    target_levels.append(lv)

        for target_lv in target_levels:
            print(f"  Generating [{channel}] Level {target_lv}...")
            if args.method == "interpolation":
                n = generate_interpolation(
                    channel_dir=ch_dir,
                    target_level=target_lv,
                    source_level_low=0,
                    source_level_high=num_levels - 1,
                    count=args.count,
                    image_size=image_size,
                    seed=args.seed,
                    dry_run=args.dry_run,
                )
            else:
                n = generate_img2img(
                    channel_dir=ch_dir,
                    target_level=target_lv,
                    count=args.count,
                    image_size=image_size,
                    seed=args.seed,
                    dry_run=args.dry_run,
                )
            print(f"    → {n} images {'(dry-run)' if args.dry_run else 'generated'}")
            total_generated += n

    print(f"\n{'='*60}")
    print(f"  Total: {total_generated} images generated")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
