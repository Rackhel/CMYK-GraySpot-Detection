"""
scripts/tsne_review.py

t-SNE 임베딩 시각화 및 의심 샘플 추출 스크립트.
t-SNE embedding visualization and suspicious sample extraction script.

목적 / Purpose:
    라벨 CSV를 기반으로 패치 이미지를 로드하고, t-SNE로 2D 임베딩을 시각화한다.
    중심에서 가장 멀리 떨어진 상위 20개 샘플을 우선 검토 대상으로 추출한다.

    Loads patch images from a label CSV, visualizes 2D t-SNE embeddings,
    and extracts the top 20 most distant samples as priority review candidates.

SSOT 근거 / SSOT Reference:
    - SSOT-CS01: cv2.imread() 사용 — BGR 색상 공간 유지
    - SSOT-SD01: cfg["train"]["seed"] 사용 — t-SNE 재현성 보장
    - SSOT-CF01: load_config() 통해 경로·파라미터 주입

입력 / Input:
    - cfg["storage"]["labeled_dir"] 내 채널/레벨 폴더 구조
    - CSV: filename, C, M, Y, K 컬럼 포함

출력 / Outputs:
    outputs/reports/tsne_plot.png               <- t-SNE 시각화 이미지
    outputs/reports/priority_review_samples.csv <- 우선 검토 샘플 목록

실행 / Run:
    python -m src.scripts.tsne_review
    python -m src.scripts.tsne_review --csv data_set/labels_v0.csv --channel C
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    from utils.utils_config import load_config
except ImportError:
    from src.utils.utils_config import load_config


# ──────────────────────────────────────────────────────────────────────────────
# 상수 / Constants
# ──────────────────────────────────────────────────────────────────────────────

_OUTPUT_DIR = Path("outputs/reports")
_TOP_N_OUTLIERS = 20


# ──────────────────────────────────────────────────────────────────────────────
# 핵심 함수 / Core functions
# ──────────────────────────────────────────────────────────────────────────────


def load_images_from_csv(
    csv_path: Path,
    labeled_dir: Path,
    image_size: int,
    channel: str | None = None,
) -> tuple[np.ndarray, list[int], list[str]]:
    """
    CSV에서 이미지·레벨·파일명을 로드한다.
    Loads images, levels, and filenames from a label CSV.

    SSOT-CS01 준수: cv2.imread() 사용 — BGR 색상 공간 유지.
    SSOT-CS01 compliant: uses cv2.imread() — BGR color space maintained.

    Args:
        csv_path:    라벨 CSV 경로 / Label CSV path
        labeled_dir: data_set/labeled/ 루트 / labeled root directory
        image_size:  이미지 리사이즈 크기 (cfg["data"]["image_size"]) / Resize target
        channel:     특정 채널만 로드 (None이면 전체) / Channel filter (None = all)

    Returns:
        (images array, labels list, filenames list)
    """
    df = pd.read_csv(csv_path)

    images: list[np.ndarray] = []
    labels: list[int] = []
    filenames: list[str] = []

    for _, row in df.iterrows():
        filename = str(row["filename"])

        # 레벨: CMYK 채널 중 최댓값 / Level: max across CMYK columns
        level = int(max(row["C"], row["M"], row["Y"], row["K"]))

        # 채널 필터 / Channel filter
        if channel is not None:
            row_channel = str(row.get("channel", "")).upper()
            if row_channel and row_channel != channel.upper():
                continue

        # 파일 검색: labeled_dir/{level}/{filename} 구조
        # File search: labeled_dir/{level}/{filename} structure
        img_path: Path | None = None
        for lvl in range(6):
            candidate = labeled_dir / str(lvl) / filename
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        # SSOT-CS01: cv2.imread() → BGR 유지 (PIL RGB 변환 금지)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        try:
            # cfg["data"]["image_size"] 기준 리사이즈 / Resize per SSOT image_size
            img_resized = cv2.resize(img, (image_size, image_size))
            img_array = img_resized.flatten().astype(np.float32) / 255.0
            images.append(img_array)
            labels.append(level)
            filenames.append(filename)
        except (cv2.error, ValueError) as exc:
            print(f"[WARN] 이미지 처리 실패 / Failed to process: {filename} — {exc}")
            continue

    return np.array(images), labels, filenames


def run_tsne(images: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    t-SNE 2D 임베딩을 계산한다.
    Computes 2D t-SNE embedding.

    Args:
        images: (N, D) float32 배열 / float32 array
        seed:   재현성 시드 (SSOT-SD01) / Reproducibility seed

    Returns:
        (N, 2) t-SNE 좌표 배열 / t-SNE coordinate array
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
    return tsne.fit_transform(images)


def save_tsne_plot(
    x_tsne: np.ndarray,
    labels: list[int],
    output_path: Path,
) -> None:
    """
    t-SNE 산점도를 PNG로 저장한다.
    Saves t-SNE scatter plot as PNG.

    Args:
        x_tsne:      (N, 2) 좌표 / Coordinates
        labels:      레벨 리스트 / Level list
        output_path: 저장 경로 / Save path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, cmap="viridis")
    plt.colorbar(scatter, ax=ax, label="Level")
    ax.set_title("t-SNE Embedding Visualization")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ t-SNE 플롯 저장 / Plot saved: {output_path}")


def extract_priority_samples(
    x_tsne: np.ndarray,
    labels: list[int],
    filenames: list[str],
    top_n: int = _TOP_N_OUTLIERS,
) -> pd.DataFrame:
    """
    중심에서 가장 멀리 떨어진 상위 N개 샘플을 추출한다.
    Extracts the top N samples farthest from the centroid.

    Args:
        x_tsne:    t-SNE 좌표 / Coordinates
        labels:    레벨 리스트 / Level list
        filenames: 파일명 리스트 / Filename list
        top_n:     추출할 샘플 수 / Number of samples to extract

    Returns:
        우선 검토 샘플 DataFrame / Priority review samples DataFrame
    """
    center = np.mean(x_tsne, axis=0)
    distances = np.linalg.norm(x_tsne - center, axis=1)
    top_idx = np.argsort(distances)[-top_n:]

    rows = [
        {
            "filename": filenames[idx],
            "label": labels[idx],
            "distance": round(float(distances[idx]), 2),
        }
        for idx in top_idx
    ]
    return pd.DataFrame(rows, columns=["filename", "label", "distance"])


def run_tsne_review(
    csv_path: Path,
    labeled_dir: Path,
    image_size: int,
    seed: int = 42,
    channel: str | None = None,
    output_dir: Path = _OUTPUT_DIR,
) -> None:
    """
    t-SNE 리뷰 파이프라인 전체를 실행한다.
    Runs the full t-SNE review pipeline.

    Args:
        csv_path:    라벨 CSV 경로 / Label CSV path
        labeled_dir: data_set/labeled/ 경로 / labeled directory path
        image_size:  이미지 크기 (cfg["data"]["image_size"]) / Image size
        seed:        시드 / Random seed (SSOT-SD01)
        channel:     채널 필터 / Channel filter (None = all)
        output_dir:  산출물 저장 경로 / Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 이미지 로드 / Load images
    print(f"[1/4] 이미지 로드 중 / Loading images from: {csv_path}")
    images, labels, filenames = load_images_from_csv(
        csv_path, labeled_dir, image_size, channel
    )
    print(f"      로드 완료 / Loaded: {len(images)} images")

    if len(images) == 0:
        raise ValueError(
            "로드된 이미지가 없습니다. 경로와 CSV를 확인하세요. / No images loaded. Check path and CSV."
        )

    # 2. t-SNE 실행 / Run t-SNE
    print("[2/4] t-SNE 계산 중 / Computing t-SNE...")
    x_tsne = run_tsne(images, seed=seed)

    # 3. 플롯 저장 / Save plot
    print("[3/4] 플롯 저장 중 / Saving plot...")
    save_tsne_plot(x_tsne, labels, output_dir / "tsne_plot.png")

    # 4. 우선 검토 샘플 추출 및 저장 / Extract and save priority samples
    print("[4/4] 우선 검토 샘플 추출 중 / Extracting priority review samples...")
    priority_df = extract_priority_samples(x_tsne, labels, filenames)
    csv_out = output_dir / "priority_review_samples.csv"
    priority_df.to_csv(csv_out, index=False)
    print(f"✓ 우선 검토 샘플 저장 / Saved: {csv_out}")
    print("\nTOP PRIORITY REVIEW SAMPLES:")
    print(priority_df.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# 진입점 / Entry point
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE embedding review")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="라벨 CSV 경로 / Label CSV path",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        choices=["Y", "M", "C", "K"],
        help="특정 채널만 처리 / Process specific channel only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help=f"산출물 저장 경로 / Output directory (default: {_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config()

    labeled_dir = Path(cfg["storage"]["labeled_dir"])
    image_size = cfg["data"]["image_size"]
    seed = cfg.get("train", {}).get("seed", 42)

    # CSV 기본값: labels_cmyk.csv → labels_v0.csv 순으로 탐색
    # Default CSV: search labels_cmyk.csv → labels_v0.csv in order
    csv_path = args.csv
    if csv_path is None:
        for candidate in ["data_set/labels_cmyk.csv", "data_set/labels_v0.csv"]:
            if Path(candidate).exists():
                csv_path = Path(candidate)
                break
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            "[SSOT-FF01] 라벨 CSV를 찾을 수 없습니다. --csv 옵션으로 경로를 지정하세요. / "
            "Label CSV not found. Specify path with --csv option."
        )

    run_tsne_review(
        csv_path=csv_path,
        labeled_dir=labeled_dir,
        image_size=image_size,
        seed=seed,
        channel=args.channel,
        output_dir=args.output_dir,
    )
