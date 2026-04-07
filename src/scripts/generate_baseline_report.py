"""
scripts/generate_baseline_report.py
====================================
Grayspot Detection Pipeline — Baseline Report CLI Script
Grayspot 탐지 파이프라인 — Baseline 리포트 CLI 스크립트

Runs the full evaluation pipeline on a trained checkpoint (or random weights)
and generates outputs/reports/baseline.html.

학습된 체크포인트(또는 랜덤 가중치)에 대해 전체 평가 파이프라인을 실행하고
outputs/reports/baseline.html을 생성합니다.

Usage / 사용법:
    # Random weights (prototype mode / 프로토타입 모드)
    python scripts/generate_baseline_report.py

    # With checkpoint / 체크포인트 지정
    python scripts/generate_baseline_report.py --checkpoint data_set/models/baseline_C.pt

    # Open in browser after generation / 생성 후 브라우저 열기
    python scripts/generate_baseline_report.py --open-browser

Source notebook : 04_evaluation.ipynb (Cells 0~15, full pipeline)
PRD reference   : Section 8.2 (Reporting), Section 5.6 (Evaluation)
Execution plan  : Stage 2 (W7~W8), Role R3

Python 3.11.5 | macOS & Windows compatible
"""

# ── Standard library / 표준 라이브러리 ────────────────────────────────────
from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

# ── Python 3.11.5 version guard / Python 3.11.5 버전 가드 ────────────────
assert sys.version_info[:2] == (3, 11), (
    f"Python 3.11.x required, got {sys.version}. "
    f"Python 3.11.x가 필요합니다. 현재: {sys.version}"
)

warnings.filterwarnings("ignore")

# ── Project root on sys.path ──────────────────────────────────────────────
# Folder layout / 폴더 구조:
#   CMYK_MAIN/
#     src/
#       evaluation/    ← import target / 임포트 대상
#       reporting/     ← import target / 임포트 대상
#     scripts/
#       generate_baseline_report.py   ← this file / 이 파일
#
# Path chain: scripts/ → CMYK_MAIN/ → src/
# 경로 체인: scripts/ → CMYK_MAIN/ → src/
_SRC  = Path(__file__).parent.parent.resolve()   # src/
_ROOT = _SRC.parent                              # CMYK_MAIN/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Third-party / 서드파티 ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ── Internal / 내부 ───────────────────────────────────────────────────────
from evaluation.evaluator import GrayspotEvaluator, EvaluatorConfig
from reporting.html_report import generate_baseline_report


# ─────────────────────────────────────────────────────────────────────────────
# 0. Device selection — identical to 04_evaluation.ipynb Cell 0-B
#    디바이스 선택 — 04_evaluation.ipynb Cell 0-B와 동일
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Auto-select compute device: CUDA → MPS → CPU.
    컴퓨팅 디바이스 자동 선택: CUDA → MPS → CPU.

    Windows: CUDA → CPU  /  macOS Apple Silicon: MPS → CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon / 애플 실리콘
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset — identical structure to 04_evaluation.ipynb Cell 3
#    데이터셋 — 04_evaluation.ipynb Cell 3과 동일한 구조
# ─────────────────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Color patch image Dataset.
    색상 패치 이미지 Dataset.

    Folder structure / 폴더 구조:
        labeled/{color}/{level}/{filename}

    Identical to 04_evaluation.ipynb PatchDataset.
    04_evaluation.ipynb PatchDataset과 동일합니다.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        patch_dir: Path,
        transform,
        image_size: int = 224,
    ) -> None:
        self.df         = df.reset_index(drop=True)
        self.patch_dir  = Path(patch_dir)
        self.transform  = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        color = row["color"]
        fname = row["filename"]
        level = row["level"]

        # Path: labeled/{color}/{level}/{filename}
        img_path = self.patch_dir / color / str(level) / fname

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            # Dummy image for pipeline validation / 파이프라인 검증용 더미 이미지
            arr = np.random.randint(100, 200, (self.image_size, self.image_size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)

        return self.transform(img), int(level), fname   # tensor, label, filename


# ─────────────────────────────────────────────────────────────────────────────
# 2. Label loading — identical to 04_evaluation.ipynb Cell 2
#    라벨 로드 — 04_evaluation.ipynb Cell 2와 동일
# ─────────────────────────────────────────────────────────────────────────────

_COLOR_COLUMNS = {"Y": "Y", "M": "M", "C": "C", "K": "K"}

def load_labels(
    labels_csv: Path,
    color_columns: dict[str, str] = _COLOR_COLUMNS,
) -> pd.DataFrame:
    """
    Load label CSV and convert to long-format DataFrame.
    라벨 CSV를 로드하여 long-format DataFrame으로 변환합니다.

    Mirrors load_labels() from 04_evaluation.ipynb Cell 2.
    04_evaluation.ipynb Cell 2의 load_labels()를 반영합니다.

    Wide format (input): filename, C, M, Y, K
    Long format (output): filename, color, level, confidence
    """
    df = pd.read_csv(labels_csv)
    print(f"[CSV 로드 / Loaded] rows={len(df)}, columns={list(df.columns)}")

    records = []
    for _, row in df.iterrows():
        for color_code, col_name in color_columns.items():
            if col_name not in df.columns:
                continue
            records.append({
                "filename":   row["filename"],
                "color":      color_code,
                "level":      int(row[col_name]),
                "confidence": row.get("confidence", "확실"),
            })

    long_df = pd.DataFrame(records)
    print(f"[변환 완료 / Converted] long-format rows={len(long_df)}")
    return long_df


def create_dummy_labels(
    channels: list[str],
    num_levels: int,
    n_per_class: int = 20,
) -> pd.DataFrame:
    """
    Generate dummy labels when real CSV is not available.
    실제 CSV가 없을 때 더미 라벨을 생성합니다.

    Mirrors create_dummy_labels() from 04_evaluation.ipynb Cell 2.
    04_evaluation.ipynb Cell 2의 create_dummy_labels()를 반영합니다.
    """
    print("[더미 데이터 / Dummy] 실제 CSV 없음 → 더미 생성 / CSV not found → generating dummy")
    records = []
    idx = 0
    for c in channels:
        for lv in range(num_levels):
            for _ in range(n_per_class):
                records.append({
                    "filename":   f"scan_dummy_{idx:04d}.png",
                    "color":      c,
                    "level":      lv,
                    "confidence": "확실" if np.random.rand() > 0.2 else "불확실",
                })
                idx += 1
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model builder — identical to 04_evaluation.ipynb Cell 4
#    모델 빌더 — 04_evaluation.ipynb Cell 4와 동일
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier(
    backbone_name: str,
    num_classes: int,
    checkpoint: Path | None,
    device: torch.device,
) -> nn.Module:
    """
    Build and load a Phase 2 classifier.
    Phase 2 분류기를 생성하고 로드합니다.

    Mirrors build_classifier() from 04_evaluation.ipynb Cell 4.
    04_evaluation.ipynb Cell 4의 build_classifier()를 반영합니다.
    """
    if backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == "resnet34":
        model = models.resnet34(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"지원하지 않는 backbone / Unsupported backbone: {backbone_name}")

    if checkpoint is not None and Path(str(checkpoint)).exists():
        # map_location ensures Windows/macOS CPU-safe loading
        # map_location으로 Windows/macOS CPU 안전 로드 보장
        state = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"✅  체크포인트 로드 / Checkpoint loaded: {checkpoint}")
    else:
        if checkpoint is not None:
            print(f"⚠️   체크포인트 없음 → 랜덤 가중치 / Not found → random weights: {checkpoint}")
        else:
            print("ℹ️   체크포인트 미설정 → 랜덤 가중치 (프로토타입 모드 / prototype mode)")

    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inference runner — identical to 04_evaluation.ipynb Cell 5
#    추론 실행기 — 04_evaluation.ipynb Cell 5와 동일
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Run model inference over the entire DataLoader.
    전체 DataLoader에 대해 모델 추론을 실행합니다.

    Mirrors run_inference() from 04_evaluation.ipynb Cell 5.
    04_evaluation.ipynb Cell 5의 run_inference()를 반영합니다.

    Returns:
        y_true, y_pred, confidences, filenames
    """
    model.eval()
    all_true, all_pred, all_conf, all_fnames = [], [], [], []

    for batch_imgs, batch_labels, batch_fnames in loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        logits     = model(batch_imgs)
        probs      = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        all_true.extend(batch_labels.numpy())
        all_pred.extend(pred.cpu().numpy())
        all_conf.extend(conf.cpu().numpy())
        all_fnames.extend(batch_fnames)

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_conf),
        all_fnames,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main pipeline
#    메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """
    Full evaluation → baseline.html pipeline.
    전체 평가 → baseline.html 파이프라인.
    """
    # ── Path resolution / 경로 설정 ──────────────────────────────────────
    # CMYK_MAIN/ 기준 경로 설정 / Path resolution from CMYK_MAIN/
    #   CMYK_MAIN/                  <- root_dir
    #     data_set/labeled/         <- labeled_dir
    #     data_set/labels_v0.csv    <- labels_csv
    #     outputs/reports/baseline.html  <- output_path (auto-created)
    root_dir    = Path(args.root_dir).resolve()          # CMYK_MAIN/
    labeled_dir = root_dir / "data_set" / "labeled"      # data_set/labeled/
    labels_csv  = root_dir / "data_set" / "labels_v0.csv"
    output_path = root_dir / "outputs" / "reports" / "baseline.html"

    channels   = ["Y", "M", "C", "K"]
    num_levels = 6
    image_size = 224
    batch_size = 32

    # ── Random seed / 랜덤 시드 ──────────────────────────────────────────
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Device / 디바이스 ────────────────────────────────────────────────
    device = get_device()
    print(f"⚙️  Device / 디바이스: {device}")

    # ── Labels / 라벨 ────────────────────────────────────────────────────
    if labels_csv.exists():
        df_labels = load_labels(labels_csv)
    else:
        df_labels = create_dummy_labels(channels, num_levels, n_per_class=20)

    # ── Transform (same as 04_evaluation.ipynb) / 전처리 변환 ─────────────
    # PRD §6.8.4: single-channel → 3-channel copy / 단일채널 → 3채널 복제
    inference_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    # ── Model / 모델 ─────────────────────────────────────────────────────
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    model = build_classifier(
        backbone_name=args.backbone,
        num_classes=num_levels,
        checkpoint=checkpoint,
        device=device,
    )

    # ── Inference per channel / 채널별 추론 ──────────────────────────────
    results: dict[str, dict] = {}
    for color in channels:
        df_color = df_labels[df_labels["color"] == color].reset_index(drop=True)
        ds = PatchDataset(df_color, labeled_dir, inference_transform, image_size)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,       # 0 = single-process, safe on both OS
                                 # 0 = 단일 프로세스, 양 OS에서 안전
            pin_memory=(device.type == "cuda"),
        )
        y_true, y_pred, confs, fnames = run_inference(model, loader, device)
        results[color] = {
            "y_true":      y_true,
            "y_pred":      y_pred,
            "confidences": confs,
            "filenames":   fnames,
        }
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred)
        print(f"  [{color}] {len(y_true):5,} samples | Accuracy: {acc:.4f}")

    # ── Evaluate / 평가 ───────────────────────────────────────────────────
    cfg = EvaluatorConfig(
        output_dir=output_path.parent,   # outputs/reports/
        channels=channels,
        num_levels=num_levels,
        backbone_name=args.backbone,
        checkpoint=str(checkpoint) if checkpoint else None,
        open_browser=False,              # HTML report handles browser opening
                                         # HTML 리포트가 브라우저 열기를 처리
    )
    evaluator = GrayspotEvaluator(cfg)
    summary   = evaluator.run(results, meta={
        "backbone":   args.backbone,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "n_samples":  int(sum(len(results[c]["y_true"]) for c in channels)),
    })

    # ── Generate baseline.html / baseline.html 생성 ───────────────────────
    print("\n📄 Generating baseline.html / baseline.html 생성 중...")
    generate_baseline_report(
        summary=summary,
        results=results,
        output_path=output_path,
        channels=channels,
        open_browser=args.open_browser,
    )

    # ── Output checklist / 출력물 체크리스트 ─────────────────────────────
    evaluator.print_output_checklist()
    print(f"\n✅ 완료 / Done → {output_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI argument parser
#    CLI 인수 파서
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    명령줄 인수를 파싱합니다.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate outputs/reports/baseline.html — "
            "Grayspot Stage 2 Baseline Evaluation Report\n"
            "Grayspot Stage 2 Baseline 평가 리포트 생성"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent.resolve()),  # CMYK_MAIN/
        help=(
            "Project root directory (default: script's parent parent).\n"
            "프로젝트 루트 디렉토리 (기본값: 스크립트의 상위 상위 디렉토리)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to Phase 2 checkpoint .pt file.\n"
            "None → random weights (prototype mode).\n"
            "Phase 2 체크포인트 .pt 파일 경로.\n"
            "None → 랜덤 가중치 (프로토타입 모드)."
        ),
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet18", "resnet34"],
        help="Model backbone (default: efficientnet_b0).",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help=(
            "Open baseline.html in the system default browser after generation.\n"
            "생성 후 시스템 기본 브라우저에서 baseline.html 열기."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
