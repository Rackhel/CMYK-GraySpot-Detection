"""
Grayspot — PyTorch Dataset
data/dataset.py

로컬 스토리지 폴더 구조 기준 / Based on local storage folder structure:
  data/labeled/{channel}/level_{N}/*.png

Phase 0: 라벨 없이 이미지만 로드 (Contrastive Learning)
         Loads images without labels (Contrastive Learning)
Phase 2: 채널별 레벨 라벨 포함 (Supervised Classification)
         Loads images with per-channel level labels (Supervised Classification)
"""

import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

from data.preprocessing import preprocess, CHANNELS
from data.augmentation import get_contrastive_transforms, get_supervised_transforms


# ──────────────────────────────────────────────
# Phase 0 Dataset — Contrastive Learning
# ──────────────────────────────────────────────
class ContrastiveDataset(Dataset):
    """
    Phase 0 비지도 Contrastive Learning용 Dataset.
    라벨 없이 이미지만 로드하며,
    같은 이미지의 두 가지 augmentation을 Positive Pair로 반환한다.

    Phase 0 unsupervised Contrastive Learning Dataset.
    Loads images without labels and returns two augmented views
    of the same image as a Positive Pair.

    폴더 구조 / Folder structure:
        data/labeled/{channel}/level_*/*.png  (라벨 무시 / labels ignored)
    """

    def __init__(self, cfg: dict, channel: str):
        """
        Args:
            cfg:     config.yaml 딕셔너리 / config.yaml dictionary
            channel: "Y" | "M" | "C" | "K"
        """
        assert channel in CHANNELS
        self.cfg       = cfg
        self.channel   = channel
        self.transform = get_contrastive_transforms(cfg)

        # 이미지 경로 수집 (labeled/ 하위 모든 레벨 포함)
        # Collect image paths (all levels under labeled/)
        roi_dir = Path(cfg["storage"]["roi_dir"]) / channel
        self.image_paths = (
            sorted(roi_dir.rglob("*.png")) +
            sorted(roi_dir.rglob("*.tiff")) +
            sorted(roi_dir.rglob("*.jpg"))
        )

        if len(self.image_paths) == 0:
            print(f"  [{channel}] 이미지 없음 / No images found: {roi_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path      = self.image_paths[idx]
        processed = preprocess(path, self.cfg, return_feature=False)
        roi       = processed[self.channel]  # (H, W, 3) float32

        # Positive Pair: 같은 이미지의 두 augmentation / Two augmented views of the same image
        view1 = self.transform(roi)
        view2 = self.transform(roi)
        return view1, view2


# ──────────────────────────────────────────────
# Phase 2 Dataset — Supervised Classification
# ──────────────────────────────────────────────
class GrayspotDataset(Dataset):
    """
    Phase 2 지도학습용 Dataset.
    Phase 2 Supervised Classification Dataset.

    폴더 구조 기반 라벨 자동 추출 / Auto-extracts labels from folder structure:
        data/labeled/{channel}/level_{N}/*.png  →  label = N

    또는 CSV 기반 라벨 로드 / Or loads labels from CSV:
        filename, yellow_level, magenta_level, cyan_level, black_level, confidence
    """

    # CSV 컬럼명과 채널명 매핑 / CSV column name to channel mapping
    CHANNEL_CSV_COL = {
        "Y": "yellow_level",
        "M": "magenta_level",
        "C": "cyan_level",
        "K": "black_level",
    }

    def __init__(
        self,
        cfg: dict,
        channel: str,
        split: str = "train",             # "train" | "val" | "test"
        label_csv: Optional[str] = None,  # CSV 경로 (없으면 폴더 구조로 자동 추출) / CSV path (auto from folder if None)
        augment: bool = True,
    ):
        assert channel in CHANNELS
        assert split in ("train", "val", "test")

        self.cfg       = cfg
        self.channel   = channel
        self.split     = split
        self.augment   = augment and (split == "train")  # train split에서만 증강 / Augment only for train split
        self.transform = get_supervised_transforms(cfg, augment=self.augment)

        if label_csv:
            self.samples = self._load_from_csv(label_csv, channel)
        else:
            self.samples = self._load_from_folder(channel)

        # Train/Val/Test 분할 (session_id 단위 분리, data leakage 방지)
        # Split by session_id to prevent data leakage between splits
        self.samples = self._split_samples(self.samples, split, cfg)

        print(f"  [{channel}] {split}: {len(self.samples)}개 샘플 / samples")

    def _load_from_folder(self, channel: str) -> list[dict]:
        """
        labeled/{channel}/level_{N}/ 폴더 구조에서 자동으로 샘플과 라벨을 수집한다.
        Automatically collects samples and labels from the folder structure.
        """
        labeled_dir = Path(self.cfg["storage"]["labeled_dir"]) / channel
        samples = []
        for level in range(self.cfg["data"]["num_levels"]):
            level_dir = labeled_dir / f"level_{level}"
            if not level_dir.exists():
                continue
            for img_path in sorted(level_dir.glob("*")):
                if img_path.suffix.lower() in {".png", ".tiff", ".tif", ".jpg"}:
                    # 파일명에서 session_id 추출 (규칙: scan_{date}_{session_id}_{seq}.png)
                    # Extract session_id from filename (format: scan_{date}_{session_id}_{seq}.png)
                    session_id = self._parse_session_id(img_path.name)
                    samples.append({
                        "path":       img_path,
                        "label":      level,
                        "channel":    channel,
                        "session_id": session_id,
                        "confidence": "확실",  # 폴더 구조 기반은 기본 확실 / Folder-based default to confirmed
                    })
        return samples

    def _load_from_csv(self, csv_path: str, channel: str) -> list[dict]:
        """
        라벨 CSV에서 샘플을 로드한다.
        Loads samples from a label CSV file.

        CSV 컬럼 / CSV columns:
            filename, yellow_level, magenta_level, cyan_level, black_level, confidence
        """
        col     = self.CHANNEL_CSV_COL[channel]
        raw_dir = Path(self.cfg["storage"]["raw_dir"])
        samples = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label      = int(row[col])
                confidence = row.get("confidence", "확실")
                session_id = self._parse_session_id(row["filename"])
                img_path   = raw_dir / row["filename"]
                if img_path.exists():
                    samples.append({
                        "path":       img_path,
                        "label":      label,
                        "channel":    channel,
                        "session_id": session_id,
                        "confidence": confidence,
                    })
        return samples

    def _parse_session_id(self, filename: str) -> str:
        """
        파일명에서 session_id를 추출한다.
        Extracts session_id from filename.
        규칙 / Format: scan_{date}_{session_id}_{seq}.png
        """
        parts = Path(filename).stem.split("_")
        if len(parts) >= 3:
            return parts[2]
        return "unknown"

    def _split_samples(self, samples: list[dict], split: str, cfg: dict) -> list[dict]:
        """
        session_id 단위로 Train/Val/Test를 분리한다 (data leakage 방지).
        Splits samples by session_id to prevent data leakage across splits.
        """
        ratios      = cfg["data"]["train_val_test_split"]  # [0.70, 0.15, 0.15]
        session_ids = sorted(set(s["session_id"] for s in samples))
        n           = len(session_ids)
        n_train     = int(n * ratios[0])
        n_val       = int(n * ratios[1])

        if split == "train":
            valid_sessions = set(session_ids[:n_train])
        elif split == "val":
            valid_sessions = set(session_ids[n_train:n_train+n_val])
        else:  # test
            valid_sessions = set(session_ids[n_train+n_val:])

        return [s for s in samples if s["session_id"] in valid_sessions]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        sample    = self.samples[idx]
        processed = preprocess(sample["path"], self.cfg, return_feature=True)

        roi     = processed[self.channel]                   # (H, W, 3)
        feature = processed.get(f"{self.channel}_feature")  # (H, W, 3) — 강조 특징 / Enhanced feature

        tensor = self.transform(roi)  # → (3, H, W) torch.Tensor

        # 메타 정보 / Metadata
        meta = {
            "path":       str(sample["path"]),
            "channel":    sample["channel"],
            "session_id": sample["session_id"],
            "confidence": sample["confidence"],
        }
        return tensor, sample["label"], meta


# ──────────────────────────────────────────────
# 유틸: 클래스 가중치 계산 / Class Weight Computation
# ──────────────────────────────────────────────
def compute_class_weights(dataset: GrayspotDataset) -> torch.Tensor:
    """
    클래스 불균형 보정을 위한 가중치를 계산한다.
    Level 5처럼 샘플 수가 적은 클래스에 더 높은 가중치를 부여한다.

    Computes class weights to correct for class imbalance.
    Assigns higher weight to underrepresented classes (e.g., Level 5).
    """
    num_levels = dataset.cfg["data"]["num_levels"]
    counts     = torch.zeros(num_levels)
    for sample in dataset.samples:
        counts[sample["label"]] += 1

    total   = counts.sum()
    weights = total / (num_levels * counts.clamp(min=1))  # 역빈도 가중치 / Inverse frequency weighting
    return weights


# ──────────────────────────────────────────────
# 유틸: 라벨 통계 출력 / Label Statistics Printer
# ──────────────────────────────────────────────
def print_dataset_stats(datasets: dict[str, GrayspotDataset], channel: str) -> None:
    """
    Train/Val/Test 각 split의 레벨 분포를 출력한다.
    Prints the level distribution for each Train/Val/Test split.
    """
    num_levels = list(datasets.values())[0].cfg["data"]["num_levels"]
    print(f"\n📊  Dataset Stats — Channel: {channel}")
    print(f"{'Level':<8}", end="")
    for split in ["train", "val", "test"]:
        print(f"{split:>10}", end="")
    print()
    print("-" * 38)

    for lv in range(num_levels):
        print(f"Level {lv:<2}", end="")
        for ds in datasets.values():
            count = sum(1 for s in ds.samples if s["label"] == lv)
            print(f"{count:>10}", end="")
        print()

    print("-" * 38)
    print(f"{'Total':<8}", end="")
    for ds in datasets.values():
        print(f"{len(ds):>10}", end="")
    print("\n")