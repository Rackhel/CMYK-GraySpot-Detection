"""
data/dataset.py

Grayspot 학습용 PyTorch Dataset 클래스.
PyTorch Dataset classes for Grayspot training.

두 가지 Dataset / Two Dataset classes:
    - CMYKDataset       : Supervised 학습(Phase 2)용 — 폴더 구조 기반 라벨 로드
    - ContrastiveDataset: Contrastive Learning(Phase 0)용 — 라벨 없이 Positive Pair 반환

폴더 구조 / Folder structure:
    data_set/labeled/{channel}/{level}/*.png
"""

import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from data.augmentation import augment_contrastive, augment_supervised
from data.normalize import _IMAGENET_NORMALIZE  # SSOT-NM01: 단일 출처 사용
from data.preprocessing import preprocess

# ──────────────────────────────────────────────────────────────
# Supervised Dataset / Phase 2
# ──────────────────────────────────────────────────────────────


class CMYKDataset(Dataset):
    """
    Supervised 학습(Phase 2)용 Dataset.
    Dataset for Supervised training (Phase 2).

    폴더 구조에서 레벨별 이미지를 로드하고 Stratified Split + Oversampling을 적용한다.
    Loads level-labeled images from folder structure,
    applies Stratified Split and optional Oversampling.

    Args:
        cfg:       config.json dict
        channel:   "Y" | "M" | "C" | "K"
        split:     "train" | "val" | "test"
        augment:   증강 적용 여부 — train split에서만 활성화
                   Whether to apply augmentation — activated only for train split
        oversample: 소수 클래스 오버샘플링 여부 / Whether to oversample minority classes
    """

    _EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    def __init__(
        self,
        cfg: dict,
        channel: str,
        split: str = "train",
        augment: bool = True,
        oversample: bool = True,
    ):
        self.augment = augment and (split == "train")
        self.image_size = cfg["data"]["image_size"]
        self.num_levels = cfg["data"]["num_levels"]
        self.sup_aug_cfg = cfg["phase2"].get("augmentation", {})

        labeled_dir = Path(cfg["storage"]["labeled_dir"])
        all_samples: list[tuple[Path, int]] = []

        # 레벨별 이미지 수집 / Collect images per level
        channel_dir = labeled_dir / channel
        for level in range(self.num_levels):
            level_dir = channel_dir / str(level)
            if not level_dir.exists():
                continue
            for img_path in sorted(level_dir.glob("*")):
                if img_path.suffix.lower() in self._EXTS:
                    all_samples.append((img_path, level))

        # Stratified Split — 레벨별 비율 유지 / Maintain level distribution per split
        level_groups: dict[int, list] = defaultdict(list)
        for sample in all_samples:
            level_groups[sample[1]].append(sample)

        ratios = cfg["data"]["split_ratios"]
        train_s, val_s, test_s = [], [], []
        for lv, items in level_groups.items():
            random.shuffle(items)
            n = len(items)
            n_train = max(1, int(n * ratios["train"]))
            n_val = max(1, int(n * ratios["val"]))
            train_s.extend(items[:n_train])
            val_s.extend(items[n_train : n_train + n_val])
            test_s.extend(items[n_train + n_val :])

        split_map = {"train": train_s, "val": val_s, "test": test_s}
        self.samples: list[tuple[Path, int]] = split_map[split]

        # 소수 클래스 오버샘플링 / Minority class oversampling
        if (
            split == "train"
            and oversample
            and cfg["phase2"].get("oversample", True)
            and self.samples
        ):
            level_counts = Counter(lv for _, lv in self.samples)
            max_count = max(level_counts.values())
            oversampled: list[tuple[Path, int]] = []
            for level in range(self.num_levels):
                level_s = [(p, lv) for p, lv in self.samples if lv == level]
                if not level_s:
                    continue
                while len(level_s) < max_count:
                    level_s.append(random.choice(level_s))
                oversampled.extend(level_s)
            self.samples = oversampled

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, level = self.samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"이미지 없음 / Image not found: {img_path}")

        # 전처리 / Preprocessing
        image = preprocess(image, self.image_size)

        # 증강 (train only) / Augmentation (train only)
        if self.augment:
            image = augment_supervised(image, self.sup_aug_cfg)

        tensor = torch.tensor(image).permute(2, 0, 1).float()
        return _IMAGENET_NORMALIZE(tensor), level


# ──────────────────────────────────────────────────────────────
# Contrastive Dataset / Phase 0
# ──────────────────────────────────────────────────────────────


class ContrastiveDataset(Dataset):
    """
    Contrastive Learning(Phase 0, SimCLR)용 Dataset.
    Dataset for Contrastive Learning (Phase 0, SimCLR).

    라벨 없이 동일 이미지의 두 가지 증강 뷰(Positive Pair)를 반환한다.
    Returns two differently-augmented views (Positive Pair) of the same image, without labels.

    Args:
        cfg:     config.json dict
        channel: "Y" | "M" | "C" | "K"
    """

    _EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    def __init__(self, cfg: dict, channel: str):
        self.image_size = cfg["data"]["image_size"]
        self.num_levels = cfg["data"]["num_levels"]
        self.aug_cfg = cfg["phase0"].get("augmentation", {})
        self.image_paths: list[Path] = []

        labeled_dir = Path(cfg["storage"]["labeled_dir"])

        # 레벨 폴더 전체에서 이미지 수집 (라벨 불필요)
        # Collect images from all level folders (labels not needed)
        channel_dir = labeled_dir / channel
        for level in range(self.num_levels):
            level_dir = channel_dir / str(level)
            if not level_dir.exists():
                continue
            for img_path in sorted(level_dir.glob("*")):
                if img_path.suffix.lower() in self._EXTS:
                    self.image_paths.append(img_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise ValueError(f"이미지 없음 / Image not found: {self.image_paths[idx]}")

        # 전처리 후 동일 이미지에 서로 다른 증강 2회 적용 → Positive Pair
        # Preprocess then apply two independent augmentations → Positive Pair
        image = preprocess(image, self.image_size)
        view1 = augment_contrastive(image.copy(), self.image_size, self.aug_cfg)
        view2 = augment_contrastive(image.copy(), self.image_size, self.aug_cfg)

        return (
            _IMAGENET_NORMALIZE(torch.tensor(view1).permute(2, 0, 1).float()),
            _IMAGENET_NORMALIZE(torch.tensor(view2).permute(2, 0, 1).float()),
        )
