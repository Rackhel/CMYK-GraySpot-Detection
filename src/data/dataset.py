"""
data/dataset.py

Grayspot 학습에 사용하는 PyTorch Dataset 정의.
PyTorch Dataset definitions used for Grayspot training.

포함된 Dataset 종류 / Included dataset classes:
    - CMYKDataset        : 지도학습(Phase 2)용 Dataset
    - ContrastiveDataset : Contrastive Learning(Phase 0)용 Dataset

기본 폴더 구조 / Expected folder structure:
    data_set/labeled/{channel}/{level}/*.png
"""

import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from data.augmentation import augment_contrastive, augment_supervised
from data.preprocessing import preprocess

# ──────────────────────────────────────────────────────────────
# Supervised Dataset
# ──────────────────────────────────────────────────────────────


class CMYKDataset(Dataset):
    """
    지도학습 단계에서 사용하는 Dataset 클래스.
    Dataset class for supervised learning.

    채널 및 레벨 기반 폴더 구조에서 이미지를 읽어온다.
    Stratified split과 optional oversampling을 적용한다.

    Args:
        cfg: 설정 정보(config.yaml)
        channel: "Y" | "M" | "C" | "K"
        split: "train" | "val" | "test"
        augment: train 데이터 증강 여부
        oversample: 클래스 불균형 보정 여부
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

        labeled_dir = Path(cfg["storage"]["labeled_dir"])
        all_samples: list[tuple[Path, int]] = []

        # 레벨별 이미지 경로 수집 / Collect image paths by level
        channel_dir = labeled_dir / channel

        for level in range(self.num_levels):
            level_dir = channel_dir / str(level)

            if not level_dir.exists():
                continue

            for img_path in sorted(level_dir.glob("*")):
                if img_path.suffix.lower() in self._EXTS:
                    all_samples.append((img_path, level))

        # 레벨 비율 유지 분할 / Stratified split
        level_groups: dict[int, list] = defaultdict(list)

        for sample in all_samples:
            level_groups[sample[1]].append(sample)

        train_s, val_s, test_s = [], [], []

        for lv, items in level_groups.items():
            random.shuffle(items)

            n = len(items)
            n_train = max(1, int(n * 0.70))
            n_val = max(1, int(n * 0.15))

            train_s.extend(items[:n_train])
            val_s.extend(items[n_train : n_train + n_val])
            test_s.extend(items[n_train + n_val :])

        split_map = {
            "train": train_s,
            "val": val_s,
            "test": test_s,
        }

        self.samples: list[tuple[Path, int]] = split_map[split]

        # 클래스 균형 보정 / Minority class oversampling
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
            raise ValueError(
                f"이미지를 불러올 수 없음 / Failed to load image: {img_path}"
            )

        # 이미지 전처리 / Image preprocessing
        image = preprocess(image, self.image_size)

        # 학습 시 증강 적용 / Apply augmentation during training
        if self.augment:
            image = augment_supervised(image)

        return torch.tensor(image).permute(2, 0, 1).float(), level


# ──────────────────────────────────────────────────────────────
# Contrastive Dataset
# ──────────────────────────────────────────────────────────────


class ContrastiveDataset(Dataset):
    """
    Contrastive Learning(SimCLR)용 Dataset 클래스.
    Dataset class for contrastive learning.

    동일 이미지에서 서로 다른 두 개의 augmented view를 생성한다.
    Labels are not required in this dataset.

    Args:
        cfg: 설정 정보(config.yaml)
        channel: "Y" | "M" | "C" | "K"
    """

    _EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    def __init__(self, cfg: dict, channel: str):
        self.image_size = cfg["data"]["image_size"]
        self.num_levels = cfg["data"]["num_levels"]

        self.image_paths: list[Path] = []

        labeled_dir = Path(cfg["storage"]["labeled_dir"])

        # 전체 레벨 폴더 탐색 / Scan all level folders
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
            raise ValueError(
                f"이미지를 불러올 수 없음 / Failed to load image: {self.image_paths[idx]}"
            )

        # 전처리 후 서로 다른 augmentation 적용
        # Apply two independent augmentations after preprocessing
        image = preprocess(image, self.image_size)

        view1 = augment_contrastive(
            image.copy(),
            self.image_size,
        )

        view2 = augment_contrastive(
            image.copy(),
            self.image_size,
        )

        return (
            torch.tensor(view1).permute(2, 0, 1).float(),
            torch.tensor(view2).permute(2, 0, 1).float(),
        )
