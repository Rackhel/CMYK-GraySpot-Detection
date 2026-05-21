"""
data/dataset.py

Grayspot 학습·평가용 PyTorch Dataset 클래스.
PyTorch Dataset classes for Grayspot training and evaluation.

세 가지 Dataset / Three Dataset classes:
    - CMYKDataset       : Supervised 학습(Phase 2)용 — 폴더 구조 기반 라벨 로드
    - ContrastiveDataset: Contrastive Learning(Phase 0)용 — 라벨 없이 Positive Pair 반환
    - _EvalDataset      : 평가 전용 — long-format DataFrame 기반, Evaluator 내부 전용

폴더 구조 / Folder structure:
    data_set/labeled/{channel}/{level}/*.png
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# pandas는 평가 시에만 필요 — 학습 경로에서 불필요한 pyarrow 의존을 피한다.
# pandas is only needed for evaluation — TYPE_CHECKING avoids importing it on the training path.
if TYPE_CHECKING:
    import pandas as pd

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

        # SSOT-SD01: 시드된 Random 인스턴스를 사용하여 재현성을 보장한다.
        # Use a seeded Random instance to guarantee reproducibility (SSOT-SD01).
        seed = cfg.get("train", {}).get("seed", 42)
        _rng = random.Random(seed)

        ratios = cfg["data"]["split_ratios"]
        train_s, val_s, test_s = [], [], []
        for lv, items in level_groups.items():
            _rng.shuffle(items)
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


# ──────────────────────────────────────────────────────────────
# Evaluation Dataset / 평가 전용 Dataset (Evaluator 내부 전용)
# ──────────────────────────────────────────────────────────────


class _EvalDataset(Dataset):
    """
    평가 전용 Dataset. Evaluator(evaluation/evaluator_inference.py) 내부에서만 사용한다.
    Evaluation-only Dataset. Used internally by Evaluator (evaluator_inference.py) only.

    CMYKDataset과의 차이 / Difference from CMYKDataset:
        CMYKDataset  : cfg 기반, 폴더 스캔, Stratified Split + Oversampling 포함
        _EvalDataset : long-format DataFrame 기반, 계층적 경로(labeled/{color}/{level}/) 직접 참조,
                       분할·오버샘플링 없음 — 전체 라벨 집합을 그대로 평가

    Args:
        df         : load_labels()가 반환한 long-format DataFrame
                     Long-format DataFrame returned by InferenceMixin.load_labels()
                     columns: ["filename", "color", "level"]
        patch_dir  : data_set/labeled/ 루트 경로 / Root path of data_set/labeled/
        image_size : 리사이즈 목표 크기 (정수) / Target resize size (int)
    """

    def __init__(self, df: pd.DataFrame, patch_dir: Path, image_size: int):
        self.df = df.reset_index(drop=True)
        self.patch_dir = Path(patch_dir)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Returns:
            tensor : (3, H, W) float32 — ImageNet 정규화 완료 / ImageNet-normalized
            level  : int — 정답 레벨 / Ground-truth level
            fname  : str — 파일명 / Filename
        """
        row = self.df.iloc[idx]
        color = row["color"]
        fname = row["filename"]
        level = int(row["level"])

        img_path = self.patch_dir / color / str(level) / fname

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found / 이미지 없음: {img_path}"
            )

        img = cv2.imread(str(img_path))
        # SSOT-CS01: BGR 유지 — RGB 변환 금지 / Keep BGR — no RGB conversion
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0

        tensor = torch.tensor(img).permute(2, 0, 1).float()
        # SSOT-NM01: ImageNet 정규화 적용 — 학습(CMYKDataset)과 동일한 변환
        # Apply ImageNet normalization — identical to training transform (CMYKDataset)
        tensor = _IMAGENET_NORMALIZE(tensor)
        return tensor, level, fname
