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
    data_set/holdout/{channel}/{level}/*.png   ← prepare_holdout.py 로 생성
    data_set/labeled/{channel}/{level}/synthetic_*.png  ← generate_synthetic.py 로 생성
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

if TYPE_CHECKING:
    import pandas as pd

from data.augmentation import augment_contrastive, augment_supervised
from data.normalize import _IMAGENET_NORMALIZE  # SSOT-NM01
from data.preprocessing import preprocess

# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────


def _resolve_per_channel(cfg: dict, channel: str, key: str, default):
    """cfg['phase2']['per_channel'][channel][key] 우선, 없으면 cfg['phase2'][key] fallback."""
    per = cfg.get("phase2", {}).get("per_channel", {}).get(channel, {})
    return per.get(key, cfg.get("phase2", {}).get(key, default))


# ──────────────────────────────────────────────────────────────
# Supervised Dataset / Phase 2
# ──────────────────────────────────────────────────────────────


class CMYKDataset(Dataset):
    """
    Supervised 학습(Phase 2)용 Dataset.
    Dataset for Supervised training (Phase 2).

    split 옵션 / Split options:
        "train" | "val" | "test"  — labeled_dir 안에서 Stratified Split
        "holdout"                 — holdout_dir 전체를 로드 (분할 없음)

    Args:
        cfg:              config.json dict
        channel:          "Y" | "M" | "C" | "K"
        split:            "train" | "val" | "test" | "holdout"
        augment:          증강 적용 여부 (train split 시만 실제 적용)
        oversample:       소수 클래스 오버샘플링 여부
        samples:          외부에서 직접 sample list 주입 (K-Fold 등)
        exclude_synthetic: True이면 synthetic_ 파일명 이미지를 제외
    """

    _EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    def __init__(
        self,
        cfg: dict,
        channel: str,
        split: str = "train",
        augment: bool = True,
        oversample: bool = True,
        samples: list[tuple[Path, int]] | None = None,
        exclude_synthetic: bool = False,
    ):
        self.augment = augment and (split == "train")
        self.image_size = cfg["data"]["image_size"]
        self.num_levels = cfg["data"]["num_levels"]
        self.sup_aug_cfg = cfg["phase2"].get("augmentation", {})
        seed = cfg.get("train", {}).get("seed", 42)
        _rng = random.Random(seed)

        if samples is not None:
            self.samples = samples
            return

        # ── holdout 분기 ──────────────────────────────────────────
        if split == "holdout":
            holdout_dir = Path(
                cfg["storage"].get(
                    "holdout_dir",
                    str(Path(cfg["storage"]["labeled_dir"]).parent / "holdout"),
                )
            )
            self.samples = self._collect(holdout_dir / channel, exclude_synthetic)
            return

        # ── labeled 분기 (train / val / test) ────────────────────
        labeled_dir = Path(cfg["storage"]["labeled_dir"])
        all_samples = self._collect(labeled_dir / channel, exclude_synthetic)

        # Stratified Split — 레벨별 비율 유지 / Maintain level distribution
        level_groups: dict[int, list] = defaultdict(list)
        for sample in all_samples:
            level_groups[sample[1]].append(sample)

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
            and _resolve_per_channel(cfg, channel, "oversample", True)
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
                    level_s.append(_rng.choice(level_s))  # D-2: seeded RNG
                oversampled.extend(level_s)
            self.samples = oversampled

    # ── helpers ──────────────────────────────────────────────────

    def _collect(
        self, channel_dir: Path, exclude_synthetic: bool
    ) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for level in range(self.num_levels):
            level_dir = channel_dir / str(level)
            if not level_dir.exists():
                continue
            for img_path in sorted(level_dir.glob("*")):
                if img_path.suffix.lower() not in self._EXTS:
                    continue
                if exclude_synthetic and img_path.stem.startswith("synthetic_"):
                    continue
                samples.append((img_path, level))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, level = self.samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"이미지 없음 / Image not found: {img_path}")

        # preprocess: BGR→RGB + resize + /255.0 (D-1 fix in preprocessing.py)
        image = preprocess(image, self.image_size)

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
    """

    _EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    def __init__(self, cfg: dict, channel: str):
        self.image_size = cfg["data"]["image_size"]
        self.num_levels = cfg["data"]["num_levels"]
        self.aug_cfg = cfg["phase0"].get("augmentation", {})
        self.image_paths: list[Path] = []

        labeled_dir = Path(cfg["storage"]["labeled_dir"])
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

        image = preprocess(image, self.image_size)  # BGR→RGB included
        view1 = augment_contrastive(image.copy(), self.image_size, self.aug_cfg)
        view2 = augment_contrastive(image.copy(), self.image_size, self.aug_cfg)

        return (
            _IMAGENET_NORMALIZE(torch.tensor(view1).permute(2, 0, 1).float()),
            _IMAGENET_NORMALIZE(torch.tensor(view2).permute(2, 0, 1).float()),
        )


# ──────────────────────────────────────────────────────────────
# Evaluation Dataset / 평가 전용 (Evaluator 내부 전용)
# ──────────────────────────────────────────────────────────────


class _EvalDataset(Dataset):
    """
    평가 전용 Dataset. Evaluator 내부에서만 사용한다.
    Evaluation-only Dataset. Used internally by Evaluator only.
    """

    def __init__(self, df: "pd.DataFrame", patch_dir: Path, image_size: int):
        self.df = df.reset_index(drop=True)
        self.patch_dir = Path(patch_dir)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        color = row["color"]
        fname = row["filename"]
        level = int(row["level"])

        img_path = self.patch_dir / color / str(level) / fname
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found / 이미지 없음: {img_path}")

        img = cv2.imread(str(img_path))
        # D-1: BGR→RGB consistent with training path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0

        tensor = _IMAGENET_NORMALIZE(torch.tensor(img).permute(2, 0, 1).float())
        return tensor, level, fname
