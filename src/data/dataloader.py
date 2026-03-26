"""
Grayspot -- DataLoader 생성 유틸리티 / DataLoader Creation Utility
data/dataloader.py

Phase 0 / Phase 2 각각에 맞는 DataLoader를 생성하는 유틸리티 모듈.
Utility module that creates DataLoaders suited for Phase 0 and Phase 2.
"""

import torch
from torch.utils.data import DataLoader

from data.dataset import ContrastiveDataset, GrayspotDataset, compute_class_weights

CHANNELS = ["Y", "M", "C", "K"]


def get_phase0_loaders(cfg: dict, channel: str) -> DataLoader:
    """
    Phase 0 Contrastive Learning용 DataLoader를 생성한다.
    Creates a DataLoader for Phase 0 Contrastive Learning.

    Args:
        cfg:     config.yaml 딕셔너리 / config.yaml dictionary
        channel: "Y" | "M" | "C" | "K"

    Returns:
        DataLoader -- (view1, view2) 배치를 반환 / Returns (view1, view2) batches

    Example:
        >>> loader = get_phase0_loaders(cfg, "Y")
        >>> for view1, view2 in loader:
        ...     pass
    """
    p       = cfg["phase0"]
    dataset = ContrastiveDataset(cfg, channel)

    if len(dataset) == 0:
        raise ValueError(
            f"[{channel}] Phase 0 데이터 없음 / No data. "
            f"labeled/ 폴더를 확인하세요 / Check the labeled/ folder."
        )

    loader = DataLoader(
        dataset,
        batch_size=p["batch_size"],
        shuffle=True,
        drop_last=True,      # 마지막 불완전 배치 제거 (Contrastive에서 중요) / Drop last incomplete batch (important for contrastive)
        num_workers=0,       # macOS 호환성 / macOS compatibility
        pin_memory=False,    # CPU 모드에서 불필요 / Not needed in CPU mode
    )

    print(f"  [{channel}] Phase 0 DataLoader -- "
          f"{len(dataset)}개 샘플 / samples | "
          f"배치 크기 / batch size: {p['batch_size']} | "
          f"{len(loader)}개 배치 / batches")

    return loader


def get_phase2_loaders(
    cfg: dict,
    channel: str,
    label_csv: str = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Phase 2 Supervised Classification용 Train / Val / Test DataLoader를 생성한다.
    Creates Train / Val / Test DataLoaders for Phase 2 Supervised Classification.

    Args:
        cfg:       config.yaml 딕셔너리 / config.yaml dictionary
        channel:   "Y" | "M" | "C" | "K"
        label_csv: 라벨 CSV 경로 (없으면 폴더 구조 기반) / Label CSV path (folder-based if None)

    Returns:
        (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = get_phase2_loaders(cfg, "Y")
        >>> for x, labels, meta in train_loader:
        ...     pass
    """
    p = cfg["phase2"]

    train_ds = GrayspotDataset(cfg, channel, split="train", label_csv=label_csv, augment=True)
    val_ds   = GrayspotDataset(cfg, channel, split="val",   label_csv=label_csv, augment=False)
    test_ds  = GrayspotDataset(cfg, channel, split="test",  label_csv=label_csv, augment=False)

    if len(train_ds) == 0:
        raise ValueError(
            f"[{channel}] Phase 2 학습 데이터 없음 / No training data. "
            f"labeled/ 폴더를 확인하세요 / Check the labeled/ folder."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=p["batch_size"],
        shuffle=True,
        drop_last=True,   # 마지막 불완전 배치 제거 / Drop last incomplete batch
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=p["batch_size"],
        shuffle=False,    # 검증 시 순서 유지 / Maintain order for validation
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=p["batch_size"],
        shuffle=False,    # 테스트 시 순서 유지 / Maintain order for testing
        num_workers=0,
        pin_memory=False,
    )

    print(f"  [{channel}] Phase 2 DataLoaders -- "
          f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader


def get_class_weights(cfg: dict, channel: str, label_csv: str = None) -> torch.Tensor:
    """
    학습 데이터셋 기반 클래스 불균형 보정 가중치를 반환한다.
    Returns class imbalance correction weights based on the training dataset.

    Args:
        cfg:       config.yaml 딕셔너리 / config.yaml dictionary
        channel:   "Y" | "M" | "C" | "K"
        label_csv: 라벨 CSV 경로 (없으면 폴더 구조 기반) / Label CSV path (folder-based if None)

    Returns:
        torch.Tensor -- 레벨별 가중치 (num_levels,) / Per-level weights

    Example:
        >>> weights = get_class_weights(cfg, "Y")
        >>> criterion = nn.CrossEntropyLoss(weight=weights)
    """
    train_ds = GrayspotDataset(cfg, channel, split="train", label_csv=label_csv, augment=False)

    if len(train_ds) == 0:
        raise ValueError(f"[{channel}] 학습 데이터 없음 / No training data")

    weights = compute_class_weights(train_ds)

    print(f"  [{channel}] 클래스 가중치 / Class weights: "
          + " ".join(f"L{i}:{w:.2f}" for i, w in enumerate(weights.tolist())))

    return weights


def print_loader_summary(cfg: dict) -> None:
    """
    전체 채널의 DataLoader 구성 요약을 출력한다.
    Prints a summary of DataLoader configuration for all channels.
    """
    print("\n" + "=" * 55)
    print("  DataLoader 구성 요약 / DataLoader Summary")
    print("=" * 55)
    labeled_dir = __import__("pathlib").Path(cfg["storage"]["labeled_dir"])
    channels    = cfg["data"]["channels"]
    n_levels    = cfg["data"]["num_levels"]
    exts        = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    for ch in channels:
        total = 0
        for lv in range(n_levels):
            folder = labeled_dir / ch / f"level_{lv}"
            if folder.exists():
                total += sum(1 for p in folder.iterdir() if p.suffix.lower() in exts)
        print(f"  [{ch}] 총 샘플 / Total samples: {total}")

    print(f"\n  배치 크기 / Batch size (Phase 0): {cfg['phase0']['batch_size']}")
    print(f"  배치 크기 / Batch size (Phase 2): {cfg['phase2']['batch_size']}")
    print(f"  이미지 크기 / Image size: {cfg['data']['image_size']} x {cfg['data']['image_size']}")
    print(f"  Split 비율 / Split ratio: "
          f"Train {cfg['data']['train_val_test_split'][0]:.0%} | "
          f"Val {cfg['data']['train_val_test_split'][1]:.0%} | "
          f"Test {cfg['data']['train_val_test_split'][2]:.0%}")
    print()