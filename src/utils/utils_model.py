"""
utils/utils_model.py

모델 관련 공용 유틸리티 / Model-related shared utilities.

모델 재현성, backbone 헬퍼, 체크포인트 로드에 관한 함수를 정의한다.
Defines functions for reproducibility, backbone helpers, and checkpoint loading.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ── 경로 상수 / Path constants ─────────────────────────────────────────────────
_UTILS_DIR = Path(__file__).resolve().parent  # src/utils/
_SRC_DIR = _UTILS_DIR.parent  # src/


# ──────────────────────────────────────────────────────────────────────────────
# 시드 설정 / Seed setup
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int, cfg: Optional[dict] = None) -> None:
    """
    재현성을 위한 전역 시드를 설정한다.
    Sets the global random seed for reproducibility.

    - `random`, `numpy`, `torch` 모두 동일한 시드 적용
    - cfg 제공 시 `cuda.deterministic` / `cuda.benchmark` 소비 (SSOT-SD01)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if cfg is not None:
        cuda_cfg = cfg.get("cuda", {})
        torch.backends.cudnn.deterministic = cuda_cfg.get("deterministic", True)
        torch.backends.cudnn.benchmark = cuda_cfg.get("benchmark", False)


# ──────────────────────────────────────────────────────────────────────────────
# Backbone 약어 / Backbone abbreviation
# ──────────────────────────────────────────────────────────────────────────────


def backbone_tag(backbone_name: str) -> str:
    """
    backbone 이름의 파일명용 약어를 반환한다.
    Returns a filename-safe abbreviation for the backbone name.

    Examples:
        "efficientnet_b0" → "effb0"
        "resnet50"        → "res50"
    """
    _MAP = {
        "efficientnet_b0": "effb0",
        "resnet50": "res50",
    }
    return _MAP.get(backbone_name, backbone_name.replace("_", "")[:8])


# ──────────────────────────────────────────────────────────────────────────────
# 모델 빌더 / Model builder
# ──────────────────────────────────────────────────────────────────────────────


def build_model(cfg: dict, checkpoint: Path, device: torch.device) -> nn.Module:
    """
    Phase 2 GrayspotModel을 체크포인트에서 로드한다.
    Loads a Phase 2 GrayspotModel from a checkpoint.

    Args:
        cfg:        config dict
        checkpoint: .pt 파일 경로 / Path to the .pt checkpoint file
        device:     연산 디바이스 / Compute device

    Returns:
        eval 모드로 설정된 GrayspotModel / GrayspotModel set to eval mode
    """
    import sys

    sys.path.insert(0, str(_SRC_DIR))
    from models.grayspot_model import GrayspotModel

    model = GrayspotModel(cfg, phase=2).to(device)
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()
