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


def _cfg_for_ckpt(cfg: dict, ckpt_path: Path) -> dict:
    """
    체크포인트 weight shape에서 hidden_dim / mid_dim / backbone을 역산해
    cfg 복사본을 패치한다. Optuna 튜닝 후 기본 config와 아키텍처가 달라도
    올바르게 모델을 로드할 수 있도록 보장한다.

    Auto-detects hidden_dim / mid_dim / backbone from checkpoint weight shapes
    and patches a copy of cfg. Ensures correct model loading even when the
    architecture differs from the default config (e.g. after Optuna tuning).
    """
    import copy

    _FEATURE_TO_BACKBONE: dict[int, str] = {
        2048: "resnet50",
        1280: "efficientnet_b0",
        1792: "efficientnet_b4",
        1536: "efficientnet_b3",
        1408: "efficientnet_b2",
        512:  "resnet18",
        1024: "densenet121",
    }

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    w0 = state.get("head.net.0.weight")
    if w0 is None:
        return cfg  # 구조를 알 수 없으면 원본 반환 / Return original if structure unknown

    in_features  = int(w0.shape[1])
    first_out    = int(w0.shape[0])
    num_classes  = cfg.get("data", {}).get("num_levels", 6)

    # head 구조 판별 / Head structure detection:
    #
    # 2-layer head (mid_dim=None):
    #   net.0: Linear(in_features, hidden_dim)
    #   net.4: Linear(hidden_dim, num_classes)  ← shape[0] == num_classes
    #
    # 3-layer head (mid_dim 존재):
    #   net.0: Linear(in_features, mid_dim)
    #   net.4: Linear(mid_dim, hidden_dim)      ← shape[0] != num_classes
    #   net.8: Linear(hidden_dim, num_classes)
    w4 = state.get("head.net.4.weight")
    if w4 is None or int(w4.shape[0]) == num_classes:
        # 2-layer head: net.4 가 최종 분류기 → mid_dim 없음
        # 2-layer head: net.4 is the classifier → no mid_dim
        mid_dim    = None
        hidden_dim = first_out
    else:
        # 3-layer head: net.4 가 중간 projection → mid_dim 존재
        # 3-layer head: net.4 is a projection → mid_dim exists
        mid_dim    = first_out
        hidden_dim = int(w4.shape[0])

    patched  = copy.deepcopy(cfg)
    detected = _FEATURE_TO_BACKBONE.get(in_features)
    if detected:
        patched.setdefault("model", {})["backbone"] = detected
        backbone = detected
    else:
        backbone = patched.get("model", {}).get("backbone", "efficientnet_b0")

    heads = patched.setdefault("phase2", {}).setdefault("heads", {})
    if backbone not in heads:
        heads[backbone] = {}
    heads[backbone]["mid_dim"]    = mid_dim
    heads[backbone]["hidden_dim"] = hidden_dim

    return patched


def build_model(cfg: dict, checkpoint: Path, device: torch.device) -> nn.Module:
    """
    Phase 2 GrayspotModel을 체크포인트에서 로드한다.
    체크포인트 weight shape를 역산해 아키텍처를 자동 감지하므로
    Optuna 튜닝 후 hidden_dim이 달라도 올바르게 로드된다.

    Loads a Phase 2 GrayspotModel from a checkpoint.
    Auto-detects architecture from weight shapes so models trained with
    Optuna-tuned hidden_dim values load correctly.

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

    patched_cfg = _cfg_for_ckpt(cfg, Path(checkpoint))
    model = GrayspotModel(patched_cfg, phase=2).to(device)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
