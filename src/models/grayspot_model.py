"""
Grayspot — 전체 모델 (Phase 0 / Phase 2 모드 전환)
models/grayspot_model.py

Phase 0: Backbone + Projection MLP Head (Contrastive Learning)
Phase 2: Backbone (Phase 0 weights 로드) + Classifier MLP Head (Supervised)

Phase 0: Backbone + Projection MLP Head (Contrastive Learning)
Phase 2: Backbone (loads Phase 0 weights) + Classifier MLP Head (Supervised)
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


# ──────────────────────────────────────────────
# Backbone 생성 / Backbone Builder
# ──────────────────────────────────────────────
def build_backbone(cfg: dict) -> tuple[nn.Module, int]:
    """
    Pretrained CNN Backbone을 생성한다.
    Builds a pretrained CNN backbone.

    Returns:
        (backbone_without_head, feature_dim)
        Head가 제거된 Backbone과 특징 차원 / Backbone without head, and its feature dimension
    """
    name       = cfg["model"]["backbone"]   # "efficientnet_b0" | "resnet50"
    pretrained = cfg["model"]["pretrained"]

    if name == "efficientnet_b0":
        model       = models.efficientnet_b0(pretrained=pretrained)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()   # Head 제거, feature만 출력 / Remove head, output features only

    elif name == "resnet50":
        model       = models.resnet50(pretrained=pretrained)
        feature_dim = model.fc.in_features
        model.fc    = nn.Identity()        # Head 제거 / Remove head

    else:
        raise ValueError(f"지원하지 않는 backbone / Unsupported backbone: {name}")

    return model, feature_dim


# ──────────────────────────────────────────────
# Phase 0 — Projection MLP Head
# ──────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    Phase 0 Contrastive Learning 전용 Projection Head.
    Backbone 출력 → 256 → 128 (ReLU)
    Phase 2 진입 시 제거하고 ClassifierHead로 교체한다.

    Projection Head exclusively for Phase 0 Contrastive Learning.
    Backbone output → 256 → 128 (ReLU)
    Removed and replaced with ClassifierHead upon entering Phase 2.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),    # 배치 정규화 / Batch normalization
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# Phase 2 — Classifier MLP Head
# ──────────────────────────────────────────────
class ClassifierHead(nn.Module):
    """
    Phase 2 Supervised Classification 전용 Classifier Head.
    Backbone 출력 → FC → Dropout → 6-class logits
    CrossEntropyLoss가 내부적으로 Softmax를 포함하므로 logit을 반환한다.

    Classifier Head exclusively for Phase 2 Supervised Classification.
    Backbone output → FC → Dropout → 6-class logits.
    Returns raw logits (CrossEntropyLoss includes Softmax internally).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),    # 배치 정규화 / Batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),           # 과적합 방지 / Prevent overfitting
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logit 반환 (Softmax 미포함) / Returns logits (no Softmax)


# ──────────────────────────────────────────────
# 전체 모델 (Phase 전환 지원)
# Full Model (Supports Phase Switching)
# ──────────────────────────────────────────────
class GrayspotModel(nn.Module):
    """
    Phase 0 / Phase 2 모드를 지원하는 Grayspot 분류 모델.
    Grayspot classification model supporting Phase 0 / Phase 2 mode switching.

    Usage:
        # Phase 0 학습 / Phase 0 training
        model = GrayspotModel(cfg, phase=0)
        z = model(x)  # (B, 128) projection vector

        # Phase 2 전환 / Switch to Phase 2
        model.switch_to_phase2(weights_path="phase0_backbone.pt")
        logits = model(x)  # (B, 6) class logits
    """

    def __init__(self, cfg: dict, phase: int = 0):
        super().__init__()
        assert phase in (0, 2)
        self.cfg   = cfg
        self.phase = phase

        self.backbone, self.feature_dim = build_backbone(cfg)

        # Phase에 따라 Head 선택 / Select head based on phase
        if phase == 0:
            self.head = ProjectionHead(
                in_dim=self.feature_dim,
                hidden_dim=cfg["phase0"]["projection_hidden_dim"],
                out_dim=cfg["phase0"]["projection_dim"],
            )
        else:
            self.head = ClassifierHead(
                in_dim=self.feature_dim,
                hidden_dim=cfg["phase2"]["fc_hidden_dim"],
                num_classes=cfg["data"]["num_levels"],
                dropout=cfg["phase2"]["dropout_rate"],
            )

        # Phase 2 Stage 1: Backbone 고정 / Freeze backbone in Phase 2 Stage 1
        if phase == 2 and cfg["phase2"]["backbone_freeze"]:
            self._freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # 특징 추출 / Extract features
        return self.head(features)   # Head 통과 / Pass through head

    def _freeze_backbone(self) -> None:
        """Backbone 전체를 고정한다 (Stage 1). / Freezes all backbone parameters (Stage 1)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒  Backbone frozen (Stage 1)")

    def unfreeze_backbone(self, num_layers: int = 2) -> None:
        """
        Phase 2 Stage 2: Backbone 마지막 N개 레이어를 unfreeze하여 fine-tuning을 수행한다.
        Phase 2 Stage 2: Unfreezes the last N backbone layers for fine-tuning.
        """
        layers = list(self.backbone.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"🔓  Backbone last {num_layers} layers unfrozen (Stage 2)")

    def switch_to_phase2(self, weights_path: str | Path) -> None:
        """
        Phase 0에서 학습된 Backbone weights를 로드하고
        Projection Head를 Classifier Head로 교체하여 Phase 2 모드로 전환한다.

        Loads Phase 0 backbone weights and replaces the Projection Head
        with a Classifier Head to switch to Phase 2 mode.
        """
        # Phase 0 backbone weights 로드 / Load Phase 0 backbone weights
        state = torch.load(weights_path, map_location="cpu")
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"  Phase 0 Backbone weights 로드 / Loaded: {weights_path}")

        # Projection Head → Classifier Head 교체 / Replace Projection Head with Classifier Head
        self.head = ClassifierHead(
            in_dim=self.feature_dim,
            hidden_dim=self.cfg["phase2"]["fc_hidden_dim"],
            num_classes=self.cfg["data"]["num_levels"],
            dropout=self.cfg["phase2"]["dropout_rate"],
        )
        self.phase = 2

        if self.cfg["phase2"]["backbone_freeze"]:
            self._freeze_backbone()

    def save(self, path: str | Path) -> None:
        """모델 가중치를 저장한다. / Saves model weights."""
        torch.save(self.state_dict(), path)
        print(f"  모델 저장 / Model saved: {path}")

    def load(self, path: str | Path) -> None:
        """모델 가중치를 로드한다. / Loads model weights."""
        self.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"📂  모델 로드 / Model loaded: {path}")