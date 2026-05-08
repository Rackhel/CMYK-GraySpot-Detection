"""
models/grayspot_model.py

Grayspot Detection 전체 모델 — Phase 0 / Phase 2 모드 전환 지원.
Full Grayspot Detection model — supports Phase 0 / Phase 2 mode switching.

Phase 0: Backbone + ProjectionHead  → embedding (contrastive learning)
Phase 2: Backbone + ClassifierHead  → logits    (supervised classification)
"""

import torch
import torch.nn as nn
from pathlib import Path

from models.backbone        import build_backbone
from models.projection_head import ProjectionHead
from models.classifier      import ClassifierHead
from utils import LoggerMixin


class GrayspotModel(nn.Module, LoggerMixin):
    """
    Grayspot Level 0~5 분류 모델.
    Grayspot Level 0~5 classification model.

    Args:
        cfg:   config.json 에서 로드한 설정 dict / Config dict loaded from config.json
        phase: 0 (Contrastive) | 2 (Supervised)
    """

    def __init__(self, cfg: dict, phase: int = 2):
        super().__init__()

        backbone_name = cfg["model"]["backbone"]
        num_levels    = cfg["data"]["num_levels"]        # 6
        proj_dim      = cfg["phase0"]["projection_dim"]  # 128
        proj_hidden   = cfg["phase0"]["hidden_dim"]      # 256 — Phase 0 head hidden dim
        cls_hidden    = cfg["phase2"]["hidden_dim"]      # 256 — Phase 2 head hidden dim
        dropout       = cfg["phase2"]["dropout"]         # 0.3

        # Backbone 로드 / Load backbone
        self.backbone, self.feature_dim = build_backbone(backbone_name)

        # Frozen backbone 지원 / Frozen backbone support
        if cfg["model"].get("frozen_backbone", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Phase에 따라 Head 구성 / Build head according to phase
        if phase == 0:
            self.head = ProjectionHead(
                in_dim=self.feature_dim,
                hidden_dim=proj_hidden,
                out_dim=proj_dim,
            )
        elif phase == 2:
            self.head = ClassifierHead(
                in_dim=self.feature_dim,
                hidden_dim=cls_hidden,
                num_classes=num_levels,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"지원하지 않는 phase / Unsupported phase: {phase}. "
                f"선택 가능 / Available: [0, 2]"
            )

        self.phase = phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image tensor

        Returns:
            Phase 0: (B, proj_dim)   projection vector
            Phase 2: (B, num_levels) logits
        """
        features = self.backbone(x)   # 특징 추출 / Extract features
        return self.head(features)    # Head 통과 / Pass through head

    def switch_to_phase2(self, backbone_path: Path, cfg: dict) -> None:
        """
        Phase 0 backbone weights를 로드하고 Head를 ClassifierHead로 교체한다.
        Loads Phase 0 backbone weights and replaces head with ClassifierHead.

        Args:
            backbone_path: Phase 0 학습 후 저장된 backbone .pt 파일 경로
                           Path to saved backbone .pt file after Phase 0 training
            cfg:           config.json dict
        """
        # backbone. 키만 선택적으로 로드 / Selectively load backbone keys only
        state          = torch.load(backbone_path, map_location="cpu")
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in state.items()
            if k.startswith("backbone.")
        }

        if backbone_state:
            self.backbone.load_state_dict(backbone_state, strict=False)
            self.logger.info(f"[PASS] Phase 0 backbone 로드 / Loaded: {Path(backbone_path).name}")
        else:
            self.logger.info("[WARN] backbone 키 없음 — pretrained weights 유지 / No backbone keys found")

        # Head를 ClassifierHead로 교체 / Replace head with ClassifierHead
        self.head  = ClassifierHead(
            in_dim=self.feature_dim,
            hidden_dim=cfg["phase2"]["hidden_dim"],
            num_classes=cfg["data"]["num_levels"],
            dropout=cfg["phase2"]["dropout"],
        )
        self.phase = 2
        self.logger.info("[PASS] Head 교체 완료 / Head replaced: ProjectionHead → ClassifierHead")
