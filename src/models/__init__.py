"""
models/__init__.py

모델 패키지 / Models package.

Grayspot Detection 시스템의 신경망 구성 요소를 내보낸다.
Exports neural network components for the Grayspot Detection system.

주요 클래스 / Key classes:
    GrayspotModel  : Phase 0 / Phase 2 전환 지원 메인 모델
    ClassifierHead : Phase 2 Supervised Classification Head
    ProjectionHead : Phase 0 Contrastive Learning Head
    build_backbone : Backbone 로드 팩토리 함수

사용법 / Usage:
    from models import GrayspotModel, build_backbone
    model = GrayspotModel(cfg, phase=2)
"""

from .grayspot_model  import GrayspotModel
from .backbone        import build_backbone
from .classifier      import ClassifierHead
from .projection_head import ProjectionHead

__all__ = [
    "GrayspotModel",
    "build_backbone",
    "ClassifierHead",
    "ProjectionHead",
]
