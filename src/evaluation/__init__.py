"""
evaluation/__init__.py
======================
Grayspot Detection Pipeline — Evaluation Package
Grayspot 탐지 파이프라인 — 평가 패키지

Public API exports for the evaluation subpackage.
평가 서브패키지의 공개 API 내보내기.

PRD reference  : Section 5.6 (Evaluation Module)
Execution plan : Stage 2 (W5~W6), Role R3

Usage / 사용법:
    from evaluation import GrayspotEvaluator, EvaluatorConfig
    from evaluation import compute_channel_metrics, EvaluationSummary

Python 3.11.5 | macOS (MPS) & Windows (CUDA/CPU) compatible
"""

# ── metrics.py public exports / metrics.py 공개 내보내기 ──────────────────
from .metrics import (
    # Constants / 상수
    NUM_LEVELS,
    CHANNELS,
    DEFAULT_TARGET_OVERALL_ACC,
    DEFAULT_TARGET_PER_CLASS_F1,
    DEFAULT_TARGET_PER_COLOR_ACC,
    DEFAULT_TARGET_MAE,

    # Dataclasses / 데이터클래스
    PerClassMetrics,
    ChannelMetrics,
    EvaluationSummary,

    # Computation functions / 계산 함수
    compute_per_class_metrics,
    compute_channel_metrics,
    compute_mae_by_level,

    # Phase 3 feedback / Phase 3 피드백
    determine_swing_feedback,

    # Serialization / 직렬화
    summary_to_dict,
    save_metrics_json,

    # Console output / 콘솔 출력
    print_summary,
)

# ── confusion.py public exports / confusion.py 공개 내보내기 ─────────────
from .confusion import (
    # Style constants / 스타일 상수
    PLOTLY_TEMPLATE,
    CMYK_COLORS,
    LEVEL_COLORS,

    # Confusion matrix / 혼동 행렬
    build_confusion_matrix_figure,
    save_confusion_matrix,
    save_all_confusion_matrices,

    # MAE heatmap / MAE 히트맵
    build_mae_heatmap_figure,
    save_mae_heatmap,
)

# ── evaluator.py public exports / evaluator.py 공개 내보내기 ─────────────
from .evaluator import (
    EvaluatorConfig,
    GrayspotEvaluator,
)

__all__ = [
    # Constants
    "NUM_LEVELS",
    "CHANNELS",
    "DEFAULT_TARGET_OVERALL_ACC",
    "DEFAULT_TARGET_PER_CLASS_F1",
    "DEFAULT_TARGET_PER_COLOR_ACC",
    "DEFAULT_TARGET_MAE",
    # Dataclasses
    "PerClassMetrics",
    "ChannelMetrics",
    "EvaluationSummary",
    # Core functions
    "compute_per_class_metrics",
    "compute_channel_metrics",
    "compute_mae_by_level",
    "determine_swing_feedback",
    "summary_to_dict",
    "save_metrics_json",
    "print_summary",
    # Visualization
    "PLOTLY_TEMPLATE",
    "CMYK_COLORS",
    "LEVEL_COLORS",
    "build_confusion_matrix_figure",
    "save_confusion_matrix",
    "save_all_confusion_matrices",
    "build_mae_heatmap_figure",
    "save_mae_heatmap",
    # Pipeline
    "EvaluatorConfig",
    "GrayspotEvaluator",
]
