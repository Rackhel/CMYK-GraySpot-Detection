"""
evaluation/

PRD Section 5.6 에서 정의한 평가 모듈 패키지.
Evaluation module package defined in PRD Section 5.6.
"""

from evaluation.metrics import (
    # Constants
    NUM_LEVELS,
    CHANNELS,
    TARGET_OVERALL_ACC,
    TARGET_PER_CLASS_F1,
    TARGET_PER_COLOR_ACC,
    TARGET_MAE,
    DEFAULT_TARGET_OVERALL_ACC,
    DEFAULT_TARGET_PER_CLASS_F1,
    DEFAULT_TARGET_PER_COLOR_ACC,
    DEFAULT_TARGET_MAE,
    CONF_THRESH_AUTO,
    CONF_THRESH_WARN,
    CONF_THRESH_MANUAL,
    # Data classes
    PerClassMetric,
    ChannelMetrics,
    EvaluationSummary,
    # Functions
    compute_metrics,
    compute_per_class_metrics,
    compute_all_channels,
    check_targets,
    print_summary,
    build_evaluation_summary,
    summary_to_dict,
    determine_swing_feedback,
)

from evaluation.confusion import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_all_channels,
)

from evaluation.evaluator import Evaluator

__all__ = [
    # Constants
    "NUM_LEVELS",
    "CHANNELS",
    "TARGET_OVERALL_ACC",
    "TARGET_PER_CLASS_F1",
    "TARGET_PER_COLOR_ACC",
    "TARGET_MAE",
    "DEFAULT_TARGET_OVERALL_ACC",
    "DEFAULT_TARGET_PER_CLASS_F1",
    "DEFAULT_TARGET_PER_COLOR_ACC",
    "DEFAULT_TARGET_MAE",
    "CONF_THRESH_AUTO",
    "CONF_THRESH_WARN",
    "CONF_THRESH_MANUAL",
    # Data classes
    "PerClassMetric",
    "ChannelMetrics",
    "EvaluationSummary",
    # Metric functions
    "compute_metrics",
    "compute_per_class_metrics",
    "compute_all_channels",
    "check_targets",
    "print_summary",
    "build_evaluation_summary",
    "summary_to_dict",
    "determine_swing_feedback",
    # Confusion
    "compute_confusion_matrix",
    "plot_confusion_matrix",
    "plot_all_channels",
    # Evaluator
    "Evaluator",
]