"""
evaluation/

PRD Section 5.6 에서 정의한 평가 모듈 패키지.
Evaluation module package defined in PRD Section 5.6.
"""

from evaluation.confusion import (compute_confusion_matrix, plot_all_channels,
                                  plot_confusion_matrix)
from evaluation.evaluator import Evaluator
from evaluation.metrics import CHANNELS  # Constants; Data classes; Functions
from evaluation.metrics import (CONF_THRESH_AUTO, CONF_THRESH_MANUAL,
                                CONF_THRESH_WARN, DEFAULT_TARGET_MAE,
                                DEFAULT_TARGET_OVERALL_ACC,
                                DEFAULT_TARGET_PER_CLASS_F1,
                                DEFAULT_TARGET_PER_COLOR_ACC, NUM_LEVELS,
                                TARGET_MAE, TARGET_OVERALL_ACC,
                                TARGET_PER_CLASS_F1, TARGET_PER_COLOR_ACC,
                                ChannelMetrics, EvaluationSummary,
                                PerClassMetric, build_evaluation_summary,
                                check_targets, compute_all_channels,
                                compute_metrics, compute_per_class_metrics,
                                determine_swing_feedback, print_summary,
                                summary_to_dict)

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
