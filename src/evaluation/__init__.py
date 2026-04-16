"""
evaluation/

PRD Section 5.6 에서 정의한 평가 모듈 패키지.
Evaluation module package defined in PRD Section 5.6.

모듈 구성 / Module layout:
    metrics.py   — SSOT 지표 계산 / SSOT metric computation
    confusion.py — Confusion Matrix 생성 / Confusion Matrix generation
    evaluator.py — 전체 평가 파이프라인 / Full evaluation pipeline

주의 / Note:
    이 패키지는 학습 루프 내 실시간 metric 과는 분리되어 있다.
    학습 루프 내 metric 은 training/metrics.py 를 사용한다.
    This package is separate from training-loop real-time metrics.
    For training-loop metrics, use training/metrics.py.
"""

from evaluation.metrics import (
    NUM_LEVELS,
    TARGET_OVERALL_ACC,
    TARGET_PER_CLASS_F1,
    TARGET_PER_COLOR_ACC,
    TARGET_MAE,
    CONF_THRESH_AUTO,
    CONF_THRESH_WARN,
    CONF_THRESH_MANUAL,
    compute_metrics,
    compute_all_channels,
    check_targets,
    print_summary,
)

from evaluation.confusion import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_all_channels,
)

from evaluation.evaluator import Evaluator

__all__ = [
    # metrics
    'NUM_LEVELS',
    'TARGET_OVERALL_ACC',
    'TARGET_PER_CLASS_F1',
    'TARGET_PER_COLOR_ACC',
    'TARGET_MAE',
    'CONF_THRESH_AUTO',
    'CONF_THRESH_WARN',
    'CONF_THRESH_MANUAL',
    'compute_metrics',
    'compute_all_channels',
    'check_targets',
    'print_summary',
    # confusion
    'compute_confusion_matrix',
    'plot_confusion_matrix',
    'plot_all_channels',
    # evaluator
    'Evaluator',
]
