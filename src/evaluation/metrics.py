"""
evaluation/metrics.py

최종 평가 및 리포팅용 SSOT 지표 모듈.
SSOT metric module for final evaluation and reporting.

PRD Section 5.6.2 에서 정의한 모든 지표를 구현한다.
Implements all metrics defined in PRD Section 5.6.2.

지표 목록 / Metric list:
    - Accuracy (전체 / per-color)
    - Macro F1
    - Per-class Precision, Recall, F1
    - Confusion Matrix (6x6)
    - MAE (Mean Absolute Error) — Level을 순서형 정수로 취급

주의 / Note:
    이 모듈은 전체 데이터셋 기준의 최종 평가에 사용한다.
    학습 루프 내 실시간 metric (epoch 단위) 은 training/metrics.py 를 사용한다.
    This module is for final evaluation on the full dataset.
    For training-loop real-time metrics (per epoch), use training/metrics.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Level 정의 / Level definition (PRD Section 1.3)
NUM_LEVELS: int = 6
LEVEL_LABELS: List[int] = list(range(NUM_LEVELS))

# 성능 목표 / Performance targets (PRD Section 1.4)
TARGET_OVERALL_ACC   = 0.90
TARGET_PER_CLASS_F1  = 0.80
TARGET_PER_COLOR_ACC = 0.85
TARGET_MAE           = 0.50

# 신뢰도 임계값 / Confidence thresholds (PRD Section 14.2)
CONF_THRESH_AUTO   = 0.8   # 자동 판정 / Auto judgment
CONF_THRESH_WARN   = 0.5   # 경고 포함 자동 / Warn + auto
CONF_THRESH_MANUAL = 0.3   # 수동 검수 대기 / Manual queue


def compute_per_class_metrics(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    num_classes : int = NUM_LEVELS,
) -> List[dict]:
    """
    클래스별 Precision, Recall, F1 을 계산한다.
    Computes per-class Precision, Recall, F1.

    Args:
        y_true      : 정답 라벨 배열 / True label array (N,)
        y_pred      : 예측 라벨 배열 / Predicted label array (N,)
        num_classes : 클래스 수 / Number of classes (default: 6)

    Returns:
        List of dicts: [{'level': int, 'precision': float, 'recall': float, 'f1': float}, ...]
    """
    labels = list(range(num_classes))
    prec   = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec    = recall_score   (y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1     = f1_score       (y_true, y_pred, labels=labels, average=None, zero_division=0)

    return [
        {
            'level'    : i,
            'precision': float(prec[i]),
            'recall'   : float(rec[i]),
            'f1'       : float(f1[i]),
        }
        for i in labels
    ]


def compute_metrics(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    num_classes : int = NUM_LEVELS,
) -> dict:
    """
    단일 채널 또는 전체 데이터에 대한 분류/순서형 지표를 계산한다.
    Computes classification and ordinal metrics for a single channel or all data.

    Args:
        y_true      : 정답 라벨 배열 / True label array (N,)
        y_pred      : 예측 라벨 배열 / Predicted label array (N,)
        num_classes : 클래스 수 / Number of classes (default: 6)

    Returns:
        dict with keys:
            accuracy    : float  — 전체 정확도 / Overall accuracy
            macro_f1    : float  — Macro F1 (클래스 균형 / class-balanced)
            mae         : float  — Mean Absolute Error (순서형 / ordinal)
            per_class   : list   — 클래스별 지표 리스트 / Per-class metric list
            n_samples   : int    — 샘플 수 / Sample count
    """
    if len(y_true) == 0:
        return {
            'accuracy' : 0.0,
            'macro_f1' : 0.0,
            'mae'      : 0.0,
            'per_class': compute_per_class_metrics(
                np.array([0]), np.array([0]), num_classes
            ),
            'n_samples': 0,
        }

    labels   = list(range(num_classes))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    # MAE: Level 을 순서형 정수로 취급 / Treat Level as ordinal integer (PRD 1.4)
    mae = float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))

    return {
        'accuracy' : accuracy,
        'macro_f1' : macro_f1,
        'mae'      : mae,
        'per_class': compute_per_class_metrics(y_true, y_pred, num_classes),
        'n_samples': int(len(y_true)),
    }


def compute_all_channels(
    results     : Dict[str, dict],
    channels    : List[str] = None,
    num_classes : int = NUM_LEVELS,
) -> Dict[str, dict]:
    """
    색상별 + 전체 통합 지표를 한번에 계산한다.
    Computes per-color and combined overall metrics in one call.

    Args:
        results     : {'Y': {'y_true': ..., 'y_pred': ...}, 'M': ..., ...}
                      run_inference() 출력 형식 / Output format of run_inference()
        channels    : 처리할 채널 목록 / Channel list to process (default: all in results)
        num_classes : 클래스 수 / Number of classes

    Returns:
        dict with keys: 'Y', 'M', 'C', 'K' (available), 'overall'
        Each value is the dict returned by compute_metrics().
    """
    if channels is None:
        channels = [c for c in ['Y', 'M', 'C', 'K'] if c in results]

    metrics: Dict[str, dict] = {}

    for color in channels:
        if color not in results:
            continue
        y_true = results[color]['y_true']
        y_pred = results[color]['y_pred']
        metrics[color] = compute_metrics(y_true, y_pred, num_classes)

    # Overall combined / 전체 통합
    active = [c for c in channels if c in metrics]
    if active:
        all_true = np.concatenate([results[c]['y_true'] for c in active])
        all_pred = np.concatenate([results[c]['y_pred'] for c in active])
        metrics['overall'] = compute_metrics(all_true, all_pred, num_classes)
    else:
        metrics['overall'] = compute_metrics(np.array([]), np.array([]), num_classes)

    return metrics


def check_targets(
    metrics  : Dict[str, dict],
    channels : List[str] = None,
) -> Dict[str, dict]:
    """
    PRD Section 1.4 성능 목표 달성 여부를 판정한다.
    Checks whether PRD Section 1.4 performance targets are met.

    Args:
        metrics  : compute_all_channels() 반환값 / Return value of compute_all_channels()
        channels : 검사할 색상 채널 리스트 / Color channel list to check

    Returns:
        dict with keys: 'overall', per-color keys
        Each value: {'acc_pass', 'f1_pass', 'mae_pass', 'all_pass'}
    """
    if channels is None:
        channels = [k for k in metrics if k != 'overall']

    results: Dict[str, dict] = {}

    # Overall targets / 전체 목표
    m   = metrics.get('overall', {})
    acc = m.get('accuracy', 0.0)
    f1  = m.get('macro_f1', 0.0)
    mae = m.get('mae', 9.9)

    per_class_f1_ok = all(
        pc['f1'] >= TARGET_PER_CLASS_F1
        for pc in m.get('per_class', [])
    )

    results['overall'] = {
        'acc_pass'       : bool(acc >= TARGET_OVERALL_ACC),
        'f1_pass'        : bool(f1  >= TARGET_PER_CLASS_F1),
        'mae_pass'       : bool(mae <= TARGET_MAE),
        'per_class_f1_ok': per_class_f1_ok,
        'all_pass'       : bool(
            acc >= TARGET_OVERALL_ACC
            and f1  >= TARGET_PER_CLASS_F1
            and mae <= TARGET_MAE
            and per_class_f1_ok
        ),
    }

    # Per-color targets / 색상별 목표
    for color in channels:
        m   = metrics.get(color, {})
        acc = m.get('accuracy', 0.0)
        f1  = m.get('macro_f1', 0.0)
        mae = m.get('mae', 9.9)
        results[color] = {
            'acc_pass': bool(acc >= TARGET_PER_COLOR_ACC),
            'f1_pass' : bool(f1  >= TARGET_PER_CLASS_F1),
            'mae_pass': bool(mae <= TARGET_MAE),
            'all_pass': bool(
                acc >= TARGET_PER_COLOR_ACC
                and f1  >= TARGET_PER_CLASS_F1
                and mae <= TARGET_MAE
            ),
        }

    return results


def print_summary(
    metrics  : Dict[str, dict],
    channels : List[str] = None,
) -> None:
    """
    지표 요약을 콘솔에 출력한다.
    Prints metric summary to console.
    """
    if channels is None:
        channels = [k for k in metrics if k != 'overall']

    targets = check_targets(metrics, channels)

    header = (
        f"{'Channel':>10}  {'Accuracy':>10}  {'Macro F1':>10}  "
        f"{'MAE':>8}  {'Acc':>4}  {'F1':>4}  {'MAE':>4}"
    )
    print('\n=== Performance Summary / 성능 요약 ===')
    print(header)
    print('-' * len(header))

    for ch in channels + ['overall']:
        if ch not in metrics:
            continue
        m  = metrics[ch]
        t  = targets[ch]
        print(
            f"{ch:>10}  {m['accuracy']:>10.4f}  {m['macro_f1']:>10.4f}  "
            f"{m['mae']:>8.4f}  "
            f"{'OK' if t['acc_pass'] else '--':>4}  "
            f"{'OK' if t['f1_pass']  else '--':>4}  "
            f"{'OK' if t['mae_pass'] else '--':>4}"
        )

    print()
    print(f'Targets (PRD 1.4):')
    print(f'  Overall Accuracy >= {TARGET_OVERALL_ACC:.0%}')
    print(f'  Per-color Acc    >= {TARGET_PER_COLOR_ACC:.0%}')
    print(f'  Per-class F1     >= {TARGET_PER_CLASS_F1:.2f}')
    print(f'  MAE              <= {TARGET_MAE:.2f}')

    overall_pass = targets.get('overall', {}).get('all_pass', False)
    all_color_ok = all(
        targets.get(c, {}).get('acc_pass', False) for c in channels
    )
    if overall_pass and all_color_ok:
        print('\n  All targets met / 모든 목표 달성 -- TERMINATE Swing')
    else:
        print('\n  One or more targets not met / 목표 미달 항목 존재')
