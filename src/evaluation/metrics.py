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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Constants / 상수
# ---------------------------------------------------------------------------

# Level 정의 / Level definition (PRD Section 1.3)
NUM_LEVELS: int = 6
LEVEL_LABELS: List[int] = list(range(NUM_LEVELS))

# Channel list / 채널 목록
CHANNELS: List[str] = ["Y", "M", "C", "K"]

# 성능 목표 / Performance targets (PRD Section 1.4)
TARGET_OVERALL_ACC: float = 0.90
TARGET_PER_CLASS_F1: float = 0.80
TARGET_PER_COLOR_ACC: float = 0.85
TARGET_MAE: float = 0.50

# Aliases used by html_report.py / html_report.py 에서 사용하는 별칭
DEFAULT_TARGET_OVERALL_ACC: float = TARGET_OVERALL_ACC
DEFAULT_TARGET_PER_CLASS_F1: float = TARGET_PER_CLASS_F1
DEFAULT_TARGET_PER_COLOR_ACC: float = TARGET_PER_COLOR_ACC
DEFAULT_TARGET_MAE: float = TARGET_MAE

# 신뢰도 임계값 / Confidence thresholds (PRD Section 14.2)
CONF_THRESH_AUTO: float = 0.8  # 자동 판정 / Auto judgment
CONF_THRESH_WARN: float = 0.5  # 경고 포함 자동 / Warn + auto
CONF_THRESH_MANUAL: float = 0.3  # 수동 검수 대기 / Manual queue


# ---------------------------------------------------------------------------
# Data classes for structured results / 결과를 위한 데이터 클래스
# ---------------------------------------------------------------------------


@dataclass
class PerClassMetric:
    """단일 레벨의 분류 지표 / Classification metrics for a single level."""

    level: int
    precision: float
    recall: float
    f1: float
    support: int = 0


@dataclass
class ChannelMetrics:
    """
    단일 채널(Y/M/C/K) 또는 전체(overall)의 평가 지표.
    Evaluation metrics for a single channel or overall.
    """

    accuracy: float
    macro_f1: float
    mae: float
    n_samples: int
    per_class: List[PerClassMetric] = field(default_factory=list)

    # Convenience pass/fail flags / 합격/불합격 플래그
    @property
    def acc_pass(self) -> bool:
        return self.accuracy >= TARGET_PER_COLOR_ACC

    @property
    def f1_pass(self) -> bool:
        return self.macro_f1 >= TARGET_PER_CLASS_F1

    @property
    def mae_pass(self) -> bool:
        return self.mae <= TARGET_MAE


@dataclass
class EvaluationSummary:
    """
    전체 평가 결과 요약 — html_report.py 에서 사용한다.
    Full evaluation result summary — used by html_report.py.
    """

    overall: ChannelMetrics
    by_channel: Dict[str, ChannelMetrics]
    meta: Dict = field(default_factory=dict)
    targets: Dict = field(
        default_factory=lambda: {
            "overall_accuracy": TARGET_OVERALL_ACC,
            "per_color_accuracy": TARGET_PER_COLOR_ACC,
            "per_class_f1": TARGET_PER_CLASS_F1,
            "mae": TARGET_MAE,
        }
    )


# ---------------------------------------------------------------------------
# Core metric functions / 핵심 지표 함수
# ---------------------------------------------------------------------------


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_LEVELS,
) -> List[dict]:
    """
    클래스별 Precision, Recall, F1 을 계산한다.
    Computes per-class Precision, Recall, F1.

    Returns:
        List of dicts: [{'level': int, 'precision': float, 'recall': float, 'f1': float}, ...]
    """
    labels = list(range(num_classes))
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # Count support per class / 클래스별 샘플 수
    from collections import Counter

    counts = Counter(y_true.tolist())

    return [
        {
            "level": i,
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(counts.get(i, 0)),
        }
        for i in labels
    ]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_LEVELS,
) -> dict:
    """
    단일 채널 또는 전체 데이터에 대한 분류/순서형 지표를 계산한다.
    Computes classification and ordinal metrics for a single channel or all data.

    Returns:
        dict with keys: accuracy, macro_f1, mae, per_class, n_samples
    """
    if len(y_true) == 0:
        empty_per_class = [
            {
                "level": i,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }
            for i in range(num_classes)
        ]
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "mae": 0.0,
            "per_class": empty_per_class,
            "n_samples": 0,
        }

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mae = float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mae": mae,
        "per_class": compute_per_class_metrics(y_true, y_pred, num_classes),
        "n_samples": int(len(y_true)),
    }


def compute_all_channels(
    results: Dict[str, dict],
    channels: List[str] = None,
    num_classes: int = NUM_LEVELS,
) -> Dict[str, dict]:
    """
    색상별 + 전체 통합 지표를 한번에 계산한다.
    Computes per-color and combined overall metrics in one call.
    """
    if channels is None:
        channels = [c for c in CHANNELS if c in results]

    metrics: Dict[str, dict] = {}

    for color in channels:
        if color not in results:
            continue
        y_true = results[color]["y_true"]
        y_pred = results[color]["y_pred"]
        metrics[color] = compute_metrics(y_true, y_pred, num_classes)

    active = [c for c in channels if c in metrics]
    if active:
        all_true = np.concatenate([results[c]["y_true"] for c in active])
        all_pred = np.concatenate([results[c]["y_pred"] for c in active])
        metrics["overall"] = compute_metrics(all_true, all_pred, num_classes)
    else:
        metrics["overall"] = compute_metrics(np.array([]), np.array([]), num_classes)

    return metrics


def check_targets(
    metrics: Dict[str, dict],
    channels: List[str] = None,
) -> Dict[str, dict]:
    """
    PRD Section 1.4 성능 목표 달성 여부를 판정한다.
    Checks whether PRD Section 1.4 performance targets are met.
    """
    if channels is None:
        channels = [k for k in metrics if k != "overall"]

    results: Dict[str, dict] = {}

    # Overall targets
    m = metrics.get("overall", {})
    acc = m.get("accuracy", 0.0)
    f1 = m.get("macro_f1", 0.0)
    mae = m.get("mae", 9.9)

    per_class_f1_ok = all(
        pc["f1"] >= TARGET_PER_CLASS_F1 for pc in m.get("per_class", [])
    )

    results["overall"] = {
        "acc_pass": bool(acc >= TARGET_OVERALL_ACC),
        "f1_pass": bool(f1 >= TARGET_PER_CLASS_F1),
        "mae_pass": bool(mae <= TARGET_MAE),
        "per_class_f1_ok": per_class_f1_ok,
        "all_pass": bool(
            acc >= TARGET_OVERALL_ACC
            and f1 >= TARGET_PER_CLASS_F1
            and mae <= TARGET_MAE
            and per_class_f1_ok
        ),
    }

    for color in channels:
        m = metrics.get(color, {})
        acc = m.get("accuracy", 0.0)
        f1 = m.get("macro_f1", 0.0)
        mae = m.get("mae", 9.9)
        results[color] = {
            "acc_pass": bool(acc >= TARGET_PER_COLOR_ACC),
            "f1_pass": bool(f1 >= TARGET_PER_CLASS_F1),
            "mae_pass": bool(mae <= TARGET_MAE),
            "all_pass": bool(
                acc >= TARGET_PER_COLOR_ACC
                and f1 >= TARGET_PER_CLASS_F1
                and mae <= TARGET_MAE
            ),
        }

    return results


def print_summary(
    metrics: Dict[str, dict],
    channels: List[str] = None,
) -> None:
    """지표 요약을 콘솔에 출력한다. / Prints metric summary to console."""
    if channels is None:
        channels = [k for k in metrics if k != "overall"]

    targets = check_targets(metrics, channels)

    header = (
        f"{'Channel':>10}  {'Accuracy':>10}  {'Macro F1':>10}  "
        f"{'MAE':>8}  {'Acc':>4}  {'F1':>4}  {'MAE':>4}"
    )
    print("\n=== Performance Summary / 성능 요약 ===")
    print(header)
    print("-" * len(header))

    for ch in channels + ["overall"]:
        if ch not in metrics:
            continue
        m = metrics[ch]
        t = targets[ch]
        print(
            f"{ch:>10}  {m['accuracy']:>10.4f}  {m['macro_f1']:>10.4f}  "
            f"{m['mae']:>8.4f}  "
            f"{'OK' if t['acc_pass'] else '--':>4}  "
            f"{'OK' if t['f1_pass']  else '--':>4}  "
            f"{'OK' if t['mae_pass'] else '--':>4}"
        )

    print()
    print("Targets (PRD 1.4):")
    print(f"  Overall Accuracy >= {TARGET_OVERALL_ACC:.0%}")
    print(f"  Per-color Acc    >= {TARGET_PER_COLOR_ACC:.0%}")
    print(f"  Per-class F1     >= {TARGET_PER_CLASS_F1:.2f}")
    print(f"  MAE              <= {TARGET_MAE:.2f}")

    overall_pass = targets.get("overall", {}).get("all_pass", False)
    all_color_ok = all(targets.get(c, {}).get("acc_pass", False) for c in channels)
    if overall_pass and all_color_ok:
        print("\n  All targets met / 모든 목표 달성 -- TERMINATE Swing")
    else:
        print("\n  One or more targets not met / 목표 미달 항목 존재")


# ---------------------------------------------------------------------------
# Helper: build EvaluationSummary from raw results dict
# ---------------------------------------------------------------------------


def build_evaluation_summary(
    results: Dict[str, dict],
    channels: List[str] = None,
    meta: Dict = None,
    num_classes: int = NUM_LEVELS,
) -> EvaluationSummary:
    """
    raw results dict 로부터 EvaluationSummary 를 구성한다.
    Builds an EvaluationSummary from a raw results dict.

    Args:
        results  : {'Y': {'y_true': ..., 'y_pred': ...}, ...}
        channels : 채널 목록 / Channel list
        meta     : 메타데이터 (backbone 이름 등) / Metadata (backbone name etc.)
        num_classes : 클래스 수 / Number of classes
    """
    if channels is None:
        channels = [c for c in CHANNELS if c in results]

    raw_metrics = compute_all_channels(results, channels, num_classes)

    def _to_channel_metrics(m: dict) -> ChannelMetrics:
        per_class = [
            PerClassMetric(
                level=pc["level"],
                precision=pc["precision"],
                recall=pc["recall"],
                f1=pc["f1"],
                support=pc.get("support", 0),
            )
            for pc in m.get("per_class", [])
        ]
        return ChannelMetrics(
            accuracy=m["accuracy"],
            macro_f1=m["macro_f1"],
            mae=m["mae"],
            n_samples=m["n_samples"],
            per_class=per_class,
        )

    by_channel = {
        ch: _to_channel_metrics(raw_metrics[ch]) for ch in channels if ch in raw_metrics
    }
    overall = _to_channel_metrics(
        raw_metrics.get("overall", compute_metrics(np.array([]), np.array([])))
    )

    return EvaluationSummary(
        overall=overall,
        by_channel=by_channel,
        meta=meta or {},
    )


def summary_to_dict(summary: EvaluationSummary) -> dict:
    """EvaluationSummary 를 JSON-직렬화 가능한 dict 로 변환한다."""

    def _cm_to_dict(cm: ChannelMetrics) -> dict:
        return {
            "accuracy": cm.accuracy,
            "macro_f1": cm.macro_f1,
            "mae": cm.mae,
            "n_samples": cm.n_samples,
            "per_class": [
                {
                    "level": pc.level,
                    "precision": pc.precision,
                    "recall": pc.recall,
                    "f1": pc.f1,
                    "support": pc.support,
                }
                for pc in cm.per_class
            ],
        }

    return {
        "overall": _cm_to_dict(summary.overall),
        "by_channel": {ch: _cm_to_dict(cm) for ch, cm in summary.by_channel.items()},
        "meta": summary.meta,
        "targets": summary.targets,
    }


def determine_swing_feedback(
    summary: "EvaluationSummary",
    channels: List[str] = None,
) -> dict:
    """
    PRD 3.3.2 피드백 복귀 판단 로직.
    Phase 3 feedback-loop decision logic.

    Returns:
        dict with keys:
            terminate  : bool — 모든 목표 달성 시 True
            decisions  : List[str] — 조치 필요 항목 목록
    """
    if channels is None:
        channels = list(summary.by_channel.keys())

    decisions: List[str] = []
    overall = summary.overall
    targets = summary.targets

    # Check 1: per-color accuracy < 0.80 -> Phase 0
    for color in channels:
        cm = summary.by_channel.get(color)
        if cm is None:
            continue
        if cm.accuracy < 0.80:
            decisions.append(
                f"[{color}] Accuracy {cm.accuracy:.3f} < 0.80"
                " -> Phase 0 (retrain representation / 표현 재학습)"
            )

    # Check 2: per-class F1 < 0.70 -> Phase 1
    for pc in overall.per_class:
        if pc.f1 < 0.70:
            decisions.append(
                f"Level {pc.level} F1={pc.f1:.3f} < 0.70"
                " -> Phase 1 (review level boundary / 레벨 경계 재검토)"
            )

    # Check 3: overall MAE > 0.80 -> Phase 0
    if overall.mae > 0.80:
        decisions.append(
            f"Overall MAE {overall.mae:.3f} > 0.80"
            " -> Phase 0 (representation learning retry / 표현 학습 재시도)"
        )

    all_color_ok = all(
        summary.by_channel[c].acc_pass for c in channels if c in summary.by_channel
    )
    terminate = (
        overall.accuracy >= targets["overall_accuracy"]
        and overall.macro_f1 >= targets["per_class_f1"]
        and overall.mae <= targets["mae"]
        and all_color_ok
        and not decisions
    )

    return {
        "terminate": terminate,
        "decisions": decisions,
    }
