"""
evaluation/metrics.py
=====================
Grayspot Detection Pipeline — Evaluation Metrics Module
Grayspot 탐지 파이프라인 — 평가 지표 모듈

This module is the Single Source of Truth (SSOT) for all final evaluation metrics.
이 모듈은 최종 평가 지표 계산을 위한 단일 표준(SSOT)입니다.

Source notebook: 04_evaluation.ipynb (Cell 6 · Metrics Computation)
PRD reference  : Section 1.4 (Performance Targets), Section 5.6, Section 5.6.2
Execution plan : Stage 2 (W5~W6), Role R3

Python 3.11.5 | macOS (MPS) & Windows (CUDA/CPU) compatible
"""

# ── Standard library / 표준 라이브러리 ────────────────────────────────────
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── Third-party / 서드파티 ────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants — PRD §1.4 Performance Targets
#    상수 — PRD §1.4 성능 목표
# ─────────────────────────────────────────────────────────────────────────────

NUM_LEVELS: int = 6  # Grayspot severity levels 0~5 / Grayspot 심각도 레벨 0~5
CHANNELS: list[str] = ["Y", "M", "C", "K"]  # CMYK channel order / CMYK 채널 순서

# Default performance targets from PRD §1.4
# PRD §1.4의 기본 성능 목표값
DEFAULT_TARGET_OVERALL_ACC: float = 0.90
DEFAULT_TARGET_PER_CLASS_F1: float = 0.80
DEFAULT_TARGET_PER_COLOR_ACC: float = 0.85
DEFAULT_TARGET_MAE: float = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataclasses — Typed result containers
#    데이터클래스 — 타입이 명시된 결과 컨테이너
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PerClassMetrics:
    """
    Metrics for a single Grayspot level (0~5).
    단일 Grayspot 레벨(0~5)에 대한 지표.

    Attributes:
        level     : Grayspot severity level (0~5) / Grayspot 심각도 레벨 (0~5)
        precision : Precision for this level / 이 레벨의 정밀도
        recall    : Recall for this level / 이 레벨의 재현율
        f1        : F1 score for this level / 이 레벨의 F1 점수
        support   : Number of true samples for this level / 이 레벨의 실제 샘플 수
        f1_pass   : Whether F1 meets the target threshold / F1이 목표 임계값을 충족하는지 여부
    """

    level: int
    precision: float
    recall: float
    f1: float
    support: int
    f1_pass: bool = field(default=False)


@dataclass
class ChannelMetrics:
    """
    Aggregated metrics for a single CMYK channel.
    단일 CMYK 채널에 대한 집계 지표.

    Attributes:
        channel      : CMYK channel identifier (Y/M/C/K or 'overall')
                       CMYK 채널 식별자 (Y/M/C/K 또는 'overall')
        accuracy     : Overall accuracy / 전체 정확도
        macro_f1     : Macro-averaged F1 score / 매크로 평균 F1 점수
        mae          : Mean Absolute Error (ordinal) / 평균 절대 오차 (순서형)
        per_class    : Per-class metrics list / 클래스별 지표 목록
        n_samples    : Total number of samples / 전체 샘플 수
        acc_pass     : Whether accuracy meets the target / 정확도가 목표를 충족하는지
        f1_pass      : Whether macro F1 meets the target / 매크로 F1이 목표를 충족하는지
        mae_pass     : Whether MAE meets the target / MAE가 목표를 충족하는지
    """

    channel: str
    accuracy: float
    macro_f1: float
    mae: float
    per_class: list[PerClassMetrics]
    n_samples: int
    acc_pass: bool = field(default=False)
    f1_pass: bool = field(default=False)
    mae_pass: bool = field(default=False)


@dataclass
class EvaluationSummary:
    """
    Full evaluation summary across all channels.
    모든 채널에 걸친 전체 평가 요약.

    Attributes:
        overall      : Combined metrics across all channels / 전체 채널 통합 지표
        by_channel   : Per-channel metrics dict / 채널별 지표 딕셔너리
        targets      : Performance targets used for evaluation / 평가에 사용된 성능 목표
        meta         : Metadata (backbone, checkpoint, n_samples, etc.)
                       메타데이터 (백본, 체크포인트, 샘플 수 등)
    """

    overall: ChannelMetrics
    by_channel: dict[str, ChannelMetrics]
    targets: dict[str, float]
    meta: dict


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core metric computation functions
#    핵심 지표 계산 함수
# ─────────────────────────────────────────────────────────────────────────────


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_LEVELS,
    target_f1: float = DEFAULT_TARGET_PER_CLASS_F1,
) -> list[PerClassMetrics]:
    """
    Compute per-class Precision, Recall, F1, and support.
    클래스별 Precision, Recall, F1, 지지도(support)를 계산합니다.

    Mirrors the logic in 04_evaluation.ipynb Cell 6 (compute_metrics → per_class).
    04_evaluation.ipynb Cell 6 (compute_metrics → per_class) 로직을 반영합니다.

    Args:
        y_true      : Ground-truth labels, shape (N,) / 정답 라벨, shape (N,)
        y_pred      : Predicted labels, shape (N,) / 예측 라벨, shape (N,)
        num_classes : Total number of Grayspot severity levels / Grayspot 심각도 레벨 수
        target_f1   : F1 target threshold for pass/fail flag / 합격/불합격 기준 F1 임계값

    Returns:
        List of PerClassMetrics, one entry per level / 레벨별 PerClassMetrics 목록
    """
    labels = list(range(num_classes))

    # sklearn returns arrays aligned to `labels` order
    # sklearn은 `labels` 순서에 맞게 배열을 반환합니다
    prec = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    rec = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # Compute support (actual count per class) / 클래스별 실제 샘플 수 계산
    support = np.array(
        [int(np.sum(y_true == lv)) for lv in labels], dtype=int
    )

    return [
        PerClassMetrics(
            level=i,
            precision=float(prec[i]),
            recall=float(rec[i]),
            f1=float(f1[i]),
            support=int(support[i]),
            f1_pass=bool(float(f1[i]) >= target_f1),
        )
        for i in labels
    ]


def compute_channel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    channel: str = "overall",
    num_classes: int = NUM_LEVELS,
    target_overall_acc: float = DEFAULT_TARGET_OVERALL_ACC,
    target_per_color_acc: float = DEFAULT_TARGET_PER_COLOR_ACC,
    target_per_class_f1: float = DEFAULT_TARGET_PER_CLASS_F1,
    target_mae: float = DEFAULT_TARGET_MAE,
) -> ChannelMetrics:
    """
    Compute all metrics for a single channel or the combined dataset.
    단일 채널 또는 통합 데이터셋에 대한 모든 지표를 계산합니다.

    This is the primary computation function called by the Evaluator.
    이 함수는 Evaluator에서 호출하는 기본 계산 함수입니다.

    PRD §5.6.2 metrics: Accuracy, Macro F1, Per-class Precision/Recall/F1, MAE
    PRD §5.6.2 지표: 정확도, 매크로 F1, 클래스별 Precision/Recall/F1, MAE

    Args:
        y_true             : Ground-truth labels, shape (N,) / 정답 라벨
        y_pred             : Predicted labels, shape (N,) / 예측 라벨
        channel            : Channel name for display / 표시용 채널 이름
        num_classes        : Number of severity levels / 심각도 레벨 수
        target_overall_acc : Overall accuracy target / 전체 정확도 목표
        target_per_color_acc: Per-color accuracy target / 색상별 정확도 목표
        target_per_class_f1: Per-class F1 target / 클래스별 F1 목표
        target_mae         : MAE target / MAE 목표

    Returns:
        ChannelMetrics dataclass / ChannelMetrics 데이터클래스
    """
    accuracy = float(accuracy_score(y_true, y_pred))

    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # MAE: treat Level as ordinal integer (PRD §1.4)
    # MAE: Level을 순서형 정수로 취급 (PRD §1.4)
    mae = float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))

    per_class = compute_per_class_metrics(
        y_true, y_pred, num_classes=num_classes, target_f1=target_per_class_f1
    )

    # Determine which accuracy target to compare against
    # 어떤 정확도 목표와 비교할지 결정
    acc_target = (
        target_overall_acc if channel == "overall" else target_per_color_acc
    )

    return ChannelMetrics(
        channel=channel,
        accuracy=accuracy,
        macro_f1=macro_f1,
        mae=mae,
        per_class=per_class,
        n_samples=int(len(y_true)),
        acc_pass=bool(accuracy >= acc_target),
        f1_pass=bool(macro_f1 >= target_per_class_f1),
        mae_pass=bool(mae <= target_mae),
    )


def compute_mae_by_level(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_LEVELS,
) -> dict[int, dict]:
    """
    Compute MAE and sample count broken down by true level.
    실제 레벨별 MAE와 샘플 수를 계산합니다.

    Used by the MAE heatmap visualization in the evaluator.
    Evaluator의 MAE 히트맵 시각화에서 사용됩니다.

    Args:
        y_true      : Ground-truth labels / 정답 라벨
        y_pred      : Predicted labels / 예측 라벨
        num_classes : Number of severity levels / 심각도 레벨 수

    Returns:
        Dict mapping level → {'mae': float, 'count': int}
        레벨 → {'mae': float, 'count': int} 매핑 딕셔너리
    """
    result: dict[int, dict] = {}
    for lv in range(num_classes):
        mask = y_true == lv
        count = int(mask.sum())
        if count > 0:
            mae_val = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        else:
            mae_val = float("nan")
        result[lv] = {"mae": mae_val, "count": count}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Phase 3 Feedback Decision Logic
#    Phase 3 피드백 복귀 판단 로직
# ─────────────────────────────────────────────────────────────────────────────


def determine_swing_feedback(
    summary: EvaluationSummary,
    channels: list[str] = CHANNELS,
    color_acc_threshold: float = 0.80,
    class_f1_threshold: float = 0.70,
    mae_phase0_threshold: float = 0.80,
) -> dict:
    """
    Implement PRD §3.3.2 Feedback Loop Decision Logic.
    PRD §3.3.2 피드백 루프 판단 기준을 구현합니다.

    Mirrors the decision logic from 04_evaluation.ipynb Cell 14.
    04_evaluation.ipynb Cell 14의 판단 로직을 반영합니다.

    Three checks are performed:
    세 가지 검사를 수행합니다:
        1. Per-color accuracy < threshold → Phase 0 (retrain representation)
           색상별 정확도 < 임계값 → Phase 0 (표현 재학습)
        2. Per-class F1 < threshold     → Phase 1 (review level boundary)
           클래스별 F1 < 임계값        → Phase 1 (레벨 경계 재검토)
        3. Overall MAE > threshold      → Phase 0 (representation retry)
           전체 MAE > 임계값           → Phase 0 (표현 학습 재시도)

    Returns a structured dict with:
    다음 내용을 담은 구조화된 딕셔너리를 반환합니다:
        'status'    : 'terminate' | 'action_required' | 'no_critical_failure'
        'decisions' : list of decision strings / 결정 문자열 목록
        'terminate' : bool flag / 종료 여부 플래그
    """
    targets = summary.targets
    overall = summary.overall
    decisions: list[str] = []

    # Check 1: Per-color accuracy → Phase 0
    # 검사 1: 색상별 정확도 → Phase 0
    for ch in channels:
        if ch not in summary.by_channel:
            continue
        acc = summary.by_channel[ch].accuracy
        if acc < color_acc_threshold:
            decisions.append(
                f"[{ch}] Accuracy {acc:.3f} < {color_acc_threshold:.2f} "
                "→ Phase 0 (표현 재학습 / Retrain representation)"
            )

    # Check 2: Per-class F1 → Phase 1
    # 검사 2: 클래스별 F1 → Phase 1
    for pc in overall.per_class:
        if pc.f1 < class_f1_threshold:
            decisions.append(
                f"Level {pc.level} F1={pc.f1:.3f} < {class_f1_threshold:.2f} "
                "→ Phase 1 (레벨 경계 재검토 / Review level boundary)"
            )

    # Check 3: Overall MAE → Phase 0
    # 검사 3: 전체 MAE → Phase 0
    if overall.mae > mae_phase0_threshold:
        decisions.append(
            f"Overall MAE {overall.mae:.3f} > {mae_phase0_threshold:.2f} "
            "→ Phase 0 (표현 학습 재시도 / Representation learning retry)"
        )

    # Check 4: All targets met → Terminate Swing
    # 검사 4: 모든 목표 달성 → Swing 종료
    overall_acc = overall.accuracy
    overall_mf1 = overall.macro_f1
    overall_mae = overall.mae

    all_color_ok = all(
        summary.by_channel.get(ch, ChannelMetrics(
            ch, 0.0, 0.0, 9.9, [], 0
        )).accuracy >= targets["per_color_accuracy"]
        for ch in channels
        if ch in summary.by_channel
    )
    all_class_ok = all(pc.f1 >= targets["per_class_f1"] for pc in overall.per_class)
    mae_ok = overall_mae <= targets["mae"]

    terminate = (
        overall_acc >= targets["overall_accuracy"]
        and all_color_ok
        and all_class_ok
        and mae_ok
    )

    if terminate:
        status = "terminate"
    elif decisions:
        status = "action_required"
    else:
        status = "no_critical_failure"

    return {
        "status": status,
        "terminate": terminate,
        "decisions": decisions,
        "overall_accuracy": overall_acc,
        "overall_macro_f1": overall_mf1,
        "overall_mae": overall_mae,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Serialization helpers
#    직렬화 헬퍼
# ─────────────────────────────────────────────────────────────────────────────


def summary_to_dict(summary: EvaluationSummary) -> dict:
    """
    Convert EvaluationSummary to a JSON-serializable dict.
    EvaluationSummary를 JSON 직렬화 가능한 딕셔너리로 변환합니다.

    Matches the metrics_summary.json structure from 04_evaluation.ipynb Cell 13.
    04_evaluation.ipynb Cell 13의 metrics_summary.json 구조와 일치합니다.

    Args:
        summary : EvaluationSummary to serialize / 직렬화할 EvaluationSummary

    Returns:
        JSON-serializable dict / JSON 직렬화 가능한 딕셔너리
    """

    def _channel_to_dict(cm: ChannelMetrics) -> dict:
        return {
            "accuracy": round(cm.accuracy, 4),
            "macro_f1": round(cm.macro_f1, 4),
            "mae": round(cm.mae, 4),
            "n_samples": cm.n_samples,
            "acc_pass": cm.acc_pass,
            "f1_pass": cm.f1_pass,
            "mae_pass": cm.mae_pass,
        }

    return {
        "meta": summary.meta,
        "targets": summary.targets,
        "global": {
            **_channel_to_dict(summary.overall),
            "per_class": [
                {
                    "level": pc.level,
                    "precision": round(pc.precision, 4),
                    "recall": round(pc.recall, 4),
                    "f1": round(pc.f1, 4),
                    "support": pc.support,
                    "f1_pass": pc.f1_pass,
                }
                for pc in summary.overall.per_class
            ],
        },
        "by_color": {
            ch: _channel_to_dict(cm)
            for ch, cm in summary.by_channel.items()
        },
    }


def save_metrics_json(
    summary: EvaluationSummary,
    output_path: str | Path,
) -> Path:
    """
    Save EvaluationSummary as a JSON file (UTF-8).
    EvaluationSummary를 JSON 파일(UTF-8)로 저장합니다.

    Equivalent to the json.dump block in 04_evaluation.ipynb Cell 13.
    04_evaluation.ipynb Cell 13의 json.dump 블록과 동일합니다.

    Args:
        summary     : EvaluationSummary to save / 저장할 EvaluationSummary
        output_path : Destination file path / 저장 경로

    Returns:
        Resolved Path of the saved file / 저장된 파일의 절대 경로
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = summary_to_dict(summary)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path.resolve()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quick summary print
#    간단한 요약 출력
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(
    summary: EvaluationSummary,
    channels: list[str] = CHANNELS,
) -> None:
    """
    Print a formatted performance summary table to stdout.
    포맷된 성능 요약 테이블을 stdout에 출력합니다.

    Mirrors the summary print block in 04_evaluation.ipynb Cell 6.
    04_evaluation.ipynb Cell 6의 요약 출력 블록을 반영합니다.

    Args:
        summary  : EvaluationSummary to display / 표시할 EvaluationSummary
        channels : CMYK channels to display / 표시할 CMYK 채널
    """
    targets = summary.targets
    print("\n=== Performance Summary / 성능 요약 ===")
    header = (
        f"{'Channel':>10}  {'Accuracy':>10}  {'Macro F1':>10}"
        f"  {'MAE':>8}  {'Acc':>4}  {'F1':>4}  {'MAE':>4}"
    )
    print(header)
    print("-" * len(header))

    for ch in channels + ["overall"]:
        if ch == "overall":
            cm = summary.overall
            tgt_acc = targets["overall_accuracy"]
        else:
            cm = summary.by_channel.get(ch)
            if cm is None:
                continue
            tgt_acc = targets["per_color_accuracy"]

        print(
            f"{cm.channel:>10}  {cm.accuracy:>10.4f}  {cm.macro_f1:>10.4f}"
            f"  {cm.mae:>8.4f}  "
            f"{'✅' if cm.accuracy >= tgt_acc else '❌':>4}  "
            f"{'✅' if cm.macro_f1 >= targets['per_class_f1'] else '❌':>4}  "
            f"{'✅' if cm.mae <= targets['mae'] else '❌':>4}"
        )

    print()
    print("Targets (PRD §1.4):")
    print(f"  Overall Accuracy ≥ {targets['overall_accuracy']:.0%}")
    print(f"  Per-color Acc    ≥ {targets['per_color_accuracy']:.0%}")
    print(f"  Per-class F1     ≥ {targets['per_class_f1']:.2f}")
    print(f"  MAE              ≤ {targets['mae']:.2f}")

    print("\n=== Per-Class Performance (Overall) / 클래스별 성능 (전체) ===")
    for pc in summary.overall.per_class:
        flag = "✅" if pc.f1 >= targets["per_class_f1"] else "❌"
        print(
            f"  Level {pc.level}  "
            f"Prec={pc.precision:.4f}  "
            f"Recall={pc.recall:.4f}  "
            f"F1={pc.f1:.4f}  "
            f"support={pc.support}  {flag}"
        )
