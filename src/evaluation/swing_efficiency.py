"""
evaluation/swing_efficiency.py

Swing Feedback 루프 효율성 지표 계산 모듈.
Module for computing Swing Feedback loop efficiency metrics.

책임 / Responsibility:
    - Swing Cycle 반복당 라벨 수정 대비 성능 향상 효율을 측정한다.
    - Measures performance improvement efficiency per label change per Swing Cycle.
    - 조기 종료(Early Stop) 여부를 판단한다.
    - Determines whether early stopping is warranted.

SSOT 근거 / SSOT Reference:
    - SSOT_Evaluation_Reporting.md §4 — Swing Efficiency 정의
    - Contract_evaluation_reporting.md §13 — compute_swing_efficiency, should_early_stop API 계약
    - BDD_SwingFeedback.md §3.5 — 조기 종료 시나리오

TDD 근거 / TDD Reference:
    - TDD_SwingEfficiency.md §2 — T-SE-01 ~ T-SE-23

Python 3.11.5
"""

from __future__ import annotations

from dataclasses import dataclass

# ── 상수 / Constants ────────────────────────────────────────────────────────
# 효율성 비율 조기 종료 임계값: 이전 사이클 대비 50% 미만이면 종료
# Early stop threshold: stop if current efficiency < previous * 0.5
_EARLY_STOP_RATIO: float = 0.5

# Swing Decision 문자열 상수 / Swing Decision string constants
_DECISION_PASS: str = "pass"
_DECISION_RETRY_PHASE2: str = "retry_phase2"
_DECISION_RETRY_PHASE0: str = "retry_phase0"


# ── 데이터 클래스 / Dataclass ────────────────────────────────────────────────


@dataclass
class SwingEfficiencyReport:
    """
    단일 Swing Cycle의 효율성 측정 결과.
    Efficiency measurement result for a single Swing Cycle.

    Attributes:
        cycle            : Swing 사이클 번호 (1부터 시작) / Cycle number (1-indexed)
        baseline_acc     : 이전 사이클 종료 시점 정확도 / Accuracy at end of previous cycle
        cycle_acc        : 현재 사이클 종료 시점 정확도 / Accuracy at end of current cycle
        delta_acc        : 성능 변화량 = cycle_acc - baseline_acc
        n_labels_changed : 이번 사이클에서 수정된 라벨 수 / Number of labels changed this cycle
        efficiency_ratio : delta_acc / n_labels_changed (분모 0 시 0.0 / 0.0 when denominator is 0)
        swing_decision   : "pass" | "retry_phase2" | "retry_phase0"
    """

    cycle: int
    baseline_acc: float
    cycle_acc: float
    delta_acc: float
    n_labels_changed: int
    efficiency_ratio: float
    swing_decision: str


# ── 공개 함수 / Public functions ─────────────────────────────────────────────


def compute_swing_efficiency(
    baseline_acc: float,
    cycle_acc: float,
    n_labels_changed: int,
    cycle: int,
    cfg: dict,
) -> SwingEfficiencyReport:
    """
    단일 Swing Cycle의 효율성 지표를 계산한다.
    Computes efficiency metrics for a single Swing Cycle.

    Contract_evaluation_reporting.md §13 / TDD_SwingEfficiency.md T-SE-01~06 준수.
    Compliant with Contract §13 and TDD T-SE-01~06.

    Args:
        baseline_acc     : 이전 사이클 종료 정확도 [0, 1] / Accuracy before this cycle
        cycle_acc        : 현재 사이클 종료 정확도 [0, 1] / Accuracy after this cycle
        n_labels_changed : 이번 사이클 라벨 수정 수 (≥ 0) / Labels changed (non-negative)
        cycle            : 현재 사이클 번호 (1부터) / Current cycle number (1-indexed)
        cfg              : 프로젝트 config dict / Project config dict
                           cfg["evaluation"]["swing_thresholds"]["overall_accuracy"] 참조

    Returns:
        SwingEfficiencyReport — 효율성 측정 결과 / Efficiency measurement result

    Notes:
        - n_labels_changed == 0 이면 efficiency_ratio = 0.0 (ZeroDivisionError 없음)
          If n_labels_changed == 0, efficiency_ratio = 0.0 (no ZeroDivisionError)
        - delta_acc 음수 가능 (성능 하락) / delta_acc can be negative (regression)
    """
    delta_acc: float = cycle_acc - baseline_acc

    # 0 / 0 처리: 라벨 수정 없으면 efficiency = 0.0 (예외 없음)
    # Handle 0/0: no label changes → efficiency = 0.0 (no exception)
    if n_labels_changed == 0:
        efficiency_ratio: float = 0.0
    else:
        efficiency_ratio = delta_acc / n_labels_changed

    swing_decision = _determine_swing_decision(cycle_acc, delta_acc, cfg)

    return SwingEfficiencyReport(
        cycle=cycle,
        baseline_acc=float(baseline_acc),
        cycle_acc=float(cycle_acc),
        delta_acc=float(delta_acc),
        n_labels_changed=int(n_labels_changed),
        efficiency_ratio=float(efficiency_ratio),
        swing_decision=swing_decision,
    )


def should_early_stop(
    current_report: SwingEfficiencyReport,
    previous_report: SwingEfficiencyReport,
) -> bool:
    """
    현재 사이클의 효율이 이전 사이클의 50% 미만이면 조기 종료를 권장한다.
    Recommends early stopping if current cycle efficiency < 50% of previous cycle.

    Contract_evaluation_reporting.md §13 / TDD_SwingEfficiency.md T-SE-10~13 준수.
    Compliant with Contract §13 and TDD T-SE-10~13.

    조기 종료 조건 / Early stop condition:
        current_report.efficiency_ratio < previous_report.efficiency_ratio * 0.5

    경계값 포함 / Boundary inclusive:
        current == previous * 0.5 → False (계속 진행 / continue)

    Args:
        current_report  : 현재 사이클 결과 / Current cycle result
        previous_report : 이전 사이클 결과 / Previous cycle result

    Returns:
        True  — 조기 종료 권장 / Early stop recommended
        False — 계속 진행 권장 / Continue recommended

    Notes:
        - previous_report.efficiency_ratio == 0.0 이면 항상 True 반환
          If previous efficiency is 0.0, always returns True
    """
    prev_eff = previous_report.efficiency_ratio

    # 이전 효율이 0이면 비교 불가 → 항상 종료
    # If previous efficiency is 0, comparison is undefined → always stop
    if prev_eff == 0.0:
        return True

    threshold = prev_eff * _EARLY_STOP_RATIO

    # 경계값 포함: current < threshold 일 때만 True (current == threshold → False)
    # Boundary inclusive: True only when current < threshold (equal → False)
    return current_report.efficiency_ratio < threshold


# ── 내부 헬퍼 / Internal helper ──────────────────────────────────────────────


def _determine_swing_decision(
    cycle_acc: float,
    delta_acc: float,
    cfg: dict,
) -> str:
    """
    성능 지표 기반으로 Swing 결정을 반환한다.
    Returns Swing decision based on performance metrics.

    결정 로직 / Decision logic:
        - cycle_acc >= overall_accuracy_target → "pass"
        - delta_acc >= 0 이지만 목표 미달 → "retry_phase2" (미세 조정 반복)
        - delta_acc < 0 (성능 하락) → "retry_phase0" (전체 재학습)

    Args:
        cycle_acc : 현재 사이클 정확도 / Current cycle accuracy
        delta_acc : 성능 변화량 / Performance delta
        cfg       : 프로젝트 config dict / Project config dict

    Returns:
        "pass" | "retry_phase2" | "retry_phase0"
    """
    overall_target: float = (
        cfg.get("evaluation", {})
        .get("swing_thresholds", {})
        .get("overall_accuracy", 0.90)
    )

    if cycle_acc >= overall_target:
        return _DECISION_PASS

    if delta_acc >= 0:
        # 목표는 미달이지만 성능 향상 중 → Phase 2 재시도
        # Below target but improving → retry Phase 2
        return _DECISION_RETRY_PHASE2

    # 성능 하락 → Phase 0 부터 전체 재학습
    # Performance regression → restart from Phase 0
    return _DECISION_RETRY_PHASE0
