"""
test_swing_efficiency.py
Tests for src/evaluation/swing_efficiency.py.
Status: FAILING — swing_efficiency.py not yet implemented.
Ref: doc/TDD/TDD_SwingEfficiency.md
"""

import sys
from pathlib import Path

import pytest

# ── sys.path 설정 ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # CMYK_MAIN/
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

# Will raise ImportError until implemented — correct failing behavior
from evaluation.swing_efficiency import SwingEfficiencyReport  # noqa: E402
from evaluation.swing_efficiency import compute_swing_efficiency, should_early_stop

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def base_cfg():
    """compute_swing_efficiency() 에 전달할 최소 cfg."""
    return {
        "evaluation": {
            "swing_thresholds": {
                "overall_accuracy": 0.90,
                "per_color_accuracy": 0.85,
            }
        }
    }


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────


def make_report(
    efficiency_ratio: float,
    swing_decision: str = "pass",
    cycle: int = 1,
    delta_acc: float = 0.09,
) -> SwingEfficiencyReport:
    """should_early_stop() 테스트용 SwingEfficiencyReport 인스턴스 생성."""
    return SwingEfficiencyReport(
        cycle=cycle,
        baseline_acc=0.72,
        cycle_acc=0.72 + delta_acc,
        delta_acc=delta_acc,
        n_labels_changed=45,
        efficiency_ratio=efficiency_ratio,
        swing_decision=swing_decision,
    )


# ── compute_swing_efficiency() ────────────────────────────────────────────────
# T-SE-01 ~ T-SE-06, T-SE-23


class TestComputeSwingEfficiency:
    """T-SE-01 ~ T-SE-06, T-SE-23: compute_swing_efficiency() 검증."""

    def test_normal_cycle_delta_acc(self, base_cfg):
        """T-SE-01a: baseline=0.72, cycle=0.81, n=45 → delta_acc ≈ 0.09"""
        report = compute_swing_efficiency(
            baseline_acc=0.72,
            cycle_acc=0.81,
            n_labels_changed=45,
            cycle=1,
            cfg=base_cfg,
        )
        assert (
            abs(report.delta_acc - 0.09) < 1e-6
        ), f"delta_acc={report.delta_acc}, 0.09 기대"

    def test_normal_cycle_efficiency_ratio(self, base_cfg):
        """T-SE-01b: efficiency_ratio = delta_acc / n_labels_changed = 0.09/45"""
        report = compute_swing_efficiency(
            baseline_acc=0.72,
            cycle_acc=0.81,
            n_labels_changed=45,
            cycle=1,
            cfg=base_cfg,
        )
        expected = 0.09 / 45
        assert (
            abs(report.efficiency_ratio - expected) < 1e-9
        ), f"efficiency_ratio={report.efficiency_ratio}, {expected} 기대"

    def test_normal_cycle_decision_is_valid(self, base_cfg):
        """T-SE-01c: swing_decision ∈ {'pass','retry_phase2','retry_phase0'}"""
        report = compute_swing_efficiency(
            baseline_acc=0.72,
            cycle_acc=0.81,
            n_labels_changed=45,
            cycle=1,
            cfg=base_cfg,
        )
        assert report.swing_decision in {
            "pass",
            "retry_phase2",
            "retry_phase0",
        }, f"swing_decision='{report.swing_decision}' 유효하지 않은 값"

    def test_zero_labels_changed_no_exception(self, base_cfg):
        """T-SE-02: n_labels_changed=0 → delta=0.0, efficiency=0.0 (ZeroDivisionError 없음)"""
        report = compute_swing_efficiency(
            baseline_acc=0.72,
            cycle_acc=0.72,
            n_labels_changed=0,
            cycle=1,
            cfg=base_cfg,
        )
        assert report.delta_acc == pytest.approx(
            0.0, abs=1e-8
        ), f"delta_acc={report.delta_acc}, 0.0 기대"
        assert report.efficiency_ratio == pytest.approx(
            0.0
        ), f"efficiency_ratio={report.efficiency_ratio}, 0.0 기대 (0/0 → 0.0)"

    def test_regression_cycle_delta_negative(self, base_cfg):
        """T-SE-03a: baseline=0.81 > cycle=0.75 → delta_acc < 0"""
        report = compute_swing_efficiency(
            baseline_acc=0.81,
            cycle_acc=0.75,
            n_labels_changed=20,
            cycle=2,
            cfg=base_cfg,
        )
        assert (
            report.delta_acc < 0
        ), f"성능 하락 시 delta_acc={report.delta_acc} < 0 이어야 함"

    def test_regression_cycle_decision(self, base_cfg):
        """T-SE-03b: 성능 하락 → decision ∈ {'retry_phase0','retry_phase2'}"""
        report = compute_swing_efficiency(
            baseline_acc=0.81,
            cycle_acc=0.75,
            n_labels_changed=20,
            cycle=2,
            cfg=base_cfg,
        )
        assert report.swing_decision in {
            "retry_phase0",
            "retry_phase2",
        }, f"성능 하락 시 swing_decision='{report.swing_decision}'"

    def test_large_improvement_delta_acc(self, base_cfg):
        """T-SE-04a: baseline=0.50, cycle=0.95, n=10 → delta_acc == 0.45"""
        report = compute_swing_efficiency(
            baseline_acc=0.50,
            cycle_acc=0.95,
            n_labels_changed=10,
            cycle=1,
            cfg=base_cfg,
        )
        assert (
            abs(report.delta_acc - 0.45) < 1e-6
        ), f"delta_acc={report.delta_acc}, 0.45 기대"

    def test_large_improvement_efficiency_ratio(self, base_cfg):
        """T-SE-04b: delta=0.45, n=10 → efficiency_ratio == 0.045"""
        report = compute_swing_efficiency(
            baseline_acc=0.50,
            cycle_acc=0.95,
            n_labels_changed=10,
            cycle=1,
            cfg=base_cfg,
        )
        assert (
            abs(report.efficiency_ratio - 0.045) < 1e-9
        ), f"efficiency_ratio={report.efficiency_ratio}, 0.045 기대"

    def test_cycle_number_stored(self, base_cfg):
        """T-SE-05: report.cycle == 입력 cycle 값(3)"""
        report = compute_swing_efficiency(
            baseline_acc=0.72,
            cycle_acc=0.81,
            n_labels_changed=45,
            cycle=3,
            cfg=base_cfg,
        )
        assert report.cycle == 3, f"report.cycle={report.cycle}, 3 기대"

    def test_return_type_is_dataclass(self, base_cfg):
        """T-SE-06: 반환값이 SwingEfficiencyReport 인스턴스"""
        report = compute_swing_efficiency(
            baseline_acc=0.72,
            cycle_acc=0.81,
            n_labels_changed=45,
            cycle=1,
            cfg=base_cfg,
        )
        assert isinstance(
            report, SwingEfficiencyReport
        ), f"반환 타입={type(report)}, SwingEfficiencyReport 기대"

    def test_swing_decision_valid_values(self, base_cfg):
        """T-SE-23: swing_decision은 {'pass','retry_phase2','retry_phase0'} 중 하나"""
        report = compute_swing_efficiency(
            baseline_acc=0.72,
            cycle_acc=0.81,
            n_labels_changed=45,
            cycle=1,
            cfg=base_cfg,
        )
        assert report.swing_decision in {
            "pass",
            "retry_phase2",
            "retry_phase0",
        }, f"swing_decision='{report.swing_decision}' 유효하지 않은 값"


# ── should_early_stop() ───────────────────────────────────────────────────────
# T-SE-10 ~ T-SE-13


class TestShouldEarlyStop:
    """T-SE-10 ~ T-SE-13: should_early_stop() 검증."""

    def test_early_stop_true_when_below_half(self):
        """T-SE-10: curr.eff=0.0009 < prev.eff*0.5=0.001 → True"""
        prev = make_report(efficiency_ratio=0.002)
        curr = make_report(efficiency_ratio=0.0009, cycle=2)
        assert (
            should_early_stop(curr, prev) is True
        ), "curr.eff(0.0009) < prev.eff*0.5(0.001) → True 기대"

    def test_early_stop_false_when_above_half(self):
        """T-SE-11: curr.eff=0.0015 >= prev.eff*0.5=0.001 → False"""
        prev = make_report(efficiency_ratio=0.002)
        curr = make_report(efficiency_ratio=0.0015, cycle=2)
        assert (
            should_early_stop(curr, prev) is False
        ), "curr.eff(0.0015) >= prev.eff*0.5(0.001) → False 기대"

    def test_early_stop_boundary_equal_to_half(self):
        """T-SE-12: curr.eff == prev.eff*0.5 (경계값 0.001) → False (경계 포함)"""
        prev = make_report(efficiency_ratio=0.002)
        curr = make_report(efficiency_ratio=0.001, cycle=2)
        assert (
            should_early_stop(curr, prev) is False
        ), "경계값(curr.eff == prev.eff*0.5) → False 기대 (포함)"

    def test_early_stop_true_when_previous_is_zero(self):
        """T-SE-13: prev.eff=0.0 → 항상 True (분모 0 처리)"""
        prev = make_report(efficiency_ratio=0.0)
        curr = make_report(efficiency_ratio=0.001, cycle=2)
        assert should_early_stop(curr, prev) is True, "prev.eff=0.0 → 항상 True 기대"


# ── SwingEfficiencyReport dataclass ───────────────────────────────────────────
# T-SE-20 ~ T-SE-22


class TestSwingEfficiencyReportDataclass:
    """T-SE-20 ~ T-SE-22: SwingEfficiencyReport 필드 타입 및 일관성 검증."""

    def test_cycle_is_int(self, base_cfg):
        """T-SE-20: report.cycle은 int 타입"""
        report = compute_swing_efficiency(0.72, 0.81, 45, cycle=1, cfg=base_cfg)
        assert isinstance(
            report.cycle, int
        ), f"report.cycle type={type(report.cycle)}, int 기대"

    def test_acc_fields_are_float(self, base_cfg):
        """T-SE-21: baseline_acc, cycle_acc는 float 타입"""
        report = compute_swing_efficiency(0.72, 0.81, 45, cycle=1, cfg=base_cfg)
        assert isinstance(
            report.baseline_acc, float
        ), f"baseline_acc type={type(report.baseline_acc)}, float 기대"
        assert isinstance(
            report.cycle_acc, float
        ), f"cycle_acc type={type(report.cycle_acc)}, float 기대"

    def test_delta_acc_equals_difference(self, base_cfg):
        """T-SE-22: delta_acc == cycle_acc - baseline_acc"""
        report = compute_swing_efficiency(0.72, 0.81, 45, cycle=1, cfg=base_cfg)
        expected_delta = report.cycle_acc - report.baseline_acc
        assert abs(report.delta_acc - expected_delta) < 1e-8, (
            f"delta_acc={report.delta_acc}, "
            f"cycle_acc - baseline_acc={expected_delta} 기대"
        )

    def test_n_labels_changed_is_int(self, base_cfg):
        """T-SE-20 확장: n_labels_changed는 int 타입"""
        report = compute_swing_efficiency(0.72, 0.81, 45, cycle=1, cfg=base_cfg)
        assert isinstance(
            report.n_labels_changed, int
        ), f"n_labels_changed type={type(report.n_labels_changed)}, int 기대"

    def test_efficiency_ratio_is_float(self, base_cfg):
        """T-SE-21 확장: efficiency_ratio는 float 타입"""
        report = compute_swing_efficiency(0.72, 0.81, 45, cycle=1, cfg=base_cfg)
        assert isinstance(
            report.efficiency_ratio, float
        ), f"efficiency_ratio type={type(report.efficiency_ratio)}, float 기대"
