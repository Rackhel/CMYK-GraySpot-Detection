"""
tests/unit/test_metrics.py

evaluation/metrics.py 단위 테스트.
Unit tests for evaluation/metrics.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from evaluation.metrics import (NUM_LEVELS, TARGET_MAE, TARGET_OVERALL_ACC,
                                TARGET_PER_CLASS_F1, TARGET_PER_COLOR_ACC,
                                ChannelMetrics, EvaluationSummary,
                                PerClassMetric, build_evaluation_summary,
                                check_targets, compute_all_channels,
                                compute_metrics, compute_per_class_metrics,
                                determine_swing_feedback)

# ── compute_metrics ──────────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_perfect_accuracy(self, perfect_predictions):
        result = compute_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )
        assert result["accuracy"] == pytest.approx(1.0)

    def test_perfect_mae_is_zero(self, perfect_predictions):
        result = compute_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )
        assert result["mae"] == pytest.approx(0.0)

    def test_perfect_macro_f1(self, perfect_predictions):
        result = compute_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )
        assert result["macro_f1"] == pytest.approx(1.0)

    def test_worst_mae(self, worst_predictions):
        result = compute_metrics(
            worst_predictions["y_true"],
            worst_predictions["y_pred"],
        )
        assert result["mae"] == pytest.approx(5.0)

    def test_empty_input_returns_zeros(self):
        result = compute_metrics(np.array([]), np.array([]))
        assert result["accuracy"] == 0.0
        assert result["mae"] == 0.0
        assert result["macro_f1"] == 0.0
        assert result["n_samples"] == 0

    def test_n_samples_correct(self, perfect_predictions):
        n = len(perfect_predictions["y_true"])
        result = compute_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )
        assert result["n_samples"] == n

    def test_per_class_length(self, perfect_predictions):
        result = compute_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )
        assert len(result["per_class"]) == NUM_LEVELS

    def test_result_keys_present(self, perfect_predictions):
        result = compute_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )
        for key in ("accuracy", "macro_f1", "mae", "per_class", "n_samples"):
            assert key in result

    def test_accuracy_between_0_and_1(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 0, 2, 3, 4, 5])
        result = compute_metrics(y_true, y_pred)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_mae_non_negative(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([5, 4, 3, 2])
        result = compute_metrics(y_true, y_pred)
        assert result["mae"] >= 0.0


# ── compute_per_class_metrics ────────────────────────────────────────────────


class TestComputePerClassMetrics:
    def test_returns_list_of_length_num_levels(self, perfect_predictions):
        result = compute_per_class_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )
        assert len(result) == NUM_LEVELS

    def test_perfect_f1_per_class(self, perfect_predictions):
        result = compute_per_class_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )
        for pc in result:
            assert pc["f1"] == pytest.approx(1.0)

    def test_each_dict_has_required_keys(self, perfect_predictions):
        result = compute_per_class_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )
        for pc in result:
            for key in ("level", "precision", "recall", "f1", "support"):
                assert key in pc

    def test_level_indices_sequential(self, perfect_predictions):
        result = compute_per_class_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )
        levels = [pc["level"] for pc in result]
        assert levels == list(range(NUM_LEVELS))


# ── compute_all_channels ─────────────────────────────────────────────────────


class TestComputeAllChannels:
    def test_overall_key_always_present(self, multi_channel_results):
        result = compute_all_channels(multi_channel_results)
        assert "overall" in result

    def test_per_channel_keys_present(self, multi_channel_results):
        result = compute_all_channels(multi_channel_results)
        for ch in ["Y", "M", "C", "K"]:
            assert ch in result

    def test_overall_n_samples_is_sum_of_channels(self, multi_channel_results):
        result = compute_all_channels(multi_channel_results)
        total = sum(len(v["y_true"]) for v in multi_channel_results.values())
        assert result["overall"]["n_samples"] == total

    def test_empty_results_returns_zero_overall(self):
        result = compute_all_channels({})
        assert result["overall"]["accuracy"] == 0.0


# ── check_targets ─────────────────────────────────────────────────────────────


class TestCheckTargets:
    def test_perfect_metrics_all_pass(self, perfect_predictions):
        metrics = {
            "Y": compute_metrics(
                perfect_predictions["y_true"],
                perfect_predictions["y_pred"],
            ),
            "overall": compute_metrics(
                perfect_predictions["y_true"],
                perfect_predictions["y_pred"],
            ),
        }
        result = check_targets(metrics, channels=["Y"])
        assert result["overall"]["all_pass"] is True
        assert result["Y"]["acc_pass"] is True

    def test_zero_accuracy_all_fail(self, worst_predictions):
        y_true = np.zeros(6, dtype=int)
        y_pred = np.ones(6, dtype=int) * 5
        metrics = {
            "overall": compute_metrics(y_true, y_pred),
        }
        result = check_targets(metrics, channels=[])
        assert result["overall"]["acc_pass"] is False

    def test_overall_key_always_in_result(self, multi_channel_results):
        metrics = compute_all_channels(multi_channel_results)
        result = check_targets(metrics)
        assert "overall" in result


# ── build_evaluation_summary ──────────────────────────────────────────────────


class TestBuildEvaluationSummary:
    def test_returns_evaluation_summary_type(self, multi_channel_results):
        summary = build_evaluation_summary(multi_channel_results)
        assert isinstance(summary, EvaluationSummary)

    def test_overall_is_channel_metrics(self, multi_channel_results):
        summary = build_evaluation_summary(multi_channel_results)
        assert isinstance(summary.overall, ChannelMetrics)

    def test_by_channel_keys_present(self, multi_channel_results):
        summary = build_evaluation_summary(multi_channel_results, channels=["Y", "M"])
        assert "Y" in summary.by_channel
        assert "M" in summary.by_channel

    def test_swing_thresholds_injected_from_cfg(self, multi_channel_results):
        cfg = {
            "evaluation": {
                "swing_thresholds": {
                    "acc_retry": 0.60,
                    "f1_retry": 0.55,
                    "mae_retry": 0.90,
                }
            }
        }
        summary = build_evaluation_summary(multi_channel_results, cfg=cfg)
        assert summary.targets["swing_acc_retry"] == pytest.approx(0.60)
        assert summary.targets["swing_f1_retry"] == pytest.approx(0.55)
        assert summary.targets["swing_mae_retry"] == pytest.approx(0.90)

    def test_swing_thresholds_use_defaults_without_cfg(self, multi_channel_results):
        summary = build_evaluation_summary(multi_channel_results)
        assert "swing_acc_retry" in summary.targets
        assert "swing_f1_retry" in summary.targets
        assert "swing_mae_retry" in summary.targets

    def test_empty_results_does_not_raise(self):
        summary = build_evaluation_summary({})
        assert summary.overall.n_samples == 0

    def test_meta_stored_in_summary(self, multi_channel_results):
        meta = {"backbone": "efficientnet_b0"}
        summary = build_evaluation_summary(multi_channel_results, meta=meta)
        assert summary.meta["backbone"] == "efficientnet_b0"


# ── determine_swing_feedback ──────────────────────────────────────────────────


class TestDetermineSwingFeedback:
    def _make_summary(self, acc: float, f1: float, mae: float) -> EvaluationSummary:
        """테스트용 EvaluationSummary 생성 헬퍼."""
        per_class = [
            PerClassMetric(level=i, precision=f1, recall=f1, f1=f1) for i in range(6)
        ]
        overall = ChannelMetrics(
            accuracy=acc, macro_f1=f1, mae=mae, n_samples=10, per_class=per_class
        )
        channel_m = ChannelMetrics(
            accuracy=acc, macro_f1=f1, mae=mae, n_samples=10, per_class=per_class
        )
        summary = EvaluationSummary(
            overall=overall,
            by_channel={"Y": channel_m},
            targets={
                "overall_accuracy": TARGET_OVERALL_ACC,
                "per_color_accuracy": TARGET_PER_COLOR_ACC,
                "per_class_f1": TARGET_PER_CLASS_F1,
                "mae": TARGET_MAE,
                "swing_acc_retry": 0.80,
                "swing_f1_retry": 0.70,
                "swing_mae_retry": 0.80,
            },
        )
        return summary

    def test_perfect_metrics_terminate_true(self):
        summary = self._make_summary(acc=0.95, f1=0.90, mae=0.20)
        result = determine_swing_feedback(summary, channels=["Y"])
        assert result["terminate"] is True
        assert result["decisions"] == []

    def test_low_accuracy_produces_decisions(self):
        summary = self._make_summary(acc=0.50, f1=0.90, mae=0.20)
        result = determine_swing_feedback(summary, channels=["Y"])
        assert result["terminate"] is False
        assert len(result["decisions"]) > 0

    def test_low_f1_produces_decisions(self):
        summary = self._make_summary(acc=0.95, f1=0.60, mae=0.20)
        result = determine_swing_feedback(summary, channels=["Y"])
        assert result["terminate"] is False
        assert len(result["decisions"]) > 0

    def test_high_mae_produces_decisions(self):
        summary = self._make_summary(acc=0.95, f1=0.90, mae=0.90)
        result = determine_swing_feedback(summary, channels=["Y"])
        assert result["terminate"] is False
        assert len(result["decisions"]) > 0

    def test_result_has_required_keys(self):
        summary = self._make_summary(acc=0.95, f1=0.90, mae=0.20)
        result = determine_swing_feedback(summary, channels=["Y"])
        assert "terminate" in result
        assert "decisions" in result

    def test_decisions_is_list(self):
        summary = self._make_summary(acc=0.50, f1=0.50, mae=1.0)
        result = determine_swing_feedback(summary, channels=["Y"])
        assert isinstance(result["decisions"], list)


# ── ChannelMetrics 프로퍼티 / properties ────────────────────────────────────


class TestChannelMetricsProperties:
    def test_acc_pass_true_above_threshold(self):
        cm = ChannelMetrics(accuracy=0.90, macro_f1=0.85, mae=0.30, n_samples=10)
        assert cm.acc_pass is True

    def test_acc_pass_false_below_threshold(self):
        cm = ChannelMetrics(accuracy=0.80, macro_f1=0.85, mae=0.30, n_samples=10)
        assert cm.acc_pass is False

    def test_mae_pass_true_below_threshold(self):
        cm = ChannelMetrics(accuracy=0.90, macro_f1=0.85, mae=0.30, n_samples=10)
        assert cm.mae_pass is True

    def test_mae_pass_false_above_threshold(self):
        cm = ChannelMetrics(accuracy=0.90, macro_f1=0.85, mae=0.60, n_samples=10)
        assert cm.mae_pass is False

    def test_f1_pass_true_above_threshold(self):
        cm = ChannelMetrics(accuracy=0.90, macro_f1=0.85, mae=0.30, n_samples=10)
        assert cm.f1_pass is True
