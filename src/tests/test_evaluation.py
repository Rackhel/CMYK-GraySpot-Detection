"""
tests/test_evaluation.py

evaluation/ 모듈 단위 테스트.
Unit tests for the evaluation/ module.

PRD S2 완료 기준: 모든 모듈 단위 테스트 통과
PRD S2 completion criteria: All module unit tests pass

실행 방법 / How to run:
    # 프로젝트 루트에서 실행 / Run from project root
    pytest src/tests/test_evaluation.py -v

    # 특정 테스트만 실행 / Run specific test
    pytest src/tests/test_evaluation.py::TestMetrics::test_compute_metrics_perfect -v

테스트 범위 / Test coverage:
    TestMetrics   — metrics.py
    TestConfusion — confusion.py
    TestEvaluator — evaluator.py (모델/이미지 없이 로직만 검증 / logic-only, no real model/images)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# sys.path 설정 / sys.path setup
# 파일 위치: src/tests/test_evaluation.py
# File location: src/tests/test_evaluation.py
#
# parents[0] = src/tests/
# parents[1] = src/          <- 여기가 evaluation/ 패키지가 있는 곳
# parents[2] = CMYK_MAIN/    <- 프로젝트 루트
#
# conftest.py 가 프로젝트 루트에 있으면 pytest 가 자동으로 처리한다.
# If conftest.py is at project root, pytest handles this automatically.
# python 으로 직접 실행할 경우를 위해 아래 코드가 필요하다.
# The code below is needed when running directly with python.
_SRC_DIR = Path(__file__).resolve().parents[1]  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluation.confusion import (
    compute_confusion_matrix,
    plot_confusion_matrix,
)
from evaluation.evaluator import Evaluator
from evaluation.metrics import (
    CONF_THRESH_AUTO,
    CONF_THRESH_MANUAL,
    CONF_THRESH_WARN,
    NUM_LEVELS,
    TARGET_MAE,
    TARGET_OVERALL_ACC,
    TARGET_PER_CLASS_F1,
    TARGET_PER_COLOR_ACC,
    check_targets,
    compute_all_channels,
    compute_metrics,
    compute_per_class_metrics,
    print_summary,
)

# ---------------------------------------------------------------------------
# Fixtures / 픽스처
# ---------------------------------------------------------------------------


def make_labels(
    n: int = 60, num_classes: int = NUM_LEVELS, seed: int = 42
) -> np.ndarray:
    """
    균일 분포 정답 라벨을 생성한다.
    Generates uniformly distributed true labels.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_classes, size=n).astype(int)


def make_perfect_preds(y_true: np.ndarray) -> np.ndarray:
    """정답과 동일한 예측 / Predictions identical to true labels."""
    return y_true.copy()


def make_random_preds(
    n: int, num_classes: int = NUM_LEVELS, seed: int = 0
) -> np.ndarray:
    """완전 랜덤 예측 / Completely random predictions."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_classes, size=n).astype(int)


def make_results_dict(
    channels: list = None,
    n_per_ch: int = 60,
    perfect: bool = False,
    seed: int = 42,
) -> dict:
    """
    run() 출력 형식의 더미 results dict 를 생성한다.
    Generates a dummy results dict in the format output by run().
    """
    if channels is None:
        channels = ["Y", "M", "C", "K"]
    results = {}
    for i, ch in enumerate(channels):
        y_true = make_labels(n_per_ch, seed=seed + i)
        y_pred = (
            make_perfect_preds(y_true)
            if perfect
            else make_random_preds(n_per_ch, seed=seed + i + 10)
        )
        confs = (
            np.random.default_rng(seed + i)
            .uniform(0.3, 1.0, size=n_per_ch)
            .astype(np.float32)
        )
        fnames = [f"scan_{i:03d}_{ch}_{j:04d}.png" for j in range(n_per_ch)]
        results[ch] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "confidences": confs,
            "filenames": fnames,
        }
    return results


# ---------------------------------------------------------------------------
# TestMetrics — metrics.py
# ---------------------------------------------------------------------------


class TestMetrics:
    """metrics.py 단위 테스트 / Unit tests for metrics.py."""

    def test_constants_defined(self):
        """
        성능 목표 상수가 정의되어 있는지 확인한다.
        Checks that performance target constants are defined.
        """
        assert 0 < TARGET_OVERALL_ACC <= 1.0
        assert 0 < TARGET_PER_CLASS_F1 <= 1.0
        assert 0 < TARGET_PER_COLOR_ACC <= 1.0
        assert TARGET_MAE > 0

        assert 0 < CONF_THRESH_MANUAL < CONF_THRESH_WARN < CONF_THRESH_AUTO <= 1.0

    def test_compute_metrics_perfect(self):
        """
        완벽한 예측에 대한 지표 검증: Accuracy=1.0, MAE=0.0.
        Validates metrics for perfect predictions: Accuracy=1.0, MAE=0.0.
        """
        y_true = make_labels(60)
        y_pred = make_perfect_preds(y_true)

        m = compute_metrics(y_true, y_pred)

        assert m["accuracy"] == pytest.approx(1.0)
        assert m["macro_f1"] == pytest.approx(1.0, abs=0.01)
        assert m["mae"] == pytest.approx(0.0)
        assert m["n_samples"] == 60

    def test_compute_metrics_random(self):
        """
        랜덤 예측에 대한 지표가 합리적 범위 내에 있는지 검증한다.
        Validates that metrics for random predictions are within reasonable range.
        """
        y_true = make_labels(120)
        y_pred = make_random_preds(120)

        m = compute_metrics(y_true, y_pred)

        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0
        assert m["mae"] >= 0.0
        assert m["n_samples"] == 120

    def test_compute_metrics_per_class_length(self):
        """
        per_class 리스트가 NUM_LEVELS 길이를 갖는지 확인한다.
        Checks that per_class list has length NUM_LEVELS.
        """
        y_true = make_labels(60)
        y_pred = make_random_preds(60)
        m = compute_metrics(y_true, y_pred)

        assert len(m["per_class"]) == NUM_LEVELS
        for pc in m["per_class"]:
            assert "level" in pc
            assert "precision" in pc
            assert "recall" in pc
            assert "f1" in pc
            assert 0.0 <= pc["precision"] <= 1.0
            assert 0.0 <= pc["recall"] <= 1.0
            assert 0.0 <= pc["f1"] <= 1.0

    def test_compute_metrics_empty(self):
        """
        빈 배열 입력 시 안전하게 0을 반환하는지 확인한다.
        Checks that empty array input safely returns zeros.
        """
        m = compute_metrics(np.array([]), np.array([]))
        assert m["accuracy"] == 0.0
        assert m["macro_f1"] == 0.0
        assert m["mae"] == 0.0
        assert m["n_samples"] == 0

    def test_compute_per_class_metrics_shape(self):
        """
        compute_per_class_metrics 반환값 형태 검증.
        Validates the shape of compute_per_class_metrics return value.
        """
        y_true = make_labels(60)
        y_pred = make_random_preds(60)
        result = compute_per_class_metrics(y_true, y_pred)

        assert len(result) == NUM_LEVELS
        for i, pc in enumerate(result):
            assert pc["level"] == i

    def test_compute_all_channels_keys(self):
        """
        compute_all_channels 반환값에 'overall' 키와 채널 키가 있는지 확인한다.
        Checks that compute_all_channels result has 'overall' and channel keys.
        """
        results = make_results_dict(["C", "K"])
        metrics = compute_all_channels(results, ["C", "K"])

        assert "overall" in metrics
        assert "C" in metrics
        assert "K" in metrics

    def test_compute_all_channels_overall_aggregation(self):
        """
        overall 지표가 채널 데이터를 합산하여 계산되는지 검증한다.
        Validates that overall metrics are computed from aggregated channel data.
        """
        results = make_results_dict(["C", "K"], n_per_ch=60, perfect=True)
        metrics = compute_all_channels(results, ["C", "K"])

        # 완벽한 예측이므로 overall accuracy = 1.0 이어야 함
        # Perfect predictions -> overall accuracy should be 1.0
        assert metrics["overall"]["accuracy"] == pytest.approx(1.0)

    def test_check_targets_all_pass(self):
        """
        완벽한 예측에 대해 모든 목표가 달성됨을 검증한다.
        Validates that all targets pass for perfect predictions.
        """
        results = make_results_dict(["C", "K"], perfect=True)
        metrics = compute_all_channels(results, ["C", "K"])
        targets = check_targets(metrics, ["C", "K"])

        assert targets["overall"]["acc_pass"]
        assert targets["overall"]["all_pass"]

    def test_check_targets_structure(self):
        """
        check_targets 반환값이 올바른 키 구조를 갖는지 확인한다.
        Checks that check_targets return value has correct key structure.
        """
        results = make_results_dict(["Y", "M"])
        metrics = compute_all_channels(results, ["Y", "M"])
        targets = check_targets(metrics, ["Y", "M"])

        for key in ["overall", "Y", "M"]:
            assert key in targets
            assert "acc_pass" in targets[key]
            assert "all_pass" in targets[key]

    def test_mae_ordinal_property(self):
        """
        MAE 가 순서형 오류를 올바르게 반영하는지 검증한다.
        Validates that MAE correctly reflects ordinal errors.
        Level 0 -> Level 5 예측의 MAE 가 0 -> 1 보다 커야 한다.
        MAE for 0->5 prediction should be larger than 0->1.
        """
        y_true = np.array([0, 0, 0])
        y_pred_far = np.array([5, 5, 5])
        y_pred_near = np.array([1, 1, 1])

        m_far = compute_metrics(y_true, y_pred_far)
        m_near = compute_metrics(y_true, y_pred_near)

        assert m_far["mae"] > m_near["mae"]

    def test_print_summary_runs_without_error(self, capsys):
        """
        print_summary 가 오류 없이 실행되는지 확인한다.
        Checks that print_summary runs without error.
        """
        results = make_results_dict(["C"])
        metrics = compute_all_channels(results, ["C"])
        print_summary(metrics, channels=["C"])

        captured = capsys.readouterr()
        assert "Performance Summary" in captured.out


# ---------------------------------------------------------------------------
# TestConfusion — confusion.py
# ---------------------------------------------------------------------------


class TestConfusion:
    """confusion.py 단위 테스트 / Unit tests for confusion.py."""

    def test_compute_confusion_matrix_shape(self):
        """
        혼동 행렬이 (NUM_LEVELS, NUM_LEVELS) 형태인지 확인한다.
        Checks that the confusion matrix has shape (NUM_LEVELS, NUM_LEVELS).
        """
        y_true = make_labels(60)
        y_pred = make_random_preds(60)
        cm_raw, cm_norm = compute_confusion_matrix(y_true, y_pred)

        assert cm_raw.shape == (NUM_LEVELS, NUM_LEVELS)
        assert cm_norm.shape == (NUM_LEVELS, NUM_LEVELS)

    def test_compute_confusion_matrix_perfect(self):
        """
        완벽한 예측에 대해 대각선이 1.0 인지 확인한다.
        Checks that the diagonal is 1.0 for perfect predictions.
        """
        y_true = make_labels(60)
        y_pred = make_perfect_preds(y_true)
        _, cm_norm = compute_confusion_matrix(y_true, y_pred, normalize=True)

        # 샘플이 없는 클래스는 NaN 또는 0 이 될 수 있으므로 존재하는 행만 확인
        # Rows with no samples may be 0; check only rows that have samples
        for lv in range(NUM_LEVELS):
            if (y_true == lv).sum() > 0:
                assert cm_norm[lv, lv] == pytest.approx(1.0)

    def test_compute_confusion_matrix_no_normalize(self):
        """
        normalize=False 시 정수 카운트를 반환하는지 확인한다.
        Checks that integer counts are returned when normalize=False.
        """
        y_true = make_labels(60)
        y_pred = make_random_preds(60)
        cm_raw, cm_norm = compute_confusion_matrix(y_true, y_pred, normalize=False)

        # normalize=False 이면 cm_raw == cm_norm
        np.testing.assert_array_equal(cm_raw, cm_norm.astype(int))

    def test_compute_confusion_matrix_row_sum(self):
        """
        행 정규화 후 각 행의 합이 0 또는 1.0 인지 확인한다.
        Checks that each row sums to 0 or 1.0 after row normalization.
        """
        y_true = make_labels(60)
        y_pred = make_random_preds(60)
        _, cm_norm = compute_confusion_matrix(y_true, y_pred, normalize=True)

        for row in cm_norm:
            s = row.sum()
            assert s == pytest.approx(0.0) or s == pytest.approx(1.0, abs=1e-6)

    def test_plot_confusion_matrix_returns_figure(self):
        """
        plot_confusion_matrix 가 go.Figure 를 반환하는지 확인한다.
        Checks that plot_confusion_matrix returns a go.Figure.
        """
        import plotly.graph_objects as go

        y_true = make_labels(60)
        y_pred = make_random_preds(60)
        fig = plot_confusion_matrix(y_true, y_pred, title="Test CM")

        assert isinstance(fig, go.Figure)

    def test_plot_confusion_matrix_saves_html(self, tmp_path):
        """
        output_path 지정 시 HTML 파일이 생성되는지 확인한다.
        Checks that an HTML file is created when output_path is specified.
        """
        y_true = make_labels(60)
        y_pred = make_random_preds(60)
        out = str(tmp_path / "test_cm.html")

        plot_confusion_matrix(y_true, y_pred, title="Test", output_path=out)

        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


# ---------------------------------------------------------------------------
# TestEvaluator — evaluator.py
# ---------------------------------------------------------------------------


class TestEvaluator:
    """
    evaluator.py 단위 테스트.
    Unit tests for evaluator.py.

    실제 모델/이미지 없이 내부 로직만 검증한다.
    Tests only internal logic without real model or images.
    """

    def _make_dummy_evaluator(self, tmp_path: Path) -> Evaluator:
        """
        더미 Evaluator 인스턴스를 생성한다.
        Creates a dummy Evaluator instance.
        """
        dummy_model = MagicMock(spec=nn.Module)
        return Evaluator(
            model=dummy_model,
            labeled_dir=tmp_path / "labeled",
            labels_csv=tmp_path / "labels_v0.csv",
            output_dir=tmp_path / "reports",
            device=torch.device("cpu"),
            image_size=128,
            batch_size=32,
        )

    def test_extract_color_c(self):
        """
        C 색상 파일명에서 'C' 를 추출하는지 확인한다.
        Checks extraction of 'C' from a C-color filename.
        """
        assert Evaluator._extract_color("scan_001_C_0007.png") == "C"

    def test_extract_color_y(self):
        """Y 색상 파일명 / Y-color filename."""
        assert Evaluator._extract_color("scan_016_Y_0004.png") == "Y"

    def test_extract_color_m(self):
        """M 색상 파일명 / M-color filename."""
        assert Evaluator._extract_color("scan_002_M_0016.png") == "M"

    def test_extract_color_k(self):
        """K 색상 파일명 / K-color filename."""
        assert Evaluator._extract_color("scan_002_K_0016.png") == "K"

    def test_extract_color_invalid(self):
        """
        색상 코드가 없는 파일명은 None 을 반환하는지 확인한다.
        Checks that None is returned for filenames without color codes.
        """
        assert Evaluator._extract_color("image_001.png") is None
        assert Evaluator._extract_color("scan_001_X_0007.png") is None

    def test_output_dir_created(self, tmp_path):
        """
        output_dir 이 자동으로 생성되는지 확인한다.
        Checks that output_dir is created automatically.
        """
        ev = self._make_dummy_evaluator(tmp_path)
        assert ev.output_dir.exists()

    def test_compute_calls_compute_all_channels(self, tmp_path):
        """
        compute() 가 compute_all_channels 를 호출하여 지표를 반환하는지 확인한다.
        Checks that compute() returns metrics via compute_all_channels.
        """
        ev = self._make_dummy_evaluator(tmp_path)
        results = make_results_dict(["C", "K"])
        metrics = ev.compute(results, channels=["C", "K"])

        assert "overall" in metrics
        assert "C" in metrics
        assert "K" in metrics

    def test_get_misclassified_structure(self, tmp_path):
        """
        get_misclassified 반환 DataFrame 의 컬럼 구조를 검증한다.
        Validates the column structure of the DataFrame returned by get_misclassified.
        """
        ev = self._make_dummy_evaluator(tmp_path)
        results = make_results_dict(["C"], n_per_ch=60, perfect=False)
        df = ev.get_misclassified(results, channels=["C"])

        expected_cols = {
            "filename",
            "color",
            "true_level",
            "pred_level",
            "confidence",
            "correct",
            "error_gap",
        }
        assert expected_cols.issubset(set(df.columns))
        if len(df) > 0:
            assert all(df["correct"] == False)
            assert all(df["error_gap"] >= 1)

    def test_get_misclassified_empty_for_perfect(self, tmp_path):
        """
        완벽한 예측에서 오분류 DataFrame 이 비어 있는지 확인한다.
        Checks that the misclassified DataFrame is empty for perfect predictions.
        """
        ev = self._make_dummy_evaluator(tmp_path)
        results = make_results_dict(["C"], n_per_ch=60, perfect=True)
        df = ev.get_misclassified(results, channels=["C"])

        assert len(df) == 0

    def test_save_csv_creates_file(self, tmp_path):
        """
        save_csv 가 CSV 파일을 생성하는지 확인한다.
        Checks that save_csv creates a CSV file.
        """
        ev = self._make_dummy_evaluator(tmp_path)
        results = make_results_dict(["C"], n_per_ch=20)
        path = ev.save_csv(results, experiment_name="test", channels=["C"])

        assert path.exists()
        import pandas as pd

        df = pd.read_csv(path)
        assert len(df) == 20
        assert "true_level" in df.columns
        assert "pred_level" in df.columns
        assert "confidence" in df.columns

    def test_save_json_creates_file(self, tmp_path):
        """
        save_json 이 JSON 파일을 생성하는지 확인한다.
        Checks that save_json creates a JSON file.
        """
        import json as _json

        ev = self._make_dummy_evaluator(tmp_path)
        results = make_results_dict(["C", "K"])
        metrics = ev.compute(results, channels=["C", "K"])
        path = ev.save_json(metrics, experiment_name="test", channels=["C", "K"])

        assert path.exists()
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
        assert "global" in data
        assert "by_color" in data
        assert "per_class_overall" in data

    def test_save_json_target_fields(self, tmp_path):
        """
        JSON 에 성능 목표 필드가 포함되어 있는지 확인한다.
        Checks that the JSON includes performance target fields.
        """
        import json as _json

        ev = self._make_dummy_evaluator(tmp_path)
        results = make_results_dict(["C"])
        metrics = ev.compute(results, channels=["C"])
        path = ev.save_json(metrics, experiment_name="targets_test", channels=["C"])

        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
        targets = data["targets"]
        assert "overall_accuracy" in targets
        assert "per_color_accuracy" in targets
        assert "per_class_f1" in targets
        assert "mae" in targets

    def test_build_phase3_decision_all_pass(self, tmp_path):
        """
        완벽한 예측에서 Phase 3 판단이 TERMINATE 를 포함하는지 확인한다.
        Checks that Phase 3 decision includes TERMINATE for perfect predictions.
        """
        ev = self._make_dummy_evaluator(tmp_path)
        results = make_results_dict(["C", "K"], perfect=True)
        metrics = ev.compute(results, channels=["C", "K"])
        text = ev._build_phase3_decision(metrics, ["C", "K"])

        assert "TERMINATE" in text

    def test_build_phase3_decision_action_required(self, tmp_path):
        """
        랜덤 예측에서 Phase 3 판단이 조치 필요를 포함하는지 확인한다.
        Checks that Phase 3 decision includes action required for random predictions.
        """
        ev = self._make_dummy_evaluator(tmp_path)
        results = make_results_dict(["C", "K"], perfect=False)
        metrics = ev.compute(results, channels=["C", "K"])
        text = ev._build_phase3_decision(metrics, ["C", "K"])

        # 랜덤 예측은 목표 미달이므로 조치 내용이 포함되어야 함
        # Random predictions miss targets, so action text should be present
        assert "Phase 3" in text


# ---------------------------------------------------------------------------
# Integration test / 통합 테스트
# ---------------------------------------------------------------------------


class TestIntegration:
    """
    metrics + confusion + evaluator 통합 테스트.
    Integration tests for metrics + confusion + evaluator.
    """

    def test_full_pipeline_without_model(self, tmp_path):
        """
        실제 모델 없이 지표 계산 -> CSV -> JSON 파이프라인 전체를 실행한다.
        Runs the full metrics -> CSV -> JSON pipeline without a real model.
        """
        dummy_model = MagicMock(spec=nn.Module)
        ev = Evaluator(
            model=dummy_model,
            labeled_dir=tmp_path / "labeled",
            labels_csv=tmp_path / "labels_v0.csv",
            output_dir=tmp_path / "reports",
            device=torch.device("cpu"),
        )

        results = make_results_dict(["Y", "M", "C", "K"], n_per_ch=30)
        metrics = ev.compute(results)
        ev.save_csv(results, experiment_name="integration")
        ev.save_json(metrics, experiment_name="integration")

        # 저장된 파일 존재 여부 확인 / Check saved files exist
        assert (tmp_path / "reports" / "evaluation_results_integration.csv").exists()
        assert (tmp_path / "reports" / "metrics_summary_integration.json").exists()

    def test_metrics_and_confusion_consistency(self):
        """
        metrics.py 와 confusion.py 의 Accuracy 가 일치하는지 확인한다.
        Checks that accuracy from metrics.py and confusion.py are consistent.
        """
        y_true = make_labels(120)
        y_pred = make_random_preds(120)

        m = compute_metrics(y_true, y_pred)
        cm_raw, _ = compute_confusion_matrix(y_true, y_pred, normalize=False)

        # 혼동 행렬 대각선 합 / 전체 합 = accuracy
        # Diagonal sum / total sum = accuracy
        acc_from_cm = cm_raw.diagonal().sum() / cm_raw.sum()

        assert m["accuracy"] == pytest.approx(acc_from_cm, abs=1e-6)
