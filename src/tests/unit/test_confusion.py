"""
tests/unit/test_confusion.py

evaluation/confusion.py 단위 테스트.
Unit tests for evaluation/confusion.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from evaluation.confusion import compute_confusion_matrix, plot_confusion_matrix


# ── compute_confusion_matrix ─────────────────────────────────────────────────

class TestComputeConfusionMatrix:
    def test_output_shape_is_6x6(self):
        y_true = np.arange(6)
        y_pred = np.arange(6)
        cm_raw, cm_norm = compute_confusion_matrix(y_true, y_pred)
        assert cm_raw.shape  == (6, 6)
        assert cm_norm.shape == (6, 6)

    def test_perfect_prediction_diagonal_ones(self):
        y_true = np.repeat(np.arange(6), 5)
        y_pred = np.repeat(np.arange(6), 5)
        _, cm_norm = compute_confusion_matrix(y_true, y_pred, normalize=True)
        np.testing.assert_array_almost_equal(np.diag(cm_norm), np.ones(6))

    def test_raw_matrix_contains_counts(self):
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 2])
        cm_raw, _ = compute_confusion_matrix(y_true, y_pred, normalize=False)
        assert cm_raw[0, 0] == 1
        assert cm_raw[0, 1] == 1
        assert cm_raw[1, 1] == 2

    def test_normalized_rows_sum_to_one(self, perfect_predictions):
        _, cm_norm = compute_confusion_matrix(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
            normalize=True,
        )
        row_sums = cm_norm.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(6))

    def test_normalize_false_returns_raw_counts(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        cm_raw, cm_no_norm = compute_confusion_matrix(y_true, y_pred, normalize=False)
        np.testing.assert_array_equal(cm_raw, cm_no_norm)

    def test_custom_num_classes(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        cm_raw, _ = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        assert cm_raw.shape == (3, 3)

    def test_all_wrong_prediction_zero_diagonal(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([5, 5, 5])
        _, cm_norm = compute_confusion_matrix(y_true, y_pred, normalize=True)
        assert cm_norm[0, 0] == pytest.approx(0.0)


# ── plot_confusion_matrix ─────────────────────────────────────────────────────

class TestPlotConfusionMatrix:
    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        y_true = np.repeat(np.arange(6), 5)
        y_pred = np.repeat(np.arange(6), 5)
        fig = plot_confusion_matrix(y_true, y_pred, title="Test CM")
        assert isinstance(fig, go.Figure)

    def test_does_not_raise_with_valid_inputs(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        plot_confusion_matrix(y_true, y_pred, title="No error test")

    def test_saves_html_file(self, tmp_path):
        y_true = np.repeat(np.arange(6), 3)
        y_pred = np.repeat(np.arange(6), 3)
        output = str(tmp_path / "cm_test.html")
        plot_confusion_matrix(y_true, y_pred, title="Save test", output_path=output)
        assert (tmp_path / "cm_test.html").exists()

    def test_normalize_false_does_not_raise(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5, 0])
        plot_confusion_matrix(y_true, y_pred, title="No norm", normalize=False)
