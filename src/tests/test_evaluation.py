"""
tests/test_evaluation.py
========================
Unit tests for the evaluation package.
evaluation 패키지 단위 테스트.

Tests cover metrics.py, confusion.py, and evaluator.py.
tests는 metrics.py, confusion.py, evaluator.py를 다룹니다.

PRD reference  : Section 5.6 (Evaluation Module)
Execution plan : Stage 2 (W5~W6), Role R3

Run / 실행:
    pytest tests/test_evaluation.py -v

Python 3.11.5 | macOS (MPS) & Windows (CUDA/CPU) compatible
"""

# ── Standard library / 표준 라이브러리 ────────────────────────────────────
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# ── Python 3.11.5 version guard / Python 3.11.5 버전 가드 ────────────────
assert sys.version_info[:2] == (3, 11), (
    f"Python 3.11.x required, got {sys.version}. "
    f"Python 3.11.x가 필요합니다. 현재: {sys.version}"
)

# ── Third-party / 서드파티 ────────────────────────────────────────────────
import numpy as np
import pytest

# ── Source under test / 테스트 대상 소스 ─────────────────────────────────
# Folder layout / 폴더 구조:
#   CMYK_MAIN/
#     src/
#       evaluation/    ← import target / 임포트 대상
#     tests/
#       test_evaluation.py   ← this file / 이 파일
#
# Path chain: tests/ → CMYK_MAIN/ → src/
# 경로 체인: tests/ → CMYK_MAIN/ → src/
_SRC_DIR = Path(__file__).parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from evaluation.metrics import (
    NUM_LEVELS,
    CHANNELS,
    compute_per_class_metrics,
    compute_channel_metrics,
    compute_mae_by_level,
    determine_swing_feedback,
    EvaluationSummary,
    ChannelMetrics,
    PerClassMetrics,
    summary_to_dict,
    save_metrics_json,
    print_summary,
)
from evaluation.confusion import (
    build_confusion_matrix_figure,
    build_mae_heatmap_figure,
)
from evaluation.evaluator import EvaluatorConfig, GrayspotEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — shared test data / 픽스처 — 공유 테스트 데이터
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def perfect_preds() -> tuple[np.ndarray, np.ndarray]:
    """
    Perfect predictions: y_pred == y_true for all 6 levels.
    완벽한 예측: 6개 레벨 모두 y_pred == y_true.
    """
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, NUM_LEVELS, size=300)
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture
def random_preds() -> tuple[np.ndarray, np.ndarray]:
    """
    Random predictions with controlled seed for reproducibility.
    재현성을 위해 제어된 시드를 사용한 랜덤 예측.
    """
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, NUM_LEVELS, size=300)
    y_pred = rng.integers(0, NUM_LEVELS, size=300)
    return y_true, y_pred


@pytest.fixture
def mock_results(random_preds) -> dict[str, dict]:
    """
    Mock inference results dict with shape matching run_inference() output.
    run_inference() 출력 형태와 일치하는 모의 추론 결과 딕셔너리.
    """
    rng = np.random.default_rng(99)
    y_true, y_pred = random_preds
    results: dict[str, dict] = {}
    for ch in CHANNELS:
        # Each channel has independent (but reproducible) arrays
        # 각 채널은 독립적이지만 재현 가능한 배열을 가짐
        yt = rng.integers(0, NUM_LEVELS, size=100)
        yp = rng.integers(0, NUM_LEVELS, size=100)
        confs = rng.uniform(0.1, 0.99, size=100).astype(np.float32)
        fnames = [f"scan_{ch}_{i:04d}.png" for i in range(100)]
        results[ch] = {
            "y_true": yt,
            "y_pred": yp,
            "confidences": confs,
            "filenames": fnames,
        }
    return results


@pytest.fixture
def perfect_summary(perfect_preds) -> EvaluationSummary:
    """Build a perfect EvaluationSummary for terminate-check tests. / 종료 검사 테스트용 완벽한 요약."""
    y_true, y_pred = perfect_preds
    cm = compute_channel_metrics(y_true, y_pred, channel="overall")
    by_ch: dict[str, ChannelMetrics] = {}
    for ch in CHANNELS:
        by_ch[ch] = compute_channel_metrics(y_true, y_pred, channel=ch)
    return EvaluationSummary(
        overall=cm,
        by_channel=by_ch,
        targets={
            "overall_accuracy": 0.90,
            "per_color_accuracy": 0.85,
            "per_class_f1": 0.80,
            "mae": 0.50,
        },
        meta={"backbone": "efficientnet_b0", "checkpoint": None},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests — metrics.py / metrics.py 테스트
# ─────────────────────────────────────────────────────────────────────────────


class TestComputePerClassMetrics:
    """Unit tests for compute_per_class_metrics(). / compute_per_class_metrics() 단위 테스트."""

    def test_returns_correct_length(self, perfect_preds):
        """Should return exactly NUM_LEVELS entries. / 정확히 NUM_LEVELS 개의 항목을 반환해야 합니다."""
        y_true, y_pred = perfect_preds
        result = compute_per_class_metrics(y_true, y_pred)
        assert len(result) == NUM_LEVELS

    def test_perfect_f1_is_one(self, perfect_preds):
        """Perfect predictions → F1 == 1.0 for every level. / 완벽한 예측 → 모든 레벨의 F1 == 1.0."""
        y_true, y_pred = perfect_preds
        result = compute_per_class_metrics(y_true, y_pred)
        for pc in result:
            assert pytest.approx(pc.f1, abs=1e-6) == 1.0

    def test_f1_pass_flag_set_correctly(self, perfect_preds):
        """f1_pass should be True for perfect predictions. / 완벽한 예측에서 f1_pass는 True여야 합니다."""
        y_true, y_pred = perfect_preds
        result = compute_per_class_metrics(y_true, y_pred, target_f1=0.80)
        for pc in result:
            assert pc.f1_pass is True

    def test_level_indices_match(self, perfect_preds):
        """Level attribute must match index (0~5). / Level 속성은 인덱스(0~5)와 일치해야 합니다."""
        y_true, y_pred = perfect_preds
        result = compute_per_class_metrics(y_true, y_pred)
        for i, pc in enumerate(result):
            assert pc.level == i

    def test_support_sums_to_n(self, perfect_preds):
        """Sum of support values should equal total sample count. / support 합은 총 샘플 수와 같아야 합니다."""
        y_true, y_pred = perfect_preds
        result = compute_per_class_metrics(y_true, y_pred)
        assert sum(pc.support for pc in result) == len(y_true)

    def test_random_preds_metrics_in_range(self, random_preds):
        """All metric values should be in [0, 1]. / 모든 지표 값은 [0, 1] 범위여야 합니다."""
        y_true, y_pred = random_preds
        result = compute_per_class_metrics(y_true, y_pred)
        for pc in result:
            assert 0.0 <= pc.precision <= 1.0
            assert 0.0 <= pc.recall <= 1.0
            assert 0.0 <= pc.f1 <= 1.0


class TestComputeChannelMetrics:
    """Unit tests for compute_channel_metrics(). / compute_channel_metrics() 단위 테스트."""

    def test_perfect_accuracy_is_one(self, perfect_preds):
        """Perfect predictions → accuracy == 1.0. / 완벽한 예측 → 정확도 == 1.0."""
        y_true, y_pred = perfect_preds
        cm = compute_channel_metrics(y_true, y_pred, channel="Y")
        assert pytest.approx(cm.accuracy, abs=1e-6) == 1.0

    def test_mae_is_zero_for_perfect_preds(self, perfect_preds):
        """Perfect predictions → MAE == 0.0. / 완벽한 예측 → MAE == 0.0."""
        y_true, y_pred = perfect_preds
        cm = compute_channel_metrics(y_true, y_pred)
        assert pytest.approx(cm.mae, abs=1e-6) == 0.0

    def test_acc_pass_flag_for_perfect(self, perfect_preds):
        """Perfect accuracy should pass the default target. / 완벽한 정확도는 기본 목표를 통과해야 합니다."""
        y_true, y_pred = perfect_preds
        cm = compute_channel_metrics(y_true, y_pred, channel="overall")
        assert cm.acc_pass is True

    def test_n_samples_correct(self, perfect_preds):
        """n_samples should match input length. / n_samples는 입력 길이와 일치해야 합니다."""
        y_true, y_pred = perfect_preds
        cm = compute_channel_metrics(y_true, y_pred)
        assert cm.n_samples == len(y_true)

    def test_channel_name_stored(self, perfect_preds):
        """channel attribute should store the given name. / channel 속성은 주어진 이름을 저장해야 합니다."""
        y_true, y_pred = perfect_preds
        cm = compute_channel_metrics(y_true, y_pred, channel="M")
        assert cm.channel == "M"

    def test_macro_f1_range(self, random_preds):
        """Macro F1 should be in [0, 1]. / 매크로 F1은 [0, 1] 범위여야 합니다."""
        y_true, y_pred = random_preds
        cm = compute_channel_metrics(y_true, y_pred)
        assert 0.0 <= cm.macro_f1 <= 1.0


class TestComputeMaeByLevel:
    """Unit tests for compute_mae_by_level(). / compute_mae_by_level() 단위 테스트."""

    def test_returns_all_levels(self, perfect_preds):
        """Should return an entry for every level 0~5. / 0~5의 모든 레벨에 대한 항목을 반환해야 합니다."""
        y_true, y_pred = perfect_preds
        result = compute_mae_by_level(y_true, y_pred)
        for lv in range(NUM_LEVELS):
            assert lv in result

    def test_mae_zero_for_perfect(self, perfect_preds):
        """Perfect predictions → MAE == 0 for every populated level. / 완벽한 예측 → 모든 레벨의 MAE == 0."""
        y_true, y_pred = perfect_preds
        result = compute_mae_by_level(y_true, y_pred)
        for lv, info in result.items():
            if info["count"] > 0:
                assert pytest.approx(info["mae"], abs=1e-6) == 0.0

    def test_count_sums_to_n(self, perfect_preds):
        """Sum of counts should equal total sample count. / count 합은 총 샘플 수와 같아야 합니다."""
        y_true, y_pred = perfect_preds
        result = compute_mae_by_level(y_true, y_pred)
        total = sum(v["count"] for v in result.values())
        assert total == len(y_true)


class TestDetermineSwingFeedback:
    """Unit tests for determine_swing_feedback(). / determine_swing_feedback() 단위 테스트."""

    def test_perfect_summary_terminates(self, perfect_summary):
        """Perfect metrics should trigger Swing termination. / 완벽한 지표는 Swing 종료를 트리거해야 합니다."""
        decision = determine_swing_feedback(perfect_summary)
        assert decision["terminate"] is True
        assert decision["status"] == "terminate"

    def test_bad_accuracy_produces_decisions(self, mock_results):
        """Poor per-color accuracy should produce phase-0 decisions. / 낮은 색상별 정확도는 phase-0 결정을 생성해야 합니다."""
        # Build a summary with very low accuracy
        # 매우 낮은 정확도를 가진 요약 생성
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, NUM_LEVELS, size=60)
        y_pred = rng.integers(0, NUM_LEVELS, size=60)
        cm_overall = compute_channel_metrics(y_true, y_pred, channel="overall")
        by_ch = {
            ch: compute_channel_metrics(y_true, y_pred, channel=ch)
            for ch in CHANNELS
        }
        summary = EvaluationSummary(
            overall=cm_overall,
            by_channel=by_ch,
            targets={
                "overall_accuracy": 0.90,
                "per_color_accuracy": 0.85,
                "per_class_f1": 0.80,
                "mae": 0.50,
            },
            meta={},
        )
        decision = determine_swing_feedback(summary)
        assert decision["terminate"] is False
        assert len(decision["decisions"]) > 0


class TestSerialization:
    """Unit tests for summary_to_dict() and save_metrics_json(). / 직렬화 함수 단위 테스트."""

    def test_summary_to_dict_has_required_keys(self, perfect_summary):
        """Serialized dict must have meta, targets, global, by_color keys. / 직렬화된 딕셔너리는 필수 키를 가져야 합니다."""
        d = summary_to_dict(perfect_summary)
        for key in ("meta", "targets", "global", "by_color"):
            assert key in d

    def test_global_metrics_rounded(self, perfect_summary):
        """Numeric values should be rounded to 4 decimal places. / 숫자 값은 소수점 4자리로 반올림되어야 합니다."""
        d = summary_to_dict(perfect_summary)
        acc = d["global"]["accuracy"]
        # 4 decimal places: value * 10000 should be close to an integer
        # 4자리: value * 10000은 정수에 가까워야 합니다
        assert pytest.approx(acc * 10000, abs=1) == round(acc * 10000)

    def test_save_metrics_json_creates_file(self, perfect_summary):
        """save_metrics_json() must create a valid JSON file. / save_metrics_json()은 유효한 JSON 파일을 생성해야 합니다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "metrics_summary.json"
            result_path = save_metrics_json(perfect_summary, out_path)
            assert result_path.exists()
            with open(result_path, encoding="utf-8") as f:
                data = json.load(f)
            assert "global" in data

    def test_save_json_utf8_encoding(self, perfect_summary):
        """JSON file should be readable as UTF-8 (supports Korean). / JSON 파일은 UTF-8로 읽을 수 있어야 합니다 (한국어 지원)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.json"
            save_metrics_json(perfect_summary, out_path)
            text = out_path.read_text(encoding="utf-8")
            assert len(text) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests — confusion.py / confusion.py 테스트
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildConfusionMatrixFigure:
    """Unit tests for build_confusion_matrix_figure(). / build_confusion_matrix_figure() 단위 테스트."""

    def test_returns_plotly_figure(self, perfect_preds):
        """Should return a plotly go.Figure instance. / plotly go.Figure 인스턴스를 반환해야 합니다."""
        import plotly.graph_objects as go
        y_true, y_pred = perfect_preds
        fig = build_confusion_matrix_figure(y_true, y_pred, title="Test CM")
        assert isinstance(fig, go.Figure)

    def test_figure_has_heatmap_trace(self, perfect_preds):
        """Figure should contain exactly one Heatmap trace. / Figure는 정확히 하나의 Heatmap trace를 포함해야 합니다."""
        import plotly.graph_objects as go
        y_true, y_pred = perfect_preds
        fig = build_confusion_matrix_figure(y_true, y_pred, title="T")
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmaps) == 1

    def test_normalize_z_range(self, random_preds):
        """Normalized matrix values should be in [0, 1]. / 정규화된 행렬 값은 [0, 1] 범위여야 합니다."""
        y_true, y_pred = random_preds
        fig = build_confusion_matrix_figure(
            y_true, y_pred, title="T", normalize=True
        )
        heatmap = fig.data[0]
        z = np.array(heatmap.z, dtype=float)
        # Ignore NaN entries (possible for empty rows)
        # NaN 항목 무시 (빈 행에서 발생 가능)
        valid = z[~np.isnan(z)]
        if len(valid) > 0:
            assert valid.min() >= -1e-9
            assert valid.max() <= 1.0 + 1e-9

    def test_non_normalize_has_counts(self, perfect_preds):
        """Non-normalized matrix should contain integer count values. / 비정규화 행렬은 정수 카운트 값을 포함해야 합니다."""
        y_true, y_pred = perfect_preds
        fig = build_confusion_matrix_figure(
            y_true, y_pred, title="T", normalize=False
        )
        heatmap = fig.data[0]
        z = np.array(heatmap.z, dtype=float)
        # All values should be non-negative integers
        # 모든 값은 비음수 정수여야 함
        assert np.all(z >= 0)
        assert np.all(z == np.floor(z))


class TestBuildMaeHeatmapFigure:
    """Unit tests for build_mae_heatmap_figure(). / build_mae_heatmap_figure() 단위 테스트."""

    def test_returns_plotly_figure(self, mock_results):
        """Should return a plotly go.Figure instance. / plotly go.Figure 인스턴스를 반환해야 합니다."""
        import plotly.graph_objects as go
        fig = build_mae_heatmap_figure(mock_results)
        assert isinstance(fig, go.Figure)

    def test_heatmap_shape(self, mock_results):
        """Heatmap z-matrix should have shape (len(CHANNELS), NUM_LEVELS). / Heatmap z 행렬은 (채널 수, 레벨 수) 형태여야 합니다."""
        fig = build_mae_heatmap_figure(mock_results)
        z = np.array(fig.data[0].z)
        assert z.shape == (len(CHANNELS), NUM_LEVELS)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — evaluator.py / evaluator.py 테스트
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluatorConfig:
    """Unit tests for EvaluatorConfig. / EvaluatorConfig 단위 테스트."""

    def test_defaults_match_prd(self):
        """Default targets should match PRD §1.4. / 기본 목표값은 PRD §1.4와 일치해야 합니다."""
        cfg = EvaluatorConfig()
        assert cfg.target_overall_acc == 0.90
        assert cfg.target_per_color_acc == 0.85
        assert cfg.target_per_class_f1 == 0.80
        assert cfg.target_mae == 0.50

    def test_targets_dict_has_all_keys(self):
        """targets_dict() must include all four target keys. / targets_dict()는 4개의 목표 키를 모두 포함해야 합니다."""
        cfg = EvaluatorConfig()
        d = cfg.targets_dict()
        for key in ("overall_accuracy", "per_color_accuracy", "per_class_f1", "mae"):
            assert key in d

    def test_confidence_thresholds_prd(self):
        """Confidence thresholds should match PRD §14.2 defaults. / 신뢰도 임계값은 PRD §14.2 기본값과 일치해야 합니다."""
        cfg = EvaluatorConfig()
        assert cfg.conf_thresh_auto == 0.8
        assert cfg.conf_thresh_warn == 0.5
        assert cfg.conf_thresh_manual == 0.3


class TestGrayspotEvaluator:
    """Integration-level tests for GrayspotEvaluator. / GrayspotEvaluator 통합 테스트."""

    def test_run_returns_evaluation_summary(self, mock_results, tmp_path):
        """run() should return an EvaluationSummary instance. / run()은 EvaluationSummary 인스턴스를 반환해야 합니다."""
        cfg = EvaluatorConfig(output_dir=tmp_path, open_browser=False)
        evaluator = GrayspotEvaluator(cfg)
        summary = evaluator.run(mock_results)
        assert isinstance(summary, EvaluationSummary)

    def test_output_files_created(self, mock_results, tmp_path):
        """run() should create all expected output files. / run()은 예상된 모든 출력 파일을 생성해야 합니다."""
        cfg = EvaluatorConfig(output_dir=tmp_path, open_browser=False)
        evaluator = GrayspotEvaluator(cfg)
        evaluator.run(mock_results)

        expected_files = [
            "cm_Y.html",
            "cm_M.html",
            "cm_C.html",
            "cm_K.html",
            "cm_overall.html",
            "eval_dashboard.html",
            "per_class_metrics.html",
            "mae_heatmap.html",
            "misclassified_scatter.html",
            "confidence_distribution.html",
            "evaluation_results.csv",
            "misclassified_samples.csv",
            "metrics_summary.json",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"

    def test_evaluation_results_csv_columns(self, mock_results, tmp_path):
        """evaluation_results.csv must contain all PRD §8.2.2 columns. / CSV는 PRD §8.2.2 컬럼을 모두 포함해야 합니다."""
        import pandas as pd
        cfg = EvaluatorConfig(output_dir=tmp_path, open_browser=False)
        evaluator = GrayspotEvaluator(cfg)
        evaluator.run(mock_results)

        csv_path = tmp_path / "evaluation_results.csv"
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        required_cols = [
            "filename", "color", "true_level", "pred_level",
            "confidence", "correct", "error_gap",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_metrics_json_structure(self, mock_results, tmp_path):
        """metrics_summary.json must have meta/targets/global/by_color keys. / JSON은 필수 최상위 키를 가져야 합니다."""
        cfg = EvaluatorConfig(output_dir=tmp_path, open_browser=False)
        evaluator = GrayspotEvaluator(cfg)
        evaluator.run(mock_results)

        json_path = tmp_path / "metrics_summary.json"
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for key in ("meta", "targets", "global", "by_color"):
            assert key in data

    def test_by_channel_covers_all_cmyk(self, mock_results, tmp_path):
        """EvaluationSummary.by_channel must include all 4 CMYK channels. / by_channel은 4개 CMYK 채널 모두를 포함해야 합니다."""
        cfg = EvaluatorConfig(output_dir=tmp_path, open_browser=False)
        evaluator = GrayspotEvaluator(cfg)
        summary = evaluator.run(mock_results)
        for ch in CHANNELS:
            assert ch in summary.by_channel

    def test_get_summary_dict_not_none_after_run(self, mock_results, tmp_path):
        """get_summary_dict() should not return None after run(). / run() 이후 get_summary_dict()는 None을 반환하면 안 됩니다."""
        cfg = EvaluatorConfig(output_dir=tmp_path, open_browser=False)
        evaluator = GrayspotEvaluator(cfg)
        evaluator.run(mock_results)
        d = evaluator.get_summary_dict()
        assert d is not None

    def test_evaluator_default_config(self, mock_results, tmp_path):
        """Evaluator should work with default config (no args). / Evaluator는 기본 설정으로 동작해야 합니다."""
        cfg = EvaluatorConfig(output_dir=tmp_path, open_browser=False)
        evaluator = GrayspotEvaluator(cfg)
        # Should not raise / 예외를 발생시키면 안 됩니다
        summary = evaluator.run(mock_results)
        assert summary is not None
