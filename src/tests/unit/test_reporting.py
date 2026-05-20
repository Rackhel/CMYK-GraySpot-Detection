"""
tests/unit/test_reporting.py

reporting/html_report.py 단위 테스트.
Unit tests for reporting/html_report.py.

테스트 대상 / Test targets:
    - generate_baseline_report()   BDD Scenario 4.6
    - summary_to_dict()            BDD Scenario 4.4

BDD 매핑 / BDD mapping:
    Scenario 4.1 — HTML 대시보드 생성 검증 (대시보드 마커 확인)
    Scenario 4.2 — 채널별 혼동 행렬 탭 확인
    Scenario 4.4 — 집계 지표 JSON 직렬화 구조 검증
    Scenario 4.6 — generate_baseline_report() 독립 실행형 HTML 생성

TDD 전략 / TDD strategy: TDD.md §3.8
    - 실제 파일 / 브라우저 의존성 없이 tmp_path 사용
    - AAA(Arrange-Act-Assert) 패턴 적용
    - 함수 1개당 검증 1개

Python 3.11.5 | pytest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

# ── sys.path 설정 / sys.path setup ──────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # CMYK_MAIN/
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from evaluation.metrics import (
    ChannelMetrics,
    EvaluationSummary,
    PerClassMetric,
    summary_to_dict,
)
from reporting.html_report import generate_baseline_report


# ── 로컬 Fixture 헬퍼 / Local fixture helpers ────────────────────────────────


def _make_per_class(f1: float = 0.90, n_levels: int = 6) -> list[PerClassMetric]:
    """
    테스트용 PerClassMetric 리스트를 생성한다.
    Creates a PerClassMetric list for testing.
    """
    return [
        PerClassMetric(
            level=i,
            precision=f1,
            recall=f1,
            f1=f1,
            support=20,
        )
        for i in range(n_levels)
    ]


def _make_channel_metrics(
    accuracy: float = 0.92,
    macro_f1: float = 0.88,
    mae: float = 0.25,
    n_samples: int = 120,
) -> ChannelMetrics:
    """
    테스트용 ChannelMetrics 객체를 생성한다.
    Creates a ChannelMetrics object for testing.
    """
    return ChannelMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        mae=mae,
        n_samples=n_samples,
        per_class=_make_per_class(macro_f1),
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def good_channel_metrics() -> ChannelMetrics:
    """목표치를 초과하는 단일 채널 지표. / Single-channel metrics that exceed all targets."""
    return _make_channel_metrics(accuracy=0.92, macro_f1=0.88, mae=0.25)


@pytest.fixture
def mock_summary(good_channel_metrics) -> EvaluationSummary:
    """
    실제 파일 없이 동작하는 EvaluationSummary fixture.
    EvaluationSummary fixture that works without real files.

    overall + 4채널 all-pass 상태로 구성.
    Configured with overall + 4 channels all in pass state.
    """
    return EvaluationSummary(
        overall=good_channel_metrics,
        by_channel={
            "Y": good_channel_metrics,
            "M": good_channel_metrics,
            "C": good_channel_metrics,
            "K": good_channel_metrics,
        },
        meta={"backbone": "efficientnet_b0", "checkpoint": None},
        targets={
            "overall_accuracy": 0.90,
            "per_color_accuracy": 0.85,
            "per_class_f1": 0.80,
            "mae": 0.50,
            "swing_acc_retry": 0.80,
            "swing_f1_retry": 0.70,
            "swing_mae_retry": 0.80,
        },
    )


@pytest.fixture
def mock_results() -> Dict[str, dict]:
    """
    채널별 추론 결과 fixture — y_true, y_pred, confidences, filenames 포함.
    Per-channel inference results fixture — includes y_true, y_pred, confidences, filenames.

    confidence 분포 차트 생성에 필요한 confidences 배열을 포함한다.
    Includes confidences array required for confidence distribution chart generation.
    """
    rng = np.random.default_rng(42)
    result: Dict[str, dict] = {}
    for ch in ["Y", "M", "C", "K"]:
        n = 30
        y_true = rng.integers(0, 6, n)
        # 근접 예측 — 현실적인 오차 범위 / Near-correct predictions — realistic error range
        y_pred = np.clip(y_true + rng.integers(-1, 2, n), 0, 5)
        confidences = rng.uniform(0.4, 1.0, n).astype(np.float32)
        result[ch] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "confidences": confidences,
            "filenames": [f"{ch}_patch_{i:04d}.png" for i in range(n)],
        }
    return result


@pytest.fixture
def failing_summary() -> EvaluationSummary:
    """
    모든 목표치 미달인 EvaluationSummary — Feedback 섹션의 '조치 필요' 분기 테스트용.
    EvaluationSummary with all targets failing — tests the 'action required' branch.
    """
    bad_cm = _make_channel_metrics(accuracy=0.50, macro_f1=0.45, mae=1.20)
    return EvaluationSummary(
        overall=bad_cm,
        by_channel={"Y": bad_cm, "M": bad_cm, "C": bad_cm, "K": bad_cm},
        meta={"backbone": "efficientnet_b0"},
        targets={
            "overall_accuracy": 0.90,
            "per_color_accuracy": 0.85,
            "per_class_f1": 0.80,
            "mae": 0.50,
            "swing_acc_retry": 0.80,
            "swing_f1_retry": 0.70,
            "swing_mae_retry": 0.80,
        },
    )


# ── generate_baseline_report ─────────────────────────────────────────────────
# TDD §3.8 | BDD Scenario 4.6


class TestGenerateBaselineReport:
    """generate_baseline_report() 단위 테스트. / Unit tests for generate_baseline_report()."""

    # ── 반환값 / Return value ─────────────────────────────────────────────

    def test_returns_path_type(self, tmp_path, mock_summary, mock_results):
        """반환 타입이 Path 이다. / Return type is Path."""
        # Arrange
        out = tmp_path / "baseline.html"
        # Act
        result = generate_baseline_report(mock_summary, mock_results, output_path=out)
        # Assert
        assert isinstance(result, Path)

    def test_returns_absolute_path(self, tmp_path, mock_summary, mock_results):
        """반환 경로가 절대 경로다. / Returned path is absolute."""
        out = tmp_path / "baseline.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        assert path.is_absolute()

    def test_output_has_html_suffix(self, tmp_path, mock_summary, mock_results):
        """반환 경로의 확장자가 .html 이다. / Returned path has .html extension."""
        out = tmp_path / "test_report.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        assert path.suffix == ".html"

    # ── 파일 생성 / File creation ─────────────────────────────────────────

    def test_html_file_is_created(self, tmp_path, mock_summary, mock_results):
        """지정 경로에 HTML 파일이 생성된다. / HTML file is created at given path."""
        # BDD Scenario 4.6 — 파일 생성 검증
        out = tmp_path / "baseline.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        assert path.exists()

    def test_creates_parent_dir_if_not_exists(self, tmp_path, mock_summary, mock_results):
        """중간 디렉토리가 없어도 자동 생성된다. / Intermediate directories are created automatically."""
        out = tmp_path / "nested" / "deep" / "report.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        assert path.exists()

    # ── HTML 내용 / HTML content ──────────────────────────────────────────

    def test_html_starts_with_doctype(self, tmp_path, mock_summary, mock_results):
        """생성된 HTML에 <!DOCTYPE html>이 포함된다 (BDD 4.6). / Generated HTML contains <!DOCTYPE html>."""
        out = tmp_path / "baseline.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_html_contains_plotly_cdn(self, tmp_path, mock_summary, mock_results):
        """
        Plotly CDN이 포함된다 — 외부 에셋 없이 독립 실행 가능 (BDD 4.6).
        Plotly CDN is included — self-contained without local assets.
        """
        out = tmp_path / "baseline.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        content = path.read_text(encoding="utf-8")
        # CDN URL 또는 plotly 스크립트 태그 확인 / Check for CDN URL or plotly script tag
        assert "plotly" in content.lower()

    def test_html_contains_all_channel_labels(self, tmp_path, mock_summary, mock_results):
        """채널 레이블(Y, M, C, K)이 HTML에 포함된다 (BDD 4.1 — 색상별 막대 차트 확인)."""
        out = tmp_path / "baseline.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        content = path.read_text(encoding="utf-8")
        for ch in ["Y", "M", "C", "K"]:
            assert ch in content

    def test_html_contains_accuracy_section(self, tmp_path, mock_summary, mock_results):
        """정확도 관련 내용이 HTML에 포함된다 (BDD 4.1 — Gauge 차트)."""
        out = tmp_path / "baseline.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        content = path.read_text(encoding="utf-8")
        assert "Accuracy" in content or "accuracy" in content

    def test_html_contains_feedback_section(self, tmp_path, mock_summary, mock_results):
        """Phase 3 Feedback 탭이 HTML에 포함된다 (BDD 4.1 — 탭 구조)."""
        out = tmp_path / "baseline.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        content = path.read_text(encoding="utf-8")
        assert "Feedback" in content or "feedback" in content

    def test_html_contains_confusion_section(self, tmp_path, mock_summary, mock_results):
        """혼동 행렬 탭이 HTML에 포함된다 (BDD 4.2 — Confusion Matrix 탭)."""
        out = tmp_path / "baseline.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        content = path.read_text(encoding="utf-8")
        assert "Confusion" in content or "confusion" in content

    def test_html_is_utf8_encoded(self, tmp_path, mock_summary, mock_results):
        """HTML 파일이 UTF-8로 읽힌다 (한글 포함). / HTML file is UTF-8 readable (including Korean)."""
        out = tmp_path / "baseline.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        # UnicodeDecodeError 없이 읽혀야 한다 / Must be readable without UnicodeDecodeError
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_html_file_is_non_empty(self, tmp_path, mock_summary, mock_results):
        """HTML 파일이 비어있지 않다. / HTML file is non-empty."""
        out = tmp_path / "baseline.html"
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        assert path.stat().st_size > 0

    # ── 경계 / 옵션 케이스 / Edge & option cases ──────────────────────────

    def test_single_channel_does_not_raise(self, tmp_path, mock_summary, mock_results):
        """단일 채널만 전달해도 예외가 발생하지 않는다. / Single channel does not raise."""
        out = tmp_path / "single_ch.html"
        generate_baseline_report(
            mock_summary, mock_results, output_path=out, channels=["Y"]
        )

    def test_open_browser_false_does_not_open(
        self, tmp_path, mock_summary, mock_results, monkeypatch
    ):
        """open_browser=False 이면 브라우저가 열리지 않는다. / open_browser=False skips browser open."""
        opened = []
        monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
        out = tmp_path / "baseline.html"
        generate_baseline_report(
            mock_summary, mock_results, output_path=out, open_browser=False
        )
        assert len(opened) == 0

    def test_failing_summary_generates_html(
        self, tmp_path, failing_summary, mock_results
    ):
        """목표치 미달 summary 에서도 HTML이 정상 생성된다. / HTML is generated even when targets are not met."""
        out = tmp_path / "fail_report.html"
        path = generate_baseline_report(
            failing_summary, mock_results, output_path=out
        )
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_custom_output_path_is_respected(
        self, tmp_path, mock_summary, mock_results
    ):
        """사용자 지정 경로에 파일이 생성된다. / File is created at the custom output path."""
        custom_name = "my_custom_report.html"
        out = tmp_path / custom_name
        path = generate_baseline_report(mock_summary, mock_results, output_path=out)
        assert path.name == custom_name


# ── summary_to_dict ────────────────────────────────────────────────────────────
# TDD §3.8 | BDD Scenario 4.4


class TestSummaryToDict:
    """summary_to_dict() 단위 테스트. / Unit tests for summary_to_dict()."""

    # ── 반환 타입 / Return type ───────────────────────────────────────────

    def test_returns_dict(self, mock_summary):
        """반환 타입이 dict 이다. / Return type is dict."""
        result = summary_to_dict(mock_summary)
        assert isinstance(result, dict)

    # ── 필수 키 존재 / Required keys present ─────────────────────────────
    # BDD Scenario 4.4: global, by_color, per_class_overall 섹션 확인

    def test_has_overall_key(self, mock_summary):
        """'overall' 키가 존재한다. / 'overall' key is present."""
        result = summary_to_dict(mock_summary)
        assert "overall" in result

    def test_has_by_channel_key(self, mock_summary):
        """'by_channel' 키가 존재한다. / 'by_channel' key is present."""
        result = summary_to_dict(mock_summary)
        assert "by_channel" in result

    def test_has_meta_key(self, mock_summary):
        """'meta' 키가 존재한다. / 'meta' key is present."""
        result = summary_to_dict(mock_summary)
        assert "meta" in result

    def test_has_targets_key(self, mock_summary):
        """'targets' 키가 존재한다. / 'targets' key is present."""
        result = summary_to_dict(mock_summary)
        assert "targets" in result

    # ── JSON 직렬화 / JSON serialization ──────────────────────────────────

    def test_is_json_serializable(self, mock_summary):
        """json.dumps()로 직렬화 가능하다 (BDD 4.4). / Serializable with json.dumps()."""
        result = summary_to_dict(mock_summary)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_roundtrip_preserves_accuracy(self, mock_summary):
        """직렬화-역직렬화 후 overall accuracy 값이 보존된다. / Round-trip preserves overall accuracy."""
        result = summary_to_dict(mock_summary)
        restored = json.loads(json.dumps(result))
        assert restored["overall"]["accuracy"] == pytest.approx(
            result["overall"]["accuracy"]
        )

    # ── overall 구조 / overall structure ─────────────────────────────────

    def test_overall_has_accuracy(self, mock_summary):
        """'overall' 에 accuracy 가 있다. / 'overall' has accuracy."""
        result = summary_to_dict(mock_summary)
        assert "accuracy" in result["overall"]

    def test_overall_has_macro_f1(self, mock_summary):
        """'overall' 에 macro_f1 가 있다. / 'overall' has macro_f1."""
        result = summary_to_dict(mock_summary)
        assert "macro_f1" in result["overall"]

    def test_overall_has_mae(self, mock_summary):
        """'overall' 에 mae 가 있다. / 'overall' has mae."""
        result = summary_to_dict(mock_summary)
        assert "mae" in result["overall"]

    def test_overall_has_n_samples(self, mock_summary):
        """'overall' 에 n_samples 가 있다. / 'overall' has n_samples."""
        result = summary_to_dict(mock_summary)
        assert "n_samples" in result["overall"]

    def test_overall_has_per_class(self, mock_summary):
        """'overall' 에 per_class 리스트가 있다. / 'overall' has per_class list."""
        result = summary_to_dict(mock_summary)
        assert "per_class" in result["overall"]
        assert isinstance(result["overall"]["per_class"], list)

    def test_per_class_has_6_entries(self, mock_summary):
        """per_class 배열이 6개 항목을 가진다. / per_class list has 6 entries."""
        result = summary_to_dict(mock_summary)
        assert len(result["overall"]["per_class"]) == 6

    def test_per_class_entry_has_required_keys(self, mock_summary):
        """per_class 각 항목에 level/precision/recall/f1/support 키가 있다."""
        result = summary_to_dict(mock_summary)
        required = {"level", "precision", "recall", "f1", "support"}
        for pc in result["overall"]["per_class"]:
            assert required <= set(pc.keys())

    # ── by_channel 구조 / by_channel structure ───────────────────────────

    def test_by_channel_contains_all_channels(self, mock_summary):
        """by_channel 에 Y/M/C/K 키가 포함된다. / by_channel contains Y/M/C/K keys."""
        result = summary_to_dict(mock_summary)
        for ch in ["Y", "M", "C", "K"]:
            assert ch in result["by_channel"]

    def test_by_channel_each_has_accuracy(self, mock_summary):
        """by_channel 각 채널에 accuracy 가 있다. / Each channel in by_channel has accuracy."""
        result = summary_to_dict(mock_summary)
        for ch in ["Y", "M", "C", "K"]:
            assert "accuracy" in result["by_channel"][ch]

    # ── meta / targets ────────────────────────────────────────────────────

    def test_meta_backbone_preserved(self, mock_summary):
        """meta 의 backbone 값이 보존된다. / meta backbone value is preserved."""
        result = summary_to_dict(mock_summary)
        assert result["meta"]["backbone"] == "efficientnet_b0"

    def test_targets_values_are_numeric(self, mock_summary):
        """targets 딕셔너리의 값이 숫자형이다. / Targets values are numeric."""
        result = summary_to_dict(mock_summary)
        for val in result["targets"].values():
            assert isinstance(val, (int, float))

    # ── 경계 케이스 / Edge cases ──────────────────────────────────────────

    def test_empty_by_channel_does_not_raise(self):
        """by_channel 이 비어있어도 예외가 발생하지 않는다. / Empty by_channel does not raise."""
        pc = _make_per_class(0.80)
        summary = EvaluationSummary(
            overall=ChannelMetrics(
                accuracy=0.0, macro_f1=0.0, mae=0.0, n_samples=0, per_class=pc
            ),
            by_channel={},
        )
        result = summary_to_dict(summary)
        assert result["by_channel"] == {}

    def test_accuracy_value_is_float(self, mock_summary):
        """overall accuracy 가 float 타입이다. / overall accuracy is float type."""
        result = summary_to_dict(mock_summary)
        assert isinstance(result["overall"]["accuracy"], float)

    def test_n_samples_value_is_int(self, mock_summary):
        """overall n_samples 가 int 타입이다. / overall n_samples is int type."""
        result = summary_to_dict(mock_summary)
        assert isinstance(result["overall"]["n_samples"], int)
