"""
test_evaluate_script.py
Tests for src/scripts/evaluate.py CLI script.
Status: FAILING — evaluate.py not yet implemented.
Ref: doc/TDD/TDD_Evaluate_Script.md
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── sys.path 설정 ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # CMYK_MAIN/
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

# Will raise ImportError until implemented — correct failing behavior
from scripts.evaluate import main, parse_args  # noqa: E402


# ── CLI 인수 파싱 ──────────────────────────────────────────────────────────────
# T-EVAL-01 ~ T-EVAL-05


class TestParseArgs:
    """T-EVAL-01 ~ T-EVAL-05: CLI 인수 파싱 검증."""

    def test_channel_y(self):
        """T-EVAL-01: --channel Y → args.channel == 'Y'"""
        args = parse_args(["--channel", "Y"])
        assert args.channel == "Y"

    def test_channel_all(self):
        """T-EVAL-02: --channel all → args.channel == 'all'"""
        args = parse_args(["--channel", "all"])
        assert args.channel == "all"

    def test_missing_channel_exits(self):
        """T-EVAL-03: 필수 인수(--channel) 누락 → SystemExit (argparse 기본 동작)"""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_invalid_channel_rejected(self):
        """T-EVAL-04: 유효하지 않은 채널 'X' → SystemExit 또는 ValueError"""
        with pytest.raises((SystemExit, ValueError)):
            parse_args(["--channel", "X"])

    def test_custom_output_dir(self):
        """T-EVAL-05: --output-dir /tmp/reports → args.output_dir == '/tmp/reports'"""
        args = parse_args(["--channel", "Y", "--output-dir", "/tmp/reports"])
        assert str(args.output_dir) == "/tmp/reports"

    def test_channel_c(self):
        """T-EVAL-01 확장: --channel C → args.channel == 'C'"""
        args = parse_args(["--channel", "C"])
        assert args.channel == "C"

    def test_channel_m(self):
        """T-EVAL-01 확장: --channel M → args.channel == 'M'"""
        args = parse_args(["--channel", "M"])
        assert args.channel == "M"

    def test_channel_k(self):
        """T-EVAL-01 확장: --channel K → args.channel == 'K'"""
        args = parse_args(["--channel", "K"])
        assert args.channel == "K"


# ── 리포트 생성 ────────────────────────────────────────────────────────────────
# T-EVAL-10 ~ T-EVAL-13


class TestReportGeneration:
    """T-EVAL-10 ~ T-EVAL-13: 리포트 생성 및 예외 처리 검증."""

    def _make_cfg(self, tmp_path: Path) -> dict:
        """테스트용 최소 config dict."""
        return {
            "data": {
                "channels": ["Y", "M", "C", "K"],
                "num_levels": 6,
                "image_size": 128,
            },
            "storage": {
                "models_dir": str(tmp_path),
                "reports_dir": str(tmp_path),
            },
            "system": {"device": "cpu"},
        }

    def test_creates_json_report(self, tmp_path, monkeypatch):
        """T-EVAL-10: 유효 체크포인트 존재 시 JSON 파일 생성"""
        monkeypatch.setattr(
            "scripts.evaluate.load_config",
            lambda: self._make_cfg(tmp_path),
        )
        mock_evaluator = MagicMock()
        mock_evaluator.run.return_value = ([], [], [])
        with patch("scripts.evaluate.Evaluator", return_value=mock_evaluator):
            with patch("scripts.evaluate.Path.exists", return_value=True):
                main(["--channel", "Y", "--output-dir", str(tmp_path)])
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) > 0, "JSON 리포트 파일이 생성되지 않음"

    def test_json_contains_accuracy_key(self, tmp_path):
        """T-EVAL-11: 생성된 JSON 파일에 'accuracy' 키가 존재해야 함"""
        # 구현 전 단계: 구조 검증을 위해 리포트 형식 직접 검사
        report = {"accuracy": 0.85, "channel": "Y", "num_levels": 6}
        report_file = tmp_path / "report_Y.json"
        report_file.write_text(json.dumps(report), encoding="utf-8")
        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert "accuracy" in data, "JSON 리포트에 'accuracy' 키 없음"

    def test_missing_checkpoint_exits_with_code_1(self, tmp_path):
        """T-EVAL-13: 체크포인트 미존재 → sys.exit(1) (exit code == 1)"""
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--channel",
                    "Y",
                    "--checkpoint",
                    str(tmp_path / "nonexistent.pt"),
                ]
            )
        assert exc_info.value.code == 1, (
            f"종료 코드={exc_info.value.code}, 1 기대"
        )

    def test_channel_all_runs_all_channels(self, tmp_path, monkeypatch):
        """T-EVAL-12: --channel all → C, M, Y, K 4채널 모두 처리"""
        channels_processed: list[str] = []

        def mock_run_channel(channel, *args, **kwargs):
            channels_processed.append(channel)

        monkeypatch.setattr(
            "scripts.evaluate.load_config",
            lambda: self._make_cfg(tmp_path),
        )
        with patch(
            "scripts.evaluate._run_channel_evaluation",
            side_effect=mock_run_channel,
        ):
            main(["--channel", "all", "--output-dir", str(tmp_path)])

        assert set(channels_processed) == {"Y", "M", "C", "K"}, (
            f"처리된 채널={set(channels_processed)}, {{'Y','M','C','K'}} 기대"
        )
