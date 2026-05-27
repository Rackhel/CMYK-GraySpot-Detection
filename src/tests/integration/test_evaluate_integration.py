"""
tests/integration/test_evaluate_integration.py

scripts/evaluate.py 통합 테스트.
Integration tests for scripts/evaluate.py.

TDD 근거 / TDD Reference:
    - TDD_Evaluate_Script.md §3 — T-EVAL-INT-01 ~ T-EVAL-INT-02

BDD 근거 / BDD Reference:
    - BDD_Evaluation.md §4.7 — evaluate.py CLI 시나리오

SSOT 근거 / SSOT Reference:
    - SSOT_Core.md §6 — SSOT-FF01 Fail-Fast
    - Contract_evaluation_reporting.md §10 — evaluate.py API 계약

전략 / Strategy:
    - 실제 GrayspotModel과 최소 이미지 데이터를 사용한다 (mock 없음).
    - Uses real GrayspotModel with minimal image data (no mocks).
    - 각 테스트는 tmp_path 격리 환경에서 실행된다.
    - Each test runs in an isolated tmp_path environment.

Python 3.11.5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # CMYK_MAIN/
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from scripts.evaluate import (
    _run_channel_evaluation,
    _write_json_summary,
    main,
    parse_args,
)

# ── 통합 테스트용 픽스처 / Integration test fixtures ─────────────────────────


@pytest.fixture
def eval_cfg(tmp_path) -> dict:
    """
    evaluate.py 통합 테스트용 config dict.
    Config dict for evaluate.py integration tests.

    labeled_dir, models_dir, reports_dir 모두 tmp_path로 격리.
    All storage dirs isolated under tmp_path.
    """
    return {
        "system": {"device": "cpu"},
        "data": {
            "channels": ["Y", "M", "C", "K"],
            "num_levels": 6,
            "image_size": 64,  # 빠른 테스트를 위해 작은 크기 / Small size for speed
        },
        "model": {
            "backbone": "efficientnet_b0",
            "frozen_backbone": False,
        },
        "phase0": {
            "projection_dim": 128,
            "hidden_dim": 256,
            "temperature": 0.1,
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "augmentation": {
                "color_jitter": 0.4,
                "blur_prob": 0.5,
                "flip_prob": 0.5,
                "crop_prob": 0.5,
                "crop_scale_min": 0.6,
                "crop_scale_max": 1.0,
                "contrast_scale_min": 0.8,
                "contrast_scale_max": 1.2,
                "blur_kernels": [3, 5],
            },
        },
        "phase2": {
            "dropout": 0.3,
            "hidden_dim": 256,
            "heads": {
                "efficientnet_b0": {"mid_dim": None, "hidden_dim": 256, "dropout": 0.2}
            },
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "oversample": False,
            "early_stopping": {"enabled": False, "patience": 5, "min_delta": 1e-4},
            "augmentation": {
                "flip_prob": 0.5,
                "brightness_prob": 0.5,
                "brightness_range": 30,
                "noise_prob": 0.5,
                "noise_range": 10,
            },
        },
        "storage": {
            "data_root": str(tmp_path),
            "labeled_dir": str(tmp_path / "labeled"),
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "evaluation": {
            "targets": {
                "overall_accuracy": 0.90,
                "per_color_accuracy": 0.85,
                "per_class_f1": 0.80,
                "mae": 0.50,
            },
            "swing_thresholds": {
                "acc_retry": 0.80,
                "f1_retry": 0.70,
                "mae_retry": 0.80,
            },
        },
        "inference": {
            "confidence_thresholds": {
                "auto_accept": 0.8,
                "warn_threshold": 0.5,
                "manual_review": 0.3,
            },
        },
    }


@pytest.fixture
def minimal_labeled_dir(tmp_path, eval_cfg) -> Path:
    """
    Y 채널 최소 labeled 디렉토리와 labels_master.csv를 생성한다.
    Creates minimal labeled directory for Y channel and labels_master.csv.
    """
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not available")

    import csv

    image_size = eval_cfg["data"]["image_size"]
    labeled_root = tmp_path / "labeled"
    rows = []

    for level in range(6):
        level_dir = labeled_root / "Y" / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            filename = f"Y_{level}_{i:04d}.png"
            img = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
            cv2.imwrite(str(level_dir / filename), img)
            rows.append(
                {"filename": filename, "Y": level, "M": level, "C": level, "K": level}
            )

    # Create labels_master.csv at data_root (tmp_path)
    csv_path = tmp_path / "labels_master.csv"  # Changed from labels_v0.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "Y", "M", "C", "K"])
        writer.writeheader()
        writer.writerows(rows)

    return labeled_root


@pytest.fixture
def mock_checkpoint(tmp_path, eval_cfg) -> Path:
    """
    더미 GrayspotModel 가중치를 체크포인트 파일로 저장한다.
    Saves dummy GrayspotModel weights as a checkpoint file.

    Returns:
        Path — 저장된 체크포인트 파일 경로
    """
    from models.grayspot_model import GrayspotModel

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "best_Y.pt"

    model = GrayspotModel(eval_cfg, phase=2)
    torch.save(model.state_dict(), str(ckpt_path))

    return ckpt_path


# ── T-EVAL-INT-01 / Integration Scenario 1 ───────────────────────────────────
# 실제 Evaluator + 모델 + 데이터로 평가 실행 후 JSON 리포트 정상 생성 확인
# Verify JSON report is created after evaluation with real Evaluator + model + data


class TestEvaluateIntegration:
    """T-EVAL-INT-01: 실제 체크포인트 로드 → Evaluator 실행 → 리포트 저장 통합 테스트."""

    def test_json_report_created_after_run(
        self, tmp_path, eval_cfg, minimal_labeled_dir, mock_checkpoint
    ):
        """T-EVAL-INT-01a: 평가 실행 후 JSON 파일이 생성된다."""
        # Given: 유효한 체크포인트와 labeled 데이터 존재
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        # When: 단일 채널 평가 실행
        with patch("scripts.evaluate.load_config", return_value=eval_cfg):
            main(
                [
                    "--channel",
                    "Y",
                    "--output-dir",
                    str(output_dir),
                    "--checkpoint",
                    str(mock_checkpoint),
                ]
            )

        # Then: report_Y.json 파일이 존재해야 한다
        report_file = output_dir / "report_Y.json"
        assert report_file.exists(), (
            "평가 완료 후 report_Y.json 파일이 생성되지 않았습니다 / "
            "report_Y.json not generated after evaluation"
        )

    def test_json_report_accuracy_is_float_in_range(
        self, tmp_path, eval_cfg, minimal_labeled_dir, mock_checkpoint
    ):
        """T-EVAL-INT-01b: JSON 내 accuracy 값이 float [0, 1] 범위에 있다."""
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch("scripts.evaluate.load_config", return_value=eval_cfg):
            main(
                [
                    "--channel",
                    "Y",
                    "--output-dir",
                    str(output_dir),
                    "--checkpoint",
                    str(mock_checkpoint),
                ]
            )

        report_file = output_dir / "report_Y.json"
        assert report_file.exists(), "report_Y.json 없음 / No report_Y.json"

        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert "accuracy" in data, "JSON에 'accuracy' 키가 없습니다 / No 'accuracy' key"
        assert isinstance(
            data["accuracy"], float
        ), f"accuracy 타입={type(data['accuracy'])}, float 기대 / float expected"
        assert (
            0.0 <= data["accuracy"] <= 1.0
        ), f"accuracy={data['accuracy']} — [0, 1] 범위 벗어남 / out of [0,1] range"

    def test_json_report_contains_channel_key(
        self, tmp_path, eval_cfg, minimal_labeled_dir, mock_checkpoint
    ):
        """T-EVAL-INT-01c: JSON 리포트에 'channel' 키와 올바른 채널명이 있다."""
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch("scripts.evaluate.load_config", return_value=eval_cfg):
            main(
                [
                    "--channel",
                    "Y",
                    "--output-dir",
                    str(output_dir),
                    "--checkpoint",
                    str(mock_checkpoint),
                ]
            )

        report_file = output_dir / "report_Y.json"
        assert report_file.exists(), "report_Y.json 없음"

        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert data.get("channel") == "Y", f"channel='{data.get('channel')}', 'Y' 기대"

    def test_exit_code_1_when_checkpoint_missing(self, tmp_path):
        """T-EVAL-INT-01d: SSOT-FF01 — 체크포인트 누락 시 exit code 1"""
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--channel",
                    "Y",
                    "--checkpoint",
                    str(tmp_path / "nonexistent.pt"),
                ]
            )
        assert (
            exc_info.value.code == 1
        ), f"종료 코드={exc_info.value.code}, 1 기대 (SSOT-FF01)"


# ── T-EVAL-INT-02 / Integration Scenario 2 ───────────────────────────────────
# --channel all 실행 시 4채널 리포트 모두 생성 확인
# Verify all 4 channel reports are generated with --channel all


class TestEvaluateAllChannels:
    """T-EVAL-INT-02: --channel all 실행 시 4채널 JSON 리포트 모두 생성."""

    def test_all_channels_produce_json_reports(self, tmp_path, eval_cfg):
        """T-EVAL-INT-02a: --channel all → 4채널 JSON 리포트 각각 생성."""
        # Given: 4채널 모두 _run_channel_evaluation이 모킹됨 (실제 모델 불필요)
        # Given: _run_channel_evaluation is mocked for all 4 channels
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        channels_evaluated = []

        def mock_run_channel(channel, output_dir, cfg, checkpoint=None):
            """채널별로 JSON 파일을 실제로 생성하는 mock."""
            channels_evaluated.append(channel)
            report_path = output_dir / f"report_{channel}.json"
            report_path.write_text(
                json.dumps({"channel": channel, "accuracy": 0.75}),
                encoding="utf-8",
            )

        with patch("scripts.evaluate.load_config", return_value=eval_cfg):
            with patch(
                "scripts.evaluate._run_channel_evaluation",
                side_effect=mock_run_channel,
            ):
                main(["--channel", "all", "--output-dir", str(output_dir)])

        # Then: 4채널 모두 평가됨
        assert set(channels_evaluated) == {
            "Y",
            "M",
            "C",
            "K",
        }, f"평가된 채널={set(channels_evaluated)}, 4채널 기대"

    def test_all_channels_json_files_exist(self, tmp_path, eval_cfg):
        """T-EVAL-INT-02b: --channel all → 4개 JSON 파일 존재"""
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        def mock_run_channel(channel, output_dir, cfg, checkpoint=None):
            report_path = output_dir / f"report_{channel}.json"
            report_path.write_text(
                json.dumps({"channel": channel, "accuracy": 0.75}),
                encoding="utf-8",
            )

        with patch("scripts.evaluate.load_config", return_value=eval_cfg):
            with patch(
                "scripts.evaluate._run_channel_evaluation",
                side_effect=mock_run_channel,
            ):
                main(["--channel", "all", "--output-dir", str(output_dir)])

        json_files = list(output_dir.glob("report_*.json"))
        assert (
            len(json_files) == 4
        ), f"JSON 파일 수={len(json_files)}, 4개 기대 (Y/M/C/K)"

    def test_single_channel_produces_one_json(self, tmp_path, eval_cfg):
        """T-EVAL-INT-02c: 단일 채널 → JSON 1개만 생성."""
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        def mock_run_channel(channel, output_dir, cfg, checkpoint=None):
            report_path = output_dir / f"report_{channel}.json"
            report_path.write_text(
                json.dumps({"channel": channel, "accuracy": 0.80}),
                encoding="utf-8",
            )

        with patch("scripts.evaluate.load_config", return_value=eval_cfg):
            with patch(
                "scripts.evaluate._run_channel_evaluation",
                side_effect=mock_run_channel,
            ):
                main(["--channel", "M", "--output-dir", str(output_dir)])

        json_files = list(output_dir.glob("report_*.json"))
        assert len(json_files) == 1, f"JSON 파일 수={len(json_files)}, 1개 기대"
        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert data["channel"] == "M"


# ── _write_json_summary 단독 통합 테스트 / Standalone integration tests ───────


class TestWriteJsonSummary:
    """_write_json_summary() 파일 I/O 통합 테스트."""

    def test_creates_valid_json_file(self, tmp_path):
        """실제 dict metrics로 유효한 JSON 파일을 생성한다."""
        metrics = {
            "overall": {
                "accuracy": 0.87,
                "macro_f1": 0.82,
                "mae": 0.35,
                "n_samples": 120,
            }
        }
        path = tmp_path / "report_Y.json"
        _write_json_summary(path, "Y", metrics)

        assert path.exists(), "JSON 파일이 생성되지 않음"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["channel"] == "Y"
        assert data["accuracy"] == pytest.approx(0.87)
        assert data["macro_f1"] == pytest.approx(0.82)
        assert data["mae"] == pytest.approx(0.35)
        assert data["n_samples"] == 120

    def test_handles_non_dict_metrics_gracefully(self, tmp_path):
        """dict가 아닌 metrics도 예외 없이 처리한다 (MagicMock 호환성)."""
        from unittest.mock import MagicMock

        path = tmp_path / "report_C.json"
        _write_json_summary(path, "C", MagicMock())

        assert path.exists(), "MagicMock 입력에도 JSON 파일이 생성되어야 함"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "channel" in data

    def test_utf8_encoding(self, tmp_path):
        """JSON 파일이 UTF-8 인코딩으로 저장된다."""
        metrics = {
            "overall": {
                "accuracy": 0.90,
                "macro_f1": 0.85,
                "mae": 0.30,
                "n_samples": 50,
            }
        }
        path = tmp_path / "report_K.json"
        _write_json_summary(path, "K", metrics)

        # UTF-8로 직접 읽기
        raw = path.read_bytes()
        decoded = raw.decode("utf-8")
        data = json.loads(decoded)
        assert data["channel"] == "K"
