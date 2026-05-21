"""
tests/unit/test_prepare_dataset.py

prepare_dataset.py 단위 테스트.
Unit tests for prepare_dataset.py.

테스트 대상 / Coverage:
    - _parse_filename() — 파일명 파싱 (level, channel 추출)
    - _load_roi_labels() — roi_labels.csv 로드 및 채널별 레벨 오버라이드
    - EXTRACT_CAP 상한선 로직

TDD Reference: doc/TDD/TDD_ROI_Pipeline.md §3
BDD Reference: doc/BDD/BDD_ROI_Pipeline.md Scenario P.1~P.3

Python 3.11.5
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

import scripts.prepare_dataset as pd_mod
from scripts.prepare_dataset import (
    CHANNELS,
    EXTRACT_CAP,
    _load_roi_labels,
    _parse_filename,
)

# ── _parse_filename() ─────────────────────────────────────────────────────────
# T-PREP-01 ~ T-PREP-06


class TestParseFilename:
    """T-PREP-01~06: _parse_filename() 기본 동작."""

    def test_valid_stem_returns_level_and_channel(self):
        """T-PREP-01: 유효 파일명 → (level, channel) 반환."""
        result = _parse_filename("lvl3_Scanned_Documents_(113)_3_1_M")
        assert result == (3, "M")

    def test_all_channels_parsed(self):
        """T-PREP-02: Y/M/C/K 모든 채널 파싱 성공."""
        for ch in ("Y", "M", "C", "K"):
            result = _parse_filename(f"lvl2_scan_001_{ch}")
            assert result is not None
            assert result == (2, ch)

    def test_level_zero_parsed(self):
        """T-PREP-03: Level 0 파싱 성공."""
        result = _parse_filename("lvl0_scan_001_Y")
        assert result == (0, "Y")

    def test_level_five_parsed(self):
        """T-PREP-04: Level 5 파싱 성공."""
        result = _parse_filename("lvl5_scan_001_C")
        assert result == (5, "C")

    def test_invalid_stem_returns_none(self):
        """T-PREP-05: 패턴에 맞지 않는 파일명 → None 반환."""
        assert _parse_filename("random_file") is None
        assert _parse_filename("scan_001") is None
        assert _parse_filename("") is None

    def test_invalid_channel_returns_none(self):
        """T-PREP-06: 유효하지 않은 채널 문자 → None 반환."""
        assert _parse_filename("lvl3_scan_001_Z") is None
        assert _parse_filename("lvl3_scan_001_R") is None


# ── _load_roi_labels() ────────────────────────────────────────────────────────
# T-PREP-10 ~ T-PREP-14


class TestLoadRoiLabels:
    """T-PREP-10~14: _load_roi_labels() 동작."""

    def test_returns_empty_dict_when_no_file(self, tmp_path):
        """T-PREP-10: roi_labels.csv 없으면 빈 dict 반환."""
        nonexistent = tmp_path / "roi_labels.csv"
        with patch.object(pd_mod, "ROI_LABELS_CSV", nonexistent):
            result = _load_roi_labels()
        assert result == {}

    def test_loads_mapping_from_csv(self, tmp_path):
        """T-PREP-11: roi_labels.csv 존재 시 매핑 로드."""
        csv_path = tmp_path / "roi_labels.csv"
        rows = [
            {"roi_filename": "lvl3_scan_001_C", "level": "1"},
            {"roi_filename": "lvl3_scan_001_M", "level": "3"},
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["roi_filename", "level"])
            writer.writeheader()
            writer.writerows(rows)

        with patch.object(pd_mod, "ROI_LABELS_CSV", csv_path):
            result = _load_roi_labels()

        assert result["lvl3_scan_001_C"] == 1
        assert result["lvl3_scan_001_M"] == 3

    def test_level_values_are_int(self, tmp_path):
        """T-PREP-12: 로드된 level 값은 int 타입."""
        csv_path = tmp_path / "roi_labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["roi_filename", "level"])
            writer.writeheader()
            writer.writerow({"roi_filename": "lvl2_scan_Y", "level": "2"})

        with patch.object(pd_mod, "ROI_LABELS_CSV", csv_path):
            result = _load_roi_labels()

        assert isinstance(result["lvl2_scan_Y"], int)

    def test_all_channels_in_csv_loaded(self, tmp_path):
        """T-PREP-13: 같은 스캔의 4개 채널 모두 로드."""
        csv_path = tmp_path / "roi_labels.csv"
        base = "lvl3_scan_001"
        channels_levels = [("C", 3), ("M", 1), ("Y", 0), ("K", 0)]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["roi_filename", "level"])
            writer.writeheader()
            for ch, lv in channels_levels:
                writer.writerow({"roi_filename": f"{base}_{ch}", "level": str(lv)})

        with patch.object(pd_mod, "ROI_LABELS_CSV", csv_path):
            result = _load_roi_labels()

        for ch, lv in channels_levels:
            assert result[f"{base}_{ch}"] == lv

    def test_empty_csv_returns_empty_dict(self, tmp_path):
        """T-PREP-14: 헤더만 있는 CSV → 빈 dict 반환."""
        csv_path = tmp_path / "roi_labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["roi_filename", "level"])
            writer.writeheader()

        with patch.object(pd_mod, "ROI_LABELS_CSV", csv_path):
            result = _load_roi_labels()

        assert result == {}


# ── 채널별 독립 라벨링 오버라이드 / Per-channel label override ────────────────
# T-PREP-20 ~ T-PREP-22


class TestLabelOverride:
    """T-PREP-20~22: roi_labels.csv 오버라이드 우선순위."""

    def test_roi_labels_overrides_filename_level(self, tmp_path):
        """T-PREP-20: roi_labels.csv 레벨이 파일명 레벨보다 우선한다."""
        csv_path = tmp_path / "roi_labels.csv"
        # 파일명은 lvl3이지만 roi_labels.csv 에서 레벨 1로 오버라이드
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["roi_filename", "level"])
            writer.writeheader()
            writer.writerow({"roi_filename": "lvl3_scan_001_M", "level": "1"})

        with patch.object(pd_mod, "ROI_LABELS_CSV", csv_path):
            label_map = _load_roi_labels()

        filename_level = 3  # lvl3_...
        overridden_level = label_map.get("lvl3_scan_001_M", filename_level)
        assert overridden_level == 1

    def test_filename_level_used_when_not_in_roi_labels(self, tmp_path):
        """T-PREP-21: roi_labels.csv 에 없는 파일은 파일명 레벨 사용."""
        csv_path = tmp_path / "roi_labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["roi_filename", "level"])
            writer.writeheader()
            writer.writerow({"roi_filename": "lvl3_scan_001_M", "level": "1"})

        with patch.object(pd_mod, "ROI_LABELS_CSV", csv_path):
            label_map = _load_roi_labels()

        # 이 파일은 roi_labels.csv 에 없음
        filename_level = 2
        result = label_map.get("lvl2_scan_002_C", filename_level)
        assert result == filename_level

    def test_different_channels_same_scan_can_have_different_levels(self, tmp_path):
        """T-PREP-22: 같은 스캔의 채널별 레벨이 독립적으로 다를 수 있다."""
        csv_path = tmp_path / "roi_labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["roi_filename", "level"])
            writer.writeheader()
            writer.writerow({"roi_filename": "lvl3_scan_001_C", "level": "3"})
            writer.writerow({"roi_filename": "lvl3_scan_001_M", "level": "0"})
            writer.writerow({"roi_filename": "lvl3_scan_001_Y", "level": "1"})
            writer.writerow({"roi_filename": "lvl3_scan_001_K", "level": "2"})

        with patch.object(pd_mod, "ROI_LABELS_CSV", csv_path):
            label_map = _load_roi_labels()

        levels = {ch: label_map[f"lvl3_scan_001_{ch}"] for ch in ("C", "M", "Y", "K")}
        assert len(set(levels.values())) > 1, "4개 채널이 서로 다른 레벨을 가져야 함"


# ── EXTRACT_CAP 상수 검증 ─────────────────────────────────────────────────────
# T-PREP-30 ~ T-PREP-31


class TestExtractCap:
    """T-PREP-30~31: EXTRACT_CAP 상수 PRD v2 일치 확인."""

    def test_extract_cap_matches_prd_v2_targets(self):
        """T-PREP-30: EXTRACT_CAP이 PRD v2 목표와 일치한다."""
        prd_v2 = {0: 330, 1: 330, 2: 330, 3: 265, 4: 165, 5: 100}
        assert EXTRACT_CAP == prd_v2

    def test_extract_cap_total_is_1520(self):
        """T-PREP-31: 채널당 총 목표 1,520장."""
        assert sum(EXTRACT_CAP.values()) == 1520

    def test_channels_constant(self):
        """T-PREP-32: CHANNELS = {Y, M, C, K}."""
        assert CHANNELS == {"Y", "M", "C", "K"}
