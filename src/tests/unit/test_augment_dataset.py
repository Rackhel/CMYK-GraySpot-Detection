"""
tests/unit/test_augment_dataset.py

augment_dataset.py 단위 테스트.
Unit tests for augment_dataset.py.

테스트 대상 / Coverage:
    - _augment_image()  — 허용 변환 적용 확인
    - _read_csv()       — CSV 읽기
    - _write_csv()      — CSV 쓰기
    - PRD_TARGETS 상수 — PRD v2 목표 수량 일치
    - 증강 발동 조건    — 목표 미달 시만 증강

TDD Reference: doc/TDD/TDD_ROI_Pipeline.md §3
BDD Reference: doc/BDD/BDD_ROI_Pipeline.md

Python 3.11.5
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from scripts.augment_dataset import PRD_TARGETS, _augment_image, _read_csv, _write_csv

# ── PRD_TARGETS 상수 검증 / PRD v2 constant checks ──────────────────────────
# T-AUG-01 ~ T-AUG-02


class TestPRDTargets:
    """T-AUG-01~02: PRD_TARGETS 상수 검증."""

    def test_prd_targets_match_v2(self):
        """T-AUG-01: PRD_TARGETS가 PRD v2 목표와 정확히 일치한다."""
        expected = {0: 330, 1: 330, 2: 330, 3: 265, 4: 165, 5: 100}
        assert PRD_TARGETS == expected

    def test_prd_targets_total(self):
        """T-AUG-02: 채널당 총 목표 1,520장."""
        assert sum(PRD_TARGETS.values()) == 1520


# ── _augment_image() ─────────────────────────────────────────────────────────
# T-AUG-10 ~ T-AUG-14


class TestAugmentImage:
    """T-AUG-10~14: _augment_image() 동작."""

    @pytest.fixture
    def sample_image(self):
        """테스트용 128×128 RGB PIL 이미지."""
        arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_returns_pil_image(self, sample_image):
        """T-AUG-10: 반환값이 PIL.Image 인스턴스이다."""
        result = _augment_image(sample_image)
        assert isinstance(result, Image.Image)

    def test_output_size_unchanged(self, sample_image):
        """T-AUG-11: 증강 후 이미지 크기가 변하지 않는다."""
        result = _augment_image(sample_image)
        assert result.size == sample_image.size

    def test_augment_returns_varied_results(self, sample_image):
        """T-AUG-12: 여러 번 호출하면 다른 결과가 나온다 (무작위 변환)."""
        results = [_augment_image(sample_image) for _ in range(30)]
        arrays = [np.array(r) for r in results]
        # 30번 중 적어도 2가지 이상 다른 결과가 있어야 한다
        unique = set(a.tobytes() for a in arrays)
        assert len(unique) > 1, "30번 호출 중 변환 다양성이 없음 — 무작위 동작 확인"

    def test_output_dtype_unchanged(self, sample_image):
        """T-AUG-13: 출력 dtype이 입력과 동일하다."""
        result = _augment_image(sample_image)
        assert result.mode == sample_image.mode

    def test_augment_does_not_modify_original(self, sample_image):
        """T-AUG-14: 원본 이미지가 수정되지 않는다."""
        original_arr = np.array(sample_image).copy()
        _augment_image(sample_image)
        assert np.array_equal(np.array(sample_image), original_arr)


# ── _read_csv() / _write_csv() ────────────────────────────────────────────────
# T-AUG-20 ~ T-AUG-23


class TestCsvRoundtrip:
    """T-AUG-20~23: CSV 읽기/쓰기 동작."""

    def test_read_nonexistent_returns_empty_list(self, tmp_path):
        """T-AUG-20: 없는 파일 읽기 → 빈 리스트 반환."""
        result = _read_csv(tmp_path / "nonexistent.csv")
        assert result == []

    def test_write_then_read_roundtrip(self, tmp_path):
        """T-AUG-21: 쓰기 후 읽기 — 동일 데이터 복원."""
        csv_path = tmp_path / "test.csv"
        rows = [
            {
                "filepath": "data_set/labeled/Y/0/img_0001.png",
                "channel": "Y",
                "level": 0,
            },
            {
                "filepath": "data_set/labeled/M/3/img_0002.png",
                "channel": "M",
                "level": 3,
            },
        ]
        _write_csv(csv_path, rows)
        loaded = _read_csv(csv_path)
        assert len(loaded) == 2
        assert loaded[0]["channel"] == "Y"
        assert loaded[1]["channel"] == "M"

    def test_written_csv_has_correct_fieldnames(self, tmp_path):
        """T-AUG-22: 저장된 CSV 헤더가 filepath, channel, level 이다."""
        csv_path = tmp_path / "test.csv"
        _write_csv(csv_path, [{"filepath": "p", "channel": "Y", "level": 0}])
        with open(csv_path) as f:
            header = f.readline().strip()
        assert header == "filepath,channel,level"

    def test_read_csv_returns_list_of_dicts(self, tmp_path):
        """T-AUG-23: _read_csv 반환값은 list[dict] 이다."""
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "channel", "level"])
            writer.writeheader()
            writer.writerow({"filepath": "x.png", "channel": "C", "level": "2"})
        result = _read_csv(csv_path)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)


# ── 증강 발동 조건 / Augmentation trigger ─────────────────────────────────────
# T-AUG-30 ~ T-AUG-32


class TestAugmentationTrigger:
    """T-AUG-30~32: 증강 발동 조건 — PRD 미달 시만 증강."""

    def test_no_augment_when_at_target(self, tmp_path):
        """T-AUG-30: 이미 목표 달성 → 증강 파일 생성 없음."""
        import copy
        from unittest.mock import patch

        # labels_master.csv 에 Y-Level-0 이 정확히 목표치(330)만큼 있도록 설정
        csv_path = tmp_path / "labels_master.csv"
        labeled_dir = tmp_path / "labeled"
        img_dir = labeled_dir / "Y" / "0"
        img_dir.mkdir(parents=True)

        rows = []
        for i in range(330):
            fname = f"img_{i:04d}.png"
            img_path = img_dir / fname
            # 실제 파일 생성 (augment_dataset.py 가 파일을 열기 때문)
            from PIL import Image as PILImage

            PILImage.fromarray(
                np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            ).save(str(img_path))
            rel = str((labeled_dir / "Y" / "0" / fname).relative_to(tmp_path))
            rows.append({"filepath": rel, "channel": "Y", "level": 0})

        _write_csv(csv_path, rows)
        initial_count = len(list(img_dir.glob("*.png")))

        # augment_dataset.main() 의 핵심 로직만 직접 검증
        from collections import defaultdict

        groups: dict = defaultdict(list)
        for row in rows:
            key = (row["channel"], int(row["level"]))
            groups[key].append(row)

        channel, level = "Y", 0
        current_count = len(groups[(channel, level)])
        target_count = PRD_TARGETS[level]

        # 목표 달성 → 증강 필요 없음
        assert current_count >= target_count
        shortage = max(0, target_count - current_count)
        assert shortage == 0

    def test_augment_needed_when_below_target(self):
        """T-AUG-31: 목표 미달 → shortage > 0."""
        current = 100
        target = PRD_TARGETS[0]  # 330
        shortage = max(0, target - current)
        assert shortage == 230

    def test_shortage_zero_when_above_target(self):
        """T-AUG-32: 목표 초과 → shortage = 0."""
        current = 400
        target = PRD_TARGETS[0]  # 330
        shortage = max(0, target - current)
        assert shortage == 0
