"""
test_label_refiner.py
Tests for LabelRefiner class (data/label_refiner.py).
Status: FAILING — LabelRefiner not yet implemented.
Ref: doc/TDD/TDD_LabelRefiner.md
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Will raise ImportError until implemented
from data.label_refiner import LabelRefiner


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def refiner():
    return LabelRefiner(cfg={})


@pytest.fixture
def sample_embeddings():
    np.random.seed(42)
    return np.random.randn(100, 128).astype(np.float32)


@pytest.fixture
def sample_labels():
    return list(range(6)) * 16 + list(range(4))  # 100 labels, 0-5


@pytest.fixture
def sample_paths():
    return [f"img_{i}.png" for i in range(100)]


@pytest.fixture
def sample_priority_df(refiner, sample_embeddings, sample_labels, sample_paths):
    return refiner.compute_priority_score(sample_embeddings, sample_labels, sample_paths)


@pytest.fixture
def labels_v0_csv(tmp_path):
    df = pd.DataFrame({
        "path": ["a.png", "b.png", "c.png", "d.png"],
        "channel": ["Y", "Y", "Y", "Y"],
        "level": [1, 2, 3, 4],
        "version": [0, 0, 0, 0],
        "reviewer": ["auto", "auto", "auto", "auto"],
    })
    csv_path = tmp_path / "labels_v0.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ── compute_priority_score() ──────────────────────────────────────────────────

class TestComputePriorityScore:
    """T-LR-01 ~ T-LR-05"""

    def test_returns_correct_row_count(self, refiner, sample_embeddings, sample_labels, sample_paths):
        """T-LR-01: 100개 입력 → 100행 DataFrame"""
        df = refiner.compute_priority_score(sample_embeddings, sample_labels, sample_paths)
        assert len(df) == 100

    def test_has_required_columns(self, refiner, sample_embeddings, sample_labels, sample_paths):
        """T-LR-02: 필수 컬럼 존재"""
        df = refiner.compute_priority_score(sample_embeddings, sample_labels, sample_paths)
        required = {"path", "true_label", "priority_score", "cluster_label"}
        assert required.issubset(set(df.columns))

    def test_priority_score_in_range(self, sample_priority_df):
        """T-LR-03: priority_score ∈ [0.0, 1.0]"""
        assert (sample_priority_df["priority_score"] >= 0.0).all()
        assert (sample_priority_df["priority_score"] <= 1.0).all()

    def test_empty_input_returns_empty_df(self, refiner):
        """T-LR-04: 빈 입력 → 빈 DataFrame (예외 없음)"""
        empty_emb = np.empty((0, 128), dtype=np.float32)
        df = refiner.compute_priority_score(empty_emb, [], [])
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_mismatched_lengths_raises(self, refiner, sample_embeddings):
        """T-LR-05: paths 길이 != embeddings 행 수 → ValueError"""
        with pytest.raises(ValueError):
            refiner.compute_priority_score(
                sample_embeddings,
                labels=[0] * 100,
                paths=["img.png"] * 50,  # 길이 불일치
            )


# ── compute_clustering_quality() ─────────────────────────────────────────────

class TestComputeClusteringQuality:
    """T-LR-10 ~ T-LR-13"""

    def test_returns_required_keys(self, refiner, sample_embeddings, sample_labels):
        """T-LR-10: 반환 dict에 ari, silhouette 키 존재"""
        result = refiner.compute_clustering_quality(sample_embeddings, sample_labels)
        assert "ari" in result
        assert "silhouette" in result

    def test_values_are_float(self, refiner, sample_embeddings, sample_labels):
        """T-LR-10: ari, silhouette 값은 float"""
        result = refiner.compute_clustering_quality(sample_embeddings, sample_labels)
        assert isinstance(result["ari"], float)
        assert isinstance(result["silhouette"], float)

    def test_random_embeddings_no_exception(self, refiner):
        """T-LR-12: 랜덤 임베딩 → 예외 없이 float 반환"""
        emb = np.random.randn(60, 128).astype(np.float32)
        labels = [i % 6 for i in range(60)]
        result = refiner.compute_clustering_quality(emb, labels)
        assert isinstance(result["ari"], float)
        assert isinstance(result["silhouette"], float)

    def test_single_label_no_exception(self, refiner):
        """T-LR-13: 단일 레이블 → 예외 없이 처리"""
        emb = np.random.randn(20, 128).astype(np.float32)
        labels = [0] * 20
        result = refiner.compute_clustering_quality(emb, labels)
        assert "silhouette" in result


# ── get_review_queue() ────────────────────────────────────────────────────────

class TestGetReviewQueue:
    """T-LR-20 ~ T-LR-23"""

    def test_top_20_percent_count(self, refiner, sample_priority_df):
        """T-LR-20: top_ratio=0.2 → 20행 반환"""
        queue = refiner.get_review_queue(sample_priority_df, top_ratio=0.2)
        assert len(queue) == 20

    def test_zero_ratio_empty(self, refiner, sample_priority_df):
        """T-LR-21: top_ratio=0.0 → 빈 DataFrame"""
        queue = refiner.get_review_queue(sample_priority_df, top_ratio=0.0)
        assert len(queue) == 0

    def test_full_ratio_returns_all(self, refiner, sample_priority_df):
        """T-LR-22: top_ratio=1.0 → 전체 반환"""
        queue = refiner.get_review_queue(sample_priority_df, top_ratio=1.0)
        assert len(queue) == len(sample_priority_df)

    def test_queue_has_highest_priority(self, refiner, sample_priority_df):
        """T-LR-23: 반환된 행의 priority_score ≥ 나머지 최솟값"""
        queue = refiner.get_review_queue(sample_priority_df, top_ratio=0.2)
        rest = sample_priority_df[~sample_priority_df.index.isin(queue.index)]
        if len(rest) > 0:
            assert queue["priority_score"].min() >= rest["priority_score"].max() - 1e-8


# ── save_labels() ─────────────────────────────────────────────────────────────

class TestSaveLabels:
    """T-LR-30 ~ T-LR-33"""

    def test_creates_output_file(self, refiner, labels_v0_csv, tmp_path):
        """T-LR-30: save_labels → 파일 생성"""
        output = tmp_path / "labels_v1.csv"
        refiner.save_labels(labels_v0_csv, corrections={}, output_path=output)
        assert output.exists()

    def test_correction_applied(self, refiner, labels_v0_csv, tmp_path):
        """T-LR-31: 수정된 경로의 level 변경"""
        output = tmp_path / "labels_v1.csv"
        refiner.save_labels(labels_v0_csv, corrections={"a.png": 5}, output_path=output)
        result = pd.read_csv(output)
        assert result[result["path"] == "a.png"]["level"].iloc[0] == 5

    def test_uncorrected_rows_preserved(self, refiner, labels_v0_csv, tmp_path):
        """T-LR-32: 수정되지 않은 행 원본 유지"""
        output = tmp_path / "labels_v1.csv"
        refiner.save_labels(labels_v0_csv, corrections={"a.png": 5}, output_path=output)
        result = pd.read_csv(output)
        assert result[result["path"] == "b.png"]["level"].iloc[0] == 2

    def test_missing_original_raises(self, refiner, tmp_path):
        """T-LR-33: 원본 CSV 미존재 → FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            refiner.save_labels(
                tmp_path / "nonexistent.csv",
                corrections={},
                output_path=tmp_path / "out.csv",
            )
