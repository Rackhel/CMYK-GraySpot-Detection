---
type: tdd
domain: label_refiner
status: failing
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "../SSOT/SSOT_ROI_Pipeline.md"
  - "../Contract/Contract_roi_pipeline.md"
---

# [TDD] LabelRefiner — 임베딩 기반 라벨 정제

> **목적**: `LabelRefiner` 클래스의 동작을 BDD/TDD로 정의한다.
> **테스트 파일**: `src/tests/unit/test_label_refiner.py`
> **상태**: 🔴 Failing — 구현 전

---

## 1. BDD 시나리오

### Feature: Priority Score 계산

**Scenario 1: 정상 임베딩으로 Priority Score 계산**
```
Given  (100, 128) float32 임베딩과 (100,) int 레이블이 주어졌을 때
When   compute_priority_score(embeddings, labels, paths) 를 호출하면
Then   반환 DataFrame의 행 수는 100이고
And    컬럼에 "path", "true_label", "priority_score", "cluster_label" 이 존재하고
And    priority_score 값은 [0.0, 1.0] 범위이다
```

**Scenario 2: Priority Score 상위 20% 검토 큐 추출**
```
Given  100개 샘플의 priority_df 가 주어졌을 때
When   get_review_queue(priority_df, top_ratio=0.2) 를 호출하면
Then   반환 DataFrame의 행 수는 20이고
And    반환된 샘플의 priority_score가 나머지보다 높다
```

**Scenario 3: 임베딩이 비어 있을 때**
```
Given  빈 임베딩 배열 (0, 128) 이 주어졌을 때
When   compute_priority_score() 를 호출하면
Then   빈 DataFrame을 반환한다 (예외 아님)
```

---

### Feature: 클러스터링 품질 평가

**Scenario 4: 완벽히 분리된 클러스터 → 높은 품질 지표**
```
Given  6개 클러스터가 명확하게 분리된 임베딩이 주어졌을 때
When   compute_clustering_quality(embeddings, labels) 를 호출하면
Then   결과 dict에 "ari"와 "silhouette" 키가 존재하고
And    ari > 0.8, silhouette > 0.6 이다
```

**Scenario 5: 무작위 임베딩 → 낮은 품질 지표**
```
Given  완전히 랜덤한 임베딩이 주어졌을 때
When   compute_clustering_quality() 를 호출하면
Then   silhouette < 0.5 이다
```

---

### Feature: 라벨 수정 및 저장

**Scenario 6: 수정된 레벨로 새 CSV 저장**
```
Given  labels_v0.csv 와 corrections={"img1.png": 2, "img2.png": 4} 가 주어졌을 때
When   save_labels(original_csv, corrections, output_path) 를 호출하면
Then   output_path에 CSV 파일이 생성되고
And    img1.png 의 level 컬럼 값은 2이고
And    수정되지 않은 행은 원본 값이 유지된다
```

---

## 2. TDD 스펙

### 2.1 compute_priority_score()

| 테스트 ID | 입력 | 기댓값 |
| --- | --- | --- |
| T-LR-01 | (100, 128) 임베딩, (100,) 레이블 | 반환 DataFrame 행 수 == 100 |
| T-LR-02 | 반환 컬럼 확인 | {"path","true_label","priority_score","cluster_label"} ⊆ columns |
| T-LR-03 | priority_score 범위 | `0.0 <= score <= 1.0` 모든 행 |
| T-LR-04 | 빈 입력 (0, 128) | 빈 DataFrame 반환, 예외 없음 |
| T-LR-05 | paths 길이 != embeddings 행 수 | `ValueError` 발생 |

```python
def test_compute_priority_score_shape():
    refiner = LabelRefiner(cfg={})
    embeddings = np.random.randn(100, 128).astype(np.float32)
    labels = np.random.randint(0, 6, 100).tolist()
    paths = [f"img_{i}.png" for i in range(100)]
    df = refiner.compute_priority_score(embeddings, labels, paths)
    assert len(df) == 100
    assert {"path", "true_label", "priority_score", "cluster_label"}.issubset(df.columns)
    assert (df["priority_score"] >= 0.0).all()
    assert (df["priority_score"] <= 1.0).all()

def test_compute_priority_score_empty():
    refiner = LabelRefiner(cfg={})
    df = refiner.compute_priority_score(
        np.empty((0, 128), dtype=np.float32), [], []
    )
    assert len(df) == 0
```

### 2.2 compute_clustering_quality()

| 테스트 ID | 입력 | 기댓값 |
| --- | --- | --- |
| T-LR-10 | 반환 키 확인 | `{"ari", "silhouette"} == set(result.keys())` |
| T-LR-11 | 완벽 분리 임베딩 | `ari > 0.8` |
| T-LR-12 | 랜덤 임베딩 | 예외 없이 float 반환 |
| T-LR-13 | 단일 레이블만 있을 때 | `silhouette == 0.0` 또는 예외 없이 처리 |

```python
def test_clustering_quality_keys():
    refiner = LabelRefiner(cfg={})
    emb = np.random.randn(60, 128).astype(np.float32)
    labels = [i // 10 for i in range(60)]  # 6클러스터 × 10샘플
    result = refiner.compute_clustering_quality(emb, labels)
    assert "ari" in result and "silhouette" in result
    assert isinstance(result["ari"], float)
    assert isinstance(result["silhouette"], float)
```

### 2.3 get_review_queue()

| 테스트 ID | 입력 | 기댓값 |
| --- | --- | --- |
| T-LR-20 | 100행 DataFrame, top_ratio=0.2 | 반환 행 수 == 20 |
| T-LR-21 | top_ratio=0.0 | 빈 DataFrame |
| T-LR-22 | top_ratio=1.0 | 전체 반환 |
| T-LR-23 | 반환된 행의 priority_score | 나머지보다 낮지 않음 (min >= threshold) |

### 2.4 save_labels()

| 테스트 ID | 입력 | 기댓값 |
| --- | --- | --- |
| T-LR-30 | corrections 적용 후 저장 | 파일 존재 |
| T-LR-31 | corrections 키의 레이블 값 | CSV에서 해당 경로 행의 level == 수정값 |
| T-LR-32 | corrections에 없는 경로 | 원본 level 유지 |
| T-LR-33 | 원본 CSV 미존재 | `FileNotFoundError` |

```python
def test_save_labels(tmp_path):
    # 원본 CSV 생성
    original = tmp_path / "labels_v0.csv"
    df = pd.DataFrame({
        "path": ["a.png", "b.png", "c.png"],
        "channel": ["Y", "Y", "Y"],
        "level": [1, 2, 3],
    })
    df.to_csv(original, index=False)

    refiner = LabelRefiner(cfg={})
    output = tmp_path / "labels_v1.csv"
    refiner.save_labels(original, corrections={"a.png": 5}, output_path=output)

    result = pd.read_csv(output)
    assert result[result["path"] == "a.png"]["level"].iloc[0] == 5
    assert result[result["path"] == "b.png"]["level"].iloc[0] == 2  # 유지
```

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_ROI_Pipeline.md](../SSOT/SSOT_ROI_Pipeline.md) | LabelRefiner 정의 |
| [Contract_roi_pipeline.md](../Contract/Contract_roi_pipeline.md) | API 계약 |
