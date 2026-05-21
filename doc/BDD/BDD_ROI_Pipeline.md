---
type: bdd
domain: roi_pipeline
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_ROI_Pipeline.md"
  - "Contract_roi_pipeline.md"
---

# [BDD] ROI 파이프라인 / ROI Pipeline

> **역할 / Role**: ROI 추출, CMYK 분판, 라벨 정제 기능의 관찰 가능한 행동을 정의한다.
> **Role**: Defines observable behavior for ROI extraction, CMYK splitting, and label refinement features.

---

## 행위자 / Actors

| 행위자 / Actor | 역할 / Role |
|---|---|
| **운영자 / Operator** | ROI 추출 및 라벨 관리를 수행하는 엔지니어 / Engineer performing ROI extraction and label management |
| **데이터 과학자 / Data Scientist** | 라벨 품질을 검토하고 정제하는 연구자 / Researcher reviewing and refining label quality |
| **시스템 / System** | ROI 추출 → CMYK 분판 → 라벨 스코어링을 자동 수행 / Automatically performs ROI extraction → CMYK split → label scoring |

---

## Feature: ROI 추출 및 CMYK 분판 / ROI Extraction and CMYK Splitting

---

### Scenario R.1 — ROI 패치 추출 성공 / ROI Patch Extraction Success

```gherkin
Feature: ROI 추출 / ROI extraction

  Scenario: 유효한 이미지 경로에서 ROI 패치가 추출된다
  Scenario: ROI patches are extracted from a valid image path

    Given 유효한 PNG 이미지 파일(H×W×3)이 존재한다
    And   ROIExtractor가 patch_size=128, stride=64로 초기화된다

    When  extract_patches(image_path)가 호출된다

    Then  반환 리스트의 각 원소 shape이 (128, 128, 3)이다
    And   반환 리스트가 비어 있지 않다
    And   dtype이 float32이다
```

---

### Scenario R.2 — 존재하지 않는 파일 처리 / Non-Existent File Handling

```gherkin
  Scenario: 존재하지 않는 이미지 경로가 입력되면 FileNotFoundError가 발생한다
  Scenario: FileNotFoundError is raised when image path does not exist

    Given 존재하지 않는 파일 경로가 주어진다

    When  extract_patches(non_existent_path)가 호출된다

    Then  FileNotFoundError가 즉시 발생한다
    And   빈 리스트를 반환하지 않는다
```

---

### Scenario R.3 — CMYK 분판 공식 검증 / CMYK Split Formula Validation

```gherkin
  Scenario: 흰색 픽셀의 CMYK 분판 결과가 수식과 일치한다
  Scenario: CMYK split of white pixel matches the formula

    Given RGB 픽셀값이 (255, 255, 255)인 흰색 이미지가 주어진다

    When  split_cmyk(img)가 호출된다

    Then  C = 0.0, M = 0.0, Y = 0.0, K = 0.0이 반환된다
    And   모든 채널 값이 [0.0, 1.0] 범위 내에 있다
```

---

### Scenario R.4 — 단채널 이미지 3채널 변환 / Single-Channel to 3-Channel Conversion

```gherkin
  Scenario: 단채널(Grayscale) 이미지가 3채널로 자동 변환된다
  Scenario: Grayscale image is automatically converted to 3-channel

    Given 단채널(H, W) 이미지가 입력된다

    When  extract_patches(image_path)가 호출된다

    Then  출력 패치 shape이 (128, 128, 3)이다
    And   3채널 모두 동일한 값을 가진다 (grayscale 복제 / grayscale duplication)
```

---

## Feature: 라벨 정제 / Label Refinement

---

### Scenario R.5 — 우선순위 스코어 정렬 / Priority Score Sorting

```gherkin
Feature: 라벨 정제 / Label refinement

  Scenario: compute_priority_score()가 우선순위 내림차순으로 DataFrame을 반환한다
  Scenario: compute_priority_score() returns DataFrame sorted by priority descending

    Given labels_v0.csv에 image_path, level 컬럼이 존재한다
    And   LabelRefiner가 초기화된다

    When  compute_priority_score()가 호출된다

    Then  반환 DataFrame에 priority_score 컬럼이 존재한다
    And   priority_score 기준 내림차순으로 정렬되어 있다
    And   모든 값이 [0.0, 1.0] 범위 내에 있다
```

---

### Scenario R.6 — 검토 큐 상위 20% 추출 / Review Queue Top 20% Extraction

```gherkin
  Scenario: get_review_queue()가 상위 20% 샘플을 반환한다
  Scenario: get_review_queue() returns top 20% samples

    Given 100개의 샘플이 labels_v0.csv에 존재한다
    And   priority_score가 계산된 상태이다

    When  get_review_queue(top_ratio=0.2)가 호출된다

    Then  반환 DataFrame의 행 수가 20이다
    And   priority_score 기준 상위 20개이다
```

---

### Scenario R.7 — 라벨 버전 저장 / Label Version Save

```gherkin
  Scenario: save_labels()가 버전 증가된 CSV를 저장한다
  Scenario: save_labels() saves CSV with incremented version

    Given labels_v0.csv가 존재하는 상태이다
    And   일부 라벨이 수정된다

    When  save_labels(updated_df)가 호출된다

    Then  labels_v1.csv가 새로 생성된다
    And   labels_v0.csv는 변경되지 않는다
    And   image_path, level 컬럼이 포함된다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 / Scenario | TDD 파일 / TDD File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| R.1 — ROI 패치 형상 / shape | `test_roi_extractor.py` | `test_extract_patches_output_shape` | Unit |
| R.2 — FileNotFoundError | `test_roi_extractor.py` | `test_extract_patches_missing_file` | Unit |
| R.3 — 흰색 픽셀 분판 / white split | `test_roi_extractor.py` | `test_split_cmyk_white_pixel` | Unit |
| R.4 — 단채널 변환 / grayscale convert | `test_roi_extractor.py` | `test_extract_patches_grayscale_equal_3ch` | Unit |
| R.5 — 우선순위 정렬 / priority sort | `test_label_refiner.py` | `test_priority_score_in_range` | Unit |
| R.6 — 검토 큐 20% / queue 20% | `test_label_refiner.py` | `test_review_queue_returns_top_ratio` | Unit |
| R.7 — 라벨 버전 / label version | `test_label_refiner.py` | `test_save_labels_increments_version` | Unit |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [Contract_roi_pipeline.md](../Contract/Contract_roi_pipeline.md) | ROI 파이프라인 계약 / ROI pipeline contract |
| [SSOT_ROI_Pipeline.md](../SSOT/SSOT_ROI_Pipeline.md) | CMYK 분판 공식, 우선순위 스코어 가중치 / CMYK split formulas and priority score weights |
| [TDD_ROI_Pipeline.md](../TDD/TDD_ROI_Pipeline.md) | ROI TDD 명세 / ROI TDD specification |
| [TDD_LabelRefiner.md](../TDD/TDD_LabelRefiner.md) | LabelRefiner TDD 명세 / LabelRefiner TDD specification |
