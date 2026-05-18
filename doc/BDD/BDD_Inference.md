---
type: bdd
domain: inference
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "Contract_inference_boundary.md"
---

# [BDD] 추론 파이프라인 / Inference Pipeline

> **역할 / Role**: 품질 검사원(QC Inspector)이 그레이스팟 결함을 분류할 때의 관찰 가능한 행동을 Given-When-Then 시나리오로 정의한다.
> **Role**: Defines observable behavior when a QC Inspector classifies grayspot defects using Given-When-Then scenarios.

---

## 행위자 / Actors

| 행위자 / Actor | 역할 / Role |
|---|---|
| **품질 검사원 / QC Inspector** | 인쇄물 결함 수준을 판정하는 현장 검사자 / Field inspector judging print defect levels |
| **시스템 / System** | CMYK Grayspot Detection Pipeline 자체 / The CMYK Grayspot Detection Pipeline itself |

---

## Feature: 그레이스팟 결함 분류 / Grayspot Defect Classification

> **비즈니스 가치 / Business Value**: 품질 검사원이 인쇄물 패치 이미지를 시스템에 입력하면, Level 0(정상) ~ Level 5(최대 결함)의 6단계 분류 결과와 신뢰도를 즉시 반환받아 수작업 검사 시간을 단축한다.
>
> **Business Value**: When a QC Inspector inputs a print patch image, the system immediately returns a 6-level classification (Level 0 normal to Level 5 maximum defect) with confidence, reducing manual inspection time.

---

### Scenario 1.1 — 정상 이미지 자동 수락 / Normal Image Auto-Accept

```gherkin
Feature: 그레이스팟 결함 분류 / Grayspot defect classification

  Scenario: 신뢰도 높은 Level 0(정상) 이미지가 자동 수락된다
  Scenario: High-confidence Level 0 (normal) image is auto-accepted

    Given 학습 완료된 Y채널 Phase 2 모델이 로드된 상태이다
    And   입력 이미지는 128×128 BGR 패치이다

    When  품질 검사원이 Level 0 (정상) 패치 이미지를 시스템에 제출한다
    And   모델 예측 신뢰도가 0.80 이상이다

    Then  시스템은 예측 Level "0"을 반환한다
    And   분류 플래그는 "AUTO_ACCEPT"이다
    And   결과가 1초 이내에 반환된다
```

---

### Scenario 1.2 — 중간 신뢰도 이미지 경고 / Medium-Confidence Image Warning

```gherkin
  Scenario: 신뢰도 중간 이미지에 대해 경고가 표시된다
  Scenario: Warning is displayed for medium-confidence image

    Given 학습 완료된 M채널 Phase 2 모델이 로드된 상태이다

    When  품질 검사원이 패치 이미지를 시스템에 제출한다
    And   모델 예측 신뢰도가 0.50 이상 0.80 미만이다

    Then  시스템은 예측 Level을 반환한다
    And   분류 플래그는 "WARN"이다
    And   검사원에게 결과를 재확인하도록 안내한다
```

---

### Scenario 1.3 — 저신뢰도 이미지 수동 검토 요청 / Low-Confidence Manual Review

```gherkin
  Scenario: 저신뢰도 이미지는 수동 검토를 요청한다
  Scenario: Low-confidence image triggers manual review request

    Given 학습 완료된 C채널 Phase 2 모델이 로드된 상태이다

    When  품질 검사원이 패치 이미지를 시스템에 제출한다
    And   모델 예측 신뢰도가 0.30 미만이다

    Then  시스템은 예측 Level을 반환하되
    And   분류 플래그는 "MANUAL_REVIEW"이다
    And   자동 수락하지 않고 검사원의 최종 판단을 요청한다
```

---

### Scenario 1.4 — 최대 결함 이미지 분류 / Maximum Defect Image Classification

```gherkin
  Scenario: Level 5(최대 결함) 이미지가 올바르게 분류된다
  Scenario: Level 5 (maximum defect) image is correctly classified

    Given 학습 완료된 K채널 Phase 2 모델이 로드된 상태이다
    And   입력 이미지는 심각한 그레이스팟 결함을 포함한다

    When  품질 검사원이 해당 패치를 시스템에 제출한다

    Then  시스템은 예측 Level "5"에 가까운 값을 반환한다
    And   MAE(평균 절대 오차 / Mean Absolute Error)는 1.0 이하이다
    And   해당 이미지는 오분류 목록에 기록되지 않는다
```

---

### Scenario 1.5 — Raw Logit 반환 보장 / Raw Logit Return Guarantee

```gherkin
  Scenario: 모델 출력은 항상 Softmax 없는 raw logit이다
  Scenario: Model output is always raw logit without Softmax

    Given Phase 2 모델이 로드된 상태이다

    When  배치 이미지 (B, 3, 128, 128)가 모델에 입력된다

    Then  출력 텐서 shape은 (B, 6)이다
    And   출력값의 합이 1.0이 아닐 수 있다 (Softmax 미적용 / Softmax not applied)
    And   CrossEntropyLoss가 올바르게 적용 가능한 형태이다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 / Scenario | TDD 파일 / TDD File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| 1.1 — AUTO_ACCEPT | `test_predictor.py` | `test_confidence_auto_accept_flag` | Unit |
| 1.2 — WARN | `test_predictor.py` | `test_confidence_warn_flag` | Unit |
| 1.3 — MANUAL_REVIEW | `test_predictor.py` | `test_confidence_manual_review_flag` | Unit |
| 1.4 — Level 5 분류 / classification | `test_predictor.py` | `test_level5_mae_below_threshold` | Integration |
| 1.5 — Raw logit | `test_models.py` | `test_output_is_logits_not_probability` | Unit |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [Contract_inference_boundary.md](../Contract/Contract_inference_boundary.md) | 추론 모듈 경계 계약 / Inference module boundary contract |
| [SSOT_Core.md](../SSOT/SSOT_Core.md) | 신뢰도 임계값 정의 / Confidence threshold definitions |
| [BDD_Safety.md](BDD_Safety.md) | 정규화 적용 시나리오 (8.3) / Normalization scenario |
