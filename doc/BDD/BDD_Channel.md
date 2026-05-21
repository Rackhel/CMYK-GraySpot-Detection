---
type: bdd
domain: channel
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "SSOT_Data_Pipeline.md"
---

# [BDD] CMYK 채널 독립 관리 / CMYK Channel Independence

> **역할 / Role**: CMYK 4색 채널이 독립 모델로 관리될 때의 관찰 가능한 행동을 정의한다.
> **Role**: Defines observable behavior when CMYK 4-color channels are managed as independent models.

---

## 행위자 / Actors

| 행위자 / Actor | 역할 / Role |
|---|---|
| **운영자 / Operator** | 채널별 학습·추론을 실행하는 엔지니어 / Engineer running per-channel training and inference |
| **시스템 / System** | 채널 불변식을 강제하는 파이프라인 / Pipeline enforcing channel invariants |

---

## Feature: CMYK 채널 독립 관리 / CMYK Channel Independence

> **비즈니스 가치 / Business Value**: CMYK 4색 분판 각각을 독립 모델로 관리함으로써 색상별 결함 패턴의 차이를 정밀하게 탐지한다.
>
> **Business Value**: Managing each of the 4 CMYK color separations as an independent model enables precise detection of channel-specific defect patterns.

---

### Scenario 6.1 — 채널 불변식 준수 / Channel Invariant Enforcement

```gherkin
Feature: CMYK 채널 독립 관리 / CMYK channel independence

  Scenario: Y채널 backbone으로 M채널 Phase 2를 시작하려 하면 실패한다
  Scenario: Phase 2 with Y-channel backbone fails when starting M-channel Phase 2

    Given Y채널로 학습된 phase0_backbone_Y_effb0.pt가 존재한다

    When  운영자가 M채널 Phase 2를 Y채널 backbone으로 시작하려 한다

    Then  시스템이 SSOT-FF01 오류를 발생시킨다
    And   "channel mismatch" 오류 메시지가 출력된다
    And   M채널 Phase 2 학습이 시작되지 않는다
```

---

### Scenario 6.2 — 4채널 독립 병렬 처리 / 4-Channel Independent Processing

```gherkin
  Scenario: Y, M, C, K 채널이 각각 독립 모델로 학습된다
  Scenario: Y, M, C, K channels are each trained as independent models

    Given 4채널 모두의 데이터가 data_set/labeled/ 하위에 존재한다

    When  운영자가 전체 채널에 대해 학습 파이프라인을 실행한다

    Then  phase0_backbone_Y_effb0.pt, phase0_backbone_M_effb0.pt,
          phase0_backbone_C_effb0.pt, phase0_backbone_K_effb0.pt가 각각 생성된다
    And   best_Y.pt, best_M.pt, best_C.pt, best_K.pt가 각각 생성된다
    And   각 모델은 서로 다른 채널의 가중치를 공유하지 않는다
```

---

### Scenario 6.3 — BGR 색상 공간 일관성 / BGR Color Space Consistency

```gherkin
  Scenario: 학습부터 평가까지 BGR 색상 공간이 유지된다
  Scenario: BGR color space is maintained throughout training and evaluation

    Given 이미지가 cv2.imread()로 로드된다 (BGR 형식 / BGR format)

    When  이미지가 전처리 파이프라인을 통과한다
    And   모델에 입력된다

    Then  RGB 변환이 적용되지 않는다
    And   학습과 추론 모두 동일한 BGR 통계(ImageNet 정규화)가 적용된다
    And   SSOT-CS01 위반이 발생하지 않는다
```

---

### Scenario 6.4 — ROI CMYK 분판 일관성 / ROI CMYK Split Consistency

```gherkin
  Scenario: ROIExtractor의 CMYK 분판이 수식 정의를 따른다
  Scenario: ROIExtractor CMYK split follows the formula definition

    Given RGB 이미지(H, W, 3)가 입력된다

    When  split_cmyk(img)가 호출된다

    Then  C = 1 - R, M = 1 - G, Y = 1 - B, K = min(C, M, Y) 공식이 적용된다
    And   각 채널 값이 [0, 1] 범위 내에 있다
    And   dtype이 float32이다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 / Scenario | TDD 파일 / TDD File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| 6.1 — 채널 불변식 / channel invariant | `test_smoke_phase2.py` | `test_channel_mismatch_raises_ff01` | Smoke |
| 6.2 — 4채널 독립 / independence | `test_smoke_phase2.py` | `test_4channel_independent_checkpoints` | Smoke |
| 6.3 — BGR 일관성 / consistency | `test_preprocessing.py` | `test_no_rgb_conversion` | Unit |
| 6.4 — CMYK 분판 / split | `test_roi_extractor.py` | `test_split_cmyk_white_pixel` | Unit |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [SSOT_Core.md](../SSOT/SSOT_Core.md) | 채널 불변식 정의 (SSOT-FF01) / Channel invariant definition |
| [SSOT_Data_Pipeline.md](../SSOT/SSOT_Data_Pipeline.md) | BGR 규약 정의 / BGR convention definition |
| [BDD_ROI_Pipeline.md](BDD_ROI_Pipeline.md) | ROI CMYK 분판 시나리오 / ROI CMYK split scenarios |
