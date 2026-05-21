---
type: bdd
domain: swing_feedback
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Evaluation_Reporting.md"
  - "Contract_evaluation_reporting.md"
---

# [BDD] Swing Feedback 루프 / Swing Feedback Loop

> **역할 / Role**: 평가 결과에 따라 시스템이 재학습 방향을 자동으로 제시하는 행동을 정의한다.
> **Role**: Defines the system's behavior of automatically suggesting retraining direction based on evaluation results.

---

## 행위자 / Actors

| 행위자 / Actor | 역할 / Role |
|---|---|
| **품질 관리자 / QC Manager** | 검사 결과를 기반으로 재학습/통과 여부를 결정하는 관리자 / Manager deciding retrain/pass based on evaluation |
| **시스템 / System** | Swing Feedback 결정을 자동 계산 / Automatically computes Swing Feedback decisions |

---

## Feature: Swing Feedback 루프 / Swing Feedback Loop

> **비즈니스 가치 / Business Value**: 평가 결과가 목표 미달일 때 시스템이 자동으로 재학습 방향을 제시하여, 품질 관리자가 수동으로 판단하지 않아도 된다.
>
> **Business Value**: When evaluation results fall below targets, the system automatically suggests a retraining direction, eliminating the need for QC Managers to decide manually.

---

### Scenario 3.1 — 모든 목표 달성 시 PASS / All Targets Met → PASS

```gherkin
Feature: Swing Feedback 루프 / Swing Feedback loop

  Scenario: 모든 성능 목표 달성 시 Swing이 종료된다
  Scenario: Swing terminates when all performance targets are met

    Given 전체 채널(Y, M, C, K) 평가가 완료된 상태이다
    And   Overall Accuracy >= 0.90이다
    And   Overall Macro F1 >= 0.85이다
    And   Overall MAE <= 0.50이다
    And   색상별 Accuracy >= 0.85이다

    When  시스템이 determine_swing_feedback()를 실행한다

    Then  피드백 결과 dict의 terminate 값이 True이다
    And   피드백 결과 dict의 decisions 목록이 비어 있다
    And   Swing Architecture가 종료된다
    And   최종 모델 best_{channel}.pt가 배포 준비 상태이다
```

---

### Scenario 3.2 — 정확도 미달 → Phase 0 재학습 / Accuracy Below Threshold → Retry Phase 0

```gherkin
  Scenario: 색상별 Accuracy가 임계값 미달 시 Phase 0 재학습이 권장된다
  Scenario: Phase 0 retraining is recommended when per-color Accuracy is below threshold

    Given Y채널 평가 Accuracy가 0.80 미만이다
    And   MAE가 0.80 초과이다

    When  시스템이 determine_swing_feedback()를 실행한다

    Then  피드백 결과 dict의 terminate 값이 False이다
    And   피드백 결과 dict의 decisions 목록에 "[Y] Accuracy … -> Phase 0 (retrain representation)" 항목이 포함된다
    And   운영자에게 Phase 0 재실행을 안내한다
```

---

### Scenario 3.3 — F1 미달 → Phase 2 재학습 / F1 Below Threshold → Retry Phase 2

```gherkin
  Scenario: 특정 Level F1이 임계값 미달 시 Phase 2 재학습이 권장된다
  Scenario: Phase 2 retraining is recommended when per-level F1 is below threshold

    Given Level 3 F1 점수가 0.70 미만이다
    And   Overall Accuracy는 0.80 이상이다

    When  시스템이 determine_swing_feedback()를 실행한다

    Then  피드백 결과 dict의 terminate 값이 False이다
    And   피드백 결과 dict의 decisions 목록에 "Level 3 F1 … -> Phase 2 (retrain classifier)" 항목이 포함된다
    And   Phase 0 backbone은 유지되고 Phase 2만 재실행된다
```

---

### Scenario 3.4 — 경계 레벨 MAE 감시 / Boundary Level MAE Monitoring

```gherkin
  Scenario: MAE가 임계값을 초과하면 표현 학습 재시도가 권장된다
  Scenario: Representation re-learning is recommended when MAE exceeds threshold

    Given Overall MAE가 0.80 초과이다

    When  시스템이 피드백을 계산한다

    Then  "Overall MAE > 0.80 → Phase 0 재시도" 결정이 반환된다
    And   현재 Phase 0 backbone이 유효하지 않은 것으로 판단된다
```

---

### Scenario 3.5 — Swing Efficiency 조기 종료 / Swing Efficiency Early Stop

```gherkin
  Scenario: 2사이클 이상에서 효율비가 50% 미만이면 사이클이 종료된다
  Scenario: Cycle terminates when efficiency ratio drops below 50% after 2+ cycles

    Given Cycle 1의 efficiency_ratio가 0.40이다
    And   Cycle 2의 efficiency_ratio가 Cycle 1의 50% 미만이다

    When  should_early_stop([cycle1_report, cycle2_report])가 호출된다

    Then  반환값이 True이다
    And   운영자에게 사이클 조기 종료를 안내한다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 / Scenario | TDD 파일 / TDD File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| 3.1 — PASS feedback | `test_metrics.py` | `test_determine_swing_feedback_pass` | Unit |
| 3.2 — retry_phase0 | `test_metrics.py` | `test_determine_swing_feedback_retry_phase0` | Unit |
| 3.3 — retry_phase2 | `test_metrics.py` | `test_determine_swing_feedback_retry_phase2` | Unit |
| 3.4 — MAE monitoring | `test_metrics.py` | `test_determine_swing_feedback_mae_exceeds` | Unit |
| 3.5 — Early stop | `test_swing_efficiency.py` | `test_should_early_stop_returns_true` | Unit |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [Contract_evaluation_reporting.md](../Contract/Contract_evaluation_reporting.md) | determine_swing_feedback() 계약 / Contract |
| [SSOT_Evaluation_Reporting.md](../SSOT/SSOT_Evaluation_Reporting.md) | 성능 목표값 정의 / Performance target definitions |
| [TDD_SwingEfficiency.md](../TDD/TDD_SwingEfficiency.md) | Swing Efficiency TDD 명세 / TDD specification |
