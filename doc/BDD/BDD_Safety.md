---
type: bdd
domain: safety
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "Contract_fail_fast.md"
  - "SSOT_Validation_Codes.md"
---

# [BDD] 시스템 안전성 / System Safety (Fail-Fast)

> **역할 / Role**: 잘못된 아티팩트나 설정이 사용될 때 시스템이 즉시 실패하는 행동을 정의한다.
> **Role**: Defines the system's behavior of failing immediately when incorrect artifacts or configurations are used.

---

## 행위자 / Actors

| 행위자 / Actor | 역할 / Role |
|---|---|
| **시스템 / System** | Fail-Fast 검증을 수행하고 즉시 오류를 발생시키는 파이프라인 / Pipeline performing Fail-Fast validation and raising errors immediately |
| **운영자 / Operator** | 학습 파이프라인을 실행하는 엔지니어 / Engineer running training pipeline |

---

## Feature: 시스템 안전성 / System Safety (Fail-Fast)

> **비즈니스 가치 / Business Value**: 잘못된 아티팩트나 설정이 사용될 경우 즉시 오류를 발생시켜 오염된 결과물이 생성되는 것을 방지한다.
>
> **Business Value**: When incorrect artifacts or configurations are used, the system fails immediately to prevent contaminated results from being produced.

---

### Scenario 7.1 — Phase 0 Backbone 누락 시 즉시 실패 / Immediate Failure on Missing Phase 0 Backbone

```gherkin
Feature: 시스템 안전성 / System safety (Fail-Fast)

  Scenario: Phase 0 backbone 파일 없이 Phase 2 실행 시 즉시 실패한다
  Scenario: Phase 2 fails immediately when Phase 0 backbone file is missing

    Given phase0_backbone_Y_effb0.pt 파일이 존재하지 않는다

    When  운영자가 Y채널 Phase 2 학습을 시작하려 한다

    Then  시스템은 FileNotFoundError를 즉시 발생시킨다
    And   SSOT-FF01 코드가 로그에 기록된다
    And   Phase 2 학습이 시작되지 않는다
    And   fallback 모델이 생성되지 않는다
```

---

### Scenario 7.2 — 필수 Config 키 누락 시 즉시 실패 / Immediate Failure on Missing Config Key

```gherkin
  Scenario: 필수 config 키 누락 시 즉시 실패한다
  Scenario: System fails immediately when required config key is missing

    Given config.json에 "data.num_levels" 키가 없다

    When  GrayspotModel이 초기화된다

    Then  시스템은 KeyError를 발생시킨다
    And   SSOT-CF01 코드가 로그에 기록된다
    And   기본값(default)으로 대체되지 않는다
```

---

### Scenario 7.3 — 정규화 적용 검증 / Normalization Application Verification

> ✅ **SSOT-NM01 해소됨 / Resolved**: `predictor_inference.py` `_preprocess_images()` 에서 `_IMAGENET_NORMALIZE` 적용 완료 (2026-05-14).

```gherkin
  Scenario: GrayspotPredictor 추론 시 ImageNet 정규화가 학습과 동일하게 적용된다
  Scenario: GrayspotPredictor applies ImageNet normalization identical to training

    Given GrayspotPredictor가 초기화되어 있다

    When  predict()가 BGR numpy 이미지 배열을 입력받는다

    Then  내부적으로 [0,1] 정규화 후 ImageNet mean/std 변환이 적용된다
    And   dataset.py의 _IMAGENET_NORMALIZE와 동일한 변환이 보장된다
    And   학습(dataset.py)과 추론(predictor_inference.py) 간 정규화 불일치가 없다
```

---

### Scenario 7.4 — Dead Config 감지 / Dead Config Detection

```gherkin
  Scenario: 코드에서 미소비되는 config 키가 경고 로그로 기록된다
  Scenario: Config keys not consumed by code are logged as warnings

    Given config.json에 코드에서 참조하지 않는 키가 선언되어 있다

    When  validate_config()가 실행된다

    Then  SSOT-CF02 경고가 Level 2 (Warning + Continue) 수준으로 로그에 기록된다
    And   실행이 중단되지 않는다
    And   Dead Config 키 목록이 출력된다
```

---

### Scenario 7.5 — switch_to_phase2() Backbone 키 없을 시 즉시 실패 / Immediate Failure When No Backbone Keys Loaded

```gherkin
  Scenario: switch_to_phase2()에서 backbone 키가 0개 로드되면 즉시 실패한다
  Scenario: switch_to_phase2() fails immediately when zero backbone keys are loaded

    Given 아키텍처가 변경되어 phase0 checkpoint의 backbone 키와 현재 모델의 키가 불일치한다
    And   phase0_backbone_Y_effb0.pt 파일은 존재한다

    When  model.switch_to_phase2(backbone_path, cfg)가 호출된다

    Then  시스템은 RuntimeError를 발생시킨다 (SSOT-FF01)
    And   오류 메시지에 "Zero backbone keys loaded" 또는 동등 내용이 포함된다
    And   잘못된 backbone weights로 Phase 2 학습이 시작되지 않는다
    And   경고(warning)로만 처리되지 않는다
```

---

### Scenario 7.6 — validate_config() 실패 시 ValueError 발생 / ValueError on Config Validation Failure

```gherkin
  Scenario: validate_config()가 필수 키 누락 시 ValueError를 발생시킨다
  Scenario: validate_config() raises ValueError when required keys are missing

    Given config dict에 "phase2.epochs" 키가 없다

    When  validate_config(cfg)가 호출된다

    Then  시스템은 ValueError를 발생시킨다 (SSOT-CF01)
    And   False를 반환하지 않는다
    And   오류 메시지에 누락된 키 이름이 포함된다
    And   실행이 즉시 중단된다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 / Scenario | TDD 파일 / TDD File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| 7.1 — FF01 backbone 누락 / missing | `test_smoke_phase2.py` | `test_phase2_fails_without_phase0_backbone` | Smoke |
| 7.2 — CF01 config 키 누락 / missing key | `test_utils_config.py` | `test_validate_config_missing_key_raises` | Unit |
| 7.3 — NM01 정규화 적용 / normalization | `test_predictor.py` | `test_preprocess_images_imagenet_normalized` | Unit |
| 7.4 — CF02 dead config | `test_utils_config.py` | `test_validate_config_dead_config_warning` | Unit |
| 7.5 — FF01 backbone 키 0개 / zero keys | `test_models.py` | `test_switch_to_phase2_zero_keys_raises` | Unit |
| 7.6 — CF01 ValueError | `test_utils_config.py` | `test_validate_config_raises_value_error` | Unit |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [Contract_fail_fast.md](../Contract/Contract_fail_fast.md) | Fail-Fast 계약 상세 / Fail-Fast contract details |
| [SSOT_Validation_Codes.md](../SSOT/SSOT_Validation_Codes.md) | 에러 코드 정의 / Error code definitions |
| [SSOT_Core.md](../SSOT/SSOT_Core.md) | Fail-Fast 정책 원칙 / Fail-Fast policy principles |
