---
type: bdd
domain: tuning
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Training_Pipeline.md"
  - "Contract_tuning_boundary.md"
---

# [BDD] Optuna 하이퍼파라미터 튜닝 / Optuna Hyperparameter Tuning

> **역할 / Role**: 데이터 과학자(Data Scientist)가 Optuna로 하이퍼파라미터를 최적화할 때의 관찰 가능한 행동을 정의한다.
> **Role**: Defines observable behavior when a Data Scientist optimizes hyperparameters using Optuna.

---

## 행위자 / Actors

| 행위자 / Actor | 역할 / Role |
|---|---|
| **데이터 과학자 / Data Scientist** | Optuna 튜닝을 실행하고 결과를 분석하는 연구자 / Researcher running Optuna tuning and analyzing results |
| **시스템 / System** | TPE 샘플러로 탐색 공간을 최적화하는 파이프라인 / Pipeline optimizing search space with TPE sampler |

---

## Feature: Optuna 하이퍼파라미터 튜닝 / Optuna Hyperparameter Tuning

> **비즈니스 가치 / Business Value**: 데이터 과학자가 수동으로 파라미터를 탐색하지 않고, Optuna가 자동으로 최적 하이퍼파라미터를 탐색하여 모델 성능을 향상시킨다.
>
> **Business Value**: Instead of manual parameter search by Data Scientists, Optuna automatically finds optimal hyperparameters to improve model performance.

---

### Scenario 5.1 — 단일 채널 Optuna 튜닝 / Single-Channel Optuna Tuning

```gherkin
Feature: Optuna 하이퍼파라미터 튜닝 / Optuna hyperparameter tuning

  Scenario: 단일 채널에 대해 Optuna 튜닝이 실행된다
  Scenario: Optuna tuning runs for a single channel

    Given config.json의 optuna.n_trials가 30으로 설정되어 있다
    And   optuna.sampler가 "tpe"로 설정되어 있다
    And   data_set/labeled/Y/ 경로에 학습 데이터가 존재한다

    When  데이터 과학자가 `python -m src.scripts.run_optuna --channel Y`를 실행한다

    Then  30회의 trial이 수행된다
    And   각 trial에서 val_acc를 목적 함수로 최적화한다
    And   outputs/optuna/best_params_y.json에 최적 파라미터가 저장된다
    And   outputs/optuna/trials_summary_y.json에 전체 trial 결과가 저장된다
```

---

### Scenario 5.2 — Backbone별 독립 탐색 공간 적용 / Per-Backbone Search Space

```gherkin
  Scenario: EfficientNet-B0와 ResNet-50은 서로 다른 탐색 공간을 사용한다
  Scenario: EfficientNet-B0 and ResNet-50 use different search spaces

    Given config.json의 model.backbone이 "resnet50"으로 설정되어 있다
    And   optuna.search_space.resnet50에 mid_dim 탐색 범위가 정의되어 있다

    When  Optuna trial이 실행된다

    Then  mid_dim 하이퍼파라미터가 탐색 범위에서 샘플링된다
    And   EfficientNet-B0 전용 탐색 공간(mid_dim 없음)과 혼용되지 않는다
```

---

### Scenario 5.3 — MedianPruner 조기 종료 / MedianPruner Early Stopping

```gherkin
  Scenario: 성능 하위 trial이 조기 종료된다
  Scenario: Under-performing trials are pruned early

    Given optuna.pruner.n_warmup_steps가 10으로 설정되어 있다

    When  10 epoch 이후 trial이 중앙값보다 낮은 val_acc를 기록한다

    Then  해당 trial이 자동으로 중단된다
    And   나머지 trial 리소스가 더 유망한 파라미터 탐색에 사용된다
```

---

### Scenario 5.4 — Optuna n_trials=0 처리 / Optuna n_trials=0 Edge Case

```gherkin
  Scenario: n_trials가 config에 없거나 0이면 기본값(5)으로 폴백된다
  Scenario: n_trials falls back to default (5) when absent or 0 in config

    Given config.json에 optuna.n_trials 키가 없다

    When  데이터 과학자가 n_trials 인자 없이 run_optuna()를 호출한다

    Then  n_trials = 5 (기본값 / default)으로 실행된다
    And   0번 실행되거나 오류가 발생하지 않는다
```

---

### Scenario 5.5 — 역방향 의존성 격리 / Reverse Dependency Isolation

```gherkin
  Scenario: optuna_tuner.py의 run_phase2 의존성이 런타임에만 활성화된다
  Scenario: run_phase2 dependency in optuna_tuner.py is activated only at runtime

    Given optuna_tuner 모듈이 import된다

    When  objective() 함수 내부에서만 run_phase2가 lazy import된다

    Then  tuning 모듈이 scripts 모듈 없이 import될 수 있다
    And   순환 import 오류가 발생하지 않는다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 / Scenario | TDD 파일 / TDD File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| 5.1 — val_acc 목적 함수 / objective | `test_smoke_optuna.py` | `test_optuna_objective_returns_float` | Smoke |
| 5.2 — backbone 탐색 공간 / search space | `test_search_space.py` | `test_resnet50_search_space_has_mid_dim` | Unit |
| 5.3 — MedianPruner | `test_smoke_optuna.py` | `test_optuna_pruner_activates` | Smoke |
| 5.4 — n_trials 기본값 / default | `test_smoke_optuna.py` | `test_optuna_default_n_trials_fallback` | Smoke |
| 5.5 — 역방향 의존성 / reverse dep | `test_smoke_optuna.py` | `test_tuning_module_imports_without_scripts` | Smoke |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [Contract_tuning_boundary.md](../Contract/Contract_tuning_boundary.md) | Optuna 튜닝 경계 계약 / Optuna tuning boundary contract |
| [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) | Phase 2 탐색 공간 정의 / Phase 2 search space definitions |
