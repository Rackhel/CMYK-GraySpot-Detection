---
type: bdd
domain: training
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Training_Pipeline.md"
  - "Contract_training_pipeline.md"
---

# [BDD] 학습 파이프라인 / Training Pipeline

> **역할 / Role**: 운영자(Operator)가 Phase 0 → Phase 2 학습을 실행할 때의 관찰 가능한 행동을 정의한다.
> **Role**: Defines observable behavior when an Operator runs Phase 0 → Phase 2 training.

---

## 행위자 / Actors

| 행위자 / Actor | 역할 / Role |
|---|---|
| **운영자 / Operator** | 학습 파이프라인을 실행하고 모델을 배포하는 엔지니어 / Engineer who runs training and deploys models |
| **시스템 / System** | CMYK Grayspot Detection Pipeline 자체 / The CMYK Grayspot Detection Pipeline itself |

---

## Feature: 학습 파이프라인 실행 / Training Pipeline Execution

> **비즈니스 가치 / Business Value**: 운영자가 학습 스크립트를 실행하면 Phase 0 → Phase 2 순서로 자동 진행되어 재현 가능한 모델이 생성된다.
>
> **Business Value**: When an Operator runs the training scripts, Phase 0 → Phase 2 proceeds automatically in order, producing a reproducible model.

---

### Scenario 2.1 — Phase 0 SimCLR 사전 학습 성공 / Phase 0 Pre-training Success

```gherkin
Feature: 학습 파이프라인 실행 / Training pipeline execution

  Scenario: Phase 0 SimCLR 사전 학습이 정상 완료된다
  Scenario: Phase 0 SimCLR pre-training completes successfully

    Given config.json에 유효한 Phase 0 파라미터가 정의되어 있다
    And   data_set/labeled/{channel}/{level}/ 경로에 이미지가 존재한다
    And   EfficientNet-B0 (또는 ResNet-50) pretrained weights를 다운로드할 수 있다

    When  운영자가 Y채널에 대해 `python -m src.scripts.run_phase0 --channel Y`를 실행한다

    Then  학습이 완료되고 data_set/models/phase0_backbone_Y_effb0.pt 파일이 생성된다
    And   파일에는 backbone.* 키와 head.* 키(ProjectionHead)가 포함된다
    And   각 epoch의 InfoNCE loss가 로그에 기록된다
    And   동일 seed 재실행 시 동일한 체크포인트가 생성된다
```

---

### Scenario 2.2 — Phase 2 지도 학습 성공 / Phase 2 Supervised Training Success

```gherkin
  Scenario: Phase 2 지도 학습이 Phase 0 backbone 위에서 정상 완료된다
  Scenario: Phase 2 supervised training completes successfully on Phase 0 backbone

    Given Phase 0 backbone 파일(phase0_backbone_Y_effb0.pt)이 존재한다
    And   config.json에 유효한 Phase 2 파라미터가 정의되어 있다

    When  운영자가 `python -m src.scripts.run_phase2 --channel Y`를 실행한다

    Then  학습이 완료되고 data_set/models/best_Y.pt 파일이 생성된다
    And   val_acc 기준 최고 성능 체크포인트만 저장된다
    And   각 epoch의 train_loss와 val_acc가 CSV로 저장된다
    And   Early Stopping이 patience 초과 시 자동으로 학습을 종료한다
```

---

### Scenario 2.3 — Backbone별 ClassifierHead 특화 구조 적용 / Backbone-Specific Head

```gherkin
  Scenario: EfficientNet-B0 backbone 사용 시 직접 압축 head가 적용된다
  Scenario: Direct-compression head is applied when using EfficientNet-B0 backbone

    Given config.json의 model.backbone이 "efficientnet_b0"로 설정되어 있다
    And   phase2.heads.efficientnet_b0에 mid_dim이 정의되어 있지 않다

    When  GrayspotModel이 phase=2로 초기화된다

    Then  ClassifierHead는 1280 → 256 → 6 구조(Linear 2개)로 생성된다
    And   mid_dim 중간 레이어가 존재하지 않는다

  Scenario: ResNet-50 backbone 사용 시 단계적 압축 head가 적용된다
  Scenario: Staged-compression head is applied when using ResNet-50 backbone

    Given config.json의 model.backbone이 "resnet50"으로 설정되어 있다
    And   phase2.heads.resnet50.mid_dim이 512로 설정되어 있다

    When  GrayspotModel이 phase=2로 초기화된다

    Then  ClassifierHead는 2048 → 512 → 256 → 6 구조(Linear 3개)로 생성된다
    And   mid_dim=512 중간 레이어가 존재한다
```

---

### Scenario 2.4 — Baseline 학습 (Phase 0 없이) / Baseline Training Without Phase 0

```gherkin
  Scenario: Baseline 모드에서 Phase 0 없이 Phase 2 지도 학습이 실행된다
  Scenario: Baseline mode runs Phase 2 supervised training without Phase 0

    Given pretrained EfficientNet-B0 weights가 사용 가능하다
    And   Phase 0 backbone 파일이 존재하지 않아도 된다

    When  운영자가 `python -m src.scripts.run_baseline --channel Y`를 실행한다

    Then  ImageNet pretrained weights에서 직접 Phase 2 학습이 시작된다
    And   data_set/baseline/best_Y.pt가 생성된다
    And   Swing Architecture와의 성능 비교에 기준선으로 사용 가능하다
```

---

### Scenario 2.5 — 재현성 보장 / Reproducibility Guarantee

```gherkin
  Scenario: 동일 seed 설정 시 학습 결과가 재현된다
  Scenario: Training result is reproduced with the same seed setting

    Given config.json의 train.seed가 42로 고정되어 있다
    And   동일한 데이터와 모델 구조가 사용된다

    When  동일한 학습 스크립트를 두 번 실행한다

    Then  두 실행의 val_acc가 동일하다
    And   두 체크포인트의 가중치가 동일하다
    And   데이터 분할(train/val/test)이 동일하다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 / Scenario | TDD 파일 / TDD File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| 2.1 — Phase 0 완료 / completion | `test_smoke_phase0.py` | `test_phase0_checkpoint_created` | Smoke |
| 2.2 — Phase 2 완료 / completion | `test_smoke_phase2.py` | `test_phase2_checkpoint_created` | Smoke |
| 2.3 — EffB0 head 구조 / structure | `test_models.py` | `test_effb0_direct_compression_mid_dim_none` | Unit |
| 2.3 — ResNet-50 head 구조 / structure | `test_models.py` | `test_resnet50_staged_compression_mid_dim_512` | Unit |
| 2.4 — Baseline 체크포인트 / checkpoint | `test_smoke_phase2.py` | `test_baseline_checkpoint_created` | Smoke |
| 2.5 — 재현성 / reproducibility | `test_utils_model.py` | `test_set_seed_reproducibility` | Unit |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [Contract_training_pipeline.md](../Contract/Contract_training_pipeline.md) | 학습 파이프라인 계약 / Training pipeline contract |
| [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) | Phase 0/2 파라미터 정의 / Phase 0/2 parameter definitions |
| [BDD_Safety.md](BDD_Safety.md) | Backbone 누락 시 Fail-Fast 시나리오 / Fail-Fast backbone missing scenario |
