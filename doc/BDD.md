---
title: BDD 명세 / Behavior-Driven Development Specification
version: 1.0.0
last_updated: 2026-05-12
scope: CMYK Grayspot Detection System — 전체 동작 명세 / Full behavioral specification
---

# BDD 명세 / Behavior-Driven Development Specification

> **설계 명세 계층 / Design Specification Hierarchy**
>
> ```
> PRD (What — 비즈니스 요구사항)
>  └─ SSOT / Contract (Semantic — 기술 사양 정의)
>      └─ BDD (Behavior — 사용자 관점 행동 시나리오)  ← 이 문서
>          └─ TDD (Unit — 함수/클래스 단위 기술 검증)
>              └─ Code (Implementation)
> ```
>
> **역할 / Role**: 시스템이 각 행위자(Actor) 관점에서 어떻게 동작해야 하는지를 Given-When-Then 시나리오로 정의한다. 구현 세부사항이 아닌 관찰 가능한 행동(observable behavior)을 기술한다.
>
> **Role**: Defines how the system must behave from each Actor's perspective using Given-When-Then scenarios. Describes observable behaviors, not implementation details.

---

## 목차 / Table of Contents

1. [행위자 정의 / Actor Definitions](#1-행위자-정의--actor-definitions)
2. [Feature 1 — 그레이스팟 결함 분류 / Grayspot Defect Classification](#2-feature-1--그레이스팟-결함-분류--grayspot-defect-classification)
3. [Feature 2 — 학습 파이프라인 실행 / Training Pipeline Execution](#3-feature-2--학습-파이프라인-실행--training-pipeline-execution)
4. [Feature 3 — Swing Feedback 루프 / Swing Feedback Loop](#4-feature-3--swing-feedback-루프--swing-feedback-loop)
5. [Feature 4 — 평가 리포트 생성 / Evaluation Report Generation](#5-feature-4--평가-리포트-생성--evaluation-report-generation)
6. [Feature 5 — Optuna 하이퍼파라미터 튜닝 / Optuna Hyperparameter Tuning](#6-feature-5--optuna-하이퍼파라미터-튜닝--optuna-hyperparameter-tuning)
7. [Feature 6 — CMYK 채널 독립 관리 / CMYK Channel Independence](#7-feature-6--cmyk-채널-독립-관리--cmyk-channel-independence)
8. [Feature 7 — 시스템 안전성 / System Safety (Fail-Fast)](#8-feature-7--시스템-안전성--system-safety-fail-fast)
9. [시나리오-테스트 추적 매트릭스 / Scenario-Test Traceability Matrix](#9-시나리오-테스트-추적-매트릭스--scenario-test-traceability-matrix)
10. [관련 문서 / Related Documents](#10-관련-문서--related-documents)

---

## 1. 행위자 정의 / Actor Definitions

| 행위자 / Actor | 역할 / Role | 시스템과의 상호작용 / Interaction with System |
|---|---|---|
| **운영자 / Operator** | 학습 파이프라인을 실행하고 모델을 배포하는 엔지니어 / Engineer who runs training and deploys models | Phase 0 · Phase 2 학습 스크립트 실행, 결과 모니터링 |
| **품질 검사원 / QC Inspector** | 인쇄물 결함 수준을 판정하는 현장 검사자 / Field inspector judging print defect levels | 추론 결과(Level 예측 + 신뢰도) 조회 |
| **품질 관리자 / QC Manager** | 검사 결과를 기반으로 재학습/통과 여부를 결정하는 관리자 / Manager deciding retrain/pass based on evaluation | 평가 대시보드 검토, Swing Feedback 결정 승인 |
| **데이터 과학자 / Data Scientist** | 모델 성능을 분석하고 하이퍼파라미터를 최적화하는 연구자 / Researcher analyzing model performance and tuning hyperparameters | Optuna 튜닝 실행, 지표 분석, 혼동 행렬 해석 |
| **시스템 / System** | CMYK Grayspot Detection Pipeline 자체 | Fail-Fast 검증, 자동 피드백 판단 |

---

## 2. Feature 1 — 그레이스팟 결함 분류 / Grayspot Defect Classification

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
    And   MAE(평균 절대 오차)는 1.0 이하이다
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
    And   출력값의 합이 1.0이 아닐 수 있다 (Softmax 미적용)
    And   CrossEntropyLoss가 올바르게 적용 가능한 형태이다
```

---

## 3. Feature 2 — 학습 파이프라인 실행 / Training Pipeline Execution

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

## 4. Feature 3 — Swing Feedback 루프 / Swing Feedback Loop

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

    Then  피드백 결과는 "pass"이다
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

    Then  피드백 결과는 "retry_phase0"이다
    And   "[Y] Accuracy < 0.80 → Phase 0 재학습" 메시지가 출력된다
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

    Then  피드백 결과는 "retry_phase2"이다
    And   "Level 3 F1 < 0.70 → Phase 2 재학습" 메시지가 출력된다
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

## 5. Feature 4 — 평가 리포트 생성 / Evaluation Report Generation

> **비즈니스 가치 / Business Value**: 데이터 과학자와 품질 관리자가 단일 HTML 대시보드에서 모델 성능의 전체 그림을 즉시 파악할 수 있다.
>
> **Business Value**: Data Scientists and QC Managers can immediately grasp the full picture of model performance from a single HTML dashboard.

---

### Scenario 4.1 — HTML 대시보드 생성 / HTML Dashboard Generation

```gherkin
Feature: 평가 리포트 생성 / Evaluation report generation

  Scenario: 평가 완료 후 HTML 대시보드가 생성된다
  Scenario: HTML dashboard is generated after evaluation

    Given 전채널 평가 결과(y_true, y_pred, confidences)가 준비되어 있다
    And   지표 계산(compute_all_channels)이 완료되어 있다

    When  운영자가 save_report()를 호출한다

    Then  outputs/reports/eval_dashboard.html이 생성된다
    And   대시보드에는 Overall Accuracy, Macro F1, MAE Gauge가 포함된다
    And   색상별(Y, M, C, K) Accuracy 막대 차트가 포함된다
    And   목표값 기준선(target threshold)이 시각적으로 표시된다
```

---

### Scenario 4.2 — 채널별 혼동 행렬 저장 / Per-Channel Confusion Matrix

```gherkin
  Scenario: 채널별 혼동 행렬 HTML이 저장된다
  Scenario: Per-channel confusion matrix HTML files are saved

    Given Y, M, C, K 채널 추론 결과가 준비되어 있다

    When  save_report()가 실행된다

    Then  outputs/reports/confusion/cm_Y.html 등 채널별 파일이 생성된다
    And   cm_overall.html도 함께 생성된다
    And   각 혼동 행렬은 정규화(normalize=True)된 값으로 표시된다
```

---

### Scenario 4.3 — 오분류 샘플 CSV 추출 / Misclassified Samples CSV Export

```gherkin
  Scenario: 오분류 샘플이 CSV로 저장된다
  Scenario: Misclassified samples are saved as CSV

    Given 평가 결과에 오분류 샘플이 존재한다

    When  save_report()가 실행된다

    Then  outputs/reports/misclassified_eval.csv가 생성된다
    And   각 행에 filename, color, true_level, pred_level, confidence, error_gap이 포함된다
    And   error_gap 내림차순, confidence 오름차순으로 정렬된다
    And   UTF-8 BOM으로 인코딩되어 Windows Excel에서 한글이 깨지지 않는다
```

---

### Scenario 4.4 — 지표 JSON 저장 / Metrics JSON Export

```gherkin
  Scenario: 집계 지표가 JSON으로 저장된다
  Scenario: Aggregated metrics are saved as JSON

    Given 전채널 지표 계산이 완료되어 있다

    When  save_json()이 호출된다

    Then  outputs/reports/metrics_summary_eval.json이 생성된다
    And   global, by_color, per_class_overall 섹션이 포함된다
    And   각 지표에 acc_pass, f1_pass, mae_pass, all_pass 판정이 포함된다
    And   목표값(targets) 섹션이 config에서 주입된 값으로 기록된다
```

---

### Scenario 4.5 — 신뢰도 분포 시각화 / Confidence Distribution Visualization

```gherkin
  Scenario: 채널별 신뢰도 분포 히스토그램이 생성된다
  Scenario: Per-channel confidence distribution histogram is generated

    Given Y, M, C, K 채널의 신뢰도(confidences) 배열이 준비되어 있다

    When  save_report()가 실행된다

    Then  outputs/reports/confidence_distribution.html이 생성된다
    And   정답/오답 신뢰도 분포가 서로 다른 색상으로 구분된다
    And   auto_accept(0.80), warn(0.50), manual_review(0.30) 임계값이 수직선으로 표시된다
```

---

## 6. Feature 5 — Optuna 하이퍼파라미터 튜닝 / Optuna Hyperparameter Tuning

> **비즈니스 가치 / Business Value**: 데이터 과학자가 수동으로 파라미터를 탐색하지 않고, Optuna가 자동으로 최적 하이퍼파라미터를 탐색하여 모델 성능을 향상시킨다.
>
> **Business Value**: Instead of manual parameter search by Data Scientists, Optuna automatically finds optimal hyperparameters to improve model performance.

---

### Scenario 6.1 — 단일 채널 Optuna 튜닝 / Single-Channel Optuna Tuning

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

### Scenario 6.2 — Backbone별 독립 탐색 공간 적용 / Per-Backbone Search Space

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

### Scenario 6.3 — MedianPruner 조기 종료 / MedianPruner Early Stopping

```gherkin
  Scenario: 성능 하위 trial이 조기 종료된다
  Scenario: Under-performing trials are pruned early

    Given optuna.pruner.n_warmup_steps가 10으로 설정되어 있다

    When  10 epoch 이후 trial이 중앙값보다 낮은 val_acc를 기록한다

    Then  해당 trial이 자동으로 중단된다
    And   나머지 trial 리소스가 더 유망한 파라미터 탐색에 사용된다
```

---

## 7. Feature 6 — CMYK 채널 독립 관리 / CMYK Channel Independence

> **비즈니스 가치 / Business Value**: CMYK 4색 분판 각각을 독립 모델로 관리함으로써 색상별 결함 패턴의 차이를 정밀하게 탐지한다.
>
> **Business Value**: Managing each of the 4 CMYK color separations as an independent model enables precise detection of channel-specific defect patterns.

---

### Scenario 7.1 — 채널 불변식 준수 / Channel Invariant Enforcement

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

### Scenario 7.2 — 4채널 독립 병렬 처리 / 4-Channel Independent Processing

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

### Scenario 7.3 — BGR 색상 공간 일관성 / BGR Color Space Consistency

```gherkin
  Scenario: 학습부터 평가까지 BGR 색상 공간이 유지된다
  Scenario: BGR color space is maintained throughout training and evaluation

    Given 이미지가 cv2.imread()로 로드된다 (BGR 형식)

    When  이미지가 전처리 파이프라인을 통과한다
    And   모델에 입력된다

    Then  RGB 변환이 적용되지 않는다
    And   학습과 추론 모두 동일한 BGR 통계(ImageNet 정규화)가 적용된다
    And   SSOT-CS01 위반이 발생하지 않는다
```

---

## 8. Feature 7 — 시스템 안전성 / System Safety (Fail-Fast)

> **비즈니스 가치 / Business Value**: 잘못된 아티팩트나 설정이 사용될 경우 즉시 오류를 발생시켜 오염된 결과물이 생성되는 것을 방지한다.
>
> **Business Value**: When incorrect artifacts or configurations are used, the system fails immediately to prevent contaminated results from being produced.

---

### Scenario 8.1 — Phase 0 Backbone 누락 시 즉시 실패 / Immediate Failure on Missing Phase 0 Backbone

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

### Scenario 8.2 — 필수 Config 키 누락 시 즉시 실패 / Immediate Failure on Missing Config Key

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

### Scenario 8.3 — 정규화 적용 검증 / Normalization Application Verification

> ✅ **SSOT-NM01 해소됨 / Resolved**: `predictor_inference.py` `_preprocess_images()` 에서 `_IMAGENET_NORMALIZE` 적용 완료 (2026-05-14).
> 시나리오 조건이 "미적용 감지"에서 "정규화 적용 보장"으로 전환됨.

```gherkin
  Scenario: GrayspotPredictor 추론 시 ImageNet 정규화가 학습과 동일하게 적용된다
  Scenario: GrayspotPredictor applies ImageNet normalization identical to training

    Given GrayspotPredictor 가 초기화되어 있다

    When  predict() 가 BGR numpy 이미지 배열을 입력받는다

    Then  내부적으로 [0,1] 정규화 후 ImageNet mean/std 변환이 적용된다
    And   dataset.py 의 _IMAGENET_NORMALIZE 와 동일한 변환이 보장된다
    And   학습(dataset.py) 과 추론(predictor_inference.py) 간 정규화 불일치가 없다
```

---

### Scenario 8.4 — Dead Config 감지 / Dead Config Detection

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

## 9. 시나리오-테스트 추적 매트릭스 / Scenario-Test Traceability Matrix

BDD 시나리오와 TDD 테스트 파일 / BDD scenarios mapped to TDD test files:

| BDD 시나리오 / Scenario | TDD 테스트 파일 / Test File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| 1.1 — AUTO_ACCEPT 플래그 / flag | `test_predictor.py` | `test_confidence_auto_accept_flag` | Unit |
| 1.2 — WARN 플래그 / flag | `test_predictor.py` | `test_confidence_warn_flag` | Unit |
| 1.3 — MANUAL_REVIEW 플래그 / flag | `test_predictor.py` | `test_confidence_manual_review_flag` | Unit |
| 8.3 — ImageNet 정규화 적용 / Applied | `test_predictor.py` | `test_preprocess_images_imagenet_normalized` | Unit |
| 1.5 — Raw logit 출력 / output | `test_models.py` | `test_output_is_logits_not_probability` | Unit |
| 2.3 — EfficientNet-B0 head 구조 / structure | `test_models.py` | `test_effb0_direct_compression_mid_dim_none` | Unit |
| 2.3 — ResNet-50 head 구조 / structure | `test_models.py` | `test_resnet50_staged_compression_mid_dim_512` | Unit |
| 2.3 — EffB0 Linear 2개 / 2 Linear layers | `test_models.py` | `test_effb0_head_layer_count_is_2_linear` | Unit |
| 2.3 — ResNet-50 Linear 3개 / 3 Linear layers | `test_models.py` | `test_resnet50_head_layer_count_is_3_linear` | Unit |
| 2.3 — Phase2 EffB0 출력 shape / output shape | `test_models.py` | `test_phase2_effb0_output_shape` | Unit (slow) |
| 2.3 — Phase2 ResNet-50 출력 shape / output shape | `test_models.py` | `test_phase2_resnet50_output_shape` | Unit (slow) |
| 2.5 — 재현성 보장 / Reproducibility guarantee | `test_utils_model.py` | `test_set_seed_reproducibility` | Unit |
| 3.1 — PASS 피드백 / feedback | `test_metrics.py` | `test_determine_swing_feedback_pass` | Unit |
| 3.2 — retry_phase0 피드백 / feedback | `test_metrics.py` | `test_determine_swing_feedback_retry_phase0` | Unit |
| 3.3 — retry_phase2 피드백 / feedback | `test_metrics.py` | `test_determine_swing_feedback_retry_phase2` | Unit |
| 4.3 — 오분류 CSV / Misclassification CSV | `test_metrics.py` | `test_get_misclassified_structure` | Unit |
| 4.4 — 지표 JSON 구조 / Metrics JSON structure | `test_metrics.py` | `test_check_targets_all_pass_structure` | Unit |
| 5.1 — val_acc 목적 함수 반환 / objective return | `test_smoke_optuna.py` | `test_optuna_objective_returns_float` | Smoke |
| 7.3 — BGR 일관성 / consistency | `test_preprocessing.py` | `test_no_rgb_conversion` | Unit |
| 8.1 — FF01 backbone 누락 / missing | `test_smoke_phase2.py` | `test_phase2_fails_without_phase0_backbone` | Smoke |
| 8.2 — CF01 config 키 누락 / missing key | `test_utils_config.py` | `test_validate_config_missing_key_raises` | Unit |

> **미구현 테스트 / Not Yet Implemented**: 시나리오 1.1–1.3, 8.3 (`test_predictor.py`), 4.1–4.2 (HTML 생성 단위 테스트), 7.1–7.2 (채널 불변식 통합 테스트)
>
> **Not Yet Implemented**: Scenarios 1.1–1.3, 8.3 (`test_predictor.py`), 4.1–4.2 (HTML generation unit tests), 7.1–7.2 (channel invariant integration tests)

---

## 10. 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [SSOT_Core.md](SSOT_Core.md) | SOLID 원칙, SSOT 정의, Fail-Fast 정책 — BDD 시나리오 7, 8의 근거 / SOLID principles, SSOT definition, Fail-Fast policy — basis for scenarios 7 & 8 |
| [Contract.md](Contract.md) | 모듈 경계 계약 — BDD 시나리오 입출력 spec의 기술적 근거 / Module boundary contracts — technical basis for scenario I/O specs |
| [SSOT_Evaluation_Reporting.md](SSOT_Evaluation_Reporting.md) | 성능 목표값, 신뢰도 임계값 — 시나리오 1, 3, 4의 수치적 근거 / Performance targets, confidence thresholds — numeric basis for scenarios 1, 3, 4 |
| [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md) | Phase 0/2 학습 파라미터 — 시나리오 2의 기술적 근거 / Phase 0/2 training parameters — technical basis for scenario 2 |
| [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md) | BGR 규약, 채널 독립성 — 시나리오 6, 7의 근거 / BGR convention, channel independence — basis for scenarios 6 & 7 |
| [ADR_Model_Select.md](ADR_Model_Select.md) | EfficientNet-B0/ResNet-50 설계 결정 — 시나리오 2.3의 근거 / EfficientNet-B0/ResNet-50 design decision — basis for scenario 2.3 |
| [TDD.md](TDD.md) | 시나리오별 단위 테스트 전략 — §9 매트릭스의 구현 세부 / Per-scenario unit test strategy — implementation detail of the §9 matrix |

---
