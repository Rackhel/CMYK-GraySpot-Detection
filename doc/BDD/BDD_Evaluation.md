---
type: bdd
domain: evaluation
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Evaluation_Reporting.md"
  - "Contract_evaluation_reporting.md"
---

# [BDD] 평가 리포트 생성 / Evaluation Report Generation

> **역할 / Role**: 데이터 과학자와 품질 관리자가 평가 결과를 조회할 때의 관찰 가능한 행동을 정의한다.
> **Role**: Defines observable behavior when Data Scientists and QC Managers review evaluation results.

---

## 행위자 / Actors

| 행위자 / Actor | 역할 / Role |
|---|---|
| **데이터 과학자 / Data Scientist** | 모델 성능을 분석하고 리포트를 생성하는 연구자 / Researcher analyzing model performance and generating reports |
| **품질 관리자 / QC Manager** | 평가 대시보드를 검토하는 관리자 / Manager reviewing the evaluation dashboard |
| **운영자 / Operator** | 리포트 생성 스크립트를 실행하는 엔지니어 / Engineer running the report generation script |

---

## Feature: 평가 리포트 생성 / Evaluation Report Generation

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

### Scenario 4.6 — Baseline HTML 리포트 생성 / Baseline HTML Report Generation

```gherkin
  Scenario: generate_baseline_report()가 독립 실행형 HTML 리포트를 생성한다
  Scenario: generate_baseline_report() generates a standalone HTML report

    Given EvaluationSummary와 채널별 추론 결과(results)가 준비되어 있다

    When  generate_baseline_report(summary, results)가 호출된다

    Then  outputs/reports/baseline.html이 생성된다
    And   생성된 HTML은 Plotly CDN이 내장된 독립 실행형 파일이다
    And   파일을 브라우저에서 열 때 외부 자산 로드 없이 동작한다
    And   summary.overall, summary.by_channel 지표가 차트로 표현된다
```

---

### Scenario 4.7 — evaluate.py CLI 스크립트 실행 / evaluate.py CLI Script Execution

```gherkin
  Scenario: evaluate.py가 채널별 평가 결과와 리포트를 생성한다
  Scenario: evaluate.py generates per-channel evaluation results and reports

    Given best_Y.pt 모델과 data_set/labeled/Y/ 데이터가 존재한다
    And   config.json이 유효하다

    When  운영자가 `python -m src.scripts.evaluate --channel Y`를 실행한다

    Then  채널 Y에 대한 평가가 수행된다
    And   metrics_summary_eval.json이 생성된다
    And   eval_dashboard.html이 생성된다
    And   종료 코드 0으로 완료된다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 / Scenario | TDD 파일 / TDD File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| 4.1 — HTML 대시보드 / dashboard | `test_reporting.py` | `test_eval_dashboard_html_created` | Unit |
| 4.2 — 혼동 행렬 / confusion matrix | `test_confusion.py` | `test_confusion_html_per_channel` | Unit |
| 4.3 — 오분류 CSV / misclassified CSV | `test_metrics.py` | `test_get_misclassified_structure` | Unit |
| 4.4 — 지표 JSON / metrics JSON | `test_metrics.py` | `test_check_targets_all_pass_structure` | Unit |
| 4.5 — 신뢰도 분포 / confidence dist | `test_reporting.py` | `test_confidence_distribution_html_created` | Unit |
| 4.6 — Baseline HTML | `test_reporting.py` | `test_generate_baseline_report_creates_html` | Unit |
| 4.7 — evaluate.py CLI | `test_evaluate_script.py` | `test_main_runs_without_error` | Integration |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [Contract_evaluation_reporting.md](../Contract/Contract_evaluation_reporting.md) | 평가 모듈 경계 계약 / Evaluation module boundary contract |
| [SSOT_Evaluation_Reporting.md](../SSOT/SSOT_Evaluation_Reporting.md) | 성능 목표값, 임계값 정의 / Performance targets and threshold definitions |
| [TDD_Evaluate_Script.md](../TDD/TDD_Evaluate_Script.md) | evaluate.py TDD 명세 / TDD specification |
