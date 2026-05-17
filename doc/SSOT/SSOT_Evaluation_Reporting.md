---
type: ssot
domain: evaluation_reporting
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "SSOT_Training_Pipeline.md"
  - "SSOT_Artifacts.md"
---

# SSOT Evaluation & Reporting — 평가 및 보고 / Evaluation and Reporting

CMYK Grayspot Detection System 의 평가 지표, 목표값, 신뢰도 임계값, 리포트 생성에 관한 단일 진실 공급원.

This document is the authoritative reference for evaluation metrics, targets, confidence thresholds, and report generation.

> **목적 / Purpose**: 평가 기준과 보고 형식의 의미(semantic) 정의
> **역할 / Role**: "What" — 지표 정의, 합격 기준, 신뢰도 정책
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Artifacts.md](SSOT_Artifacts.md)

---

## 1. Evaluator 개요 / Evaluator Overview

`Evaluator`는 4개의 Mixin을 조합한 **조율자(Orchestrator)** 클래스로, SRP/ISP 원칙에 따라 분리된 내부 모듈을 단일 공개 인터페이스로 노출한다.

`Evaluator` is an **Orchestrator** composing 4 Mixins. Its internal modules are separated per SRP/ISP and exposed via a single public interface.

```
evaluation/
├── evaluator_inference.py  — InferenceMixin  (추론 루프 / Inference loop)
├── evaluator_metrics.py    — MetricsMixin    (지표·오분류 / Metrics & misclassification)
├── evaluator_export.py     — ExportMixin     (CSV/JSON 내보내기 / CSV/JSON export)
├── evaluator_charts.py     — ChartsMixin     (7개 Plotly 차트 / 7 Plotly charts)
└── evaluator.py            — Evaluator       (__init__ + save_report 조율자 / Orchestrator)
```

### 1.1 공개 API / Public API

| 함수 / Function | 모듈 / Module | Mixin / 역할 | 설명 / Description |
|---|---|---|---|
| `Evaluator.__init__` | `evaluation.evaluator` | Orchestrator | 상태 초기화 / State initialisation |
| `Evaluator.run` | `evaluation.evaluator_inference` | `InferenceMixin` | 채널별 추론 실행 / Per-channel inference |
| `Evaluator.load_labels` | `evaluation.evaluator_inference` | `InferenceMixin` | CSV → DataFrame 레이블 로딩 / Labels CSV loading |
| `Evaluator.compute` | `evaluation.evaluator_metrics` | `MetricsMixin` | 전 채널 지표 계산 / All-channel metrics computation |
| `Evaluator.get_misclassified` | `evaluation.evaluator_metrics` | `MetricsMixin` | 오분류 DataFrame 추출 / Misclassified samples extraction |
| `Evaluator.save_csv` | `evaluation.evaluator_export` | `ExportMixin` | 예측 결과 CSV 저장 / Prediction results CSV export |
| `Evaluator.save_json` | `evaluation.evaluator_export` | `ExportMixin` | 지표 요약 JSON 저장 / Metrics summary JSON export |
| `Evaluator.save_report` | `evaluation.evaluator` | Orchestrator | HTML + CSV + JSON 전체 리포트 생성 / Full HTML+CSV+JSON report |
| `compute_metrics` | `evaluation.metrics` | — | 단일 채널 지표 계산 / Single-channel metrics computation |
| `compute_all_channels` | `evaluation.metrics` | — | 전 채널 지표 일괄 계산 / Batch metrics for all channels |
| `EvaluationSummary` | `evaluation.metrics` | — | 지표 결과 데이터클래스 / Metrics result dataclass |
| `determine_swing_feedback` | `evaluation.metrics` | — | Swing 재실행 여부 결정 / Determine whether to retry Swing |
| `plot_confusion_matrix` | `evaluation.confusion` | — | 채널별 혼동 행렬 시각화 / Per-channel confusion matrix |
| `plot_all_channels` | `evaluation.confusion` | — | 전 채널 혼동 행렬 일괄 생성 / Batch confusion matrices |

---

## 2. 성능 목표 / Performance Targets

PRD Section 1.4 기준 / Based on PRD Section 1.4 targets

| 지표 / Metric | 목표값 / Target | config 키 / Key | 판정 / Pass Condition |
|---|---|---|---|
| Overall Accuracy / 전체 정확도 | ≥ 90% | `evaluation.targets.overall_accuracy` 🟢 | `accuracy >= 0.90` |
| Per-channel Accuracy / 채널별 정확도 | ≥ 85% | `evaluation.targets.per_color_accuracy` 🟢 | `accuracy >= 0.85` per channel |
| Per-class F1 / 클래스별 F1 | ≥ 0.80 | `evaluation.targets.per_class_f1` 🟢 | `f1 >= 0.80` per class |
| MAE | ≤ 0.50 | `evaluation.targets.mae` 🟢 | `mae <= 0.50` |

### 2.1 상수 정의 / Constants

`evaluation/metrics.py`에서 정의됨 / Defined in `evaluation/metrics.py`:

| 상수 / Constant | 값 / Value | config 키 / Key |
|---|---|---|
| `NUM_LEVELS` | `6` | Hard SSOT — `data.num_levels`와 동기화 필수 |
| `TARGET_OVERALL_ACC` | `0.90` | `evaluation.targets.overall_accuracy` |
| `TARGET_PER_COLOR_ACC` | `0.85` | `evaluation.targets.per_color_accuracy` |
| `TARGET_PER_CLASS_F1` | `0.80` | `evaluation.targets.per_class_f1` |
| `TARGET_MAE` | `0.50` | `evaluation.targets.mae` |

### 2.2 Swing 재시도 임계값 / Swing Retry Thresholds

`EvaluationSummary.targets`에 자동 주입 (`evaluation.swing_thresholds.*`) / Auto-injected into `EvaluationSummary.targets`:

| 키 / Key | 값 / Value | config 키 / Key |
|---|---|---|
| `"swing_acc_retry"` | `0.80` | `evaluation.swing_thresholds.acc_retry` 🟢 |
| `"swing_f1_retry"` | `0.70` | `evaluation.swing_thresholds.f1_retry` 🟢 |
| `"swing_mae_retry"` | `0.80` | `evaluation.swing_thresholds.mae_retry` 🟢 |

---

## 3. 신뢰도 임계값 / Confidence Thresholds

| 임계값 / Threshold | 값 / Value | 처리 / Action | 비고 / Note |
|---|---|---|---|
| `CONF_THRESH_AUTO` | 0.8 | 자동 수락 / Auto accept | `inference.confidence_thresholds.auto_accept` 🟢 |
| `CONF_THRESH_WARN` | 0.5 | 경고 포함 수락 / Accept with warning (warn flag) | `inference.confidence_thresholds.warn_threshold` 🟢 |
| `CONF_THRESH_MANUAL` | 0.3 | 수동 검토 대기 / Queue for manual review | `inference.confidence_thresholds.manual_review` 🟢 |

> `CONF_THRESH_*` 상수는 `evaluation/metrics.py`에 backward-compatibility용으로 유지되나,
> `Evaluator` 내부에서는 `self.conf_thresh_*` 인스턴스 속성(cfg 기반)을 사용한다.
>
> `CONF_THRESH_*` constants remain in `evaluation/metrics.py` for backward compatibility,
> but `Evaluator` internally uses `self.conf_thresh_*` instance attributes (cfg-based).

---

## 4. 평가 지표 정의 / Metric Definitions

### 4.1 Overall Accuracy / 전체 정확도

전체 샘플 기준 정답 비율 (모든 레벨 포함) / Fraction of correct predictions across all samples and all levels.

### 4.2 Per-class F1

클래스별 F1의 비가중 평균 (`macro`). / Unweighted average of per-class F1 scores (`macro` average).

| 지표 / Metric | 설명 / Description |
|---|---|
| `per_class_f1` | 클래스별 F1 배열 / Per-class F1 array |
| `macro_f1` | 클래스별 F1의 비가중 평균 / Unweighted mean of per-class F1 |

### 4.3 MAE (Mean Absolute Error)

레벨 간 절대 오차의 평균. ordinal 속성 활용 — 레벨 간 거리를 숫자로 측정.
Mean absolute difference between predicted and true levels. Leverages ordinal nature of the 6-class label space.

### 4.4 Per-class Metrics / 클래스별 지표

`compute_metrics(y_true, y_pred, confidences)` → `EvaluationSummary`

- `EvaluationSummary.per_class`: `List[PerClassMetric]`
- `PerClassMetric` 필드: `precision`, `recall`, `f1`, `support` (레벨별 / per level)

---

## 5. 평가 흐름 / Evaluation Flow

```
best_{channel}.pt
    ↓ (model.eval())
Evaluator.run(channel)
    ↓
y_true, y_pred, confidences
    ↓
compute_metrics(y_true, y_pred, confidences)
    ↓
EvaluationSummary {accuracy, macro_f1, mae, per_class, targets}
    ↓
determine_swing_feedback(summary)     ← 목표 달성 여부 판단 (swing_thresholds 기반) / Determine if targets met (based on swing_thresholds)
    ↓
save_report(results, name)
    ↓
evaluation_results_{name}.csv + metrics_summary_{name}.json
```

### 5.1 Evaluator 입력 / Input Contract

| 입력 / Input | 타입 / Type | 조건 / Condition |
|---|---|---|
| 모델 / Model | `nn.Module` | `model.eval()` 상태 / in eval mode |
| 이미지 / Images | `Tensor (B, 3, 128, 128)` | BGR float32 [0, 1] |
| 레이블 / Labels | `Tensor (B,)` | int [0, 5] |

> **⚠️ SSOT-CS01**: `Evaluator`에 전달되는 이미지는 학습 시와 동일한 BGR 색상 공간이어야 한다.
> Images passed to `Evaluator` must use the same BGR color space as training.

### 5.2 평가 배치 크기 / Evaluation Batch Size

| config 키 / Key | 값 / Value | 소비자 / Consumer |
|---|---|---|
| `inference.batch_size` 🟢 | `32` | `GrayspotPredictor` |

---

## 6. Swing Feedback 정책 / Swing Feedback Policy

`determine_swing_feedback(summary: EvaluationSummary, channels: List[str] = None) → dict`

반환값 / Returns:

| 키 / Key | 타입 / Type | 설명 / Description |
|---|---|---|
| `terminate` | `bool` | 모든 목표 달성 시 `True` / `True` when all targets are met |
| `decisions` | `List[str]` | 조치 필요 항목 목록 / List of required actions |

**결과 판단 로직 / Decision Logic**:

| 조건 / Condition | decisions 항목 / Entry | 다음 단계 / Next Step |
|---|---|---|
| `decisions` 비어 있음 / empty | — | `terminate=True` → 시스템 종료 / System exit |
| 채널별 Acc < `swing_acc_retry` (0.80) | `"[{ch}] Accuracy … -> Phase 0 (retrain representation)"` | Phase 0 재학습 → Phase 2 / Retrain Phase 0 → Phase 2 |
| per-class F1 < `swing_f1_retry` (0.70) | `"Level {n} F1 … -> Phase 2 (retrain classifier)"` | Phase 2 재학습 / Retrain Phase 2 |
| Overall MAE > `swing_mae_retry` (0.80) | `"MAE … -> Phase 0 (retrain representation)"` | Phase 0 재학습 → Phase 2 / Retrain Phase 0 → Phase 2 |

> Swing 임계값은 `evaluation.swing_thresholds.*` config에서 주입 — `EvaluationSummary.targets["swing_*_retry"]` 로 전달.
> Swing thresholds are injected from `evaluation.swing_thresholds.*` config into `EvaluationSummary.targets`.

---

## 7. 보고서 산출물 / Report Artifacts

```
outputs/reports/
├── eval_dashboard.html               ← Gauge + Bar 대시보드 / Gauge + Bar dashboard
├── per_class_metrics.html            ← 클래스별 F1 차트 / Per-class F1 chart
├── mae_heatmap.html                  ← MAE 히트맵 / MAE heatmap
├── misclassified_scatter.html        ← 오분류 Scatter / Misclassification scatter plot
├── confidence_distribution.html      ← 신뢰도 분포 / Confidence distribution
├── confusion/
│   ├── cm_{channel}.html             ← 채널별 혼동 행렬 / Per-channel confusion matrix
│   └── cm_overall.html               ← 전체 혼동 행렬 / Overall confusion matrix
├── evaluation_results_{name}.csv     ← 샘플별 예측 결과 / Per-sample prediction results
├── misclassified_{name}.csv          ← 오분류 목록 / Misclassification list
└── metrics_summary_{name}.json       ← 집계 지표 / Aggregated metrics
```

### 7.1 metrics_summary JSON 스키마 / Schema

| 필드 / Field | 타입 / Type | 설명 / Description |
|---|---|---|
| `channel` | `str` | CMYK 채널 식별자 / Channel identifier |
| `accuracy` | `float` | 전체 정확도 / Overall accuracy |
| `macro_f1` | `float` | Macro F1 점수 / Macro F1 score |
| `mae` | `float` | Mean Absolute Error |
| `per_class` | `dict[str, dict]` | 레벨별 precision/recall/f1/support / Per-level metrics |
| `confusion_matrix` | `list[list[int]]` | 6×6 혼동 행렬 / 6×6 confusion matrix |
| `timestamp` | `str` | ISO 8601 생성 시각 / Generation timestamp |

### 7.2 HTML 리포트 탭 구조 / HTML Report Tab Structure

`reporting/html_report.py` 가 생성하는 self-contained HTML 의 탭 구성.
Tab layout of the self-contained HTML generated by `reporting/html_report.py`.

| 탭 / Tab | 내용 / Content |
|---|---|
| Summary / 요약 | KPI 카드 (Acc, Macro F1, MAE) + 목표 달성 여부 / KPI cards + pass/fail against targets |
| Per-Class / 클래스별 | 레벨별 Precision / Recall / F1 테이블 + radar chart / Per-level P/R/F1 table + radar chart |
| Confusion / 혼동 행렬 | 6×6 히트맵 (Plotly) / 6×6 heatmap |
| MAE | 레벨별 MAE 분포 / MAE distribution per level |
| Confidence / 신뢰도 | 예측 confidence histogram / Prediction confidence histogram |
| Feedback | PRD §3.3.2 기반 개선 방향 / Improvement directions based on PRD §3.3.2 |

> 의존성: Plotly CDN + inline CSS. Jinja2 미사용, 순수 string concat 방식.
> Dependencies: Plotly CDN + inline CSS. No Jinja2 — pure string concatenation.

### 7.3 보고서 생성 / Report Generation

`reporting.html_report.generate_baseline_report` 를 통해 단일 self-contained HTML 파일로 생성.
Generates a single self-contained HTML file via `reporting.html_report.generate_baseline_report`.

| 파라미터 / Parameter | 타입 / Type | 기본값 / Default | 설명 / Description |
|---|---|---|---|
| `summary` | `EvaluationSummary` | — | `GrayspotEvaluator.run()` 결과 / Result from evaluator run |
| `results` | `dict[str, dict]` | — | 채널별 추론 결과 / Per-channel inference results |
| `output_path` | `str \| Path` | `outputs/reports/baseline.html` | 출력 HTML 경로 / Output HTML path |
| `channels` | `list[str]` | `["Y","M","C","K"]` | 포함할 채널 목록 / Channels to include |
| `open_browser` | `bool` | `False` | 완료 후 브라우저 자동 열기 / Auto-open browser on completion |
| `logger` | `logging.Logger \| None` | `None` | 로거 인스턴스 / Logger instance |

---

## 8. 학습 vs 평가 알려진 불일치 / Known Training–Evaluation Mismatches

학습(`Phase2Trainer`)과 평가(`Evaluator`) 간 처리 방식 비교.
Comparison of processing between training (`Phase2Trainer`) and evaluation (`Evaluator`).

| 항목 / Item | 학습 / Training | 평가 / Evaluation | 영향 / Impact |
|---|---|---|---|
| 색상 순서 / Color order | BGR (`cv2.imread`) | BGR (`cv2.imread`) | ✅ 동일 / Identical |
| ImageNet 정규화 / Norm | `_IMAGENET_NORMALIZE` 적용 / Applied | `_IMAGENET_NORMALIZE` 적용 / Applied | ✅ 동일 / Identical |
| 이미지 크기 / Image size | 128×128 | 128×128 | ✅ 동일 / Identical |
| Best 저장 기준 / Save criterion | `val_acc` 최대화 / Maximize `val_acc` | — | ⚠️ PRD 목표 `macro_f1`과 불일치 / Differs from PRD target `macro_f1` |
| Optuna 목적 함수 / Objective | `best_val_acc` | `best_val_acc` | ⚠️ PRD 목표 `macro_f1`과 불일치 / Differs from PRD target `macro_f1` |

---

## 9. Fail-Fast 규칙 / Fail-Fast Rules

평가 파이프라인에서 발동되는 SSOT 검증 코드. 자세한 정의는 [SSOT_Validation_Codes.md](SSOT_Validation_Codes.md) 참조.
SSOT validation codes triggered in the evaluation pipeline. See [SSOT_Validation_Codes.md](SSOT_Validation_Codes.md) for full definitions.

| 조건 / Condition | SSOT 코드 / Code | 등급 / Level | 동작 / Action |
|---|---|---|---|
| 평가 대상 모델 파일 누락 / Model file missing for evaluation | `SSOT-FF01` | Level 1 — Error | 평가 중단 / Abort evaluation |
| 학습/평가 색상 공간 불일치 / Color space mismatch | `SSOT-CS01` | Level 1 — Error | 결과 신뢰 불가 / Results unreliable |
| ImageNet 정규화 미적용 (pretrained) / Missing ImageNet norm | `SSOT-NM01` | Level 2 — Warning | 경고 출력 + 계속 / Warn and continue |

---

## 체크리스트 / Checklist

- [ ] 성능 목표 변경 시 §2 업데이트 / Update §2 on performance target change
- [ ] Swing Feedback 임계값 변경 시 §6 동기화 / Sync §6 on Swing Feedback threshold change
- [ ] 새 지표 추가 시 §4 목록 + §7.1 JSON 스키마 갱신 / Update §4 list + §7.1 JSON schema when adding new metric
- [ ] Best 저장 기준 → macro_f1 전환 시 Training Pipeline 동기화 / Sync Training Pipeline on best-save criterion change to macro_f1
- [ ] `_EvalDataset` ImageNet 정규화 적용 후 N-01 해소 / Resolve N-01 after applying ImageNet normalization in `_EvalDataset`

---

## See Also

| 문서 / Document | 관계 / Relation |
| --- | --- |
| [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md) | 학습 흐름, Swing Architecture / Training flow, Swing Architecture |
| [SSOT_Model_Architecture.md](SSOT_Model_Architecture.md) | 모델 구조 / Model architecture |
| [SSOT_Artifacts.md](SSOT_Artifacts.md) | 산출물 파일명 패턴 / Artifact filename patterns |
| [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md) | 평가 데이터 전처리 / Evaluation data preprocessing |

