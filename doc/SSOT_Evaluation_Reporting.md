# SSOT Evaluation & Reporting — 평가 및 보고 / Evaluation and Reporting

CMYK Grayspot Detection System 의 평가 지표, 목표값, 신뢰도 임계값, 리포트 생성에 관한 단일 진실 공급원.

This document is the authoritative reference for evaluation metrics, targets, confidence thresholds, and report generation.

> **목적 / Purpose**: 평가 기준과 보고 형식의 의미(semantic) 정의
> **역할 / Role**: "What" — 지표 정의, 합격 기준, 신뢰도 정책
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Artifacts.md](SSOT_Artifacts.md)

---

## Table of Contents / 목차

1. [Evaluator 개요 / Evaluator Overview](#1-evaluator-개요--evaluator-overview)
2. [성능 목표 / Performance Targets](#2-성능-목표--performance-targets)
3. [신뢰도 임계값 / Confidence Thresholds](#3-신뢰도-임계값--confidence-thresholds)
4. [평가 지표 정의 / Metric Definitions](#4-평가-지표-정의--metric-definitions)
5. [평가 흐름 / Evaluation Flow](#5-평가-흐름--evaluation-flow)
6. [Swing Feedback 정책 / Swing Feedback Policy](#6-swing-feedback-정책--swing-feedback-policy)
7. [보고서 산출물 / Report Artifacts](#7-보고서-산출물--report-artifacts)
8. [학습 vs 평가 알려진 불일치 / Known Mismatches](#8-학습-vs-평가-알려진-불일치--known-trainingevaluation-mismatches)
9. [Fail-Fast 규칙 / Fail-Fast Rules](#9-fail-fast-규칙--fail-fast-rules)
10. [SSOT 위반 현황 / Violations](#10-ssot-위반-현황--violations)

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

```python
from evaluation.evaluator import Evaluator
from pathlib import Path
import torch

ev = Evaluator(
    model       = model,           # nn.Module (model.eval() 상태)
    labeled_dir = Path("data_set/labeled"),
    labels_csv  = Path("data_set/labels_v0.csv"),
    output_dir  = Path("outputs/reports"),
    device      = torch.device("cpu"),
    cfg         = cfg,             # 신뢰도·swing 임계값 주입 / confidence & swing threshold injection
)

results = ev.run(channels=["Y", "M", "C", "K"])
# results: {"Y": {"y_true": ndarray, "y_pred": ndarray, "confidences": ndarray}, ...}

metrics = ev.compute(results)
ev.save_report(results, metrics, experiment_name="baseline")
# → outputs/reports/eval_dashboard.html
# → outputs/reports/evaluation_results_baseline.csv
# → outputs/reports/metrics_summary_baseline.json
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

```python
# evaluation/metrics.py
NUM_LEVELS          = 6      # Hard SSOT — data.num_levels와 동기화 필수 / must be in sync with data.num_levels
TARGET_OVERALL_ACC  = 0.90   # evaluation.targets.overall_accuracy
TARGET_PER_COLOR_ACC= 0.85   # evaluation.targets.per_color_accuracy
TARGET_PER_CLASS_F1 = 0.80   # evaluation.targets.per_class_f1
TARGET_MAE          = 0.50   # evaluation.targets.mae
```

### 2.2 Swing 재시도 임계값 / Swing Retry Thresholds

```python
# EvaluationSummary.targets 에 자동 주입 / Auto-injected into EvaluationSummary.targets from config
# evaluation.swing_thresholds.*
"swing_acc_retry" : 0.80   # evaluation.swing_thresholds.acc_retry 🟢
"swing_f1_retry"  : 0.70   # evaluation.swing_thresholds.f1_retry  🟢
"swing_mae_retry" : 0.80   # evaluation.swing_thresholds.mae_retry 🟢
```

---

## 3. 신뢰도 임계값 / Confidence Thresholds

| 임계값 / Threshold | 값 / Value | 처리 / Action | 비고 / Note |
|---|---|---|---|
| `CONF_THRESH_AUTO` | 0.8 | 자동 수락 / Auto accept | `inference.confidence_thresholds.auto_accept` 🟢 |
| `CONF_THRESH_WARN` | 0.5 | 경고 포함 수락 / Accept with warning (warn flag) | `inference.confidence_thresholds.warn_threshold` 🟢 |
| `CONF_THRESH_MANUAL` | 0.3 | 수동 검토 대기 / Queue for manual review | `inference.confidence_thresholds.manual_review` 🟢 |

```python
# evaluation/evaluator.py — Evaluator.__init__(model, labeled_dir, labels_csv, output_dir, device, ..., cfg=None)
ct = (cfg or {}).get("inference", {}).get("confidence_thresholds", {})
self.conf_thresh_auto   = float(ct.get("auto_accept",    0.8))   # 🟢 inference.confidence_thresholds.auto_accept
self.conf_thresh_warn   = float(ct.get("warn_threshold", 0.5))   # 🟢 inference.confidence_thresholds.warn_threshold
self.conf_thresh_manual = float(ct.get("manual_review",  0.3))   # 🟢 inference.confidence_thresholds.manual_review
```

> `CONF_THRESH_*` 상수는 `evaluation/metrics.py`에 backward-compatibility용으로 유지되나,
> `Evaluator` 내부에서는 `self.conf_thresh_*` 인스턴스 속성(cfg 기반)을 사용한다.
>
> `CONF_THRESH_*` constants remain in `evaluation/metrics.py` for backward compatibility,
> but `Evaluator` internally uses `self.conf_thresh_*` instance attributes (cfg-based).

---

## 4. 평가 지표 정의 / Metric Definitions

### 4.1 Overall Accuracy / 전체 정확도

```python
accuracy = (y_pred == y_true).mean()
# 전체 샘플 기준 (전 레벨 포함) / Based on all samples across all levels
```

### 4.2 Per-class F1

```python
from sklearn.metrics import f1_score
per_class_f1 = f1_score(y_true, y_pred, average=None)       # per-class array / 클래스별 배열
macro_f1     = f1_score(y_true, y_pred, average="macro")    # unweighted mean / 클래스별 F1의 비가중 평균
# 클래스별 F1의 비가중 평균 / Unweighted average of per-class F1
```

### 4.3 MAE (Mean Absolute Error)

```python
mae = np.mean(np.abs(y_pred - y_true))
# ordinal 속성 활용 — 레벨 간 오차를 숫자로 측정 / Leverages ordinal nature — measures numeric distance between levels
```

### 4.4 Per-class Metrics / 클래스별 지표

```python
from evaluation import compute_metrics

metrics = compute_metrics(y_true, y_pred, confidences)
# Returns EvaluationSummary with per_class: List[PerClassMetric]
# PerClassMetric: precision, recall, f1, support per level
```

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

```json
"inference": {
  "batch_size": 32    // 🟢 소비됨 / Consumed — GrayspotPredictor
}
```

---

## 6. Swing Feedback 정책 / Swing Feedback Policy

```python
from evaluation import determine_swing_feedback

decision = determine_swing_feedback(summary: EvaluationSummary)
# Returns: list of feedback strings
# summary.targets 의 swing_thresholds 값 기반으로 판단 / Decision based on swing_thresholds values in summary.targets
```

| 결과 / Result | 조건 / Condition | 다음 단계 / Next Step |
|---|---|---|
| `"pass"` | 모든 목표 달성 / All targets met | 시스템 종료 / System exit |
| `"retry_phase2"` | Acc < `swing_acc_retry` (0.80) | Phase 2 재학습 / Retrain Phase 2 |
| `"retry_phase0"` | F1 < `swing_f1_retry` (0.70) | Phase 0 재학습 → Phase 2 / Retrain Phase 0 → Phase 2 |

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

```json
{
  "channel": "Y",
  "accuracy": 0.92,
  "macro_f1": 0.87,
  "mae": 0.35,
  "per_class": {
    "0": {"precision": 0.95, "recall": 0.93, "f1": 0.94, "support": 120},
    "1": {"precision": 0.88, "recall": 0.85, "f1": 0.86, "support": 115},
    "2": {"precision": 0.90, "recall": 0.89, "f1": 0.89, "support": 110},
    "3": {"precision": 0.85, "recall": 0.83, "f1": 0.84, "support": 108},
    "4": {"precision": 0.82, "recall": 0.80, "f1": 0.81, "support": 105},
    "5": {"precision": 0.87, "recall": 0.86, "f1": 0.86, "support": 112}
  },
  "confusion_matrix": [[...], ...],
  "timestamp": "2026-05-08T12:00:00"
}
```

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

```python
from reporting import html_report

html_report(
    results=eval_results,
    output_dir=Path("outputs/reports"),
    channel="Y",
)
```

---

## 8. 학습 vs 평가 알려진 불일치 / Known Training–Evaluation Mismatches

학습(`Phase2Trainer`)과 평가(`Evaluator`) 간 처리 방식 비교.
Comparison of processing between training (`Phase2Trainer`) and evaluation (`Evaluator`).

| 항목 / Item | 학습 / Training | 평가 / Evaluation | 영향 / Impact |
|---|---|---|---|
| 색상 순서 / Color order | BGR (`cv2.imread`) | BGR (`cv2.imread`) — SSOT-CS01 해소됨 | ✅ 동일 / Identical |
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

## 10. SSOT 위반 현황 / Violations

| 코드 / Code | 위반 내용 / Violation | 등급 / Level | 해결 방법 / Fix |
|---|---|---|---|
| SSOT-CS01 | `evaluator_inference.py` `_EvalDataset.__getitem__()` 에서 `cv2.cvtColor(BGR→RGB)` 호출 → 학습(BGR)과 불일치 / `cv2.cvtColor(BGR→RGB)` call in `_EvalDataset.__getitem__()` caused mismatch with training (BGR) | Level 1 | ✅ **해소됨 / Resolved** — `evaluator_inference.py` line 59 제거 (BGR 유지 주석 추가) (2026-05-14) |
| SSOT-NM01 | 평가 전처리에서도 ImageNet norm 미적용 → 학습과 일치는 되나 pretrained 기대치와 다름 / ImageNet norm not applied in eval preprocessing — consistent with training but differs from pretrained expectations | Level 2 | ✅ **해소됨 / Resolved** — `predictor_inference.py` `_preprocess_images()` 에서 ImageNet norm 적용 (2026-05-14) |

> ✅ **해소됨 / Resolved**: SSOT-CS01 — `evaluator_inference.py` `_EvalDataset.__getitem__()` 의 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 제거. BGR 색상 공간 유지.
> ✅ **해소됨 / Resolved**: `evaluation.swing_thresholds.*` → `determine_swing_feedback()` 소비 완료.
> ✅ **해소됨 / Resolved**: config 키명 실제 config.json 구조(`evaluation.targets.*`)로 전면 정정.
> ✅ **해소됨 / Resolved**: SSOT-NM01 — `predictor_inference.py` `_preprocess_images()` 에서 `_IMAGENET_NORMALIZE` 적용 완료. 학습(`dataset.py`)·추론(`predictor_inference.py`) 정규화 완전 일치.

---
