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
8. [SSOT 위반 현황 / Violations](#8-ssot-위반-현황--violations)

---

## 1. Evaluator 개요 / Evaluator Overview

```python
from evaluation import Evaluator

evaluator = Evaluator(cfg, model, channel="Y")
results = evaluator.run()
# results: {"y_true": np.ndarray, "y_pred": np.ndarray, "confidences": np.ndarray}

evaluator.save_report(results, name="baseline_Y")
# → outputs/reports/evaluation_results_baseline_Y.csv
# → outputs/reports/metrics_summary_baseline_Y.json
```

### 1.1 공개 API / Public API

| 함수 / Function | 모듈 / Module | 설명 / Description |
|---|---|---|
| `Evaluator` | `evaluation.evaluator` | 추론 + 지표 + 리포트 통합 클래스 |
| `compute_metrics` | `evaluation.metrics` | 단일 채널 지표 계산 |
| `compute_all_channels` | `evaluation.metrics` | 전 채널 지표 일괄 계산 |
| `EvaluationSummary` | `evaluation.metrics` | 지표 결과 데이터클래스 |
| `determine_swing_feedback` | `evaluation.metrics` | Swing 재실행 여부 결정 |
| `plot_confusion_matrix` | `evaluation.confusion` | 채널별 혼동 행렬 시각화 |
| `plot_all_channels` | `evaluation.confusion` | 전 채널 혼동 행렬 일괄 생성 |

---

## 2. 성능 목표 / Performance Targets

PRD Section 1.4 기준 / Based on PRD Section 1.4

| 지표 / Metric | 목표값 / Target | config 키 / Key | 판정 / Pass Condition |
|---|---|---|---|
| Overall Accuracy / 전체 정확도 | ≥ 90% | `evaluation.targets.overall_accuracy` 🟢 | `accuracy >= 0.90` |
| Per-channel Accuracy / 채널별 정확도 | ≥ 85% | `evaluation.targets.per_color_accuracy` 🟢 | `accuracy >= 0.85` per channel |
| Per-class F1 / 클래스별 F1 | ≥ 0.80 | `evaluation.targets.per_class_f1` 🟢 | `f1 >= 0.80` per class |
| MAE | ≤ 0.50 | `evaluation.targets.mae` 🟢 | `mae <= 0.50` |

### 2.1 상수 정의 / Constants

```python
# evaluation/metrics.py
NUM_LEVELS          = 6      # Hard SSOT — data.num_levels와 동기화 필수
TARGET_OVERALL_ACC  = 0.90   # evaluation.targets.overall_accuracy
TARGET_PER_COLOR_ACC= 0.85   # evaluation.targets.per_color_accuracy
TARGET_PER_CLASS_F1 = 0.80   # evaluation.targets.per_class_f1
TARGET_MAE          = 0.50   # evaluation.targets.mae
```

### 2.2 Swing 재시도 임계값 / Swing Retry Thresholds

```python
# EvaluationSummary.targets 에 자동 주입 / Auto-injected from config
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
| `CONF_THRESH_WARN` | 0.5 | 경고 포함 수락 / Accept with warning | `inference.confidence_thresholds.warn_threshold` 🟢 |
| `CONF_THRESH_MANUAL` | 0.3 | 수동 검토 대기 / Queue for manual review | `inference.confidence_thresholds.manual_review` 🟢 |

```python
# evaluation/evaluator.py — Evaluator.__init__(cfg=...)
ct = (cfg or {}).get("inference", {}).get("confidence_thresholds", {})
self.conf_thresh_auto   = ct.get("auto_accept",   0.8)   # 🟢 inference.confidence_thresholds.auto_accept
self.conf_thresh_warn   = ct.get("warn_threshold", 0.5)  # 🟢 inference.confidence_thresholds.warn_threshold
self.conf_thresh_manual = ct.get("manual_review",  0.3)  # 🟢 inference.confidence_thresholds.manual_review
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
# 전체 샘플 기준 / Based on all samples across all levels
```

### 4.2 Per-class F1

```python
from sklearn.metrics import f1_score
per_class_f1 = f1_score(y_true, y_pred, average=None)       # per-class array
macro_f1     = f1_score(y_true, y_pred, average="macro")    # unweighted mean
# 클래스별 F1의 비가중 평균 / Unweighted average of per-class F1
```

### 4.3 MAE (Mean Absolute Error)

```python
mae = np.mean(np.abs(y_pred - y_true))
# ordinal 속성 활용 — 레벨 간 오차를 숫자로 측정
# Leverages ordinal nature — measures numeric distance between levels
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
determine_swing_feedback(summary)     ← 목표 달성 여부 판단 (swing_thresholds 기반)
    ↓
save_report(results, name)
    ↓
evaluation_results_{name}.csv + metrics_summary_{name}.json
```

### 5.1 Evaluator 입력 / Input Contract

| 입력 / Input | 타입 / Type | 조건 / Condition |
|---|---|---|
| 모델 / Model | `nn.Module` | `model.eval()` 상태 |
| 이미지 / Images | `Tensor (B, 3, 128, 128)` | BGR float32 [0, 1] |
| 레이블 / Labels | `Tensor (B,)` | int [0, 5] |

> **⚠️ SSOT-CS01**: `Evaluator`에 전달되는 이미지는 학습 시와 동일한 BGR 색상 공간이어야 한다.
> Images passed to `Evaluator` must use the same BGR color space as training.

### 5.2 평가 배치 크기 / Evaluation Batch Size

```json
"inference": {
  "batch_size": 32    // 🟢 소비됨 — GrayspotPredictor
}
```

---

## 6. Swing Feedback 정책 / Swing Feedback Policy

```python
from evaluation import determine_swing_feedback

decision = determine_swing_feedback(summary: EvaluationSummary)
# Returns: list of feedback strings
# summary.targets 의 swing_thresholds 값 기반으로 판단
```

| 결과 / Result | 조건 / Condition | 다음 단계 / Next Step |
|---|---|---|
| `"pass"` | 모든 목표 달성 / All targets met | 시스템 종료 / System exit |
| `"retry_phase2"` | Acc < `swing_acc_retry` (0.80) | Phase 2 재학습 |
| `"retry_phase0"` | F1 < `swing_f1_retry` (0.70) | Phase 0 재학습 → Phase 2 |

> Swing 임계값은 `evaluation.swing_thresholds.*` config에서 주입 — `EvaluationSummary.targets["swing_*_retry"]` 로 전달.
> Swing thresholds are injected from `evaluation.swing_thresholds.*` config into `EvaluationSummary.targets`.

---

## 7. 보고서 산출물 / Report Artifacts

```
outputs/reports/
├── eval_dashboard.html               ← Gauge + Bar 대시보드
├── per_class_metrics.html            ← 클래스별 F1 차트
├── mae_heatmap.html                  ← MAE 히트맵
├── misclassified_scatter.html        ← 오분류 Scatter
├── confidence_distribution.html      ← 신뢰도 분포
├── confusion/
│   ├── cm_{channel}.html             ← 채널별 혼동 행렬
│   └── cm_overall.html               ← 전체 혼동 행렬
├── evaluation_results_{name}.csv     ← 샘플별 예측 결과
├── misclassified_{name}.csv          ← 오분류 목록
└── metrics_summary_{name}.json       ← 집계 지표
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

### 7.2 보고서 생성 / Report Generation

```python
from reporting import html_report

html_report(
    results=eval_results,
    output_dir=Path("outputs/reports"),
    channel="Y",
)
```

---

## 8. SSOT 위반 현황 / Violations

| 코드 / Code | 위반 내용 / Violation | 등급 / Level | 해결 방법 / Fix |
|---|---|---|---|
| SSOT-CS01 | 평가 시 RGB로 읽으면 학습(BGR)과 불일치 | Level 1 | Evaluator 입력 전 BGR 검증 추가 |
| SSOT-NM01 | 평가 전처리에서도 ImageNet norm 미적용 → 학습과 일치는 되나 pretrained 기대치와 다름 | Level 2 | 의도 명시 또는 norm 추가 |

> ✅ **해소됨 / Resolved**: `evaluation.swing_thresholds.*` → `determine_swing_feedback()` 소비 완료.
> ✅ **해소됨 / Resolved**: config 키명 실제 config.json 구조(`evaluation.targets.*`)로 전면 정정.

---

**Version**: 0.2.0
**Last Updated**: 2026-05-08
**Applies to**: CMYK Grayspot Detection System v0.1.0+
