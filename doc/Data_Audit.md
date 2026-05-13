# Data Audit — 데이터셋 현황 감사 / Dataset Status Audit

CMYK Grayspot Detection System 의 실제 데이터셋 현황을 PRD 목표 수량과 대조한 감사 보고서.

This report audits the current dataset against PRD Section 6.3 target quantities and documents structural findings discovered during the audit.

> **작성일 / Date**: 2026-05-13
> **대상 경로 / Target Path**: `data_set/`
> **PRD 근거 / PRD Reference**: `Grayspot_pipeline_design(PRD).pdf` Section 6.3

---

## Table of Contents / 목차

1. [데이터 파이프라인 구조 / Data Pipeline Structure](#1-데이터-파이프라인-구조--data-pipeline-structure)
2. [채널별 데이터 현황 / Per-Channel Data Status](#2-채널별-데이터-현황--per-channel-data-status)
3. [PRD 목표 대비 달성 현황 / PRD Target Achievement](#3-prd-목표-대비-달성-현황--prd-target-achievement)
4. [구조적 발견사항 / Structural Findings](#4-구조적-발견사항--structural-findings)
5. [학습-평가 불일치 / Train-Eval Mismatch](#5-학습-평가-불일치--train-eval-mismatch)
6. [권장 조치사항 / Recommended Actions](#6-권장-조치사항--recommended-actions)

---

## 1. 데이터 파이프라인 구조 / Data Pipeline Structure

```
data_set/raw/         (378장 JPEG)
    ↓ ROI 추출 / ROI extraction
data_set/roi/         (채널별 각 189장, 총 756장)
    ↓ 패치 추출 + 라벨링 / Patch extraction + labeling
data_set/labeled/     (총 10,556장)
    ├── Y/  (771장)
    ├── M/  (7,394장)
    ├── C/  (1,604장)
    └── K/  (787장)
```

### 핵심 관찰 / Key Observation

ROI 단계까지는 4채널 모두 **189장으로 동일**하다.
채널 간 불균형은 ROI 이후 패치 추출·증강 단계에서 발생했다.

ROI output is **identical across all 4 channels (189 each)**.
The inter-channel imbalance originates in the patch extraction and augmentation stage after ROI.

| 단계 / Stage | Y | M | C | K |
|---|---|---|---|---|
| raw → roi | 189 | 189 | 189 | 189 |
| roi → labeled | 771 | 7,394 | 1,604 | 787 |
| **증강 배율 / Aug multiplier** | **×4.1** | **×39.1** | **×8.5** | **×4.2** |

> ⚠️ M 채널 증강 배율(×39)은 PRD에 명시된 기준이 없다. 다른 채널 대비 약 10배이며 근거 문서가 확인되지 않는다.
>
> ⚠️ M channel augmentation multiplier (×39) has no stated basis in the PRD. It is approximately 10× other channels with no documented justification.

---

## 2. 채널별 데이터 현황 / Per-Channel Data Status

### Y 채널 (771장 / 771 images)

| Level | 설명 / Description | 실제 / Actual | PRD 목표 / Target | 상태 / Status |
|---|---|---|---|---|
| 0 | 정상 / Normal | 189 | 100+ | ✅ |
| 1 | 매우 미세 / Very subtle | **76** | 100+ | ❌ **–24장** |
| 2 | 약간 인지 / Slightly visible | 101 | 100+ | ✅ |
| 3 | 명확 / Clear | 112 | 80+ | ✅ |
| 4 | 심각 / Severe | 250 | 50+ | ✅ |
| 5 | 극심 / Extreme | 43 | 30+ | ✅ |

### M 채널 (7,394장 / 7,394 images)

| Level | 실제 / Actual | PRD 목표 / Target | 상태 / Status |
|---|---|---|---|
| 0 | 193 | 100+ | ✅ |
| 1 | 1,854 | 100+ | ✅ |
| 2 | 1,235 | 100+ | ✅ |
| 3 | 1,972 | 80+ | ✅ |
| 4 | 1,193 | 50+ | ✅ |
| 5 | 947 | 30+ | ✅ |

> M 채널은 모든 레벨에서 PRD 목표를 초과 달성했다. 단, 증강 배율 기준 부재로 다른 채널과의 데이터 불균형이 심화되었다.
>
> M channel exceeds all PRD targets. However, the lack of augmentation ratio standards has deepened inter-channel imbalance.

### C 채널 (1,604장 / 1,604 images)

| Level | 실제 / Actual | PRD 목표 / Target | 상태 / Status |
|---|---|---|---|
| 0 | 151 | 100+ | ✅ |
| 1 | 241 | 100+ | ✅ |
| 2 | 399 | 100+ | ✅ |
| 3 | 271 | 80+ | ✅ |
| 4 | 362 | 50+ | ✅ |
| 5 | 180 | 30+ | ✅ |

### K 채널 (787장 / 787 images)

| Level | 설명 / Description | 실제 / Actual | PRD 목표 / Target | 상태 / Status |
|---|---|---|---|---|
| 0 | 정상 / Normal | 158 | 100+ | ✅ |
| 1 | 매우 미세 / Very subtle | **93** | 100+ | ❌ **–7장** |
| 2 | 약간 인지 / Slightly visible | **83** | 100+ | ❌ **–17장** |
| 3 | 명확 / Clear | 274 | 80+ | ✅ |
| 4 | 심각 / Severe | 153 | 50+ | ✅ |
| 5 | 극심 / Extreme | **26** | 30+ | ❌ **–4장** |

---

## 3. PRD 목표 대비 달성 현황 / PRD Target Achievement

PRD Section 6.3 기준 채널별 달성 요약 / Achievement summary per channel vs PRD Section 6.3:

| 채널 / Channel | 합계 / Total | PRD 미달 레벨 / Below Target | 상태 / Status |
|---|---|---|---|
| Y | 771 | Level 1 (76장, –24) | ⚠️ 1개 레벨 미달 / 1 level below target |
| M | 7,394 | 없음 / None | ✅ 전 레벨 달성 / All levels met |
| C | 1,604 | 없음 / None | ✅ 전 레벨 달성 / All levels met |
| K | 787 | Level 1 (93, –7) · Level 2 (83, –17) · Level 5 (26, –4) | ❌ 3개 레벨 미달 / 3 levels below target |

### PRD Section 6.3 원문 / PRD Section 6.3 Original

```
Level  목표 샘플 수 (per color) / Target samples (per color)
  0    100+
  1    100+
  2    100+
  3     80+
  4     50+
  5     30+
```

---

## 4. 구조적 발견사항 / Structural Findings

### 4.1 두 개의 라벨링 배치 / Two Labeling Batches

`data_set/labeled/K/` 폴더를 조사한 결과, 두 가지 파일명 형식이 혼재함을 발견했다.

Investigation of `data_set/labeled/K/` revealed two distinct filename formats coexisting in the folder.

| 배치 / Batch | 파일명 형식 / Filename Format | CSV | K 채널 파일 수 / K files |
|---|---|---|---|
| 1차 라벨링 / 1st labeling | `scan_002_K_0008.png` | `labels_v0.csv` | 446장 |
| 2차 라벨링 / 2nd labeling | `0_6_1_K_0007.png` | `labels_cmyk.csv` | 341장 |
| **합계 / Total** | | | **787장** |

> - 두 CSV 간 공통 파일은 0개 — 완전히 다른 파일 세트 / Zero files in common between the two CSVs — entirely separate file sets
> - 두 배치 모두 `data_set/labeled/` 폴더에 레벨별 디렉토리 구조로 저장되어 있음 / Both batches are stored in the `data_set/labeled/` folder under level-based subdirectories

### 4.2 학습 시 사용 범위 / Training Scope

`CMYKDataset`은 CSV를 읽지 않고 `labeled/{channel}/{level}/*.png` 폴더를 직접 스캔한다.

`CMYKDataset` does not read any CSV — it directly scans `labeled/{channel}/{level}/*.png`.

```python
# src/data/dataset.py line 79
for img_path in sorted(level_dir.glob("*")):   # CSV 미사용 / No CSV used
```

따라서 두 배치(787장)가 **모두 학습에 사용**되고 있다.
Both batches (787 images) are **used in training**.

---

## 5. 학습-평가 불일치 / Train-Eval Mismatch

`Evaluator`는 `labels_csv` 파라미터로 전달된 CSV 파일만을 기준으로 평가를 수행한다.

`Evaluator` performs evaluation only on samples defined in the `labels_csv` parameter.

```python
ev = Evaluator(
    model       = model,
    labeled_dir = Path("data_set/labeled"),
    labels_csv  = Path("data_set/labels_v0.csv"),   # ← 어느 CSV를 넘기는가?
    ...
)
```

| 구분 / Scope | 사용 파일 수 / Files used | K 채널 / K channel |
|---|---|---|
| **학습 / Training** | 전체 폴더 스캔 | 787장 |
| **평가 (labels_cmyk.csv 전달 시)** | CSV 기준 | 341장 (나머지 446장 평가 제외) |
| **평가 (labels_v0.csv 전달 시)** | CSV 기준 | 446장 (나머지 341장 평가 제외) |

> ⚠️ 어느 CSV를 사용하더라도 학습(787장)과 평가(341 or 446장) 간 커버리지 불일치가 발생한다.
> 두 CSV를 병합한 단일 CSV 파일을 사용하면 이 문제를 해소할 수 있다.
>
> ⚠️ Regardless of which CSV is used, a training (787) vs evaluation (341 or 446) coverage mismatch exists.
> Merging the two CSVs into a single file resolves this issue.

---

## 6. 권장 조치사항 / Recommended Actions

### 즉시 가능 / Immediately Actionable

| 우선순위 / Priority | 조치 / Action | 효과 / Effect |
|---|---|---|
| 🔴 **1** | `labels_v0.csv` + `labels_cmyk.csv` 병합 → `labels_merged.csv` 생성 후 Evaluator에 적용 | 평가 커버리지 341장 → 787장으로 확대 (K 채널) |
| 🔴 **2** | K Level 5 추가 스캔 (4장 이상) | PRD 목표 달성 |

### 추가 수집 필요 / Requires Additional Data Collection

| 우선순위 / Priority | 조치 / Action | 필요 수량 / Required |
|---|---|---|
| 🟠 **3** | K Level 2 추가 라벨링 | 17장 이상 |
| 🟠 **4** | K Level 1 추가 라벨링 | 7장 이상 |
| 🟡 **5** | Y Level 1 추가 라벨링 | 24장 이상 |

### 문서화 권장 / Documentation Recommended

| 항목 / Item | 내용 / Content |
|---|---|
| M 채널 증강 배율 근거 문서화 | ×39 배율 적용 이유 및 기준 명시 |
| 채널별 목표 증강 배율 정의 | PRD 또는 SSOT_Data_Pipeline.md에 추가 |
| 두 라벨링 배치 이력 기록 | 1차(labels_v0) / 2차(labels_cmyk) 수집 시점 및 기준 차이 명시 |

---

## 부록 / Appendix

### A. labels_v0.csv K 채널 레벨 분포 / labels_v0.csv K Channel Level Distribution

```
Level 0:   5장
Level 1:  16장
Level 2:  71장
Level 3: 231장
Level 4: 121장
Level 5:   2장
합계:    446장
```

### B. labels_cmyk.csv K 채널 레벨 분포 / labels_cmyk.csv K Channel Level Distribution

```
Level 0: 153장
Level 1:  77장
Level 2:  12장
Level 3:  43장
Level 4:  32장
Level 5:  24장
합계:    341장
```

### C. CSV 병합 시 K 채널 예상 분포 / Projected K Distribution After CSV Merge

| Level | labels_v0 | labels_cmyk | 병합 합계 / Merged Total | PRD 목표 / Target | 상태 / Status |
|---|---|---|---|---|---|
| 0 | 5 | 153 | 158 | 100+ | ✅ |
| 1 | 16 | 77 | 93 | 100+ | ❌ –7장 |
| 2 | 71 | 12 | 83 | 100+ | ❌ –17장 |
| 3 | 231 | 43 | 274 | 80+ | ✅ |
| 4 | 121 | 32 | 153 | 50+ | ✅ |
| 5 | 2 | 24 | 26 | 30+ | ❌ –4장 |

> CSV 병합은 평가 커버리지를 높이지만 PRD 미달 자체는 추가 수집으로만 해결된다.
> CSV merge improves evaluation coverage, but PRD shortfalls can only be resolved by collecting additional data.

---

**Version**: 1.0.0
**Date**: 2026-05-13
**Author**: CMYK Project Team
**Applies to**: CMYK Grayspot Detection System v0.1.0+
