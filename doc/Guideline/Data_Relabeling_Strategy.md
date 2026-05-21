# 재라벨링 전략 / Re-Labeling Strategy

> **문서 유형 / Document Type**: 데이터 수집·라벨링 전략 / Data Collection & Labeling Strategy
> **관련 문서 / See also**: [Data_Audit.md](../Errors/Data_Audit.md), [SSOT_Data_Pipeline.md](../SSOT/SSOT_Data_Pipeline.md), [augmentation_policy.md](augmentation_policy.md)
> **근거 / Basis**: Data_Audit.md 채널별 현황 분석 기준

---

## Table of Contents / 목차

1. [배경 / Background](#1-배경--background)
2. [C 채널 기준 채택 근거 / Rationale for C Channel Baseline](#2-c-채널-기준-채택-근거--rationale-for-c-channel-baseline)
3. [×8.5 배율만으로 충분하지 않은 이유 / Why ×8.5 Alone Is Not Enough](#3-배율만으로-충분하지-않은-이유--why-85-alone-is-not-enough)
4. [권장 재라벨링 프로세스 / Recommended Re-Labeling Process](#4-권장-재라벨링-프로세스--recommended-re-labeling-process)
5. [채널별 조치 방향 / Per-Channel Action Plan](#5-채널별-조치-방향--per-channel-action-plan)
6. [M 채널 특이사항 / M Channel Special Case](#6-m-채널-특이사항--m-channel-special-case)
7. [PRD 목표 역산표 / PRD Target Back-Calculation](#7-prd-목표-역산표--prd-target-back-calculation)
8. [완료 기준 / Definition of Done](#8-완료-기준--definition-of-done)

---

## 1. 배경 / Background

현재 데이터셋은 ROI 추출 이후 패치 추출·증강 단계에서 채널 간 심각한 불균형이 발생했다.

The current dataset has severe inter-channel imbalance originating in the patch extraction and augmentation stage after ROI.

```
ROI 단계 (균일) / ROI stage (uniform):
  Y: 189장   M: 189장   C: 189장   K: 189장

labeled 단계 (불균형) / labeled stage (imbalanced):
  Y: 771장   M: 7,394장   C: 1,604장   K: 787장
  (×4.1)     (×39.1 ⚠️)   (×8.5)       (×4.2)
```

**PRD Section 6.3 v2 목표 (채널당 1,520장) / PRD v2 targets (1,520 per channel):**

| Level | 목표 / Target | 비율 근거 |
|---|---|---|
| 0 | 330 | ×3.3 (v1:100) |
| 1 | 330 | ×3.3 (v1:100) |
| 2 | 330 | ×3.3 (v1:100) |
| 3 | 265 | ×3.3 (v1:80) |
| 4 | 165 | ×3.3 (v1:50) |
| 5 | 100 | ×3.3 (v1:30) |
| **합계** | **1,520** | |

> ✅ **2026-05-21**: labeled/ OLD 데이터 삭제 완료. 새 데이터로 재구성 예정.
> ✅ **2026-05-21**: labeled/ old data cleared. Reconstruction with new data in progress.

---

## 2. PRD v2 기준 채택 근거 / Rationale for PRD v2 Targets

PRD v2 목표는 채널당 총 1,400~1,600장을 달성하기 위해 v1 레벨 비율을 ×3.3 스케일한 결과다.

PRD v2 targets are derived by scaling the v1 level ratios ×3.3 to achieve 1,400–1,600 total samples per channel.

| 기준 / Criterion | 평가 / Assessment |
|---|---|
| 채널별 총 목표 달성 | ✅ 1,520장 (1,400~1,600 범위 내) |
| v1 레벨 비율 유지 | ✅ 100:100:100:80:50:30 비율 그대로 |
| 채널 간 균형 | ✅ 4개 채널 동일 목표 적용 |
| 채널 독립 라벨링 | ✅ C/M/Y/K 각각 독립 레벨 부여 필수 |

> **결론**: 모든 채널(C/M/Y/K)에 동일한 PRD v2 목표를 적용한다. 같은 스캔의 채널이라도 레벨은 독립적으로 부여한다.
> **Conclusion**: PRD v2 targets apply uniformly to all channels. Channels from the same scan must be labeled independently.

---

## 3. 채널별 독립 라벨링이 필요한 이유 / Why Per-Channel Independent Labeling Is Required

스캔 파일명에 포함된 레벨(`lvlX_...`)은 **인쇄 스캔 전체**의 레벨이다. 그러나 CMYK 각 채널은 서로 다른 결함 심각도를 가질 수 있다.

The level encoded in the scan filename (`lvlX_...`) represents the **overall scan level**. Each CMYK channel can have a different defect severity.

```
예시 / Example:
  lvl3_Scanned Documents (116)_4_1.jpeg → ROI 분리 후:
    Y 채널: 결함 없음  → Level 0
    M 채널: 경미한 결함 → Level 2
    C 채널: 중간 결함   → Level 3   ← 파일명 기준 레벨
    K 채널: 결함 없음  → Level 0
```

파일명 레벨을 4개 채널 전체에 복사하면 레벨 분포가 채널별로 동일해지는 오류가 발생한다.

Copying the filename level to all 4 channels produces identical level distributions across channels — which is incorrect.

### 핵심 원칙 / Core Principle

```
각 ROI 이미지(채널)를 개별적으로 육안 검사하여 레벨을 부여해야 한다.
Each ROI image (per channel) must be visually inspected and labeled individually.
```

---

## 4. 권장 재라벨링 프로세스 / Recommended Re-Labeling Process

### Step 1 — 채널별 ROI 이미지 개별 육안 검사

새 ROI 파일(`data_set/roi/lvlX_..._CH.png`)을 채널별로 열어 각 채널의 실제 결함 레벨을 독립적으로 판정한다.

Open new ROI files (`data_set/roi/lvlX_..._CH.png`) and assign each channel's defect level independently.

```
현재 신규 ROI 파일 수 / Current new ROI files:
  Y: 288장   M: 288장   C: 288장   K: 288장
  (lvlX_..._Y.png / _M.png / _C.png / _K.png)
```

### Step 2 — 패치 추출 후 labeled/ 배치

라벨링 완료 후 ROI 이미지에서 패치를 추출하여 `data_set/labeled/{channel}/{level}/`에 배치한다.

After labeling, extract patches from ROI images and place under `data_set/labeled/{channel}/{level}/`.

### Step 3 — 레벨별 필요 증강량 역산

```python
# PRD v2 목표 / PRD v2 targets
PRD_targets = {0: 330, 1: 330, 2: 330, 3: 265, 4: 165, 5: 100}

for level in range(6):
    labeled_count = labeled_distribution[channel][level]
    shortage = max(0, PRD_targets[level] - labeled_count)
    # augment_dataset.py 가 자동 계산하여 증강 수행
```

### Step 4 — augment_dataset.py 실행

```bash
python -m src.scripts.augment_dataset
```

PRD v2 목표 미달 (channel, level) 쌍에 대해 자동 증강 후 `labels_master.csv` 갱신.

### Step 4 — 라벨 CSV 통합

재라벨링 완료 후 단일 CSV 파일로 통합한다.

```
이전 / Previous:  labels_v0.csv  +  labels_cmyk.csv  (K 채널 이중 배치 문제)
현재 완료 / Done: labels_master.csv  (전 채널 통합 단일 파일, long-format)
```

> ✅ **2026-05-21 완료**: `data_set/labels_master.csv` 가 생성되어 현재 K 채널의 학습(787장) vs 평가 커버리지 불일치 문제가 해소되었다.
> ✅ **2026-05-21 Complete**: `data_set/labels_master.csv` has been created, resolving the K channel training (787) vs evaluation coverage mismatch.

재라벨링으로 새 이미지가 추가될 경우 `data_set/labeled/{channel}/{level}/` 에 파일을 추가한 뒤 `src/scripts/augment_dataset.py` 를 실행해 부족 레벨을 보충하고 `labels_master.csv` 를 갱신한다.
When new images are added via re-labeling, place them under `data_set/labeled/{channel}/{level}/`, then run `src/scripts/augment_dataset.py` to fill any shortage levels and update `labels_master.csv`.

---

## 5. 채널별 조치 방향 / Per-Channel Action Plan

| 채널 | 현황 (2026-05-21) | 조치 방향 |
|---|---|---|
| **Y** | ROI 288장 (재라벨링 필요) | 개별 육안 라벨링 → 패치 추출 → augment_dataset.py |
| **M** | ROI 288장 (재라벨링 필요) | 개별 육안 라벨링 → 패치 추출 → augment_dataset.py |
| **C** | ROI 288장 (재라벨링 필요) | 개별 육안 라벨링 → 패치 추출 → augment_dataset.py |
| **K** | ROI 288장 (재라벨링 필요) | 개별 육안 라벨링 → 패치 추출 → augment_dataset.py |

> OLD labeled 데이터(8,969장) 삭제 완료 (2026-05-21). 4개 채널 모두 재구성 필요.
> Old labeled data (8,969 images) cleared on 2026-05-21. All 4 channels require reconstruction.

---

## 6. 신규 ROI 현황 / Current New ROI Status

신규 RAW 스캔 288장 → CMYK 분리 → ROI 1,152장 (채널당 288장)

New raw scans: 288 → CMYK split → 1,152 ROI files (288 per channel)

현재 ROI 레벨 분포 (파일명 기준, 채널 미분리) / Current ROI level distribution (filename-based, not channel-differentiated):

| Level | 현재 ROI 수 (채널당) | PRD v2 목표 | 부족 (증강 전) |
|---|---|---|---|
| 0 | 8 | 330 | **−322** |
| 1 | 55 | 330 | **−275** |
| 2 | 54 | 330 | **−276** |
| 3 | 63 | 265 | **−202** |
| 4 | 54 | 165 | **−111** |
| 5 | 54 | 100 | **−46** |

> ⚠️ 위 분포는 채널별 독립 라벨링 전 수치다. 실제 분포는 라벨링 후 달라진다.
> The above distribution is pre-labeling. Actual per-channel distribution will differ after independent labeling.

---

## 7. PRD v2 목표표 / PRD v2 Target Table

| Level | PRD v2 목표 (채널당) | v1 목표 | 배율 |
|---|---|---|---|
| 0 | 330 | 100 | ×3.3 |
| 1 | 330 | 100 | ×3.3 |
| 2 | 330 | 100 | ×3.3 |
| 3 | 265 | 80 | ×3.3 |
| 4 | 165 | 50 | ×3.3 |
| 5 | 100 | 30 | ×3.3 |
| **합계** | **1,520** | 460 | |

---

## 8. 완료 기준 / Definition of Done

| 항목 / Item | 기준 / Criterion |
|---|---|
| PRD v2 목표 달성 | Y / M / C / K 전 채널, Level 0~5 모두 PRD v2 목표 충족 (채널당 1,520장) |
| 채널 독립 라벨링 | 각 ROI 이미지가 채널별로 개별 라벨링됨 |
| CSV 통합 | `labels_master.csv` — 전 채널 단일 파일 (long-format: filepath/channel/level) |
| 학습-평가 일치 | `CMYKDataset` (학습) 과 `Evaluator` (평가) 가 동일 `labels_master.csv` 참조 |
| 채널 간 균형 | 4개 채널 모두 1,400~1,600장 범위 내 |

---
