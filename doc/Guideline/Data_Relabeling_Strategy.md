# 재라벨링 전략 / Re-Labeling Strategy

> **문서 유형 / Document Type**: 데이터 수집·라벨링 전략 / Data Collection & Labeling Strategy
> **관련 문서 / See also**: [Data_Audit.md](Data_Audit.md), [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md)
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

**PRD Section 6.3 목표 달성 현황 / PRD target achievement:**

| 채널 | 합계 | PRD 미달 레벨 | 증강 배율 | 종합 평가 |
|---|---|---|---|---|
| Y | 771장 | Level 1 (-24장) | ×4.1 | ⚠️ 1개 레벨 미달 |
| M | 7,394장 | 없음 | ×39.1 | ⚠️ 배율 근거 없음 |
| C | 1,604장 | **없음** | ×8.5 | ✅ **기준 채널** |
| K | 787장 | Level 1/2/5 (-28장) | ×4.2 | ❌ 3개 레벨 미달 |

---

## 2. C 채널 기준 채택 근거 / Rationale for C Channel Baseline

C 채널을 재라벨링 기준으로 삼는 이유:

| 기준 / Criterion | C 채널 평가 / C Channel Assessment |
|---|---|
| PRD 전 레벨 달성 | ✅ Level 0~5 모두 목표 초과 |
| 증강 배율 적정성 | ✅ ×8.5 — 과도하지 않고 문서화 가능한 수준 |
| 배율 근거 존재 | ✅ PRD 목표 달성 결과로 역추적 가능 |
| 팀 기준 통일성 | ✅ 4개 채널에 동일 기준 적용 가능 |

> **결론**: C 채널의 ×8.5 증강 배율을 **모든 채널의 기본 기준**으로 삼는다.
> **Conclusion**: C channel's ×8.5 augmentation multiplier serves as the **default baseline for all channels**.

---

## 3. 배율만으로 충분하지 않은 이유 / Why ×8.5 Alone Is Not Enough

×8.5 배율이 C 채널에서 성공한 건 **ROI 189장 내 각 레벨의 분포가 충분했기 때문**이다.

The ×8.5 multiplier succeeded for C because **the 189 ROI images had sufficient per-level distribution**.

### 문제 시나리오 / Problem Scenario

```
가정: K 채널 ROI 189장 중 Level 5가 3장뿐인 경우
Assumption: Only 3 Level 5 images in K channel's 189 ROI images

  3장 × 8.5 = 25.5장 → 반올림 25장
  PRD 목표: 30장 이상
  → ❌ ×8.5 적용해도 여전히 PRD 미달
```

따라서 **배율 설계 전에 ROI 단계의 레벨별 분포를 먼저 파악하는 것이 필수**다.

Therefore, **understanding the per-level distribution in the ROI stage must precede multiplier design**.

### 핵심 원칙 / Core Principle

```
PRD 목표 (레벨별) ÷ ROI 레벨별 장수 = 해당 레벨에 필요한 최소 배율
PRD target (per level) ÷ ROI count (per level) = minimum multiplier needed for that level
```

---

## 4. 권장 재라벨링 프로세스 / Recommended Re-Labeling Process

### Step 1 — ROI 레벨별 분포 파악 (선행 필수)

ROI 189장을 레벨별로 몇 장 보유하는지 채널별로 집계한다.

```
조사 항목:
  각 채널(Y/M/C/K)의 ROI 189장 중
  Level 0: ?장
  Level 1: ?장
  Level 2: ?장
  Level 3: ?장
  Level 4: ?장
  Level 5: ?장
```

> 이 숫자가 확정되어야 Step 2~3 설계가 가능하다.
> Steps 2~3 cannot be designed without this data.

### Step 2 — 레벨별 필요 배율 역산

```python
# 역산 공식 / Back-calculation formula
PRD_targets = {0: 100, 1: 100, 2: 100, 3: 80, 4: 50, 5: 30}

for level in range(6):
    roi_count = roi_distribution[channel][level]
    required_multiplier = math.ceil(PRD_targets[level] / roi_count)
    effective_multiplier = max(required_multiplier, 8.5)  # C 기준 최솟값
```

- 모든 레벨에서 **최소 ×8.5** 적용 (C 기준)
- PRD 달성에 ×8.5 이상이 필요한 레벨은 **해당 레벨만 높은 배율** 적용
- 채널 전체에 균일한 단일 배율을 강제하지 않음

### Step 3 — 증강 파라미터 통일

C 채널에 적용된 것과 **동일한 증강 파이프라인**을 사용한다.

```python
# cfg["phase2"]["augmentation"] 기준 / Based on cfg["phase2"]["augmentation"]
aug_cfg = {
    "flip_prob":        0.5,
    "brightness_prob":  0.3,
    "brightness_range": 20,
    "noise_prob":       0.2,
    "noise_range":      10,
}
```

> 증강 종류와 강도까지 C와 동일하게 맞춰야 데이터 품질 일관성이 유지된다.
> Augmentation type and intensity must match C to maintain data quality consistency.

### Step 4 — 라벨 CSV 통합

재라벨링 완료 후 단일 CSV 파일로 통합한다.

```
현재 / Current:  labels_v0.csv  +  labels_cmyk.csv  (K 채널 이중 배치 문제)
목표 / Target:   labels_merged.csv  (전 채널 통합 단일 파일)
```

> 이 조치는 현재 K 채널의 학습(787장) vs 평가(341 or 446장) 커버리지 불일치도 함께 해소한다.
> This also resolves the current K channel training (787) vs evaluation (341 or 446) coverage mismatch.

---

## 5. 채널별 조치 방향 / Per-Channel Action Plan

| 채널 | 현황 | 기본 방향 | 특이사항 |
|---|---|---|---|
| **C** | ✅ 기준 채널 | 변경 없음 — 기준으로 삼음 | — |
| **Y** | ⚠️ Level 1 부족 | ×8.5 기본 + Level 1 ROI 분포 확인 후 조정 | Level 1: ROI 분포에 따라 추가 수집 필요할 수 있음 |
| **M** | ⚠️ 배율 과다 | ×8.5로 재조정 권장 | 총 장수 7,394 → 약 1,600 수준으로 감소 예상 — 팀 결정 필요 |
| **K** | ❌ 3개 레벨 미달 | ×8.5 기본 + 미달 레벨 집중 증강 | Level 5 ROI가 적으면 추가 스캔 필수 |

---

## 6. M 채널 특이사항 / M Channel Special Case

M 채널은 별도 결정이 필요하다.

M channel requires a separate team decision.

| 옵션 / Option | 설명 / Description | 결과 / Outcome |
|---|---|---|
| **A. ×8.5로 재조정** | C 기준에 맞춰 통일 | 7,394 → ~1,600장. 채널 간 균형 확보. 재라벨링 작업 발생 |
| **B. 현행 유지** | ×39 그대로 유지 | 장수 유지. 단, 과도 증강 근거를 문서화해야 함 |
| **C. 절충 (×15 내외)** | PRD 목표 3~4배 수준으로 타협 | 균형과 실용성 사이 타협 |

> **권장 / Recommended**: 옵션 A (×8.5 재조정) — 채널 간 데이터 불균형이 모델 학습 시 편향을 유발할 수 있음.
> 단, M 채널의 기존 라벨링 작업이 많으므로 팀 합의 후 결정.
>
> **Recommended**: Option A (reset to ×8.5) — inter-channel imbalance can introduce training bias.
> However, team agreement is required given the existing M channel labeling effort.

---

## 7. PRD 목표 역산표 / PRD Target Back-Calculation

ROI 레벨별 분포 조사 후 아래 표를 채워 넣으면 채널별 필요 배율이 확정된다.

Fill in the table below after surveying ROI per-level distribution to finalize required multipliers per channel.

### Y 채널

| Level | PRD 목표 | ROI 장수 (조사 필요) | 필요 최소 배율 | 채택 배율 |
|---|---|---|---|---|
| 0 | 100+ | ? | ? | max(?, 8.5) |
| 1 | 100+ | ? | ? | max(?, 8.5) |
| 2 | 100+ | ? | ? | max(?, 8.5) |
| 3 | 80+ | ? | ? | max(?, 8.5) |
| 4 | 50+ | ? | ? | max(?, 8.5) |
| 5 | 30+ | ? | ? | max(?, 8.5) |

### K 채널

| Level | PRD 목표 | ROI 장수 (조사 필요) | 필요 최소 배율 | 채택 배율 |
|---|---|---|---|---|
| 0 | 100+ | ? | ? | max(?, 8.5) |
| 1 | 100+ | ? | ? | max(?, 8.5) |
| 2 | 100+ | ? | ? | max(?, 8.5) |
| 3 | 80+ | ? | ? | max(?, 8.5) |
| 4 | 50+ | ? | ? | max(?, 8.5) |
| 5 | 30+ | ? | ? | max(?, 8.5) |

> ROI 레벨별 장수가 확인되면 이 표를 완성하고 `SSOT_Data_Pipeline.md` 에 반영한다.
> Once ROI per-level counts are confirmed, complete this table and update `SSOT_Data_Pipeline.md`.

---

## 8. 완료 기준 / Definition of Done

재라벨링 작업이 완료된 것으로 판단하는 기준:

| 항목 / Item | 기준 / Criterion |
|---|---|
| PRD 목표 달성 | Y / M / C / K 전 채널, Level 0~5 모두 PRD Section 6.3 목표 충족 |
| 증강 배율 문서화 | 채널별·레벨별 적용 배율이 이 문서에 기록됨 |
| CSV 통합 | `labels_merged.csv` 생성 완료 — 전 채널 단일 파일 |
| 학습-평가 일치 | `CMYKDataset` (학습) 과 `Evaluator` (평가) 가 동일 파일 세트를 참조 |
| 채널 간 균형 | 가장 적은 채널과 가장 많은 채널의 총 장수 비율이 ×5 이내 |

---
