# ADR — ImageNet 정규화 채택 결정 / ImageNet Normalization Adoption Decision

> **문서 유형 / Document Type**: Architecture Decision Record (ADR)
> **관련 문서 / See also**: [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md), [Contract.md](Contract.md)
> **SSOT 코드 / SSOT Code**: SSOT-NM01

---

## Table of Contents / 목차

1. [결정 요약 / Decision Summary](#1-결정-요약--decision-summary)
2. [배경 / Context](#2-배경--context)
3. [결정 / Decision](#3-결정--decision)
4. [이유 / Rationale](#4-이유--rationale)
5. [적용하지 않았을 때의 결과 / Consequences of Not Applying](#5-적용하지-않았을-때의-결과--consequences-of-not-applying)
6. [일관성 요구사항 / Consistency Requirements](#6-일관성-요구사항--consistency-requirements)
7. [기각된 대안 / Rejected Alternatives](#7-기각된-대안--rejected-alternatives)
8. [위반 이력 / Violation History](#8-위반-이력--violation-history)

---

## 1. 결정 요약 / Decision Summary

모든 이미지 입력에 **ImageNet 채널별 mean/std** 를 사용하여 정규화한다.

All image inputs are normalized using **ImageNet channel-wise mean and standard deviation**.

```python
_IMAGENET_NORMALIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225],
)
```

> **적용 범위 / Scope**: 학습(training), 추론(inference), 평가(evaluation) 전 단계 동일하게 적용.
> Applied identically across training, inference, and evaluation.

---

## 2. 배경 / Context

이 프로젝트는 그레이스팟 결함 감지를 위해 **전이 학습(Transfer Learning)** 방식을 채택한다.

- **Phase 0**: SimCLR 기반 대조 학습 — EfficientNet-B0 또는 ResNet-50 backbone 사용
- **Phase 2**: Supervised fine-tuning — Phase 0 backbone 위에 ClassifierHead 추가

두 backbone 모두 **ImageNet 데이터셋으로 사전 학습된 가중치**로 초기화된다.

This project uses **Transfer Learning** for grayspot defect detection.

- **Phase 0**: SimCLR-based contrastive learning — EfficientNet-B0 or ResNet-50 backbone
- **Phase 2**: Supervised fine-tuning — ClassifierHead added on top of Phase 0 backbone

Both backbones are initialized with **weights pretrained on the ImageNet dataset**.

---

## 3. 결정 / Decision

### 정규화 파라미터 / Normalization Parameters

| 채널 / Channel | Mean | Std |
|---|---|---|
| R (Red) | 0.485 | 0.229 |
| G (Green) | 0.456 | 0.224 |
| B (Blue) | 0.406 | 0.225 |

> ⚠️ **SSOT-CS01 주의**: 이 프로젝트는 BGR 색상 공간을 유지한다. 위 파라미터는 채널 순서 그대로 (B→R, G→G, R→B 위치에) 적용된다.
> The project maintains BGR color space. The parameters above are applied in channel order as-is.

### 적용 순서 / Processing Order

```
cv2.imread()          →  (H, W, 3)  uint8  [0, 255]  BGR
resize(128, 128)      →  (128, 128, 3)  float32
/ 255.0               →  (128, 128, 3)  float32  [0.0, 1.0]
permute(2, 0, 1)      →  (3, 128, 128)  float32
_IMAGENET_NORMALIZE   →  (3, 128, 128)  float32  ImageNet-normalized
```

---

## 4. 이유 / Rationale

### 4.1 사전 학습 가중치 호환성 / Pretrained Weight Compatibility

EfficientNet-B0 와 ResNet-50 은 ImageNet 사전 학습 시 **모든 입력이 ImageNet mean/std 로 정규화된 상태**에서 가중치를 최적화했다.

EfficientNet-B0 and ResNet-50 optimized their weights during ImageNet pretraining with **all inputs normalized by ImageNet mean/std**.

이는 backbone 의 모든 레이어 — 합성곱 필터, Batch Normalization 통계 — 가 해당 입력 분포를 기대하도록 보정되어 있음을 의미한다.

This means every layer of the backbone — convolutional filters, Batch Normalization statistics — is calibrated to expect that specific input distribution.

```
사전 학습 시 입력 분포  ≡  학습·추론 시 입력 분포
Pretraining input distribution  ≡  Training & inference input distribution
```

### 4.2 전이 학습 최적 시작점 / Optimal Transfer Learning Starting Point

정규화를 맞추면 Phase 0 대조 학습이 **이미 유의미한 특징(엣지, 텍스처, 형태)을 인식하는 backbone** 에서 시작된다.
정규화를 맞추지 않으면 backbone 의 초기 특징 추출이 왜곡된 상태에서 학습이 시작된다.

Matching normalization means Phase 0 contrastive learning starts from a **backbone that already recognizes meaningful features (edges, textures, shapes)**.
Without matching, the backbone's initial feature extraction is distorted from the start.

### 4.3 소규모 데이터셋에서의 안정성 / Stability on Small Datasets

CMYK 그레이스팟 데이터셋은 채널당 최소 771장으로 소규모다.
소규모 데이터에서 커스텀 통계를 계산하면 노이즈가 크고 신뢰도가 낮다.
ImageNet 통계는 1.2억 장 기반으로 안정적이고 검증된 값이다.

The CMYK grayspot dataset is small (minimum 771 images per channel).
Computing custom statistics from a small dataset produces noisy, unreliable values.
ImageNet statistics are stable, validated values derived from 1.2 million images.

### 4.4 산업 표준 / Industry Standard

ImageNet 사전 학습 backbone 을 사용할 때 ImageNet 정규화를 적용하는 것은
컴퓨터 비전 분야의 **보편적 표준 관행**이다.

Applying ImageNet normalization when using ImageNet-pretrained backbones
is the **universally accepted standard practice** in computer vision.

---

## 5. 적용하지 않았을 때의 결과 / Consequences of Not Applying

| 결과 / Consequence | 설명 / Description |
|---|---|
| 활성화 왜곡 / Activation distortion | 초기 레이어의 합성곱 필터가 잘못된 신호를 받음 / Early convolutional filters receive incorrect signals |
| 특징 추출 품질 저하 / Degraded feature quality | 사전 학습된 특징 표현이 신뢰 불가 상태 / Pretrained feature representations become unreliable |
| 수렴 저하 / Slow convergence | Fine-tuning 시 잘못된 초기화에서 시작하므로 수렴 지연 / Fine-tuning starts from degraded initialization |
| 무음 정확도 저하 / Silent accuracy degradation | 오류 없이 예측만 틀림 — 감지 어려움 / No errors thrown, only wrong predictions — hard to detect |

> ⚠️ 이것이 **SSOT-NM01** 위반의 실제 결과다.
> This is the real-world consequence of an **SSOT-NM01** violation.

---

## 6. 일관성 요구사항 / Consistency Requirements

학습, 추론, 평가 **세 단계 모두** 동일한 정규화가 적용되어야 한다.
어느 한 단계라도 다르면 **입력 분포 불일치(distribution shift)** 가 발생하여 정확도가 무음으로 저하된다.

Training, inference, and evaluation **must all apply identical normalization**.
Any discrepancy causes **distribution shift**, leading to silent accuracy degradation.

| 단계 / Stage | 파일 / File | 정규화 / Normalization | 상태 / Status |
|---|---|---|---|
| 학습 Phase 0 / Training Phase 0 | `data/dataset.py` | `_IMAGENET_NORMALIZE` | ✅ |
| 학습 Phase 2 / Training Phase 2 | `data/dataset.py` | `_IMAGENET_NORMALIZE` | ✅ |
| 추론 / Inference | `inference/predictor_inference.py` | `_IMAGENET_NORMALIZE` | ✅ |
| 평가 / Evaluation | `evaluation/evaluator_inference.py` | `_IMAGENET_NORMALIZE` (via Dataset) | ✅ |

> ⚠️ **미해소 / Open**: `_IMAGENET_NORMALIZE` 가 `dataset.py` 와 `predictor_inference.py` 두 곳에 정의되어 있다. SSOT 단일 출처 원칙 위반. 향후 `data/normalize.py` 로 통합 예정.
>
> `_IMAGENET_NORMALIZE` is currently defined in both `dataset.py` and `predictor_inference.py`. Violates SSOT single-source principle. Planned consolidation into `data/normalize.py`.

---

## 7. 기각된 대안 / Rejected Alternatives

### 대안 1 — 커스텀 통계 사용 / Custom Dataset Statistics

```python
# 기각 / Rejected
mean = compute_mean(grayspot_dataset)
std  = compute_std(grayspot_dataset)
```

**기각 이유 / Reason for rejection**:
- 데이터셋이 소규모(채널당 771~7,394장)여서 통계 신뢰도 낮음
- 사전 학습 가중치와의 분포 불일치 해결 불가
- 채널별로 서로 다른 통계가 생성되어 일관성 저해

### 대안 2 — `[0, 1]` 정규화만 적용 / Only [0,1] Normalization

```python
# 기각 / Rejected
tensor = tensor / 255.0  # mean/std 미적용
```

**기각 이유 / Reason for rejection**:
- 사전 학습 backbone 의 기대 입력 분포와 불일치
- SSOT-NM01 위반 상태 그대로 유지
- 실제로 이 상태가 `predictor_inference.py` 에서 2026-05-14 이전까지 존재했던 버그

---

## 8. 위반 이력 / Violation History

| 위반 / Violation | 위치 / Location | 발견 / Detected | 해소 / Resolved |
|---|---|---|---|
| SSOT-NM01: 추론 시 정규화 미적용 | `inference/predictor_inference.py` | 2026-05-14 | ✅ 2026-05-14 |
| SSOT-CS01: 평가 시 BGR→RGB 변환 | `evaluation/evaluator_inference.py` line 59 | 2026-05-14 | ✅ 2026-05-14 |

---
