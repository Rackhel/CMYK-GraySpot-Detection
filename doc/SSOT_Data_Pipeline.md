# SSOT Data Pipeline — 데이터 파이프라인 / Data Pipeline

CMYK Grayspot Detection System 의 데이터 로딩·분할·전처리·증강에 관한 단일 진실 공급원.

This document is the authoritative reference for data loading, splitting, preprocessing, and augmentation in the CMYK Grayspot Detection System.

> **목적 / Purpose**: 데이터 파이프라인의 의미(semantic) 정의
> **역할 / Role**: "What" — 입력 분포, 분할 규칙, 색상 공간 규약 정의
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md)

---

## Table of Contents / 목차

1. [디렉토리 구조 / Directory Structure](#1-디렉토리-구조--directory-structure)
2. [데이터셋 클래스 / Dataset Classes](#2-데이터셋-클래스--dataset-classes)
3. [전처리 파이프라인 / Preprocessing Pipeline](#3-전처리-파이프라인--preprocessing-pipeline)
4. [증강 정책 / Augmentation Policy](#4-증강-정책--augmentation-policy)
5. [데이터 분할 규칙 / Split Rules](#5-데이터-분할-규칙--split-rules)
6. [DataLoader 설정 / DataLoader Settings](#6-dataloader-설정--dataloader-settings)
7. [인터페이스 계약 / Interface Contracts](#7-인터페이스-계약--interface-contracts)
8. [Fail-Fast 규칙 / Fail-Fast Rules](#8-fail-fast-규칙--fail-fast-rules)
9. [SSOT 위반 현황 / Violations](#9-ssot-위반-현황--violations)

---

## 1. 디렉토리 구조 / Directory Structure

```
data_set/
└── labeled/
    ├── Y/
    │   ├── 0/   ← Level 0 (정상 / Normal)
    │   │   └── *.png
    │   ├── 1/   ← Level 1
    │   ├── 2/   ← Level 2
    │   ├── 3/   ← Level 3
    │   ├── 4/   ← Level 4
    │   └── 5/   ← Level 5 (최대 결함 / Maximum defect)
    ├── M/
    ├── C/
    └── K/
```

### 1.1 경로 패턴 / Path Pattern

```
data_set/labeled/{channel}/{level}/*.png
```

| 필드 / Field | 값 / Values | 설명 / Description |
|---|---|---|
| `{channel}` | `Y`, `M`, `C`, `K` | CMYK 색상 채널 / Color channel |
| `{level}` | `0`, `1`, `2`, `3`, `4`, `5` | Grayspot 결함 수준 / Defect level |

### 1.2 config.json 키 / Config Keys

```json
"data": {
  "channels":   ["Y", "M", "C", "K"],
  "num_levels": 6,
  "image_size": 128,
  "split_ratios": { "train": 0.70, "val": 0.15, "test": 0.15 },
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225]
  }
},
"storage": {
  "labeled_dir": "data_set/labeled"
}
```

- `storage.labeled_dir` 🟢 — `dataset.py`에서 소비 / consumed in `dataset.py`
- `data.channels` 🟢 — `run_baseline.py`, `run_phase0.py`에서 소비 / consumed in `run_baseline.py`, `run_phase0.py`
- `data.num_levels` 🟢 — `grayspot_model.py`에서 소비 (Hard SSOT) / consumed in `grayspot_model.py`
- `data.image_size` 🟢 — `dataset.py`에서 소비 (Hard SSOT) / consumed in `dataset.py`
- `data.split_ratios` 🟢 — `dataset.py` stratified split에서 소비 / consumed in `dataset.py` stratified split

---

## 2. 데이터셋 클래스 / Dataset Classes

### 2.1 CMYKDataset — Phase 2 (Supervised)

```python
from data import CMYKDataset

ds = CMYKDataset(cfg, channel="Y", split="train")
image, label = ds[0]
# image: Tensor (3, 128, 128) float32, ImageNet-normalized (mean/std subtracted)
# label: int  [0, 5]
```

| 속성 / Attribute | 값 / Value | 비고 / Note |
|---|---|---|
| 입력 소스 / Input | `data_set/labeled/{channel}/{level}/*.png` | BGR → float32 |
| 출력 형상 / Output shape | `(3, 128, 128)` | C, H, W 순서 / C, H, W order |
| 레이블 타입 / Label type | `int` [0, 5] | ordinal 6-class |
| 분할 / Split | `train` / `val` / `test` | stratified (`data.split_ratios` 🟢) |
| 오버샘플링 / Oversample | `phase2.oversample` 🟢 | 클래스 불균형 보정 / Class imbalance correction |

### 2.2 ContrastiveDataset — Phase 0 (SimCLR)

```python
from data import ContrastiveDataset

ds = ContrastiveDataset(cfg, channel="Y")
view1, view2 = ds[0]
# view1, view2: Tensor (3, 128, 128) float32, ImageNet-normalized
# Positive pair: same image with different augmentations
```

| 속성 / Attribute | 값 / Value | 비고 / Note |
|---|---|---|
| 입력 소스 / Input | `data_set/labeled/{channel}/{level}/*.png` | 전체 레벨 수집 (라벨 불필요) / All levels collected (label not needed) |
| 출력 / Output | `(view1, view2)` Tensor pair | augmented positive pair |
| 레이블 / Label | ❌ 없음 / None | unsupervised |
| 증강 설정 / Aug config | `phase0.augmentation` 🟢 | `aug_cfg`로 전달 / passed as `aug_cfg` |

---

## 3. 전처리 파이프라인 / Preprocessing Pipeline

### 3.1 처리 순서 / Processing Order

```
cv2.imread(path)                                 # BGR uint8 (H, W, 3)
    → preprocess(image, image_size=128)
        → cv2.resize((128, 128))                 # 크기 통일 / Uniform size (128×128)
        → .astype(np.float32) / 255.0           # [0, 255] → [0.0, 1.0]
    → [augment_supervised(image)]                # train split only (Phase 2) / train split 전용
    → torch.tensor(image).permute(2,0,1).float() # (H,W,C) → (C,H,W)
    → _IMAGENET_NORMALIZE(tensor)               # ImageNet mean/std normalization
    → DataLoader batching
```

### 3.2 색상 공간 규약 / Color Space Contract

| 단계 / Stage | 색상 공간 / Color Space | 비고 / Note |
|---|---|---|
| 파일 로드 / File load | BGR (OpenCV `cv2.imread`) | Hard SSOT |
| 전처리 후 / After preprocess | BGR float32 [0, 1] | 변환 없음 / No conversion |
| 텐서 정규화 후 / After normalize | BGR float32, ImageNet-normalized | `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` |
| 모델 입력 / Model input | BGR float32, ImageNet-normalized | ✅ pretrained backbone 호환 |
| 추론 / Inference | BGR float32, ImageNet-normalized | **학습과 일치 필수** / Must match training |

> **⚠️ SSOT-CS01**: 학습과 평가에서 동일 색상 공간(BGR)을 유지해야 한다.
> Training and evaluation **must** use the same color space (BGR).

> ✅ **SSOT-NM01 해소됨 / Resolved**: `dataset.py`의 `_IMAGENET_NORMALIZE`가 `CMYKDataset`과 `ContrastiveDataset` 모두에 적용된다.
> `_IMAGENET_NORMALIZE` in `dataset.py` is applied in both `CMYKDataset` and `ContrastiveDataset`.

### 3.3 `preprocess()` 함수 시그니처 / Function Signature

```python
from data import preprocess

image_tensor = preprocess(image: np.ndarray, image_size: int = 128) -> np.ndarray
# Input:  BGR uint8 (H, W, 3)
# Output: BGR float32 (128, 128, 3) in [0.0, 1.0]
```

---

## 4. 증강 정책 / Augmentation Policy

### 4.1 Supervised 증강 / Supervised Augmentation

Phase 2 학습 전용 (train split only) / For Phase 2 training only (train split only):

```python
from data import augment_supervised

augmented = augment_supervised(image: np.ndarray, aug_cfg: dict = None) -> np.ndarray
# Input/Output: BGR float32 (128, 128, 3) [0, 1]
# aug_cfg = cfg["phase2"]["augmentation"]  — CMYKDataset 에서 자동 전달 / auto-passed from CMYKDataset
```

| 변환 / Transform | 파라미터 / Parameters | config 키 / Key | 소비 여부 / Status |
|---|---|---|---|
| 수평 뒤집기 / Horizontal flip | p=0.5 | `phase2.augmentation.flip_prob` | 🟢 |
| 밝기 조절 / Brightness jitter | p=0.5, delta ±30/255 | `phase2.augmentation.brightness_prob`, `brightness_range` | 🟢 |
| 가산 노이즈 / Additive noise | p=0.5, 0~10/255 | `phase2.augmentation.noise_prob`, `noise_range` | 🟢 |

> 모든 Supervised 증강 파라미터가 `phase2.augmentation.*` config에서 소비됨. / All Supervised augmentation params are consumed from `phase2.augmentation.*` config.
> 모듈 상수(`_SUP_*`)는 fallback 기본값으로만 유지. / Module constants (`_SUP_*`) remain as fallback defaults only.
>
> All Supervised augmentation params are consumed from `phase2.augmentation.*` config.
> Module constants (`_SUP_*`) remain as fallback defaults only.

### 4.2 Contrastive 증강 / Contrastive Augmentation

Phase 0 SimCLR용 강한 증강 / Strong augmentation for Phase 0 SimCLR:

```python
from data import augment_contrastive

view = augment_contrastive(image: np.ndarray, image_size: int, aug_cfg: dict) -> np.ndarray
# Two independent calls produce the positive pair (view1, view2)
# aug_cfg = cfg["phase0"]["augmentation"]
```

| 변환 / Transform | 파라미터 / Parameters | config 키 / Key | 소비 여부 / Status |
|---|---|---|---|
| 수평 뒤집기 / Horizontal flip | p=0.5 | `phase0.augmentation.flip_prob` | 🟢 |
| 랜덤 크롭 + 리사이즈 / Random crop + resize | p=0.5, scale 0.6~1.0 | `phase0.augmentation.crop_prob`, `crop_scale_min`, `crop_scale_max` | 🟢 |
| 밝기 조절 / Brightness jitter | p=`blur_prob`, delta ±`color_jitter` | `phase0.augmentation.color_jitter`, `blur_prob` | 🟢 |
| 대비 조절 / Contrast jitter | p=`blur_prob`, scale 0.8~1.2 | `phase0.augmentation.contrast_scale_min`, `contrast_scale_max` | 🟢 |
| 가우시안 블러 / Gaussian blur | p=`blur_prob`, kernel [3, 5] | `phase0.augmentation.blur_prob`, `blur_kernels` | 🟢 |

> 모든 Contrastive 증강 파라미터가 `phase0.augmentation.*` config에서 소비됨. / All Contrastive augmentation params are consumed from `phase0.augmentation.*` config.
> 모듈 상수(`_CON_*`)는 fallback 기본값으로만 유지. / Module constants (`_CON_*`) remain as fallback defaults only.
>
> All Contrastive augmentation params are consumed from `phase0.augmentation.*` config.
> Module constants (`_CON_*`) remain as fallback defaults only.

---

## 5. 데이터 분할 규칙 / Split Rules

### 5.1 분할 비율 / Split Ratios

| 분할 / Split | config 키 / Key | 기본값 / Default | 비고 / Note |
|---|---|---|---|
| train | `data.split_ratios.train` 🟢 | 0.70 | dataset.py 소비 / consumed in dataset.py |
| val | `data.split_ratios.val` 🟢 | 0.15 | dataset.py 소비 / consumed in dataset.py |
| test | `data.split_ratios.test` 🟢 | 0.15 | dataset.py 소비 / consumed in dataset.py |

### 5.2 분할 방식 / Split Method

- Stratified split (클래스 비율 유지 / Class proportion preserved) — 코드에서 항상 적용 / always applied in code
- Seed: `train.seed` 🟢 (기본값 / default 42) — 재현성 보장 / Reproducibility guaranteed

> **⚠️ SSOT-SD01**: 동일 seed에서 항상 동일한 분할이 생성되어야 한다.
> The same seed must always produce the same split.

---

## 6. DataLoader 설정 / DataLoader Settings

컨텍스트별 DataLoader 파라미터가 다르며, 특히 `batch_size`와 `num_workers`의 출처가 다르다.
DataLoader parameters differ by context — note the difference in `batch_size` and `num_workers` sources.

| 항목 / Item | `run_phase2.py` (학습 / Training) | `Evaluator` (평가 / Evaluation) | `GrayspotPredictor` (추론 / Inference) |
|---|---|---|---|
| `batch_size` | `phase2.batch_size` 🟢 (config) | `inference.batch_size` 🟢 (config, 기본 32) | 호출자 지정 / caller-specified |
| `shuffle` | train=True, val/test=False | False | False |
| `num_workers` | `train.num_workers` 🟢 (min, cpu_count) | 0 🟡 하드코딩 / Hardcoded | 0 🟡 하드코딩 / Hardcoded |
| `pin_memory` | `train.pin_memory` 🟢 (config) | device==cuda 🟡 하드코딩 / Hardcoded | — |
| `drop_last` | `train.drop_last` 🟢 (config, 기본 false / default false) | — | — |
| `persistent_workers` | `train.persistent_workers` 🟢 | — | — |

> 평가·추론의 `num_workers=0` 하드코딩은 성능 최적화 여지가 있으나 현재 의도적 기본값이다.
> Hardcoded `num_workers=0` in evaluation/inference leaves performance on the table but is the current intentional default.

---

## 7. 인터페이스 계약 / Interface Contracts

| 계약 / Contract | 타입 / Type | 형상 / Shape | 값 범위 / Range |
|---|---|---|---|
| Dataset 출력 / Output (Phase 2) | `(Tensor, int)` | `(3, 128, 128)`, scalar | ImageNet-normalized float32, label [0, 5] |
| Dataset 출력 / Output (Phase 0) | `(Tensor, Tensor)` | `(3, 128, 128)`, `(3, 128, 128)` | ImageNet-normalized float32 |
| 모델 입력 / Model input | `Tensor` | `(B, 3, 128, 128)` | ImageNet-normalized float32 |
| 평가 입력 / Eval input | `np.ndarray` | `(N,)` | int [0, 5] |

---

## 8. Fail-Fast 규칙 / Fail-Fast Rules

데이터 파이프라인에서 발동되는 SSOT 검증 코드 목록. 자세한 정의는 [SSOT_Validation_Codes.md](SSOT_Validation_Codes.md) 참조.
SSOT validation codes triggered within the data pipeline. See [SSOT_Validation_Codes.md](SSOT_Validation_Codes.md) for full definitions.

| 조건 / Condition | SSOT 코드 / Code | 등급 / Level | 동작 / Action |
|---|---|---|---|
| 데이터 디렉토리 미존재 / Data directory not found | `SSOT-FF01` | Level 1 — Error | 학습 중단 / Abort training |
| 학습(BGR) ↔ 추론(RGB) 색상 공간 불일치 / Color space mismatch | `SSOT-CS01` | Level 1 — Error | 결과 신뢰 불가 / Results unreliable |
| ImageNet 정규화 미적용 / ImageNet normalization missing | `SSOT-NM01` | Level 2 — Warning | 경고 출력 + 계속 / Warn and continue |
| 동일 seed에서 다른 데이터 분할 / Non-deterministic split | `SSOT-SD01` | Level 2 — Warning | 경고 출력 / Log warning |

---

## 9. SSOT 위반 현황 / Violations

| 코드 / Code | 위반 내용 / Violation | 등급 / Level | 상태 / Status |
|---|---|---|---|
| SSOT-CS01 | `evaluator_inference.py` `_EvalDataset.__getitem__()` 에서 `cv2.cvtColor(BGR→RGB)` 적용 → 학습(BGR)과 불일치 / BGR→RGB conversion in `_EvalDataset` caused mismatch with training | Level 1 | ✅ **해소됨 / Resolved** — `evaluator_inference.py` line 59 제거 (2026-05-14) |
| SSOT-NM01 | ImageNet 정규화 미적용 — pretrained backbone 성능 저하 가능 / ImageNet normalization not applied — may degrade pretrained backbone performance | Level 2 | ✅ **해소됨 / Resolved** (dataset.py 2026-05-08, predictor_inference.py 2026-05-14) |
| `_IMAGENET_NORMALIZE` 이중 정의 | `dataset.py` 와 `predictor_inference.py` 각각 정의 — 단일 출처 원칙 위반 / Defined in both files — violates single-source principle | Level 2 | ⚠️ **미해소 / Unresolved** — 향후 `data/normalize.py` 모듈로 통합 예정 / Planned consolidation into `data/normalize.py` |

> ✅ **해소됨 / Resolved**: SSOT-CS01 — `evaluator_inference.py` `_EvalDataset.__getitem__()` 의 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 제거. BGR 색상 공간 유지 (2026-05-14).
> ✅ **해소됨 / Resolved**: `data.split_ratios` Dead Config → `dataset.py`에서 소비 완료.
> ✅ **해소됨 / Resolved**: `phase0.augmentation.color_jitter` / `blur_prob` → `augment_contrastive(aug_cfg)` 소비 완료.
> ✅ **해소됨 / Resolved**: `phase0.augmentation` 나머지 키 (flip/crop/contrast/blur_kernels) → `augment_contrastive()` 소비 완료.
> ✅ **해소됨 / Resolved**: `phase2.augmentation.*` 신규 추가 → `CMYKDataset.sup_aug_cfg` 경유 `augment_supervised()` 소비 완료.
> ✅ **해소됨 / Resolved**: SSOT-NM01 — `_IMAGENET_NORMALIZE` 적용 완료 (`dataset.py` 2026-05-08, `predictor_inference.py` 2026-05-14 — 학습·추론 정규화 완전 일치 / training-inference normalization fully aligned).

---
