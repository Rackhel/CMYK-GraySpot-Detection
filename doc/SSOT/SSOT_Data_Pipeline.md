# SSOT Data Pipeline — 데이터 파이프라인 / Data Pipeline

CMYK Grayspot Detection System 의 데이터 로딩·분할·전처리·증강에 관한 단일 진실 공급원.

This document is the authoritative reference for data loading, splitting, preprocessing, and augmentation in the CMYK Grayspot Detection System.

> **목적 / Purpose**: 데이터 파이프라인의 의미(semantic) 정의
> **역할 / Role**: "What" — 입력 분포, 분할 규칙, 색상 공간 규약 정의
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md)
> **last_updated**: 2026-06-01

---

## 0. 데이터 생산 파이프라인 흐름 / Data Production Pipeline Flow

```
RAW PDF/JPEG (원본 스캔)
    ↓ ROI 추출 / ROI Extraction     → data_set/roi/
    ↓ 채널별 독립 라벨링 / Per-Channel Independent Labeling
    ↓ 패치 추출 / Patch Extraction  → data_set/labeled/{ch}/{lv}/*.png
    ↓ [한 번만 실행] prepare_holdout.py
    │   └─ labeled/ → holdout/ 15% stratified 분리 (이후 절대 수정 금지)
    ↓ [선택] generate_synthetic.py
    │   └─ labeled/{ch}/{lv}/synthetic_*.png 생성 (holdout은 건드리지 않음)
    ↓ CSV 통합 / CSV Unification     → data_set/labels_master.csv
    ↓ ML 학습 / Training             → CMYKDataset / ContrastiveDataset
    ↓ [학습 완전 완료 후 딱 한 번] 최종 평가
        └─ evaluate.py --holdout     → holdout/ 기준 성능 보고
```

> **⚠️ 채널 독립 라벨링 필수 / Per-Channel Independent Labeling Required**: Y/M/C/K 채널은 각자 독립적으로 라벨링해야 한다.

---

## 1. 디렉토리 구조 / Directory Structure

```
data_set/
├── labeled/                ← 학습에만 사용 / Training use only
│   ├── Y/
│   │   ├── 0/   ← Level 0 (정상 / Normal)
│   │   │   ├── *.png
│   │   │   └── synthetic_*.png   ← generate_synthetic.py 생성 (선택)
│   │   ├── 1/ ~ 5/
│   ├── M/ / C/ / K/
└── holdout/                ← 최종 평가 전용 — 학습 중 절대 참조 금지
    ├── Y/
    │   ├── 0/ ~ 5/
    ├── M/ / C/ / K/
```

### 1.1 경로 패턴 / Path Pattern

```
data_set/labeled/{channel}/{level}/*.png          ← 학습 / Training
data_set/labeled/{channel}/{level}/synthetic_*.png ← 합성 (선택) / Synthetic (optional)
data_set/holdout/{channel}/{level}/*.png           ← 최종 평가 전용 / Final eval only
```

### 1.2 라벨 파일 / Label File

| 파일 | 형식 | 컬럼 | 상태 |
|---|---|---|---|
| `data_set/labels_master.csv` | long-format | `filepath`, `channel`, `level` | ✅ Canonical |
| `data_set/labels_v0.csv` | wide-format | `filename`, `C`, `M`, `Y`, `K` | ⚠️ Legacy |

### 1.3 관리 스크립트 / Management Scripts

| 스크립트 | 명령 | 역할 | 실행 횟수 |
|---|---|---|---|
| `prepare_dataset.py` | `python -m src.scripts.prepare_dataset` | ROI 패치 추출 + augment_dataset.py 호출 | 반복 가능 |
| `augment_dataset.py` | `python -m src.scripts.augment_dataset` | PRD v2 미달 레벨 증강 + CSV 갱신 | 반복 가능 |
| `prepare_holdout.py` | `python -m src.scripts.prepare_holdout` | labeled/ → holdout/ stratified 15% 분리 | **딱 한 번만** |
| `generate_synthetic.py` | `python -m src.scripts.generate_synthetic` | 희귀 클래스 합성 이미지 생성 | prepare_holdout 이후 반복 가능 |

### 1.4 config.json 키 / Config Keys

- `storage.labeled_dir` 🟢 — `dataset.py`에서 소비
- `storage.holdout_dir` 🟢 — `prepare_holdout.py`, `evaluate.py --holdout`에서 소비
- `data.channels` 🟢 — `run_baseline.py`, `run_phase0.py`에서 소비
- `data.num_levels` 🟢 — `grayspot_model.py`에서 소비 (Hard SSOT)
- `data.image_size` 🟢 — `dataset.py`에서 소비 (Hard SSOT)
- `data.split_ratios` 🟢 — `dataset.py` stratified split에서 소비

---

## 2. 데이터셋 클래스 / Dataset Classes

### 2.1 CMYKDataset — Phase 2 (Supervised)

| 속성 | 값 | 비고 |
|---|---|---|
| 입력 소스 | `data_set/labeled/{channel}/{level}/*.png` | RGB float32 (D-1 수정 이후) |
| 출력 형상 | `(3, 128, 128)` | C, H, W 순서 |
| 레이블 타입 | `int` [0, 5] | ordinal 6-class |
| 분할 | `train` / `val` / `test` / `holdout` | stratified (`data.split_ratios` 🟢) |
| 오버샘플링 | `phase2.oversample` 또는 `phase2.per_channel.{ch}.oversample` 🟢 | per-channel 오버라이드 가능 |
| 합성 제외 | `exclude_synthetic=True` 옵션 | 평가 시 synthetic_* 파일 제외 가능 |

**split 옵션 상세:**
- `"train"` / `"val"` / `"test"`: `labeled_dir` 안에서 stratified split
- `"holdout"`: `holdout_dir` 전체 로드 (분할 없음, prepare_holdout.py 실행 후 사용)

### 2.2 ContrastiveDataset — Phase 0 (SimCLR)

| 속성 | 값 | 비고 |
|---|---|---|
| 입력 소스 | `data_set/labeled/{channel}/{level}/*.png` | 전체 레벨 (라벨 불필요) |
| 출력 | `(view1, view2)` Tensor pair | augmented positive pair |
| 레이블 | ❌ 없음 | unsupervised |

### 2.3 _EvalDataset — 평가 전용

| 속성 | 값 | 비고 |
|---|---|---|
| 입력 소스 | `labeled_dir` + `labels_master.csv` | Evaluator 내부 전용 |
| 출력 | `(Tensor, int, str)` | 이미지, 레벨, 파일명 |
| 색상 공간 | **RGB** (D-1 수정 이후) | 학습 경로와 동일 |
| 정규화 | ImageNet mean/std (SSOT-NM01) | |

---

## 3. 전처리 파이프라인 / Preprocessing Pipeline

### 3.1 처리 순서 / Processing Order

```
cv2.imread (BGR uint8)
  → cv2.cvtColor(BGR2RGB)   ← D-1 수정 (2026-06-01)
  → resize (128×128)
  → float32 /255.0
  → augment_supervised (train split only)
  → tensor permute (H,W,C → C,H,W)
  → ImageNet normalize
  → DataLoader batching
```

### 3.2 색상 공간 규약 / Color Space Contract

> ⚠️ **SSOT-CS01 개정 (2026-06-01)**: 기존 BGR 유지 정책에서 **RGB 변환 적용**으로 변경.
> pretrained EfficientNet-B0 / ResNet-50이 RGB 입력을 가정하므로 학습·평가 모두 RGB 사용.
> **기존 BGR 기준으로 학습된 모델은 재학습 필요.**

| 단계 | 색상 공간 | 비고 |
|---|---|---|
| 파일 로드 | BGR (cv2.imread) | |
| `preprocess()` 내부 | BGR → **RGB** (cvtColor) | |
| `preprocess()` 출력 | **RGB** float32 [0, 1] | |
| 증강 후 | **RGB** float32 [0, 1] | |
| 텐서 정규화 후 | **RGB** float32, ImageNet-normalized | |
| 모델 입력 | **RGB** float32, ImageNet-normalized | pretrained backbone 요구사항 충족 |
| 추론 경로 | **RGB** float32, ImageNet-normalized | 학습과 동일 |

### 3.3 `preprocess()` 함수 시그니처

`preprocess(image: np.ndarray, image_size: int = 128) -> np.ndarray`

- Input: BGR uint8 `(H, W, 3)` ← cv2.imread 출력
- Internal: `cv2.cvtColor(BGR2RGB)` → resize → /255.0
- Output: **RGB** float32 `(128, 128, 3)` in `[0.0, 1.0]`

---

## 4. 증강 정책 / Augmentation Policy

### 4.1 학습 시 증강 (Runtime Augmentation) — Phase 2

`augment_supervised(image, aug_cfg)` — train split 전용

| 변환 | 파라미터 | config 키 | 상태 |
|---|---|---|---|
| 수평 뒤집기 | p=0.5 | `phase2.augmentation.flip_prob` | 🟢 |
| 수직 뒤집기 | p=0.0 (기본 꺼짐) | `phase2.augmentation.vflip_prob` | 🟢 |
| 랜덤 회전 | p=0.3, ±15° | `phase2.augmentation.rotation_prob`, `rotation_max` | 🟢 |
| 밝기 조절 | p=0.5, ±30/255 | `phase2.augmentation.brightness_prob`, `brightness_range` | 🟢 |
| 가산 노이즈 | p=0.5, 0~10/255 | `phase2.augmentation.noise_prob`, `noise_range` | 🟢 |
| **증강 강도 정책** | `"light"` / `"strong"` | `phase2.augmentation.policy` | 🟢 |

> `policy="strong"` 시 모든 확률이 ×1.3 적용 — K/Y 채널 per_channel 설정에서 자동 활성화.

### 4.2 배치 레벨 증강 (Batch-Level Augmentation)

DataLoader 이후 배치 텐서에 적용하는 선택적 증강:

| 함수 | 설명 | import |
|---|---|---|
| `mixup_batch(x, y, alpha, num_classes)` | 두 이미지 alpha 비율 혼합, soft label 반환 | `from data.augmentation import mixup_batch` |
| `cutmix_batch(x, y, alpha, num_classes)` | 직사각형 패치 교체, soft label 반환 | `from data.augmentation import cutmix_batch` |

### 4.3 채널별 증강 정책 / Per-Channel Augmentation Policy

`config.json`의 `phase2.per_channel.{ch}.augmentation.policy`로 채널별 독립 설정:

| 채널 | 권장 policy | 이유 |
|---|---|---|
| K | `"strong"` | Lv2: 8장 → 과적합 위험 극심 |
| Y | `"strong"` | Lv5: ~30장 → 과적합 위험 높음 |
| C | `"light"` | 적당한 불균형 |
| M | `"light"` | 비교적 균형적 |

### 4.4 합성 데이터 정책 / Synthetic Data Policy

`generate_synthetic.py`로 사전 생성, `labeled/`에 저장.

| 방법 | 파일명 패턴 | 설명 |
|---|---|---|
| Level Interpolation | `synthetic_{N:04d}.png` | Level 0 + Level N 보간 |
| Stable Diffusion img2img | `synthetic_{N:04d}.png` | 결함 강도 조건부 생성 |

> **원칙 / Principle**:
> - `prepare_holdout.py` 실행 이후에만 생성 가능 (holdout은 보강하지 않음)
> - 평가 시 `exclude_synthetic=True`로 합성 데이터 제외 가능
> - `synthetic_` prefix로 실제 데이터와 구분

---

## 5. 데이터 분할 규칙 / Split Rules

### 5.1 Three-Tier Split Strategy

```
전체 데이터 (100%)
├── Holdout Test (15%)  ← prepare_holdout.py로 한 번만 분리, 이후 잠금
│   └── 학습·검증·HPO 중 절대 사용 금지
└── Dev Pool (85%)      ← labeled/에 유지
    ├── Train (~70%)    ← CMYKDataset(split="train")
    └── Val  (~15%)     ← CMYKDataset(split="val") — early stopping, Optuna
```

### 5.2 분할 비율 / Split Ratios (Dev Pool 내부)

| 분할 | config 키 | 기본값 | 비고 |
|---|---|---|---|
| train | `data.split_ratios.train` 🟢 | 0.70 | labeled/ 기준 |
| val | `data.split_ratios.val` 🟢 | 0.15 | labeled/ 기준 |
| test | `data.split_ratios.test` 🟢 | 0.15 | labeled/ 기준 (중간 평가용) |

### 5.3 분할 방식

- Stratified split (클래스 비율 유지)
- Seed: `train.seed` 🟢 (기본값 42) — **seeded `random.Random` 인스턴스** 사용 (D-2 수정)
- Oversampling RNG도 동일 seeded 인스턴스 사용으로 재현성 보장

> **⚠️ SSOT-SD01**: 동일 seed에서 항상 동일한 분할이 생성되어야 한다.

---

## 6. DataLoader 설정

| 항목 | 학습 | 평가 | 추론 |
|---|---|---|---|
| `batch_size` | `phase2.batch_size` 🟢 | `inference.batch_size` 🟢 | caller 지정 |
| `shuffle` | train=True, else=False | False | False |
| `num_workers` | `train.num_workers` 🟢 | 0 | 0 |

---

## 7. 인터페이스 계약 / Interface Contracts

| 계약 | 타입 | 형상 | 값 범위 |
|---|---|---|---|
| Dataset 출력 (Phase 2) | `(Tensor, int)` | `(3, 128, 128)`, scalar | ImageNet-normalized RGB float32, [0,5] |
| Dataset 출력 (Phase 0) | `(Tensor, Tensor)` | `(3, 128, 128)` × 2 | ImageNet-normalized RGB float32 |
| MixUp 출력 | `(Tensor, Tensor)` | `(B,3,128,128)`, `(B, num_classes)` | mixed image, soft label |
| 모델 입력 | `Tensor` | `(B, 3, 128, 128)` | ImageNet-normalized RGB float32 |

---

## 8. Fail-Fast 규칙

| 조건 | SSOT 코드 | 등급 | 동작 |
|---|---|---|---|
| 데이터 디렉토리 미존재 | `SSOT-FF01` | Level 1 | 학습 중단 |
| 학습(RGB) ↔ 추론(BGR) 색상 공간 불일치 | `SSOT-CS01` | Level 1 | 결과 신뢰 불가 |
| ImageNet 정규화 미적용 | `SSOT-NM01` | Level 2 | 경고 + 계속 |
| 동일 seed에서 다른 데이터 분할 | `SSOT-SD01` | Level 2 | 경고 |
| holdout/ 학습 중 참조 시도 | `SSOT-HO01` | Level 1 | 즉시 중단 |

> **SSOT-HO01 (신규)**: holdout 디렉토리는 `evaluate.py --holdout` 실행 외 모든 경로에서 참조 금지.

---

## 변경 이력 / Changelog

| 날짜 | 변경 내용 |
|---|---|
| 2026-06-01 | **D-1**: SSOT-CS01 개정 — BGR 유지 → RGB 변환 적용으로 변경 (pretrained 호환성 수정) |
| 2026-06-01 | **D-2**: oversampling RNG를 전역 random → seeded 인스턴스로 교체 |
| 2026-06-01 | **A-1/A-2**: holdout 분리 전략 추가 (prepare_holdout.py, split="holdout") |
| 2026-06-01 | **B-1**: 합성 데이터 생성 정책 추가 (generate_synthetic.py, synthetic_* prefix) |
| 2026-06-01 | **C-3**: 증강 강화 — 수직 뒤집기, 랜덤 회전, policy 옵션, MixUp/CutMix 추가 |
