---
type: contract
domain: data_pipeline
status: Active
last_updated: 2026-06-01
owner: CMYK WooSong Team
---

# [Contract] Data Pipeline Boundary — 데이터 파이프라인 경계 계약

> **목적 / Purpose**: Dataset 클래스의 입출력 형상, 증강 함수 계약, DataLoader 배치 형태를 정의한다.
> **상태 / Status**: ✅ Accepted [Hard]
> **관련 문서**:
> - [SSOT_Data_Pipeline.md](../SSOT/SSOT_Data_Pipeline.md)
> - [SSOT_Core.md](../SSOT/SSOT_Core.md)

> 🔒 **SSOT 경계 원칙**: 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.

---

## 변경 이력 / Changelog

| 날짜 | 항목 | 내용 |
|---|---|---|
| 2026-06-01 | D-1 | **SSOT-CS01 개정** — BGR 유지 → **RGB 변환** 적용으로 변경 |
| 2026-06-01 | D-2 | oversampling RNG 전역 random → seeded 인스턴스로 교체 |
| 2026-06-01 | A-2 | CMYKDataset `split="holdout"`, `exclude_synthetic` 옵션 추가 |
| 2026-06-01 | C-3 | augment_supervised 변환 추가 (수직 뒤집기, 회전, policy, MixUp, CutMix) |

---

## 1. 계약 목적 / Contract Purpose

파일 로드부터 모델 입력까지의 데이터 변환 경계에서 타입, 형상, 값 범위, 색상 공간 보장을 정의한다.

---

## 2. 전처리 경계 타입표 / Preprocessing Boundary Type Table

> ⚠️ **SSOT-CS01 개정 (2026-06-01)**: 기존 BGR 유지 정책 → **RGB 변환** 으로 변경.
> pretrained EfficientNet-B0 / ResNet-50이 RGB 입력을 가정하기 때문.
> **BGR 기준으로 학습된 모든 기존 모델은 재학습 필요.**

| 경계 | 타입 | 형상 | 값 범위 | 색상 공간 |
|---|---|---|---|---|
| 파일 로드 후 | `np.ndarray` | `(H, W, 3)` | `[0, 255]` uint8 | **BGR** (cv2.imread) |
| `preprocess()` 내부 변환 후 | `np.ndarray` | `(H, W, 3)` | `[0, 255]` uint8 | **RGB** (cvtColor 적용) |
| `preprocess()` 출력 | `np.ndarray` | `(128, 128, 3)` | `[0.0, 1.0]` float32 | **RGB** |
| 증강 후 | `np.ndarray` | `(128, 128, 3)` | `[0.0, 1.0]` float32 | **RGB** |
| Tensor 변환 후 | `torch.Tensor` | `(3, 128, 128)` | `[0.0, 1.0]` float32 | **RGB** (CHW) |
| ImageNet 정규화 후 | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized | **RGB** (CHW) |

---

## 3. Dataset 출력 계약 / Dataset Output Contract

### 3.1 `ContrastiveDataset` (Phase 0)

```python
view1, view2 = dataset[i]
```

| 출력 | 타입 | 형상 | 범위 |
|---|---|---|---|
| `view1` | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized RGB |
| `view2` | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized RGB |

### 3.2 `CMYKDataset` (Phase 2)

```python
image, label = dataset[i]
```

| 출력 | 타입 | 형상 | 범위 |
|---|---|---|---|
| `image` | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized RGB |
| `label` | `int` | scalar | `[0, 5]` (ordinal 6-class) |

**split 옵션:**

| split | 데이터 소스 | 용도 |
|---|---|---|
| `"train"` | `labeled_dir` — stratified 70% | 학습 (증강 적용) |
| `"val"` | `labeled_dir` — stratified 15% | Early stopping, Optuna |
| `"test"` | `labeled_dir` — stratified 15% | 중간 평가 |
| `"holdout"` | `holdout_dir` — 전체 | **최종 평가 전용** — 학습 중 사용 금지 |

**선택적 파라미터:**

```python
CMYKDataset(cfg, channel, split="train",
            exclude_synthetic=False)  # True: synthetic_* 파일 제외
```

---

## 4. DataLoader 배치 계약 / DataLoader Batch Contract

| Phase | 배치 타입 | 형상 |
|---|---|---|
| Phase 0 | `(Tensor, Tensor)` | `(B, 3, 128, 128)`, `(B, 3, 128, 128)` |
| Phase 2 | `(Tensor, Tensor)` | `(B, 3, 128, 128)`, `(B,)` int |
| Phase 2 + MixUp | `(Tensor, Tensor)` | `(B, 3, 128, 128)`, `(B, num_classes)` soft label |
| Phase 2 + CutMix | `(Tensor, Tensor)` | `(B, 3, 128, 128)`, `(B, num_classes)` soft label |

---

## 5. 증강 함수 계약 / Augmentation Function Contract

### 5.1 `augment_supervised()`

```python
from data.augmentation import augment_supervised

aug_image = augment_supervised(
    image: np.ndarray,            # (H, W, 3) float32, [0.0, 1.0], RGB
    aug_cfg: Optional[dict] = None  # cfg["phase2"]["augmentation"]
) -> np.ndarray                   # 입력과 동일 형상·범위·색상 공간 유지
```

| 변환 | 파라미터 | 기본값 |
|---|---|---|
| Random horizontal flip | `flip_prob` | 0.5 |
| Random vertical flip | `vflip_prob` | 0.0 (기본 꺼짐) |
| Random rotation | `rotation_prob`, `rotation_max` | 0.3, 15° |
| Brightness jitter | `brightness_prob`, `brightness_range` | 0.5, 30 |
| Additive noise | `noise_prob`, `noise_range` | 0.5, 10 |
| Augmentation policy | `policy` | `"light"` (`"strong"` 시 ×1.3 확률) |

### 5.2 `augment_contrastive()`

```python
from data.augmentation import augment_contrastive

aug_image = augment_contrastive(
    image: np.ndarray,              # (H, W, 3) float32, [0.0, 1.0], RGB
    image_size: int,
    aug_cfg: Optional[dict] = None  # cfg["phase0"]["augmentation"]
) -> np.ndarray                     # (image_size, image_size, 3) float32 [0,1] RGB
```

### 5.3 `mixup_batch()` / `cutmix_batch()`

```python
from data.augmentation import mixup_batch, cutmix_batch

mixed_x, mixed_y = mixup_batch(
    x: torch.Tensor,    # (B, C, H, W) — DataLoader 배치 이후 적용
    y: torch.Tensor,    # (B,) int labels
    alpha: float = 0.2,
    num_classes: int = 6
) -> (Tensor, Tensor)   # (B,C,H,W), (B, num_classes) soft labels
```

- `mixup_batch`: lambda × img_a + (1-lambda) × img_b, soft label 반환
- `cutmix_batch`: 직사각형 영역 교체, 영역 비율 기반 soft label 반환
- 두 함수 모두 `CrossEntropyLoss`가 아닌 **soft label 지원 손실 함수** 필요 (또는 `(mixed_y.argmax(1)`로 hard label 변환)

---

## 6. 금지 패턴 / Prohibited Patterns

```python
# ❌ preprocess() 후 BGR 그대로 사용 — SSOT-CS01 위반
# (2026-06-01 이전 코드 패턴 — 더 이상 유효하지 않음)
img = cv2.imread(path)
image = img.astype(np.float32) / 255.0  # BGR 그대로 — 금지

# ✅ 올바른 패턴 (preprocess() 내부에서 자동 처리)
image = preprocess(img, image_size)  # BGR→RGB, resize, /255.0 모두 처리됨
```

```python
# ❌ val split에 증강 적용
if split == "val":
    image = augment_supervised(image, aug_cfg)  # 금지

# ❌ 학습 중 holdout 참조 (SSOT-HO01)
dataset = CMYKDataset(cfg, "Y", split="holdout")  # 학습 루프 내 사용 금지

# ✅ holdout은 최종 평가에서만
# python -m src.scripts.evaluate --channel Y --holdout
```

```python
# ❌ synthetic_ 파일이 평가 지표에 포함
eval_ds = CMYKDataset(cfg, "Y", split="test")  # synthetic_ 포함됨

# ✅ 실제 데이터 기준 평가
eval_ds = CMYKDataset(cfg, "Y", split="test", exclude_synthetic=True)
```

---

## 7. 체크리스트 / Checklist

- [x] ~~BGR 유지 확인~~ → **RGB 변환 적용 확인** (D-1, 2026-06-01)
- [x] preprocess()에서 cv2.cvtColor(BGR2RGB) 적용
- [x] _EvalDataset에서 동일 RGB 변환 적용
- [x] oversampling RNG seeded 인스턴스 사용 (D-2)
- [x] CMYKDataset split="holdout" 옵션 추가 (A-2)
- [x] CMYKDataset exclude_synthetic 옵션 추가 (B-2)
- [x] ImageNet 정규화 학습/추론 동일 적용 (SSOT-NM01)
- [x] val split 증강 미적용 확인
- [x] augment_supervised 수직 뒤집기, 회전 추가 (C-3)
- [x] mixup_batch / cutmix_batch 함수 추가 (C-3)

---

## See Also

| 문서 | 관계 |
|---|---|
| [SSOT_Data_Pipeline.md](../SSOT/SSOT_Data_Pipeline.md) | 데이터 파이프라인 정의 |
| [Contract_model_boundary.md](Contract_model_boundary.md) | 모델 입력 형상 계약 |
| [Contract_fail_fast.md](Contract_fail_fast.md) | SSOT-CS01, SSOT-HO01 정의 |
