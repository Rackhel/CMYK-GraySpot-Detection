---
type: contract
domain: data_pipeline
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
---

# [Contract] Data Pipeline Boundary — 데이터 파이프라인 경계 계약

> **목적**: Dataset 클래스의 입출력 형상, 증강 함수 계약, DataLoader 배치 형태를 정의한다.
> **상태**: ✅ Accepted [Hard]
> **작성일**: 2026-05-17
> **관련 문서**:
>
> - [SSOT_Data_Pipeline.md](../SSOT/SSOT_Data_Pipeline.md) (데이터 파이프라인 정의)
> - [SSOT_Core.md](../SSOT/SSOT_Core.md) (BGR/ImageNet 불변량)

> 🔒 **SSOT 경계 원칙**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다.
> 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.

---

## 1. 계약 목적

파일 로드부터 모델 입력까지의 데이터 변환 경계에서 타입, 형상, 값 범위, 색상 공간 보장을 정의한다.

---

## 2. 전처리 경계 타입표

| 경계 | 타입 | 형상 | 값 범위 | 색상 공간 |
| --- | --- | --- | --- | --- |
| 파일 로드 후 | `np.ndarray` | `(H, W, 3)` | `[0, 255]` uint8 | **BGR** |
| `preprocess()` 출력 | `np.ndarray` | `(128, 128, 3)` | `[0.0, 1.0]` float32 | **BGR** |
| 증강 후 | `np.ndarray` | `(128, 128, 3)` | `[0.0, 1.0]` float32 | **BGR** |
| Tensor 변환 후 | `torch.Tensor` | `(3, 128, 128)` | `[0.0, 1.0]` float32 | **BGR** (CHW) |
| ImageNet 정규화 후 | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized | **BGR** (CHW) |

> ⚠️ **SSOT-CS01**: 파일 로드부터 모델 입력까지 BGR을 유지한다. RGB 변환 금지.

---

## 3. Dataset 출력 계약

### 3.1 `ContrastiveDataset` (Phase 0)

```python
view1, view2 = dataset[i]
```

| 출력 | 타입 | 형상 | 범위 |
| --- | --- | --- | --- |
| `view1` | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized |
| `view2` | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized |

> 동일 이미지에 독립 증강 2회 적용 → Positive pair.

### 3.2 `CMYKDataset` (Phase 2)

```python
image, label = dataset[i]
```

| 출력 | 타입 | 형상 | 범위 |
| --- | --- | --- | --- |
| `image` | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized |
| `label` | `int` | scalar | `[0, 5]` (ordinal 6-class) |

> train split만 증강 적용. val split은 증강 없음.

---

## 4. DataLoader 배치 계약

| Phase | 배치 타입 | 형상 |
| --- | --- | --- |
| Phase 0 | `(Tensor, Tensor)` | `(B, 3, 128, 128)`, `(B, 3, 128, 128)` |
| Phase 2 | `(Tensor, Tensor)` | `(B, 3, 128, 128)`, `(B,)` int |

---

## 5. 증강 함수 계약

### 5.1 `augment_supervised()`

```python
from data.augmentation import augment_supervised

aug_image = augment_supervised(
    image: np.ndarray,            # (H, W, 3) float32, [0.0, 1.0], BGR
    aug_cfg: Optional[dict] = None  # cfg["phase2"]["augmentation"]
) -> np.ndarray                   # 입력과 동일 형상·범위·색상 공간 유지
```

| 변환 | 파라미터 | 기본값 |
| --- | --- | --- |
| Random horizontal flip | `flip_prob` | 0.5 |
| Brightness jitter | `brightness_prob`, `brightness_range` | 0.5, 30 |
| Additive noise | `noise_prob`, `noise_range` | 0.5, 10 |

### 5.2 `augment_contrastive()`

```python
from data.augmentation import augment_contrastive

aug_image = augment_contrastive(
    image: np.ndarray,              # (H, W, 3) float32, [0.0, 1.0], BGR
    image_size: int,                # 출력 크기 (e.g. 128)
    aug_cfg: Optional[dict] = None  # cfg["phase0"]["augmentation"]
) -> np.ndarray                     # (image_size, image_size, 3) float32 [0,1] BGR
```

| 변환 | 파라미터 |
| --- | --- |
| Random horizontal flip | `flip_prob` |
| Random crop + resize | `crop_prob`, `crop_scale_min`, `crop_scale_max` |
| Brightness jitter | `blur_prob`, `color_jitter` |
| Contrast jitter | `blur_prob`, `contrast_scale_min`, `contrast_scale_max` |
| Gaussian blur | `blur_prob`, `blur_kernels` |

> ⚠️ **SSOT-CS01 준수**: 두 증강 함수 모두 BGR 색상 공간을 유지한다. RGB 변환 없음.

---

## 6. 금지 패턴

```python
# ❌ RGB 변환 — BGR 일관성 위반 (SSOT-CS01)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

```python
# ❌ val split에 증강 적용
if split == "val":
    image = augment_supervised(image, aug_cfg)  # 금지
```

```python
# ✅ 올바른 패턴
img = cv2.imread(path)  # BGR 그대로 사용
image = image / 255.0   # float32 [0,1]
```

---

## 7. 체크리스트

- [x] 전처리 전 경로에서 BGR 유지 확인
- [x] ImageNet 정규화 학습/추론 동일 적용
- [x] val split 증강 미적용 확인
- [ ] `_EvalDataset`에 ImageNet 정규화 적용 (N-01)

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_Data_Pipeline.md](../SSOT/SSOT_Data_Pipeline.md) | 데이터 파이프라인 정의 (What) |
| [Contract_model_boundary.md](Contract_model_boundary.md) | 모델 입력 형상 계약 |
| [Contract_fail_fast.md](Contract_fail_fast.md) | SSOT-CS01 정의 |
