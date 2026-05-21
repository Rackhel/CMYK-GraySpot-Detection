---
type: contract
domain: inference_boundary
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] Inference Boundary — 추론 경계 계약 / Inference Boundary Contract

> **목적 / Purpose**: `GrayspotPredictor`의 공개 API 입출력, Mixin 구조, Fail-Fast 조건을 정의한다. / Defines the public API I/O, Mixin structure, and Fail-Fast conditions for `GrayspotPredictor`.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-17
> **관련 문서 / Related Docs**:
>
> - [SSOT_Model_Architecture.md](../SSOT/SSOT_Model_Architecture.md) (모델 구조 / Model architecture)
> - [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) (체크포인트 로드 / Checkpoint loading)

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.
> / This document does not redefine SSOT semantic definitions. Follow SSOT_Core.md as the final authority for semantic interpretation.

---

## 1. 계약 목적 / Contract Purpose

`GrayspotPredictor`의 모델 로드, 단일/배치 추론, 캐시 관리의 인터페이스 규약을 정의한다.

Defines the interface contracts for `GrayspotPredictor`'s model loading, single/batch inference, and cache management.

---

## 2. 모듈 트리 / Module Tree

```
inference/
├── predictor_device.py    — DeviceMixin      (장치 감지·설정 / device detection and setup)
├── predictor_loader.py    — ModelLoaderMixin (모델 로딩·캐시 / model loading and caching)
├── predictor_inference.py — InferenceMixin   (추론 실행 / inference execution)
└── predictor.py           — GrayspotPredictor (Orchestrator)
```

---

## 3. `GrayspotPredictor` 공개 API / GrayspotPredictor Public API

```python
from inference.predictor import GrayspotPredictor

predictor = GrayspotPredictor(config_path=None)
predictor.load_model(channel="Y", model_path=None)
result    = predictor.predict(images, channel="Y", batch_size=32, return_confidences=True)
results   = predictor.predict_batch(images_dict, batch_size=32)
predictor.clear_cache(channel=None)
info      = predictor.get_model_info(channel=None)
```

| 메서드 / Method | 모듈 / Module | Mixin | 설명 / Description |
| --- | --- | --- | --- |
| `__init__` | `inference.predictor` | Orchestrator | config 로딩, 장치 설정, 캐시 초기화 / Config loading, device setup, cache init |
| `load_model` | `inference.predictor_loader` | `ModelLoaderMixin` | 채널별 모델 로드 및 캐시 / Per-channel model load and cache |
| `predict` | `inference.predictor_inference` | `InferenceMixin` | 단일 채널 배치 추론 / Single-channel batch inference |
| `predict_batch` | `inference.predictor_inference` | `InferenceMixin` | 멀티 채널 배치 추론 / Multi-channel batch inference |
| `clear_cache` | `inference.predictor_loader` | `ModelLoaderMixin` | 모델 캐시 비우기 / Clear model cache |
| `get_model_info` | `inference.predictor_loader` | `ModelLoaderMixin` | 로드된 모델 정보 조회 / Query loaded model info |

---

## 4. `__init__` 입출력 / `__init__` I/O

```python
GrayspotPredictor(config_path: Optional[str | Path] = None) -> GrayspotPredictor
```

| 항목 / Item | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `config_path` | `Optional[str \| Path]` | None 이면 기본 config.json 경로 사용 / Uses default config.json path if None |
| **실패 조건 / Failure Condition** | `FileNotFoundError` | config.json 없음 — Fail-Fast (fallback 없음) / config.json missing — no fallback |

---

## 5. `load_model` 입출력 / load_model I/O

```python
load_model(channel: str, model_path: Optional[str | Path] = None) -> None
```

| 항목 / Item | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `channel` | `str` | `"Y" \| "M" \| "C" \| "K"` (대소문자 무관 / case-insensitive) |
| `model_path` | `Optional[str \| Path]` | None 이면 `storage.models_dir/best_{channel}.pt` 자동 탐색 / Auto-resolves if None |
| **실패 조건 / Failure Condition** | `ValueError` | 지원하지 않는 채널 / Unsupported channel |
| **실패 조건 / Failure Condition** | `FileNotFoundError` | 모델 파일 없음 — `SSOT-FF01` / Model file missing |

---

## 6. `predict` 입출력 / predict I/O

```python
predict(
    images: np.ndarray,           # (N, H, W, 3) or (N, H, W) uint8 or float32, BGR
    channel: str,                 # "Y" | "M" | "C" | "K"
    batch_size: int = 32,
    return_confidences: bool = True,
) -> Dict[str, np.ndarray]
```

| 반환 키 / Return Key | 형상 / Shape | 타입 / Type | 설명 / Description |
| --- | --- | --- | --- |
| `predictions` | `(N,)` | `int64` | 예측 클래스 / Predicted class [0-5] |
| `logits` | `(N, 6)` | `float32` | 원시 로짓 / Raw logits |
| `probabilities` | `(N, 6)` | `float32` | Softmax 확률 / Softmax probabilities |
| `confidences` | `(N,)` | `float32` | Max-softmax 신뢰도 (`return_confidences=True` 시 / when True) |

> **SSOT-NM01 준수 / Compliance**: `predict()` 내부에서 `[0,1]` 정규화 후 반드시 ImageNet mean/std 를 적용한다. / Inside `predict()`, `[0,1]` normalization must be followed by ImageNet mean/std.
> 학습(`dataset.py _IMAGENET_NORMALIZE`)과 동일한 변환 — 불일치 시 성능 저하. / Same transform as training — mismatch causes performance degradation.

---

## 7. `predict_batch` 입출력 / predict_batch I/O

```python
predict_batch(
    images_dict: Dict[str, np.ndarray],   # {channel: (N,H,W,3) BGR array}
    batch_size: int = 32,
) -> Dict[str, Dict[str, np.ndarray]]     # {channel: predict() 반환값 / return value}
```

> 로드되지 않은 채널은 경고 로그 후 결과에서 제외된다.
> / Unloaded channels are excluded from results after a warning log.

---

## 8. 필수 Config 키 / Required Config Keys

| Config 키 / Config Key | 사용처 / Usage Location | 설명 / Description |
| --- | --- | --- |
| `system.device` | `DeviceMixin._setup_device` | `"auto" \| "cuda" \| "mps" \| "cpu"` |
| `storage.models_dir` | `ModelLoaderMixin._resolve_model_path` | 모델 아티팩트 디렉토리 / Model artifact directory |
| `data.channels` | `GrayspotPredictor.__init__` | 지원 채널 목록 / Supported channel list |
| `data.image_size` | `GrayspotPredictor.__init__` | 입력 이미지 크기 / Input image size |
| `data.num_levels` | `GrayspotPredictor.__init__` | 분류 클래스 수 / Number of classification classes |

---

## 9. 금지 패턴 / Prohibited Patterns

```python
# ❌ 추론 전 ImageNet 정규화 미적용
# / ImageNet normalization not applied before inference
predictions = model(torch.tensor(images / 255.0))  # 정규화 없음 / no normalization

# ❌ BGR → RGB 변환 후 추론
# / Inference after BGR → RGB conversion
images_rgb = images[:, :, :, ::-1]  # SSOT-CS01 위반 / violation

# ✅ 올바른 패턴 (predict() 내부 처리 보장) / Correct pattern (handled inside predict())
result = predictor.predict(images, channel="Y")
```

---

## 10. 체크리스트 / Checklist

- [x] `load_model()` — 모델 파일 없으면 `FileNotFoundError` (SSOT-FF01) / `FileNotFoundError` if model file missing
- [x] `predict()` — BGR 색상 공간 유지 (SSOT-CS01) / BGR color space maintained
- [x] `predict()` — ImageNet 정규화 적용 (SSOT-NM01) / ImageNet normalization applied
- [x] `weights_only=True` 로드 확인 / Verify `weights_only=True` load
- [ ] `_EvalDataset`에 ImageNet 정규화 적용 (N-01) / Apply ImageNet normalization to `_EvalDataset`

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Model_Architecture.md](../SSOT/SSOT_Model_Architecture.md) | 모델 구조 정의 (What) / Model architecture definition |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | 체크포인트 로드 계약 / Checkpoint load contract |
| [Contract_model_boundary.md](Contract_model_boundary.md) | 모델 입출력 계약 / Model I/O contract |
| [Contract_fail_fast.md](Contract_fail_fast.md) | SSOT-FF01 정의 / SSOT-FF01 definition |
