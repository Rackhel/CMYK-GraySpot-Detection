---
type: contract
domain: inference_boundary
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
---

# [Contract] Inference Boundary — 추론 경계 계약

> **목적**: `GrayspotPredictor`의 공개 API 입출력, Mixin 구조, Fail-Fast 조건을 정의한다.
> **상태**: ✅ Accepted [Hard]
> **작성일**: 2026-05-17
> **관련 문서**:
>
> - [SSOT_Model_Architecture.md](../SSOT/SSOT_Model_Architecture.md) (모델 구조)
> - [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) (체크포인트 로드)

> 🔒 **SSOT 경계 원칙**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다.
> 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.

---

## 1. 계약 목적

`GrayspotPredictor`의 모델 로드, 단일/배치 추론, 캐시 관리의 인터페이스 규약을 정의한다.

---

## 2. 모듈 트리

```
inference/
├── predictor_device.py    — DeviceMixin      (장치 감지·설정)
├── predictor_loader.py    — ModelLoaderMixin (모델 로딩·캐시)
├── predictor_inference.py — InferenceMixin   (추론 실행)
└── predictor.py           — GrayspotPredictor (Orchestrator)
```

---

## 3. `GrayspotPredictor` 공개 API

```python
from inference.predictor import GrayspotPredictor

predictor = GrayspotPredictor(config_path=None)
predictor.load_model(channel="Y", model_path=None)
result    = predictor.predict(images, channel="Y", batch_size=32, return_confidences=True)
results   = predictor.predict_batch(images_dict, batch_size=32)
predictor.clear_cache(channel=None)
info      = predictor.get_model_info(channel=None)
```

| 메서드 | 모듈 | Mixin | 설명 |
| --- | --- | --- | --- |
| `__init__` | `inference.predictor` | Orchestrator | config 로딩, 장치 설정, 캐시 초기화 |
| `load_model` | `inference.predictor_loader` | `ModelLoaderMixin` | 채널별 모델 로드 및 캐시 |
| `predict` | `inference.predictor_inference` | `InferenceMixin` | 단일 채널 배치 추론 |
| `predict_batch` | `inference.predictor_inference` | `InferenceMixin` | 멀티 채널 배치 추론 |
| `clear_cache` | `inference.predictor_loader` | `ModelLoaderMixin` | 모델 캐시 비우기 |
| `get_model_info` | `inference.predictor_loader` | `ModelLoaderMixin` | 로드된 모델 정보 조회 |

---

## 4. `__init__` 입출력

```python
GrayspotPredictor(config_path: Optional[str | Path] = None) -> GrayspotPredictor
```

| 항목 | 타입 | 설명 |
| --- | --- | --- |
| `config_path` | `Optional[str \| Path]` | None 이면 기본 config.json 경로 사용 |
| **실패 조건** | `FileNotFoundError` | config.json 없음 — Fail-Fast (fallback 없음) |

---

## 5. `load_model` 입출력

```python
load_model(channel: str, model_path: Optional[str | Path] = None) -> None
```

| 항목 | 타입 | 설명 |
| --- | --- | --- |
| `channel` | `str` | `"Y" \| "M" \| "C" \| "K"` (대소문자 무관) |
| `model_path` | `Optional[str \| Path]` | None 이면 `storage.models_dir/best_{channel}.pt` 자동 탐색 |
| **실패 조건** | `ValueError` | 지원하지 않는 채널 |
| **실패 조건** | `FileNotFoundError` | 모델 파일 없음 — `SSOT-FF01` |

---

## 6. `predict` 입출력

```python
predict(
    images: np.ndarray,           # (N, H, W, 3) or (N, H, W) uint8 or float32, BGR
    channel: str,                 # "Y" | "M" | "C" | "K"
    batch_size: int = 32,
    return_confidences: bool = True,
) -> Dict[str, np.ndarray]
```

| 반환 키 | 형상 | 타입 | 설명 |
| --- | --- | --- | --- |
| `predictions` | `(N,)` | `int64` | 예측 클래스 [0-5] |
| `logits` | `(N, 6)` | `float32` | 원시 로짓 |
| `probabilities` | `(N, 6)` | `float32` | Softmax 확률 |
| `confidences` | `(N,)` | `float32` | Max-softmax 신뢰도 (`return_confidences=True` 시) |

> **SSOT-NM01 준수**: `predict()` 내부에서 `[0,1]` 정규화 후 반드시 ImageNet mean/std 를 적용한다.
> 학습(`dataset.py _IMAGENET_NORMALIZE`)과 동일한 변환 — 불일치 시 성능 저하.

---

## 7. `predict_batch` 입출력

```python
predict_batch(
    images_dict: Dict[str, np.ndarray],   # {channel: (N,H,W,3) BGR array}
    batch_size: int = 32,
) -> Dict[str, Dict[str, np.ndarray]]     # {channel: predict() 반환값}
```

> 로드되지 않은 채널은 경고 로그 후 결과에서 제외된다.

---

## 8. 필수 Config 키

| Config 키 | 사용처 | 설명 |
| --- | --- | --- |
| `system.device` | `DeviceMixin._setup_device` | `"auto" \| "cuda" \| "mps" \| "cpu"` |
| `storage.models_dir` | `ModelLoaderMixin._resolve_model_path` | 모델 아티팩트 디렉토리 |
| `data.channels` | `GrayspotPredictor.__init__` | 지원 채널 목록 |
| `data.image_size` | `GrayspotPredictor.__init__` | 입력 이미지 크기 |
| `data.num_levels` | `GrayspotPredictor.__init__` | 분류 클래스 수 |

---

## 9. 금지 패턴

```python
# ❌ 추론 전 ImageNet 정규화 미적용
predictions = model(torch.tensor(images / 255.0))  # 정규화 없음

# ❌ BGR → RGB 변환 후 추론
images_rgb = images[:, :, :, ::-1]  # SSOT-CS01 위반

# ✅ 올바른 패턴 (predict() 내부 처리 보장)
result = predictor.predict(images, channel="Y")
```

---

## 10. 체크리스트

- [x] `load_model()` — 모델 파일 없으면 `FileNotFoundError` (SSOT-FF01)
- [x] `predict()` — BGR 색상 공간 유지 (SSOT-CS01)
- [x] `predict()` — ImageNet 정규화 적용 (SSOT-NM01)
- [x] `weights_only=True` 로드 확인
- [ ] `_EvalDataset`에 ImageNet 정규화 적용 (N-01)

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_Model_Architecture.md](../SSOT/SSOT_Model_Architecture.md) | 모델 구조 정의 (What) |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | 체크포인트 로드 계약 |
| [Contract_model_boundary.md](Contract_model_boundary.md) | 모델 입출력 계약 |
| [Contract_fail_fast.md](Contract_fail_fast.md) | SSOT-FF01 정의 |
