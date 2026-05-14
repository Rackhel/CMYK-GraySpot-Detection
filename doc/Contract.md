# Contract — 모듈 간 인터페이스 계약 / Inter-Module Interface Contracts

CMYK Grayspot Detection System 의 모든 모듈 경계에서의 입력·출력 타입, 형상, 값 범위, 의존 config 키를 정의한다.

This document defines the input/output types, shapes, value ranges, and required config keys at every module boundary.

> **목적 / Purpose**: 데이터 의존성과 흐름을 한눈에 파악
> **역할 / Role**: "How" — 실제 인터페이스 규약 (의미 정의는 SSOT 개별 문서 참조)
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md), [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md), [SSOT_Artifacts.md](SSOT_Artifacts.md)

---

## Table of Contents / 목차

1. [전체 데이터 흐름 / Full Data Flow](#1-전체-데이터-흐름--full-data-flow)
2. [Config 로딩 계약 / Config Loading Contract](#2-config-로딩-계약--config-loading-contract)
3. [데이터 경계 계약 / Data Boundary Contracts](#3-데이터-경계-계약--data-boundary-contracts)
4. [모델 경계 계약 / Model Boundary Contracts](#4-모델-경계-계약--model-boundary-contracts)
5. [학습 경계 계약 / Training Boundary Contracts](#5-학습-경계-계약--training-boundary-contracts)
6. [아티팩트 경계 계약 / Artifact Boundary Contracts](#6-아티팩트-경계-계약--artifact-boundary-contracts)
7. [평가 경계 계약 / Evaluation Boundary Contracts](#7-평가-경계-계약--evaluation-boundary-contracts)
8. [Fail-Fast 집행 포인트 / Fail-Fast Enforcement Points](#8-fail-fast-집행-포인트--fail-fast-enforcement-points)
9. [모듈별 필수 Config 키 요약 / Required Config Keys per Module](#9-모듈별-필수-config-키-요약--required-config-keys-per-module)
10. [추론 경계 계약 / Inference Boundary Contracts](#10-추론-경계-계약--inference-boundary-contracts)
11. [튜닝 경계 계약 / Tuning Boundary Contracts](#11-튜닝-경계-계약--tuning-boundary-contracts)

---

## 1. 전체 데이터 흐름 / Full Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  config/config.json  ──►  utils_config.load_config()  ──►  cfg: dict   │
│                               (모든 모듈이 cfg를 수령 / all modules receive cfg)    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ cfg
                    ┌──────────────▼──────────────┐
                    │    data_set/labeled/         │
                    │  {channel}/{level}/*.png     │
                    │  BGR uint8  (H, W, 3)        │
                    └──────┬──────────────┬────────┘
                           │              │
               ┌───────────▼──┐    ┌──────▼────────────┐
               │ContrastiveDS │    │   CMYKDataset      │
               │ (Phase 0)    │    │   (Phase 2)        │
               │(view1, view2)│    │ (image, label)     │
               │(3,128,128)×2 │    │ (3,128,128), [0-5] │
               └──────┬───────┘    └──────┬─────────────┘
                      │                   │
               ┌──────▼───────┐    ┌──────▼─────────────┐
               │ DataLoader   │    │   DataLoader        │
               │(B,3,128,128) │    │ (B,3,128,128)+(B,) │
               └──────┬───────┘    └──────┬─────────────┘
                      │                   │
               ┌──────▼───────┐    ┌──────▼─────────────┐
               │GrayspotModel │    │ GrayspotModel       │
               │  phase=0     │    │   phase=2           │
               │  ↓           │    │   ↓                 │
               │(B,proj_dim)  │    │  (B,6) logits       │
               │ L2-norm      │    │  (no softmax)       │
               └──────┬───────┘    └──────┬─────────────┘
                      │                   │
               ┌──────▼───────┐    ┌──────▼─────────────┐
               │ InfoNCELoss  │    │CrossEntropyLoss     │
               │ scalar loss  │    │ scalar loss + acc   │
               └──────┬───────┘    └──────┬─────────────┘
                      │                   │
               ┌──────▼───────┐    ┌──────▼─────────────┐
               │Phase0Trainer │    │ Phase2Trainer       │
               │save_backbone │    │  best val_acc save  │
               └──────┬───────┘    └──────┬─────────────┘
                      │                   │
          ┌───────────▼──────┐   ┌────────▼───────────────┐
          │phase0_backbone_  │   │    best_{ch}.pt         │
          │{ch}_{tag}.pt     │   │  (backbone+ClassHead)   │
          └───────────┬───────┘   └────────┬───────────────┘
                      │ switch_to_phase2()  │ load (eval mode)
                      └──────────┬──────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │        Evaluator             │
                    │  run() → y_true, y_pred,     │
                    │          confidences         │
                    └────────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │    compute_all_channels()    │
                    │    EvaluationSummary         │
                    │  {accuracy, macro_f1, mae,   │
                    │   per_class, targets}        │
                    └────────────┬────────────────┘
                                 │
               ┌─────────────────┼──────────────────┐
               │                 │                  │
    ┌──────────▼──────┐  ┌───────▼────────┐  ┌─────▼──────────────┐
    │determine_swing_ │  │  save_report() │  │ html_report()       │
    │feedback()       │  │  CSV + JSON    │  │ HTML dashboard      │
    │ pass/retry_ph*  │  │  outputs/      │  │ outputs/reports/    │
    └─────────────────┘  └───────────────┘  └────────────────────┘
```

---

## 2. Config 로딩 계약 / Config Loading Contract

### 생산자 / Producer: `utils/utils_config.py`

```python
from src.utils import load_config, validate_config, create_directories, get_nested

cfg = load_config()                          # → dict
validate_config(cfg)                         # → None  (실패 시 ValueError 발생, SSOT-CF01 / Raises ValueError on failure)
create_directories(cfg)                      # → None  (storage.* 경로 생성 / creates storage.* paths)
val = get_nested(cfg, "phase2.learning_rate") # → Any   (None-safe dot-notation)
```

### 반환 타입 / Return Type

| 항목 / Item | 타입 / Type | 보장 / Guarantee |
|---|---|---|
| `load_config()` 반환값 / return value | `dict` | 경로 절대화 + device 자동 감지 완료 / Path absolutized + device auto-detected |
| `cfg["system"]["device"]` | `str` | `"cpu"` / `"cuda"` / `"mps"` (절대 `"auto"` 아님 / never `"auto"`) |
| `cfg["storage"]["*_dir"]` | `str` | 절대 경로 (root 기준 해소 완료) / Absolute path (resolved from root) |

### 금지 패턴 / Prohibited

```python
# ❌ 삭제된 패턴 — 사용 금지 / Deleted pattern — do not use
config = load_config()   # ConfigManager 반환 — 더 이상 존재하지 않음 / Returns ConfigManager — no longer exists
cfg = config.config
config.get("phase2.learning_rate")
```

---

## 3. 데이터 경계 계약 / Data Boundary Contracts

### 3.1 파일 → `preprocess()` → Dataset

| 경계 / Boundary | 타입 / Type | 형상 / Shape | 값 범위 / Range | 색상 공간 / Color |
|---|---|---|---|---|
| 파일 로드 후 / After file load | `np.ndarray` | `(H, W, 3)` | `[0, 255]` uint8 | **BGR** (`cv2.imread`) |
| `preprocess()` 출력 / output | `np.ndarray` | `(128, 128, 3)` | `[0.0, 1.0]` float32 | **BGR** |
| 증강 후 / After augmentation | `np.ndarray` | `(128, 128, 3)` | `[0.0, 1.0]` float32 | **BGR** |
| Tensor 변환 후 / After to_tensor | `torch.Tensor` | `(3, 128, 128)` | `[0.0, 1.0]` float32 | **BGR** (C,H,W) |
| ImageNet 정규화 후 / After normalize | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized float32 | **BGR** (C,H,W) |

> ⚠️ **SSOT-CS01**: 파일 로드부터 모델 입력까지 BGR을 유지한다. RGB 변환 금지. / BGR must be maintained from file load to model input. RGB conversion is prohibited.

### 3.2 `ContrastiveDataset` 출력 계약

```python
view1, view2 = dataset[i]
```

| 출력 / Output | 타입 / Type | 형상 / Shape | 범위 / Range | 비고 / Note |
|---|---|---|---|---|
| `view1` | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized | 동일 이미지 다른 증강 / Same image with different augmentation |
| `view2` | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized | Positive pair |

### 3.3 `CMYKDataset` 출력 계약

```python
image, label = dataset[i]
```

| 출력 / Output | 타입 / Type | 형상 / Shape | 범위 / Range | 비고 / Note |
|---|---|---|---|---|
| `image` | `torch.Tensor` | `(3, 128, 128)` | ImageNet-normalized | train split만 증강 적용 / Augmentation applied to train split only |
| `label` | `int` | scalar | `[0, 5]` | ordinal 6-class |

### 3.4 DataLoader 배치 계약

| Phase | 배치 타입 / Type | 형상 / Shape |
|---|---|---|
| Phase 0 | `(Tensor, Tensor)` | `(B, 3, 128, 128)`, `(B, 3, 128, 128)` |
| Phase 2 | `(Tensor, Tensor)` | `(B, 3, 128, 128)`, `(B,)` int |

### 3.5 `augment_supervised()` 계약

```python
from data.augmentation import augment_supervised

aug_image = augment_supervised(image: np.ndarray, aug_cfg: Optional[dict] = None) -> np.ndarray
```

| 항목 / Item | 타입 / Type | 제약 / Constraint |
|---|---|---|
| `image` (입력) | `np.ndarray` | `(H, W, 3)` float32, 범위 `[0.0, 1.0]`, **BGR** |
| `aug_cfg` | `Optional[dict]` | `cfg["phase2"]["augmentation"]` — None 이면 모듈 기본값 사용 |
| 반환 / Return | `np.ndarray` | 입력과 동일 형상·범위·색상 공간 유지 |

**적용 변환 / Applied transforms** (all controlled by `aug_cfg`):

| 변환 / Transform | 파라미터 / Parameter | 기본값 / Default |
|---|---|---|
| Random horizontal flip | `flip_prob` | 0.5 |
| Brightness jitter | `brightness_prob`, `brightness_range` | 0.3, 20 |
| Additive noise | `noise_prob`, `noise_range` | 0.2, 10 |

### 3.6 `augment_contrastive()` 계약

```python
from data.augmentation import augment_contrastive

aug_image = augment_contrastive(
    image     : np.ndarray,    # (H, W, 3) float32 [0,1] BGR
    image_size: int,           # 출력 크기 / Output size (e.g. 128)
    aug_cfg   : Optional[dict] = None,  # cfg["phase0"]["augmentation"]
) -> np.ndarray                # (image_size, image_size, 3) float32 [0,1] BGR
```

**적용 변환 / Applied transforms** (all controlled by `aug_cfg`):

| 변환 / Transform | 파라미터 / Parameter |
|---|---|
| Random horizontal flip | `flip_prob` |
| Random crop + resize | `crop_prob`, `crop_scale_min`, `crop_scale_max` |
| Brightness jitter | `blur_prob`, `color_jitter` |
| Contrast jitter | `blur_prob`, `contrast_scale_min`, `contrast_scale_max` |
| Gaussian blur | `blur_prob`, `blur_kernels` |

> ⚠️ **SSOT-CS01** 준수: 두 증강 함수 모두 BGR 색상 공간을 유지한다. RGB 변환 없음.
> Both augmentation functions preserve BGR color space. No RGB conversion.

---

## 4. 모델 경계 계약 / Model Boundary Contracts

### 4.1 `GrayspotModel` 입출력

| Phase | 입력 / Input | 출력 / Output | 비고 / Note |
|---|---|---|---|
| Phase 0 | `(B, 3, 128, 128)` float32 | `(B, 128)` float32 | L2-정규화된 projection vector / L2-normalized projection vector |
| Phase 2 | `(B, 3, 128, 128)` float32 | `(B, 6)` float32 | Raw logits — Softmax 없음 / No Softmax applied |

### 4.2 생성 계약 / Construction Contract

```python
# Phase 0 모델 생성 / Phase 0 model construction
model = GrayspotModel(cfg, phase=0)
# 필수 cfg 키 / Required cfg keys: model.backbone, phase0.projection_dim, phase0.hidden_dim

# Phase 2 모델 생성 / Phase 2 model construction
model = GrayspotModel(cfg, phase=2)
# 필수 cfg 키 / Required cfg keys: model.backbone, model.frozen_backbone, data.num_levels
# backbone별 head cfg / Backbone-specific head cfg (fallback to phase2.hidden_dim/dropout if absent):
#   phase2.heads.efficientnet_b0.{hidden_dim, dropout}         → 직접 압축 구조 / Direct compression
#   phase2.heads.resnet50.{mid_dim, hidden_dim, dropout}       → 단계적 압축 구조 / Staged compression
```

### 4.2.1 Backbone별 ClassifierHead 생성 규칙 / Backbone-Specific ClassifierHead Build Rules

| Backbone | `mid_dim` | ClassifierHead 구조 / Structure |
|---|---|---|
| `efficientnet_b0` | `None` | `in_dim(1280) → hidden_dim → num_levels` |
| `resnet50` | `512` | `in_dim(2048) → mid_dim → hidden_dim → num_levels` |

`phase2.heads.{backbone}` 부재 시 → `phase2.hidden_dim` / `phase2.dropout` fallback.
If `phase2.heads.{backbone}` absent → fallback to `phase2.hidden_dim` / `phase2.dropout`.

### 4.3 `build_backbone()` 계약

```python
backbone, feature_dim = build_backbone(backbone_name: str)
```

| Backbone | `feature_dim` | Hard SSOT |
|---|---|---|
| `efficientnet_b0` | `1280` | ✅ |
| `resnet50` | `2048` | ✅ |

### 4.4 Phase 전환 계약 / Phase Transition Contract

```python
model.switch_to_phase2(backbone_path: Path, cfg: dict) -> None
# 입력 / Input: phase0_backbone_{channel}_{tag}.pt, config dict
# 동작 / Action: backbone.* 키만 로드 (strict=False) / Load only backbone.* keys (strict=False)
# 결과 / Result: Backbone weights 유지 + ClassifierHead 새로 초기화 / Backbone weights preserved + ClassifierHead freshly initialized
# 실패 / Fail: backbone 키 0개 로드 시 RuntimeError (SSOT-FF01) / RuntimeError if zero backbone keys loaded
```

### 4.5 `build_model()` 유틸리티 계약

```python
from utils.utils_model import build_model

model = build_model(cfg: dict, checkpoint: Path, device: torch.device) -> nn.Module
```

| 항목 / Item | 타입 / Type | 설명 / Description |
|---|---|---|
| `cfg` | `dict` | config dict (Phase 2 head 구성에 사용) |
| `checkpoint` | `Path` | `best_{ch}.pt` 절대 경로 |
| `device` | `torch.device` | 연산 디바이스 |
| 반환 / Return | `nn.Module` | `GrayspotModel(phase=2)`, `model.eval()` 상태 |
| 주의 / Note | — | `strict=False` 로드 — 키 불일치 허용 / Allows key mismatch |

---

## 5. 학습 경계 계약 / Training Boundary Contracts

### 5.1 `Phase0Trainer`

```python
trainer = Phase0Trainer(model, cfg, channel="Y", device=device)
history = trainer.train(loader)           # → List[dict]  에폭별 기록 / per-epoch record
trainer.save_backbone()                   # → Path  저장 경로 반환 / returns save path
```

| 경계 / Boundary | 입력 타입 / Input | 출력 타입 / Output |
|---|---|---|
| `train()` 입력 / Input | `DataLoader` — `(B,3,128,128), (B,3,128,128)` | `List[{"epoch", "loss"}]` |
| `save_backbone()` 출력 / Output | — | `data_set/models/phase0_backbone_{ch}_{tag}.pt` |

**필수 cfg 키**: `phase0.epochs`, `phase0.batch_size`, `phase0.learning_rate`,
`phase0.weight_decay`, `phase0.temperature`, `train.optimizer`, `train.scheduler`,
`train.gradient_clip`, `train.eta_min`, `train.seed`, `storage.models_dir`

### 5.2 `Phase2Trainer`

```python
trainer = Phase2Trainer(model, cfg, channel="Y", device=device, train_ds=train_ds)
history = trainer.train(train_loader, val_loader)  # → List[dict]
trainer.save_history(history)                       # → Path  CSV 저장 / saves CSV and returns path
```

| 경계 / Boundary | 입력 타입 / Input | 출력 타입 / Output |
|---|---|---|
| `train()` 입력 / Input | `DataLoader` × 2 — `(B,3,128,128), (B,)` | `List[{"epoch", "train_loss", "val_acc", ...}]` |
| 체크포인트 / Checkpoint | 내부 자동 저장 / Auto-saved internally | `data_set/models/best_{ch}.pt` |
| `save_history()` 출력 / Output | `List[dict]` | `data_set/reports/phase2_history_{ch}.csv` |

**필수 cfg 키**: `phase2.epochs`, `phase2.batch_size`, `phase2.learning_rate`,
`phase2.weight_decay`, `phase2.early_stopping.*`, `train.optimizer`, `train.scheduler`,
`train.gradient_clip`, `train.eta_min`, `train.seed`, `storage.models_dir`, `storage.reports_dir`

### 5.3 손실 함수 계약 / Loss Function Contract

```python
from training import InfoNCELoss, get_loss

# Phase 0
loss_fn = InfoNCELoss(temperature=cfg["phase0"]["temperature"])
loss = loss_fn(z1, z2)
# z1, z2: (B, projection_dim)  L2-normalized — 반드시 정규화 후 전달 / must be normalized before passing

# Phase 2
loss_fn = get_loss(cfg, phase=2, train_samples=train_ds.samples)
loss = loss_fn(logits, labels)
# logits: (B, 6)  raw logits (Softmax 미적용 상태 / Softmax not applied)
# labels: (B,)   int [0, 5]
```

---

## 6. 아티팩트 경계 계약 / Artifact Boundary Contracts

### 6.1 Phase 0 체크포인트

| 항목 / Item | 값 / Value |
|---|---|
| 경로 패턴 / Path | `{storage.models_dir}/phase0_backbone_{channel}_{tag}.pt` |
| 형식 / Format | `torch.save(model.state_dict())` |
| 포함 키 / Keys | `backbone.*` + `head.*` (ProjectionHead) |
| 생산자 / Producer | `Phase0Trainer.save_backbone()` |
| 소비자 / Consumer | `GrayspotModel.switch_to_phase2()` — `backbone.*` 키만 선택 로드 |

### 6.2 Phase 2 체크포인트 (`best_{ch}.pt`)

| 항목 / Item | 값 / Value |
|---|---|
| 경로 패턴 / Path | `{storage.models_dir}/best_{channel}.pt` |
| 형식 / Format | `torch.save(model.state_dict())` |
| 포함 키 / Keys | `backbone.*` + `head.*` (ClassifierHead) |
| 저장 기준 / Trigger | `val_acc > best_val_acc + early_stopping.min_delta` |
| 생산자 / Producer | `Phase2Trainer.train()` 내부 / Inside `Phase2Trainer.train()` |
| 소비자 / Consumer | `Evaluator`, `GrayspotPredictor.load_model()` |

### 6.3 추론 시 로드 계약 / Inference Load Contract

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint, strict=False)
# weights_only=True : pickle 보안 / pickle security
# strict=False      : 버전 간 키 불일치 허용 / allows key mismatches between versions
```

---

## 7. 평가 경계 계약 / Evaluation Boundary Contracts

> ✅ **리팩토링 완료 / Refactoring Complete (2026-05-12)**: `evaluator.py`(~950줄)를 4 Mixin + Orchestrator 패턴으로 분해하여 SRP·ISP 위반 해결.
>
> **분해 결과 / Decomposition Result**:
> - `evaluator_inference.py` — `_EvalDataset`, `InferenceMixin` (추론 전담 / Inference only)
> - `evaluator_metrics.py` — `MetricsMixin` (지표 계산 전담 / Metrics only)
> - `evaluator_export.py` — `ExportMixin` (CSV/JSON 저장 전담 / Export only)
> - `evaluator_charts.py` — `ChartsMixin` (차트 7종 + Phase 3 판단 / Charts + Phase 3 decision)
> - `evaluator.py` — `Evaluator` 조율자 (`__init__` + `save_report` / Orchestrator)
>
> 외부 API (`from evaluation.evaluator import Evaluator`) 및 §7.1–§7.5 계약은 변경 없음.
> External API and §7.1–§7.5 contracts are unchanged.

### 7.1 `Evaluator` 입력 계약

```python
evaluator = Evaluator(
    model       = model,            # nn.Module, model.eval() 상태 / in model.eval() state
    labeled_dir = Path("data_set/labeled"),
    labels_csv  = Path("data_set/labels_v0.csv"),
    output_dir  = Path("outputs/reports"),
    device      = device,
    cfg         = cfg,              # swing_thresholds + confidence_thresholds 주입 / injected
)
```

| 입력 / Input | 타입 / Type | 제약 / Constraint |
|---|---|---|
| `model` | `nn.Module` | 반드시 `model.eval()` 상태 / Must be in `model.eval()` state |
| 이미지 배치 / Image batch | `Tensor (B, 3, 128, 128)` | BGR float32, ImageNet-normalized — 학습과 동일 / Same as training |
| 레이블 / Labels | `Tensor (B,)` | int [0, 5] |

### 7.2 `Evaluator.run()` 출력 계약

```python
results = evaluator.run(channels=["Y", "M", "C", "K"])
```

| 출력 키 / Key | 타입 / Type | 형상 / Shape | 범위 / Range |
|---|---|---|---|
| `"y_true"` | `np.ndarray` | `(N,)` | int [0, 5] |
| `"y_pred"` | `np.ndarray` | `(N,)` | int [0, 5] |
| `"confidences"` | `np.ndarray` | `(N,)` | float [0.0, 1.0] |

### 7.3 `build_evaluation_summary()` 계약

```python
from evaluation import build_evaluation_summary

summary = build_evaluation_summary(
    results  = {"Y": {"y_true": ..., "y_pred": ...}, ...},
    channels = ["Y", "M", "C", "K"],
    meta     = {"backbone": "efficientnet_b0"},
    cfg      = cfg,   # evaluation.swing_thresholds → summary.targets 주입
)
```

| 출력 / Output | 타입 / Type | 보장 / Guarantee |
|---|---|---|
| `summary.overall` | `ChannelMetrics` | 전 채널 집계 지표 / Aggregated metrics across all channels |
| `summary.by_channel[ch]` | `ChannelMetrics` | 채널별 지표 / Per-channel metrics |
| `summary.targets["swing_*_retry"]` | `float` | cfg 주입값 (없으면 기본값) / cfg-injected value (default if absent) |

### 7.4 `ChannelMetrics` 구조

```python
@dataclass
class ChannelMetrics:
    accuracy : float       # 전체 정확도 / Overall accuracy
    macro_f1 : float       # Macro-averaged F1
    mae      : float       # Mean Absolute Error (ordinal)
    n_samples: int
    per_class: List[PerClassMetric]  # level별 precision/recall/f1/support / per-level

    @property
    def acc_pass(self) -> bool: ...  # accuracy >= TARGET_PER_COLOR_ACC (0.85)
    def f1_pass (self) -> bool: ...  # macro_f1  >= TARGET_PER_CLASS_F1  (0.80)
    def mae_pass(self) -> bool: ...  # mae       <= TARGET_MAE            (0.50)
```

### 7.5 `determine_swing_feedback()` 계약

```python
from evaluation import determine_swing_feedback

feedback = determine_swing_feedback(summary, channels=None)
# channels: List[str] | None — None 이면 summary.by_channel.keys() 전체 사용
```

**반환 타입 / Return Type**: `dict`

```python
{
    "terminate": bool,        # True = 모든 목표 달성 → Swing 종료 / All targets met → exit Swing
    "decisions": List[str],   # 조치 필요 항목 목록 (비어있으면 계속 진행) / Action items (empty = continue)
}
```

| `terminate` | `decisions` | 조건 / Condition | 다음 단계 / Next Step |
|---|---|---|---|
| `True` | `[]` | 모든 채널 및 목표 달성 / All channels & targets met | Swing 종료 / Exit Swing |
| `False` | 채움 | per-color accuracy < `swing_acc_retry` (0.80) | Phase 0 재시작 / Restart Phase 0 |
| `False` | 채움 | per-class F1 < `swing_f1_retry` (0.70) | Phase 1 레벨 경계 재검토 / Review level boundary |
| `False` | 채움 | overall MAE > `swing_mae_retry` (0.80) | Phase 0 표현학습 재시도 / Retry representation learning |

> **임계값 출처 / Threshold source**: `cfg["evaluation"]["swing_thresholds"]` 에서 읽음. 기본값은 위 괄호 안 값.
> Read from `cfg["evaluation"]["swing_thresholds"]`. Defaults shown in parentheses.

### 7.6 `MetricsMixin.compute()` 계약

```python
metrics = evaluator.compute(results, channels=None)
```

| 항목 / Item | 타입 / Type | 설명 / Description |
|---|---|---|
| `results` (입력) | `Dict[str, dict]` | `run()` 반환값 — 채널별 `y_true/y_pred/confidences/filenames` |
| `channels` (입력) | `Optional[List[str]]` | None 이면 `results.keys()` 전체 사용 |
| 반환 / Return | `Dict[str, dict]` | `compute_all_channels()` 반환값 — `"overall"` 키 포함 |

### 7.7 `MetricsMixin.get_misclassified()` 계약

```python
df = evaluator.get_misclassified(results, channels=None)
```

| 반환 컬럼 / Column | 타입 / Type | 설명 / Description |
|---|---|---|
| `filename` | `str` | 파일명 |
| `color` | `str` | 채널 (`Y`/`M`/`C`/`K`) |
| `true_level` | `int` | 실제 레벨 [0-5] |
| `pred_level` | `int` | 예측 레벨 [0-5] |
| `confidence` | `float` | Max-softmax 신뢰도 [0.0-1.0] |
| `correct` | `bool` | 항상 `False` — 오분류 필터 결과 |

### 7.8 `ExportMixin.save_report()` 계약

```python
dashboard_path = evaluator.save_report(
    results,
    metrics,
    experiment_name="eval",
    channels=None,
    open_browser=False,
    checkpoint_path=None,
) -> Path
```

**생성 파일 / Generated files** (모두 `output_dir/` 아래):

| 파일 / File | 설명 / Description |
|---|---|
| `confusion/cm_{channel}.html` | 채널별 정규화 혼동행렬 / Per-channel normalized CM |
| `confusion/cm_overall.html` | 전체 혼동행렬 / Overall CM |
| `eval_dashboard.html` | Gauge + Bar 대시보드 (반환 경로) / Dashboard (returned path) |
| `per_class_metrics.html` | 클래스별 F1 바 차트 / Per-class F1 bar chart |
| `mae_heatmap.html` | MAE 히트맵 |
| `misclassified_scatter.html` | 오분류 scatter |
| `confidence_distribution.html` | 신뢰도 분포 |
| `evaluation_results_{name}.csv` | 채널별 지표 CSV |
| `misclassified_{name}.csv` | 오분류 샘플 CSV |
| `metrics_summary_{name}.json` | JSON 요약 |

### 7.9 `summary_to_dict()` 계약

```python
from evaluation.metrics import summary_to_dict

d = summary_to_dict(summary)  # → dict (JSON-직렬화 가능 / JSON-serializable)
```

```python
{
    "overall": {"accuracy": float, "macro_f1": float, "mae": float,
                "n_samples": int, "per_class": [{"level", "precision", "recall", "f1", "support"}, ...]},
    "by_channel": {"Y": {...}, "M": {...}, "C": {...}, "K": {...}},
    "meta": dict,
    "targets": dict,
}
```

### 7.10 `generate_baseline_report()` 계약

```python
from reporting.html_report import generate_baseline_report

path = generate_baseline_report(
    summary      : EvaluationSummary,
    results      : Dict[str, dict],
    output_path  : str | Path = "outputs/reports/baseline.html",
    channels     : List[str]  = ["Y", "M", "C", "K"],
    open_browser : bool       = False,
    logger       = None,
) -> Path
```

| 항목 / Item | 설명 / Description |
|---|---|
| 반환 / Return | 생성된 HTML 파일의 절대 경로 / Absolute path of generated HTML file |
| 산출물 / Output | 단일 독립 HTML (Plotly CDN 내장) / Single self-contained HTML with Plotly CDN |
| 필수 입력 / Required | `summary` (EvaluationSummary), `results` (run() 반환값) |

---

## 8. Fail-Fast 집행 포인트 / Fail-Fast Enforcement Points

모든 경계에서 아래 조건은 **즉시 예외를 발생시켜야 한다**. 우회·임시 생성 금지. / At every boundary, the following conditions **must raise an exception immediately**. No fallbacks.

| 위치 / Location | 조건 / Condition | 코드 / Code | 예외 / Exception |
|---|---|---|---|
| `run_phase2.py` 시작 / start | `phase0_backbone_{ch}_{tag}.pt` 미존재 / not found | `SSOT-FF01` | `FileNotFoundError` |
| `Evaluator.run()` 시작 / start | `best_{ch}.pt` 미존재 / not found | `SSOT-FF01` | `FileNotFoundError` |
| `GrayspotModel.__init__` | `cfg["data"]["num_levels"]` 키 미존재 / key missing | `SSOT-CF01` | `KeyError` |
| `Phase0Trainer.train()` | DataLoader 배치 언패킹 실패 (형상 불일치) / DataLoader batch unpack failure (shape mismatch) | `SSOT-CS01` | `ValueError` |
| `InfoNCELoss.forward()` | z1, z2 가 L2-정규화되지 않은 경우 / z1, z2 not L2-normalized | — | `RuntimeError` (수치 이상 / numerical anomaly) |
| `switch_to_phase2()` | backbone 키 0개 로드 (구조 불일치) / zero backbone keys loaded (architecture mismatch) | `SSOT-FF01` | `RuntimeError` |
| `validate_config()` | 필수 섹션 누락 (`data`, `model`, `phase2` 등) / required section missing (`data`, `model`, `phase2`, etc.) | `SSOT-CF01` | `ValueError` |

---

## 10. 추론 경계 계약 / Inference Boundary Contracts

### 10.1 모듈 트리 / Module Tree

```
inference/
├── predictor_device.py    — DeviceMixin      (장치 감지·설정 / Device detection & setup)
├── predictor_loader.py    — ModelLoaderMixin (모델 로딩·캐시 / Model loading & cache)
├── predictor_inference.py — InferenceMixin   (추론 실행 / Inference execution)
└── predictor.py           — GrayspotPredictor (Orchestrator)
```

### 10.2 `GrayspotPredictor` 공개 API / Public API

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
|---|---|---|---|
| `__init__` | `inference.predictor` | Orchestrator | config 로딩, 장치 설정, 캐시 초기화 / Config loading, device setup, cache init |
| `load_model` | `inference.predictor_loader` | `ModelLoaderMixin` | 채널별 모델 로드 및 캐시 / Per-channel model load & cache |
| `predict` | `inference.predictor_inference` | `InferenceMixin` | 단일 채널 배치 추론 / Single-channel batch inference |
| `predict_batch` | `inference.predictor_inference` | `InferenceMixin` | 멀티 채널 배치 추론 / Multi-channel batch inference |
| `clear_cache` | `inference.predictor_loader` | `ModelLoaderMixin` | 모델 캐시 비우기 / Clear model cache |
| `get_model_info` | `inference.predictor_loader` | `ModelLoaderMixin` | 로드된 모델 정보 조회 / Query loaded model info |

### 10.3 `__init__` 입출력 / Init I/O

```python
GrayspotPredictor(config_path: Optional[str | Path] = None) -> GrayspotPredictor
```

| 항목 / Item | 타입 / Type | 설명 / Description |
|---|---|---|
| `config_path` | `Optional[str \| Path]` | None 이면 기본 config.json 경로 사용 / None uses default config.json |
| **실패 조건 / Fail** | `FileNotFoundError` | config.json 없음 — Fail-Fast (fallback 없음 / no fallback) |

### 10.4 `load_model` 입출력 / Load Model I/O

```python
load_model(channel: str, model_path: Optional[str | Path] = None) -> None
```

| 항목 / Item | 타입 / Type | 설명 / Description |
|---|---|---|
| `channel` | `str` | `"Y" \| "M" \| "C" \| "K"` (대소문자 무관 / case-insensitive) |
| `model_path` | `Optional[str \| Path]` | None 이면 `storage.models_dir/best_{channel}.pt` 자동 탐색 |
| **실패 조건 / Fail** | `ValueError` | 지원하지 않는 채널 / Unsupported channel |
| **실패 조건 / Fail** | `FileNotFoundError` | 모델 파일 없음 — `SSOT-FF01` |

### 10.5 `predict` 입출력 / Predict I/O

```python
predict(
    images: np.ndarray,           # (N, H, W, 3) or (N, H, W) uint8 or float32, BGR
    channel: str,                 # "Y" | "M" | "C" | "K"
    batch_size: int = 32,
    return_confidences: bool = True,
) -> Dict[str, np.ndarray]
```

| 반환 키 / Return Key | 형상 / Shape | 타입 / Type | 설명 / Description |
|---|---|---|---|
| `predictions` | `(N,)` | `int64` | 예측 클래스 [0-5] / Predicted class |
| `logits` | `(N, 6)` | `float32` | 원시 로짓 / Raw logits |
| `probabilities` | `(N, 6)` | `float32` | Softmax 확률 / Softmax probabilities |
| `confidences` | `(N,)` | `float32` | Max-softmax 신뢰도 (`return_confidences=True` 시) |

> **SSOT-NM01 준수**: `predict()` 내부에서 `[0,1]` 정규화 후 반드시 ImageNet mean/std 를 적용한다.
> 학습(`dataset.py _IMAGENET_NORMALIZE`)과 동일한 변환 — 불일치 시 성능 저하.

### 10.6 `predict_batch` 입출력 / Predict Batch I/O

```python
predict_batch(
    images_dict: Dict[str, np.ndarray],   # {channel: (N,H,W,3) BGR array}
    batch_size: int = 32,
) -> Dict[str, Dict[str, np.ndarray]]     # {channel: predict() 반환값}
```

> 로드되지 않은 채널은 경고 로그 후 결과에서 제외된다.
> Channels without loaded models are skipped with a warning log.

### 10.7 필수 Config 키 / Required Config Keys

| Config 키 / Key | 사용처 / Used by | 설명 / Description |
|---|---|---|
| `system.device` | `DeviceMixin._setup_device` | `"auto" \| "cuda" \| "mps" \| "cpu"` |
| `storage.models_dir` | `ModelLoaderMixin._resolve_model_path` | 모델 아티팩트 디렉토리 / Model artifact directory |
| `data.channels` | `GrayspotPredictor.__init__` | 지원 채널 목록 / Supported channels |
| `data.image_size` | `GrayspotPredictor.__init__` | 입력 이미지 크기 / Input image size |
| `data.num_levels` | `GrayspotPredictor.__init__` | 분류 클래스 수 / Number of classification levels |

---

## 9. 모듈별 필수 Config 키 요약 / Required Config Keys per Module

| 모듈 / Module | 필수 config 키 / Required Keys |
|---|---|
| `data.dataset` | `data.channels`, `data.num_levels`, `data.image_size`, `data.split_ratios.*`, `storage.labeled_dir`, `train.seed` |
| `data.augmentation` (Phase 0) | `phase0.augmentation.*` (flip/crop/color_jitter/contrast/blur) |
| `data.augmentation` (Phase 2) | `phase2.augmentation.*` (flip/brightness/noise), `phase2.oversample` |
| `models.grayspot_model` | `model.backbone`, `model.frozen_backbone`, `data.num_levels`, `phase0.projection_dim`, `phase0.hidden_dim`, `phase2.heads.{backbone}.hidden_dim`, `phase2.heads.{backbone}.dropout`, `phase2.heads.resnet50.mid_dim` (resnet50 시 / when resnet50) |
| `training.trainer` (Phase 0) | `phase0.epochs`, `phase0.batch_size`, `phase0.learning_rate`, `phase0.weight_decay`, `phase0.temperature`, `train.optimizer`, `train.scheduler`, `train.gradient_clip`, `train.eta_min`, `storage.models_dir` |
| `training.trainer` (Phase 2) | `phase2.epochs`, `phase2.batch_size`, `phase2.learning_rate`, `phase2.weight_decay`, `phase2.early_stopping.*`, `train.*`, `storage.models_dir`, `storage.reports_dir` |
| `training.losses` | `phase0.temperature`, `data.num_levels` |
| `evaluation.evaluator` | `inference.confidence_thresholds.*`, `evaluation.swing_thresholds.*` |
| `evaluation.metrics` | `evaluation.targets.*`, `data.num_levels` |
| `tuning.optuna_tuner` | `optuna.*`, `system.device`, `train.seed` |
| `tuning.search_space` | `optuna.search_space.{backbone}.*` (backbone별 분기 / backbone-branched) |
| `utils.utils_config` | — (config 자체를 로드하므로 의존 없음 / loads config itself, no dependencies) |
| `utils.utils_model` | `model.backbone`, `storage.models_dir`, `system.device` (`build_model` 사용 시 / when using build_model) |
| `inference.predictor` | `system.device`, `storage.models_dir`, `data.channels`, `data.image_size`, `data.num_levels` |
| `scripts.run_phase0` | `data.channels`, `system.device`, `train.seed`, `storage.*` |
| `scripts.run_phase2` | `data.channels`, `system.device`, `train.seed`, `storage.*` |
| `scripts.run_baseline` | `data.channels`, `system.device`, `train.seed`, `storage.*` |
| `scripts.run_optuna` | `optuna.*`, `system.device` |

---

## 11. 튜닝 경계 계약 / Tuning Boundary Contracts

### 11.1 `run_optuna()` 계약

```python
from tuning.optuna_tuner import run_optuna

run_optuna(n_trials: int | None = None, channel: str = "all") -> None
```

| 항목 / Item | 타입 / Type | 설명 / Description |
|---|---|---|
| `n_trials` | `int \| None` | None 이면 `cfg["optuna"]["n_trials"]` 사용 (기본 5) / Uses cfg default if None |
| `channel` | `str` | `"all"` 또는 단일 채널 (`"Y"/"M"/"C"/"K"`) — 대소문자 무관 |
| 반환 / Return | `None` | — |
| 산출물 / Output | — | `outputs/optuna/` 아래 trial 결과 JSON / Trial results JSON under `outputs/optuna/` |

**필수 cfg 키 / Required config keys**: `optuna.n_trials`, `optuna.search_space.{backbone}.*`, `system.device`, `train.seed`

> ⚠️ **역방향 의존성 경고 / Reverse dependency warning**: `optuna_tuner.py` 가 `src.scripts.run_baseline` 을 import 한다. 이는 `tuning → scripts` 방향으로 레이어 위반. 향후 리팩토링 필요.
> `optuna_tuner.py` imports from `src.scripts.run_baseline` — a layer violation (`tuning → scripts`). Refactor planned.

### 11.2 하이퍼파라미터 탐색 공간 계약 / Hyperparameter Search Space Contract

```python
from tuning.search_space import get_phase2_search_space

space = get_phase2_search_space(trial: optuna.Trial, cfg: dict = None) -> dict
```

| 항목 / Item | 설명 / Description |
|---|---|
| `trial` | `optuna.Trial` 객체 |
| `cfg` | config dict — `None` 이면 기본값 사용 (`efficientnet_b0` 기준) |
| backbone 감지 | `cfg["model"]["backbone"]` — 없으면 `"efficientnet_b0"` fallback |
| 반환 / Return | `dict` — 샘플링된 하이퍼파라미터. ResNet-50 에만 `"mid_dim"` 포함 |
| Config 참조 / Config ref | `cfg["optuna"]["search_space"][backbone_name].*` → 최상위 → 기본값 순 fallback |

**반환 dict 필수 키 / Required return keys** (항상 / always):

| 키 / Key | 타입 / Type | 설명 / Description |
|---|---|---|
| `learning_rate` | `float` | log-uniform 탐색 |
| `batch_size` | `int` | categorical 선택 |
| `weight_decay` | `float` | log-uniform 탐색 |
| `epochs` | `int` | 정수 범위 탐색 |
| `dropout` | `float` | uniform 탐색 |
| `hidden_dim` | `int` | categorical 선택 |
| `mid_dim` | `int` | **ResNet-50 전용** — EfficientNet-B0 시 키 없음 |

### 11.3 `tuning/optuna_utils.py` 공개 API 계약

> **위치 / Location**: `src/tuning/optuna_utils.py` (not `src/utils/`)
>
> Optuna artifact 저장·불러오기, 채널 정규화, Phase 2 config 적용을 담당한다.
> 저장·불러오기 책임은 `optuna_tuner.py` 에서 분리 (SRP).

```python
from tuning.optuna_utils import (
    normalize_channel,
    load_best_params,
    save_best_params,
    save_trials_summary,
    apply_phase2_params,
)
```

#### `normalize_channel(channel: str) -> str`

| 항목 / Item | 내용 / Detail |
|---|---|
| 입력 / Input | `"Y"` / `"M"` / `"C"` / `"K"` / `"all"` (대소문자 무관) |
| 반환 / Return | 소문자 suffix (`"y"` / `"m"` / `"c"` / `"k"` / `"all"`) |
| **실패 / Fail** | `ValueError` — `VALID_CHANNELS = {Y,M,C,K,ALL}` 외 입력 |

#### `load_best_params(channel: str, output_dir: str | Path = "outputs/optuna") -> dict`

| 항목 / Item | 내용 / Detail |
|---|---|
| 입력 / Input | `channel` (normalize_channel 경유), `output_dir` |
| 반환 / Return | best params `dict` |
| **실패 / Fail** | `FileNotFoundError` — `best_params_{ch}.json` 미존재 (SSOT-FF01) |
| **실패 / Fail** | `ValueError` — 유효하지 않은 channel |

#### `save_best_params(params: dict, channel: str, output_dir) -> Path`

| 항목 / Item | 내용 / Detail |
|---|---|
| 반환 / Return | 저장된 `Path` 객체 |
| 파일명 / Filename | `best_params_{channel_lower}.json` |

#### `save_trials_summary(trials: list, channel: str, output_dir) -> Path`

| 항목 / Item | 내용 / Detail |
|---|---|
| `trials` | `optuna.trial.FrozenTrial` 목록 (study.trials) |
| 파일명 / Filename | `trials_summary_{channel_lower}.json` |
| 저장 스키마 / Schema | `[{number, value, state, params}, ...]` |

#### `apply_phase2_params(cfg: dict, params: dict) -> dict`

```python
cfg = apply_phase2_params(cfg, params)
```

| 항목 / Item | 내용 / Detail |
|---|---|
| 입력 `params` 필수 키 | `learning_rate`, `batch_size`, `weight_decay`, `epochs`, `dropout`, `hidden_dim` |
| ResNet-50 추가 키 | `mid_dim` (있으면 적용, 없으면 스킵) |
| 적용 대상 / Applied to | `cfg["phase2"]["learning_rate/batch_size/weight_decay/epochs"]`, `cfg["phase2"]["heads"][backbone]["dropout/hidden_dim"]` |
| **Backbone 불변식** | `cfg["model"]["backbone"]` 값은 변경되지 않는다 / Must not be modified |
| 반환 / Return | 수정된 `cfg` dict (in-place + return) |
| **실패 / Fail** | `KeyError` — 필수 params 키 누락 |

---
