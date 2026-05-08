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

---

## 1. 전체 데이터 흐름 / Full Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  config/config.json  ──►  utils_config.load_config()  ──►  cfg: dict   │
│                               (모든 모듈이 cfg를 수령)                     │
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
validate_config(cfg)                         # → bool  (실패 시 ValueError)
create_directories(cfg)                      # → None  (storage.* 경로 생성)
val = get_nested(cfg, "phase2.learning_rate") # → Any   (None-safe dot-notation)
```

### 반환 타입 / Return Type

| 항목 / Item | 타입 / Type | 보장 / Guarantee |
|---|---|---|
| `load_config()` 반환값 | `dict` | 경로 절대화 + device 자동 감지 완료 |
| `cfg["system"]["device"]` | `str` | `"cpu"` / `"cuda"` / `"mps"` (절대 `"auto"` 아님) |
| `cfg["storage"]["*_dir"]` | `str` | 절대 경로 (root 기준 해소 완료) |

### 금지 패턴 / Prohibited

```python
# ❌ 삭제된 패턴 — 사용 금지
config = load_config()   # ConfigManager 반환 — 더 이상 존재하지 않음
cfg = config.config
config.get("phase2.learning_rate")
```

---

## 3. 데이터 경계 계약 / Data Boundary Contracts

### 3.1 파일 → `preprocess()` → Dataset

| 경계 / Boundary | 타입 / Type | 형상 / Shape | 값 범위 / Range | 색상 공간 / Color |
|---|---|---|---|---|
| 파일 로드 후 / After file load | `np.ndarray` | `(H, W, 3)` | `[0, 255]` uint8 | **BGR** (`cv2.imread`) |
| `preprocess()` 출력 | `np.ndarray` | `(128, 128, 3)` | `[0.0, 1.0]` float32 | **BGR** |
| 증강 후 / After augmentation | `np.ndarray` | `(128, 128, 3)` | `[0.0, 1.0]` float32 | **BGR** |
| Tensor 변환 후 / After to_tensor | `torch.Tensor` | `(3, 128, 128)` | `[0.0, 1.0]` float32 | **BGR** (C,H,W) |

> ⚠️ **SSOT-CS01**: 파일 로드부터 모델 입력까지 BGR을 유지한다. RGB 변환 금지.

### 3.2 `ContrastiveDataset` 출력 계약

```python
view1, view2 = dataset[i]
```

| 출력 / Output | 타입 / Type | 형상 / Shape | 범위 / Range | 비고 / Note |
|---|---|---|---|---|
| `view1` | `torch.Tensor` | `(3, 128, 128)` | `[0.0, 1.0]` | 동일 이미지 다른 증강 |
| `view2` | `torch.Tensor` | `(3, 128, 128)` | `[0.0, 1.0]` | Positive pair |

### 3.3 `CMYKDataset` 출력 계약

```python
image, label = dataset[i]
```

| 출력 / Output | 타입 / Type | 형상 / Shape | 범위 / Range | 비고 / Note |
|---|---|---|---|---|
| `image` | `torch.Tensor` | `(3, 128, 128)` | `[0.0, 1.0]` | train split만 증강 적용 |
| `label` | `int` | scalar | `[0, 5]` | ordinal 6-class |

### 3.4 DataLoader 배치 계약

| Phase | 배치 타입 / Type | 형상 / Shape |
|---|---|---|
| Phase 0 | `(Tensor, Tensor)` | `(B, 3, 128, 128)`, `(B, 3, 128, 128)` |
| Phase 2 | `(Tensor, Tensor)` | `(B, 3, 128, 128)`, `(B,)` int |

---

## 4. 모델 경계 계약 / Model Boundary Contracts

### 4.1 `GrayspotModel` 입출력

| Phase | 입력 / Input | 출력 / Output | 비고 / Note |
|---|---|---|---|
| Phase 0 | `(B, 3, 128, 128)` float32 | `(B, 128)` float32 | L2-정규화된 projection vector |
| Phase 2 | `(B, 3, 128, 128)` float32 | `(B, 6)` float32 | Raw logits — Softmax 없음 |

### 4.2 생성 계약 / Construction Contract

```python
# Phase 0 모델 생성
model = GrayspotModel(cfg, phase=0)
# 필수 cfg 키: model.backbone, phase0.projection_dim, phase0.hidden_dim

# Phase 2 모델 생성
model = GrayspotModel(cfg, phase=2)
# 필수 cfg 키: model.backbone, model.frozen_backbone,
#              data.num_levels, phase2.hidden_dim, phase2.dropout
```

### 4.3 `build_backbone()` 계약

```python
backbone, feature_dim = build_backbone(cfg)
```

| Backbone | `feature_dim` | Hard SSOT |
|---|---|---|
| `efficientnet_b0` | `1280` | ✅ |
| `resnet50` | `2048` | ✅ |

### 4.4 Phase 전환 계약 / Phase Transition Contract

```python
model.switch_to_phase2(backbone_path: Path)
# 입력: phase0_backbone_{channel}_{tag}.pt
# 동작: backbone.* 키만 로드 (strict=False)
# 결과: Backbone weights 유지 + ClassifierHead 새로 초기화
```

---

## 5. 학습 경계 계약 / Training Boundary Contracts

### 5.1 `Phase0Trainer`

```python
trainer = Phase0Trainer(model, cfg, channel="Y", device=device)
history = trainer.train(loader)           # → List[dict]  에폭별 기록
trainer.save_backbone()                   # → Path  저장 경로 반환
```

| 경계 / Boundary | 입력 타입 / Input | 출력 타입 / Output |
|---|---|---|
| `train()` 입력 | `DataLoader` — `(B,3,128,128), (B,3,128,128)` | `List[{"epoch", "loss"}]` |
| `save_backbone()` 출력 | — | `data_set/models/phase0_backbone_{ch}_{tag}.pt` |

**필수 cfg 키**: `phase0.epochs`, `phase0.batch_size`, `phase0.learning_rate`,
`phase0.weight_decay`, `phase0.temperature`, `train.optimizer`, `train.scheduler`,
`train.gradient_clip`, `train.eta_min`, `train.seed`, `storage.models_dir`

### 5.2 `Phase2Trainer`

```python
trainer = Phase2Trainer(model, cfg, channel="Y", device=device, train_ds=train_ds)
history = trainer.train(train_loader, val_loader)  # → List[dict]
trainer.save_history(history)                       # → Path  CSV 저장
```

| 경계 / Boundary | 입력 타입 / Input | 출력 타입 / Output |
|---|---|---|
| `train()` 입력 | `DataLoader` × 2 — `(B,3,128,128), (B,)` | `List[{"epoch", "train_loss", "val_acc", ...}]` |
| 체크포인트 | 내부 자동 저장 | `data_set/models/best_{ch}.pt` |
| `save_history()` 출력 | `List[dict]` | `data_set/reports/phase2_history_{ch}.csv` |

**필수 cfg 키**: `phase2.epochs`, `phase2.batch_size`, `phase2.learning_rate`,
`phase2.weight_decay`, `phase2.early_stopping.*`, `train.optimizer`, `train.scheduler`,
`train.gradient_clip`, `train.eta_min`, `train.seed`, `storage.models_dir`, `storage.reports_dir`

### 5.3 손실 함수 계약 / Loss Function Contract

```python
from training import InfoNCELoss, get_loss

# Phase 0
loss_fn = InfoNCELoss(temperature=cfg["phase0"]["temperature"])
loss = loss_fn(z1, z2)
# z1, z2: (B, projection_dim)  L2-normalized — 반드시 정규화 후 전달

# Phase 2
loss_fn = get_loss(cfg, phase=2, train_samples=train_ds.samples)
loss = loss_fn(logits, labels)
# logits: (B, 6)  raw logits (Softmax 미적용 상태)
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
| 생산자 / Producer | `Phase2Trainer.train()` 내부 |
| 소비자 / Consumer | `Evaluator`, `GrayspotPredictor.load_model()` |

### 6.3 추론 시 로드 계약

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint, strict=False)
# weights_only=True : pickle 보안
# strict=False      : 버전 간 키 불일치 허용
```

---

## 7. 평가 경계 계약 / Evaluation Boundary Contracts

### 7.1 `Evaluator` 입력 계약

```python
evaluator = Evaluator(
    model       = model,            # nn.Module, model.eval() 상태
    labeled_dir = Path("data_set/labeled"),
    labels_csv  = Path("data_set/labels_v0.csv"),
    output_dir  = Path("outputs/reports"),
    device      = device,
    cfg         = cfg,              # swing_thresholds + confidence_thresholds 주입
)
```

| 입력 / Input | 타입 / Type | 제약 / Constraint |
|---|---|---|
| `model` | `nn.Module` | 반드시 `model.eval()` 상태 |
| 이미지 배치 | `Tensor (B, 3, 128, 128)` | BGR float32 [0, 1] — 학습과 동일 |
| 레이블 | `Tensor (B,)` | int [0, 5] |

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
| `summary.overall` | `ChannelMetrics` | 전 채널 집계 지표 |
| `summary.by_channel[ch]` | `ChannelMetrics` | 채널별 지표 |
| `summary.targets["swing_*_retry"]` | `float` | cfg 주입값 (없으면 기본값) |

### 7.4 `ChannelMetrics` 구조

```python
@dataclass
class ChannelMetrics:
    accuracy : float       # 전체 정확도
    macro_f1 : float       # Macro-averaged F1
    mae      : float       # Mean Absolute Error (ordinal)
    n_samples: int
    per_class: List[PerClassMetric]  # level별 precision/recall/f1/support

    @property
    def acc_pass(self) -> bool: ...  # accuracy >= TARGET_PER_COLOR_ACC (0.85)
    def f1_pass (self) -> bool: ...  # macro_f1  >= TARGET_PER_CLASS_F1  (0.80)
    def mae_pass(self) -> bool: ...  # mae       <= TARGET_MAE            (0.50)
```

### 7.5 `determine_swing_feedback()` 계약

```python
from evaluation import determine_swing_feedback

feedback = determine_swing_feedback(summary)
```

| 반환값 / Return | 조건 / Condition | 다음 단계 / Next Step |
|---|---|---|
| `"pass"` | 모든 목표 달성 | 시스템 종료 |
| `"retry_phase2"` | `accuracy < swing_acc_retry` (0.80) | Phase 2 재학습 |
| `"retry_phase0"` | `macro_f1 < swing_f1_retry` (0.70) | Phase 0 → Phase 2 재학습 |

---

## 8. Fail-Fast 집행 포인트 / Fail-Fast Enforcement Points

모든 경계에서 아래 조건은 **즉시 예외를 발생시켜야 한다**. 우회·임시 생성 금지.
At every boundary, the following conditions **must raise an exception immediately**. No fallbacks.

| 위치 / Location | 조건 / Condition | 코드 / Code | 예외 / Exception |
|---|---|---|---|
| `run_phase2.py` 시작 | `phase0_backbone_{ch}_{tag}.pt` 미존재 | `SSOT-FF01` | `FileNotFoundError` |
| `Evaluator.run()` 시작 | `best_{ch}.pt` 미존재 | `SSOT-FF01` | `FileNotFoundError` |
| `GrayspotModel.__init__` | `cfg["data"]["num_levels"]` 키 미존재 | `SSOT-CF01` | `KeyError` |
| `Phase0Trainer.train()` | DataLoader 배치 언패킹 실패 (형상 불일치) | `SSOT-CS01` | `ValueError` |
| `InfoNCELoss.forward()` | z1, z2 가 L2-정규화되지 않은 경우 | — | `RuntimeError` (수치 이상) |
| `switch_to_phase2()` | backbone 키 0개 로드 (구조 불일치) | `SSOT-FF01` | `RuntimeError` |
| `validate_config()` | 필수 섹션 누락 (`data`, `model`, `phase2` 등) | `SSOT-CF01` | `ValueError` |

---

## 9. 모듈별 필수 Config 키 요약 / Required Config Keys per Module

| 모듈 / Module | 필수 config 키 / Required Keys |
|---|---|
| `data.dataset` | `data.channels`, `data.num_levels`, `data.image_size`, `data.split_ratios.*`, `storage.labeled_dir`, `train.seed` |
| `data.augmentation` (Phase 0) | `phase0.augmentation.*` (flip/crop/color_jitter/contrast/blur) |
| `data.augmentation` (Phase 2) | `phase2.augmentation.*` (flip/brightness/noise), `phase2.oversample` |
| `models.grayspot_model` | `model.backbone`, `model.frozen_backbone`, `data.num_levels`, `phase0.projection_dim`, `phase0.hidden_dim`, `phase2.hidden_dim`, `phase2.dropout` |
| `training.trainer` (Phase 0) | `phase0.epochs`, `phase0.batch_size`, `phase0.learning_rate`, `phase0.weight_decay`, `phase0.temperature`, `train.optimizer`, `train.scheduler`, `train.gradient_clip`, `train.eta_min`, `storage.models_dir` |
| `training.trainer` (Phase 2) | `phase2.epochs`, `phase2.batch_size`, `phase2.learning_rate`, `phase2.weight_decay`, `phase2.early_stopping.*`, `train.*`, `storage.models_dir`, `storage.reports_dir` |
| `training.losses` | `phase0.temperature`, `data.num_levels` |
| `evaluation.evaluator` | `inference.confidence_thresholds.*`, `evaluation.swing_thresholds.*` |
| `evaluation.metrics` | `evaluation.targets.*`, `data.num_levels` |
| `tuning.optuna_tuner` | `optuna.*`, `system.device`, `train.seed` |
| `tuning.search_space` | `optuna.search_space.*` |
| `utils.utils_config` | — (config 자체를 로드하므로 의존 없음) |
| `utils.utils_model` | `model.backbone`, `storage.models_dir`, `system.device` (`build_model` 사용 시) |
| `scripts.run_phase0` | `data.channels`, `system.device`, `train.seed`, `storage.*` |
| `scripts.run_phase2` | `data.channels`, `system.device`, `train.seed`, `storage.*` |
| `scripts.run_baseline` | `data.channels`, `system.device`, `train.seed`, `storage.*` |
| `scripts.run_optuna` | `optuna.*`, `system.device` |

---

**Version**: 0.1.0
**Last Updated**: 2026-05-08
**Python**: 3.11.5
**PyTorch**: 2.x
**Applies to**: CMYK Grayspot Detection System v0.1.0+
