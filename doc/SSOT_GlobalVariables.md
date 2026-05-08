# SSOT Global Variables — 전역 변수 및 하드코딩 정책 / Global Variables and Hardcoding Policy

CMYK Grayspot Detection System 의 모든 전역 변수, 상수, 하드코딩 값의 Hard/Soft SSOT 분류에 관한 단일 진실 공급원.

This document catalogs every global variable, constant, and hardcoded value and classifies them as Hard SSOT, Soft SSOT, or Dead Config.

> **목적 / Purpose**: 전역 변수/상수의 의미와 Hard/Soft 분류 정의
> **역할 / Role**: "What" — 변경 시 재학습 필요 여부 판단의 기준
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Config_Resolution.md](SSOT_Config_Resolution.md)

---

## Table of Contents / 목차

1. [분류 기준 / Classification Criteria](#1-분류-기준--classification-criteria)
2. [Hard SSOT 변수 / Hard SSOT Variables](#2-hard-ssot-변수--hard-ssot-variables)
3. [Soft SSOT 변수 / Soft SSOT Variables](#3-soft-ssot-변수--soft-ssot-variables)
4. [하드코딩 값 / Hardcoded Values](#4-하드코딩-값--hardcoded-values)
5. [Dead Config 전체 목록 / Complete Dead Config List](#5-dead-config-전체-목록--complete-dead-config-list)
6. [변경 영향 요약 / Change Impact Summary](#6-변경-영향-요약--change-impact-summary)

---

## 1. 분류 기준 / Classification Criteria

### Hard SSOT — 재학습 필수 / Retrain Required

변경 시 이미 학습된 모든 결과가 무효화되어 재학습이 필요한 파라미터.

Changes invalidate all trained results downstream — retraining is required.

**해당 조건 / Applies when:**

1. 모델 **구조**가 달라지는 경우 (`num_levels`, `backbone`, `hidden_dim`, `projection_dim`)
2. 모델 **입력 분포**가 달라지는 경우 (`image_size`, 정규화 방식, 색상 공간)
3. **학습 결과의 의미**가 달라지는 경우 (Phase 순서, loss function, temperature)

### Soft SSOT — 재학습 선택 / Retrain Optional

변경 시 성능이나 속도에만 영향을 주는 파라미터.

Changes only affect performance or speed — retraining is optional.

**해당 조건 / Applies when:**

1. **성능만** 달라지는 하이퍼파라미터 (`lr`, `weight_decay`, `batch_size`)
2. **실행 환경에만** 영향을 미치는 파라미터 (`num_workers`, `pin_memory`, `device`)

---

## 2. Hard SSOT 변수 / Hard SSOT Variables

변경 시 **재학습 필수**. / Changing **requires retraining**.

| 변수 / Variable | 값 / Value | config 키 / Key | 위치 / Location | 영향 / Impact |
|---|---|---|---|---|
| `NUM_LEVELS` | `6` | `data.num_levels` 🟢 | `evaluation/metrics.py` | ClassifierHead 출력 차원 / Output dim |
| `num_levels` (model) | `6` | `data.num_levels` 🟢 | `models/grayspot_model.py` | ClassifierHead 출력 차원 |
| `image_size` | `128` | `data.image_size` 🟢 | `data/dataset.py` | 모델 입력 크기 / Input size |
| `backbone` | `"efficientnet_b0"` | `model.backbone` 🟢 | `models/backbone.py` | 전체 구조 / Full architecture |
| `feature_dim` (effb0) | `1280` | 🟡 하드코딩 | `models/backbone.py` | Head 입력 차원 / Head input dim |
| `feature_dim` (res50) | `2048` | 🟡 하드코딩 | `models/backbone.py` | Head 입력 차원 |
| `projection_dim` | `128` | `phase0.projection_dim` 🟢 | `models/grayspot_model.py` | ProjectionHead 출력 차원 |
| `proj_hidden` (Phase 0 head) | `256` | `phase0.hidden_dim` 🟢 | `models/grayspot_model.py` | ProjectionHead 중간 차원 |
| `cls_hidden` (Phase 2 head) | `256` | `phase2.hidden_dim` 🟢 | `models/grayspot_model.py` | ClassifierHead 중간 차원 |
| `temperature` (τ) | `0.1` | `phase0.temperature` 🟢 | `training/contrastive_loss.py` | InfoNCE 손실 스케일 |
| Phase 순서 | Phase 0 → Phase 2 | — 🟡 하드코딩 | `scripts/run_phase2.py` | backbone 의존성 / Backbone dependency |
| 색상 공간 / Color space | BGR | — 🟡 하드코딩 | `data/preprocessing.py` | 입력 분포 / Input distribution |
| 정규화 / Normalization | `/ 255.0` only | — 🟡 하드코딩 | `data/preprocessing.py` | 입력 스케일 / Input scale |

**Hard SSOT 수 / Count: 13개 (config 연결 8개, 하드코딩 5개)**

---

## 3. Soft SSOT 변수 / Soft SSOT Variables

변경 시 **성능 또는 속도만 영향**. / Changing only affects **performance or speed**.

| 변수 / Variable | 값 / Value | config 키 / Key | 위치 / Location | 영향 / Impact |
|---|---|---|---|---|
| `batch_size` (phase0) | `16` | `phase0.batch_size` 🟢 | `training/trainer.py` | 학습 속도 / Speed |
| `batch_size` (phase2) | `16` | `phase2.batch_size` 🟢 | `training/trainer.py` | 학습 속도 |
| `lr` (phase0) | `1e-3` | `phase0.learning_rate` 🟢 | `training/trainer.py` | 성능 / Performance |
| `lr` (phase2) | `1e-4` | `phase2.learning_rate` 🟢 | `training/trainer.py` | 성능 |
| `weight_decay` (phase0) | `1e-5` | `phase0.weight_decay` 🟢 | `training/trainer.py` | 정규화 / Regularization |
| `weight_decay` (phase2) | `1e-4` | `phase2.weight_decay` 🟢 | `training/trainer.py` | 정규화 |
| `dropout` | `0.3` | `phase2.dropout` 🟢 | `models/grayspot_model.py` | 정규화 |
| `epochs` (phase0) | `10` | `phase0.epochs` 🟢 | `training/trainer.py` | 수렴 / Convergence |
| `epochs` (phase2) | `30` | `phase2.epochs` 🟢 | `training/trainer.py` | 수렴 |
| `num_workers` | `4` | `train.num_workers` 🟢 | DataLoader (`run_phase2.py`) | 데이터 로딩 속도 |
| `pin_memory` | `true` | `train.pin_memory` 🟢 | DataLoader | GPU 전송 속도 |
| `device` | `"auto"` | `system.device` 🟢 | `scripts/run_*.py` | 연산 장치 |
| `oversample` | `true` | `phase2.oversample` 🟢 | `data/dataset.py` | 클래스 균형 / Class balance |
| `seed` | `42` | `train.seed` 🟢 | `scripts/run_*.py` | 재현성 / Reproducibility |
| `gradient_clip` | `1.0` | `train.gradient_clip` 🟢 | `training/trainer.py` | 학습 안정성 / Stability |
| `early_stopping.patience` | `5` | `phase2.early_stopping.patience` 🟢 | `Phase2Trainer` | 과적합 방지 / Overfitting |
| `optimizer` | `"adamw"` | `train.optimizer` 🟢 | `training/trainer.py` `_build_optimizer()` | optimizer 선택 |
| `scheduler` | `"cosine"` | `train.scheduler` 🟢 | `training/trainer.py` `_build_scheduler()` | scheduler 선택 |
| `momentum` | `0.9` | `train.momentum` 🟢 | `_build_optimizer()` — SGD 선택 시 | SGD 모멘텀 |
| `betas` | `[0.9, 0.999]` | `train.betas` 🟢 | `_build_optimizer()` — AdamW | AdamW beta 파라미터 |
| `gamma` | `0.1` | `train.gamma` 🟢 | `_build_scheduler()` — StepLR 선택 시 | StepLR 감쇠율 |

**Soft SSOT 수 / Count: 21개**

---

## 4. 하드코딩 값 / Hardcoded Values

config 키 없이 코드에 직접 고정된 값들. / Values fixed directly in code without config keys.

### 4.1 Hard SSOT 하드코딩 / Hardcoded Hard SSOT

| 값 / Value | 위치 / Location | 비고 / Note |
|---|---|---|
| `feature_dim = 1280` (EfficientNet-B0) | `models/backbone.py` | backbone 선택 시 결정됨 |
| `feature_dim = 2048` (ResNet50) | `models/backbone.py` | backbone 선택 시 결정됨 |
| BGR 색상 공간 (`cv2.imread`) | `data/preprocessing.py` | 변경 시 재학습 필수 |
| `/ 255.0` 정규화 | `data/preprocessing.py` | 변경 시 재학습 필수 |
| Phase 순서 (0 → 2) | `scripts/run_phase2.py` | 아키텍처 의존성 |

### 4.2 Soft SSOT 하드코딩 / Hardcoded Soft SSOT

| 값 / Value | 위치 / Location | 비고 / Note |
|---|---|---|
| aug fallback constants (`_SUP_*`, `_CON_*`) | `data/augmentation.py` | 🟢 모든 파라미터 config 연결 완료 — 상수는 fallback 기본값으로만 유지 |
| `CONF_THRESH_AUTO/WARN/MANUAL` | `evaluation/metrics.py` | 🟢 `Evaluator(cfg=...)` 에서 config 소비 — 상수는 backward-compat 및 fallback용으로 유지 |

---

## 5. Dead Config 전체 목록 / Complete Dead Config List

config.json에 선언되었으나 코드에서 소비되지 않는 주요 키.
Key config.json entries declared but not consumed by code.

현재 잔존 Dead Config 는 없음 (구현 대상 또는 삭제 완료).
No remaining actionable Dead Config — all resolved or removed.

**이전 대비 해소된 Dead Config (리팩토링 완료)**:

| config 키 / Key | 해소 내용 / Resolution |
|---|---|
| `data.split_ratios` | `dataset.py` 에서 `cfg["data"]["split_ratios"]` 소비 |
| `model.frozen_backbone` | `grayspot_model.py` 에서 backbone freeze 구현 |
| `train.gradient_clip` | `Phase0Trainer` / `Phase2Trainer` 에서 `clip_grad_norm_` 적용 |
| `phase2.early_stopping.*` | `Phase2Trainer.train()` 에서 early stopping 구현 |
| `cuda.deterministic` | `set_seed()` 에서 `torch.backends.cudnn.deterministic` 소비 |
| `cuda.benchmark` | `set_seed()` 에서 `torch.backends.cudnn.benchmark` 소비 |
| `phase0.weight_decay` | `Phase0Trainer` AdamW optimizer의 `weight_decay` 인자로 소비 |
| `phase0.augmentation.color_jitter` | `augment_contrastive()` 의 `aug_cfg` 매개변수로 소비 |
| `phase0.augmentation.blur_prob` | `augment_contrastive()` 의 `aug_cfg` 매개변수로 소비 |
| `evaluation.swing_thresholds.*` | `EvaluationSummary.targets` 에 주입 → `determine_swing_feedback()` 소비 |
| `phase0.hidden_dim` | `grayspot_model.py` 에서 ProjectionHead `hidden_dim=proj_hidden`으로 연결 (버그 수정) |
| `train.optimizer` | `trainer.py` `_build_optimizer()` factory 구현 — AdamW/SGD 선택 지원 |
| `train.scheduler` | `trainer.py` `_build_scheduler()` factory 구현 — cosine/step 선택 지원 |
| `train.momentum` | `_build_optimizer()` SGD 경로에서 소비 |
| `train.betas` | `_build_optimizer()` AdamW 경로에서 소비 |
| `train.gamma` | `_build_scheduler()` StepLR 경로에서 소비 |
| `optuna.direction` | `run_optuna()` 에서 `cfg.get("optuna", {}).get("direction", "maximize")` 소비 |
| `optuna.sampler` | `run_optuna()` 에서 sampler factory 구현 — tpe/random 선택 지원 |
| `optuna.search_space.learning_rate` | `get_phase2_search_space(trial, cfg)` 에서 소비 |
| `optuna.search_space.weight_decay` | `get_phase2_search_space(trial, cfg)` 에서 소비 |
| `optuna.search_space.dropout` | `search_space.py` 에서 ss dict에 포함 (future use) |
| `optuna.search_space.batch_size` | `get_phase2_search_space(trial, cfg)` 에서 소비 (config에 신규 추가) |
| `optuna.search_space.epochs` | `get_phase2_search_space(trial, cfg)` 에서 소비 (config에 신규 추가) |

**config.json에서 제거된 Dead Config (플레이스홀더 삭제)**:

| config 키 / Key | 제거 사유 / Reason for Removal |
|---|---|
| `system.project_name` | 코드에서 미사용 — 메타 정보만 / Not consumed by code |
| `system.version` | 코드에서 미사용 — 메타 정보만 |
| `system.mixed_precision` | 미구현 기능 플레이스홀더 / Unimplemented feature placeholder |
| `storage.raw_dir` | 코드에서 미사용 |
| `storage.outputs_dir` | 코드에서 미사용 (optuna_tuner.py 내부 고정 경로 사용) |
| `storage.model_checkpoint` | 코드에서 동적 생성 (`best_{channel}.pt`) |
| `storage.model_last` | 코드에서 미사용 |
| `storage.history_file` | 코드에서 미사용 |
| `model.weights` | 커스텀 가중치 로드 미구현 |
| `phase0.enabled` | 코드에서 미사용 (Phase 0 항상 수동 실행) |
| `phase0.warmup_epochs` | warmup 스케줄러 미구현 |
| `phase0.augmentation.normalize` | `preprocessing.py` 고정 처리 |
| `phase2.warmup_epochs` | warmup 스케줄러 미구현 |
| `phase2.class_weights` | `losses.py` 내부 자체 처리 |
| `phase2.loss_type` | 항상 CrossEntropy 사용 — Focal Loss 미구현 |
| `phase2.focal_alpha` | Focal Loss 미구현 |
| `phase2.focal_gamma` | Focal Loss 미구현 |
| `train.grad_accumulation_steps` | 미구현 기능 플레이스홀더 |
| `evaluation.metrics` | 코드 내부 고정 목록 |
| `cuda.visible_devices` | 코드에서 미사용 |
| `logging.log_interval` | 코드에서 미사용 |
| `logging.val_log_interval` | 코드에서 미사용 |
| `logging.log_file` | `setup_logging()` 이 타임스탬프 기반 파일명 자동 생성 |

---

## 6. 변경 영향 요약 / Change Impact Summary

### 6.1 변경 불가 (재학습 필수) / Cannot Change Without Retraining

```
num_levels = 6           → ClassifierHead 출력 차원 변경
image_size = 128         → 모든 전처리 파이프라인 영향
backbone = efficientnet  → feature_dim 변경 → head 구조 변경
projection_dim = 128     → Phase 0 head 구조 변경
phase0.hidden_dim = 256  → ProjectionHead 중간 차원 변경
phase2.hidden_dim = 256  → ClassifierHead 중간 차원 변경
temperature = 0.1        → InfoNCE 손실 스케일 변경
색상 공간 = BGR          → 입력 분포 전면 변경
정규화 = /255.0 only     → 입력 스케일 변경
```

### 6.2 변경 가능 (성능 조정) / Can Change for Tuning

```
lr, weight_decay, batch_size, dropout, epochs
num_workers, pin_memory, device
oversample, seed
optimizer (adamw | sgd), scheduler (cosine | step)
momentum, betas, gamma, eta_min
```

### 6.3 해결 완료 Dead Config / Resolved Dead Config

```
✅ DONE  data.split_ratios              → dataset.py 소비 완료
✅ DONE  cuda.deterministic             → set_seed() 소비 완료
✅ DONE  cuda.benchmark                 → set_seed() 소비 완료
✅ DONE  phase2.early_stopping.*        → Phase2Trainer 소비 완료
✅ DONE  train.gradient_clip            → Phase0/2Trainer 소비 완료
✅ DONE  model.frozen_backbone          → GrayspotModel 소비 완료
✅ DONE  phase0.weight_decay            → Phase0Trainer AdamW 소비 완료
✅ DONE  phase0.augmentation.color_jitter → augment_contrastive() aug_cfg 소비 완료
✅ DONE  phase0.augmentation.blur_prob  → augment_contrastive() aug_cfg 소비 완료
✅ DONE  evaluation.swing_thresholds.*  → determine_swing_feedback() 소비 완료
✅ DONE  phase0.hidden_dim              → GrayspotModel ProjectionHead hidden_dim 버그 수정
✅ DONE  train.optimizer                → _build_optimizer() factory 구현
✅ DONE  train.scheduler                → _build_scheduler() factory 구현
✅ DONE  train.momentum / betas / gamma → factory 함수 내 소비
✅ DONE  optuna.direction               → run_optuna() cfg 소비
✅ DONE  optuna.sampler                 → run_optuna() sampler factory 구현
✅ DONE  optuna.search_space.*          → get_phase2_search_space(trial, cfg) 소비

🗑️ REMOVED  system.project_name / version / mixed_precision
🗑️ REMOVED  storage.raw_dir / outputs_dir / model_checkpoint / model_last / history_file
🗑️ REMOVED  model.weights
🗑️ REMOVED  phase0.enabled / warmup_epochs / augmentation.normalize
🗑️ REMOVED  phase2.warmup_epochs / class_weights / loss_type / focal_alpha / focal_gamma
🗑️ REMOVED  train.grad_accumulation_steps
🗑️ REMOVED  evaluation.metrics
🗑️ REMOVED  cuda.visible_devices
🗑️ REMOVED  logging.log_interval / val_log_interval / log_file
```

---

**Version**: 0.4.0
**Last Updated**: 2026-05-08
**Applies to**: CMYK Grayspot Detection System v0.1.0+
