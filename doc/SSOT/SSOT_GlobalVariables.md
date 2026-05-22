---
type: ssot
domain: global_variables
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "SSOT_Config_Resolution.md"
---

# SSOT Global Variables — 전역 변수 및 하드코딩 정책 / Global Variables and Hardcoding Policy

CMYK Grayspot Detection System 의 모든 전역 변수, 상수, 하드코딩 값의 Hard/Soft SSOT 분류에 관한 단일 진실 공급원.

This document catalogs every global variable, constant, and hardcoded value and classifies them as Hard SSOT, Soft SSOT, or Dead Config.

> **목적 / Purpose**: 전역 변수/상수의 의미와 Hard/Soft 분류 정의
> **역할 / Role**: "What" — 변경 시 재학습 필요 여부 판단의 기준
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Config_Resolution.md](SSOT_Config_Resolution.md)

---

## 1. 분류 기준 / Classification Criteria

### Hard SSOT — 재학습 필수 / Retrain Required

변경 시 이미 학습된 모든 결과가 무효화되어 재학습이 필요한 파라미터.
Parameters whose change invalidates all previously trained results — retraining is required.

**해당 조건 / Applies when:**

1. 모델 **구조**가 달라지는 경우 (`num_levels`, `backbone`, `hidden_dim`, `projection_dim`) / Model **structure** changes
2. 모델 **입력 분포**가 달라지는 경우 (`image_size`, 정규화 방식, 색상 공간) / Model **input distribution** changes (e.g., `image_size`, normalization, color space)
3. **학습 결과의 의미**가 달라지는 경우 (Phase 순서, loss function, temperature) / **Semantic meaning of training results** changes (Phase ordering, loss function, temperature)

### Soft SSOT — 재학습 선택 / Retrain Optional

변경 시 성능이나 속도에만 영향을 주는 파라미터.
Parameters whose change only affects performance or speed — retraining is optional.

**해당 조건 / Applies when:**

1. **성능만** 달라지는 하이퍼파라미터 (`lr`, `weight_decay`, `batch_size`) / Only **performance** changes (`lr`, `weight_decay`, `batch_size`)
2. **실행 환경에만** 영향을 미치는 파라미터 (`num_workers`, `pin_memory`, `device`) / Only **execution environment** is affected (`num_workers`, `pin_memory`, `device`)

---

## 2. Hard SSOT 변수 / Hard SSOT Variables

변경 시 **재학습 필수**. / Changing **requires retraining**.

| 변수 / Variable | 값 / Value | config 키 / Key | 위치 / Location | 영향 / Impact |
|---|---|---|---|---|
| `NUM_LEVELS` | `6` | `data.num_levels` 🟢 | `evaluation/metrics.py` | ClassifierHead 출력 차원 / Output dim |
| `num_levels` (model) | `6` | `data.num_levels` 🟢 | `models/grayspot_model.py` | ClassifierHead 출력 차원 / Output dim |
| `image_size` | `128` | `data.image_size` 🟢 | `data/dataset.py` | 모델 입력 크기 / Input size |
| `backbone` | `"efficientnet_b0"` | `model.backbone` 🟢 | `models/backbone.py` | 전체 구조 / Full architecture |
| `feature_dim` (effb0) | `1280` | 🟡 하드코딩 / Hardcoded | `models/backbone.py` | Head 입력 차원 / Head input dim |
| `feature_dim` (res50) | `2048` | 🟡 하드코딩 / Hardcoded | `models/backbone.py` | Head 입력 차원 / Head input dim |
| `projection_dim` | `128` | `phase0.projection_dim` 🟢 | `models/grayspot_model.py` | ProjectionHead 출력 차원 / Output dim |
| `proj_hidden` (Phase 0 head) | `256` | `phase0.hidden_dim` 🟢 | `models/grayspot_model.py` | ProjectionHead 중간 차원 / Intermediate dim |
| `cls_hidden` (EffB0 head) | `256` | `phase2.heads.efficientnet_b0.hidden_dim` 🟢 | `models/grayspot_model.py` | ClassifierHead 최종 은닉 차원 / Final hidden dim |
| `cls_hidden` (Res50 head) | `256` | `phase2.heads.resnet50.hidden_dim` 🟢 | `models/grayspot_model.py` | ClassifierHead 최종 은닉 차원 / Final hidden dim |
| `mid_dim` (Res50 전용 / only) | `512` | `phase2.heads.resnet50.mid_dim` 🟢 | `models/grayspot_model.py` | ResNet-50 단계적 압축 중간 차원 / ResNet-50 staged compression dim |
| `temperature` (τ) | `0.1` | `phase0.temperature` 🟢 | `training/contrastive_loss.py` | InfoNCE 손실 스케일 / InfoNCE loss scale |
| Phase 순서 / Phase ordering | Phase 0 → Phase 2 | — 🟡 하드코딩 / Hardcoded | `scripts/run_phase2.py` | backbone 의존성 / Backbone dependency |
| 색상 공간 / Color space | BGR | — 🟡 하드코딩 / Hardcoded | `data/preprocessing.py` | 입력 분포 / Input distribution |
| 정규화 / Normalization | `/ 255.0` only | — 🟡 하드코딩 / Hardcoded | `data/preprocessing.py` | 입력 스케일 / Input scale |

**Hard SSOT 수 / Count: 15개 / 15 (config 연결 10개 / 10 config-linked, 하드코딩 5개 / 5 hardcoded)**

---

## 3. Soft SSOT 변수 / Soft SSOT Variables

변경 시 **성능 또는 속도만 영향**. / Changing only affects **performance or speed**.

| 변수 / Variable | 값 / Value | config 키 / Key | 위치 / Location | 영향 / Impact |
|---|---|---|---|---|
| `batch_size` (phase0) | `16` | `phase0.batch_size` 🟢 | `training/trainer.py` | 학습 속도 / Training speed |
| `batch_size` (phase2) | `16` | `phase2.batch_size` 🟢 | `training/trainer.py` | 학습 속도 / Training speed |
| `lr` (phase0) | `1e-3` | `phase0.learning_rate` 🟢 | `training/trainer.py` | 성능 / Performance |
| `lr` (phase2) | `1e-4` | `phase2.learning_rate` 🟢 | `training/trainer.py` | 성능 / Performance |
| `weight_decay` (phase0) | `1e-5` | `phase0.weight_decay` 🟢 | `training/trainer.py` | 정규화 / Regularization |
| `weight_decay` (phase2) | `1e-4` | `phase2.weight_decay` 🟢 | `training/trainer.py` | 정규화 / Regularization |
| `dropout` (EffB0) | `0.2` | `phase2.heads.efficientnet_b0.dropout` 🟢 | `models/grayspot_model.py` | 정규화 (compact head) / Regularization (compact head) |
| `dropout` (Res50) | `0.4` | `phase2.heads.resnet50.dropout` 🟢 | `models/grayspot_model.py` | 정규화 (staged head, 강하게) / Regularization (staged head, heavier) |
| `epochs` (phase0) | `10` | `phase0.epochs` 🟢 | `training/trainer.py` | 수렴 / Convergence |
| `epochs` (phase2) | `30` | `phase2.epochs` 🟢 | `training/trainer.py` | 수렴 / Convergence |
| `num_workers` | `4` | `train.num_workers` 🟢 | DataLoader (`run_phase2.py`) | 데이터 로딩 속도 / Data loading speed |
| `pin_memory` | `true` | `train.pin_memory` 🟢 | DataLoader | GPU 전송 속도 / GPU transfer speed |
| `device` | `"auto"` | `system.device` 🟢 | `scripts/run_*.py` | 연산 장치 / Compute device |
| `oversample` | `true` | `phase2.oversample` 🟢 | `data/dataset.py` | 클래스 균형 / Class balance |
| `seed` | `42` | `train.seed` 🟢 | `scripts/run_*.py` | 재현성 / Reproducibility |
| `gradient_clip` | `1.0` | `train.gradient_clip` 🟢 | `training/trainer.py` | 학습 안정성 / Training stability |
| `early_stopping.patience` | `5` | `phase2.early_stopping.patience` 🟢 | `Phase2Trainer` | 과적합 방지 / Overfitting prevention |
| `optimizer` | `"adamw"` | `train.optimizer` 🟢 | `training/trainer.py` `_build_optimizer()` | optimizer 선택 / Optimizer selection |
| `scheduler` | `"cosine"` | `train.scheduler` 🟢 | `training/trainer.py` `_build_scheduler()` | scheduler 선택 / Scheduler selection |
| `momentum` | `0.9` | `train.momentum` 🟢 | `_build_optimizer()` — SGD 선택 시 / when SGD selected | SGD 모멘텀 / SGD momentum |
| `betas` | `[0.9, 0.999]` | `train.betas` 🟢 | `_build_optimizer()` — AdamW | AdamW beta 파라미터 / AdamW beta parameters |
| `gamma` | `0.1` | `train.gamma` 🟢 | `_build_scheduler()` — StepLR 선택 시 / when StepLR selected | StepLR 감쇠율 / StepLR decay rate |

**Soft SSOT 수 / Count: 22개 / 22**

---

## 4. 하드코딩 값 / Hardcoded Values

config 키 없이 코드에 직접 고정된 값들. / Values fixed directly in code without config keys.

### 4.1 Hard SSOT 하드코딩 / Hardcoded Hard SSOT

| 값 / Value | 위치 / Location | 비고 / Note |
|---|---|---|
| `feature_dim = 1280` (EfficientNet-B0) | `models/backbone.py` | backbone 선택 시 결정됨 / Determined by backbone selection |
| `feature_dim = 2048` (ResNet50) | `models/backbone.py` | backbone 선택 시 결정됨 / Determined by backbone selection |
| BGR 색상 공간 (`cv2.imread`) / BGR color space | `data/preprocessing.py` | 변경 시 재학습 필수 / Retrain required on change |
| `/ 255.0` 정규화 / Normalization | `data/preprocessing.py` | 변경 시 재학습 필수 / Retrain required on change |
| Phase 순서 (0 → 2) / Phase ordering | `scripts/run_phase2.py` | 아키텍처 의존성 / Architecture dependency |

### 4.2 Soft SSOT 하드코딩 / Hardcoded Soft SSOT

| 값 / Value | 위치 / Location | 비고 / Note |
|---|---|---|
| aug fallback constants (`_SUP_*`, `_CON_*`) | `data/augmentation.py` | 🟢 모든 파라미터 config 연결 완료 — 상수는 fallback 기본값으로만 유지 / All params config-linked — constants kept as fallback defaults only |
| `CONF_THRESH_AUTO/WARN/MANUAL` | `evaluation/metrics.py` | 🟢 `Evaluator(cfg=...)` 에서 config 소비 — 상수는 backward-compat 및 fallback용으로 유지 / Config consumed in `Evaluator(cfg=...)` — constants kept for backward-compat and fallback |

---

## 5. Dead Config 전체 목록 / Complete Dead Config List

config.json에 선언되었으나 코드에서 소비되지 않는 주요 키. / Key config.json entries declared but not consumed by code.

현재 잔존 Dead Config 는 없음 (구현 대상 또는 삭제 완료). / No remaining actionable Dead Config — all resolved or removed.

**이전 대비 해소된 Dead Config (리팩토링 완료) / Previously resolved Dead Config (refactoring complete)**:

| config 키 / Key | 해소 내용 / Resolution |
|---|---|
| `data.split_ratios` | `dataset.py` 에서 `cfg["data"]["split_ratios"]` 소비 / consumed via `cfg["data"]["split_ratios"]` in `dataset.py` |
| `model.frozen_backbone` | `grayspot_model.py` 에서 backbone freeze 구현 / backbone freeze implemented in `grayspot_model.py` |
| `train.gradient_clip` | `Phase0Trainer` / `Phase2Trainer` 에서 `clip_grad_norm_` 적용 / `clip_grad_norm_` applied in `Phase0Trainer` / `Phase2Trainer` |
| `phase2.early_stopping.*` | `Phase2Trainer.train()` 에서 early stopping 구현 / early stopping implemented in `Phase2Trainer.train()` |
| `cuda.deterministic` | `set_seed()` 에서 `torch.backends.cudnn.deterministic` 소비 / consumed as `torch.backends.cudnn.deterministic` in `set_seed()` |
| `cuda.benchmark` | `set_seed()` 에서 `torch.backends.cudnn.benchmark` 소비 / consumed as `torch.backends.cudnn.benchmark` in `set_seed()` |
| `phase0.weight_decay` | `Phase0Trainer` AdamW optimizer의 `weight_decay` 인자로 소비 / consumed as `weight_decay` arg of `Phase0Trainer` AdamW optimizer |
| `phase0.augmentation.color_jitter` | `augment_contrastive()` 의 `aug_cfg` 매개변수로 소비 / consumed as `aug_cfg` parameter of `augment_contrastive()` |
| `phase0.augmentation.blur_prob` | `augment_contrastive()` 의 `aug_cfg` 매개변수로 소비 / consumed as `aug_cfg` parameter of `augment_contrastive()` |
| `evaluation.swing_thresholds.*` | `EvaluationSummary.targets` 에 주입 → `determine_swing_feedback()` 소비 / injected into `EvaluationSummary.targets` → consumed by `determine_swing_feedback()` |
| `phase0.hidden_dim` | `grayspot_model.py` 에서 ProjectionHead `hidden_dim=proj_hidden`으로 연결 (버그 수정) / linked to ProjectionHead `hidden_dim=proj_hidden` in `grayspot_model.py` (bug fix) |
| `train.optimizer` | `trainer.py` `_build_optimizer()` factory 구현 — AdamW/SGD 선택 지원 / `_build_optimizer()` factory implemented in `trainer.py` — AdamW/SGD selection supported |
| `train.scheduler` | `trainer.py` `_build_scheduler()` factory 구현 — cosine/step 선택 지원 / `_build_scheduler()` factory implemented — cosine/step selection supported |
| `train.momentum` | `_build_optimizer()` SGD 경로에서 소비 / consumed in SGD path of `_build_optimizer()` |
| `train.betas` | `_build_optimizer()` AdamW 경로에서 소비 / consumed in AdamW path of `_build_optimizer()` |
| `train.gamma` | `_build_scheduler()` StepLR 경로에서 소비 / consumed in StepLR path of `_build_scheduler()` |
| `optuna.direction` | `run_optuna()` 에서 `cfg.get("optuna", {}).get("direction", "maximize")` 소비 / consumed via `cfg.get("optuna", {}).get("direction", "maximize")` in `run_optuna()` |
| `optuna.sampler` | `run_optuna()` 에서 sampler factory 구현 — tpe/random 선택 지원 / sampler factory implemented in `run_optuna()` — tpe/random selection supported |
| `optuna.search_space.learning_rate` | `get_phase2_search_space(trial, cfg)` 에서 소비 / consumed in `get_phase2_search_space(trial, cfg)` |
| `optuna.search_space.weight_decay` | `get_phase2_search_space(trial, cfg)` 에서 소비 / consumed in `get_phase2_search_space(trial, cfg)` |
| `optuna.search_space.dropout` | `search_space.py` 에서 ss dict에 포함 (future use) / included in ss dict in `search_space.py` (future use) |
| `optuna.search_space.batch_size` | `get_phase2_search_space(trial, cfg)` 에서 소비 (config에 신규 추가) / consumed in `get_phase2_search_space(trial, cfg)` (newly added to config) |
| `optuna.search_space.epochs` | `get_phase2_search_space(trial, cfg)` 에서 소비 (config에 신규 추가) / consumed in `get_phase2_search_space(trial, cfg)` (newly added to config) |
| `optuna.search_space.resnet50.*` | `get_phase2_search_space(trial, cfg)` backbone별 분기 탐색 / backbone-branched search in `get_phase2_search_space` |
| `optuna.search_space.efficientnet_b0.*` | `get_phase2_search_space(trial, cfg)` backbone별 분기 탐색 / backbone-branched search in `get_phase2_search_space` |
| `optuna.pruner.n_warmup_steps` | `run_optuna()` 에서 `MedianPruner(n_warmup_steps=...)` 소비 / consumed as `MedianPruner(n_warmup_steps=...)` in `run_optuna()` |
| `phase2.heads.efficientnet_b0.*` | `grayspot_model.py` ClassifierHead 직접 압축 구조 소비 / consumed for direct-compression ClassifierHead in `grayspot_model.py` |
| `phase2.heads.resnet50.*` | `grayspot_model.py` ClassifierHead 단계적 압축 구조 소비 / consumed for staged-compression ClassifierHead in `grayspot_model.py` |

**config.json에서 제거된 Dead Config (플레이스홀더 삭제) / Dead Config removed from config.json (placeholders deleted)**:

| config 키 / Key | 제거 사유 / Reason for Removal |
|---|---|
| `system.project_name` | 코드에서 미사용 — 메타 정보만 / Not consumed by code — metadata only |
| `system.version` | 코드에서 미사용 — 메타 정보만 / Not consumed by code — metadata only |
| `system.mixed_precision` | 미구현 기능 플레이스홀더 / Unimplemented feature placeholder |
| `storage.raw_dir` | 코드에서 미사용 / Not consumed by code |
| `storage.outputs_dir` | 코드에서 미사용 (optuna_tuner.py 내부 고정 경로 사용) / Not consumed (fixed path used inside optuna_tuner.py) |
| `storage.model_checkpoint` | 코드에서 동적 생성 (`best_{channel}.pt`) / Dynamically generated in code (`best_{channel}.pt`) |
| `storage.model_last` | 코드에서 미사용 / Not consumed by code |
| `storage.history_file` | 코드에서 미사용 / Not consumed by code |
| `model.weights` | 커스텀 가중치 로드 미구현 / Custom weight loading not implemented |
| `phase0.enabled` | 코드에서 미사용 (Phase 0 항상 수동 실행) / Not consumed (Phase 0 always run manually) |
| `phase0.warmup_epochs` | warmup 스케줄러 미구현 / Warmup scheduler not implemented |
| `phase0.augmentation.normalize` | `preprocessing.py` 고정 처리 / Fixed handling in `preprocessing.py` |
| `phase2.warmup_epochs` | warmup 스케줄러 미구현 / Warmup scheduler not implemented |
| `phase2.class_weights` | `losses.py` 내부 자체 처리 / Handled internally in `losses.py` |
| `phase2.loss_type` | Focal Loss 삭제 — CrossEntropy 고정 사용 / Focal Loss deleted — CrossEntropy fixed |
| `phase2.focal_alpha` | Focal Loss 삭제 / Focal Loss deleted |
| `phase2.focal_gamma` | Focal Loss 삭제 / Focal Loss deleted |
| `train.grad_accumulation_steps` | 미구현 기능 플레이스홀더 / Unimplemented feature placeholder |
| `evaluation.metrics` | 코드 내부 고정 목록 / Fixed list inside code |
| `cuda.visible_devices` | 코드에서 미사용 / Not consumed by code |
| `logging.log_interval` | 코드에서 미사용 / Not consumed by code |
| `logging.val_log_interval` | 코드에서 미사용 / Not consumed by code |
| `logging.log_file` | `setup_logging()` 이 타임스탬프 기반 파일명 자동 생성 / `setup_logging()` auto-generates timestamp-based filename |

---

## 6. 변경 영향 요약 / Change Impact Summary

### 6.1 변경 불가 (재학습 필수) / Cannot Change Without Retraining

```
num_levels = 6           → ClassifierHead 출력 차원 변경 / ClassifierHead output dim changes
image_size = 128         → 모든 전처리 파이프라인 영향 / Affects all preprocessing pipeline
backbone = efficientnet  → feature_dim 변경 → head 구조 변경 / feature_dim changes → head structure changes
projection_dim = 128     → Phase 0 head 구조 변경 / Phase 0 head structure changes
phase0.hidden_dim = 256  → ProjectionHead 중간 차원 변경 / ProjectionHead intermediate dim changes
phase2.heads.{backbone}.hidden_dim = 256  → ClassifierHead 최종 은닉 차원 변경 / ClassifierHead final hidden dim changes
phase2.heads.resnet50.mid_dim = 512       → ResNet-50 단계적 압축 구조 변경 / ResNet-50 staged compression structure changes
temperature = 0.1        → InfoNCE 손실 스케일 변경 / InfoNCE loss scale changes
색상 공간 = BGR          → 입력 분포 전면 변경 / Full input distribution change / Color space = BGR
정규화 = /255.0 only     → 입력 스케일 변경 / Input scale changes / Normalization = /255.0 only
```

### 6.2 변경 가능 (성능 조정) / Can Change for Tuning

```
lr, weight_decay, batch_size, dropout, epochs
num_workers, pin_memory, device
oversample, seed
optimizer (adamw | sgd), scheduler (cosine | step)
momentum, betas, gamma, eta_min
```

---

## 체크리스트 / Checklist

- [ ] 새 변수 추가 시 Hard/Soft 분류 판정 / Classify as Hard/Soft when adding new variable
- [ ] Hard 값 변경 시 모든 체크포인트 폐기 확인 / Verify all checkpoints are discarded on Hard value change
- [ ] Dead Config 발견 시 소비 연결 또는 제거 / Connect or remove Dead Config when found

---

## See Also

| 문서 / Document | 관계 / Relation |
| --- | --- |
| [SSOT_Core.md](SSOT_Core.md) | Hard/Soft 판단 기준 / Hard/Soft decision criteria |
| [SSOT_Config_Resolution.md](SSOT_Config_Resolution.md) | config 키 소비 상세 / Config key consumption detail |

