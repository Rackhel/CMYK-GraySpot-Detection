# SSOT Config Resolution — 설정 해소 대조표 / Config Resolution Cross-Reference

CMYK Grayspot Detection System 의 `config.json` 키별 소비 현황과 해석 규칙에 관한 단일 진실 공급원.

This document tracks the consumption status of every key in `config.json` and defines their resolution rules.

> **목적 / Purpose**: config.json 키별 소비 현황과 해석 규칙 정의
> **역할 / Role**: "What" — config 키의 의미와 소비 상태 추적 (How는 Contract 참조)
> **근거 파일 / Basis**: `src/config/config.json`, 전체 `src/` 코드 분석
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_GlobalVariables.md](SSOT_GlobalVariables.md)
> **Fail-Fast 코드 / Codes**: `SSOT-CF01` (필수 키 누락), `SSOT-CF02` (Dead Config 감지)

---

## 1. Config 로딩 API / Config Loading Interface

> `src/config/config_manager.py` 및 `ConfigManager` 클래스는 **삭제됨**. / `src/config/config_manager.py` and the `ConfigManager` class have been **deleted**.

config 로딩은 `src/utils/utils_config.py` 의 함수들로 대체되었다. / Config loading is now handled by functions in `src/utils/utils_config.py`.

### 1.1 공개 API / Public API

| 함수 / Function | 반환 타입 / Return | 설명 / Description |
|---|---|---|
| `load_config(config_path=None, root_dir=None)` | `dict` | `config.json` 로드 → 경로 해소 → 디바이스 감지 후 **dict 반환** / Load → path resolution → device detection → **return dict** |
| `validate_config(cfg)` | `bool` | 필수 키 및 값 범위 검증 / Validate required keys and value ranges |
| `create_directories(cfg)` | `None` | `storage.*` 경로 자동 생성 / Auto-create `storage.*` paths |
| `get_nested(cfg, key, default=None)` | `Any` | `"phase2.learning_rate"` 형식의 dot-notation으로 optional 키 조회 / Optional key lookup via dot-notation (e.g., `"phase2.learning_rate"`) |

모든 함수는 `from src.utils import ...` 또는 `from utils import ...` (shim 환경) 로 임포트한다. / All functions are imported via `from src.utils import ...` or `from utils import ...` (shim context).

### 1.3 utils 모듈 구조 / utils Module Structure

```
src/utils/
├── __init__.py        ← 모든 공개 심볼 re-export / Re-exports all public symbols
├── utils_config.py    ← load_config, validate_config, create_directories, get_nested  # 설정 로딩 / Config loading
├── utils_model.py     ← set_seed, backbone_tag, build_model  # 모델 유틸 / Model utilities
└── logger.py          ← get_logger, setup_logging, LoggerMixin, log_* helpers  # 로깅 / Logging
```

---

## 2. 목적 및 아이콘 / Purpose and Icons

`config.json`에 선언된 모든 키가 실제 코드에서 소비되는지를 추적한다. / Track whether every key declared in `config.json` is actually consumed by code.

| 아이콘 / Icon | 의미 / Meaning | 설명 / Description |
|---|---|---|
| 🟢 | **소비됨 / Consumed** | 코드에서 `cfg["key"]`로 읽혀 런타임에 반영 / Read by code at runtime |
| 🔴 | **미연결 / Dead** | 선언만 되고 코드에서 읽는 곳 없음 (Dead Config) / Declared but never read |
| 🟡 | **하드코딩 상충 / Conflict** | config 키가 있지만 코드는 다른 값 사용 / Key exists but code uses different value |
| ⚠️ | **부분 소비 / Partial** | 일부 하위 키만 사용 / Only some sub-keys consumed |

---

## 3. 전체 요약 / Summary

| 섹션 / Section | 총 키 수 / Total Keys | 🟢 소비 / Consumed | 🔴 미연결 / Dead | 소비율 / Rate |
|------|---------|---------|----------|-------|
| system | 1 | 1 | 0 | 100% |
| data | 6 | 6 | 0 | 100% |
| storage | 5 | 5 | 0 | 100% |
| model | 2 | 2 | 0 | 100% |
| phase0 | 16 | 16 | 0 | 100% |
| phase2 | 15 | 15 | 0 | 100% |
| train | 13 | 13 | 0 | 100% |
| evaluation | 7 | 7 | 0 | 100% |
| logging | 4 | 4 | 0 | 100% |
| reporting | ~10 | 3 | ~7 | ~30% |
| inference | ~10 | 0 | ~10 | 0% |
| optuna | 8 | 8 | 0 | 100% |
| cuda | 2 | 2 | 0 | 100% |
| **합계 / Grand Total** | **~87** | **~70** | **~17** | **~80%** |

---

## 4. 상세 대조표 / Detailed Cross-Reference

### 4.1 system

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `system.device` | `"auto"` | 🟢 | `run_baseline.py`, `run_phase0.py`, `run_phase2.py` |

### 4.2 data

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `data.channels` | `["Y","M","C","K"]` | 🟢 | `run_baseline.py`, `run_phase0.py` |
| `data.num_levels` | `6` | 🟢 | `grayspot_model.py`, `dataset.py` |
| `data.image_size` | `128` | 🟢 | `dataset.py` |
| `data.split_ratios.train` | `0.70` | 🟢 | `dataset.py` (dict: train/val/test) |
| `data.split_ratios.val` | `0.15` | 🟢 | `dataset.py` |
| `data.split_ratios.test` | `0.15` | 🟢 | `dataset.py` |

### 4.3 storage

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `storage.data_root` | `"data_set"` | 🟢 | `run_baseline.py` |
| `storage.labeled_dir` | `"data_set/labeled"` | 🟢 | `dataset.py` |
| `storage.models_dir` | `"data_set/models"` | 🟢 | `trainer.py` (Phase0/Phase2 save) |
| `storage.reports_dir` | `"data_set/reports"` | 🟢 | `trainer.py` (history CSV) |
| `storage.logs_dir` | `"outputs/logs"` | 🟢 | `setup_logging()` |

### 4.4 model

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `model.backbone` | `"efficientnet_b0"` | 🟢 | `backbone.py`, `grayspot_model.py` |
| `model.frozen_backbone` | `false` | 🟢 | `grayspot_model.py` (backbone freeze) |

### 4.5 phase0

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `phase0.epochs` | `10` | 🟢 | `Phase0Trainer` |
| `phase0.batch_size` | `16` | 🟢 | `Phase0Trainer` DataLoader |
| `phase0.learning_rate` | `1e-3` | 🟢 | `Phase0Trainer` optimizer |
| `phase0.weight_decay` | `1e-5` | 🟢 | `Phase0Trainer` optimizer |
| `phase0.temperature` | `0.1` | 🟢 | `InfoNCELoss` |
| `phase0.projection_dim` | `128` | 🟢 | `GrayspotModel` ProjectionHead out_dim |
| `phase0.hidden_dim` | `256` | 🟢 | `GrayspotModel` ProjectionHead hidden_dim |
| `phase0.augmentation.color_jitter` | `0.4` | 🟢 | `augment_contrastive()` via `aug_cfg` |
| `phase0.augmentation.blur_prob` | `0.5` | 🟢 | `augment_contrastive()` via `aug_cfg` |
| `phase0.augmentation.flip_prob` | `0.5` | 🟢 | `augment_contrastive()` via `aug_cfg` |
| `phase0.augmentation.crop_prob` | `0.5` | 🟢 | `augment_contrastive()` via `aug_cfg` |
| `phase0.augmentation.crop_scale_min` | `0.6` | 🟢 | `augment_contrastive()` via `aug_cfg` |
| `phase0.augmentation.crop_scale_max` | `1.0` | 🟢 | `augment_contrastive()` via `aug_cfg` |
| `phase0.augmentation.contrast_scale_min` | `0.8` | 🟢 | `augment_contrastive()` via `aug_cfg` |
| `phase0.augmentation.contrast_scale_max` | `1.2` | 🟢 | `augment_contrastive()` via `aug_cfg` |
| `phase0.augmentation.blur_kernels` | `[3, 5]` | 🟢 | `augment_contrastive()` via `aug_cfg` |

### 4.6 phase2

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `phase2.epochs` | `30` | 🟢 | `Phase2Trainer` |
| `phase2.batch_size` | `16` | 🟢 | `Phase2Trainer` DataLoader |
| `phase2.learning_rate` | `1e-4` | 🟢 | `Phase2Trainer` optimizer |
| `phase2.weight_decay` | `1e-4` | 🟢 | `Phase2Trainer` optimizer |
| `phase2.dropout` | `0.3` | 🟢 | `GrayspotModel` ClassifierHead |
| `phase2.hidden_dim` | `256` | 🟢 | `GrayspotModel` ClassifierHead hidden_dim |
| `phase2.oversample` | `true` | 🟢 | `CMYKDataset` |
| `phase2.early_stopping.enabled` | `true` | 🟢 | `Phase2Trainer.train()` |
| `phase2.early_stopping.patience` | `5` | 🟢 | `Phase2Trainer.train()` |
| `phase2.early_stopping.min_delta` | `1e-4` | 🟢 | `Phase2Trainer.train()` |
| `phase2.augmentation.flip_prob` | `0.5` | 🟢 | `CMYKDataset.sup_aug_cfg` → `augment_supervised()` |
| `phase2.augmentation.brightness_prob` | `0.5` | 🟢 | `CMYKDataset.sup_aug_cfg` → `augment_supervised()` |
| `phase2.augmentation.brightness_range` | `30` | 🟢 | `CMYKDataset.sup_aug_cfg` → `augment_supervised()` |
| `phase2.augmentation.noise_prob` | `0.5` | 🟢 | `CMYKDataset.sup_aug_cfg` → `augment_supervised()` |
| `phase2.augmentation.noise_range` | `10` | 🟢 | `CMYKDataset.sup_aug_cfg` → `augment_supervised()` |

### 4.7 train — 공통 학습 설정 / Common Training Settings

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `train.seed` | `42` | 🟢 | `run_baseline.py`, `run_phase0.py`, `run_phase2.py` |
| `train.num_workers` | `4` | 🟢 | DataLoader (`run_phase2.py`, `run_baseline.py`) |
| `train.pin_memory` | `true` | 🟢 | DataLoader |
| `train.prefetch_factor` | `2` | 🟢 | DataLoader |
| `train.persistent_workers` | `true` | 🟢 | DataLoader |
| `train.drop_last` | `false` | 🟢 | DataLoader |
| `train.eta_min` | `1e-6` | 🟢 | `_build_scheduler()` CosineAnnealingLR (trainer.py) |
| `train.scheduler` | `"cosine"` | 🟢 | `_build_scheduler()` (trainer.py) |
| `train.optimizer` | `"adamw"` | 🟢 | `_build_optimizer()` (trainer.py) |
| `train.gradient_clip` | `1.0` | 🟢 | `Phase0Trainer`, `Phase2Trainer` |
| `train.momentum` | `0.9` | 🟢 | `_build_optimizer()` — SGD 선택 시 소비 / consumed when optimizer=sgd |
| `train.betas` | `[0.9, 0.999]` | 🟢 | `_build_optimizer()` — AdamW betas / AdamW beta parameters |
| `train.gamma` | `0.1` | 🟢 | `_build_scheduler()` — StepLR 선택 시 소비 / consumed when scheduler=step |

### 4.8 evaluation

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `evaluation.targets.overall_accuracy` | `0.90` | 🟢 | `evaluation/metrics.py` TARGET_OVERALL_ACC |
| `evaluation.targets.per_color_accuracy` | `0.85` | 🟢 | `evaluation/metrics.py` TARGET_PER_COLOR_ACC |
| `evaluation.targets.per_class_f1` | `0.80` | 🟢 | `evaluation/metrics.py` TARGET_PER_CLASS_F1 |
| `evaluation.targets.mae` | `0.50` | 🟢 | `evaluation/metrics.py` TARGET_MAE |
| `evaluation.swing_thresholds.acc_retry` | `0.80` | 🟢 | `determine_swing_feedback()` via `summary.targets["swing_acc_retry"]` |
| `evaluation.swing_thresholds.f1_retry` | `0.70` | 🟢 | `determine_swing_feedback()` via `summary.targets["swing_f1_retry"]` |
| `evaluation.swing_thresholds.mae_retry` | `0.80` | 🟢 | `determine_swing_feedback()` via `summary.targets["swing_mae_retry"]` |

### 4.8.1 inference.confidence_thresholds

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `inference.confidence_thresholds.auto_accept` | `0.8` | 🟢 | `Evaluator.__init__(cfg)` → `self.conf_thresh_auto` |
| `inference.confidence_thresholds.warn_threshold` | `0.5` | 🟢 | `Evaluator.__init__(cfg)` → `self.conf_thresh_warn` |
| `inference.confidence_thresholds.manual_review` | `0.3` | 🟢 | `Evaluator.__init__(cfg)` → `self.conf_thresh_manual` |

### 4.9 logging

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `logging.level` | `"INFO"` | 🟢 | `setup_logging()` |
| `logging.format` | `"detailed"` | 🟢 | `setup_logging()` |
| `logging.console_output` | `true` | 🟢 | `setup_logging()` |
| `logging.file_output` | `true` | 🟢 | `setup_logging()` |

### 4.10 optuna

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `optuna.enabled` | `false` | 🟢 | `run_optuna.py` (guard) |
| `optuna.n_trials` | `20` | 🟢 | `optuna_tuner.py` |
| `optuna.n_jobs` | `1` | 🟢 | `optuna_tuner.py` |
| `optuna.direction` | `"maximize"` | 🟢 | `_build_study()` in `optuna_tuner.py` |
| `optuna.sampler` | `"tpe"` | 🟢 | `_build_study()` in `optuna_tuner.py` |
| `optuna.search_space.learning_rate` | `[1e-5, 1e-2]` | 🟢 | `search_space.py` `get_phase2_search_space()` |
| `optuna.search_space.weight_decay` | `[1e-6, 1e-3]` | 🟢 | `search_space.py` `get_phase2_search_space()` |
| `optuna.search_space.dropout` | `[0.1, 0.5]` | 🟢 | `search_space.py` (reserved for future Phase2Trainer use) |
| `optuna.search_space.batch_size` | `[16, 32, 64]` | 🟢 | `search_space.py` `get_phase2_search_space()` |
| `optuna.search_space.epochs` | `[10, 30]` | 🟢 | `search_space.py` `get_phase2_search_space()` |

### 4.11 cuda

| 키 / Key | 기본값 / Default | 소비 여부 / Status | 소비 위치 / Consumer |
|---|---|---|---|
| `cuda.deterministic` | `true` | 🟢 | `set_seed()` → `torch.backends.cudnn.deterministic` |
| `cuda.benchmark` | `true` | 🟢 | `set_seed()` → `torch.backends.cudnn.benchmark` |

---

## 5. Dead Config 요약 / Dead Config Summary

현재 이 시스템에 남아 있는 Dead Config는 `reporting` 및 `inference` 섹션의 일부 키로, 해당 모듈이 config를 직접 소비하지 않기 때문입니다. / The remaining Dead Config keys belong to `reporting` and `inference` sections where the modules do not yet consume config values directly.

| 우선순위 / Priority | config 키 / Key | 구현 난이도 / Effort | 성능/재현성 영향 / Impact |
|---------|-----------|-----------|----------|
| 🟢 LOW | `reporting.*` (html 설정 외 / excluding html settings) | 쉬움 / Easy | 리포트 포맷 유연성 / Report format flexibility |
| 🟢 LOW | `inference.*` (confidence_thresholds 제외 / excluding confidence_thresholds) | 보통 / Medium | 추론 설정 유연성 / Inference config flexibility |

---

## 6. 사용 규칙 / Usage Rules

1. **새 config 키 추가 시 / When adding a new config key**: 이 문서에 동시 등록 (소비 위치 명시) / Register in this document simultaneously (specify consumer location)
2. **코드에서 config 읽을 때 / When reading config in code**: 이 문서에서 키 존재 확인 / Verify key existence in this document
3. **리팩토링 시 / During refactoring**: Dead Config → 구현 또는 config.json에서 제거 결정 / Decide to implement or remove from config.json
4. **PR 리뷰 시 / During PR review**: config 키 추가/변경이 이 문서에 반영되었는지 확인 / Verify that config key additions/changes are reflected in this document

> `SSOT-CF01`: 코드가 참조하는 config 키가 config.json에 없으면 **즉시 실패**.
> `SSOT-CF01`: If code references a key absent from config.json, **fail immediately**.

---
