---
type: contract
domain: config_resolution
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] Config Resolution — Config 로딩 및 해석 계약 / Config Loading and Resolution Contract

> **목적 / Purpose**: `load_config()` API의 반환 타입, 보장 사항, 금지 패턴을 정의한다. / Defines the return type, guarantees, and prohibited patterns for the `load_config()` API.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-17
> **관련 문서 / Related Docs**:
>
> - [SSOT_Config_Resolution.md](../SSOT/SSOT_Config_Resolution.md) (config 키 전수 매핑 / Full config key mapping)
> - [SSOT_Core.md](../SSOT/SSOT_Core.md) (Hard/Soft 판단 기준 / Hard/Soft decision criteria)

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.
> / This document does not redefine SSOT semantic definitions. Follow SSOT_Core.md as the final authority for semantic interpretation.

---

## 1. 계약 목적 / Contract Purpose

Config 로딩 체인(`load_config → validate_config → create_directories → get_nested`)의
입출력 타입, 보장 사항, 사전/사후 조건을 정의한다.

Defines input/output types, guarantees, and pre/post-conditions for the Config loading chain
(`load_config → validate_config → create_directories → get_nested`).

---

## 2. 공개 API 계약 / Public API Contract

### 2.1 `load_config()`

```python
from src.utils import load_config

cfg = load_config()  # → dict
```

| 항목 / Item | 타입 / Type | 보장 / Guarantee |
| --- | --- | --- |
| 반환값 / Return value | `dict` | 경로 절대화 + device 자동 감지 완료 / Paths resolved to absolute + device auto-detected |
| `cfg["system"]["device"]` | `str` | `"cpu"` / `"cuda"` / `"mps"` (**절대** `"auto"` 아님 / **Never** `"auto"`) |
| `cfg["storage"]["*_dir"]` | `str` | 절대 경로 (root 기준 해소 완료) / Absolute path (resolved relative to root) |

### 2.2 `validate_config()`

```python
from src.utils import validate_config

validate_config(cfg)  # → None (실패 시 ValueError, SSOT-CF01 / raises ValueError on failure)
```

| 조건 / Condition | 예외 / Exception | 코드 / Code |
| --- | --- | --- |
| 필수 섹션 누락 (`data`, `model`, `phase2` 등) / Required sections missing | `ValueError` | SSOT-CF01 |
| 키 타입 불일치 / Key type mismatch | `ValueError` | SSOT-CF01 |

### 2.3 `create_directories()`

```python
from src.utils import create_directories

create_directories(cfg)  # → None (storage.* 경로 생성 / creates storage.* paths)
```

### 2.4 `get_nested()`

```python
from src.utils import get_nested

val = get_nested(cfg, "phase2.learning_rate")  # → Any (None-safe dot-notation)
```

| 항목 / Item | 보장 / Guarantee |
| --- | --- |
| 키 존재 시 / When key exists | 해당 값 반환 / Returns the value |
| 키 부재 시 / When key absent | `None` 반환 (예외 없음) / Returns `None` (no exception) |

---

## 3. 모듈별 필수 Config 키 / Required Config Keys per Module

| 모듈 / Module | 필수 config 키 / Required Config Keys |
| --- | --- |
| `data.dataset` | `data.channels`, `data.num_levels`, `data.image_size`, `data.split_ratios.*`, `storage.labeled_dir`, `train.seed` |
| `data.augmentation` (Phase 0) | `phase0.augmentation.*` (flip/crop/color_jitter/contrast/blur) |
| `data.augmentation` (Phase 2) | `phase2.augmentation.*` (flip/brightness/noise), `phase2.oversample` |
| `models.grayspot_model` | `model.backbone`, `model.frozen_backbone`, `data.num_levels`, `phase0.projection_dim`, `phase0.hidden_dim`, `phase2.heads.{backbone}.*` |
| `training.trainer` (Phase 0) | `phase0.epochs/batch_size/learning_rate/weight_decay/temperature`, `train.optimizer/scheduler/gradient_clip/eta_min/seed`, `storage.models_dir` |
| `training.trainer` (Phase 2) | `phase2.epochs/batch_size/learning_rate/weight_decay/early_stopping.*`, `phase2.k_fold.*`, `train.*`, `storage.models_dir/reports_dir` |
| `training.losses` | `phase2.loss`, `phase2.class_weights`, `phase2.label_smoothing`, `phase2.focal_gamma`, `data.num_levels` |
| `evaluation.evaluator` | `inference.confidence_thresholds.*`, `evaluation.swing_thresholds.*` |
| `evaluation.metrics` | `evaluation.targets.*`, `data.num_levels` |
| `tuning.optuna_tuner` | `optuna.*`, `system.device`, `train.seed` |
| `tuning.search_space` | `optuna.search_space.{backbone}.*` |
| `inference.predictor` | `system.device`, `storage.models_dir`, `data.channels/image_size/num_levels` |
| `scripts.run_phase0` | `data.channels`, `system.device`, `train.seed`, `storage.*` |
| `scripts.run_phase2` | `data.channels`, `system.device`, `train.seed`, `storage.*` |
| `scripts.run_baseline` | `data.channels`, `system.device`, `train.seed`, `storage.*` |
| `scripts.run_optuna` | `optuna.*`, `system.device` |

---

## 4. 금지 패턴 / Prohibited Patterns

```python
# ❌ 삭제된 ConfigManager 패턴 — 사용 금지
# / Deleted ConfigManager pattern — do not use
config = load_config()
cfg = config.config
config.get("phase2.learning_rate")
```

```python
# ✅ 현재 올바른 패턴 / Current correct pattern
cfg = load_config()                           # dict 직접 반환 / returns dict directly
val = get_nested(cfg, "phase2.learning_rate")  # dot-notation 접근 / dot-notation access
```

---

## 5. 체크리스트 / Checklist

- [x] `load_config()` → `dict` 반환 확인 / Verify `load_config()` returns `dict`
- [x] `cfg["system"]["device"]` → `"auto"` 아닌 실제 디바이스 / Verify actual device, not `"auto"`
- [x] `cfg["storage"]["*_dir"]` → 절대 경로 / Verify absolute paths
- [x] Dead Config 0개 확인 / Verify 0 dead config keys
- [ ] 새 모듈 추가 시 §3 필수 키 목록 갱신 / Update §3 required key list when adding new modules

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Config_Resolution.md](../SSOT/SSOT_Config_Resolution.md) | config 키 전수 매핑 (What) / Full config key mapping |
| [Contract_fail_fast.md](Contract_fail_fast.md) | SSOT-CF01 정의 / SSOT-CF01 definition |
