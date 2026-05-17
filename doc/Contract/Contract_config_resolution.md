---
type: contract
domain: config_resolution
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
---

# [Contract] Config Resolution — Config 로딩 및 해석 계약

> **목적**: `load_config()` API의 반환 타입, 보장 사항, 금지 패턴을 정의한다.
> **상태**: ✅ Accepted [Hard]
> **작성일**: 2026-05-17
> **관련 문서**:
>
> - [SSOT_Config_Resolution.md](../SSOT/SSOT_Config_Resolution.md) (config 키 전수 매핑)
> - [SSOT_Core.md](../SSOT/SSOT_Core.md) (Hard/Soft 판단 기준)

> 🔒 **SSOT 경계 원칙**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다.
> 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.

---

## 1. 계약 목적

Config 로딩 체인(`load_config → validate_config → create_directories → get_nested`)의
입출력 타입, 보장 사항, 사전/사후 조건을 정의한다.

---

## 2. 공개 API 계약

### 2.1 `load_config()`

```python
from src.utils import load_config

cfg = load_config()  # → dict
```

| 항목 | 타입 | 보장 |
| --- | --- | --- |
| 반환값 | `dict` | 경로 절대화 + device 자동 감지 완료 |
| `cfg["system"]["device"]` | `str` | `"cpu"` / `"cuda"` / `"mps"` (**절대** `"auto"` 아님) |
| `cfg["storage"]["*_dir"]` | `str` | 절대 경로 (root 기준 해소 완료) |

### 2.2 `validate_config()`

```python
from src.utils import validate_config

validate_config(cfg)  # → None (실패 시 ValueError, SSOT-CF01)
```

| 조건 | 예외 | 코드 |
| --- | --- | --- |
| 필수 섹션 누락 (`data`, `model`, `phase2` 등) | `ValueError` | SSOT-CF01 |
| 키 타입 불일치 | `ValueError` | SSOT-CF01 |

### 2.3 `create_directories()`

```python
from src.utils import create_directories

create_directories(cfg)  # → None (storage.* 경로 생성)
```

### 2.4 `get_nested()`

```python
from src.utils import get_nested

val = get_nested(cfg, "phase2.learning_rate")  # → Any (None-safe dot-notation)
```

| 항목 | 보장 |
| --- | --- |
| 키 존재 시 | 해당 값 반환 |
| 키 부재 시 | `None` 반환 (예외 없음) |

---

## 3. 모듈별 필수 Config 키

| 모듈 | 필수 config 키 |
| --- | --- |
| `data.dataset` | `data.channels`, `data.num_levels`, `data.image_size`, `data.split_ratios.*`, `storage.labeled_dir`, `train.seed` |
| `data.augmentation` (Phase 0) | `phase0.augmentation.*` (flip/crop/color_jitter/contrast/blur) |
| `data.augmentation` (Phase 2) | `phase2.augmentation.*` (flip/brightness/noise), `phase2.oversample` |
| `models.grayspot_model` | `model.backbone`, `model.frozen_backbone`, `data.num_levels`, `phase0.projection_dim`, `phase0.hidden_dim`, `phase2.heads.{backbone}.*` |
| `training.trainer` (Phase 0) | `phase0.epochs/batch_size/learning_rate/weight_decay/temperature`, `train.optimizer/scheduler/gradient_clip/eta_min/seed`, `storage.models_dir` |
| `training.trainer` (Phase 2) | `phase2.epochs/batch_size/learning_rate/weight_decay/early_stopping.*`, `train.*`, `storage.models_dir/reports_dir` |
| `training.losses` | `phase0.temperature`, `data.num_levels` |
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

## 4. 금지 패턴

```python
# ❌ 삭제된 ConfigManager 패턴 — 사용 금지
config = load_config()
cfg = config.config
config.get("phase2.learning_rate")
```

```python
# ✅ 현재 올바른 패턴
cfg = load_config()                           # dict 직접 반환
val = get_nested(cfg, "phase2.learning_rate")  # dot-notation 접근
```

---

## 5. 체크리스트

- [x] `load_config()` → `dict` 반환 확인
- [x] `cfg["system"]["device"]` → `"auto"` 아닌 실제 디바이스
- [x] `cfg["storage"]["*_dir"]` → 절대 경로
- [x] Dead Config 0개 확인
- [ ] 새 모듈 추가 시 §3 필수 키 목록 갱신

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_Config_Resolution.md](../SSOT/SSOT_Config_Resolution.md) | config 키 전수 매핑 (What) |
| [Contract_fail_fast.md](Contract_fail_fast.md) | SSOT-CF01 정의 |
