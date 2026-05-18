---
type: contract
domain: tuning_boundary
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] Tuning Boundary — Optuna 튜닝 경계 계약 / Optuna Tuning Boundary Contract

> **목적 / Purpose**: `run_optuna()`, `get_phase2_search_space()`, `optuna_utils` 공개 API의 입출력 계약을 정의한다. / Defines I/O contracts for the `run_optuna()`, `get_phase2_search_space()`, and `optuna_utils` public APIs.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-17
> **관련 문서 / Related Docs**:
>
> - [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) (학습 파이프라인 / Training pipeline)
> - [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) (Optuna 산출물 / Optuna artifacts)

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.
> / This document does not redefine SSOT semantic definitions. Follow SSOT_Core.md as the final authority for semantic interpretation.

---

## 1. 계약 목적 / Contract Purpose

Optuna 기반 하이퍼파라미터 튜닝의 진입점, 탐색 공간, 산출물 저장/로드의 인터페이스 규약을 정의한다.

Defines the interface contracts for Optuna-based hyperparameter tuning entry points, search spaces, and artifact save/load operations.

---

## 2. `run_optuna()` 계약 / run_optuna() Contract

```python
from tuning.optuna_tuner import run_optuna

run_optuna(n_trials: int | None = None, channel: str = "all") -> None
```

| 항목 / Item | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `n_trials` | `int \| None` | None 이면 `cfg["optuna"]["n_trials"]` 사용 (기본 5) / Uses cfg value if None (default 5) |
| `channel` | `str` | `"all"` 또는 단일 채널 (`"Y"/"M"/"C"/"K"`) — 대소문자 무관 / `"all"` or single channel — case-insensitive |
| 반환 / Return | `None` | — |
| 산출물 / Output | — | `outputs/optuna/` 아래 trial 결과 JSON / Trial result JSON under `outputs/optuna/` |

**필수 cfg 키 / Required cfg Keys**: `optuna.n_trials`, `optuna.search_space.{backbone}.*`, `system.device`, `train.seed`

> ⚠️ **역방향 의존성 경고 / Reverse Dependency Warning**: `optuna_tuner.py` 가 `src.scripts.run_baseline` 을 import 한다. 이는 `tuning → scripts` 방향으로 레이어 위반. 향후 리팩토링 필요. / `optuna_tuner.py` imports `src.scripts.run_baseline`. This is a `tuning → scripts` layer violation requiring future refactoring.

---

## 3. 하이퍼파라미터 탐색 공간 계약 / Hyperparameter Search Space Contract

```python
from tuning.search_space import get_phase2_search_space

space = get_phase2_search_space(trial: optuna.Trial, cfg: dict = None) -> dict
```

| 항목 / Item | 설명 / Description |
| --- | --- |
| `trial` | `optuna.Trial` 객체 / object |
| `cfg` | config dict — `None` 이면 기본값 사용 (`efficientnet_b0` 기준) / Uses defaults if `None` |
| backbone 감지 / backbone detection | `cfg["model"]["backbone"]` — 없으면 `"efficientnet_b0"` fallback |
| 반환 / Return | `dict` — 샘플링된 하이퍼파라미터. ResNet-50 에만 `"mid_dim"` 포함 / Sampled hyperparameters. `"mid_dim"` only for ResNet-50 |
| Config 참조 / Config reference | `cfg["optuna"]["search_space"][backbone_name].*` → 최상위 → 기본값 순 fallback / Falls back to top-level → defaults |

**반환 dict 필수 키 / Required Return Dict Keys** (항상 / always):

| 키 / Key | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `learning_rate` | `float` | log-uniform 탐색 / log-uniform search |
| `batch_size` | `int` | categorical 선택 / categorical selection |
| `weight_decay` | `float` | log-uniform 탐색 / log-uniform search |
| `epochs` | `int` | 정수 범위 탐색 / integer range search |
| `dropout` | `float` | uniform 탐색 / uniform search |
| `hidden_dim` | `int` | categorical 선택 / categorical selection |
| `mid_dim` | `int` | **ResNet-50 전용** — EfficientNet-B0 시 키 없음 / **ResNet-50 only** — absent for EfficientNet-B0 |

---

## 4. `tuning/optuna_utils.py` 공개 API 계약 / tuning/optuna_utils.py Public API Contract

> **위치 / Location**: `src/tuning/optuna_utils.py` (not `src/utils/`)
>
> Optuna artifact 저장·불러오기, 채널 정규화, Phase 2 config 적용을 담당한다.
> / Responsible for Optuna artifact save/load, channel normalization, and Phase 2 config application.
> 저장·불러오기 책임은 `optuna_tuner.py` 에서 분리 (SRP).
> / Save/load responsibility is separated from `optuna_tuner.py` (SRP).

```python
from tuning.optuna_utils import (
    normalize_channel,
    load_best_params,
    save_best_params,
    save_trials_summary,
    apply_phase2_params,
)
```

### 4.1 `normalize_channel(channel: str) -> str`

| 항목 / Item | 내용 / Content |
| --- | --- |
| 입력 / Input | `"Y"` / `"M"` / `"C"` / `"K"` / `"all"` (대소문자 무관 / case-insensitive) |
| 반환 / Return | 소문자 suffix (`"y"` / `"m"` / `"c"` / `"k"` / `"all"`) |
| **실패 / Failure** | `ValueError` — `VALID_CHANNELS = {Y,M,C,K,ALL}` 외 입력 / input outside VALID_CHANNELS |

### 4.2 `load_best_params(channel: str, output_dir: str | Path = "outputs/optuna") -> dict`

| 항목 / Item | 내용 / Content |
| --- | --- |
| 입력 / Input | `channel` (normalize_channel 경유 / via normalize_channel), `output_dir` |
| 반환 / Return | best params `dict` |
| **실패 / Failure** | `FileNotFoundError` — `best_params_{ch}.json` 미존재 (SSOT-FF01) / not found |
| **실패 / Failure** | `ValueError` — 유효하지 않은 channel / invalid channel |

### 4.3 `save_best_params(params: dict, channel: str, output_dir) -> Path`

| 항목 / Item | 내용 / Content |
| --- | --- |
| 반환 / Return | 저장된 `Path` 객체 / Saved `Path` object |
| 파일명 / Filename | `best_params_{channel_lower}.json` |

### 4.4 `save_trials_summary(trials: list, channel: str, output_dir) -> Path`

| 항목 / Item | 내용 / Content |
| --- | --- |
| `trials` | `optuna.trial.FrozenTrial` 목록 (study.trials) / list |
| 파일명 / Filename | `trials_summary_{channel_lower}.json` |
| 저장 스키마 / Save Schema | `[{number, value, state, params}, ...]` |

### 4.5 `apply_phase2_params(cfg: dict, params: dict) -> dict`

```python
cfg = apply_phase2_params(cfg, params)
```

| 항목 / Item | 내용 / Content |
| --- | --- |
| 입력 `params` 필수 키 / Required `params` input keys | `learning_rate`, `batch_size`, `weight_decay`, `epochs`, `dropout`, `hidden_dim` |
| ResNet-50 추가 키 / Additional ResNet-50 key | `mid_dim` (있으면 적용, 없으면 스킵 / applied if present, skipped if absent) |
| 적용 대상 / Applied to | `cfg["phase2"]["learning_rate/batch_size/weight_decay/epochs"]`, `cfg["phase2"]["heads"][backbone]["dropout/hidden_dim"]` |
| **Backbone 불변식 / Backbone Invariant** | `cfg["model"]["backbone"]` 값은 변경되지 않는다 / value is not changed |
| 반환 / Return | 수정된 `cfg` dict (in-place + return) / Modified cfg dict |
| **실패 / Failure** | `KeyError` — 필수 params 키 누락 / Required params key missing |

---

## 5. 산출물 계약 / Artifact Contract

| 아티팩트 / Artifact | 경로 패턴 / Path Pattern | 생산자 / Producer |
| --- | --- | --- |
| Best params | `outputs/optuna/best_params_{ch}.json` | `save_best_params()` |
| Trial 요약 / Trial summary | `outputs/optuna/trials_summary_{ch}.json` | `save_trials_summary()` |
| Optuna DB | `outputs/optuna/study_{ch}.db` | `run_optuna()` |

---

## 6. 금지 패턴 / Prohibited Patterns

```python
# ❌ cfg["model"]["backbone"] 변경
# / Changing cfg["model"]["backbone"]
cfg["model"]["backbone"] = "resnet50"  # apply_phase2_params 내부에서 금지 / prohibited inside

# ❌ 탐색 공간 범위를 코드에 하드코딩 (config 기반이어야 함)
# / Hardcoding search space ranges in code (must be config-based)
lr = trial.suggest_float("learning_rate", 1e-5, 1e-2)  # config 무시 / ignores config

# ✅ 올바른 패턴 / Correct pattern
space = get_phase2_search_space(trial, cfg)  # config 기반 탐색 / config-based search
```

---

## 7. 체크리스트 / Checklist

- [x] `normalize_channel()` — 대소문자 무관 처리 / Case-insensitive handling
- [x] `load_best_params()` — 파일 없으면 `FileNotFoundError` (SSOT-FF01) / `FileNotFoundError` if file missing
- [x] `apply_phase2_params()` — backbone 불변식 보장 / Backbone invariant guaranteed
- [ ] `tuning → scripts` 레이어 위반 리팩토링 / Refactor `tuning → scripts` layer violation

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) | 학습 파이프라인 정의 (What) / Training pipeline definition |
| [Contract_training_pipeline.md](Contract_training_pipeline.md) | Phase2Trainer 계약 / Phase2Trainer contract |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | 산출물 저장 계약 / Artifact save contract |
