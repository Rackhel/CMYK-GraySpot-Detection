---
type: contract
domain: tuning_boundary
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
---

# [Contract] Tuning Boundary — Optuna 튜닝 경계 계약

> **목적**: `run_optuna()`, `get_phase2_search_space()`, `optuna_utils` 공개 API의 입출력 계약을 정의한다.
> **상태**: ✅ Accepted [Hard]
> **작성일**: 2026-05-17
> **관련 문서**:
>
> - [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) (학습 파이프라인)
> - [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) (Optuna 산출물)

> 🔒 **SSOT 경계 원칙**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다.
> 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.

---

## 1. 계약 목적

Optuna 기반 하이퍼파라미터 튜닝의 진입점, 탐색 공간, 산출물 저장/로드의 인터페이스 규약을 정의한다.

---

## 2. `run_optuna()` 계약

```python
from tuning.optuna_tuner import run_optuna

run_optuna(n_trials: int | None = None, channel: str = "all") -> None
```

| 항목 | 타입 | 설명 |
| --- | --- | --- |
| `n_trials` | `int \| None` | None 이면 `cfg["optuna"]["n_trials"]` 사용 (기본 5) |
| `channel` | `str` | `"all"` 또는 단일 채널 (`"Y"/"M"/"C"/"K"`) — 대소문자 무관 |
| 반환 | `None` | — |
| 산출물 | — | `outputs/optuna/` 아래 trial 결과 JSON |

**필수 cfg 키**: `optuna.n_trials`, `optuna.search_space.{backbone}.*`, `system.device`, `train.seed`

> ⚠️ **역방향 의존성 경고**: `optuna_tuner.py` 가 `src.scripts.run_baseline` 을 import 한다. 이는 `tuning → scripts` 방향으로 레이어 위반. 향후 리팩토링 필요.

---

## 3. 하이퍼파라미터 탐색 공간 계약

```python
from tuning.search_space import get_phase2_search_space

space = get_phase2_search_space(trial: optuna.Trial, cfg: dict = None) -> dict
```

| 항목 | 설명 |
| --- | --- |
| `trial` | `optuna.Trial` 객체 |
| `cfg` | config dict — `None` 이면 기본값 사용 (`efficientnet_b0` 기준) |
| backbone 감지 | `cfg["model"]["backbone"]` — 없으면 `"efficientnet_b0"` fallback |
| 반환 | `dict` — 샘플링된 하이퍼파라미터. ResNet-50 에만 `"mid_dim"` 포함 |
| Config 참조 | `cfg["optuna"]["search_space"][backbone_name].*` → 최상위 → 기본값 순 fallback |

**반환 dict 필수 키** (항상):

| 키 | 타입 | 설명 |
| --- | --- | --- |
| `learning_rate` | `float` | log-uniform 탐색 |
| `batch_size` | `int` | categorical 선택 |
| `weight_decay` | `float` | log-uniform 탐색 |
| `epochs` | `int` | 정수 범위 탐색 |
| `dropout` | `float` | uniform 탐색 |
| `hidden_dim` | `int` | categorical 선택 |
| `mid_dim` | `int` | **ResNet-50 전용** — EfficientNet-B0 시 키 없음 |

---

## 4. `tuning/optuna_utils.py` 공개 API 계약

> **위치**: `src/tuning/optuna_utils.py` (not `src/utils/`)
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

### 4.1 `normalize_channel(channel: str) -> str`

| 항목 | 내용 |
| --- | --- |
| 입력 | `"Y"` / `"M"` / `"C"` / `"K"` / `"all"` (대소문자 무관) |
| 반환 | 소문자 suffix (`"y"` / `"m"` / `"c"` / `"k"` / `"all"`) |
| **실패** | `ValueError` — `VALID_CHANNELS = {Y,M,C,K,ALL}` 외 입력 |

### 4.2 `load_best_params(channel: str, output_dir: str | Path = "outputs/optuna") -> dict`

| 항목 | 내용 |
| --- | --- |
| 입력 | `channel` (normalize_channel 경유), `output_dir` |
| 반환 | best params `dict` |
| **실패** | `FileNotFoundError` — `best_params_{ch}.json` 미존재 (SSOT-FF01) |
| **실패** | `ValueError` — 유효하지 않은 channel |

### 4.3 `save_best_params(params: dict, channel: str, output_dir) -> Path`

| 항목 | 내용 |
| --- | --- |
| 반환 | 저장된 `Path` 객체 |
| 파일명 | `best_params_{channel_lower}.json` |

### 4.4 `save_trials_summary(trials: list, channel: str, output_dir) -> Path`

| 항목 | 내용 |
| --- | --- |
| `trials` | `optuna.trial.FrozenTrial` 목록 (study.trials) |
| 파일명 | `trials_summary_{channel_lower}.json` |
| 저장 스키마 | `[{number, value, state, params}, ...]` |

### 4.5 `apply_phase2_params(cfg: dict, params: dict) -> dict`

```python
cfg = apply_phase2_params(cfg, params)
```

| 항목 | 내용 |
| --- | --- |
| 입력 `params` 필수 키 | `learning_rate`, `batch_size`, `weight_decay`, `epochs`, `dropout`, `hidden_dim` |
| ResNet-50 추가 키 | `mid_dim` (있으면 적용, 없으면 스킵) |
| 적용 대상 | `cfg["phase2"]["learning_rate/batch_size/weight_decay/epochs"]`, `cfg["phase2"]["heads"][backbone]["dropout/hidden_dim"]` |
| **Backbone 불변식** | `cfg["model"]["backbone"]` 값은 변경되지 않는다 |
| 반환 | 수정된 `cfg` dict (in-place + return) |
| **실패** | `KeyError` — 필수 params 키 누락 |

---

## 5. 산출물 계약

| 아티팩트 | 경로 패턴 | 생산자 |
| --- | --- | --- |
| Best params | `outputs/optuna/best_params_{ch}.json` | `save_best_params()` |
| Trial 요약 | `outputs/optuna/trials_summary_{ch}.json` | `save_trials_summary()` |
| Optuna DB | `outputs/optuna/study_{ch}.db` | `run_optuna()` |

---

## 6. 금지 패턴

```python
# ❌ cfg["model"]["backbone"] 변경
cfg["model"]["backbone"] = "resnet50"  # apply_phase2_params 내부에서 금지

# ❌ 탐색 공간 범위를 코드에 하드코딩 (config 기반이어야 함)
lr = trial.suggest_float("learning_rate", 1e-5, 1e-2)  # config 무시

# ✅ 올바른 패턴
space = get_phase2_search_space(trial, cfg)  # config 기반 탐색
```

---

## 7. 체크리스트

- [x] `normalize_channel()` — 대소문자 무관 처리
- [x] `load_best_params()` — 파일 없으면 `FileNotFoundError` (SSOT-FF01)
- [x] `apply_phase2_params()` — backbone 불변식 보장
- [ ] `tuning → scripts` 레이어 위반 리팩토링

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) | 학습 파이프라인 정의 (What) |
| [Contract_training_pipeline.md](Contract_training_pipeline.md) | Phase2Trainer 계약 |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | 산출물 저장 계약 |
