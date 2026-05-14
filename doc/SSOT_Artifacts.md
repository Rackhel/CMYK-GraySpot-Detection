# SSOT Artifacts — 산출물 의미 스키마 / Artifact Semantic Schema

CMYK Grayspot Detection System 의 산출물 파일명 패턴, 경로, 내용 스키마에 관한 단일 진실 공급원.

This document is the authoritative reference for artifact filename patterns, directory layout, and content schemas.

> **목적 / Purpose**: 산출물의 의미(semantic)와 스키마 정의
> **역할 / Role**: "What" — 산출물 의미 정의 (파일명 패턴, 경로, 내용 스키마)
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md)

---

## Table of Contents / 목차

1. [절대 원칙 / Mandatory Rules](#1-절대-원칙--mandatory-rules)
2. [파일명 패턴 원칙 / Filename Pattern Principles](#2-파일명-패턴-원칙--filename-pattern-principles)
3. [산출물 카탈로그 / Artifact Catalog](#3-산출물-카탈로그--artifact-catalog)
4. [디렉토리 구조 / Directory Structure](#4-디렉토리-구조--directory-structure)
5. [아티팩트 의존 관계 / Artifact Dependency Graph](#5-아티팩트-의존-관계--artifact-dependency-graph)
6. [state_dict 호환 규칙 / state_dict Compatibility Rules](#6-state_dict-호환-규칙--state_dict-compatibility-rules)

---

## 1. 절대 원칙 / Mandatory Rules

### ❌ 하드코딩 금지 / No Hardcoded Paths

- `"best_Y.pt"` 등 **문자열 리터럴 경로 사용 금지** / **String literal paths such as `"best_Y.pt"` are prohibited**
- 모든 경로는 **config + channel + tag 변수에서 동적 생성** / All paths must be **dynamically generated from config + channel + tag variables**

```python
# ❌ 금지 / Prohibited
path = "data_set/models/best_Y.pt"

# ✅ 허용 / Allowed
path = model_dir / f"best_{channel}.pt"
path = model_dir / f"phase0_backbone_{channel}_{backbone_tag(backbone)}.pt"
```

### ❌ Fallback 금지 / No Fallbacks

- SSOT 아티팩트 없으면 **즉시 에러** (`SSOT-FF01`) / If a SSOT artifact is missing, **raise an error immediately** (`SSOT-FF01`)
- Phase 0 backbone 없이 Phase 2 시작 금지 / Starting Phase 2 without a Phase 0 backbone is prohibited

### ✅ SSOT 우선순위 / SSOT Priority

| 순위 / Rank | 원천 / Source | 용도 / Purpose |
|------|------|------|
| 1 | 학습 산출물 (`.pt` 파일) / Training artifacts (`.pt` files) | 모델 가중치의 유일 원천 / Sole source of model weights |
| 2 | `config/config.json` | 파라미터의 유일 원천 / Sole source of parameters |
| 3 | ❌ 에러 / Error | Fallback 금지 / No fallbacks |

---

## 2. 파일명 패턴 원칙 / Filename Pattern Principles

### 2.1 기본 구조 / Base Structure

```
{artifact_type}_{channel}_{tag}.{ext}
```

### 2.2 필드 의미 / Field Semantics

| 필드 / Field | 의미 / Meaning | 예시 / Example |
|------|------|------|
| `{artifact_type}` | 산출물 유형 / Artifact type | `phase0_backbone`, `best`, `phase2` |
| `{channel}` | CMYK 색상 채널 / CMYK color channel | `Y`, `M`, `C`, `K` |
| `{tag}` | Backbone 단축 식별자 / Backbone short identifier | `effb0`, `res50` |
| `{ver}` | 버전 번호 / Version number | `v1`, `v2` |
| `{ext}` | 파일 확장자 / File extension | `.pt`, `.json`, `.csv`, `.html` |

### 2.3 Backbone 태그 / Backbone Tags

```python
from utils import backbone_tag          # ✅ 권장 / Recommended (utils_model.py 정의)
# from training import backbone_tag     # 하위 호환 유지 / Also works (trainer.py 로컬 복사본)

backbone_tag("efficientnet_b0")  # → "effb0"
backbone_tag("resnet50")         # → "res50"
```

### 2.4 Channel Invariant

동일 `{channel}` = 동일 의미 공간 / Same `{channel}` = same semantic space:

```
phase0_backbone_Y_effb0.pt → best_Y.pt   (Y 채널 연속 학습 / Continuous Y-channel training ✅)
phase0_backbone_Y_effb0.pt → best_M.pt   (채널 교차 금지 / Cross-channel prohibited ❌ SSOT-FF01)
```

---

## 3. 산출물 카탈로그 / Artifact Catalog

### 3.1 Phase 0 산출물 / Phase 0 Artifacts

| 파일명 패턴 / Pattern | 형식 / Format | 내용 / Content | SSOT |
|------------|------|------|-----------|
| `phase0_backbone_{ch}_{tag}.pt` | PyTorch state_dict | Phase 0 완료 backbone + ProjectionHead / Completed Phase 0 backbone + ProjectionHead | ✅ Hard |

**예시 / Example**: `phase0_backbone_Y_effb0.pt`

**state_dict 스키마 / Schema**:

```python
{
    "backbone.features.0.0.weight": Tensor,  # EfficientNet-B0 첫 레이어 / EfficientNet-B0 first layer
    # ... EfficientNet-B0 layers ...
    "head.fc1.weight":  Tensor,              # ProjectionHead fc1
    "head.fc1.bias":    Tensor,
    "head.bn1.weight":  Tensor,
    "head.fc2.weight":  Tensor,              # ProjectionHead fc2 (→ projection_dim=128)
    "head.fc2.bias":    Tensor,
}
```

### 3.2 Phase 2 산출물 / Phase 2 Artifacts

| 파일명 패턴 / Pattern | 형식 / Format | 내용 / Content | SSOT |
|------------|------|------|-----------|
| `best_{ch}.pt` | PyTorch state_dict | Best val_acc 기준 저장 모델 / Model saved by best val_acc | ✅ Hard |
| `phase2_{ch}_{tag}_{ver}.pt` | PyTorch state_dict | 버전 번호 포함 복사본 / Copy with version number | ✅ Hard |

**예시 / Examples**:
- `best_Y.pt`
- `phase2_Y_effb0_v1.pt`

**state_dict 스키마 / Schema**:

```python
{
    "backbone.features.0.0.weight": Tensor,  # backbone (Phase 0에서 이월 / carried over from Phase 0)
    # ... backbone layers ...
    "head.fc1.weight":  Tensor,              # ClassifierHead fc1
    "head.bn1.weight":  Tensor,
    "head.fc1.bias":    Tensor,
    "head.fc2.weight":  Tensor,              # ClassifierHead fc2 (→ num_levels=6)
    "head.fc2.bias":    Tensor,
}
```

### 3.3 학습 이력 / Training History Artifacts

| 파일명 패턴 / Pattern | 형식 / Format | 내용 / Content | SSOT |
|------------|------|------|-----------|
| `phase0_history_{ch}.csv` | CSV | Phase 0 에폭별 손실 이력 / Phase 0 per-epoch loss history | ❌ Soft |
| `phase2_history_{ch}.csv` | CSV | Phase 2 에폭별 지표 이력 / Phase 2 per-epoch metric history | ❌ Soft |
| `phase0_summary.json` | JSON | Phase 0 실행 요약 / Phase 0 run summary | ❌ Soft |
| `phase2_summary_{ver}.json` | JSON | Phase 2 실행 요약 / Phase 2 run summary | ❌ Soft |

### 3.4 설정 스냅샷 / Config Snapshot Artifacts

| 파일명 패턴 / Pattern | 형식 / Format | 내용 / Content | SSOT |
|------------|------|------|-----------|
| `config_snapshot_{tag}_{ts}.json` | JSON | 실행 시점 설정 + 환경 + git 상태 / Config + environment + git state at run time | ❌ Soft |

**예시 / Example**: `config_snapshot_Y_phase0_20260508_143000.json`

**스키마 / Schema**:

```json
{
  "timestamp": "2026-05-08T14:30:00.123456",
  "tag": "Y_phase0",
  "environment": {
    "python": "3.11.5 (main, ...)",
    "platform": "macOS-...",
    "torch": "2.1.0",
    "cuda_available": false,
    "cuda_version": null
  },
  "git": {
    "commit": "a1b2c3d",
    "dirty": false
  },
  "config": { ... }
}
```

### 3.5 평가 산출물 / Evaluation Artifacts

| 파일명 패턴 / Pattern | 형식 / Format | 내용 / Content | SSOT |
|------------|------|------|-----------|
| `evaluation_results_{name}.csv` | CSV | 샘플별 예측 결과 / Per-sample prediction results | ❌ Soft |
| `misclassified_{name}.csv` | CSV | 오분류 샘플 목록 / Misclassified sample list | ❌ Soft |
| `metrics_summary_{name}.json` | JSON | 채널별 집계 지표 / Per-channel aggregated metrics | ❌ Soft |
| `baseline_report_{ch}.html` | HTML | 시각화 리포트 / Visualization report | ❌ Soft |

### 3.6 Optuna 산출물 / Optuna Artifacts

> **채널 suffix 규칙**: Optuna 산출물의 `{ch}` 는 **소문자**를 사용한다 (`y`, `m`, `c`, `k`, `all`).
> 학습 모델 산출물(`best_Y.pt` 등)이 대문자를 사용하는 것과 다름에 주의.
>
> **Channel suffix rule**: Optuna artifacts use **lowercase** `{ch}` (`y`, `m`, `c`, `k`, `all`).
> Note this differs from training model artifacts (e.g. `best_Y.pt`) which use uppercase.

| 파일명 패턴 / Pattern | 형식 / Format | 내용 / Content | SSOT | 생산자 / Producer |
|------------|------|------|-----------|---|
| `study_{ch}.db` | SQLite | Optuna study 전체 trial 기록 / All trial records for Optuna study | ❌ Soft | `optuna_tuner.py` |
| `best_params_{ch}.json` | JSON | 최적 하이퍼파라미터 / Optimal hyperparameters | ❌ Soft | `optuna_utils.save_best_params()` |
| `trials_summary_{ch}.json` | JSON | 전체 trial 결과 요약 (number, value, state, params) / All trial results summary | ❌ Soft | `optuna_utils.save_trials_summary()` |

**예시 / Examples** (all-channel 튜닝 기준):
- `outputs/optuna/study_all.db`
- `outputs/optuna/best_params_all.json`
- `outputs/optuna/trials_summary_all.json`

**best_params JSON 스키마 / Schema** (EfficientNet-B0 기준):

```json
{
  "learning_rate": 0.0001,
  "batch_size": 16,
  "weight_decay": 0.0001,
  "epochs": 10,
  "dropout": 0.2,
  "hidden_dim": 128
}
```

**ResNet-50 추가 필드 / ResNet-50 additional field**:
```json
{
  "mid_dim": 512
}
```

---

## 4. 디렉토리 구조 / Directory Structure

```
CMYK_MAIN/
├── doc/                              # 아키텍처 & 설계 문서 / Architecture & design documents
│   ├── SSOT_*.md                    ← 단일 진실 공급원 문서 / Single Source of Truth documents
│   ├── Contract.md                  ← 모듈 인터페이스 계약 / Module interface contracts
│   ├── ADR_*.md                     ← 아키텍처 결정 기록 / Architecture Decision Records
│   └── TDD.md                       ← TDD 전략 문서 / TDD strategy document
├── src/
│   ├── config/
│   │   ├── config.json              ← 모든 런타임 파라미터 SSOT / SSOT for all runtime parameters
│   │   ├── dependencies.json        ← 의존성 레지스트리 / Dependency registry
│   │   └── pyproject.toml           ← 빌드 & 도구 설정 / Build & tool settings
│   ├── data/                        dataset.py, augmentation.py, preprocessing.py
│   ├── models/                      backbone.py, classifier.py, grayspot_model.py, projection_head.py
│   ├── training/                    trainer.py, contrastive_loss.py, losses.py
│   ├── evaluation/                  metrics.py, confusion.py, evaluator.py (Orchestrator), evaluator_inference.py, evaluator_metrics.py, evaluator_export.py, evaluator_charts.py
│   ├── inference/                   predictor.py (Orchestrator), predictor_device.py, predictor_loader.py, predictor_inference.py
│   ├── reporting/                   html_report.py
│   ├── tuning/                      optuna_tuner.py, search_space.py, optuna_utils.py
│   ├── utils/                       utils_config.py, utils_model.py, logger.py
│   ├── scripts/                     train.py, run_phase0.py, run_phase2.py, run_baseline.py, run_optuna.py
│   ├── tests/
│   │   ├── unit/                    ← 단위 테스트 (I/O 없음, < 1초) / Unit tests (no I/O, < 1s)
│   │   ├── integration/             ← 통합 테스트 (모듈 연결 검증) / Integration tests (module connection)
│   │   └── smoke/                   ← 스모크 테스트 (실 데이터, 전체 파이프라인) / Smoke tests (real data, full pipeline)
│   └── notebooks/                   01~06_*.ipynb
├── data_set/                        # 원본 데이터 (git-ignored) / Raw data (git-ignored)
│   └── labeled/
│       └── {channel}/{level}/*.png  ← 원본 라벨 이미지 / Original labeled images
├── outputs/                         # 모든 학습 산출물 / All training artifacts
│   ├── checkpoints/                 ← 모델 가중치 & 학습 이력 / Model weights & training history
│   │   ├── best_{ch}.pt                    ← Phase 2 채널별 최적 모델 / Best model per channel
│   │   ├── phase0_v1.pt                    ← Phase 0 전 채널 통합 체크포인트 / Phase 0 all-channel checkpoint
│   │   ├── phase2_{ch}_{tag}_{ver}.pt      ← Phase 2 버전 체크포인트 / Phase 2 versioned checkpoint
│   │   ├── phase0_history_{ch}.csv
│   │   ├── phase2_history_{ch}.csv
│   │   ├── phase0_summary.json
│   │   └── phase2_summary_{ver}.json
│   ├── snapshots/                   ← 실행 시점 config 스냅샷 / Config snapshot at run time
│   │   └── config_snapshot_{tag}_{ts}.json
│   ├── logs/                        ← 실행 로그 / Execution logs
│   ├── reports/                     ← HTML 평가 리포트 / HTML evaluation reports
│   │   ├── phase2_{ver}.html
│   │   └── confusion/
│   │       ├── cm_{ch}.html
│   │       └── cm_overall.html
│   └── optuna/                      ← Optuna 튜닝 결과 / Optuna tuning results
│       ├── study_{ch}.db                  ← ch = 소문자 / lowercase (y/m/c/k/all)
│       ├── best_params_{ch}.json          ← ch = 소문자 / lowercase
│       └── trials_summary_{ch}.json       ← ch = 소문자 / lowercase
├── pytest.ini                       ← Pytest 설정
├── requirements.txt
├── Dockerfile
└── LICENSE
```

---

## 5. 아티팩트 의존 관계 / Artifact Dependency Graph

```
config/config.json
    ↓ (파라미터 / parameters)
Phase 0 학습 / Training (Phase0Trainer)
    ↓ (backbone weights)
phase0_backbone_{ch}_{tag}.pt        ← SSOT-FF01 검증 대상 / FF01 validation target
    ↓ (switch_to_phase2)
Phase 2 학습 / Training (Phase2Trainer)
    ↓ (best val_acc checkpoint)
best_{ch}.pt
    ↓ (model.eval())
평가 / Evaluation (Evaluator.run)
    ↓ (metrics dict)
metrics_summary_{name}.json + baseline_report_{ch}.html
```

### 5.1 Fail-Fast 검증 포인트 / Validation Checkpoints

| 전환 / Transition | 검증 / Check | SSOT 코드 / Code |
|------|------|-----------|
| Phase 0 → Phase 2 | `phase0_backbone_{ch}_{tag}.pt` 존재 확인 / Verify existence | `SSOT-FF01` |
| Phase 2 → 평가 / Evaluation | `best_{ch}.pt` 존재 확인 / Verify existence | `SSOT-FF01` |
| 평가 / Evaluation → 리포트 / Report | metrics dict 필수 키 (`accuracy`, `macro_f1`, `mae`) 존재 확인 / Verify required keys present | `SSOT-FF01` |

---

## 6. state_dict 호환 규칙 / state_dict Compatibility Rules

### 6.1 Phase 0 → Phase 2 전환 / Phase 0 → Phase 2 Switch

```python
# GrayspotModel.switch_to_phase2()에서의 로드 규칙 / Load rules in GrayspotModel.switch_to_phase2()
state = torch.load(backbone_path)
backbone_keys = {k: v for k, v in state.items() if k.startswith("backbone.")}
model.load_state_dict(backbone_keys, strict=False)
```

- `backbone.*` prefix 키만 선택적 로드 / Only `backbone.*` prefix keys are loaded
- Head 키 무시 (ProjectionHead → ClassifierHead 교체) / Head keys discarded (ProjectionHead replaced by ClassifierHead)
- `strict=False`: 새 ClassifierHead 키 불일치 허용 / Allows new ClassifierHead key mismatches

### 6.2 추론 시 로드 / Inference Load

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint, strict=False)
```

- `weights_only=True`: pickle 보안 — 임의 코드 실행 방지 / Pickle security — prevent arbitrary code execution
- `strict=False`: 버전 간 호환성 / Cross-version compatibility

---
