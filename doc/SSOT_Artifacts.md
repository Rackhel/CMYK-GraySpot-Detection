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

- `"best_Y.pt"` 등 **문자열 리터럴 경로 사용 금지**
- 모든 경로는 **config + channel + tag 변수에서 동적 생성**

```python
# ❌ 금지 / Prohibited
path = "data_set/models/best_Y.pt"

# ✅ 허용 / Allowed
path = model_dir / f"best_{channel}.pt"
path = model_dir / f"phase0_backbone_{channel}_{backbone_tag(backbone)}.pt"
```

### ❌ Fallback 금지 / No Fallbacks

- SSOT 아티팩트 없으면 **즉시 에러** (`SSOT-FF01`)
- Phase 0 backbone 없이 Phase 2 시작 금지

### ✅ SSOT 우선순위 / SSOT Priority

| 순위 / Rank | 원천 / Source | 용도 / Purpose |
|------|------|------|
| 1 | 학습 산출물 (`.pt` 파일) | 모델 가중치의 유일 원천 / Sole source of model weights |
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
| `{artifact_type}` | 산출물 유형 | `phase0_backbone`, `best`, `phase2` |
| `{channel}` | CMYK 색상 채널 | `Y`, `M`, `C`, `K` |
| `{tag}` | Backbone 단축 식별자 | `effb0`, `res50` |
| `{ver}` | 버전 번호 | `v1`, `v2` |
| `{ext}` | 파일 확장자 | `.pt`, `.json`, `.csv`, `.html` |

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
phase0_backbone_Y_effb0.pt → best_Y.pt   (Y 채널 연속 학습 ✅)
phase0_backbone_Y_effb0.pt → best_M.pt   (채널 교차 금지 ❌ SSOT-FF01)
```

---

## 3. 산출물 카탈로그 / Artifact Catalog

### 3.1 Phase 0 산출물 / Phase 0 Artifacts

| 파일명 패턴 / Pattern | 형식 / Format | 내용 / Content | SSOT |
|------------|------|------|-----------|
| `phase0_backbone_{ch}_{tag}.pt` | PyTorch state_dict | Phase 0 완료 backbone + ProjectionHead | ✅ Hard |

**예시 / Example**: `phase0_backbone_Y_effb0.pt`

**state_dict 스키마 / Schema**:

```python
{
    "backbone.features.0.0.weight": Tensor,  # EfficientNet-B0 첫 레이어
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
| `best_{ch}.pt` | PyTorch state_dict | Best val_acc 기준 저장 모델 | ✅ Hard |
| `phase2_{ch}_{tag}_{ver}.pt` | PyTorch state_dict | 버전 번호 포함 복사본 | ✅ Hard |

**예시 / Examples**:
- `best_Y.pt`
- `phase2_Y_effb0_v1.pt`

**state_dict 스키마 / Schema**:

```python
{
    "backbone.features.0.0.weight": Tensor,  # backbone (Phase 0에서 이월)
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
| `phase0_history_{ch}.csv` | CSV | Phase 0 에폭별 손실 이력 | ❌ Soft |
| `phase2_history_{ch}.csv` | CSV | Phase 2 에폭별 지표 이력 | ❌ Soft |
| `phase0_summary.json` | JSON | Phase 0 실행 요약 | ❌ Soft |
| `phase2_summary_{ver}.json` | JSON | Phase 2 실행 요약 | ❌ Soft |

### 3.4 설정 스냅샷 / Config Snapshot Artifacts

| 파일명 패턴 / Pattern | 형식 / Format | 내용 / Content | SSOT |
|------------|------|------|-----------|
| `config_snapshot_{tag}_{ts}.json` | JSON | 실행 시점 설정 + 환경 + git 상태 | ❌ Soft |

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
| `evaluation_results_{name}.csv` | CSV | 샘플별 예측 결과 | ❌ Soft |
| `misclassified_{name}.csv` | CSV | 오분류 샘플 목록 | ❌ Soft |
| `metrics_summary_{name}.json` | JSON | 채널별 집계 지표 | ❌ Soft |
| `baseline_report_{ch}.html` | HTML | 시각화 리포트 | ❌ Soft |

### 3.6 Optuna 산출물 / Optuna Artifacts

| 파일명 패턴 / Pattern | 형식 / Format | 내용 / Content | SSOT |
|------------|------|------|-----------|
| `study_{ch}.db` | SQLite | Optuna study 전체 trial 기록 | ❌ Soft |
| `best_params_{ch}.json` | JSON | 최적 하이퍼파라미터 | ❌ Soft |

---

## 4. 디렉토리 구조 / Directory Structure

```
CMYK_MAIN/
├── data_set/
│   ├── labeled/
│   │   └── {channel}/{level}/*.png      ← 원본 데이터
│   └── models/
│       ├── phase0_backbone_{ch}_{tag}.pt    ← Phase 0 backbone (채널별)
│       ├── best_{ch}.pt                     ← Phase 2 최적 모델 (채널별)
│       └── last_{ch}.pt                     ← Phase 2 마지막 모델
└── outputs/
    ├── checkpoints/
    │   ├── phase0_history_{ch}.csv
    │   ├── phase0_summary.json
    │   ├── phase2_{ch}_{tag}_{ver}.pt        ← 버전 체크포인트
    │   ├── phase2_history_{ch}.csv
    │   └── phase2_summary_{ver}.json
    ├── snapshots/
    │   └── config_snapshot_{tag}_{ts}.json  ← 실행 시점 스냅샷
    ├── reports/
    │   ├── eval_dashboard.html
    │   ├── per_class_metrics.html
    │   ├── mae_heatmap.html
    │   ├── misclassified_scatter.html
    │   ├── confidence_distribution.html
    │   ├── confusion/
    │   │   ├── cm_{ch}.html
    │   │   └── cm_overall.html
    │   ├── evaluation_results_{name}.csv
    │   ├── misclassified_{name}.csv
    │   └── metrics_summary_{name}.json
    └── optuna/
        ├── study_{ch}.db
        └── best_params_{ch}.json
```

---

## 5. 아티팩트 의존 관계 / Artifact Dependency Graph

```
config/config.json
    ↓ (파라미터 / parameters)
Phase 0 학습 (Phase0Trainer)
    ↓ (backbone weights)
phase0_backbone_{ch}_{tag}.pt        ← SSOT-FF01 검증 대상 / FF01 validation target
    ↓ (switch_to_phase2)
Phase 2 학습 (Phase2Trainer)
    ↓ (best val_acc checkpoint)
best_{ch}.pt
    ↓ (model.eval())
평가 (Evaluator.run)
    ↓ (metrics dict)
metrics_summary_{name}.json + baseline_report_{ch}.html
```

### 5.1 Fail-Fast 검증 포인트 / Validation Checkpoints

| 전환 / Transition | 검증 / Check | SSOT 코드 / Code |
|------|------|-----------|
| Phase 0 → Phase 2 | `phase0_backbone_{ch}_{tag}.pt` 존재 확인 | `SSOT-FF01` |
| Phase 2 → 평가 | `best_{ch}.pt` 존재 확인 | `SSOT-FF01` |
| 평가 → 리포트 | metrics dict 필수 키 (`accuracy`, `macro_f1`, `mae`) 존재 확인 | `SSOT-FF01` |

---

## 6. state_dict 호환 규칙 / state_dict Compatibility Rules

### 6.1 Phase 0 → Phase 2 전환 / Phase 0 → Phase 2 Switch

```python
# GrayspotModel.switch_to_phase2()에서의 로드 규칙
state = torch.load(backbone_path)
backbone_keys = {k: v for k, v in state.items() if k.startswith("backbone.")}
model.load_state_dict(backbone_keys, strict=False)
```

- `backbone.*` prefix 키만 선택적 로드 / Only `backbone.*` prefix keys are loaded
- Head 키 무시 (ProjectionHead → ClassifierHead 교체) / Head keys discarded
- `strict=False`: 새 ClassifierHead 키 불일치 허용

### 6.2 추론 시 로드 / Inference Load

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint, strict=False)
```

- `weights_only=True`: pickle 보안 — 임의 코드 실행 방지 / Prevent arbitrary code execution
- `strict=False`: 버전 간 호환성 / Cross-version compatibility

---

**Version**: 0.2.0
**Last Updated**: 2026-05-08
**Applies to**: CMYK Grayspot Detection System v0.1.0+
