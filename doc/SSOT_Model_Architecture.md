# SSOT Model Architecture — 모델 구조 / Model Architecture

CMYK Grayspot Detection System 의 신경망 구조, backbone, head에 관한 단일 진실 공급원.

This document is the authoritative reference for the neural network architecture, backbone, and head components.

> **목적 / Purpose**: 모델 구조의 의미(semantic) 정의
> **역할 / Role**: "What" — 레이어 구성, feature_dim, phase 전환 규약
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md), [SSOT_Artifacts.md](SSOT_Artifacts.md)

---

## Table of Contents / 목차

1. [모델 구성 요소 / Components](#1-모델-구성-요소--components)
2. [GrayspotModel — 통합 모델 / Unified Model](#2-grayspotmodel--통합-모델--unified-model)
3. [Backbone 지원 목록 / Supported Backbones](#3-backbone-지원-목록--supported-backbones)
4. [ProjectionHead — Phase 0 Head](#4-projectionhead--phase-0-head)
5. [ClassifierHead — Phase 2 Head](#5-classifierhead--phase-2-head)
6. [Phase 전환 규칙 / Phase Transition Rules](#6-phase-전환-규칙--phase-transition-rules)
7. [Hard SSOT 파라미터 / Hard SSOT Parameters](#7-hard-ssot-파라미터--hard-ssot-parameters)
8. [SSOT 위반 현황 / Violations](#8-ssot-위반-현황--violations)

---

## 1. 모델 구성 요소 / Components

| 클래스 / Class | 모듈 / Module | Phase | 역할 / Role |
|---|---|---|---|
| `GrayspotModel` | `models.grayspot_model` | 0, 2 | 통합 모델 — Backbone + Head 조합 / Unified model combining backbone and head |
| `build_backbone` | `models.backbone` | 0, 2 | Backbone 로드 팩토리 함수 / Factory function for loading backbones |
| `ProjectionHead` | `models.projection_head` | 0 | SimCLR Contrastive 투영 Head |
| `ClassifierHead` | `models.classifier` | 2 | Supervised 분류 Head |

```python
from models import GrayspotModel, build_backbone, ProjectionHead, ClassifierHead
```

---

## 2. GrayspotModel — 통합 모델 / Unified Model

### 2.1 Phase 0 구성 / Phase 0 Configuration

```python
model = GrayspotModel(cfg, phase=0)
# Backbone (feature extractor) + ProjectionHead
# Input:  (B, 3, 128, 128) float32
# Output: (B, projection_dim=128) L2-normalized projection vector
```

### 2.2 Phase 2 구성 / Phase 2 Configuration

```python
model = GrayspotModel(cfg, phase=2)
# Backbone (feature extractor) + ClassifierHead
# Input:  (B, 3, 128, 128) float32
# Output: (B, num_levels=6) logits (Softmax 없음 / No Softmax)
```

### 2.3 Phase 전환 / Phase Switching

```python
# Phase 0 backbone을 Phase 2 모델로 전환
# Switch Phase 0 backbone weights into Phase 2 model
model.switch_to_phase2(backbone_path)
# backbone_path: data_set/models/phase0_backbone_{channel}_{tag}.pt
```

---

## 3. Backbone 지원 목록 / Supported Backbones

| 이름 / Name | 태그 / Tag | feature_dim | 출처 / Source | Hard SSOT |
|---|---|---|---|---|
| `efficientnet_b0` | `effb0` | 1280 | `torchvision EfficientNet_B0_Weights.DEFAULT` | ✅ |
| `resnet50` | `res50` | 2048 | `torchvision ResNet50_Weights.DEFAULT` | ✅ |

### 3.1 Backbone 태그 함수 / Tag Function

```python
from training import backbone_tag

backbone_tag("efficientnet_b0")  # → "effb0"
backbone_tag("resnet50")         # → "res50"
```

태그는 아티팩트 파일명 생성에 사용된다. / Tags are used for artifact filename generation.

### 3.2 Backbone 로드 팩토리 / Factory Function

```python
from models import build_backbone

backbone, feature_dim = build_backbone(cfg)
# Returns: (nn.Sequential, int)
# feature_dim: 1280 for efficientnet_b0, 2048 for resnet50
```

### 3.3 Channel Invariant

> **필수 / Required**: Phase 0 과 Phase 2 는 반드시 **동일 backbone, 동일 channel** 을 사용해야 한다.
> Phase 0 and Phase 2 **must** use the same backbone **and** the same channel.

```
phase0_backbone_Y_effb0.pt → GrayspotModel(backbone="efficientnet_b0", channel="Y") Phase 2 ✅
phase0_backbone_Y_effb0.pt → GrayspotModel(backbone="efficientnet_b0", channel="M") Phase 2 ❌  (SSOT-FF01)
phase0_backbone_Y_res50.pt → GrayspotModel(backbone="efficientnet_b0", channel="Y") Phase 2 ❌  (구조 불일치)
```

---

## 4. ProjectionHead — Phase 0 Head

Phase 0 SimCLR Contrastive Learning용 투영 Head.

```python
# 구조 / Architecture
Linear(feature_dim, projection_dim) → BatchNorm → ReLU
→ Linear(projection_dim, projection_dim)
→ L2 Normalize

# 파라미터 / Parameters
feature_dim    = 1280 (effb0) or 2048 (res50)   ← Hard SSOT
projection_dim = config["phase0"]["projection_dim"]  = 128  ← Hard SSOT
```

| 파라미터 / Parameter | config 키 / Key | 기본값 / Default | Hard SSOT |
|---|---|---|---|
| `projection_dim` | `phase0.projection_dim` 🟢 | 128 | ✅ |

---

## 5. ClassifierHead — Phase 2 Head

Phase 2 Supervised Classification용 분류 Head.

```python
# 구조 / Architecture
Linear(feature_dim, hidden_dim) → BatchNorm → ReLU → Dropout(dropout)
→ Linear(hidden_dim, num_levels)
# 출력: raw logits (Softmax 없음 / No Softmax applied)

# 파라미터 / Parameters
feature_dim = 1280 (effb0) or 2048 (res50)      ← Hard SSOT
hidden_dim  = config["phase2"]["hidden_dim"]  = 256  ← Hard SSOT
num_levels  = config["data"]["num_levels"]    = 6    ← Hard SSOT
dropout     = config["phase2"]["dropout"]     = 0.3  ← Soft SSOT
```

| 파라미터 / Parameter | config 키 / Key | 기본값 / Default | SSOT 분류 |
|---|---|---|---|
| `hidden_dim` | `phase2.hidden_dim` 🟢 | 256 | Hard |
| `num_levels` | `data.num_levels` 🟢 | 6 | Hard |
| `dropout` | `phase2.dropout` 🟢 | 0.3 | Soft |

---

## 6. Phase 전환 규칙 / Phase Transition Rules

### 6.1 switch_to_phase2() 로드 규칙 / Load Rules

```python
state = torch.load(backbone_path)
backbone_keys = {k: v for k, v in state.items() if k.startswith("backbone.")}
model.load_state_dict(backbone_keys, strict=False)
```

- `backbone.*` prefix 키만 선택적 로드 / Only `backbone.*` prefix keys are loaded
- Head 키 무시 (ProjectionHead → ClassifierHead 교체) / Head keys ignored (head is replaced)
- `strict=False`: 새 head 키 불일치 허용 / Allows new head key mismatches

### 6.2 추론 시 로드 / Inference Load Rules

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint, strict=False)
```

- `weights_only=True`: pickle 보안 (임의 코드 실행 방지) / Pickle security
- `strict=False`: 버전 간 호환성 / Cross-version compatibility

---

## 7. Hard SSOT 파라미터 / Hard SSOT Parameters

변경 시 **재학습 필수** / **Retrain required** on change:

| 파라미터 / Parameter | 값 / Value | config 키 / Key | 영향 / Impact |
|---|---|---|---|
| `num_levels` | 6 | `data.num_levels` | ClassifierHead 출력 차원 / Output dim |
| `image_size` | 128 | `data.image_size` | 모델 입력 크기 / Model input size |
| `backbone` | `efficientnet_b0` | `model.backbone` | 전체 구조 변경 / Entire architecture |
| `projection_dim` | 128 | `phase0.projection_dim` | Phase 0 head 구조 |
| `hidden_dim` | 256 | `phase2.hidden_dim` | Phase 2 head 구조 |
| Phase 순서 | Phase 0 → 2 | — | backbone 가중치 의존성 |
| 색상 공간 / Color space | BGR [0, 1] | — | 입력 분포 / Input distribution |

---

## 8. SSOT 위반 현황 / Violations

| 코드 / Code | 위반 내용 / Violation | 등급 / Level |
|---|---|---|
| SSOT-FF01 | Phase 0 backbone 없이 Phase 2 `switch_to_phase2()` 호출 | Level 1 — 즉시 중단 |
| SSOT-PH01 | Phase 0 학습 없이 Phase 2 직행 (backbone 미존재) | Level 1 — 즉시 중단 |

---

**Version**: 0.1.0
**Last Updated**: 2026-05-08
**Applies to**: CMYK Grayspot Detection System v0.1.0+
