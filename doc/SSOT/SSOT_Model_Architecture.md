---
type: ssot
domain: model_architecture
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "SSOT_Artifacts.md"
  - "SSOT_Training_Pipeline.md"
---

# SSOT Model Architecture — 모델 구조 / Model Architecture

CMYK Grayspot Detection System 의 신경망 구조, backbone, head에 관한 단일 진실 공급원.

This document is the authoritative reference for the neural network architecture, backbone, and head components.

> **목적 / Purpose**: 모델 구조의 의미(semantic) 정의
> **역할 / Role**: "What" — 레이어 구성, feature_dim, phase 전환 규약
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md), [SSOT_Artifacts.md](SSOT_Artifacts.md)

---

## 1. 모델 구성 요소 / Components

| 클래스 / Class | 모듈 / Module | Phase | 역할 / Role |
|---|---|---|---|
| `GrayspotModel` | `models.grayspot_model` | 0, 2 | 통합 모델 — Backbone + Head 조합 / Unified model combining backbone and head |
| `build_backbone` | `models.backbone` | 0, 2 | Backbone 로드 팩토리 함수 / Factory function for loading backbones |
| `ProjectionHead` | `models.projection_head` | 0 | SimCLR Contrastive 투영 Head / SimCLR Contrastive projection head |
| `ClassifierHead` | `models.classifier` | 2 | Supervised 분류 Head / Supervised classification head |

---

## 2. GrayspotModel — 통합 모델 / Unified Model

### 2.1 Phase 0 구성 / Phase 0 Configuration

| 항목 / Item | 값 / Value |
|---|---|
| 구성 / Architecture | Backbone + ProjectionHead |
| 입력 / Input | `(B, 3, 128, 128)` float32 |
| 출력 / Output | `(B, projection_dim=128)` L2-normalized projection vector |

### 2.2 Phase 2 구성 / Phase 2 Configuration

| 항목 / Item | 값 / Value |
|---|---|
| 구성 / Architecture | Backbone + ClassifierHead |
| 입력 / Input | `(B, 3, 128, 128)` float32 |
| 출력 / Output | `(B, num_levels=6)` raw logits (Softmax 없음 / No Softmax) |

### 2.3 Phase 전환 / Phase Switching

`GrayspotModel.switch_to_phase2(backbone_path)` — Phase 0 backbone 가중치를 Phase 2 모델로 전환.

- `backbone_path`: `{models_dir}/phase0_backbone_{channel}_{tag}.pt`

---

## 3. Backbone 지원 목록 / Supported Backbones

| 이름 / Name | 태그 / Tag | feature_dim | 출처 / Source | Hard SSOT |
|---|---|---|---|---|
| `efficientnet_b0` | `effb0` | 1280 | `torchvision EfficientNet_B0_Weights.DEFAULT` | ✅ |
| `resnet50` | `res50` | 2048 | `torchvision ResNet50_Weights.DEFAULT` | ✅ |

### 3.1 Backbone 태그 함수 / Tag Function

`utils.utils_model.backbone_tag(backbone_name: str) → str` — 아티팩트 파일명 생성에 사용.
Tags are used for artifact filename generation (e.g., `phase0_backbone_Y_effb0.pt`).

| backbone 이름 / Name | 반환 태그 / Tag |
|---|---|
| `efficientnet_b0` | `effb0` |
| `resnet50` | `res50` |

### 3.2 Backbone 로드 팩토리 / Factory Function

`models.build_backbone(backbone_name: str) → (nn.Sequential, int)`

- 반환: `(backbone module, feature_dim)`
- `feature_dim`: `1280` (efficientnet_b0), `2048` (resnet50)

### 3.3 Channel Invariant

> **필수 / Required**: Phase 0 과 Phase 2 는 반드시 **동일 backbone, 동일 channel** 을 사용해야 한다.
> Phase 0 and Phase 2 **must** use the same backbone **and** the same channel.

```
phase0_backbone_Y_effb0.pt → GrayspotModel(backbone="efficientnet_b0", channel="Y") Phase 2 ✅
phase0_backbone_Y_effb0.pt → GrayspotModel(backbone="efficientnet_b0", channel="M") Phase 2 ❌  (SSOT-FF01)
phase0_backbone_Y_res50.pt → GrayspotModel(backbone="efficientnet_b0", channel="Y") Phase 2 ❌  (구조 불일치 / architecture mismatch)
```

---

## 4. ProjectionHead — Phase 0 Head

Phase 0 SimCLR Contrastive Learning용 투영 Head. / Projection head for Phase 0 SimCLR Contrastive Learning.

**구조 / Architecture**: `Linear(feature_dim, proj_hidden) → BatchNorm → ReLU → Linear(proj_hidden, projection_dim) → L2 Normalize`

| 파라미터 / Parameter | config 키 / Key | 기본값 / Default | Hard SSOT |
|---|---|---|---|
| `feature_dim` | — | 1280 (effb0) / 2048 (res50) | ✅ |
| `proj_hidden` | `phase0.hidden_dim` 🟢 | 256 | ✅ |
| `projection_dim` | `phase0.projection_dim` 🟢 | 128 | ✅ |

---

## 5. ClassifierHead — Phase 2 Head

Phase 2 Supervised Classification용 분류 Head. Backbone별로 구조가 다르다. / Classification head for Phase 2 Supervised Classification. Structure differs per backbone.

### 5.1 EfficientNet-B0 특화 / EfficientNet-B0 Specialized

SE-attention이 backbone 내부에서 채널 선택을 완료하므로 직접 압축 구조를 사용한다. / SE-attention completes channel selection inside backbone — use direct compression.

**구조 / Architecture** (`mid_dim=None`): `Linear(1280, hidden_dim) → BatchNorm → ReLU → Dropout → Linear(hidden_dim, num_levels)`

출력: raw logits (Softmax 없음 / No Softmax applied)

### 5.2 ResNet-50 특화 / ResNet-50 Specialized

2048차원 비선별 features를 단계적으로 압축한다. / Staged compression of unfiltered 2048-dim features.

**구조 / Architecture** (`mid_dim=512`): `Linear(2048, mid_dim) → BatchNorm → ReLU → Dropout → Linear(mid_dim, hidden_dim) → BatchNorm → ReLU → Dropout → Linear(hidden_dim, num_levels)`

출력: raw logits (Softmax 없음 / No Softmax applied)

### 5.3 파라미터 비교 / Parameter Comparison

| 파라미터 / Parameter | config 키 / Key | EfficientNet-B0 | ResNet-50 | SSOT 분류 / Classification |
|---|---|---|---|---|
| `mid_dim` | `phase2.heads.resnet50.mid_dim` 🟢 | — (없음 / absent) | 512 | Hard |
| `hidden_dim` | `phase2.heads.{backbone}.hidden_dim` 🟢 | 256 | 256 | Hard |
| `dropout` | `phase2.heads.{backbone}.dropout` 🟢 | 0.2 | 0.4 | Soft |
| `num_levels` | `data.num_levels` 🟢 | 6 | 6 | Hard |

---

## 6. Phase 전환 규칙 / Phase Transition Rules

### 6.1 switch_to_phase2() 로드 규칙 / Load Rules

- `backbone.*` prefix 키만 선택적 로드 / Only `backbone.*` prefix keys are loaded selectively
- Head 키 무시 (ProjectionHead → ClassifierHead 교체) / Head keys ignored (ProjectionHead replaced by ClassifierHead)
- `strict=False`: 새 head 키 불일치 허용 / Allows new head key mismatches

### 6.2 추론 시 로드 / Inference Load Rules

- `weights_only=True`: pickle 보안 (임의 코드 실행 방지) / Pickle security (prevent arbitrary code execution)
- `strict=False`: 버전 간 호환성 / Cross-version compatibility

---

## 7. Hard SSOT 파라미터 / Hard SSOT Parameters

변경 시 **재학습 필수** / **Retrain required** on change:

| 파라미터 / Parameter | 값 / Value | config 키 / Key | 영향 / Impact |
|---|---|---|---|
| `num_levels` | 6 | `data.num_levels` | ClassifierHead 출력 차원 / Output dim |
| `image_size` | 128 | `data.image_size` | 모델 입력 크기 / Model input size |
| `backbone` | `efficientnet_b0` / `resnet50` | `model.backbone` | 전체 구조 변경 / Entire architecture changed |
| `projection_dim` | 128 | `phase0.projection_dim` | Phase 0 head 구조 / Phase 0 head structure |
| `hidden_dim` (EffB0) | 256 | `phase2.heads.efficientnet_b0.hidden_dim` | Phase 2 head 구조 / Phase 2 head structure |
| `hidden_dim` (Res50) | 256 | `phase2.heads.resnet50.hidden_dim` | Phase 2 head 구조 / Phase 2 head structure |
| `mid_dim` (Res50 전용 / only) | 512 | `phase2.heads.resnet50.mid_dim` | ResNet-50 단계적 압축 구조 / ResNet-50 staged compression structure |
| Phase 순서 / Phase ordering | Phase 0 → 2 | — | backbone 가중치 의존성 / Backbone weight dependency |
| 색상 공간 / Color space | BGR [0, 1] | — | 입력 분포 / Input distribution |

---

## 체크리스트 / Checklist

- [ ] 새 Backbone 추가 시 §3 목록 + `backbone_tag()` 매핑 갱신 / Update §3 list + `backbone_tag()` when adding new backbone
- [ ] Head 구조 변경 시 §4/§5 업데이트 / Update §4/§5 on head structure change
- [ ] Phase 전환 로직 변경 시 §6 동기화 / Sync §6 on phase transition logic change
- [ ] feature_dim 변경 시 Hard SSOT 영향 분석 / Analyze Hard SSOT impact on feature_dim change

---

## See Also

| 문서 / Document | 관계 / Relation |
| --- | --- |
| [SSOT_Core.md](SSOT_Core.md) | Hard/Soft 판단 기준 / Hard/Soft decision criteria |
| [SSOT_Artifacts.md](SSOT_Artifacts.md) | state_dict 스키마 / state_dict schema |
| [SSOT_Training_Pipeline.md](SSOT_Training_Pipeline.md) | 학습 흐름 / Training flow |

