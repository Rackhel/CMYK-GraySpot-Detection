---
type: contract
domain: model_boundary
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] Model Boundary — 모델 경계 계약 / Model Boundary Contract

> **목적 / Purpose**: `GrayspotModel`, `build_backbone()`, Phase 전환의 입출력 계약을 정의한다. / Defines I/O contracts for `GrayspotModel`, `build_backbone()`, and Phase transitions.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-17
> **관련 문서 / Related Docs**:
>
> - [SSOT_Model_Architecture.md](../SSOT/SSOT_Model_Architecture.md) (모델 구조 정의 / Model architecture definition)
> - [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) (state_dict 스키마 / state_dict schema)

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.
> / This document does not redefine SSOT semantic definitions. Follow SSOT_Core.md as the final authority for semantic interpretation.

---

## 1. 계약 목적 / Contract Purpose

`GrayspotModel`의 생성, Phase별 입출력, backbone 팩토리, Phase 전환의 인터페이스 규약을 정의한다.

Defines the interface contracts for `GrayspotModel` creation, per-Phase I/O, backbone factory, and Phase transitions.

---

## 2. `GrayspotModel` 입출력 계약 / GrayspotModel I/O Contract

| Phase | 입력 / Input | 출력 / Output | 비고 / Note |
| --- | --- | --- | --- |
| Phase 0 | `(B, 3, 128, 128)` float32 | `(B, 128)` float32 | L2-정규화된 projection vector / L2-normalized projection vector |
| Phase 2 | `(B, 3, 128, 128)` float32 | `(B, 6)` float32 | Raw logits — Softmax 없음 / No Softmax |

---

## 3. 생성 계약 / Creation Contract

### 3.1 Phase 0 모델 생성 / Phase 0 Model Creation

```python
model = GrayspotModel(cfg, phase=0)
```

**필수 cfg 키 / Required cfg Keys**: `model.backbone`, `phase0.projection_dim`, `phase0.hidden_dim`

### 3.2 Phase 2 모델 생성 / Phase 2 Model Creation

```python
model = GrayspotModel(cfg, phase=2)
```

**필수 cfg 키 / Required cfg Keys**: `model.backbone`, `model.frozen_backbone`, `data.num_levels`

Backbone별 head 키 / Per-backbone head keys:

- `phase2.heads.efficientnet_b0.{hidden_dim, dropout}` → 직접 압축 구조 / Direct compression structure
- `phase2.heads.resnet50.{mid_dim, hidden_dim, dropout}` → 단계적 압축 구조 / Staged compression structure
- 부재 시 / If absent: `phase2.hidden_dim` / `phase2.dropout` fallback

### 3.3 Backbone별 ClassifierHead 구조 / Per-Backbone ClassifierHead Structure

| Backbone | `mid_dim` | 구조 / Structure |
| --- | --- | --- |
| `efficientnet_b0` | `None` | `in_dim(1280) → hidden_dim → num_levels` |
| `resnet50` | `512` | `in_dim(2048) → mid_dim → hidden_dim → num_levels` |

---

## 4. `build_backbone()` 계약 / build_backbone() Contract

```python
from models.backbone import build_backbone

backbone, feature_dim = build_backbone(backbone_name: str)
```

| Backbone | `feature_dim` | Hard SSOT |
| --- | --- | --- |
| `efficientnet_b0` | `1280` | ✅ |
| `resnet50` | `2048` | ✅ |

> `feature_dim`은 torchvision 구조에 의해 고정 — config로 override 불가.
> / `feature_dim` is fixed by torchvision architecture — cannot be overridden by config.

---

## 5. Phase 전환 계약 / Phase Transition Contract

```python
model.switch_to_phase2(backbone_path: Path, cfg: dict) -> None
```

| 항목 / Item | 설명 / Description |
| --- | --- |
| 입력 / Input | `phase0_backbone_{channel}_{tag}.pt`, config dict |
| 동작 / Behavior | `backbone.*` 키만 로드 (`strict=False`) / Load only `backbone.*` keys |
| 결과 / Result | Backbone weights 유지 + ClassifierHead 새로 초기화 / Backbone weights retained + ClassifierHead re-initialized |
| 실패 / Failure | backbone 키 0개 로드 시 `RuntimeError` (SSOT-FF01) / `RuntimeError` if 0 backbone keys loaded |

### Channel Invariant 규칙 / Channel Invariant Rules

```
phase0_backbone_Y_effb0.pt → Phase 2 channel=Y    ✅
phase0_backbone_Y_effb0.pt → Phase 2 channel=M    ❌ (채널 교차 금지 / cross-channel prohibited)
phase0_backbone_Y_res50.pt → Phase 2 backbone=effb0  ❌ (구조 불일치 / architecture mismatch)
```

---

## 6. `build_model()` 유틸리티 계약 / build_model() Utility Contract

```python
from utils.utils_model import build_model

model = build_model(cfg: dict, checkpoint: Path, device: torch.device) -> nn.Module
```

| 항목 / Item | 타입 / Type | 설명 / Description |
| --- | --- | --- |
| `cfg` | `dict` | config dict (Phase 2 head 구성에 사용 / used for Phase 2 head configuration) |
| `checkpoint` | `Path` | `best_{ch}.pt` 절대 경로 / absolute path |
| `device` | `torch.device` | 연산 디바이스 / computation device |
| 반환 / Return | `nn.Module` | `GrayspotModel(phase=2)`, `model.eval()` 상태 / in eval state |

> `strict=False` 로드 — 키 불일치 허용. / `strict=False` load — key mismatch allowed.

---

## 7. 금지 패턴 / Prohibited Patterns

```python
# ❌ Phase 0 backbone을 다른 채널의 Phase 2에 사용
# / Using Phase 0 backbone for a different channel's Phase 2
model.switch_to_phase2("phase0_backbone_Y_effb0.pt")  # channel=M일 때 금지 / prohibited when channel=M

# ❌ Softmax를 모델 출력에 적용한 후 CrossEntropyLoss 전달
# / Applying Softmax to model output before passing to CrossEntropyLoss
loss = F.cross_entropy(F.softmax(logits, dim=1), labels)  # 이중 softmax / double softmax

# ✅ 올바른 패턴 / Correct pattern
loss = F.cross_entropy(logits, labels)  # raw logits 직접 전달 / pass raw logits directly
```

---

## 8. 체크리스트 / Checklist

- [x] Phase 0 → Phase 2 전환 시 `backbone.*` 키만 로드 / Load only `backbone.*` keys during Phase 0 → Phase 2 transition
- [x] `strict=False` 로드 확인 / Verify `strict=False` load
- [x] Channel Invariant 위반 시 `SSOT-FF01` 발생 / `SSOT-FF01` raised on Channel Invariant violation
- [ ] 새 Backbone 추가 시 §4 + ClassifierHead 구조 갱신 / Update §4 + ClassifierHead structure when adding new backbone

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Model_Architecture.md](../SSOT/SSOT_Model_Architecture.md) | 모델 구조 정의 (What) / Model architecture definition |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | 체크포인트 저장/로드 계약 / Checkpoint save/load contract |
| [Contract_training_pipeline.md](Contract_training_pipeline.md) | Trainer → Model 호출 / Trainer → Model calls |
