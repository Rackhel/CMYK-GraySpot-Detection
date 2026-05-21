---
type: contract
domain: training_pipeline
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] Training Pipeline — 학습 파이프라인 경계 계약 / Training Pipeline Boundary Contract

> **목적 / Purpose**: Phase0/Phase2 Trainer의 입출력, 손실 함수 계약, Optimizer/Scheduler 팩토리를 정의한다. / Defines I/O contracts for Phase0/Phase2 Trainers, loss functions, and Optimizer/Scheduler factories.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-17
> **관련 문서 / Related Docs**:
>
> - [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) (학습 파이프라인 정의 / Training pipeline definition)
> - [SSOT_Model_Architecture.md](../SSOT/SSOT_Model_Architecture.md) (모델 구조 / Model architecture)

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.
> / This document does not redefine SSOT semantic definitions. Follow SSOT_Core.md as the final authority.

---

## 1. 계약 목적 / Contract Purpose

`Phase0Trainer`, `Phase2Trainer`, 손실 함수(`InfoNCELoss`, `get_loss`)의 인터페이스 규약을 정의한다.

Defines the interface contracts for `Phase0Trainer`, `Phase2Trainer`, and loss functions (`InfoNCELoss`, `get_loss`).

---

## 2. `Phase0Trainer` 계약 / Phase0Trainer Contract

```python
trainer = Phase0Trainer(model, cfg, channel="Y", device=device)
history = trainer.train(loader)    # → List[dict]
trainer.save_backbone()            # → Path
```

### 입출력 / I/O

| 경계 / Boundary | 입력 타입 / Input Type | 출력 타입 / Output Type |
| --- | --- | --- |
| `train()` 입력 | `DataLoader` — `(B,3,128,128), (B,3,128,128)` | `List[{"epoch", "loss"}]` |
| `save_backbone()` 출력 | — | `{models_dir}/phase0_backbone_{ch}_{tag}.pt` |

### 필수 cfg 키 / Required cfg Keys

```
phase0.epochs, phase0.batch_size, phase0.learning_rate,
phase0.weight_decay, phase0.temperature,
train.optimizer, train.scheduler, train.gradient_clip,
train.eta_min, train.seed, storage.models_dir
```

---

## 3. `Phase2Trainer` 계약 / Phase2Trainer Contract

```python
trainer = Phase2Trainer(model, cfg, channel="Y", device=device, train_ds=train_ds)
history = trainer.train(train_loader, val_loader)  # → List[dict]
trainer.save_history(history)                       # → Path
```

### 입출력 / I/O

| 경계 / Boundary | 입력 타입 / Input Type | 출력 타입 / Output Type |
| --- | --- | --- |
| `train()` 입력 | `DataLoader` × 2 — `(B,3,128,128), (B,)` | `List[{"epoch", "train_loss", "val_acc", ...}]` |
| 체크포인트 / Checkpoint | 내부 자동 저장 / Saved internally | `{models_dir}/best_{ch}.pt` |
| `save_history()` 출력 | `List[dict]` | `{reports_dir}/phase2_history_{ch}.csv` |

### 저장 기준 / Save Criterion

> ⚠️ **현재 / Current**: `val_acc > best_val_acc + early_stopping.min_delta`
> ⚠️ **PRD 목표 / Target**: `macro_f1` — 미정렬 상태 / Not yet aligned

### 필수 cfg 키 / Required cfg Keys

```
phase2.epochs, phase2.batch_size, phase2.learning_rate,
phase2.weight_decay, phase2.early_stopping.*,
train.optimizer, train.scheduler, train.gradient_clip,
train.eta_min, train.seed, storage.models_dir, storage.reports_dir
```

---

## 4. 손실 함수 계약 / Loss Function Contract

### 4.1 `InfoNCELoss` (Phase 0)

```python
from training import InfoNCELoss

loss_fn = InfoNCELoss(temperature=cfg["phase0"]["temperature"])
loss = loss_fn(z1, z2)
```

| 입력 / Input | 타입 / Type | 제약 / Constraint |
| --- | --- | --- |
| `z1` | `Tensor (B, projection_dim)` | **반드시 L2-정규화 후 전달 / Must be L2-normalized** |
| `z2` | `Tensor (B, projection_dim)` | **반드시 L2-정규화 후 전달 / Must be L2-normalized** |
| 반환 / Return | `scalar Tensor` | contrastive loss |

### 4.2 `get_loss()` (Phase 2)

```python
from training import get_loss

loss_fn = get_loss(phase=2, cfg=cfg, train_samples=train_ds.samples)
loss = loss_fn(logits, labels)
```

| 입력 / Input | 타입 / Type | 제약 / Constraint |
| --- | --- | --- |
| `logits` | `Tensor (B, 6)` | raw logits — **Softmax 미적용 / Softmax NOT applied** |
| `labels` | `Tensor (B,)` | int `[0, 5]` |
| 반환 / Return | `scalar Tensor` | CrossEntropyLoss (class_weights 적용 / applied) |

---

## 5. Optimizer / Scheduler 팩토리 계약 / Factory Contract

### 5.1 `_build_optimizer()`

| config 값 / Value | Optimizer | 파라미터 / Parameters |
| --- | --- | --- |
| `"adamw"` | `AdamW` | lr, weight_decay, betas `[0.9, 0.999]` |
| `"sgd"` | `SGD` | lr, weight_decay, momentum `0.9` |

### 5.2 `_build_scheduler()`

| config 값 / Value | Scheduler | 파라미터 / Parameters |
| --- | --- | --- |
| `"cosine"` | `CosineAnnealingLR` | T_max=epochs, eta_min |
| `"step"` | `StepLR` | step_size, gamma |

---

## 6. 금지 패턴 / Prohibited Patterns

```python
# ❌ L2-정규화 없이 InfoNCELoss에 전달 / InfoNCELoss called without L2-normalization
loss = loss_fn(backbone_output, backbone_output2)  # 정규화 안 됨 / not normalized

# ❌ Softmax 적용 후 CrossEntropyLoss 전달 / CrossEntropyLoss with Softmax pre-applied
loss = loss_fn(F.softmax(logits, dim=1), labels)  # 이중 softmax / double softmax

# ✅ 올바른 패턴 / Correct pattern
z1 = F.normalize(projection_head(backbone(x1)), dim=1)
loss = info_nce(z1, z2)
```

---

## 7. 체크리스트 / Checklist

- [x] `InfoNCELoss` 입력 L2-정규화 확인 / Verify L2-normalization on InfoNCELoss input
- [x] `get_loss` logits → raw logits (Softmax 미적용 / not applied)
- [x] EarlyStopping patience config 연결 / EarlyStopping patience connected to config
- [x] Gradient clipping 적용 / Gradient clipping applied
- [ ] Best 저장 기준 val_acc → macro_f1 전환 / Switch best checkpoint criterion from val_acc to macro_f1

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) | 학습 파이프라인 정의 (What) / Training pipeline definition |
| [Contract_model_boundary.md](Contract_model_boundary.md) | 모델 입출력 계약 / Model I/O contract |
| [Contract_data_pipeline.md](Contract_data_pipeline.md) | DataLoader 배치 계약 / DataLoader batch contract |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | 체크포인트 저장 계약 / Checkpoint save contract |
