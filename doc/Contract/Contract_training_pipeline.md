---
type: contract
domain: training_pipeline
status: Active
last_updated: 2026-06-01
owner: CMYK WooSong Team
---

# [Contract] Training Pipeline — 학습 파이프라인 경계 계약

> **목적 / Purpose**: Phase0/Phase2 Trainer의 입출력, 손실 함수 계약, Optimizer/Scheduler 팩토리를 정의한다.
> **상태 / Status**: ✅ Accepted [Hard]
> **관련 문서**:
> - [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md)
> - [SSOT_Model_Architecture.md](../SSOT/SSOT_Model_Architecture.md)

> 🔒 **SSOT 경계 원칙**: 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.

---

## 변경 이력 / Changelog

| 날짜 | 항목 | 내용 |
|---|---|---|
| 2026-06-01 | C-2 | Phase2Trainer per-channel config 오버라이드, frozen_backbone 자동 적용 추가 |
| 2026-06-01 | losses | FocalLoss 추가, label_smoothing, class_weights 지원 확장 |
| 2026-06-01 | D-3 | GradScaler API `torch.cuda.amp` → `torch.amp` 업데이트 |
| 2026-06-01 | GUI | Training tab 누락 파라미터 추가 (warmup_epochs, loss type, optimizer 등) |

---

## 1. 계약 목적 / Contract Purpose

`Phase0Trainer`, `Phase2Trainer`, 손실 함수(`InfoNCELoss`, `FocalLoss`, `get_loss`)의 인터페이스 규약을 정의한다.

---

## 2. `Phase0Trainer` 계약

```python
trainer = Phase0Trainer(model, cfg, channel="Y", device=device)
history = trainer.train(loader)    # → List[dict]
trainer.save_backbone()            # → Path
```

### 입출력 / I/O

| 경계 | 입력 타입 | 출력 타입 |
|---|---|---|
| `train()` 입력 | `DataLoader` — `(B,3,128,128), (B,3,128,128)` | `List[{"epoch","loss","lr","elapsed"}]` |
| `save_backbone()` 출력 | — | `{models_dir}/phase0_backbone_{ch}_{tag}.pt` |

### 필수 cfg 키

```
phase0.epochs, phase0.batch_size, phase0.learning_rate,
phase0.weight_decay, phase0.temperature, phase0.warmup_epochs,
train.optimizer, train.scheduler, train.gradient_clip,
train.mixed_precision, train.grad_accumulation_steps,
train.eta_min, train.seed, storage.models_dir
```

---

## 3. `Phase2Trainer` 계약

```python
trainer = Phase2Trainer(model, cfg, channel="Y", device=device, train_ds=train_ds)
history = trainer.train(train_loader, val_loader)  # → List[dict]
```

### 3.1 Per-Channel Override 자동 적용

`Phase2Trainer.__init__`에서 `cfg["phase2"]["per_channel"][channel]`이 존재하면 자동으로 cfg에 병합됩니다:

```python
# config.json 예시
"phase2": {
  "epochs": 50,
  "per_channel": {
    "K": { "frozen_backbone": true, "dropout": 0.5, "epochs": 10, "patience": 3 },
    "Y": { "frozen_backbone": true, "dropout": 0.5, "epochs": 15, "patience": 5 }
  }
}
```

- `frozen_backbone=true`이면 `model.backbone.parameters()`의 `requires_grad=False` 자동 설정
- 오버라이드된 키는 로그에 `[per-channel] Applying overrides for [K]: {...}` 형태로 출력

### 3.2 입출력 / I/O

| 경계 | 입력 타입 | 출력 타입 |
|---|---|---|
| `train()` 입력 | `DataLoader` × 2 — `(B,3,128,128),(B,)` | `List[{"epoch","train_loss","val_acc",...}]` |
| 체크포인트 | 내부 자동 저장 | `{models_dir}/best_{ch}.pt` |

### 3.3 저장 기준 / Save Criterion

> ⚠️ **현재**: `val_acc > best_val_acc + early_stopping.min_delta`
> ⚠️ **PRD 목표**: `macro_f1` — 미정렬 상태

### 3.4 필수 cfg 키

```
phase2.epochs, phase2.batch_size, phase2.learning_rate,
phase2.weight_decay, phase2.warmup_epochs,
phase2.loss, phase2.class_weights, phase2.label_smoothing, phase2.focal_gamma,
phase2.early_stopping.{enabled,patience,min_delta},
phase2.per_channel.{ch}.{*} (선택적 오버라이드),
train.optimizer, train.scheduler, train.gradient_clip,
train.mixed_precision, train.grad_accumulation_steps,
storage.models_dir, storage.reports_dir
```

---

## 4. 손실 함수 계약 / Loss Function Contract

### 4.1 `InfoNCELoss` (Phase 0)

```python
from training import InfoNCELoss

loss_fn = InfoNCELoss(temperature=cfg["phase0"]["temperature"])
loss = loss_fn(z1, z2)
```

| 입력 | 타입 | 제약 |
|---|---|---|
| `z1` | `Tensor (B, projection_dim)` | **반드시 L2-정규화** |
| `z2` | `Tensor (B, projection_dim)` | **반드시 L2-정규화** |
| 반환 | scalar Tensor | contrastive loss |

### 4.2 `get_loss()` (Phase 2)

```python
from training import get_loss

loss_fn = get_loss(phase=2, cfg=cfg, train_samples=train_ds.samples)
loss = loss_fn(logits, labels)
```

`phase2.loss` 값에 따른 반환:

| `phase2.loss` | 반환 | 파라미터 |
|---|---|---|
| `"cross_entropy"` (기본) | `nn.CrossEntropyLoss` | `weight`, `label_smoothing` |
| `"focal"` | `FocalLoss` | `gamma=phase2.focal_gamma`, `weight` |

공통 입력 계약:

| 입력 | 타입 | 제약 |
|---|---|---|
| `logits` | `Tensor (B, 6)` | raw logits — **Softmax 미적용** |
| `labels` | `Tensor (B,)` | int `[0, 5]` |
| 반환 | scalar Tensor | |

`phase2.class_weights="balanced"` 시 `train_samples`로 클래스 가중치 자동 계산.

### 4.3 `FocalLoss`

```python
from training.losses import FocalLoss

loss_fn = FocalLoss(gamma=2.0, weight=class_weights, reduction="mean")
```

- `gamma > 0`: 쉬운 샘플(정상 클래스) 가중치↓, 어려운 샘플(희귀 결함) 가중치↑
- K/Y 채널 등 극심한 불균형 시 `loss="focal"`, `class_weights="balanced"` 조합 권장

---

## 5. Optimizer / Scheduler 팩토리 계약

### 5.1 `_build_optimizer()`

| config 값 | Optimizer | 파라미터 |
|---|---|---|
| `"adamw"` | `AdamW` | lr, weight_decay, betas `[0.9, 0.999]` |
| `"sgd"` | `SGD` | lr, weight_decay, momentum `0.9` |

### 5.2 `_build_scheduler()`

| config 값 | Scheduler | 파라미터 |
|---|---|---|
| `"cosine"` | `CosineAnnealingLR` | T_max, eta_min |
| `"step"` | `StepLR` | step_size, gamma |

`warmup_epochs > 0`이면 `LinearLR + base_scheduler`를 `SequentialLR`로 자동 조합.

### 5.3 GradScaler (D-3 수정)

```python
# ❌ Deprecated (PyTorch 2.1+)
scaler = torch.cuda.amp.GradScaler()

# ✅ Current
scaler = torch.amp.GradScaler("cuda")
```

---

## 6. 금지 패턴 / Prohibited Patterns

```python
# ❌ L2-정규화 없이 InfoNCELoss에 전달
loss = loss_fn(backbone_output, backbone_output2)

# ❌ Softmax 적용 후 CrossEntropyLoss 전달
loss = loss_fn(F.softmax(logits, dim=1), labels)

# ✅ 올바른 패턴
z1 = F.normalize(projection_head(backbone(x1)), dim=1)
loss = info_nce(z1, z2)
```

```python
# ❌ per_channel 오버라이드를 Phase2Trainer 외부에서 수동 적용
cfg["phase2"]["epochs"] = 10  # K 채널 전용으로 전역 수정 — 다른 채널에 영향

# ✅ config.json의 per_channel에 선언 → Phase2Trainer가 자동 처리
# config.json: "per_channel": { "K": { "epochs": 10 } }
```

---

## 7. 체크리스트 / Checklist

- [x] `InfoNCELoss` 입력 L2-정규화 확인
- [x] `get_loss` logits → raw logits (Softmax 미적용)
- [x] EarlyStopping patience, min_delta config 연결
- [x] Gradient clipping 적용
- [x] GradScaler API torch.amp 업데이트 (D-3)
- [x] Phase2Trainer per-channel override 자동 병합 (C-2)
- [x] frozen_backbone per-channel 적용 (C-2)
- [x] FocalLoss 추가, get_loss 팩토리 확장
- [x] warmup_epochs Phase 0/2 모두 config에서 소비
- [ ] Best 저장 기준 val_acc → macro_f1 전환 (미완료)

---

## See Also

| 문서 | 관계 |
|---|---|
| [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) | 학습 파이프라인 정의 |
| [Contract_model_boundary.md](Contract_model_boundary.md) | 모델 입출력 계약 |
| [Contract_data_pipeline.md](Contract_data_pipeline.md) | DataLoader 배치 계약 |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | 체크포인트 저장 계약 |
