# SSOT Training Pipeline — 학습 파이프라인 / Training Pipeline

CMYK Grayspot Detection System 의 학습 루프, optimizer, loss 함수에 관한 단일 진실 공급원.

This document is the authoritative reference for training loops, optimizers, and loss functions.

> **목적 / Purpose**: 학습 파이프라인의 의미(semantic) 정의
> **역할 / Role**: "What" — Phase 순서, 실제 학습 파라미터, 손실 함수 규약
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Model_Architecture.md](SSOT_Model_Architecture.md), [SSOT_Artifacts.md](SSOT_Artifacts.md)

---

## Table of Contents / 목차

1. [Swing Architecture 개요 / Swing Architecture Overview](#1-swing-architecture-개요--swing-architecture-overview)
2. [Phase 0 — SimCLR Contrastive Learning](#2-phase-0--simclr-contrastive-learning)
3. [Phase 2 — Supervised Classification](#3-phase-2--supervised-classification)
4. [손실 함수 / Loss Functions](#4-손실-함수--loss-functions)
5. [Optimizer / Scheduler](#5-optimizer--scheduler)
6. [학습 파라미터 요약 / Parameter Summary](#6-학습-파라미터-요약--parameter-summary)
7. [실행 명령어 / Run Commands](#7-실행-명령어--run-commands)
8. [멀티 채널 실행 / Multi-Channel Execution](#8-멀티-채널-실행--multi-channel-execution)
9. [SSOT 위반 현황 / Violations](#9-ssot-위반-현황--violations)

---

## 1. Swing Architecture 개요 / Swing Architecture Overview

```
[Raw Images]
     │
     ▼
Phase 0 ─ SimCLR Contrastive Learning
     │     ContrastiveDataset → GrayspotModel(phase=0)
     │     InfoNCELoss → Phase0Trainer → phase0_backbone_{ch}_{tag}.pt
     │
     ▼
Phase 1 ─ Label Refinement (R1 담당 / R1 responsibility)
     │     labels_v0.csv → labels_v1.csv
     │
     ▼
Phase 2 ─ Supervised Classification
     │     CMYKDataset → GrayspotModel.switch_to_phase2()
     │     CrossEntropyLoss → Phase2Trainer → best_{ch}.pt
     │
     ▼
Phase 3 ─ Evaluation & Feedback
          Evaluator.run() → compute_all_channels()
          → determine_swing_feedback() → Terminate or retry Phase 0
```

### 1.1 Phase 순서 제약 / Phase Ordering Constraint

> **SSOT-PH01**: Phase 0 학습 없이 Phase 2를 시작하면 즉시 실패.
> **SSOT-PH01**: Starting Phase 2 without a trained Phase 0 backbone **must fail immediately**.

```python
# Phase 2 시작 전 반드시 확인 / Must check before Phase 2
backbone_path = model_dir / f"phase0_backbone_{channel}_{tag}.pt"
if not backbone_path.exists():
    raise FileNotFoundError(f"[SSOT-FF01] {backbone_path}")
```

---

## 2. Phase 0 — SimCLR Contrastive Learning

### 2.1 학습기 / Trainer

```python
from training import Phase0Trainer

trainer = Phase0Trainer(model, cfg, channel="Y", device=device)
history = trainer.train(loader)
trainer.save_backbone()
# 출력 / Output: data_set/models/phase0_backbone_{channel}_{tag}.pt
```

### 2.2 실제 학습 파라미터 / Actual Training Parameters

| 파라미터 / Parameter | config 키 / Key | 기본값 / Default | 소비 여부 / Consumed |
|---|---|---|---|
| epochs | `phase0.epochs` | 10 | 🟢 |
| batch_size | `phase0.batch_size` | 16 | 🟢 |
| learning_rate | `phase0.learning_rate` | 1e-3 | 🟢 |
| weight_decay | `phase0.weight_decay` | 1e-5 | 🟢 |
| temperature (τ) | `phase0.temperature` | 0.1 | 🟢 (Hard SSOT) |
| projection_dim | `phase0.projection_dim` | 128 | 🟢 (Hard SSOT) |
| proj_hidden_dim | `phase0.hidden_dim` | 256 | 🟢 (Hard SSOT — ProjectionHead 중간 차원 / intermediate dimension) |
| optimizer | `train.optimizer` | `"adamw"` | 🟢 (`_build_optimizer()`) |
| scheduler | `train.scheduler` | `"cosine"` | 🟢 (`_build_scheduler()`) |
| gradient_clip | `train.gradient_clip` | 1.0 | 🟢 |
| color_jitter | `phase0.augmentation.color_jitter` | 0.4 | 🟢 (augment_contrastive) |
| blur_prob | `phase0.augmentation.blur_prob` | 0.5 | 🟢 (augment_contrastive) |

### 2.3 체크포인트 저장 / Checkpoint Save

```python
Phase0Trainer.save_backbone()
# → data_set/models/phase0_backbone_{channel}_{tag}.pt
# 전체 model.state_dict() 저장 (backbone + ProjectionHead) / Save full model.state_dict() (backbone + ProjectionHead)
```

---

## 3. Phase 2 — Supervised Classification

### 3.1 학습기 / Trainer

```python
from training import Phase2Trainer

trainer = Phase2Trainer(model, cfg, channel="Y", device=device, train_ds=train_ds)
history = trainer.train(train_loader, val_loader)
trainer.save_history(history)
# 출력 / Output: data_set/models/best_{channel}.pt
#               data_set/reports/phase2_history_{channel}.csv
```

### 3.2 실제 학습 파라미터 / Actual Training Parameters

| 파라미터 / Parameter | config 키 / Key | 기본값 / Default | 소비 여부 / Consumed |
|---|---|---|---|
| epochs | `phase2.epochs` | 30 | 🟢 |
| batch_size | `phase2.batch_size` | 16 | 🟢 |
| learning_rate | `phase2.learning_rate` | 1e-4 | 🟢 |
| weight_decay | `phase2.weight_decay` | 1e-4 | 🟢 |
| hidden_dim | `phase2.hidden_dim` | 256 | 🟢 (Hard SSOT — ClassifierHead 중간 차원 / intermediate dimension) |
| dropout | `phase2.dropout` | 0.3 | 🟢 |
| oversample | `phase2.oversample` | true | 🟢 |
| early_stopping.enabled | `phase2.early_stopping.enabled` | true | 🟢 |
| early_stopping.patience | `phase2.early_stopping.patience` | 5 | 🟢 |
| early_stopping.min_delta | `phase2.early_stopping.min_delta` | 1e-4 | 🟢 |
| optimizer | `train.optimizer` | `"adamw"` | 🟢 (`_build_optimizer()`) |
| scheduler | `train.scheduler` | `"cosine"` | 🟢 (`_build_scheduler()`) |
| gradient_clip | `train.gradient_clip` | 1.0 | 🟢 |

### 3.3 Best Model 기준 / Best Model Criteria

```python
# val_acc 기준 best 모델 저장 (early stopping 포함) / Save best model by val_acc (including early stopping)
if val_acc > best_val_acc + es_delta:
    best_val_acc = val_acc
    torch.save(model.state_dict(), model_dir / f"best_{channel}.pt")
```

> ⚠️ 저장 기준이 `val_acc`(accuracy)이며, PRD 목표 지표인 `macro_f1`과 다름. Optuna도 `best_val_acc`를 목적 함수로 사용한다.
> ⚠️ Best model is saved by `val_acc` (accuracy), which differs from the PRD target metric `macro_f1`. Optuna also uses `best_val_acc` as its objective function.

### 3.4 체크포인트 흐름 / Checkpoint Flow

```
Phase2Trainer.train()
    → data_set/models/best_{ch}.pt           (best val_acc 기준 / by best val_acc)
    → data_set/reports/phase2_history_{ch}.csv  (에폭별 이력 CSV / per-epoch history CSV)
```

---

## 4. 손실 함수 / Loss Functions

### 4.1 InfoNCELoss — Phase 0

```python
from training import InfoNCELoss

loss_fn = InfoNCELoss(temperature=cfg["phase0"]["temperature"])
loss = loss_fn(z1, z2)
# z1, z2: (B, projection_dim) L2-normalized vectors
# Returns: scalar contrastive loss
```

| 파라미터 / Parameter | config 키 / Key | 값 / Value | Hard SSOT |
|---|---|---|---|
| temperature (τ) | `phase0.temperature` 🟢 | 0.1 | ✅ |

### 4.2 CrossEntropyLoss — Phase 2

```python
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
loss = loss_fn(logits, labels)
# logits: (B, 6) raw logits
# labels: (B,) int [0, 5]
# class_weights: 학습 데이터 분포 기반 자동 계산 / Auto-computed from training distribution
```

### 4.3 get_loss() 팩토리 / Factory

```python
from training import get_loss

loss_fn = get_loss(cfg, phase=0)                      # InfoNCELoss
loss_fn = get_loss(cfg, phase=2, train_samples=...)   # CrossEntropyLoss (balanced weights)
```

---

## 5. Optimizer / Scheduler

`training/trainer.py`의 `_build_optimizer()` / `_build_scheduler()` factory를 통해 config에서 동적 선택.
Dynamically selected from config via `_build_optimizer()` / `_build_scheduler()` in `training/trainer.py`.

### 5.1 Optimizer Factory

```python
def _build_optimizer(model, lr, weight_decay, cfg):
    name = cfg["train"].get("optimizer", "adamw").lower()
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                   momentum=cfg["train"].get("momentum", 0.9))
    else:  # adamw (default)
        betas = tuple(cfg["train"].get("betas", [0.9, 0.999]))
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
```

| 컴포넌트 / Component | config 키 / Key | 기본값 / Default | 소비 여부 / Consumed |
|---|---|---|---|
| optimizer 종류 / Optimizer type | `train.optimizer` | `"adamw"` | 🟢 |
| AdamW betas | `train.betas` | `[0.9, 0.999]` | 🟢 (adamw 선택 시 / when adamw selected) |
| SGD momentum | `train.momentum` | `0.9` | 🟢 (sgd 선택 시 / when sgd selected) |

### 5.2 Scheduler Factory

```python
def _build_scheduler(optimizer, epochs, cfg):
    name = cfg["train"].get("scheduler", "cosine").lower()
    if name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3),
                      gamma=cfg["train"].get("gamma", 0.1))
    else:  # cosine (default)
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["train"]["eta_min"])
```

| 컴포넌트 / Component | config 키 / Key | 기본값 / Default | 소비 여부 / Consumed |
|---|---|---|---|
| scheduler 종류 / Scheduler type | `train.scheduler` | `"cosine"` | 🟢 |
| CosineAnnealingLR eta_min | `train.eta_min` | 1e-6 | 🟢 |
| StepLR gamma | `train.gamma` | 0.1 | 🟢 (step 선택 시 / when step selected) |
| Gradient clipping | `train.gradient_clip` | 1.0 | 🟢 |

---

## 6. 학습 파라미터 요약 / Parameter Summary

### 6.1 공통 학습 설정 / Common Train Settings

| 파라미터 / Parameter | config 키 / Key | 기본값 / Default | 소비 여부 / Consumed |
|---|---|---|---|
| seed | `train.seed` 🟢 | 42 | 🟢 |
| num_workers | `train.num_workers` 🟢 | 4 | 🟢 |
| pin_memory | `train.pin_memory` 🟢 | true | 🟢 |
| prefetch_factor | `train.prefetch_factor` 🟢 | 2 | 🟢 |
| persistent_workers | `train.persistent_workers` 🟢 | true | 🟢 |
| drop_last | `train.drop_last` 🟢 | false | 🟢 |
| device | `system.device` 🟢 | `"auto"` | 🟢 |
| cuda.deterministic | `cuda.deterministic` 🟢 | true | 🟢 (`set_seed()`) |
| cuda.benchmark | `cuda.benchmark` 🟢 | true | 🟢 (`set_seed()`) |

### 6.2 Snapshot 통합 / Snapshot Integration

학습 시작 시 설정 스냅샷 저장 / Config snapshot saved at training start:

```python
from utils import log_snapshot
from pathlib import Path

snap_path = log_snapshot(
    config=cfg,
    snapshot_dir=Path("outputs/snapshots"),
    tag=f"{channel}_phase0",
)
# → outputs/snapshots/config_snapshot_{channel}_phase0_{timestamp}.json
```

---

## 7. 실행 명령어 / Run Commands

```bash
# Phase 0 — SimCLR 사전 학습 / Contrastive pretraining
python -m src.scripts.run_phase0

# Phase 2 — Supervised 학습 / Supervised classification
python -m src.scripts.run_phase2

# Baseline (Phase 0 없이 비교용 / Without Phase 0, for comparison)
python -m src.scripts.run_baseline

# Optuna HPO 튜닝 / Hyperparameter optimization
python -m src.scripts.run_optuna

# 통합 학습 진입점 / Unified training entry point
python -m src.scripts.train
```

---

## 8. 멀티 채널 실행 / Multi-Channel Execution

4개 CMYK 채널은 **독립적으로** 학습된다. 모델을 공유하지 않으며, 채널별로 Phase 0 → Phase 2 전체 파이프라인이 개별 실행된다.
The 4 CMYK channels are trained **independently**. No model is shared — each channel runs through the full Phase 0 → Phase 2 pipeline separately.

```python
# scripts/run_phase2.py 기준 / Based on scripts/run_phase2.py
channels = cfg["data"]["channels"]   # ["Y", "M", "C", "K"]
for channel in channels:
    run_phase2(channel, cfg)
# 결과: 채널별 독립 모델 4개 생성 / Result: 4 independent models, one per channel
# outputs/checkpoints/best_Y.pt
# outputs/checkpoints/best_M.pt
# outputs/checkpoints/best_C.pt
# outputs/checkpoints/best_K.pt
```

| 채널 / Channel | Phase 0 backbone | Phase 2 모델 / Model | 독립성 / Independence |
|---|---|---|---|
| Y | `phase0_backbone_Y_{tag}.pt` | `best_Y.pt` | ✅ 타 채널과 완전 독립 / Fully independent |
| M | `phase0_backbone_M_{tag}.pt` | `best_M.pt` | ✅ |
| C | `phase0_backbone_C_{tag}.pt` | `best_C.pt` | ✅ |
| K | `phase0_backbone_K_{tag}.pt` | `best_K.pt` | ✅ |

> **Channel Invariant**: 동일 channel의 Phase 0 backbone만 해당 channel의 Phase 2에 사용할 수 있다. (SSOT-FF01)
> Only the Phase 0 backbone of the **same** channel may be used for that channel's Phase 2 training.

---

## 9. SSOT 위반 현황 / Violations

| 코드 / Code | 위반 내용 / Violation | 등급 / Level | 해결 방법 / Fix |
|---|---|---|---|
| SSOT-PH01 | Phase 0 없이 Phase 2 직행 시 FF01 미발생 위험 / Risk of missing FF01 when skipping Phase 0 and going directly to Phase 2 | Level 1 | `backbone_path.exists()` 검증 구현 완료 ✅ |

> ✅ **해소됨 / Resolved**: `train.optimizer`, `train.scheduler` → `_build_optimizer()` / `_build_scheduler()` factory 구현 완료.
> ✅ **해소됨 / Resolved**: `train.gradient_clip` → Phase0/2Trainer `clip_grad_norm_` 적용 완료.
> ✅ **해소됨 / Resolved**: `phase2.early_stopping.*` → `Phase2Trainer.train()` 구현 완료.
> ✅ **해소됨 / Resolved**: `cuda.deterministic` / `cuda.benchmark` → `set_seed()` 소비 완료.
> ✅ **해소됨 / Resolved**: `phase0.weight_decay` → Phase0Trainer AdamW 소비 완료.
> ✅ **해소됨 / Resolved**: `phase0.hidden_dim` → `ProjectionHead` hidden_dim 소비 완료.

---

**Version**: 0.3.0
**Last Updated**: 2026-05-11
**Applies to**: CMYK Grayspot Detection System v0.1.0+
