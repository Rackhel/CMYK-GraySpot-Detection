---
type: ssot
domain: training_pipeline
status: Active
last_updated: 2026-06-01
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
  - "SSOT_Model_Architecture.md"
  - "SSOT_Data_Pipeline.md"
---

# SSOT Training Pipeline — 학습 파이프라인 / Training Pipeline

CMYK Grayspot Detection System 의 학습 루프, optimizer, loss 함수에 관한 단일 진실 공급원.

This document is the authoritative reference for training loops, optimizers, and loss functions.

> **목적 / Purpose**: 학습 파이프라인의 의미(semantic) 정의
> **역할 / Role**: "What" — Phase 순서, 실제 학습 파라미터, 손실 함수 규약
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_Model_Architecture.md](SSOT_Model_Architecture.md), [SSOT_Artifacts.md](SSOT_Artifacts.md)

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

Phase 2 시작 전 `{models_dir}/phase0_backbone_{channel}_{tag}.pt` 존재 여부를 반드시 검증한다. 미존재 시 `SSOT-FF01` 오류를 발생시킨다.

---

## 2. Phase 0 — SimCLR Contrastive Learning

### 2.1 학습기 / Trainer

`training.Phase0Trainer(model, cfg, channel, device)` → `.train(loader)` → `.save_backbone()`

- 출력 / Output: `{models_dir}/phase0_backbone_{channel}_{tag}.pt`

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

`Phase0Trainer.save_backbone()` → `{models_dir}/phase0_backbone_{channel}_{tag}.pt`

전체 `model.state_dict()` 저장 (backbone + ProjectionHead 포함) / Saves full `model.state_dict()` (backbone + ProjectionHead included).

---

## 3. Phase 2 — Supervised Classification

### 3.1 학습기 / Trainer

`training.Phase2Trainer(model, cfg, channel, device, train_ds)` → `.train(train_loader, val_loader)` → `.save_history(history)`

- 출력 / Output: `{models_dir}/best_{channel}.pt`, `{reports_dir}/phase2_history_{channel}.csv`

### 3.2 실제 학습 파라미터 / Actual Training Parameters

| 파라미터 / Parameter | config 키 / Key | 기본값 / Default | 소비 여부 / Consumed |
|---|---|---|---|
| epochs | `phase2.epochs` | 50 | 🟢 |
| batch_size | `phase2.batch_size` | 16 | 🟢 |
| learning_rate | `phase2.learning_rate` | 1e-4 | 🟢 |
| weight_decay | `phase2.weight_decay` | 1e-4 | 🟢 |
| warmup_epochs | `phase2.warmup_epochs` | 3 | 🟢 |
| loss type | `phase2.loss` | `"cross_entropy"` | 🟢 (`"cross_entropy"` \| `"focal"`) |
| class_weights | `phase2.class_weights` | `"none"` | 🟢 (`"none"` \| `"balanced"`) |
| label_smoothing | `phase2.label_smoothing` | 0.0 | 🟢 (CE 전용) |
| focal_gamma | `phase2.focal_gamma` | 2.0 | 🟢 (focal 전용) |
| oversample | `phase2.oversample` | true | 🟢 |
| early_stopping.enabled | `phase2.early_stopping.enabled` | true | 🟢 |
| early_stopping.patience | `phase2.early_stopping.patience` | 10 | 🟢 |
| early_stopping.min_delta | `phase2.early_stopping.min_delta` | 1e-4 | 🟢 |
| optimizer | `train.optimizer` | `"adamw"` | 🟢 |
| scheduler | `train.scheduler` | `"cosine"` | 🟢 |
| gradient_clip | `train.gradient_clip` | 1.0 | 🟢 |
| mixed_precision | `train.mixed_precision` | false | 🟢 |
| grad_accumulation_steps | `train.grad_accumulation_steps` | 1 | 🟢 |

### 3.2-a 채널별 파라미터 오버라이드 / Per-Channel Parameter Override

`phase2.per_channel.{ch}` 키가 존재하면 해당 채널 학습 시 전역 설정을 덮어씁니다.
`Phase2Trainer.__init__`에서 자동으로 병합됩니다.

| 채널 | frozen_backbone | dropout | epochs | patience | policy |
|---|---|---|---|---|---|
| K | `true` | 0.5 | 10 | 3 | `"strong"` |
| Y | `true` | 0.5 | 15 | 5 | `"strong"` |
| C | `false` | 0.3 | 30 | 7 | `"light"` |
| M | `false` | 0.2 | 50 | 10 | `"light"` |

> `frozen_backbone=true`이면 `Phase2Trainer.__init__`에서 backbone 파라미터를 자동 freeze합니다.

### 3.3 Backbone별 ClassifierHead 파라미터 / Backbone-Specific ClassifierHead Parameters

Head 구조와 정규화 강도가 backbone에 따라 다르다. `phase2.heads.{backbone}` 에서 읽는다.
Head structure and regularization strength differ per backbone. Read from `phase2.heads.{backbone}`.

| 파라미터 / Parameter | config 키 / Key | EfficientNet-B0 | ResNet-50 | Hard SSOT |
|---|---|---|---|---|
| `mid_dim` | `phase2.heads.resnet50.mid_dim` 🟢 | — (없음 / absent) | 512 | ✅ |
| `hidden_dim` | `phase2.heads.{backbone}.hidden_dim` 🟢 | 256 | 256 | ✅ |
| `dropout` | `phase2.heads.{backbone}.dropout` 🟢 | 0.2 | 0.4 | Soft |

> `phase2.heads.{backbone}` 키가 없으면 `phase2.hidden_dim` / `phase2.dropout` 최상위 기본값으로 fallback한다.
> Falls back to top-level `phase2.hidden_dim` / `phase2.dropout` if `phase2.heads.{backbone}` is absent.

### 3.4 Best Model 기준 / Best Model Criteria

`val_acc > best_val_acc + es_delta` 조건 충족 시 `best_{channel}.pt` 저장.

> ⚠️ 저장 기준이 `val_acc`(accuracy)이며, PRD 목표 지표인 `macro_f1`과 다름. Optuna도 `best_val_acc`를 목적 함수로 사용한다.
> ⚠️ Best model is saved by `val_acc` (accuracy), which differs from the PRD target metric `macro_f1`. Optuna also uses `best_val_acc` as its objective function.

### 3.5 체크포인트 흐름 / Checkpoint Flow

```
Phase2Trainer.train()
    → {models_dir}/best_{ch}.pt              (best val_acc 기준 / by best val_acc)
    → {reports_dir}/phase2_history_{ch}.csv  (에폭별 이력 CSV / per-epoch history CSV)
```

---

## 4. 손실 함수 / Loss Functions

### 4.1 InfoNCELoss — Phase 0

`training.InfoNCELoss(temperature)` — SimCLR NT-Xent 기반 대조 손실.

| 파라미터 / Parameter | config 키 / Key | 값 / Value | Hard SSOT |
|---|---|---|---|
| temperature (τ) | `phase0.temperature` 🟢 | 0.1 | ✅ |

입력: `z1, z2` — `(B, projection_dim)` L2-normalized vectors  
출력: scalar contrastive loss

### 4.2 CrossEntropyLoss / FocalLoss — Phase 2

`phase2.loss` 설정에 따라 손실 함수가 선택됩니다.

| `phase2.loss` | 반환 | 비고 |
|---|---|---|
| `"cross_entropy"` (기본) | `nn.CrossEntropyLoss` | `label_smoothing` 지원 |
| `"focal"` | `FocalLoss` | `focal_gamma` 지원, 소수 클래스 가중치↑ |

공통 입력 계약:

| 입력 | 형상 | 비고 |
|---|---|---|
| `logits` | `(B, 6)` | raw logits — **Softmax 미적용** |
| `labels` | `(B,)` | int `[0, 5]` |
| `class_weights` | `(6,)` | `class_weights="balanced"` 시 자동 계산 |

> `FocalLoss`: 쉬운 샘플(정상 클래스) 가중치↓, 어려운 샘플(결함 희귀 클래스) 가중치↑.
> K/Y 채널처럼 클래스 불균형이 극심한 경우 `loss="focal"`, `class_weights="balanced"` 조합 권장.

### 4.3 get_loss() 팩토리 / Factory

`training.get_loss(phase: int, cfg: dict, train_samples: list = None) → nn.Module`

| `phase` | `phase2.loss` | 반환 |
|---|---|---|
| `0` | — | `InfoNCELoss` |
| `2` | `"cross_entropy"` | `nn.CrossEntropyLoss(weight=..., label_smoothing=...)` |
| `2` | `"focal"` | `FocalLoss(gamma=..., weight=...)` |

---

## 5. Optimizer / Scheduler

`training/trainer.py`의 `_build_optimizer()` / `_build_scheduler()` factory를 통해 config에서 동적 선택.
Dynamically selected from config via `_build_optimizer()` / `_build_scheduler()` in `training/trainer.py`.

### 5.1 Optimizer Factory

| 컴포넌트 / Component | config 키 / Key | 기본값 / Default | 소비 여부 / Consumed |
|---|---|---|---|
| optimizer 종류 / Optimizer type | `train.optimizer` | `"adamw"` | 🟢 |
| AdamW betas | `train.betas` | `[0.9, 0.999]` | 🟢 (adamw 선택 시 / when adamw selected) |
| SGD momentum | `train.momentum` | `0.9` | 🟢 (sgd 선택 시 / when sgd selected) |

지원 optimizer: `adamw` (기본 / default), `sgd`

### 5.2 Scheduler Factory

| 컴포넌트 / Component | config 키 / Key | 기본값 / Default | 소비 여부 / Consumed |
|---|---|---|---|
| scheduler 종류 / Scheduler type | `train.scheduler` | `"cosine"` | 🟢 |
| CosineAnnealingLR eta_min | `train.eta_min` | 1e-6 | 🟢 |
| StepLR gamma | `train.gamma` | 0.1 | 🟢 (step 선택 시 / when step selected) |
| Gradient clipping | `train.gradient_clip` | 1.0 | 🟢 |

지원 scheduler: `cosine` (기본 / default), `step`

---

## 실행 명령 / Run Commands

| 명령 / Command | 진입점 / Entry Point | 설명 / Description |
| --- | --- | --- |
| Phase 0 단독 | `python -m src.scripts.run_phase0 --channel Y` | SimCLR 학습 / SimCLR training |
| Phase 2 단독 | `python -m src.scripts.run_phase2 --channel Y` | Supervised 학습 / Supervised training |
| Baseline 전체 | `python -m src.scripts.run_baseline --channel all` | Phase 0 → Phase 2 순차 / Sequential Phase 0 → Phase 2 |
| Optuna 튜닝 | `python -m src.scripts.run_optuna --channel Y` | HPO 탐색 / HPO search |
| GUI 학습 | Optional Streamlit UI (legacy) → Train 페이지 | 동일 Trainer 호출 / Same Trainer invocation |
| 평가 단독 | `python -m src.scripts.evaluate --channel Y` | 체크포인트 평가 + 리포트 생성 |

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

학습 시작 시 `utils.log_snapshot()` 을 통해 config 스냅샷을 `outputs/snapshots/` 에 저장한다.
Config snapshot is saved to `outputs/snapshots/` at training start via `utils.log_snapshot()`.

- 출력 경로 / Output path: `outputs/snapshots/config_snapshot_{channel}_{phase}_{timestamp}.json`

---

## 7. Optuna 하이퍼파라미터 탐색 / Optuna Hyperparameter Search

### 7.1 Backbone별 탐색 공간 분리 / Backbone-Specific Search Space

`optuna.search_space.{backbone_name}` 에서 backbone별 독립 범위를 읽는다.
Reads backbone-specific ranges from `optuna.search_space.{backbone_name}`.

| 파라미터 | EfficientNet-B0 범위 | ResNet-50 범위 | 비고 |
|---|---|---|---|
| `learning_rate` | `[5e-5, 3e-4]` | `[1e-4, 5e-4]` | |
| `weight_decay` | `[1e-4, 1e-3]` | `[1e-3, 1e-2]` | |
| `dropout` | `[0.1, 0.3]` | `[0.3, 0.5]` | |
| `hidden_dim` | `[128, 256]` | `[256, 512]` | |
| `mid_dim` (ResNet 전용) | — | `[256, 512, 1024]` | |
| `batch_size` | `[16, 32, 64]` | `[16, 32, 64]` | |
| `epochs` | `[10, 30]` | `[10, 30]` | |
| `label_smoothing` ★ | `[0.0, 0.2]` | `[0.0, 0.2]` | Tier 1 신규 |
| `warmup_epochs` ★ | `[0, 5]` | `[0, 5]` | Tier 1 신규 |
| `class_weights` ★ | `["none","balanced"]` | `["none","balanced"]` | Tier 1 신규 |
| `frozen_backbone` ★ | `[false, true]` | `[false, true]` | Tier 1 신규 |

**Phase 0 탐색 공간 (신규 추가):**

| 파라미터 | 범위 | 비고 |
|---|---|---|
| `learning_rate` | `[1e-4, 1e-2]` | |
| `weight_decay` | `[1e-6, 1e-4]` | |
| `batch_size` | `[16, 32, 64]` | |
| `epochs` | `[5, 15]` | |
| `temperature` ★ | `[0.05, 0.5]` | Tier 1 신규 — SimCLR 표현 품질에 직결 |
| `warmup_epochs` ★ | `[0, 5]` | Tier 1 신규 |

### 7.2 소규모 데이터셋 Optuna 주의사항 / Small Dataset Optuna Cautions

| 위험 / Risk | 대응 / Mitigation |
|---|---|
| Trial 수가 많으면 탐색 자체가 test set에 과적합 / Many trials overfit to test set | `n_trials: 30`으로 제한 / Cap at 30 |
| val_acc 단일 지표의 운에 좌우됨 / val_acc vulnerable to random variation | 목적함수를 `macro_f1` 또는 k-fold 평균으로 변경 고려 / Consider `macro_f1` or k-fold mean |
| MedianPruner 조기 종료 오판 / MedianPruner premature termination | `n_warmup_steps: 10` 보장 (`optuna.pruner.n_warmup_steps`) |

---

## 8. 멀티 채널 실행 / Multi-Channel Execution

4개 CMYK 채널은 **독립적으로** 학습된다. 모델을 공유하지 않으며, 채널별로 Phase 0 → Phase 2 전체 파이프라인이 개별 실행된다.
The 4 CMYK channels are trained **independently**. No model is shared — each channel runs through the full Phase 0 → Phase 2 pipeline separately.

| 채널 / Channel | Phase 0 backbone | Phase 2 모델 / Model | 독립성 / Independence |
|---|---|---|---|
| Y | `phase0_backbone_Y_{tag}.pt` | `best_Y.pt` | ✅ 타 채널과 완전 독립 / Fully independent |
| M | `phase0_backbone_M_{tag}.pt` | `best_M.pt` | ✅ |
| C | `phase0_backbone_C_{tag}.pt` | `best_C.pt` | ✅ |
| K | `phase0_backbone_K_{tag}.pt` | `best_K.pt` | ✅ |

> **Channel Invariant**: 동일 channel의 Phase 0 backbone만 해당 channel의 Phase 2에 사용할 수 있다. (SSOT-FF01)
> Only the Phase 0 backbone of the **same** channel may be used for that channel's Phase 2 training.

---

## 체크리스트 / Checklist

- [ ] 새 Phase 추가 시 Swing Architecture 흐름도 갱신 / Update Swing Architecture diagram when adding new phase
- [ ] Optimizer/Scheduler 추가 시 §5 팩토리 업데이트 / Update §5 factory when adding optimizer/scheduler
- [ ] Best 기준 변경 시 PRD 정렬 확인 / Verify PRD alignment on best-save criterion change
- [ ] Optuna 탐색 범위 변경 시 §7 동기화 / Sync §7 on Optuna search range change

---

## See Also

| 문서 / Document | 관계 / Relation |
| --- | --- |
| [SSOT_Model_Architecture.md](SSOT_Model_Architecture.md) | 모델 구조 / Model architecture |
| [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md) | 데이터 입력 계약 / Data input contract |
| [SSOT_Evaluation_Reporting.md](SSOT_Evaluation_Reporting.md) | 평가/피드백 / Evaluation/feedback |
| [SSOT_Config_Resolution.md](SSOT_Config_Resolution.md) | config 키 매핑 / Config key mapping |

