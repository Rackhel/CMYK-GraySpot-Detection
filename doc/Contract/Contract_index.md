---
type: contract
domain: data_flow_overview
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] Data Flow Overview — 전체 데이터 흐름 및 Contract 인덱스 / Full Data Flow and Contract Index

> **목적 / Purpose**: 시스템 전체 데이터 흐름도와 도메인별 Contract 문서 인덱스를 제공한다. / Provides a system-wide data flow diagram and a domain-specific Contract document index.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-17

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.
> / This document does not redefine SSOT semantic definitions. Follow SSOT_Core.md as the final authority for semantic interpretation.

---

## 1. Contract 문서 인덱스 / Contract Document Index

| # | Contract | 도메인 / Domain | 위치 / Location |
| --- | --- | --- | --- |
| 1 | [Config Resolution](Contract_config_resolution.md) | Config 로딩/검증 / Config loading/validation | `doc/Contract/` |
| 2 | [Data Pipeline](Contract_data_pipeline.md) | 데이터 전처리/증강 / Data preprocessing/augmentation | `doc/Contract/` |
| 3 | [Model Boundary](Contract_model_boundary.md) | 모델 구조/Phase 전환 / Model architecture/Phase transition | `doc/Contract/` |
| 4 | [Training Pipeline](Contract_training_pipeline.md) | 학습/손실/Optimizer / Training/loss/optimizer | `doc/Contract/` |
| 5 | [Artifact Boundary](Contract_artifact_boundary.md) | 체크포인트/이력/스냅샷 / Checkpoint/history/snapshot | `doc/Contract/` |
| 6 | [Evaluation & Reporting](Contract_evaluation_reporting.md) | 평가/리포트/Swing / Evaluation/report/Swing | `doc/Contract/` |
| 7 | [Inference Boundary](Contract_inference_boundary.md) | 추론 (GrayspotPredictor) / Inference | `doc/Contract/` |
| 8 | [Tuning Boundary](Contract_tuning_boundary.md) | Optuna 튜닝 / Optuna tuning | `doc/Contract/` |
| 9 | [Fail-Fast Enforcement](Contract_fail_fast.md) | Fail-Fast 집행 포인트 / Fail-Fast enforcement points | `doc/Contract/` |
| 10 | [GUI](Contract_gui.md) | PyQt6 6탭 GUI / PyQt6 6-tab GUI | `doc/Contract/` |
| 11 | [ROI Pipeline](Contract_roi_pipeline.md) | ROI 추출 / 라벨 정제 / ROI extraction / label refinement | `doc/Contract/` |

---

## 2. SSOT ↔ Contract 매핑 / SSOT ↔ Contract Mapping

| SSOT 문서 (What) / SSOT Document | Contract 문서 (How) / Contract Document |
| --- | --- |
| [SSOT_Core.md](../SSOT/SSOT_Core.md) | 모든 Contract의 의미 권위자 / Semantic authority for all Contracts |
| [SSOT_Config_Resolution.md](../SSOT/SSOT_Config_Resolution.md) | [Contract_config_resolution.md](Contract_config_resolution.md) |
| [SSOT_Data_Pipeline.md](../SSOT/SSOT_Data_Pipeline.md) | [Contract_data_pipeline.md](Contract_data_pipeline.md) |
| [SSOT_Model_Architecture.md](../SSOT/SSOT_Model_Architecture.md) | [Contract_model_boundary.md](Contract_model_boundary.md) |
| [SSOT_Training_Pipeline.md](../SSOT/SSOT_Training_Pipeline.md) | [Contract_training_pipeline.md](Contract_training_pipeline.md), [Contract_tuning_boundary.md](Contract_tuning_boundary.md) |
| [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) | [Contract_artifact_boundary.md](Contract_artifact_boundary.md) |
| [SSOT_Evaluation_Reporting.md](../SSOT/SSOT_Evaluation_Reporting.md) | [Contract_evaluation_reporting.md](Contract_evaluation_reporting.md) |
| [SSOT_Validation_Codes.md](../SSOT/SSOT_Validation_Codes.md) | [Contract_fail_fast.md](Contract_fail_fast.md) |
| [SSOT_GlobalVariables.md](../SSOT/SSOT_GlobalVariables.md) | (Cross-cutting — 각 Contract에서 Hard/Soft 참조 / Referenced as Hard/Soft in each Contract) |
| [SSOT_GUI.md](../SSOT/SSOT_GUI.md) | [Contract_gui.md](Contract_gui.md) |
| [SSOT_ROI_Pipeline.md](../SSOT/SSOT_ROI_Pipeline.md) | [Contract_roi_pipeline.md](Contract_roi_pipeline.md) |

---

## 3. 전체 데이터 흐름 / Full Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  config/config.json  ──►  utils_config.load_config()  ──►  cfg: dict   │
│                               (모든 모듈이 cfg를 수령 / all modules receive cfg)│
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ cfg
                    ┌──────────────▼──────────────┐
                    │    data_set/labeled/         │
                    │  {channel}/{level}/*.png     │
                    │  BGR uint8  (H, W, 3)        │
                    └──────┬──────────────┬────────┘
                           │              │
               ┌───────────▼──┐    ┌──────▼────────────┐
               │ContrastiveDS │    │   CMYKDataset      │
               │ (Phase 0)    │    │   (Phase 2)        │
               │(view1, view2)│    │ (image, label)     │
               │(3,128,128)×2 │    │ (3,128,128), [0-5] │
               └──────┬───────┘    └──────┬─────────────┘
                      │                   │
               ┌──────▼───────┐    ┌──────▼─────────────┐
               │GrayspotModel │    │ GrayspotModel       │
               │  phase=0     │    │   phase=2           │
               │(B,128) L2    │    │  (B,6) logits       │
               └──────┬───────┘    └──────┬─────────────┘
                      │                   │
               ┌──────▼───────┐    ┌──────▼─────────────┐
               │ InfoNCELoss  │    │CrossEntropyLoss     │
               │ scalar loss  │    │ scalar loss + acc   │
               └──────┬───────┘    └──────┬─────────────┘
                      │                   │
               ┌──────▼───────┐    ┌──────▼─────────────┐
               │Phase0Trainer │    │ Phase2Trainer       │
               │save_backbone │    │  best val_acc save  │
               └──────┬───────┘    └──────┬─────────────┘
                      │                   │
          ┌───────────▼──────┐   ┌────────▼───────────────┐
          │phase0_backbone_  │   │    best_{ch}.pt         │
          │{ch}_{tag}.pt     │   │  (backbone+ClassHead)   │
          └───────────┬───────┘   └────────┬───────────────┘
                      │ switch_to_phase2()  │ load (eval mode)
                      └──────────┬──────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │        Evaluator             │
                    │  run() → y_true, y_pred,     │
                    │          confidences         │
                    └────────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │    compute_all_channels()    │
                    │    EvaluationSummary         │
                    └────────────┬────────────────┘
                                 │
               ┌─────────────────┼──────────────────┐
               │                 │                  │
    ┌──────────▼──────┐  ┌───────▼────────┐  ┌─────▼──────────────┐
    │determine_swing_ │  │  save_report() │  │generate_baseline_  │
    │feedback()       │  │  CSV + JSON    │  │report()            │
    │{terminate, ...} │  │  outputs/      │  │ HTML dashboard      │
    └─────────────────┘  └───────────────┘  └────────────────────┘
```
