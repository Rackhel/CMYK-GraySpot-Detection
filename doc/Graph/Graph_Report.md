---
type: graph_report
domain: dependency
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
---

# [Graph] 전체 프로젝트 파일 의존성 보고서

> **목적**: CMYK Grayspot 프로젝트의 Python 소스 파일 및 문서 간 의존성 구조를 기록한다.
> **원천**: `doc/Graph/graph.json` (노드 63개, 엣지 81개)

---

## 1. 패키지 구성 개요

| 패키지 | 파일 수 | 역할 |
| --- | --- | --- |
| `src/utils/` | 3 | 공통 유틸리티 (config 로딩, 모델 유틸, 로거) |
| `src/data/` | 4 | 데이터 파이프라인 (Dataset, 증강, 전처리, 정규화) |
| `src/models/` | 4 | 모델 구조 (Backbone, ClassifierHead, ProjectionHead, GrayspotModel) |
| `src/training/` | 3 | 학습 (Phase0/Phase2Trainer, InfoNCELoss, get_loss) |
| `src/evaluation/` | 7 | 평가 (Evaluator 4-Mixin, metrics, confusion) |
| `src/inference/` | 4 | 추론 (GrayspotPredictor 3-Mixin) |
| `src/reporting/` | 1 | HTML 리포트 생성 |
| `src/tuning/` | 3 | Optuna HPO (optuna_tuner, search_space, optuna_utils) |
| `src/scripts/` | 7 | 진입점 (run_phase0/2/baseline/optuna, generate_report, train, tsne) |
| `src/tests/unit/` | 8 | 단위 테스트 |
| `src/tests/integration/` | 3 | 통합 테스트 |
| `src/tests/smoke/` | 3 | 스모크 테스트 |
| `doc/SSOT/` | 9 | SSOT 문서 (What 정의) |
| `doc/Contract/` | 10 | Contract 문서 (How 계약) |

---

## 2. 의존성 계층도

```
Layer 0 — 외부 의존 없음 (Leaf)
├── utils/logger.py
├── utils/utils_config.py
├── utils/utils_model.py
├── data/augmentation.py
├── data/preprocessing.py
├── data/normalize.py
├── models/backbone.py
├── models/classifier.py
├── models/projection_head.py
├── training/contrastive_loss.py
├── tuning/optuna_utils.py
├── tuning/search_space.py
└── evaluation/metrics.py

Layer 1 — Layer 0에만 의존
├── data/dataset.py           → augmentation, normalize, preprocessing
├── models/grayspot_model.py  → backbone, classifier, projection_head
├── training/losses.py        → contrastive_loss
├── evaluation/confusion.py   → metrics
├── evaluation/evaluator_charts.py   → metrics
├── evaluation/evaluator_export.py   → metrics
├── evaluation/evaluator_metrics.py  → metrics
├── inference/predictor_device.py    → utils/logger
├── inference/predictor_inference.py → data/normalize, utils/logger
└── inference/predictor_loader.py    → models/grayspot_model, utils/logger

Layer 2 — Layer 1에 의존
├── training/trainer.py       → data/dataset, models/grayspot_model, training/losses
├── evaluation/evaluator_inference.py (외부 전용)
├── inference/predictor.py    → predictor_device, predictor_inference,
│                               predictor_loader, utils/utils_config
└── evaluation/evaluator.py   → confusion, evaluator_charts, evaluator_export,
                                evaluator_inference, evaluator_metrics, metrics

Layer 3 — 진입점 / 통합
├── reporting/html_report.py          → evaluation/confusion, evaluation/metrics
├── tuning/optuna_tuner.py            → tuning/optuna_utils, tuning/search_space
├── scripts/run_phase0.py             → data/dataset, models/grayspot_model,
│                                       utils, training/trainer
├── scripts/run_phase2.py             → data/dataset, models/grayspot_model,
│                                       utils, training/trainer
├── scripts/run_baseline.py           → data/dataset, models/grayspot_model,
│                                       utils, training/trainer
├── scripts/run_optuna.py             → tuning/optuna_tuner
├── scripts/generate_baseline_report.py → evaluation/confusion, evaluator,
│                                          metrics, utils
└── scripts/train.py                  → utils
```

---

## 3. 핵심 허브 파일 (High In-Degree)

| 파일 | 피참조 수 | 참조하는 파일 |
| --- | --- | --- |
| `evaluation/metrics.py` | 7 | confusion, evaluator_charts, evaluator_export, evaluator_metrics, evaluator, html_report, generate_baseline_report |
| `models/grayspot_model.py` | 5 | predictor_loader, trainer, run_phase0, run_phase2, run_baseline |
| `training/trainer.py` | 4 | run_phase0, run_phase2, run_baseline, smoke_phase0/phase2 |
| `data/dataset.py` | 4 | trainer, run_phase0, run_phase2, run_baseline |
| `utils/utils_config.py` | 4 | predictor, run_phase0/phase2/baseline/train, generate_report |
| `utils/logger.py` | 3 | predictor_device, predictor_inference, predictor_loader |
| `evaluation/confusion.py` | 3 | evaluator, html_report, generate_baseline_report |
| `evaluation/evaluator.py` | 2 | generate_baseline_report, test_evaluation |

---

## 4. 순환 의존성

> ✅ **순환 의존성 없음** — 모든 의존성이 단방향.

---

## 5. 테스트 커버리지 매핑

### 5.1 단위 테스트 (Unit)

| 테스트 파일 | 대상 모듈 |
| --- | --- |
| `test_augmentation.py` | `data/augmentation.py` |
| `test_confusion.py` | `evaluation/confusion.py` |
| `test_losses.py` | `training/contrastive_loss.py`, `training/losses.py` |
| `test_metrics.py` | `evaluation/metrics.py` |
| `test_models.py` | `models/classifier.py`, `models/projection_head.py` |
| `test_preprocessing.py` | `data/preprocessing.py` |
| `test_utils_config.py` | `utils/utils_config.py` |
| `test_utils_model.py` | `utils/utils_model.py` |

### 5.2 통합 테스트 (Integration)

| 테스트 파일 | 대상 모듈 |
| --- | --- |
| `test_data_pipeline.py` | `data/dataset.py`, `data/augmentation.py`, `data/preprocessing.py` |
| `test_evaluation.py` | `evaluation/evaluator.py`, `evaluation/metrics.py`, `evaluation/confusion.py` |
| `test_predictor_integration.py` | `inference/predictor.py` |

### 5.3 스모크 테스트 (Smoke)

| 테스트 파일 | 대상 모듈 |
| --- | --- |
| `test_smoke_phase0.py` | `training/trainer.py`, `models/grayspot_model.py` |
| `test_smoke_phase2.py` | `training/trainer.py`, `models/grayspot_model.py` |
| `test_smoke_optuna.py` | `tuning/optuna_tuner.py`, `tuning/search_space.py` |

### 5.4 미커버 모듈

| 모듈 | 테스트 없음 |
| --- | --- |
| `models/backbone.py` | — |
| `models/grayspot_model.py` | 직접 단위테스트 없음 (smoke에서 간접 커버) |
| `inference/predictor_*.py` | integration에서만 커버 |
| `reporting/html_report.py` | — |
| `tuning/optuna_utils.py` | — |
| `utils/logger.py` | — |
| `utils/utils_model.py` | 단위테스트 부분 커버 |

---

## 6. 문서 의존성 (SSOT → Contract)

| SSOT 문서 (What) | Contract 문서 (How) |
| --- | --- |
| `SSOT_Core.md` | 모든 Contract의 의미 권위자 |
| `SSOT_Config_Resolution.md` | `Contract_config_resolution.md` |
| `SSOT_Data_Pipeline.md` | `Contract_data_pipeline.md` |
| `SSOT_Model_Architecture.md` | `Contract_model_boundary.md` |
| `SSOT_Training_Pipeline.md` | `Contract_training_pipeline.md`, `Contract_tuning_boundary.md` |
| `SSOT_Artifacts.md` | `Contract_artifact_boundary.md` |
| `SSOT_Evaluation_Reporting.md` | `Contract_evaluation_reporting.md` |
| `SSOT_Validation_Codes.md` | `Contract_fail_fast.md` |
| `SSOT_GlobalVariables.md` | Cross-cutting — 각 Contract에서 Hard/Soft 참조 |

---

## 7. 데이터 흐름 요약

```
config.json
    │ load_config()
    ▼
utils_config ──────────────────────────────────────┐
    │                                               │
    ├─► data/dataset (CMYKDataset / ContrastiveDS)  │
    │       │ augmentation / preprocessing          │
    │       ▼                                       │
    ├─► models/grayspot_model ◄─ backbone           │
    │         │                   classifier        │
    │         │                   projection_head   │
    │         ▼                                     │
    ├─► training/trainer (Phase0 / Phase2) ─────────┤
    │         │ losses / InfoNCELoss                │
    │         ▼                                     │
    │    checkpoint (best_{ch}.pt)                  │
    │         │                                     │
    ├─► evaluation/evaluator ──────────────────────-┤
    │         │ metrics / confusion / charts        │
    │         ▼                                     │
    │    determine_swing_feedback()                 │
    │         │                                     │
    ├─► reporting/html_report                       │
    │                                               │
    └─► inference/predictor ◄───────────────────────┘
              │ predictor_device / _inference / _loader
              ▼
         predict() / predict_batch()
```

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [graph.json](graph.json) | 기계가독 그래프 데이터 (노드/엣지) |
| [Contract_index.md](../Contract/Contract_index.md) | 전체 데이터 흐름도 |
| [SSOT_Core.md](../SSOT/SSOT_Core.md) | 프로젝트 정체성 및 원칙 |
