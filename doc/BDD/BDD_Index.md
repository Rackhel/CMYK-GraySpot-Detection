---
type: bdd_index
domain: all
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [BDD] 명세 인덱스 / Specification Index

> **역할 / Role**: 프로젝트 전체 BDD 명세 파일의 목록과 도메인 매핑을 제공한다.
> **Role**: Provides a list of all BDD specification files and domain mappings for the project.

---

## BDD 파일 목록 / BDD File List

| 파일 / File | 도메인 / Domain | 시나리오 수 / Scenarios | 행위자 / Actors |
|---|---|---|---|
| [BDD_Inference.md](BDD_Inference.md) | 추론 파이프라인 / Inference Pipeline | 5 | QC Inspector, System |
| [BDD_Training.md](BDD_Training.md) | 학습 파이프라인 / Training Pipeline | 5 | Operator, System |
| [BDD_SwingFeedback.md](BDD_SwingFeedback.md) | Swing Feedback 루프 / Swing Feedback Loop | 5 | QC Manager, System |
| [BDD_Evaluation.md](BDD_Evaluation.md) | 평가 리포트 / Evaluation Reports | 7 | Data Scientist, QC Manager, Operator |
| [BDD_Tuning.md](BDD_Tuning.md) | Optuna 튜닝 / Optuna Tuning | 5 | Data Scientist, System |
| [BDD_Channel.md](BDD_Channel.md) | CMYK 채널 독립 / Channel Independence | 4 | Operator, System |
| [BDD_Safety.md](BDD_Safety.md) | 시스템 안전성 / System Safety | 6 | System, Operator |
| [BDD_ROI_Pipeline.md](BDD_ROI_Pipeline.md) | ROI 파이프라인 / ROI Pipeline | 7 | Operator, Data Scientist |
| [BDD_GUI.md](BDD_GUI.md) | PyQt6 GUI | 6 | Operator, Data Scientist |
| [BDD_ONNX.md](BDD_ONNX.md) | ONNX 내보내기 / ONNX Export | 5 | Operator, System |

**총 시나리오 / Total Scenarios**: 55

---

## 행위자 정의 / Actor Definitions

| 행위자 / Actor | 역할 / Role | 주요 시나리오 / Key Scenarios |
|---|---|---|
| **운영자 / Operator** | 학습 파이프라인을 실행하고 모델을 배포하는 엔지니어 / Engineer running training and deploying models | Training, Safety, ONNX |
| **품질 검사원 / QC Inspector** | 인쇄물 결함 수준을 판정하는 현장 검사자 / Field inspector judging print defect levels | Inference |
| **품질 관리자 / QC Manager** | 검사 결과를 기반으로 재학습/통과 여부를 결정하는 관리자 / Manager deciding retrain/pass | SwingFeedback, Evaluation |
| **데이터 과학자 / Data Scientist** | 모델 성능을 분석하고 하이퍼파라미터를 최적화하는 연구자 / Researcher analyzing performance and tuning hyperparameters | Tuning, Evaluation, GUI, ROI |
| **시스템 / System** | CMYK Grayspot Detection Pipeline 자체 / The CMYK Grayspot Detection Pipeline itself | All domains |

---

## SSOT → BDD 매핑 / SSOT to BDD Mapping

| SSOT 문서 / SSOT Doc | 연관 BDD 파일 / Related BDD Files |
|---|---|
| `SSOT_Core.md` | BDD_Safety.md, BDD_Channel.md |
| `SSOT_Data_Pipeline.md` | BDD_Channel.md, BDD_ROI_Pipeline.md |
| `SSOT_Model_Architecture.md` | BDD_Training.md, BDD_ONNX.md |
| `SSOT_Training_Pipeline.md` | BDD_Training.md, BDD_Tuning.md |
| `SSOT_Evaluation_Reporting.md` | BDD_Evaluation.md, BDD_SwingFeedback.md |
| `SSOT_Artifacts.md` | BDD_ONNX.md |
| `SSOT_Validation_Codes.md` | BDD_Safety.md |
| `SSOT_GUI.md` | BDD_GUI.md |
| `SSOT_ROI_Pipeline.md` | BDD_ROI_Pipeline.md |

---

## BDD → TDD 매핑 / BDD to TDD Mapping

| BDD 파일 / BDD File | TDD 파일 / TDD Files |
|---|---|
| BDD_Inference.md | `test_predictor.py`, `test_models.py` |
| BDD_Training.md | `test_models.py`, `test_losses.py`, `test_smoke_phase0.py`, `test_smoke_phase2.py` |
| BDD_SwingFeedback.md | `test_metrics.py`, `test_swing_efficiency.py` |
| BDD_Evaluation.md | `test_metrics.py`, `test_confusion.py`, `test_reporting.py`, `test_evaluate_script.py` |
| BDD_Tuning.md | `test_search_space.py`, `test_smoke_optuna.py` |
| BDD_Channel.md | `test_preprocessing.py`, `test_roi_extractor.py`, `test_smoke_phase2.py` |
| BDD_Safety.md | `test_utils_config.py`, `test_models.py`, `test_predictor.py` |
| BDD_ROI_Pipeline.md | `test_roi_extractor.py`, `test_label_refiner.py` |
| BDD_GUI.md | `test_gui_workers.py`, `test_gui_tabs.py` |
| BDD_ONNX.md | `test_onnx_export.py` |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [doc/TDD/](../TDD/) | 각 시나리오의 단위 테스트 명세 / Unit test specification for each scenario |
| [doc/SSOT/](../SSOT/) | 시나리오 수치 근거 / Numeric basis for scenarios |
| [doc/Contract/](../Contract/) | 모듈 경계 계약 / Module boundary contracts |
