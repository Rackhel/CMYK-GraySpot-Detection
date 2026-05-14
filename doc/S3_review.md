# S3 Review / S3 리뷰

S3 스프린트 완료 보고서 — CMYK Grayspot Detection Pipeline.
Sprint S3 completion report for the CMYK Grayspot Detection Pipeline.

---

## Table of Contents / 목차

1. [S3 완료 기준 / Completion Criteria](#1-s3-완료-기준--completion-criteria)
2. [전체 파이프라인 완주 기록 / Full Pipeline Run Log](#2-전체-파이프라인-완주-기록--full-pipeline-run-log)
3. [성능 비교 / Performance Comparison vs Baseline](#3-성능-비교--performance-comparison-vs-baseline)
4. [Swing Efficiency 지표 / Swing Efficiency Metrics](#4-swing-efficiency-지표--swing-efficiency-metrics)
5. [피드백 판단 결과 / Feedback Decision](#5-피드백-판단-결과--feedback-decision)
6. [GUI 현황 / GUI Status](#6-gui-현황--gui-status)
7. [SSOT 문서 변경 이력 / SSOT Change Log](#7-ssot-문서-변경-이력--ssot-change-log)
8. [Contract 변경 이력 / Contract Change Log](#8-contract-변경-이력--contract-change-log)
9. [TDD 진행 현황 / TDD Progress](#9-tdd-진행-현황--tdd-progress)
10. [ADR 기술 결정 / Architecture Decision Records](#10-adr-기술-결정--architecture-decision-records)
11. [팀원 회고 / Team Retrospective](#11-팀원-회고--team-retrospective)

---

## 1. S3 완료 기준 / Completion Criteria

### 체크리스트 / Checklist

- [ ] Phase 0 → 1 → 2 → 3 전체 파이프라인 1회 완주 / Phase 0 → 1 → 2 → 3 full pipeline completed once
- [ ] Baseline 대비 성능 비교 수치 존재 / Performance comparison numbers vs Baseline exist
- [ ] Swing Efficiency 지표 (라벨 수정수, ΔAcc) 기록 / Swing Efficiency metrics (label edit count, ΔAcc) recorded
- [ ] 피드백 판단 결과 문서화 (Cycle 2 방향 결정) / Feedback decision result documented (Cycle 2 direction decided)
- [ ] GUI에서 학습/평가/Embedding 분석 기본 동작 / Basic operation of training/evaluation/Embedding analysis in GUI

> **📝 참고 / Note**: GUI 기능이 미완성이어도 Swing Cycle은 CLI (`scripts/train.py`, `scripts/evaluate.py`) 기준으로 진행 가능하다. GUI는 블로커가 아닌 병행 작업이다.
>
> Even if GUI features are incomplete, the Swing Cycle can proceed based on CLI (`scripts/train.py`, `scripts/evaluate.py`). GUI is a parallel task, not a blocker.

---

## 2. 전체 파이프라인 완주 기록 / Full Pipeline Run Log

### 2.1 실행 환경 / Execution Environment

| 항목 / Item | 값 / Value |
|---|---|
| 실행일 / Run Date | 작성 예정 / TBD |
| 디바이스 / Device | 작성 예정 / TBD |
| Python 버전 / Python Version | 3.11.x |
| PyTorch 버전 / PyTorch Version | 2.1.0+ |
| Config 파일 / Config File | `src/config/config.json` |

### 2.2 Phase별 실행 결과 / Per-Phase Results

| Phase | 스크립트 / Script | 상태 / Status | 비고 / Note |
|---|---|---|---|
| Phase 0 — Contrastive | `scripts/run_phase0.py` | 🚧 진행 예정 / Planned | SimCLR InfoNCE |
| Phase 2 — Supervised | `scripts/run_phase2.py` | 🚧 진행 예정 / Planned | EfficientNet-B0 |
| Baseline Eval | `scripts/run_baseline.py` | 🚧 진행 예정 / Planned | 채널별 평가 / Per-channel eval |
| Report | `scripts/generate_baseline_report.py` | 🚧 진행 예정 / Planned | HTML 리포트 / HTML report |

### 2.3 산출 파일 / Output Artifacts

```
outputs/
├── checkpoints/
│   ├── best_{ch}.pt                  ← Phase 2 최적 모델 / Best model per channel
│   ├── phase0_v1.pt                  ← Phase 0 통합 체크포인트 / Combined checkpoint
│   ├── phase0_history_{ch}.csv       ← Phase 0 학습 이력 / Training history
│   ├── phase2_history_{ch}.csv       ← Phase 2 학습 이력 / Training history
│   └── phase2_summary_v1.json        ← 실행 요약 / Run summary
├── snapshots/
│   └── config_snapshot_{tag}_{ts}.json  ← 실행 시점 설정 스냅샷 / Config snapshot
└── reports/
    └── phase2_v1.html                ← HTML 평가 리포트 / HTML evaluation report
```

---

## 3. 성능 비교 / Performance Comparison vs Baseline

### 3.1 채널별 정확도 / Per-Channel Accuracy

| 채널 / Channel | Baseline Acc | S3 Acc | ΔAcc | 목표 / Target | 달성 여부 / Met |
|---|---|---|---|---|---|
| Y | — | — | — | ≥ 85% | — |
| M | — | — | — | ≥ 85% | — |
| C | — | — | — | ≥ 85% | — |
| K | — | — | — | ≥ 85% | — |
| **Overall / 전체** | — | — | — | ≥ 90% | — |

> **📝 비고 / Note**: 학습 완료 후 `outputs/checkpoints/phase2_summary_v1.json` 수치로 채워 넣는다.
>
> Fill in from `outputs/checkpoints/phase2_summary_v1.json` after training completes.

### 3.2 추가 지표 / Additional Metrics

| 지표 / Metric | Baseline | S3 | 목표 / Target |
|---|---|---|---|
| 전체 정확도 / Overall Accuracy | — | — | ≥ 90% |
| 클래스별 F1 (macro) / Per-class F1 (macro) | — | — | ≥ 0.80 |
| 평균 절대 오차 / MAE | — | — | ≤ 0.50 |

---

## 4. Swing Efficiency 지표 / Swing Efficiency Metrics

> Swing Cycle: 모델 예측 → 사용자 피드백 → 라벨 수정 → 재학습 → 성능 향상 반복 루프.
>
> Swing Cycle: model prediction → user feedback → label correction → retraining → accuracy improvement loop.

### 4.1 Cycle 1 Swing 지표 / Cycle 1 Metrics

| 지표 / Metric | 값 / Value | 비고 / Note |
|---|---|---|
| 총 검수 이미지 수 / Images reviewed | — | 작성 예정 / TBD |
| 라벨 수정 수 / Label corrections | — | 작성 예정 / TBD |
| 수정률 / Correction rate | — | 수정 수 ÷ 검수 수 / corrections ÷ reviewed |
| ΔAcc (수정 전→후) / ΔAcc (before→after) | — | 작성 예정 / TBD |
| Swing 소요 시간 / Swing duration | — | 작성 예정 / TBD |

### 4.2 채널별 Swing 상세 / Per-Channel Swing Detail

| 채널 / Channel | 검수 수 / Reviewed | 수정 수 / Corrections | 수정률 / Rate | ΔAcc |
|---|---|---|---|---|
| Y | — | — | — | — |
| M | — | — | — | — |
| C | — | — | — | — |
| K | — | — | — | — |

---

## 5. 피드백 판단 결과 / Feedback Decision

### 5.1 Swing 피드백 판단 기준 / Swing Feedback Criteria

`config.json` → `evaluation.swing_thresholds` 기준 적용 / Applied from `evaluation.swing_thresholds`:

| 조건 / Condition | 임계값 / Threshold | 판단 / Decision |
|---|---|---|
| `acc_retry` | 0.80 | 정확도 미달 시 재학습 / Retrain if accuracy below threshold |
| `f1_retry` | 0.70 | F1 미달 시 재학습 / Retrain if F1 below threshold |
| `mae_retry` | 0.80 | MAE 미달 시 재학습 / Retrain if MAE below threshold |

### 5.2 Cycle 1 판단 결과 / Cycle 1 Decision Result

| 항목 / Item | 결과 / Result |
|---|---|
| 최종 판정 / Final verdict | 🚧 작성 예정 / TBD |
| Cycle 2 진행 여부 / Proceed to Cycle 2 | 🚧 작성 예정 / TBD |
| 주요 결정 사항 / Key decisions | 작성 예정 / TBD |

### 5.3 Cycle 2 방향 / Cycle 2 Direction

> 작성 예정 — Cycle 1 결과 분석 후 기재.
>
> To be filled in after Cycle 1 results are analyzed.

- [ ] 추가 데이터 수집 필요 여부 / Additional data collection needed
- [ ] 하이퍼파라미터 재튜닝 필요 여부 / Hyperparameter re-tuning (Optuna) needed
- [ ] 모델 구조 변경 필요 여부 / Model architecture change needed
- [ ] Phase 0 재실행 필요 여부 / Phase 0 re-run needed

---

## 6. GUI 현황 / GUI Status

### 6.1 기능 구현 현황 / Feature Implementation Status

| 기능 / Feature | 상태 / Status | 비고 / Note |
|---|---|---|
| 학습 실행 / Training trigger | ❌ 미시작 / Not Started | CLI로 대체 가능 / Replaceable via CLI |
| 평가 실행 / Evaluation trigger | ❌ 미시작 / Not Started | CLI로 대체 가능 / Replaceable via CLI |
| Embedding 분석 / Embedding analysis | ❌ 미시작 / Not Started | CLI로 대체 가능 / Replaceable via CLI |
| 결과 시각화 / Result visualization | ❌ 미시작 / Not Started | HTML 리포트로 대체 / Replaced by HTML report |

### 6.2 CLI 대체 실행 명령 / CLI Alternative Commands

```bash
# Phase 0 학습 / Phase 0 Training
python src/scripts/run_phase0.py

# Phase 2 학습 / Phase 2 Training
python src/scripts/run_phase2.py

# 평가 및 리포트 / Evaluation & Report
python src/scripts/run_baseline.py

# HTML 리포트 생성 / Generate HTML Report
python src/scripts/generate_baseline_report.py
```

> **📝 참고 / Note**: GUI는 병행 작업으로 S3 완료 기준의 블로커가 아니다.
>
> GUI is a parallel task and is not a blocker for S3 completion criteria.

---

## 7. SSOT 문서 변경 이력 / SSOT Change Log

### 7.1 S3 기간 중 변경된 SSOT 문서 / SSOT Documents Changed During S3

| 문서 / Document | 버전 변경 / Version Change | 주요 변경 내용 / Key Changes |
|---|---|---|
| `SSOT_Data_Pipeline.md` | v0.2.0 → v0.3.0 | SSOT-NM01 해소 — ImageNet 정규화 적용 / SSOT-NM01 resolved — ImageNet normalization applied |
| `SSOT_Artifacts.md` | — | 디렉토리 구조 현행화 / Directory structure updated to current state |

### 7.2 해소된 SSOT 위반 / Resolved Violations

| 코드 / Code | 내용 / Description | 해결 / Fix |
|---|---|---|
| SSOT-NM01 | ImageNet 정규화 미적용 — pretrained backbone 성능 저하 가능 / ImageNet normalization not applied — potential pretrained backbone performance degradation | `dataset.py`에 `_IMAGENET_NORMALIZE` 추가 / Added `_IMAGENET_NORMALIZE` to `dataset.py` |

### 7.3 잔존 SSOT 위반 / Remaining Violations

| 코드 / Code | 내용 / Description | 등급 / Level | 비고 / Note |
|---|---|---|---|
| SSOT-CS01 | BGR/RGB 불일치 위험 — 추론 시 동일 색상 공간 유지 필수 / BGR/RGB mismatch risk — must maintain same color space at inference | Level 1 | 지속 모니터링 / Ongoing monitoring |

---

## 8. Contract 변경 이력 / Contract Change Log

### 8.1 S3 기간 중 변경된 계약 / Contracts Changed During S3

| 섹션 / Section | 변경 전 / Before | 변경 후 / After |
|---|---|---|
| §3.1 Tensor 변환 후 범위 / Tensor range after conversion | `[0.0, 1.0]` float32 | ImageNet-normalized float32 |
| §3.2 ContrastiveDataset 출력 범위 / Output range | `[0.0, 1.0]` | ImageNet-normalized |
| §3.3 CMYKDataset 출력 범위 / Output range | `[0.0, 1.0]` | ImageNet-normalized |
| §7.1 Evaluator 이미지 배치 입력 / Image batch input | `BGR float32 [0, 1]` | `BGR float32, ImageNet-normalized` |

### 8.2 변경 근거 / Change Rationale

> pretrained EfficientNet-B0 / ResNet50은 ImageNet 통계(`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)로 사전학습되었다. 이를 적용하지 않으면 backbone의 feature 분포가 학습 분포와 불일치하여 성능이 저하된다 (SSOT-NM01).
>
> Pretrained EfficientNet-B0 / ResNet50 were pre-trained with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Without applying them, backbone feature distributions mismatch training distributions, degrading performance (SSOT-NM01).

---

## 9. TDD 진행 현황 / TDD Progress

> 참고 문서 / Reference: [`doc/TDD.md`](TDD.md) — TDD 전략 전문 문서 / TDD strategy document.

### 9.1 테스트 피라미드 현황 / Test Pyramid Status

| 계층 / Layer | 파일 수 / Files | 상태 / Status | 실행 명령 / Run Command |
|---|---|---|---|
| Unit | 8개 / 8 files | ✅ 작성 완료 / Complete | `pytest src/tests/unit/ -v` |
| Integration | 3개 / 3 files | ✅ 작성 완료 / Complete | `pytest src/tests/integration/ -v` |
| Smoke | 3개 / 3 files | ✅ 작성 완료 / Complete | `pytest src/tests/smoke/ -v -m smoke` |

### 9.2 Unit 테스트 목록 / Unit Test Files

| 파일 / File | 테스트 대상 / Target | 상태 / Status |
|---|---|---|
| `test_preprocessing.py` | `preprocess()` | ✅ |
| `test_augmentation.py` | `augment_supervised()`, `augment_contrastive()` | ✅ |
| `test_losses.py` | `InfoNCELoss`, `get_loss()` | ✅ |
| `test_metrics.py` | `compute_metrics()`, `build_evaluation_summary()` | ✅ |
| `test_models.py` | `ClassifierHead`, `ProjectionHead` | ✅ |
| `test_confusion.py` | `compute_confusion_matrix()` | ✅ |
| `test_utils_config.py` | `load_config()`, `validate_config()` | ✅ |
| `test_utils_model.py` | `set_seed()`, `backbone_tag()` | ✅ |

### 9.3 테스트 실행 결과 요약 / Test Run Summary

| 구분 / Category | PASS | FAIL | SKIP | 비고 / Note |
|---|---|---|---|---|
| Unit | — | — | — | 작성 예정 / TBD |
| Integration | — | — | — | 작성 예정 / TBD |
| Smoke | — | — | — | 실 데이터 필요 / Real data required |

> **⚠️ 환경 주의 / Environment Note**: NumPy 2.x 환경에서 `scikit-learn`, `scipy` 임포트 오류 발생 가능. `pip install "numpy<2"` 적용 후 실행 권장.
>
> `scikit-learn` and `scipy` may raise ImportError under NumPy 2.x. Run `pip install "numpy<2"` before executing tests.

---

## 10. ADR 기술 결정 / Architecture Decision Records

### 10.1 S3 기간 중 확정된 기술 결정 / Decisions Finalized During S3

| ADR | 제목 / Title | 결정 / Decision |
|---|---|---|
| ADR-001 | ImageNet 정규화 적용 / Apply ImageNet Normalization | `_IMAGENET_NORMALIZE`를 `dataset.py` 모듈 상수로 정의, 양쪽 Dataset에 적용 / Defined as module constant in `dataset.py`, applied to both Datasets |
| ADR-002 | NumPy 버전 제약 / NumPy Version Constraint | Anaconda 환경 호환성 — `numpy>=1.24.0,<2.0` 상한 고정 / Upper-bounded for Anaconda compatibility |
| ADR-003 | 테스트 3계층 구조 도입 / Introduce 3-Layer Test Structure | Unit / Integration / Smoke 분리 — `pytest.ini` + markers 관리 / Separated with `pytest.ini` + markers |

### 10.2 ADR-001 상세 / ADR-001 Detail — ImageNet 정규화 / ImageNet Normalization

| 항목 / Item | 내용 / Content |
|---|---|
| 결정 일자 / Date | 2026-05-08 |
| 결정자 / Decided by | R2 (Yang Jin-hyeong / 양진형) |
| 문제 / Problem | SSOT-NM01 — pretrained backbone 입력 분포 불일치로 성능 저하 가능 / Potential performance degradation due to backbone input distribution mismatch |
| 결정 / Decision | `torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`를 `_IMAGENET_NORMALIZE` 상수로 정의 후 `CMYKDataset`, `ContrastiveDataset` 양쪽 `__getitem__` 에 적용 / Defined as `_IMAGENET_NORMALIZE` constant and applied in both `CMYKDataset` and `ContrastiveDataset.__getitem__` |
| 기각된 대안 / Rejected Alternatives | ① 정규화 미적용 — pretrained 활용 효율 저하 / No normalization — reduces pretrained backbone efficiency ② `preprocessing.py`에 통합 — numpy/torch 경계 혼재 / Integrate into `preprocessing.py` — mixes numpy/torch boundary |
| 영향 범위 / Impact | `dataset.py`, `config.json`, `Contract.md`, `SSOT_Data_Pipeline.md`, `test_data_pipeline.py` |

### 10.3 ADR-002 상세 / ADR-002 Detail — NumPy 버전 제약 / NumPy Version Constraint

| 항목 / Item | 내용 / Content |
|---|---|
| 결정 일자 / Date | 2026-05-08 |
| 문제 / Problem | NumPy 2.x 설치 시 Anaconda base 패키지(pyarrow, pandas, scipy)가 NumPy 1.x 기준 컴파일되어 `ImportError` 발생 / NumPy 2.x causes `ImportError` in Anaconda base packages (pyarrow, pandas, scipy) compiled against NumPy 1.x |
| 결정 / Decision | `numpy>=1.24.0,<2.0` 버전 상한 고정 — `requirements.txt`, `pyproject.toml`, `dependencies.json` 일괄 반영 / Upper-bounded at `<2.0`, applied across `requirements.txt`, `pyproject.toml`, `dependencies.json` |

### 10.4 ADR-003 상세 / ADR-003 Detail — TDD 3계층 구조 / 3-Layer Test Structure

| 항목 / Item | 내용 / Content |
|---|---|
| 결정 일자 / Date | 2026-05-08 |
| 문제 / Problem | 기존 테스트 파일이 단일 디렉토리에 혼재 — 실행 범위와 데이터 의존성이 불명확 / Previous test files were mixed in a single directory, making execution scope and data dependencies unclear |
| 결정 / Decision | `unit/` (I/O 없음, < 1s / No I/O, < 1s) · `integration/` (모듈 연결 검증 / Module wiring) · `smoke/` (실 데이터 필요, `@pytest.mark.smoke` / Real data, `@pytest.mark.smoke`) 3계층으로 분리 / Separated into 3 layers |

---

## 11. 팀원 회고 / Team Retrospective

### 11.1 Koshoi (팀 리더 / Team Lead)

> 작성 예정 / To be filled in.

### 11.2 Jin-Hyeong Yang / 양진형 (R2) — 모델 & 학습 / Model & Training

> 작성 예정 / To be filled in.

### 11.3 Habin Ham / 함하빈 (R3) — 평가 & 리포팅 / Evaluation & Reporting

> 작성 예정 / To be filled in.

### 11.4 Jeahwan Lee / 이재환 (R4) — 튜닝 & 최적화 / Tuning & Optimization

> 작성 예정 / To be filled in.

### 11.5 Rackhel (R6) — 통합 & 인프라 / Integration & Infrastructure

> 작성 예정 / To be filled in.

---

**Version**: 0.1.0 (초안 / Draft)
**Last Updated**: 2026-05-11
**Applies to**: CMYK Grayspot Detection System — Sprint S3
