---
type: ssot
domain: validation_codes
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Core.md"
---

# SSOT Validation Codes — 검증 에러 코드 정의 / Validation Error Code Definitions

CMYK Grayspot Detection System 의 모든 Fail-Fast 에러 코드와 처리 방법에 관한 단일 진실 공급원.

This document is the authoritative reference for all Fail-Fast error codes, their conditions, and remediation steps.

> **목적 / Purpose**: 검증 에러 코드의 의미와 처리 방법 정의
> **역할 / Role**: "What" — SSOT 위반 코드 목록, 감지 조건, 등급
> **관련 문서 / See also**: [SSOT_Core.md](SSOT_Core.md), [SSOT_GlobalVariables.md](SSOT_GlobalVariables.md)

---

## 1. 코드 명명 규칙 / Code Naming Convention

```
SSOT-{PREFIX}{NUM}

PREFIX: 위반 카테고리 약어 / Violation category abbreviation
  FF = Fail-Fast artifact (산출물 누락 / Artifact missing)
  PH = Phase (단계 순서 위반 / Phase ordering violation)
  CS = Color Space (색상 공간 불일치 / Color space mismatch)
  NM = Normalization Mismatch (정규화 불일치 / Normalization mismatch)
  CF = Config (설정 키 위반 / Config key violation)
  SD = Seed / Determinism (재현성 위반 / Reproducibility violation)

NUM: 순번 (01, 02, ...) / Sequential number
```

---

## 2. 등급 체계 / Severity Levels

| 등급 / Level | 이름 / Name | 동작 / Action | 예시 / Example |
|------|------|------|------|
| **Level 0** | Panic | 시스템 즉시 중단 / Immediate system halt | 모델 파일 손상, CUDA OOM / Corrupt model file, CUDA OOM |
| **Level 1** | Error | 현재 실행 중단, 명시적 에러 반환 / Abort with explicit error | SSOT-FF01, SSOT-PH01, SSOT-CS01 |
| **Level 2** | Warning | 경고 로그 + 실행 계속 / Log warning, continue | SSOT-CF02, SSOT-NM01, SSOT-SD01 |
| **Level 3** | Info | 기록만, 실행 계속 / Log only | 권장값과 다른 설정 / Settings different from recommended values |

---

## 3. 코드 전체 목록 / Complete Code List

| 코드 / Code | 카테고리 / Category | 등급 / Level | 한 줄 요약 / One-line Summary |
|---|---|---|---|
| `SSOT-FF01` | Artifact | Level 1 | 필수 아티팩트 파일 누락 / Required artifact file missing |
| `SSOT-PH01` | Phase | Level 1 | Phase 순서 위반 / Phase ordering violation |
| `SSOT-CS01` | Color Space / 색상 공간 | Level 1 | BGR/RGB 색상 공간 불일치 / Color space mismatch |
| `SSOT-NM01` | Normalization / 정규화 | Level 2 | ImageNet 정규화 미적용 / ImageNet normalization not applied |
| `SSOT-CF01` | Config | Level 1 | 코드 참조 config 키 누락 / Config key referenced but missing |
| `SSOT-CF02` | Config | Level 2 | Dead Config 감지 / Dead config key detected |
| `SSOT-SD01` | Seed | Level 2 | 재현성 seed 미설정 / Reproducibility seed not enforced |

---

## 4. 코드별 상세 정의 / Detailed Definitions

### SSOT-FF01 — 필수 아티팩트 누락 / Missing Required Artifact

| 항목 / Item | 내용 / Detail |
|---|---|
| 등급 / Level | Level 1 — Error |
| 트리거 조건 / Trigger | Phase 0 backbone 없이 Phase 2 시작, `best_{ch}.pt` 없이 평가 시작 / Starting Phase 2 without Phase 0 backbone, starting evaluation without `best_{ch}.pt` |
| 감지 위치 / Detection | `Phase2Trainer.__init__()`, `GrayspotModel.switch_to_phase2()`, `Evaluator.__init__()` |
| 즉각 조치 / Immediate Action | `raise FileNotFoundError(f"[SSOT-FF01] {path}")` |
| 해결 방법 / Fix | Phase 0 학습 완료 후 Phase 2 진행. 평가 전 Phase 2 학습 완료 확인 / Complete Phase 0 training before starting Phase 2. Verify Phase 2 training is complete before evaluation. |

---

### SSOT-PH01 — Phase 순서 위반 / Phase Ordering Violation

| 항목 / Item | 내용 / Detail |
|---|---|
| 등급 / Level | Level 1 — Error |
| 트리거 조건 / Trigger | Phase 0 학습 없이 Phase 2 직행, backbone 미학습 상태에서 `switch_to_phase2()` 호출 / Going directly to Phase 2 without Phase 0 training, calling `switch_to_phase2()` with untrained backbone |
| 감지 위치 / Detection | `run_phase2.py` 시작 부분, `Phase2Trainer.__init__()` / Start of `run_phase2.py`, `Phase2Trainer.__init__()` |
| 즉각 조치 / Immediate Action | 즉시 중단 + 명시적 에러 메시지 / Immediate halt + explicit error message |
| 해결 방법 / Fix | `python -m src.scripts.run_phase0` 먼저 실행 후 Phase 2 진행 / Run `python -m src.scripts.run_phase0` first, then proceed to Phase 2 |

---

### SSOT-CS01 — 색상 공간 불일치 / Color Space Mismatch

| 항목 / Item | 내용 / Detail |
|---|---|
| 등급 / Level | Level 1 — Error |
| 트리거 조건 / Trigger | 학습 시 BGR 사용, 평가/추론 시 RGB 사용 / BGR used during training, RGB used during evaluation/inference |
| 감지 위치 / Detection | `Evaluator`, `GrayspotPredictor` 입력 전처리 단계 / Input preprocessing stage of `Evaluator` and `GrayspotPredictor` |
| 즉각 조치 / Immediate Action | 색상 공간 강제 확인 후 불일치 시 중단 / Force color space check and halt on mismatch |
| 해결 방법 / Fix | `cv2.imread()` 사용 → BGR 유지. PIL/torchvision 사용 시 BGR 변환 명시 / Use `cv2.imread()` to keep BGR. Explicitly convert to BGR when using PIL/torchvision. |

적용 범위 / Scope: `data/dataset.py`, `data/augmentation.py`, `inference/predictor_inference.py`

---

### SSOT-NM01 — 정규화 기준 불일치 / Normalization Mismatch

| 항목 / Item | 내용 / Detail |
|---|---|
| 등급 / Level | Level 2 — Warning |
| 트리거 조건 / Trigger | Pretrained backbone 사용 시 ImageNet mean/std 정규화 미적용 / ImageNet mean/std normalization not applied when using pretrained backbone |
| SSOT 원천 / SSOT Source | `data/normalize.py` — `_IMAGENET_NORMALIZE` 단일 정의 / Single definition of `_IMAGENET_NORMALIZE` |
| 정규화 값 / Values | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |
| 즉각 조치 / Immediate Action | 경고 로그 출력 후 계속 실행 / Log warning and continue execution |
| 해결 방법 / Fix | `data.normalize._IMAGENET_NORMALIZE`를 모든 소비 모듈에서 import / Import `_IMAGENET_NORMALIZE` from `data.normalize` in all consuming modules |

---

### SSOT-CF01 — config 키 미존재 / Missing Config Key

| 항목 / Item | 내용 / Detail |
|---|---|
| 등급 / Level | Level 1 — Error |
| 트리거 조건 / Trigger | 코드에서 `cfg["phase0"]["temperature"]` 등 참조하는 키가 `config.json`에 없음 / Key referenced in code (e.g., `cfg["phase0"]["temperature"]`) is absent from `config.json` |
| 감지 위치 / Detection | `cfg["key"]` 또는 `get_nested(cfg, "key")` 접근 시 `KeyError`, 각 모듈의 config 접근 시점 / `KeyError` on `cfg["key"]` or `get_nested(cfg, "key")` access, at each module's config access point |
| 즉각 조치 / Immediate Action | `KeyError` 즉시 발생 + 에러 메시지에 코드 SSOT-CF01 명시 / Raise `KeyError` immediately + include code SSOT-CF01 in error message |
| 해결 방법 / Fix | `config.json`에 키 추가 또는 코드의 잘못된 키 참조 수정 / Add key to `config.json` or fix the incorrect key reference in code |

`validate_config(cfg) → None`: 필수 키 부재 시 `ValueError("[CONFIG ERROR / SSOT-CF01] …")` 발생. 반환값 없음 (bool 아님).

---

### SSOT-CF02 — Dead Config 감지 / Dead Config Detected

| 항목 / Item | 내용 / Detail |
|---|---|
| 등급 / Level | Level 2 — Warning |
| 트리거 조건 / Trigger | `config.json`에 선언된 키가 코드에서 소비되지 않음 / Key declared in `config.json` is not consumed by code |
| 즉각 조치 / Immediate Action | Warning 로그 출력, 실행 계속 / Log warning, continue execution |
| 해결 방법 / Fix | 해당 키 기능 구현 또는 `config.json`에서 제거 / Implement the key's feature or remove it from `config.json` |

상세 목록 / Full list: [SSOT_GlobalVariables.md §5](SSOT_GlobalVariables.md)

---

### SSOT-SD01 — Seed 재현성 실패 / Seed Reproducibility Failure

| 항목 / Item | 내용 / Detail |
|---|---|
| 등급 / Level | Level 2 — Warning |
| 트리거 조건 / Trigger | 동일 seed에서 다른 데이터 분할 또는 학습 결과 / Same seed produces different data splits or training results |
| 즉각 조치 / Immediate Action | Warning 로그 출력 / Log warning |
| 해결 방법 / Fix | `utils.set_seed(seed, cfg)` 호출 — `random`, `numpy`, `torch`, `cuda.deterministic`, `cuda.benchmark` 모두 설정 |

`set_seed()` 적용 범위: `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic`, `cudnn.benchmark`

---

## 5. 감지 가이드 / Detection Guide

### 5.1 자동 감지 체크리스트 / Automated Detection Checklist

Phase 2 시작 전 확인 사항 / Before starting Phase 2:

| 항목 / Item | 검증 대상 / Target | 코드 / Code |
|---|---|---|
| Phase 0 backbone 존재 여부 | `{models_dir}/phase0_backbone_{ch}_{tag}.pt` | SSOT-FF01 / PH01 |
| config 필수 키 / Required config keys | `validate_config(cfg)` — ValueError 발생 시 중단 | SSOT-CF01 |
| 색상 공간 / Color space | 이미지 로드 모듈이 `cv2.imread()` 사용 여부 | SSOT-CS01 |
| 정규화 / Normalization | `_IMAGENET_NORMALIZE`가 `data.normalize`에서 import 되는지 여부 | SSOT-NM01 |

### 5.2 Dead Config 감지 / Dead Config Detection

Dead Config 목록은 [SSOT_GlobalVariables.md §5](SSOT_GlobalVariables.md)에서 관리된다.
Dead Config list is maintained in [SSOT_GlobalVariables.md §5](SSOT_GlobalVariables.md).

---

## 체크리스트 / Checklist

- [ ] 새 에러 코드 추가 시 §3 목록 + §4 상세 정의 작성 / Add to §3 list + §4 detailed definition when adding new error code
- [ ] 에러 코드 해소 시 §3에 날짜 기록 / Record resolution date in §3 when error code is resolved
- [ ] Level 1 코드 해소 시 관련 SSOT 문서 동기화 / Sync related SSOT documents when Level 1 code is resolved

---

## See Also

| 문서 / Document | 관계 / Relation |
| --- | --- |
| [SSOT_Core.md](SSOT_Core.md) | Fail-Fast 정책 정의 / Fail-Fast policy definition |
| [SSOT_Data_Pipeline.md](SSOT_Data_Pipeline.md) | CS01, NM01, SD01 발생 지점 / CS01, NM01, SD01 occurrence points |
| [SSOT_Artifacts.md](SSOT_Artifacts.md) | FF01 검증 대상 / FF01 validation targets |

