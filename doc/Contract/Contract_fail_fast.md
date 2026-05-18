---
type: contract
domain: fail_fast
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] Fail-Fast Enforcement — Fail-Fast 집행 포인트 계약 / Fail-Fast Enforcement Point Contract

> **목적 / Purpose**: 시스템 전체 Fail-Fast 조건, SSOT 검증 코드, 예외 타입을 정의한다. / Defines system-wide Fail-Fast conditions, SSOT validation codes, and exception types.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-17
> **관련 문서 / Related Docs**:
>
> - [SSOT_Validation_Codes.md](../SSOT/SSOT_Validation_Codes.md) (검증 코드 전수 목록 / Full list of validation codes)
> - [SSOT_Core.md](../SSOT/SSOT_Core.md) (Hard/Soft 판단 기준 / Hard/Soft decision criteria)

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.
> / This document does not redefine SSOT semantic definitions. Follow SSOT_Core.md as the final authority for semantic interpretation.

---

## 1. 계약 목적 / Contract Purpose

모든 모듈 경계에서 **즉시 예외를 발생시켜야 하는** Fail-Fast 조건을 중앙 집중 정의한다.
우회·임시 생성·기본값 대체 금지.

Centrally defines Fail-Fast conditions that **must immediately raise exceptions** at all module boundaries.
Bypassing, temporary creation, and silent default substitution are prohibited.

---

## 2. SSOT 검증 코드 정의 / SSOT Validation Code Definitions

| 코드 / Code | 의미 / Meaning | 예외 타입 / Exception Type |
| --- | --- | --- |
| `SSOT-FF01` | 파일/아티팩트 미존재 / File/artifact not found | `FileNotFoundError` / `RuntimeError` |
| `SSOT-CF01` | Config 키 누락 또는 타입 불일치 / Config key missing or type mismatch | `ValueError` / `KeyError` |
| `SSOT-CS01` | 색상 공간 위반 (BGR 아닌 RGB 사용) / Color space violation (RGB used instead of BGR) | `ValueError` |
| `SSOT-NM01` | ImageNet 정규화 미적용 / ImageNet normalization not applied | 런타임 성능 저하 (Hard 경고) / Runtime performance degradation (Hard warning) |

---

## 3. 전체 Fail-Fast 집행 포인트 / All Fail-Fast Enforcement Points

| 위치 / Location | 조건 / Condition | 코드 / Code | 예외 / Exception |
| --- | --- | --- | --- |
| `run_phase2.py` 시작 / startup | `phase0_backbone_{ch}_{tag}.pt` 미존재 / not found | SSOT-FF01 | `FileNotFoundError` |
| `Evaluator.run()` 시작 / startup | `best_{ch}.pt` 미존재 / not found | SSOT-FF01 | `FileNotFoundError` |
| `GrayspotPredictor.load_model()` | 모델 파일 없음 / model file missing | SSOT-FF01 | `FileNotFoundError` |
| `GrayspotPredictor.__init__()` | config.json 없음 / config.json missing | SSOT-FF01 | `FileNotFoundError` |
| `load_best_params()` | `best_params_{ch}.json` 미존재 / not found | SSOT-FF01 | `FileNotFoundError` |
| `switch_to_phase2()` | backbone 키 0개 로드 / 0 backbone keys loaded | SSOT-FF01 | `RuntimeError` |
| `GrayspotModel.__init__` | `cfg["data"]["num_levels"]` 키 미존재 / key missing | SSOT-CF01 | `KeyError` |
| `validate_config()` | 필수 섹션 누락 (`data`, `model`, `phase2` 등) / Required sections missing | SSOT-CF01 | `ValueError` |
| `validate_config()` | 키 타입 불일치 / Key type mismatch | SSOT-CF01 | `ValueError` |
| `Phase0Trainer.train()` | DataLoader 배치 언패킹 실패 (형상 불일치) / DataLoader batch unpack failure (shape mismatch) | SSOT-CS01 | `ValueError` |
| `normalize_channel()` | `VALID_CHANNELS` 외 채널 입력 / Channel outside VALID_CHANNELS | — | `ValueError` |
| `apply_phase2_params()` | 필수 params 키 누락 / Required params key missing | — | `KeyError` |

---

## 4. 금지 패턴 / Prohibited Patterns

```python
# ❌ 파일 미존재 시 기본값 생성 — Fail-Fast 위반
# / Creating default value when file missing — Fail-Fast violation
if not path.exists():
    model = GrayspotModel(cfg, phase=2)  # 학습 안 된 모델 반환 금지 / Never return untrained model

# ❌ Config 키 부재 시 None 반환 후 계속 진행
# / Returning None and continuing when config key is absent
device = cfg.get("system", {}).get("device", None)
if device is None:
    device = "cpu"  # 조용한 기본값 — 오류 은폐 / Silent default — hides error

# ✅ 올바른 패턴 / Correct pattern
if not path.exists():
    raise FileNotFoundError(f"[SSOT-FF01] {path} not found")

validate_config(cfg)  # 키 누락 시 즉시 ValueError / Immediately raises ValueError if key missing
```

---

## 5. Hard vs Soft 판단 기준 / Hard vs Soft Decision Criteria

| 판단 / Decision | 기준 / Criterion | 적용 코드 / Applicable Code |
| --- | --- | --- |
| **Hard** | 위반 시 학습/추론 결과가 무효화됨 / Violation invalidates training/inference results | SSOT-FF01, SSOT-CF01, SSOT-CS01 |
| **Soft** | 위반 시 성능 저하 가능, 경고 수준 / Violation may degrade performance, warning level | SSOT-NM01 |

> Hard 위반은 반드시 즉시 예외. Soft 위반은 경고 로그 + 실행 계속 허용.
> / Hard violations must raise exceptions immediately. Soft violations allow warning log + continued execution.

---

## 6. 체크리스트 / Checklist

- [x] 모든 아티팩트 로드 전 존재 확인 후 `FileNotFoundError` / Verify existence before loading all artifacts, raise `FileNotFoundError`
- [x] `validate_config()` — 필수 섹션 누락 시 `ValueError` / Raises `ValueError` on missing required sections
- [x] `switch_to_phase2()` — backbone 키 0개 시 `RuntimeError` / Raises `RuntimeError` on 0 backbone keys
- [x] BGR 유지 경계에서 `ValueError` (SSOT-CS01) / `ValueError` at BGR maintenance boundaries
- [ ] `InfoNCELoss` L2-정규화 미적용 시 명시적 검증 추가 / Add explicit validation when InfoNCELoss L2-normalization is not applied

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Validation_Codes.md](../SSOT/SSOT_Validation_Codes.md) | 검증 코드 전수 목록 (What) / Full validation code list |
| [Contract_config_resolution.md](Contract_config_resolution.md) | SSOT-CF01 발생 위치 / SSOT-CF01 occurrence locations |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | SSOT-FF01 발생 위치 / SSOT-FF01 occurrence locations |
| [Contract_data_pipeline.md](Contract_data_pipeline.md) | SSOT-CS01 발생 위치 / SSOT-CS01 occurrence locations |
