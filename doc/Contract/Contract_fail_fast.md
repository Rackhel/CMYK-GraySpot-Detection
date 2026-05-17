---
type: contract
domain: fail_fast
status: Active
last_updated: 2026-05-17
owner: CMYK WooSong Team
---

# [Contract] Fail-Fast Enforcement — Fail-Fast 집행 포인트 계약

> **목적**: 시스템 전체 Fail-Fast 조건, SSOT 검증 코드, 예외 타입을 정의한다.
> **상태**: ✅ Accepted [Hard]
> **작성일**: 2026-05-17
> **관련 문서**:
>
> - [SSOT_Validation_Codes.md](../SSOT/SSOT_Validation_Codes.md) (검증 코드 전수 목록)
> - [SSOT_Core.md](../SSOT/SSOT_Core.md) (Hard/Soft 판단 기준)

> 🔒 **SSOT 경계 원칙**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다.
> 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.

---

## 1. 계약 목적

모든 모듈 경계에서 **즉시 예외를 발생시켜야 하는** Fail-Fast 조건을 중앙 집중 정의한다.
우회·임시 생성·기본값 대체 금지.

---

## 2. SSOT 검증 코드 정의

| 코드 | 의미 | 예외 타입 |
| --- | --- | --- |
| `SSOT-FF01` | 파일/아티팩트 미존재 | `FileNotFoundError` / `RuntimeError` |
| `SSOT-CF01` | Config 키 누락 또는 타입 불일치 | `ValueError` / `KeyError` |
| `SSOT-CS01` | 색상 공간 위반 (BGR 아닌 RGB 사용) | `ValueError` |
| `SSOT-NM01` | ImageNet 정규화 미적용 | 런타임 성능 저하 (Hard 경고) |

---

## 3. 전체 Fail-Fast 집행 포인트

| 위치 | 조건 | 코드 | 예외 |
| --- | --- | --- | --- |
| `run_phase2.py` 시작 | `phase0_backbone_{ch}_{tag}.pt` 미존재 | SSOT-FF01 | `FileNotFoundError` |
| `Evaluator.run()` 시작 | `best_{ch}.pt` 미존재 | SSOT-FF01 | `FileNotFoundError` |
| `GrayspotPredictor.load_model()` | 모델 파일 없음 | SSOT-FF01 | `FileNotFoundError` |
| `GrayspotPredictor.__init__()` | config.json 없음 | SSOT-FF01 | `FileNotFoundError` |
| `load_best_params()` | `best_params_{ch}.json` 미존재 | SSOT-FF01 | `FileNotFoundError` |
| `switch_to_phase2()` | backbone 키 0개 로드 | SSOT-FF01 | `RuntimeError` |
| `GrayspotModel.__init__` | `cfg["data"]["num_levels"]` 키 미존재 | SSOT-CF01 | `KeyError` |
| `validate_config()` | 필수 섹션 누락 (`data`, `model`, `phase2` 등) | SSOT-CF01 | `ValueError` |
| `validate_config()` | 키 타입 불일치 | SSOT-CF01 | `ValueError` |
| `Phase0Trainer.train()` | DataLoader 배치 언패킹 실패 (형상 불일치) | SSOT-CS01 | `ValueError` |
| `normalize_channel()` | `VALID_CHANNELS` 외 채널 입력 | — | `ValueError` |
| `apply_phase2_params()` | 필수 params 키 누락 | — | `KeyError` |

---

## 4. 금지 패턴

```python
# ❌ 파일 미존재 시 기본값 생성 — Fail-Fast 위반
if not path.exists():
    model = GrayspotModel(cfg, phase=2)  # 학습 안 된 모델 반환 금지

# ❌ Config 키 부재 시 None 반환 후 계속 진행
device = cfg.get("system", {}).get("device", None)
if device is None:
    device = "cpu"  # 조용한 기본값 — 오류 은폐

# ✅ 올바른 패턴
if not path.exists():
    raise FileNotFoundError(f"[SSOT-FF01] {path} not found")

validate_config(cfg)  # 키 누락 시 즉시 ValueError
```

---

## 5. Hard vs Soft 판단 기준

| 판단 | 기준 | 적용 코드 |
| --- | --- | --- |
| **Hard** | 위반 시 학습/추론 결과가 무효화됨 | SSOT-FF01, SSOT-CF01, SSOT-CS01 |
| **Soft** | 위반 시 성능 저하 가능, 경고 수준 | SSOT-NM01 |

> Hard 위반은 반드시 즉시 예외. Soft 위반은 경고 로그 + 실행 계속 허용.

---

## 6. 체크리스트

- [x] 모든 아티팩트 로드 전 존재 확인 후 `FileNotFoundError`
- [x] `validate_config()` — 필수 섹션 누락 시 `ValueError`
- [x] `switch_to_phase2()` — backbone 키 0개 시 `RuntimeError`
- [x] BGR 유지 경계에서 `ValueError` (SSOT-CS01)
- [ ] `InfoNCELoss` L2-정규화 미적용 시 명시적 검증 추가

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_Validation_Codes.md](../SSOT/SSOT_Validation_Codes.md) | 검증 코드 전수 목록 (What) |
| [Contract_config_resolution.md](Contract_config_resolution.md) | SSOT-CF01 발생 위치 |
| [Contract_artifact_boundary.md](Contract_artifact_boundary.md) | SSOT-FF01 발생 위치 |
| [Contract_data_pipeline.md](Contract_data_pipeline.md) | SSOT-CS01 발생 위치 |
