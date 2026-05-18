---
type: contract
domain: artifact_boundary
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] Artifact Boundary — 아티팩트 저장/로드 경계 계약 / Artifact Save/Load Boundary Contract

> **목적 / Purpose**: 체크포인트, 학습 이력, config 스냅샷의 저장/로드 형식과 생산자-소비자 관계를 정의한다. / Defines the save/load format and producer-consumer relationships for checkpoints, training history, and config snapshots.
> **상태 / Status**: ✅ Accepted [Hard]
> **작성일 / Created**: 2026-05-17
> **관련 문서 / Related Docs**:
>
> - [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) (아티팩트 목록 및 명명 패턴 / Artifact list and naming patterns)
> - [SSOT_Validation_Codes.md](../SSOT/SSOT_Validation_Codes.md) (SSOT-FF01 정의 / SSOT-FF01 definition)

> 🔒 **SSOT 경계 원칙 / SSOT Boundary Principle**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다. 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.
> / This document does not redefine SSOT semantic definitions. Follow SSOT_Core.md as the final authority for semantic interpretation.

---

## 1. 계약 목적 / Contract Purpose

모든 아티팩트(체크포인트, 이력, config 스냅샷)의 저장 형식, 경로 패턴, 생산자-소비자 관계를 정의한다.

Defines the storage format, path patterns, and producer-consumer relationships for all artifacts (checkpoints, history, config snapshots).

---

## 2. Phase 0 체크포인트 계약 / Phase 0 Checkpoint Contract

| 항목 / Item | 값 / Value |
| --- | --- |
| 경로 패턴 / Path Pattern | `{storage.models_dir}/phase0_backbone_{channel}_{tag}.pt` |
| 형식 / Format | `torch.save(model.state_dict())` |
| 포함 키 / Included Keys | `backbone.*` + `head.*` (ProjectionHead) |
| 생산자 / Producer | `Phase0Trainer.save_backbone()` |
| 소비자 / Consumer | `GrayspotModel.switch_to_phase2()` — `backbone.*` 키만 선택 로드 / Selectively loads only `backbone.*` keys |

---

## 3. Phase 2 체크포인트 계약 (`best_{ch}.pt`) / Phase 2 Checkpoint Contract

| 항목 / Item | 값 / Value |
| --- | --- |
| 경로 패턴 / Path Pattern | `{storage.models_dir}/best_{channel}.pt` |
| 형식 / Format | `torch.save(model.state_dict())` |
| 포함 키 / Included Keys | `backbone.*` + `head.*` (ClassifierHead) |
| 저장 기준 / Save Criterion | `val_acc > best_val_acc + early_stopping.min_delta` |
| 생산자 / Producer | `Phase2Trainer.train()` 내부 / internally |
| 소비자 / Consumer | `Evaluator`, `GrayspotPredictor.load_model()` |

---

## 4. 추론 시 로드 계약 / Inference Load Contract

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint, strict=False)
```

| 옵션 / Option | 보장 / Guarantee |
| --- | --- |
| `weights_only=True` | pickle 보안 — 임의 코드 실행 방지 / Pickle safety — prevents arbitrary code execution |
| `strict=False` | 버전 간 키 불일치 허용 / Allows cross-version key mismatches |
| `map_location="cpu"` | GPU 부재 환경에서도 로드 가능 / Loadable in environments without GPU |

---

## 5. 학습 이력 계약 / Training History Contract

### 5.1 Phase 0 이력 / Phase 0 History

| 항목 / Item | 값 / Value |
| --- | --- |
| 경로 / Path | `{models_dir}/phase0_history_{channel}_{tag}.json` |
| 형식 / Format | JSON — `List[{"epoch": int, "loss": float}]` |
| 생산자 / Producer | `Phase0Trainer.train()` |

### 5.2 Phase 2 이력 / Phase 2 History

| 항목 / Item | 값 / Value |
| --- | --- |
| 경로 / Path | `{reports_dir}/phase2_history_{channel}_{tag}.csv` |
| 형식 / Format | CSV — epoch, train_loss, val_acc, val_f1, ... |
| 생산자 / Producer | `Phase2Trainer.save_history()` |

---

## 6. Config 스냅샷 계약 / Config Snapshot Contract

| 항목 / Item | 값 / Value |
| --- | --- |
| 경로 / Path | `{models_dir}/phase2_config_{channel}_{tag}.yaml` |
| 형식 / Format | YAML — 학습 시점 config 전체 / Full config at training time |
| 생산자 / Producer | `Phase2Trainer.train()` |
| 용도 / Purpose | 재현성 보장 — 어떤 config로 학습했는지 추적 / Reproducibility — tracks which config was used for training |

---

## 7. 생산자-소비자 요약 / Producer-Consumer Summary

| 아티팩트 / Artifact | 생산자 / Producer | 소비자 / Consumer | 쓰기 권한 / Write Permission |
| --- | --- | --- | --- |
| `phase0_backbone_*.pt` | Phase0Trainer | GrayspotModel.switch_to_phase2 | ✅ Trainer만 / Trainer only |
| `best_{ch}.pt` | Phase2Trainer | Evaluator, GrayspotPredictor | ✅ Trainer만 / Trainer only |
| `phase0_history_*.json` | Phase0Trainer | 리포팅 / Reporting | ✅ Trainer만 / Trainer only |
| `phase2_history_*.csv` | Phase2Trainer | 리포팅, GUI / Reporting, GUI | ✅ Trainer만 / Trainer only |
| `phase2_config_*.yaml` | Phase2Trainer | 재현성 감사 / Reproducibility audit | ✅ Trainer만 / Trainer only |
| `study_*.db` | OptunaTuner | 분석 / Analysis | ✅ Tuner만 / Tuner only |

> ❌ **금지 / Prohibited**: Evaluator나 Predictor가 체크포인트를 쓰는 것은 금지. 읽기 전용. / Evaluator or Predictor writing checkpoints is prohibited. Read-only access.

---

## 8. Fail-Fast 검증 포인트 / Fail-Fast Verification Points

| 위치 / Location | 조건 / Condition | 코드 / Code | 예외 / Exception |
| --- | --- | --- | --- |
| `run_phase2.py` 시작 / startup | `phase0_backbone_*.pt` 미존재 / not found | SSOT-FF01 | `FileNotFoundError` |
| `Evaluator.run()` 시작 / startup | `best_{ch}.pt` 미존재 / not found | SSOT-FF01 | `FileNotFoundError` |
| `switch_to_phase2()` | backbone 키 0개 로드 / 0 backbone keys loaded | SSOT-FF01 | `RuntimeError` |

---

## 9. ONNX 산출물 계약 / ONNX Artifact Contract

### export_to_onnx()

```python
from inference.onnx_export import export_to_onnx

output_path = export_to_onnx(
    checkpoint_path="outputs/models/best_Y.pt",
    output_path="outputs/onnx/model_Y_effb0.onnx",
    cfg=cfg,
    opset_version=17,
)
```

| 항목 / Item | 값 / Value |
| --- | --- |
| 입력 shape / Input shape | `(1, 3, 128, 128)` float32 |
| 출력 shape / Output shape | `(1, 6)` float32 |
| 파일명 패턴 / Filename Pattern | `model_{channel}_{backbone_tag}.onnx` |
| 저장 위치 / Save Location | `outputs/onnx/` |
| 검증 / Validation | `onnx.checker.check_model()` 내부 실행 / executed internally |

---

## 10. 체크리스트 / Checklist

- [x] `weights_only=True` 로드 확인 / Verify `weights_only=True` load
- [x] `strict=False` 로드 확인 / Verify `strict=False` load
- [x] `map_location="cpu"` 로드 확인 / Verify `map_location="cpu"` load
- [x] 아티팩트 생산자-소비자 분리 확인 / Verify artifact producer-consumer separation
- [ ] Optuna 산출물 경로 패턴 일관성 확인 / Verify Optuna artifact path pattern consistency

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) | 아티팩트 명명 및 스키마 (What) / Artifact naming and schema |
| [Contract_training_pipeline.md](Contract_training_pipeline.md) | Trainer 생산자 / Trainer producer |
| [Contract_evaluation_reporting.md](Contract_evaluation_reporting.md) | Evaluator 소비자 / Evaluator consumer |
| [Contract_inference_boundary.md](Contract_inference_boundary.md) | Predictor 소비자 / Predictor consumer |
