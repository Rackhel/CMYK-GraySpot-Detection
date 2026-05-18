---
type: contract
domain: artifact_boundary
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
---

# [Contract] Artifact Boundary — 아티팩트 저장/로드 경계 계약

> **목적**: 체크포인트, 학습 이력, config 스냅샷의 저장/로드 형식과 생산자-소비자 관계를 정의한다.
> **상태**: ✅ Accepted [Hard]
> **작성일**: 2026-05-17
> **관련 문서**:
>
> - [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) (아티팩트 목록 및 명명 패턴)
> - [SSOT_Validation_Codes.md](../SSOT/SSOT_Validation_Codes.md) (SSOT-FF01 정의)

> 🔒 **SSOT 경계 원칙**: 본 문서는 SSOT 문서의 의미 정의를 재정의하지 않는다.
> 의미적 해석이 필요한 경우 [SSOT_Core.md](../SSOT/SSOT_Core.md)를 최종 판결자로 따른다.

---

## 1. 계약 목적

모든 아티팩트(체크포인트, 이력, config 스냅샷)의 저장 형식, 경로 패턴, 생산자-소비자 관계를 정의한다.

---

## 2. Phase 0 체크포인트 계약

| 항목 | 값 |
| --- | --- |
| 경로 패턴 | `{storage.models_dir}/phase0_backbone_{channel}_{tag}.pt` |
| 형식 | `torch.save(model.state_dict())` |
| 포함 키 | `backbone.*` + `head.*` (ProjectionHead) |
| 생산자 | `Phase0Trainer.save_backbone()` |
| 소비자 | `GrayspotModel.switch_to_phase2()` — `backbone.*` 키만 선택 로드 |

---

## 3. Phase 2 체크포인트 계약 (`best_{ch}.pt`)

| 항목 | 값 |
| --- | --- |
| 경로 패턴 | `{storage.models_dir}/best_{channel}.pt` |
| 형식 | `torch.save(model.state_dict())` |
| 포함 키 | `backbone.*` + `head.*` (ClassifierHead) |
| 저장 기준 | `val_acc > best_val_acc + early_stopping.min_delta` |
| 생산자 | `Phase2Trainer.train()` 내부 |
| 소비자 | `Evaluator`, `GrayspotPredictor.load_model()` |

---

## 4. 추론 시 로드 계약

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint, strict=False)
```

| 옵션 | 보장 |
| --- | --- |
| `weights_only=True` | pickle 보안 — 임의 코드 실행 방지 |
| `strict=False` | 버전 간 키 불일치 허용 |
| `map_location="cpu"` | GPU 부재 환경에서도 로드 가능 |

---

## 5. 학습 이력 계약

### 5.1 Phase 0 이력

| 항목 | 값 |
| --- | --- |
| 경로 | `{models_dir}/phase0_history_{channel}_{tag}.json` |
| 형식 | JSON — `List[{"epoch": int, "loss": float}]` |
| 생산자 | `Phase0Trainer.train()` |

### 5.2 Phase 2 이력

| 항목 | 값 |
| --- | --- |
| 경로 | `{reports_dir}/phase2_history_{channel}_{tag}.csv` |
| 형식 | CSV — epoch, train_loss, val_acc, val_f1, ... |
| 생산자 | `Phase2Trainer.save_history()` |

---

## 6. Config 스냅샷 계약

| 항목 | 값 |
| --- | --- |
| 경로 | `{models_dir}/phase2_config_{channel}_{tag}.yaml` |
| 형식 | YAML — 학습 시점 config 전체 |
| 생산자 | `Phase2Trainer.train()` |
| 용도 | 재현성 보장 — 어떤 config로 학습했는지 추적 |

---

## 7. 생산자-소비자 요약

| 아티팩트 | 생산자 | 소비자 | 쓰기 권한 |
| --- | --- | --- | --- |
| `phase0_backbone_*.pt` | Phase0Trainer | GrayspotModel.switch_to_phase2 | ✅ Trainer만 |
| `best_{ch}.pt` | Phase2Trainer | Evaluator, GrayspotPredictor | ✅ Trainer만 |
| `phase0_history_*.json` | Phase0Trainer | 리포팅 | ✅ Trainer만 |
| `phase2_history_*.csv` | Phase2Trainer | 리포팅, GUI | ✅ Trainer만 |
| `phase2_config_*.yaml` | Phase2Trainer | 재현성 감사 | ✅ Trainer만 |
| `study_*.db` | OptunaTuner | 분석 | ✅ Tuner만 |

> ❌ **금지**: Evaluator나 Predictor가 체크포인트를 쓰는 것은 금지. 읽기 전용.

---

## 8. Fail-Fast 검증 포인트

| 위치 | 조건 | 코드 | 예외 |
| --- | --- | --- | --- |
| `run_phase2.py` 시작 | `phase0_backbone_*.pt` 미존재 | SSOT-FF01 | `FileNotFoundError` |
| `Evaluator.run()` 시작 | `best_{ch}.pt` 미존재 | SSOT-FF01 | `FileNotFoundError` |
| `switch_to_phase2()` | backbone 키 0개 로드 | SSOT-FF01 | `RuntimeError` |

---

## 9. ONNX 산출물 계약

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

| 항목 | 값 |
| --- | --- |
| 입력 shape | `(1, 3, 128, 128)` float32 |
| 출력 shape | `(1, 6)` float32 |
| 파일명 패턴 | `model_{channel}_{backbone_tag}.onnx` |
| 저장 위치 | `outputs/onnx/` |
| 검증 | `onnx.checker.check_model()` 내부 실행 |

---

## 10. 체크리스트

- [x] `weights_only=True` 로드 확인
- [x] `strict=False` 로드 확인
- [x] `map_location="cpu"` 로드 확인
- [x] 아티팩트 생산자-소비자 분리 확인
- [ ] Optuna 산출물 경로 패턴 일관성 확인

---

## See Also

| 문서 | 관계 |
| --- | --- |
| [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) | 아티팩트 명명 및 스키마 (What) |
| [Contract_training_pipeline.md](Contract_training_pipeline.md) | Trainer 생산자 |
| [Contract_evaluation_reporting.md](Contract_evaluation_reporting.md) | Evaluator 소비자 |
| [Contract_inference_boundary.md](Contract_inference_boundary.md) | Predictor 소비자 |
