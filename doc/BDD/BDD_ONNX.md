---
type: bdd
domain: onnx_export
status: Active
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "SSOT_Artifacts.md"
  - "Contract_artifact_boundary.md"
  - "Contract_roi_pipeline.md"
---

# [BDD] ONNX 내보내기 / ONNX Export

> **역할 / Role**: GrayspotModel을 ONNX 형식으로 내보내는 관찰 가능한 행동을 정의한다.
> **Role**: Defines observable behavior for exporting GrayspotModel to ONNX format.

---

## 행위자 / Actors

| 행위자 / Actor | 역할 / Role |
|---|---|
| **운영자 / Operator** | ONNX 내보내기를 실행하는 엔지니어 / Engineer running ONNX export |
| **시스템 / System** | Phase 2 체크포인트를 ONNX로 변환하는 파이프라인 / Pipeline converting Phase 2 checkpoint to ONNX |

---

## Feature: ONNX 내보내기 / ONNX Export

> **비즈니스 가치 / Business Value**: 배포 환경에서 PyTorch 의존 없이 추론이 가능하도록 ONNX 형식으로 모델을 내보낸다.
>
> **Business Value**: Exports the model to ONNX format so inference can run without PyTorch dependency in deployment environments.

---

### Scenario O.1 — ONNX 파일 생성 성공 / ONNX File Creation Success

```gherkin
Feature: ONNX 내보내기 / ONNX export

  Scenario: export_to_onnx()가 ONNX 파일을 생성한다
  Scenario: export_to_onnx() creates an ONNX file

    Given Phase 2 학습 완료된 체크포인트(best_Y.pt)가 존재한다
    And   config.json에 backbone 정보가 정의되어 있다

    When  export_to_onnx(cfg, channel="Y", output_dir=...)가 호출된다

    Then  model_Y_effb0.onnx 파일이 생성된다
    And   onnx.checker.check_model()이 오류 없이 완료된다
```

---

### Scenario O.2 — ONNX 입출력 형상 검증 / ONNX Input/Output Shape Validation

```gherkin
  Scenario: ONNX 모델의 입출력 shape이 계약과 일치한다
  Scenario: ONNX model input/output shape matches contract

    Given model_Y_effb0.onnx 파일이 존재한다

    When  onnxruntime으로 더미 입력(1, 3, 128, 128)을 추론한다

    Then  출력 shape이 (1, 6)이다
    And   dtype이 float32이다
    And   추론이 5초 이내에 완료된다
```

---

### Scenario O.3 — Opset 버전 준수 / Opset Version Compliance

```gherkin
  Scenario: 내보낸 ONNX 파일의 opset 버전이 17이다
  Scenario: Exported ONNX file uses opset version 17

    Given export_to_onnx()가 호출된다

    When  onnx.load()로 파일을 로드한다

    Then  opset_version이 17이다
```

---

### Scenario O.4 — 파일명 패턴 준수 / File Naming Pattern Compliance

```gherkin
  Scenario: 내보낸 파일명이 model_{channel}_{tag}.onnx 패턴을 따른다
  Scenario: Exported filename follows model_{channel}_{tag}.onnx pattern

    Given channel="Y", backbone="efficientnet_b0"가 설정되어 있다

    When  export_to_onnx()가 호출된다

    Then  파일명이 model_Y_effb0.onnx이다
    And   다른 채널/backbone 조합으로 동일한 디렉터리에 중복 저장되지 않는다
```

---

### Scenario O.5 — Phase 0 체크포인트 내보내기 금지 / Phase 0 Checkpoint Export Prohibited

```gherkin
  Scenario: Phase 0 체크포인트는 ONNX로 내보낼 수 없다
  Scenario: Phase 0 checkpoint cannot be exported to ONNX

    Given phase0_backbone_Y_effb0.pt가 존재한다

    When  export_to_onnx()에 phase=0 체크포인트 경로가 전달된다

    Then  ValueError가 발생한다
    And   "Phase 0 checkpoint not supported" 메시지가 포함된다
```

---

## 추적 매트릭스 / Traceability Matrix

| 시나리오 / Scenario | TDD 파일 / TDD File | 테스트 함수 / Test Function | 계층 / Layer |
|---|---|---|---|
| O.1 — 파일 생성 / file creation | `test_onnx_export.py` | `test_export_creates_onnx_file` | Unit |
| O.2 — 입출력 형상 / shape | `test_onnx_export.py` | `test_onnx_inference_output_shape` | Unit |
| O.3 — Opset 버전 / version | `test_onnx_export.py` | `test_onnx_opset_version_is_17` | Unit |
| O.4 — 파일명 패턴 / filename | `test_onnx_export.py` | `test_export_filename_pattern` | Unit |
| O.5 — Phase 0 금지 / prohibited | `test_onnx_export.py` | `test_export_phase0_checkpoint_raises` | Unit |

---

## 관련 문서 / Related Documents

| 문서 / Document | 관계 / Relationship |
|---|---|
| [Contract_artifact_boundary.md](../Contract/Contract_artifact_boundary.md) | ONNX 아티팩트 계약 / ONNX artifact contract |
| [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) | ONNX 파일 스키마 / ONNX file schema |
| [TDD_ONNX_Export.md](../TDD/TDD_ONNX_Export.md) | ONNX TDD 명세 / ONNX TDD specification |
