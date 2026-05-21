---
type: tdd
domain: onnx_export
status: failing
last_updated: 2026-05-18
owner: CMYK WooSong Team
related_docs:
  - "../SSOT/SSOT_Artifacts.md"
  - "../Contract/Contract_artifact_boundary.md"
  - "../Contract/Contract_roi_pipeline.md"
---

# [TDD] ONNX Export — 모델 변환 / Model Conversion

> **목적 / Purpose**: `onnx_export.py`의 동작을 BDD/TDD로 정의한다.
> **테스트 파일 / Test Files**: `src/tests/unit/test_onnx_export.py`, `src/tests/integration/test_onnx_integration.py`
> **상태 / Status**: 🔴 Failing — 구현 전

---

## 1. BDD 시나리오 / BDD Scenarios

### Feature: GrayspotModel → ONNX 변환 / GrayspotModel to ONNX Conversion

**Scenario 1: 유효한 Phase 2 체크포인트 → ONNX 생성 / Valid Phase 2 checkpoint → ONNX creation**
```
Given  유효한 best_Y.pt (Phase 2 체크포인트) 와 출력 경로가 주어졌을 때
When   export_to_onnx(checkpoint_path, output_path, cfg) 를 호출하면
Then   output_path 에 .onnx 파일이 생성되고
And    onnx.checker.check_model() 이 통과된다
```

**Scenario 2: ONNX 모델 입출력 shape 검증 / ONNX model I/O shape validation**
```
Given  생성된 .onnx 파일이 있을 때
When   onnxruntime으로 (1, 3, 128, 128) 입력을 추론하면
Then   출력 shape는 (1, 6) 이다
```

**Scenario 3: 손상된 체크포인트 → RuntimeError**
```
Given  손상된(corrupt) .pt 파일 경로가 주어졌을 때
When   export_to_onnx() 를 호출하면
Then   RuntimeError 가 발생하고
And    .onnx 파일은 생성되지 않는다
```

**Scenario 4: 출력 디렉토리 자동 생성 / Automatic output directory creation**
```
Given  존재하지 않는 출력 디렉토리가 주어졌을 때
When   export_to_onnx() 를 호출하면
Then   디렉토리가 자동 생성되고 .onnx 파일이 저장된다
```

**Scenario 5: 반환값 검증 / Return value validation**
```
Given  유효한 입력이 주어졌을 때
When   export_to_onnx() 가 완료되면
Then   반환값은 생성된 .onnx 파일의 Path 객체이다
```

---

## 2. TDD 스펙 / TDD Specifications

### 2.1 export_to_onnx()

| 테스트 ID / Test ID | 입력 / Input | 기댓값 / Expected |
| --- | --- | --- |
| T-ONNX-01 | 유효 체크포인트 | .onnx 파일 생성 (`Path.exists()`) |
| T-ONNX-02 | 유효 체크포인트 | `onnx.checker.check_model()` 예외 없음 |
| T-ONNX-03 | 손상된 체크포인트 | `RuntimeError` 발생 |
| T-ONNX-04 | 미존재 출력 디렉토리 | 디렉토리 자동 생성 후 파일 저장 |
| T-ONNX-05 | 반환 타입 | `isinstance(result, Path)` |
| T-ONNX-06 | opset_version=17 | ONNX 파일 내 opset == 17 |
| T-ONNX-07 | 채널 Y 체크포인트 | 파일명에 "Y" 포함 가능 |

```python
def test_export_creates_onnx_file(tmp_phase2_checkpoint, tmp_path, minimal_cfg):
    from inference.onnx_export import export_to_onnx
    output = tmp_path / "model_Y_effb0.onnx"
    result = export_to_onnx(tmp_phase2_checkpoint, output, minimal_cfg)
    assert result.exists()
    assert result.suffix == ".onnx"

def test_export_onnx_checker_passes(tmp_phase2_checkpoint, tmp_path, minimal_cfg):
    import onnx
    from inference.onnx_export import export_to_onnx
    output = tmp_path / "model_Y_effb0.onnx"
    export_to_onnx(tmp_phase2_checkpoint, output, minimal_cfg)
    model = onnx.load(str(output))
    onnx.checker.check_model(model)  # 예외 없으면 통과 / passes if no exception

def test_export_corrupted_checkpoint(tmp_path, minimal_cfg):
    from inference.onnx_export import export_to_onnx
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a valid checkpoint")
    output = tmp_path / "out.onnx"
    with pytest.raises(RuntimeError):
        export_to_onnx(corrupt, output, minimal_cfg)
    assert not output.exists()

def test_export_returns_path(tmp_phase2_checkpoint, tmp_path, minimal_cfg):
    from pathlib import Path
    from inference.onnx_export import export_to_onnx
    output = tmp_path / "model.onnx"
    result = export_to_onnx(tmp_phase2_checkpoint, output, minimal_cfg)
    assert isinstance(result, Path)
```

### 2.2 ONNX 추론 shape 검증 / ONNX Inference Shape Validation (통합 테스트 / Integration Test)

| 테스트 ID / Test ID | 검증 포인트 / Verification |
| --- | --- |
| T-ONNX-INT-01 | onnxruntime 세션 생성 성공 |
| T-ONNX-INT-02 | 입력 (1,3,128,128) → 출력 shape (1,6) |
| T-ONNX-INT-03 | 출력값이 logits (float, unbounded) |

```python
def test_onnx_inference_shape(tmp_onnx_file):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(tmp_onnx_file))
    dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
    outputs = sess.run(None, {"input": dummy})
    assert outputs[0].shape == (1, 6)
```

---

## 3. Conftest 픽스처 / Conftest Fixtures

```python
# src/tests/unit/conftest.py 에 추가

@pytest.fixture
def tmp_phase2_checkpoint(tmp_path, minimal_cfg):
    """최소한의 유효 Phase 2 체크포인트 생성 / Create a minimal valid Phase 2 checkpoint"""
    from models.grayspot_model import GrayspotModel
    model = GrayspotModel(minimal_cfg, phase=2)
    ckpt_path = tmp_path / "best_Y.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path
```

---

## See Also

| 문서 / Document | 관계 / Relationship |
| --- | --- |
| [SSOT_Artifacts.md](../SSOT/SSOT_Artifacts.md) | ONNX 산출물 정의 / ONNX artifact definition |
| [Contract_artifact_boundary.md](../Contract/Contract_artifact_boundary.md) | export_to_onnx() API 계약 / API contract |
| [Contract_model_boundary.md](../Contract/Contract_model_boundary.md) | GrayspotModel 구조 / GrayspotModel structure |
