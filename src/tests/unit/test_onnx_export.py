"""
test_onnx_export.py
Tests for src/inference/onnx_export.py.
Status: FAILING — onnx_export.py not yet implemented.
Ref: doc/TDD/TDD_ONNX_Export.md
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Skip entire module if onnx not installed
onnx = pytest.importorskip("onnx")

# Will raise ImportError until implemented
from inference.onnx_export import export_to_onnx


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_cfg():
    return {
        "model": {"backbone": "efficientnet_b0", "frozen_backbone": False},
        "data": {"num_levels": 6, "image_size": 128},
        "phase0": {"projection_dim": 128, "hidden_dim": 256},
        "phase2": {
            "heads": {
                "efficientnet_b0": {"hidden_dim": 256, "dropout": 0.2}
            }
        },
        "system": {"device": "cpu"},
    }


@pytest.fixture
def tmp_phase2_checkpoint(tmp_path, minimal_cfg):
    """유효한 Phase 2 체크포인트 생성"""
    from models.grayspot_model import GrayspotModel
    model = GrayspotModel(minimal_cfg, phase=2)
    ckpt_path = tmp_path / "best_Y.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


@pytest.fixture
def corrupt_checkpoint(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"this is not a valid pytorch checkpoint file")
    return path


@pytest.fixture
def exported_onnx(tmp_phase2_checkpoint, tmp_path, minimal_cfg):
    output = tmp_path / "model_Y_effb0.onnx"
    export_to_onnx(tmp_phase2_checkpoint, output, minimal_cfg)
    return output


# ── export_to_onnx() ──────────────────────────────────────────────────────────

class TestExportToOnnx:
    """T-ONNX-01 ~ T-ONNX-07"""

    def test_creates_onnx_file(self, tmp_phase2_checkpoint, tmp_path, minimal_cfg):
        """T-ONNX-01: 유효 체크포인트 → .onnx 파일 생성"""
        output = tmp_path / "model_Y_effb0.onnx"
        export_to_onnx(tmp_phase2_checkpoint, output, minimal_cfg)
        assert output.exists()

    def test_onnx_checker_passes(self, exported_onnx):
        """T-ONNX-02: onnx.checker.check_model() 통과"""
        model = onnx.load(str(exported_onnx))
        onnx.checker.check_model(model)  # 예외 없으면 통과

    def test_corrupted_checkpoint_raises_runtime_error(self, corrupt_checkpoint, tmp_path, minimal_cfg):
        """T-ONNX-03: 손상된 체크포인트 → RuntimeError, 파일 미생성"""
        output = tmp_path / "should_not_exist.onnx"
        with pytest.raises(RuntimeError):
            export_to_onnx(corrupt_checkpoint, output, minimal_cfg)
        assert not output.exists()

    def test_auto_creates_output_directory(self, tmp_phase2_checkpoint, tmp_path, minimal_cfg):
        """T-ONNX-04: 미존재 출력 디렉토리 → 자동 생성"""
        new_dir = tmp_path / "new_subdir" / "onnx"
        output = new_dir / "model.onnx"
        export_to_onnx(tmp_phase2_checkpoint, output, minimal_cfg)
        assert output.exists()

    def test_returns_path_object(self, tmp_phase2_checkpoint, tmp_path, minimal_cfg):
        """T-ONNX-05: 반환값 == Path"""
        output = tmp_path / "model.onnx"
        result = export_to_onnx(tmp_phase2_checkpoint, output, minimal_cfg)
        assert isinstance(result, Path)

    def test_onnx_opset_version(self, exported_onnx):
        """T-ONNX-06: opset version == 17"""
        model = onnx.load(str(exported_onnx))
        opsets = {op.domain: op.version for op in model.opset_import}
        assert opsets.get("", opsets.get("ai.onnx", 0)) == 17

    def test_output_suffix_is_onnx(self, tmp_phase2_checkpoint, tmp_path, minimal_cfg):
        """T-ONNX-07: 출력 파일 확장자 .onnx"""
        output = tmp_path / "model_Y_effb0.onnx"
        result = export_to_onnx(tmp_phase2_checkpoint, output, minimal_cfg)
        assert result.suffix == ".onnx"


# ── ONNX 추론 shape 검증 ───────────────────────────────────────────────────────

class TestOnnxInferenceShape:
    """T-ONNX-INT-01 ~ T-ONNX-INT-03"""

    def test_onnxruntime_session_created(self, exported_onnx):
        """T-ONNX-INT-01: onnxruntime 세션 생성 성공"""
        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(str(exported_onnx))
        assert sess is not None

    def test_output_shape_is_1x6(self, exported_onnx):
        """T-ONNX-INT-02: 입력 (1,3,128,128) → 출력 (1,6)"""
        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(str(exported_onnx))
        input_name = sess.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
        outputs = sess.run(None, {input_name: dummy})
        assert outputs[0].shape == (1, 6)

    def test_output_is_float(self, exported_onnx):
        """T-ONNX-INT-03: 출력값은 float (logits)"""
        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(str(exported_onnx))
        input_name = sess.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
        outputs = sess.run(None, {input_name: dummy})
        assert outputs[0].dtype == np.float32
