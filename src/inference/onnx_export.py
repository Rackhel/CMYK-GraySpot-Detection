"""ONNX export utilities.

Provides a reusable export path for Phase 2 GrayspotModel checkpoints and
loaded models.

본 모듈은 Phase 2 GrayspotModel 체크포인트와 로드된 모델을 ONNX로 내보내는
공통 인터페이스를 제공합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from utils.utils_model import build_model


def export_model_to_onnx(
    model: nn.Module,
    output_path: Path | str,
    sample_input: Optional[torch.Tensor] = None,
) -> Path:
    """Export a loaded PyTorch model instance to ONNX.

    Args:
        model: Loaded PyTorch model to export.
        output_path: Path to save the .onnx file.
        sample_input: Optional sample tensor. If omitted, the caller should
            provide a valid tensor matching model input shape.

    Returns:
        Path: The exported ONNX model path.

    Raises:
        ValueError: If output_path does not use .onnx extension.
        RuntimeError: If ONNX export fails.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("output_path must end with .onnx")

    # 출력 디렉터리 생성 / Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sample_input is None:
        raise ValueError("sample_input must be provided when exporting a loaded model")

    if sample_input.ndim != 4:
        raise ValueError("sample_input must be a 4D tensor of shape (B, C, H, W)")

    # 모델과 텐서를 CPU로 이동 / Move model and input to CPU for export
    sample_input = sample_input.to(dtype=torch.float32)
    model.eval()
    model = model.to(torch.device("cpu"))
    sample_input = sample_input.to(torch.device("cpu"))

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_input,
                str(output_path),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to export ONNX model: {exc}") from exc

    return output_path


def export_to_onnx(
    checkpoint_path: Path | str,
    output_path: Path | str,
    cfg: dict,
    sample_input: Optional[torch.Tensor] = None,
) -> Path:
    """Export a Phase 2 GrayspotModel checkpoint to ONNX.

    Args:
        checkpoint_path: Path to a PyTorch checkpoint containing the model state.
        output_path: Path to write the exported .onnx model.
        cfg: Configuration dictionary used to build the model.
        sample_input: Optional sample tensor for tracing. If omitted, a random
            tensor of shape (1, 3, image_size, image_size) is generated.

    Returns:
        Path: The path to the exported ONNX model.

    Raises:
        ValueError: If the output file extension is not .onnx.
        RuntimeError: If checkpoint loading or ONNX export fails.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".onnx":
        raise ValueError("output_path must end with .onnx")

    # 출력 디렉터리 생성 / Create output directory if missing
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    try:
        model = build_model(cfg, checkpoint_path, device)
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to load checkpoint for ONNX export: {exc}") from exc

    if sample_input is None:
        image_size = cfg.get("data", {}).get("image_size")
        if image_size is None:
            raise ValueError(
                "cfg['data']['image_size'] must be set for default sample input"
            )
        sample_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    elif sample_input.ndim != 4:
        raise ValueError("sample_input must be a 4D tensor of shape (B, C, H, W)")

    return export_model_to_onnx(model, output_path, sample_input)
