"""
Grayspot -- 모델 변환 / Model Converter
inference/model_converter.py

학습된 PyTorch 모델을 모바일 배포용 포맷으로 변환한다.
Converts trained PyTorch models to mobile deployment formats.

지원 변환 포맷 / Supported conversion formats:
    - PyTorch (.pt)   --> ONNX (.onnx)         : Android / 범용 / General
    - ONNX (.onnx)    --> TFLite (.tflite)      : Android
    - PyTorch (.pt)   --> CoreML (.mlpackage)   : iOS / macOS

사용법 / Usage:
    # 단일 채널 ONNX 변환 / Convert single channel to ONNX
    python inference/model_converter.py --channel Y --format onnx

    # 전체 채널 ONNX 변환 / Convert all channels to ONNX
    python inference/model_converter.py --channel all --format onnx

    # CoreML 변환 (macOS 전용) / CoreML conversion (macOS only)
    python inference/model_converter.py --channel Y --format coreml

사용 전 설치 / Before Use
ONNX = pip install onnx onnxruntime
TFLite = pip install onnx2tf tensorflow
CoreML = pip install coremltools
"""

import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.grayspot_model import GrayspotModel
from utils.logger import get_logger

CHANNELS = ["Y", "M", "C", "K"]


def load_config(path: str = "config/config.yaml") -> dict:
    """config.yaml을 로드한다. / Loads config.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# PyTorch --> ONNX 변환 / PyTorch to ONNX
# ──────────────────────────────────────────────
def convert_to_onnx(
    cfg: dict,
    channel: str,
    logger,
    opset_version: int = 17,
) -> Path:
    """
    학습된 PyTorch 모델을 ONNX 포맷으로 변환한다.
    Converts a trained PyTorch model to ONNX format.

    ONNX는 Android 및 범용 추론 엔진(ONNX Runtime)에서 사용된다.
    ONNX is used for Android and general-purpose inference engines (ONNX Runtime).

    Args:
        cfg:           config.yaml 딕셔너리 / config.yaml dictionary
        channel:       "Y" | "M" | "C" | "K"
        logger:        logger 인스턴스 / Logger instance
        opset_version: ONNX opset 버전 / ONNX opset version (default: 17)

    Returns:
        저장된 ONNX 파일 경로 / Path to saved ONNX file

    Example:
        >>> path = convert_to_onnx(cfg, "Y", logger)
        >>> # data/models/grayspot_Y.onnx
    """
    model_dir  = Path(cfg["inference"]["model_dir"])
    pt_path    = model_dir / f"best_{channel}.pt"
    onnx_path  = model_dir / f"grayspot_{channel}.onnx"
    size       = cfg["data"]["image_size"]

    if not pt_path.exists():
        raise FileNotFoundError(
            f"[{channel}] PyTorch 모델 없음 / Model not found: {pt_path}\n"
            f"train.py를 먼저 실행하세요 / Run train.py first."
        )

    # 모델 로드 / Load model
    model = GrayspotModel(cfg, phase=2)
    model.load(pt_path)
    model.eval()

    # 더미 입력 생성 (배치 크기 1) / Create dummy input (batch size 1)
    dummy_input = torch.randn(1, 3, size, size)

    logger.info(f"[{channel}] ONNX 변환 시작 / Starting ONNX conversion")
    logger.info(f"  입력 크기 / Input size: (1, 3, {size}, {size})")
    logger.info(f"  opset version: {opset_version}")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,            # 가중치 포함 / Include weights
        opset_version=opset_version,
        do_constant_folding=True,      # 상수 폴딩 최적화 / Constant folding optimization
        input_names=["input"],         # 입력 노드 이름 / Input node name
        output_names=["output"],       # 출력 노드 이름 / Output node name
        dynamic_axes={
            "input":  {0: "batch_size"},   # 배치 크기 동적 처리 / Dynamic batch size
            "output": {0: "batch_size"},
        },
    )

    # 변환 결과 검증 / Validate conversion result
    _verify_onnx(onnx_path, dummy_input, model, logger, channel)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"[{channel}] ONNX 변환 완료 / Conversion done: {onnx_path} ({size_mb:.1f} MB)")

    return onnx_path


def _verify_onnx(
    onnx_path: Path,
    dummy_input: torch.Tensor,
    original_model: nn.Module,
    logger,
    channel: str,
) -> None:
    """
    ONNX 모델 출력이 PyTorch 모델과 일치하는지 검증한다.
    Verifies that ONNX model output matches PyTorch model output.
    """
    try:
        import onnx
        import onnxruntime as ort

        # ONNX 모델 구조 검증 / Validate ONNX model structure
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # 출력값 비교 / Compare outputs
        session = ort.InferenceSession(str(onnx_path))
        ort_out = session.run(
            None,
            {"input": dummy_input.numpy()}
        )[0]

        with torch.no_grad():
            pt_out = original_model(dummy_input).numpy()

        max_diff = np.abs(ort_out - pt_out).max()
        if max_diff < 1e-4:
            logger.info(f"[{channel}] ONNX 검증 통과 / Verification passed -- max diff: {max_diff:.2e}")
        else:
            logger.warning(f"[{channel}] ONNX 출력 차이 발생 / Output mismatch -- max diff: {max_diff:.2e}")

    except ImportError:
        logger.warning(
            "onnx / onnxruntime 미설치 / Not installed -- 검증 건너뜀 / Skipping verification\n"
            "설치 / Install: pip install onnx onnxruntime"
        )


# ──────────────────────────────────────────────
# ONNX --> TFLite 변환 / ONNX to TFLite
# ──────────────────────────────────────────────
def convert_to_tflite(
    cfg: dict,
    channel: str,
    logger,
) -> Path:
    """
    ONNX 모델을 TFLite 포맷으로 변환한다.
    Converts an ONNX model to TFLite format.

    TFLite는 Android 온디바이스 추론에 사용된다.
    TFLite is used for Android on-device inference.

    Args:
        cfg:     config.yaml 딕셔너리 / config.yaml dictionary
        channel: "Y" | "M" | "C" | "K"
        logger:  logger 인스턴스 / Logger instance

    Returns:
        저장된 TFLite 파일 경로 / Path to saved TFLite file

    Note:
        ONNX 파일이 먼저 존재해야 한다. / ONNX file must exist first.
        변환 도구 필요 / Requires: pip install onnx2tf tensorflow
    """
    model_dir   = Path(cfg["inference"]["model_dir"])
    onnx_path   = model_dir / f"grayspot_{channel}.onnx"
    tflite_path = model_dir / f"grayspot_{channel}.tflite"

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"[{channel}] ONNX 모델 없음 / ONNX model not found: {onnx_path}\n"
            f"먼저 ONNX 변환을 실행하세요 / Run ONNX conversion first: --format onnx"
        )

    try:
        import onnx2tf

        logger.info(f"[{channel}] TFLite 변환 시작 / Starting TFLite conversion")

        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(model_dir),
            output_tfv3_tflite_file_path=str(tflite_path),
            non_verbose=True,
        )

        size_mb = tflite_path.stat().st_size / (1024 * 1024)
        logger.info(f"[{channel}] TFLite 변환 완료 / Conversion done: {tflite_path} ({size_mb:.1f} MB)")
        return tflite_path

    except ImportError:
        logger.error(
            "onnx2tf 미설치 / Not installed\n"
            "설치 / Install: pip install onnx2tf tensorflow"
        )
        raise


# ──────────────────────────────────────────────
# PyTorch --> CoreML 변환 / PyTorch to CoreML
# ──────────────────────────────────────────────
def convert_to_coreml(
    cfg: dict,
    channel: str,
    logger,
) -> Path:
    """
    학습된 PyTorch 모델을 CoreML 포맷으로 변환한다.
    Converts a trained PyTorch model to CoreML format.

    CoreML은 iOS / macOS 온디바이스 추론에 사용된다.
    CoreML is used for iOS / macOS on-device inference.

    Args:
        cfg:     config.yaml 딕셔너리 / config.yaml dictionary
        channel: "Y" | "M" | "C" | "K"
        logger:  logger 인스턴스 / Logger instance

    Returns:
        저장된 CoreML 패키지 경로 / Path to saved CoreML package

    Note:
        macOS 전용 / macOS only
        변환 도구 필요 / Requires: pip install coremltools
    """
    model_dir    = Path(cfg["inference"]["model_dir"])
    pt_path      = model_dir / f"best_{channel}.pt"
    coreml_path  = model_dir / f"grayspot_{channel}.mlpackage"
    size         = cfg["data"]["image_size"]

    if not pt_path.exists():
        raise FileNotFoundError(
            f"[{channel}] PyTorch 모델 없음 / Model not found: {pt_path}\n"
            f"train.py를 먼저 실행하세요 / Run train.py first."
        )

    try:
        import coremltools as ct

        # 모델 로드 및 TorchScript 변환 / Load model and convert to TorchScript
        model = GrayspotModel(cfg, phase=2)
        model.load(pt_path)
        model.eval()

        dummy_input  = torch.randn(1, 3, size, size)
        traced_model = torch.jit.trace(model, dummy_input)

        logger.info(f"[{channel}] CoreML 변환 시작 / Starting CoreML conversion")

        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name="input",
                shape=(1, 3, size, size),
            )],
            outputs=[ct.TensorType(name="output")],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL,
        )

        # 모델 메타데이터 설정 / Set model metadata
        coreml_model.short_description = f"Grayspot Level Classifier -- Channel {channel}"
        coreml_model.input_description["input"]  = f"CMYK {channel} channel ROI image (224x224)"
        coreml_model.output_description["output"] = "Level 0-5 logits"

        coreml_model.save(str(coreml_path))

        logger.info(f"[{channel}] CoreML 변환 완료 / Conversion done: {coreml_path}")
        return coreml_path

    except ImportError:
        logger.error(
            "coremltools 미설치 / Not installed\n"
            "설치 / Install: pip install coremltools"
        )
        raise


# ──────────────────────────────────────────────
# 변환 요약 출력 / Conversion Summary
# ──────────────────────────────────────────────
def print_conversion_summary(cfg: dict) -> None:
    """
    모델 폴더 내 변환된 파일 목록과 크기를 출력한다.
    Prints a list and sizes of converted files in the model directory.
    """
    model_dir = Path(cfg["inference"]["model_dir"])

    print("\n" + "=" * 55)
    print("  변환 파일 목록 / Converted Files")
    print("=" * 55)

    exts = {".pt": "PyTorch", ".onnx": "ONNX", ".tflite": "TFLite", ".mlpackage": "CoreML"}

    for ext, fmt in exts.items():
        files = sorted(model_dir.glob(f"*{ext}"))
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  [{fmt:<8}] {f.name:<35} {size_mb:>6.1f} MB")

    if not any(model_dir.glob("*")):
        print("  변환된 파일 없음 / No converted files found")

    print()


# ──────────────────────────────────────────────
# 메인 실행 / Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Grayspot 모델 변환 스크립트 / Model Conversion Script"
    )
    parser.add_argument("--channel", type=str, default="all",
                        help="변환할 채널 / Channel to convert (Y/M/C/K/all)")
    parser.add_argument("--format",  type=str, default="onnx",
                        choices=["onnx", "tflite", "coreml"],
                        help="변환 포맷 / Target format (onnx / tflite / coreml)")
    parser.add_argument("--config",  type=str, default="config/config.yaml",
                        help="config.yaml 경로 / Path to config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    logger = get_logger("converter", cfg)

    target_channels = CHANNELS if args.channel == "all" else [args.channel.upper()]

    print("=" * 55)
    print("  Grayspot -- Model Converter")
    print(f"  Channels: {target_channels} | Format: {args.format.upper()}")
    print("=" * 55)

    for ch in target_channels:
        try:
            if args.format == "onnx":
                convert_to_onnx(cfg, ch, logger)
            elif args.format == "tflite":
                convert_to_tflite(cfg, ch, logger)
            elif args.format == "coreml":
                convert_to_coreml(cfg, ch, logger)
        except FileNotFoundError as e:
            logger.error(str(e))
        except Exception as e:
            logger.error(f"[{ch}] 변환 실패 / Conversion failed: {e}")

    print_conversion_summary(cfg)


if __name__ == "__main__":
    main()