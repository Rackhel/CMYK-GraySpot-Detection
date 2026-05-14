"""
inference/predictor_device.py

책임 / Responsibility: 추론 장치(Device) 감지 및 설정
Responsibility: Inference device detection and setup

SRP 준수: 이 모듈은 "장치 선택"만 담당한다.
SRP compliant: this module handles only "device selection".

SSOT 근거 / SSOT Reference:
    - SSOT_Evaluation_Reporting.md §3 — 추론 장치 config 키
    - Contract.md §9 — system.device config 키
"""

from __future__ import annotations

import torch

from utils.logger import LoggerMixin


class DeviceMixin(LoggerMixin):
    """
    장치 감지 및 설정 Mixin / Device detection and setup Mixin.

    OCP 준수: 장치 타입별 전략을 메서드로 분리하여 확장 시 기존 코드 수정 불필요.
    OCP compliant: each device type is a separate method — no modification needed to add new devices.
    """

    def _setup_device(self) -> torch.device:
        """
        config['system']['device'] 값에 따라 장치를 선택한다.
        Selects device according to config['system']['device'].

        Config key: system.device — "auto" | "cuda" | "mps" | "cpu"

        Returns:
            torch.device — 선택된 추론 장치 / Selected inference device
        """
        device_cfg: str = (
            (self.cfg or {}).get("system", {}).get("device") or "auto"
        ).lower()

        handlers = {
            "auto": self._device_auto,
            "cuda": self._device_cuda,
            "mps": self._device_mps,
            "cpu": self._device_cpu,
        }

        handler = handlers.get(device_cfg, self._device_cpu)
        return handler()

    # ------------------------------------------------------------------
    # 장치별 전략 메서드 / Per-device strategy methods
    # ------------------------------------------------------------------

    def _device_auto(self) -> torch.device:
        """우선순위: CUDA → MPS → CPU / Priority: CUDA → MPS → CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _device_cuda(self) -> torch.device:
        """CUDA 요청 — 불가 시 MPS → CPU 순으로 fallback 경고 / CUDA fallback with warning."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.logger.warning("[Device] CUDA unavailable — falling back to MPS.")
            return torch.device("mps")
        self.logger.warning("[Device] CUDA unavailable — falling back to CPU.")
        return torch.device("cpu")

    def _device_mps(self) -> torch.device:
        """MPS 요청 — 불가 시 CPU fallback 경고 / MPS fallback with warning."""
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        self.logger.warning("[Device] MPS unavailable — falling back to CPU.")
        return torch.device("cpu")

    def _device_cpu(self) -> torch.device:
        """CPU 강제 지정 / Force CPU."""
        return torch.device("cpu")
