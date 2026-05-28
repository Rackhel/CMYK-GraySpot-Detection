"""GradCAMWorker — EfficientNet-B0 / ResNet50 GradCAM 히트맵 계산.
Computes a Grad-CAM heatmap for the predicted class and returns an overlay image.

No external library required — uses pure PyTorch hooks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ._ckpt_utils import auto_find_checkpoint
from .base_worker import BaseWorker

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class GradCAMWorker(BaseWorker):
    """Compute GradCAM heatmap for a single image.

    Args:
        cfg             : dict  — load_config() return value
        image_path      : str   — path to the image
        checkpoint_path : str   — .pt path (empty → auto-detect)
        channel         : str   — "Y"|"M"|"C"|"K"
        target_level    : int | None — class index to explain; None = argmax prediction
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        image_path: str,
        checkpoint_path: str,
        channel: str = "Y",
        target_level: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_path = image_path
        self.checkpoint_path = checkpoint_path
        self.channel = channel
        self.target_level = target_level

    def run(self) -> None:
        try:
            import cv2
            import numpy as np
            import torch
            import torch.nn.functional as F
            from torchvision import transforms as T

            from gui.workers.inference_worker import _resolve_device
            from src.data.normalize import _IMAGENET_NORMALIZE
            from src.utils.utils_model import build_model

            self.emit_progress(10, "이미지 로드 / Loading image…")

            image_size = self.cfg.get("data", {}).get("image_size", 128)
            d = self.cfg.get("system", {}).get("device", "cpu")
            device = _resolve_device(d)

            img_bgr = cv2.imread(self.image_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot open: {self.image_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (image_size, image_size))

            tensor = T.ToTensor()(img_resized)
            tensor = _IMAGENET_NORMALIZE(tensor)
            inp = tensor.unsqueeze(0).to(device)

            self.emit_progress(30, "모델 로드 / Loading model…")

            ckpt = self.checkpoint_path or auto_find_checkpoint(self.cfg, self.channel)
            if not ckpt:
                raise FileNotFoundError(
                    f"Checkpoint not found for channel {self.channel}"
                )
            model = build_model(self.cfg, Path(ckpt), device)
            model.eval()

            # ── 마지막 conv feature map 레이어 탐지 / Find last conv layer ─
            target_layer = _find_last_conv(model)
            if target_layer is None:
                raise RuntimeError("Cannot find a Conv2d layer in the model")

            self.emit_progress(50, "GradCAM 계산 중 / Computing GradCAM…")

            # ── hook 등록 / Register hooks ─────────────────────────────────
            activations: list[torch.Tensor] = []
            gradients: list[torch.Tensor] = []

            def fwd_hook(_, __, output):
                activations.append(output.detach())

            def bwd_hook(_, __, grad_output):
                gradients.append(grad_output[0].detach())

            h_fwd = target_layer.register_forward_hook(fwd_hook)
            h_bwd = target_layer.register_full_backward_hook(bwd_hook)

            inp.requires_grad_(True)
            logits = model(inp)
            probs = F.softmax(logits, dim=1)[0]

            target_cls = (
                self.target_level
                if self.target_level is not None
                else int(torch.argmax(probs).item())
            )
            pred_level = int(torch.argmax(probs).item())
            confidence = float(probs[pred_level])

            model.zero_grad()
            logits[0, target_cls].backward()

            h_fwd.remove()
            h_bwd.remove()

            act = activations[0][0]  # (C, H, W)
            grad = gradients[0][0]  # (C, H, W)

            # Global average pool gradients over spatial dims
            weights = grad.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
            cam = (weights * act).sum(dim=0)  # (H, W)
            cam = F.relu(cam)
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)

            cam_np = cam.cpu().numpy()
            cam_np = cv2.resize(cam_np, (image_size, image_size))

            self.emit_progress(80, "히트맵 렌더링 / Rendering heatmap…")

            # ── 히트맵 오버레이 / Overlay heatmap ─────────────────────────
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(
                img_resized.astype(np.float32), 0.6, heatmap.astype(np.float32), 0.4, 0
            )
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            self.emit_progress(100, f"완료 / Done — L{pred_level} ({confidence:.1%})")
            self.finished.emit(
                {
                    "overlay": overlay,  # np.ndarray (H, W, 3) RGB
                    "cam": cam_np,  # np.ndarray (H, W) 0-1
                    "pred_level": pred_level,
                    "confidence": confidence,
                    "channel": self.channel,
                    "image_path": self.image_path,
                }
            )

        except Exception as exc:
            import traceback

            self.error_occurred.emit(f"{exc}\n{traceback.format_exc()}")


def _find_last_conv(model) -> "torch.nn.Conv2d | None":
    """Return the last Conv2d module in a model (depth-first)."""
    import torch.nn as nn

    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last
