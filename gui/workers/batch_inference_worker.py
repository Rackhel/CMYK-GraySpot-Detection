"""BatchInferenceWorker — 폴더 내 모든 이미지를 일괄 추론하는 QThread 워커.
Batch-inference worker: runs InferenceWorker logic over every image in a folder.

Contract: Contract_gui.md §2.5  /  SSOT_GUI.md §3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .base_worker import BaseWorker

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class BatchInferenceWorker(BaseWorker):
    """폴더 안의 이미지 파일을 전부 순회하며 추론한다.
    Iterates every image file in a folder and runs inference on each.

    Args:
        cfg             : dict — load_config() 반환값
        folder_path     : str  — 이미지 폴더 경로
        checkpoint_path : str  — .pt 체크포인트 경로

    finished payload keys:
        results   : list[dict]  — per-image result dicts
        total     : int
        succeeded : int
        failed    : int
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        folder_path: str,
        checkpoint_path: str,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.folder_path = folder_path
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:  # noqa: C901
        try:
            import cv2
            import torch
            import torch.nn.functional as F

            from src.data.normalize import _IMAGENET_NORMALIZE
            from src.utils.utils_model import build_model

            folder = Path(self.folder_path)
            if not folder.is_dir():
                raise NotADirectoryError(f"Not a directory: {self.folder_path}")

            images = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in _IMG_EXTS]
            total = len(images)
            if total == 0:
                self.emit_progress(100, "이미지 없음 / No images found")
                self.finished.emit({"results": [], "total": 0, "succeeded": 0, "failed": 0})
                return

            self.emit_progress(5, f"{total}개 이미지 발견 / Found {total} images")

            # ── 모델 1회 로드 후 재사용 ────────────────────────────────────────
            d = self.cfg.get("system", {}).get("device", "cpu")
            if d == "auto":
                device = torch.device(
                    "cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu"
                )
            else:
                device = torch.device(d)

            ckpt_path = Path(self.checkpoint_path) if self.checkpoint_path else None
            model = build_model(self.cfg, ckpt_path, device)
            model.eval()

            image_size = self.cfg.get("data", {}).get("image_size", 128)

            from torchvision import transforms as T

            results: list[dict] = []
            succeeded = 0
            failed = 0

            for idx, img_path in enumerate(images):
                if self._cancelled:
                    break

                # per-image progress
                pct = 10 + int(85 * idx / total)
                self.emit_progress(pct, f"[{idx + 1}/{total}] {img_path.name}")

                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise ValueError("Cannot read image")

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (image_size, image_size))
                    tensor = T.ToTensor()(img)
                    tensor = _IMAGENET_NORMALIZE(tensor)       # SSOT-NM01
                    tensor = tensor.unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(tensor)
                        probs  = F.softmax(logits, dim=1)[0]

                    probs_list  = probs.cpu().tolist()
                    pred_level  = int(torch.argmax(probs).item())
                    confidence  = float(probs[pred_level])
                    sorted_idx  = sorted(range(len(probs_list)), key=lambda i: probs_list[i], reverse=True)
                    top3        = [(i, probs_list[i]) for i in sorted_idx[:3]]

                    results.append({
                        "filename":   img_path.name,
                        "path":       str(img_path),
                        "pred_level": pred_level,
                        "confidence": confidence,
                        "top3":       top3,
                        "error":      None,
                    })
                    succeeded += 1

                    # 실시간 행 추가를 위해 log_emitted로 JSON 전달
                    self.log_emitted.emit(
                        "__ROW__" + json.dumps({
                            "filename":   img_path.name,
                            "pred_level": pred_level,
                            "confidence": confidence,
                            "top3":       top3,
                        })
                    )

                except Exception as row_exc:
                    failed += 1
                    results.append({
                        "filename":   img_path.name,
                        "path":       str(img_path),
                        "pred_level": -1,
                        "confidence": 0.0,
                        "top3":       [],
                        "error":      str(row_exc),
                    })
                    self.log_emitted.emit(f"⚠️  {img_path.name}: {row_exc}")

            self.emit_progress(100, f"완료 / Done — {succeeded}/{total} 성공")
            self.finished.emit({
                "results":   results,
                "total":     total,
                "succeeded": succeeded,
                "failed":    failed,
            })

        except Exception as exc:
            import traceback
            self.error_occurred.emit(f"{exc}\n{traceback.format_exc()}")
