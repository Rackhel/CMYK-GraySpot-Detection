"""EmbeddingWorker — t-SNE 임베딩 추출을 백그라운드 QThread에서 실행.
Extracts model embeddings and runs t-SNE in a background QThread.

Contract: Contract_gui.md §2.5
SSOT:     SSOT_GUI.md §3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .base_worker import BaseWorker

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class EmbeddingWorker(BaseWorker):
    """Extract embedding payloads away from the GUI thread.

    Constructor args (Contract_gui.md §2.5):
        cfg             : dict — load_config() 반환값
        channel         : str  — "Y" | "M" | "C" | "K"
        checkpoint_path : str  — .pt 파일 경로 (비어있으면 models_dir/best_{channel}.pt 자동 탐색)
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        channel: str,
        checkpoint_path: str,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.channel = channel
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:
        """GrayspotModel에서 feature를 추출하고 t-SNE 2D 변환 후 결과 emit.
        Extracts features from GrayspotModel, runs t-SNE, emits 2D result.
        """
        try:
            import numpy as np
            import torch
            from torch.utils.data import DataLoader

            from src.data.dataset import CMYKDataset
            from src.utils.utils_model import build_model

            self.emit_progress(0, f"[{self.channel}] 임베딩 추출 시작 / Embedding extraction started")

            if self.is_cancelled():
                self.log_emitted.emit("Embedding cancelled before start")
                return

            # ── 디바이스 설정 / Device setup ──────────────────────────────────
            device_str = self.cfg.get("system", {}).get("device", "cpu")
            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            ) if device_str == "auto" else torch.device(device_str)

            # ── 체크포인트 경로 해소 / Resolve checkpoint path ────────────────
            storage = self.cfg.get("storage", {})
            models_dir = Path(storage.get("models_dir", "data_set/models"))
            ckpt_path = (
                Path(self.checkpoint_path)
                if self.checkpoint_path
                else models_dir / f"best_{self.channel}.pt"
            )
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            # ── 모델 로드 (backbone feature extractor 모드) / Load model ──────
            self.emit_progress(20, f"[{self.channel}] 모델 로드 중 / Loading model...")
            model = build_model(self.cfg, ckpt_path, device)
            model.eval()

            # ── 데이터셋 구성 (test split 전체) / Build dataset ───────────────
            self.emit_progress(30, f"[{self.channel}] 데이터 로드 중 / Loading dataset...")
            dataset = CMYKDataset(
                self.cfg, self.channel, split="test", augment=False, oversample=False
            )
            loader = DataLoader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

            # ── Feature 추출 / Extract features ───────────────────────────────
            self.emit_progress(40, f"[{self.channel}] feature 추출 중 / Extracting features...")
            all_feats, all_labels, all_paths = [], [], []

            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    if self.is_cancelled():
                        self.log_emitted.emit("Embedding interrupted during extraction")
                        return

                    # CMYKDataset returns (tensor, label) or (tensor, label, path)
                    if len(batch) == 3:
                        imgs, labels, paths = batch
                        all_paths.extend(paths)
                    else:
                        imgs, labels = batch

                    imgs = imgs.to(device)
                    # backbone feature 추출 (head 전) / Extract backbone features (before head)
                    feats = model.backbone(imgs)
                    if hasattr(feats, "flatten"):
                        feats = feats.flatten(1)
                    all_feats.append(feats.cpu().numpy())
                    all_labels.extend(labels.tolist())

                    progress = 40 + int(batch_idx / len(loader) * 40)
                    self.emit_progress(progress, f"[{self.channel}] {batch_idx+1}/{len(loader)} 배치 처리")

            features = np.concatenate(all_feats, axis=0)
            labels_arr = np.array(all_labels, dtype=int)

            # ── t-SNE 2D 변환 / t-SNE projection ────────────────────────────
            self.emit_progress(85, f"[{self.channel}] t-SNE 변환 중 / Running t-SNE...")
            from sklearn.manifold import TSNE

            n_samples = features.shape[0]
            perplexity = min(30, max(5, n_samples // 10))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=500)
            embeddings_2d = tsne.fit_transform(features).tolist()

            self.emit_progress(100, f"[{self.channel}] 임베딩 완료 / Embedding finished ({n_samples} samples)")
            self.finished.emit({
                "embeddings_2d": embeddings_2d,
                "labels": labels_arr.tolist(),
                "paths": all_paths if all_paths else [f"sample_{i}" for i in range(n_samples)],
                "channel": self.channel,
            })

        except Exception as exc:
            import traceback
            self.error_occurred.emit(f"{exc}\n{traceback.format_exc()}")
