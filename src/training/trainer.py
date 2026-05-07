"""
training/trainer.py

Phase 0 / Phase 2 학습 루프 — 모드 전환 지원.
Phase 0 / Phase 2 training loop — supports mode switching.

Phase 0 Trainer: SimCLR Contrastive Learning
Phase 2 Trainer: Supervised Classification (CE Loss)

Dataset 클래스는 data/dataset.py 에 위치한다.
Dataset classes are located in data/dataset.py.
"""

import csv
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.utils import get_logger
from data.dataset       import CMYKDataset, ContrastiveDataset   # noqa: F401 — re-export
from models.grayspot_model import GrayspotModel
from training.losses    import get_loss

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Backbone 약어 / Backbone abbreviation helper
# ──────────────────────────────────────────────────────────────

def backbone_tag(backbone_name: str) -> str:
    """
    backbone 이름의 파일명용 약어를 반환한다.
    Returns a filename-safe abbreviation for the backbone name.

    Examples:
        "efficientnet_b0" → "effb0"
        "resnet50"        → "res50"
    """
    _MAP = {
        "efficientnet_b0": "effb0",
        "resnet50":        "res50",
    }
    return _MAP.get(backbone_name, backbone_name.replace("_", "")[:8])


# ──────────────────────────────────────────────────────────────
# Phase 0 Trainer
# ──────────────────────────────────────────────────────────────

class Phase0Trainer:
    """
    Phase 0 Contrastive Learning 학습기.
    Phase 0 Contrastive Learning trainer.
    """

    def __init__(self, model: GrayspotModel, cfg: dict,
                 channel: str, device: torch.device):
        self.model     = model
        self.cfg       = cfg
        self.channel   = channel
        self.device    = device
        self.criterion = get_loss(phase=0, cfg=cfg)
        self.optimizer = AdamW(model.parameters(), lr=cfg["phase0"]["learning_rate"])
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg["phase0"]["epochs"], eta_min=cfg["train"]["eta_min"]
        )

    def train(self, loader: DataLoader) -> list[dict]:
        """학습 루프 실행 / Run training loop."""
        epochs  = self.cfg["phase0"]["epochs"]
        history = []

        logger.info(f"{'='*55}\n  Phase 0 Training — Channel: [{self.channel}]\n{'='*55}")
        logger.info(f"  {'Epoch':<8} {'Loss':<14} {'LR':<14} Elapsed")
        logger.info(f"  {'-'*50}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            total_loss = 0.0

            for view1, view2 in loader:
                view1, view2 = view1.to(self.device), view2.to(self.device)
                loss         = self.criterion(self.model(view1), self.model(view2))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(loader), 1)
            self.scheduler.step()
            elapsed = time.time() - t0
            lr      = self.optimizer.param_groups[0]["lr"]

            history.append({"epoch": epoch, "loss": round(avg_loss, 6),
                            "lr": round(lr, 8), "elapsed": round(elapsed, 2)})
            logger.info(f"  {epoch:<8} {avg_loss:<14.4f} {lr:<14.2e} {elapsed:.1f}s")

        logger.info(f"  {'-'*50}\n  Done — Loss: {history[0]['loss']:.4f} → {history[-1]['loss']:.4f}")
        return history

    def save_backbone(self) -> Path:
        """학습된 모델 state_dict 저장 / Save trained model state_dict."""
        models_dir = Path(self.cfg["storage"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        tag       = backbone_tag(self.cfg["model"]["backbone"])
        save_path = models_dir / f"phase0_backbone_{self.channel}_{tag}.pt"
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"  Backbone saved / 저장: {save_path}")
        return save_path


# ──────────────────────────────────────────────────────────────
# Phase 2 Trainer
# ──────────────────────────────────────────────────────────────

class Phase2Trainer:
    """
    Phase 2 Supervised Classification 학습기.
    Phase 2 Supervised Classification trainer.
    """

    def __init__(self, model: GrayspotModel, cfg: dict,
                 channel: str, device: torch.device,
                 train_ds: CMYKDataset = None):
        self.model   = model
        self.cfg     = cfg
        self.channel = channel
        self.device  = device

        self.criterion = get_loss(phase=2, cfg=cfg,
                                  train_samples=train_ds.samples if train_ds else None)
        if hasattr(self.criterion, "weight") and self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg["phase2"]["learning_rate"],
            weight_decay=cfg["phase2"]["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg["phase2"]["epochs"], eta_min=cfg["train"]["eta_min"]
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
        """학습 루프 실행 / Run training loop."""
        epochs     = self.cfg["phase2"]["epochs"]
        models_dir = Path(self.cfg["storage"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        best_path  = models_dir / f"best_{self.channel}.pt"

        history, best_val_acc, best_epoch = [], 0.0, 0

        logger.info(f"{'='*65}\n  Phase 2 Training — Channel: [{self.channel}]\n{'='*65}")
        logger.info(f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<10} LR")
        logger.info(f"  {'-'*60}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for x, labels in train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                logits    = self.model(x)
                loss      = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss    += loss.item()
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total   += len(labels)

            train_loss_avg = train_loss / max(len(train_loader), 1)
            train_acc      = train_correct / max(train_total, 1)

            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for x, labels in val_loader:
                    x, labels = x.to(self.device), labels.to(self.device)
                    logits    = self.model(x)
                    loss      = self.criterion(logits, labels)
                    val_loss    += loss.item()
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total   += len(labels)

            val_loss_avg = val_loss / max(len(val_loader), 1)
            val_acc      = val_correct / max(val_total, 1)

            self.scheduler.step()
            elapsed = time.time() - t0
            lr      = self.optimizer.param_groups[0]["lr"]

            history.append({
                "epoch": epoch, "train_loss": round(train_loss_avg, 6),
                "train_acc": round(train_acc, 4), "val_loss": round(val_loss_avg, 6),
                "val_acc": round(val_acc, 4), "lr": round(lr, 8), "elapsed": round(elapsed, 2),
            })

            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                torch.save(self.model.state_dict(), best_path)

            logger.info(f"  {epoch:<8} {train_loss_avg:<14.4f} {train_acc:<12.4f} "
                        f"{val_loss_avg:<12.4f} {val_acc:<10.4f} {lr:.2e}")

        logger.info(f"  {'-'*60}")
        logger.info(f"  Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
        logger.info(f"  Model saved / 저장: {best_path}\n{'='*65}")
        return history

    def save_history(self, history: list[dict]) -> Path:
        """학습 이력 CSV 저장 / Save training history to CSV."""
        reports_dir = Path(self.cfg["storage"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path    = reports_dir / f"phase2_history_{self.channel}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        logger.info(f"  History saved / 저장: {csv_path}")
        return csv_path
