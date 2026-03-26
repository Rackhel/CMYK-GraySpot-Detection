"""
Grayspot — 학습 루프 (Phase 0 / Phase 2)
training/trainer.py

Phase0Trainer: SimCLR 기반 Contrastive Learning 학습기
Phase2Trainer: Supervised Classification 학습기 (Stage 1 → Stage 2)

Phase0Trainer: SimCLR-based Contrastive Learning trainer
Phase2Trainer: Supervised Classification trainer (Stage 1 → Stage 2)
"""

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from training.contrastive_loss import InfoNCELoss


CHANNELS = ["Y", "M", "C", "K"]


# ──────────────────────────────────────────────
# Phase 0 Trainer — Contrastive Learning
# ──────────────────────────────────────────────
class Phase0Trainer:
    """
    SimCLR 기반 Contrastive Learning 학습기.
    채널별(Y/M/C/K) 독립적으로 모델을 학습한다.

    SimCLR-based Contrastive Learning trainer.
    Trains a model independently per channel (Y/M/C/K).
    """

    def __init__(self, model, cfg: dict, channel: str):
        self.model   = model
        self.cfg     = cfg
        self.channel = channel
        self.device  = torch.device("cpu")  # GPU 불필요 / No GPU required

        p = cfg["phase0"]
        self.criterion = InfoNCELoss(temperature=p["temperature"])
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=p["learning_rate"],
        )
        # Cosine Annealing: 학습률을 코사인 곡선으로 점진적으로 감소
        # Cosine Annealing: gradually decreases learning rate following a cosine curve
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=p["epochs"])
        self.history: list[dict] = []

    def train_epoch(self, loader: DataLoader) -> float:
        """에폭 1회 학습 / Runs one training epoch."""
        self.model.train()
        total_loss = 0.0
        for view1, view2 in loader:
            view1, view2 = view1.to(self.device), view2.to(self.device)

            # Positive Pair forward pass
            z1   = self.model(view1)
            z2   = self.model(view2)
            loss = self.criterion(z1, z2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def train(self, loader: DataLoader) -> list[dict]:
        """전체 학습 실행 / Runs the full training loop."""
        p = self.cfg["phase0"]
        print(f"\n  Phase 0 학습 시작 / Training started — Channel: {self.channel}")
        print(f"    Epochs: {p['epochs']} | Batch: {p['batch_size']} | τ: {p['temperature']}")

        for epoch in range(1, p["epochs"] + 1):
            t0      = time.time()
            loss    = self.train_epoch(loader)
            self.scheduler.step()
            elapsed = time.time() - t0

            record = {
                "phase":   0,
                "channel": self.channel,
                "epoch":   epoch,
                "loss":    round(loss, 6),
                "lr":      round(self.optimizer.param_groups[0]["lr"], 8),
                "elapsed": round(elapsed, 2),
            }
            self.history.append(record)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>4}/{p['epochs']} | Loss: {loss:.4f} | "
                      f"LR: {record['lr']:.2e} | {elapsed:.1f}s")

        self._save_history()
        return self.history

    def save_backbone(self) -> Path:
        """Phase 0 학습된 Backbone weights를 저장한다. / Saves Phase 0 trained backbone weights."""
        model_dir = Path(self.cfg["inference"]["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"phase0_backbone_{self.channel}.pt"
        self.model.save(path)
        return path

    def _save_history(self) -> None:
        """학습 이력을 CSV로 저장한다. / Saves training history to CSV."""
        reports_dir = Path(self.cfg["storage"]["reports_dir"])
        path        = reports_dir / f"phase0_history_{self.channel}.csv"
        if self.history:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
                writer.writeheader()
                writer.writerows(self.history)


# ──────────────────────────────────────────────
# Phase 2 Trainer — Supervised Classification
# ──────────────────────────────────────────────
class Phase2Trainer:
    """
    Supervised Classification 학습기.
    Stage 1 (Backbone freeze) → Stage 2 (fine-tuning) 2단계 학습을 수행한다.

    Supervised Classification trainer.
    Performs two-stage training: Stage 1 (Backbone freeze) → Stage 2 (fine-tuning).
    """

    def __init__(self, model, cfg: dict, channel: str, class_weights=None):
        self.model   = model
        self.cfg     = cfg
        self.channel = channel
        self.device  = torch.device("cpu")

        # 클래스 불균형 보정 가중치 적용 / Apply class imbalance correction weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self._build_optimizer()
        self.history:      list[dict] = []
        self.best_val_acc: float      = 0.0  # 최고 검증 정확도 추적 / Track best validation accuracy
        self.best_epoch:   int        = 0

    def _build_optimizer(self) -> None:
        """Optimizer와 Scheduler를 생성한다. / Builds optimizer and scheduler."""
        p = self.cfg["phase2"]
        self.optimizer = AdamW(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=p["learning_rate"],
            weight_decay=p["weight_decay"],  # L2 정규화 / L2 regularization
        )
        total_epochs   = p["stage1_epochs"] + p["stage2_epochs"]
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs,
            eta_min=1e-6,  # 최소 학습률 / Minimum learning rate
        )

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """에폭 1회 학습. (loss, accuracy) 반환 / Runs one training epoch. Returns (loss, accuracy)."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, labels, _ in loader:
            x, labels = x.to(self.device), labels.to(self.device)
            logits     = self.model(x)
            loss       = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        return total_loss / max(len(loader), 1), correct / max(total, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """검증 데이터 평가. (loss, accuracy) 반환 / Evaluates on validation data. Returns (loss, accuracy)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for x, labels, _ in loader:
            x, labels  = x.to(self.device), labels.to(self.device)
            logits      = self.model(x)
            loss        = self.criterion(logits, labels)
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)
        return total_loss / max(len(loader), 1), correct / max(total, 1)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
        """전체 학습 실행 (Stage 1 → Stage 2). / Runs full training (Stage 1 → Stage 2)."""
        p            = self.cfg["phase2"]
        s1_epochs    = p["stage1_epochs"]
        s2_epochs    = p["stage2_epochs"]
        total_epochs = s1_epochs + s2_epochs

        print(f"\n  Phase 2 학습 시작 / Training started — Channel: {self.channel}")
        print(f"    Stage1: {s1_epochs}ep (Backbone freeze) | Stage2: {s2_epochs}ep (fine-tune)")

        for epoch in range(1, total_epochs + 1):

            # Stage 1 → Stage 2 전환: Backbone unfreeze
            # Stage 1 → Stage 2 transition: unfreeze backbone
            if epoch == s1_epochs + 1:
                self.model.unfreeze_backbone(p["stage2_unfreeze_layers"])
                self._build_optimizer()  # unfreeze 후 optimizer 재생성 / Rebuild optimizer after unfreeze
                print(f"\n  ▶ Stage 2 진입 / Entering Stage 2 (Epoch {epoch})")

            t0                        = time.time()
            train_loss, train_acc     = self.train_epoch(train_loader)
            val_loss,   val_acc       = self.evaluate(val_loader)
            self.scheduler.step()
            elapsed                   = time.time() - t0

            stage  = 1 if epoch <= s1_epochs else 2
            record = {
                "phase":      2,
                "channel":    self.channel,
                "epoch":      epoch,
                "stage":      stage,
                "train_loss": round(train_loss, 6),
                "train_acc":  round(train_acc,  4),
                "val_loss":   round(val_loss,   6),
                "val_acc":    round(val_acc,    4),
                "lr":         round(self.optimizer.param_groups[0]["lr"], 8),
                "elapsed":    round(elapsed, 2),
            }
            self.history.append(record)

            # Best 모델 저장 (val_acc 기준) / Save best model (based on val_acc)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch   = epoch
                self._save_best()

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>4}/{total_epochs} [S{stage}] | "
                      f"Train {train_acc:.3f} | Val {val_acc:.3f} | "
                      f"LR: {record['lr']:.2e} | {elapsed:.1f}s")

        print(f"\n  🏆  Best Val Acc: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        self._save_history()
        return self.history

    def _save_best(self) -> None:
        """최고 성능 모델을 저장한다. / Saves the best-performing model."""
        model_dir = Path(self.cfg["inference"]["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"best_{self.channel}.pt"
        self.model.save(path)

    def _save_history(self) -> None:
        """학습 이력을 CSV로 저장한다 (기존 파일에 append). / Saves training history to CSV (appends if exists)."""
        reports_dir = Path(self.cfg["storage"]["reports_dir"])
        path        = reports_dir / self.cfg["reporting"]["csv_files"]["training_history"]
        mode        = "a" if path.exists() else "w"
        with open(path, mode, newline="") as f:
            if self.history:
                writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
                if mode == "w":
                    writer.writeheader()  # 새 파일이면 헤더 작성 / Write header for new file
                writer.writerows(self.history)