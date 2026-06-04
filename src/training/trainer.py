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
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from utils.logger import get_logger
    from utils.utils_model import backbone_tag
except ImportError:
    from src.utils.logger import get_logger
    from src.utils.utils_model import backbone_tag

from data.dataset import CMYKDataset, ContrastiveDataset  # noqa: F401 — re-export
from data.normalize import _IMAGENET_NORMALIZE
from models.grayspot_model import GrayspotModel
from training.losses import get_loss

logger = get_logger(__name__)


def _save_normalize_meta(ckpt_path: Path, cfg: dict, channel: str) -> None:
    """체크포인트와 함께 정규화 메타데이터를 JSON으로 저장한다.
    Saves normalization metadata alongside the checkpoint as JSON.

    추론 시 동일한 정규화를 적용하기 위해 사용된다.
    Used to apply the exact same normalization during inference.

    저장 경로 / Save path: best_Y.pt → best_Y.meta.json
    """
    mean = list(_IMAGENET_NORMALIZE.mean)
    std = list(_IMAGENET_NORMALIZE.std)
    meta = {
        "normalize_mean": mean,
        "normalize_std": std,
        "image_size": int(cfg.get("data", {}).get("image_size", 128)),
        "channel": channel,
        "backbone": cfg.get("model", {}).get("backbone", "efficientnet_b0"),
    }
    meta_path = ckpt_path.parent / (ckpt_path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Optimizer / Scheduler factory helpers
# ──────────────────────────────────────────────────────────────


def _build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, cfg: dict):
    """
    cfg["train"]["optimizer"] 값에 따라 optimizer를 생성한다.
    Builds an optimizer based on cfg["train"]["optimizer"].

    지원 값 / Supported values: "adamw" (default), "sgd"
    """
    name = cfg["train"].get("optimizer", "adamw").lower()
    if name == "sgd":
        from torch.optim import SGD

        return SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=cfg["train"].get("momentum", 0.9),
        )
    else:  # adamw (default)
        betas = tuple(cfg["train"].get("betas", [0.9, 0.999]))
        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )


def _build_scheduler(optimizer, epochs: int, cfg: dict, phase: str = "train"):
    """
    cfg["train"]["scheduler"] 값에 따라 LR scheduler를 생성한다.
    Builds an LR scheduler based on cfg["train"]["scheduler"].

    warmup_epochs 설정이 있으면 LinearWarmup + 기본 스케줄러 조합.
    If warmup_epochs is set, combines LinearWarmup + base scheduler.

    지원 값 / Supported values: "cosine" (default), "step"
    phase: "phase0" | "phase2" | "train"
    """
    phase_cfg = cfg.get(phase, {}) if phase in ("phase0", "phase2") else {}
    warmup_epochs = int(phase_cfg.get("warmup_epochs", 0))

    name = cfg["train"].get("scheduler", "cosine").lower()
    if name == "step":
        from torch.optim.lr_scheduler import StepLR

        base_scheduler = StepLR(
            optimizer,
            step_size=max(1, epochs // 3),
            gamma=cfg["train"].get("gamma", 0.1),
        )
    else:  # cosine (default)
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup_epochs, 1),
            eta_min=cfg["train"]["eta_min"],
        )

    if warmup_epochs > 0:
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        warmup = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, base_scheduler], milestones=[warmup_epochs]
        )
    return base_scheduler


# ──────────────────────────────────────────────────────────────
# Phase 0 Trainer
# ──────────────────────────────────────────────────────────────


class Phase0Trainer:
    """
    Phase 0 Contrastive Learning 학습기.
    Phase 0 Contrastive Learning trainer.
    """

    def __init__(
        self, model: GrayspotModel, cfg: dict, channel: str, device: torch.device
    ):
        self.model = model
        self.cfg = cfg
        self.channel = channel
        self.device = device
        self.criterion = get_loss(phase=0, cfg=cfg)
        self.optimizer = _build_optimizer(
            model,
            lr=cfg["phase0"]["learning_rate"],
            weight_decay=cfg["phase0"].get("weight_decay", 0.0),
            cfg=cfg,
        )
        self.scheduler = _build_scheduler(
            self.optimizer, epochs=cfg["phase0"]["epochs"], cfg=cfg, phase="phase0"
        )

    def train(self, loader: DataLoader, optuna_trial=None) -> list[dict]:
        """학습 루프 실행 / Run training loop."""
        epochs = self.cfg["phase0"]["epochs"]
        history = []

        logger.info(
            f"{'='*55}\n  Phase 0 Training — Channel: [{self.channel}]\n{'='*55}"
        )
        logger.info(f"  {'Epoch':<8} {'Loss':<14} {'LR':<14} Elapsed")
        logger.info(f"  {'-'*50}")

        use_amp = self.cfg["train"].get(
            "mixed_precision", False
        ) and self.device.type in ("cuda", "mps")
        grad_accum = max(1, int(self.cfg["train"].get("grad_accumulation_steps", 1)))
        scaler = (
            torch.amp.GradScaler("cuda")
            if use_amp and self.device.type == "cuda"
            else None
        )
        grad_clip = self.cfg["train"].get("gradient_clip", 0.0)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            total_loss = 0.0
            self.optimizer.zero_grad()  # accum 방식: epoch 시작 시 초기화 / zero_grad at epoch start for accumulation

            for step, (view1, view2) in enumerate(loader):
                view1, view2 = view1.to(self.device), view2.to(self.device)

                if use_amp:
                    with torch.autocast(device_type=self.device.type):
                        loss = (
                            self.criterion(self.model(view1), self.model(view2))
                            / grad_accum
                        )
                else:
                    loss = (
                        self.criterion(self.model(view1), self.model(view2))
                        / grad_accum
                    )

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # accumulation step 완료 시 optimizer.step / optimizer.step when accumulation complete
                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    if grad_clip:
                        if scaler:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )

                    if scaler:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += (
                    loss.item() * grad_accum
                )  # 원래 scale로 기록 / record at original scale

            avg_loss = total_loss / max(len(loader), 1)
            self.scheduler.step()
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            history.append(
                {
                    "epoch": epoch,
                    "loss": round(avg_loss, 6),
                    "lr": round(lr, 8),
                    "elapsed": round(elapsed, 2),
                }
            )
            logger.info(f"  {epoch:<8} {avg_loss:<14.4f} {lr:<14.2e} {elapsed:.1f}s")

            # Optuna MedianPruner: 에폭별 중간 결과 보고 (Phase 0 — minimize loss)
            # Report per-epoch intermediate value for Optuna pruning (Phase 0 — minimize loss)
            if optuna_trial is not None:
                import optuna as _optuna

                optuna_trial.report(avg_loss, epoch)
                if optuna_trial.should_prune():
                    logger.info(f"  [Optuna] Trial pruned at epoch {epoch}")
                    raise _optuna.exceptions.TrialPruned()

        logger.info(
            f"  {'-'*50}\n  Done — Loss: {history[0]['loss']:.4f} → {history[-1]['loss']:.4f}"
        )
        return history

    def save_backbone(self) -> Path:
        """학습된 모델 state_dict 저장 / Save trained model state_dict."""
        models_dir = Path(self.cfg["storage"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        tag = backbone_tag(self.cfg["model"]["backbone"])
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

    def __init__(
        self,
        model: GrayspotModel,
        cfg: dict,
        channel: str,
        device: torch.device,
        train_ds: CMYKDataset = None,
    ):
        self.model = model
        self.channel = channel
        self.device = device

        # per-channel override를 cfg에 병합 / Merge per-channel overrides into cfg copy
        import copy

        self.cfg = copy.deepcopy(cfg)
        per = cfg.get("phase2", {}).get("per_channel", {}).get(channel, {})
        if per:
            for k, v in per.items():
                if k != "augmentation":
                    self.cfg["phase2"][k] = v
            if "augmentation" in per:
                self.cfg["phase2"].setdefault("augmentation", {}).update(
                    per["augmentation"]
                )
            logger.info(f"  [per-channel] Applying overrides for [{channel}]: {per}")

        # frozen_backbone 적용 / Apply frozen_backbone
        frozen = self.cfg["phase2"].get(
            "frozen_backbone", self.cfg.get("model", {}).get("frozen_backbone", False)
        )
        if frozen:
            for param in model.backbone.parameters():
                param.requires_grad = False
            logger.info(f"  [per-channel] Backbone frozen for [{channel}]")

        self.criterion = get_loss(
            phase=2, cfg=self.cfg, train_samples=train_ds.samples if train_ds else None
        )
        if hasattr(self.criterion, "weight") and self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(device)

        self.optimizer = _build_optimizer(
            model,
            lr=self.cfg["phase2"]["learning_rate"],
            weight_decay=self.cfg["phase2"]["weight_decay"],
            cfg=self.cfg,
        )
        self.scheduler = _build_scheduler(
            self.optimizer,
            epochs=self.cfg["phase2"]["epochs"],
            cfg=self.cfg,
            phase="phase2",
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optuna_trial=None,
    ) -> list[dict]:
        """학습 루프 실행 / Run training loop."""
        epochs = self.cfg["phase2"]["epochs"]
        models_dir = Path(self.cfg["storage"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        best_path = models_dir / f"best_{self.channel}.pt"

        es_cfg = self.cfg["phase2"].get("early_stopping", {})
        es_enabled = es_cfg.get("enabled", False)
        es_patience = es_cfg.get("patience", 10)
        es_delta = es_cfg.get("min_delta", 0.0)
        grad_clip = self.cfg["train"].get("gradient_clip", 0.0)

        history, best_val_acc, best_epoch, no_improve = [], 0.0, 0, 0

        logger.info(
            f"{'='*65}\n  Phase 2 Training — Channel: [{self.channel}]\n{'='*65}"
        )
        logger.info(
            f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<10} LR"
        )
        logger.info(f"  {'-'*60}")

        use_amp = self.cfg["train"].get(
            "mixed_precision", False
        ) and self.device.type in ("cuda", "mps")
        grad_accum = max(1, int(self.cfg["train"].get("grad_accumulation_steps", 1)))
        scaler = (
            torch.amp.GradScaler("cuda")
            if use_amp and self.device.type == "cuda"
            else None
        )

        # MixUp / CutMix 설정 / MixUp/CutMix config
        aug2_cfg = self.cfg.get("phase2", {}).get("augmentation", {})
        mixup_alpha = float(aug2_cfg.get("mixup_alpha", 0.0))
        cutmix_prob = float(aug2_cfg.get("cutmix_prob", 0.0))
        num_classes = int(self.cfg.get("data", {}).get("num_levels", 6))
        use_mixup = mixup_alpha > 0
        use_cutmix = cutmix_prob > 0

        import random as _random

        if use_mixup or use_cutmix:
            from data.augmentation import cutmix_batch, mixup_batch

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            self.optimizer.zero_grad()

            for step, (x, labels) in enumerate(train_loader):
                x, labels = x.to(self.device), labels.to(self.device)

                # MixUp / CutMix 적용 (배치 레벨) / Apply batch-level augmentation
                mixed_labels = None
                if use_cutmix and _random.random() < cutmix_prob:
                    x, mixed_labels = cutmix_batch(
                        x, labels, alpha=1.0, num_classes=num_classes
                    )
                elif use_mixup:
                    x, mixed_labels = mixup_batch(
                        x, labels, alpha=mixup_alpha, num_classes=num_classes
                    )

                if use_amp:
                    with torch.autocast(device_type=self.device.type):
                        logits = self.model(x)
                        if mixed_labels is not None:
                            import torch.nn.functional as _F

                            loss = _F.cross_entropy(logits, mixed_labels) / grad_accum
                        else:
                            loss = self.criterion(logits, labels) / grad_accum
                else:
                    logits = self.model(x)
                    if mixed_labels is not None:
                        import torch.nn.functional as _F

                        loss = _F.cross_entropy(logits, mixed_labels) / grad_accum
                    else:
                        loss = self.criterion(logits, labels) / grad_accum

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # accumulation step 완료 시 optimizer.step / optimizer.step when accumulation complete
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    if grad_clip:
                        if scaler:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )

                    if scaler:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss += loss.item() * grad_accum
                if mixed_labels is not None:
                    # soft label: argmax로 dominant class 비교
                    train_correct += (
                        (logits.argmax(1) == mixed_labels.argmax(1)).sum().item()
                    )
                else:
                    train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += len(labels)

            train_loss_avg = train_loss / max(len(train_loader), 1)
            train_acc = train_correct / max(train_total, 1)

            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for x, labels in val_loader:
                    x, labels = x.to(self.device), labels.to(self.device)
                    logits = self.model(x)
                    loss = self.criterion(logits, labels)
                    val_loss += loss.item()
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += len(labels)

            val_loss_avg = val_loss / max(len(val_loader), 1)
            val_acc = val_correct / max(val_total, 1)

            self.scheduler.step()
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss_avg, 6),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_loss_avg, 6),
                    "val_acc": round(val_acc, 4),
                    "lr": round(lr, 8),
                    "elapsed": round(elapsed, 2),
                }
            )

            if val_acc > best_val_acc + es_delta:
                best_val_acc, best_epoch = val_acc, epoch
                no_improve = 0
                torch.save(self.model.state_dict(), best_path)
                _save_normalize_meta(best_path, self.cfg, self.channel)
            else:
                no_improve += 1

            logger.info(
                f"  {epoch:<8} {train_loss_avg:<14.4f} {train_acc:<12.4f} "
                f"{val_loss_avg:<12.4f} {val_acc:<10.4f} {lr:.2e}"
            )

            # Optuna MedianPruner: 에폭별 중간 결과 보고 / Report per-epoch intermediate value
            if optuna_trial is not None:
                import optuna as _optuna

                optuna_trial.report(val_acc, epoch)
                if optuna_trial.should_prune():
                    logger.info(f"  [Optuna] Trial pruned at epoch {epoch}")
                    raise _optuna.exceptions.TrialPruned()

            if es_enabled and no_improve >= es_patience:
                logger.info(
                    f"  [Early Stop] patience={es_patience} 도달 / reached at epoch {epoch}"
                )
                break

        logger.info(f"  {'-'*60}")
        logger.info(f"  Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
        logger.info(f"  Model saved / 저장: {best_path}\n{'='*65}")
        return history

    def save_history(self, history: list[dict]) -> Path:
        """학습 이력 CSV 저장 / Save training history to CSV."""
        reports_dir = Path(self.cfg["storage"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / f"phase2_history_{self.channel}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        logger.info(f"  History saved / 저장: {csv_path}")
        return csv_path
