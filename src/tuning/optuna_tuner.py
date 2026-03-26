"""
Grayspot — Optuna 하이퍼파라미터 자동 최적화 / Automatic Hyperparameter Optimization
tuning/optuna_tuner.py

사용법 / Usage:
    from tuning.optuna_tuner import run_hpo
    best = run_hpo(cfg, phase=0, channel="Y")
    best = run_hpo(cfg, phase=2, channel="Y")
"""

import copy
import csv
import optuna
import yaml
from torch.utils.data import DataLoader
from pathlib import Path

from data.dataset import ContrastiveDataset, GrayspotDataset, compute_class_weights
from models.grayspot_model import GrayspotModel
from training.trainer import Phase0Trainer, Phase2Trainer

# Optuna 로그 레벨 설정 — WARNING만 출력 / Set Optuna log level — show WARNING only
optuna.logging.set_verbosity(optuna.logging.WARNING)

CHANNELS = ["Y", "M", "C", "K"]


# ──────────────────────────────────────────────
# Phase 0 — Contrastive Learning HPO
# ──────────────────────────────────────────────
def _phase0_objective(trial: optuna.Trial, cfg: dict, channel: str) -> float:
    """
    Phase 0 Optuna Objective 함수.
    탐색 대상: temperature, projection_dim, learning_rate, batch_size
    최적화 기준: Contrastive Loss 최솟값 (마지막 10 epoch 평균)

    Phase 0 Optuna Objective function.
    Search targets: temperature, projection_dim, learning_rate, batch_size
    Optimization criterion: Minimum Contrastive Loss (average of last 10 epochs)
    """
    sp = cfg["optuna"]["phase0_search_space"]

    # ── 탐색 공간 샘플링 / Sample from search space ──
    temperature    = trial.suggest_float("temperature",   *sp["temperature"],   log=True)
    projection_dim = trial.suggest_categorical("projection_dim", sp["projection_dim"])
    learning_rate  = trial.suggest_float("learning_rate", *sp["learning_rate"], log=True)
    batch_size     = trial.suggest_categorical("batch_size", sp["batch_size"])

    # cfg 깊은 복사 후 임시 수정 (원본 보존) / Deep copy cfg before modifying (preserve original)
    trial_cfg = _deep_copy_cfg(cfg)
    trial_cfg["phase0"]["temperature"]    = temperature
    trial_cfg["phase0"]["projection_dim"] = projection_dim
    trial_cfg["phase0"]["learning_rate"]  = learning_rate
    trial_cfg["phase0"]["batch_size"]     = batch_size
    trial_cfg["phase0"]["epochs"]         = 30   # HPO는 빠른 수렴 확인용으로 축약 실행 / Shortened for quick convergence check

    dataset = ContrastiveDataset(trial_cfg, channel)
    if len(dataset) == 0:
        raise optuna.TrialPruned()  # 데이터 없으면 Trial 건너뜀 / Skip trial if no data

    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model   = GrayspotModel(trial_cfg, phase=0)
    trainer = Phase0Trainer(model, trial_cfg, channel)

    history = trainer.train(loader)

    # 마지막 10 epoch 평균 loss를 목적 함수 값으로 사용
    # Use average loss of last 10 epochs as objective value
    last_losses = [h["loss"] for h in history[-10:]]
    return sum(last_losses) / len(last_losses)


# ──────────────────────────────────────────────
# Phase 2 — Supervised Classification HPO
# ──────────────────────────────────────────────
def _phase2_objective(trial: optuna.Trial, cfg: dict, channel: str) -> float:
    """
    Phase 2 Optuna Objective 함수.
    탐색 대상: learning_rate, weight_decay, dropout_rate, fc_hidden_dim, backbone_freeze, backbone_arch
    최적화 기준: Val Accuracy 최댓값

    Phase 2 Optuna Objective function.
    Search targets: learning_rate, weight_decay, dropout_rate, fc_hidden_dim, backbone_freeze, backbone_arch
    Optimization criterion: Maximum Validation Accuracy
    """
    sp = cfg["optuna"]["phase2_search_space"]

    # ── 탐색 공간 샘플링 / Sample from search space ──
    learning_rate   = trial.suggest_float("learning_rate",  *sp["learning_rate"],  log=True)
    weight_decay    = trial.suggest_float("weight_decay",   *sp["weight_decay"],   log=True)
    dropout_rate    = trial.suggest_float("dropout_rate",   *sp["dropout_rate"])
    fc_hidden_dim   = trial.suggest_int("fc_hidden_dim",    *sp["fc_hidden_dim"],  step=64)
    backbone_freeze = trial.suggest_categorical("backbone_freeze", sp["backbone_freeze"])
    backbone_arch   = trial.suggest_categorical("backbone_arch",   sp["backbone_arch"])

    # cfg 깊은 복사 후 임시 수정 / Deep copy cfg before modifying
    trial_cfg = _deep_copy_cfg(cfg)
    trial_cfg["phase2"]["learning_rate"]   = learning_rate
    trial_cfg["phase2"]["weight_decay"]    = weight_decay
    trial_cfg["phase2"]["dropout_rate"]    = dropout_rate
    trial_cfg["phase2"]["fc_hidden_dim"]   = fc_hidden_dim
    trial_cfg["phase2"]["backbone_freeze"] = backbone_freeze
    trial_cfg["phase2"]["stage1_epochs"]   = 10   # HPO 축약 실행 / Shortened for HPO
    trial_cfg["phase2"]["stage2_epochs"]   = 10
    trial_cfg["model"]["backbone"]         = backbone_arch

    train_ds = GrayspotDataset(trial_cfg, channel, split="train", augment=True)
    val_ds   = GrayspotDataset(trial_cfg, channel, split="val",   augment=False)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise optuna.TrialPruned()  # 데이터 없으면 Trial 건너뜀 / Skip if no data

    train_loader = DataLoader(train_ds, batch_size=trial_cfg["phase2"]["batch_size"],
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=trial_cfg["phase2"]["batch_size"],
                              shuffle=False)

    class_weights = (
        compute_class_weights(train_ds)
        if trial_cfg["phase2"]["class_weights"] == "balanced" else None
    )

    model   = GrayspotModel(trial_cfg, phase=2)
    trainer = Phase2Trainer(model, trial_cfg, channel, class_weights)
    trainer.train(train_loader, val_loader)

    return trainer.best_val_acc  # 최고 검증 정확도 반환 / Return best validation accuracy


# ──────────────────────────────────────────────
# HPO 실행 진입점 / HPO Entry Point
# ──────────────────────────────────────────────
def run_hpo(cfg: dict, phase: int, channel: str) -> dict:
    """
    Optuna HPO를 실행하고 최적 하이퍼파라미터를 반환한다.
    Runs Optuna HPO and returns the best hyperparameters.

    Args:
        cfg:     config.yaml 딕셔너리 / config.yaml dictionary
        phase:   0 또는 2 / 0 or 2
        channel: "Y" | "M" | "C" | "K"

    Returns:
        {
          "best_params": {...},          # 최적 파라미터 / Best parameters
          "best_value":  0.94,           # 최적 목적 함수 값 / Best objective value
          "n_trials":    50,             # 실행된 trial 수 / Number of executed trials
        }

    Example:
        >>> import yaml
        >>> cfg = yaml.safe_load(open("config/config.yaml"))
        >>> result = run_hpo(cfg, phase=2, channel="Y")
        >>> print(result["best_params"])
    """
    assert phase in (0, 2)
    assert channel in CHANNELS

    opt_cfg   = cfg["optuna"]
    direction = "minimize" if phase == 0 else "maximize"  # Phase 0: loss 최소화, Phase 2: accuracy 최대화

    print(f"\n🔬  Optuna HPO 시작 / Starting — Phase {phase} | Channel: {channel}")
    print(f"    Trials: {opt_cfg['n_trials']} | Direction: {direction}\n")

    study = optuna.create_study(
        direction=direction,
        study_name=f"grayspot_phase{phase}_{channel}",
        sampler=optuna.samplers.TPESampler(seed=42),          # TPE 알고리즘 / TPE algorithm
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),  # 조기 종료 / Early stopping
    )

    objective = _phase0_objective if phase == 0 else _phase2_objective

    study.optimize(
        lambda trial: objective(trial, cfg, channel),
        n_trials=opt_cfg["n_trials"],
        timeout=opt_cfg["timeout"],      # 최대 탐색 시간 제한 / Max search time limit
        show_progress_bar=True,
    )

    best   = study.best_trial
    result = {
        "phase":       phase,
        "channel":     channel,
        "best_params": best.params,
        "best_value":  round(best.value, 4),
        "n_trials":    len(study.trials),
    }

    print(f"\n🏆  Best Trial #{best.number}")
    print(f"    Value  : {best.value:.4f}")
    print(f"    Params : {best.params}")

    # 최적 파라미터를 config에 반영하여 저장 / Apply best params to config and save
    _apply_best_params(cfg, phase, channel, best.params)
    _save_hpo_result(result, cfg)

    return result


def _apply_best_params(cfg: dict, phase: int, channel: str, params: dict) -> None:
    """최적 파라미터를 config에 반영하고 새 yaml 파일로 저장한다. / Applies best params to config and saves as new yaml."""
    phase_key = f"phase{phase}"
    for k, v in params.items():
        if k in cfg[phase_key]:
            cfg[phase_key][k] = v
        elif k == "backbone_arch":
            cfg["model"]["backbone"] = v  # backbone_arch → model.backbone으로 매핑 / Map backbone_arch to model.backbone

    # 채널별 최적 config 저장 / Save per-channel optimal config
    out_path = Path("config") / f"config_best_phase{phase}_{channel}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    print(f"    최적 config 저장 / Best config saved: {out_path}")


def _save_hpo_result(result: dict, cfg: dict) -> None:
    """HPO 결과를 CSV로 저장한다. / Saves HPO results to CSV."""
    reports_dir = Path(cfg["storage"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"hpo_result_phase{result['phase']}_{result['channel']}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param", "value"])       # 헤더 / Header
        for k, v in result["best_params"].items():
            writer.writerow([k, v])
        writer.writerow(["best_value", result["best_value"]])  # 최적 목적 함수 값 / Best objective value
        writer.writerow(["n_trials",   result["n_trials"]])    # 총 trial 수 / Total trials
    print(f"    HPO 결과 저장 / HPO result saved: {path}")


def _deep_copy_cfg(cfg: dict) -> dict:
    """config 딕셔너리를 깊은 복사한다. / Returns a deep copy of the config dictionary."""
    return copy.deepcopy(cfg)