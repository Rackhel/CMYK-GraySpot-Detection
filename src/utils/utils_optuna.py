"""
tuning/optuna_utils.py

Optuna artifact utility functions.
Optuna 산출물 유틸리티 함수.

Purpose:
    Load saved best hyperparameters from outputs/optuna.

Contract:
    Input:
        channel: Y / M / C / K / all
    Output:
        dict: best hyperparameter values

Fail-Fast:
    Raise FileNotFoundError if best_params file does not exist.
    Raise ValueError if channel is invalid.
"""

from __future__ import annotations

import json
from pathlib import Path

VALID_CHANNELS = {"Y", "M", "C", "K", "ALL"}


def normalize_channel(channel: str) -> str:
    """Normalize channel string to lowercase suffix."""
    ch = channel.upper()
    if ch not in VALID_CHANNELS:
        raise ValueError(f"Unsupported channel: {channel}. Available: Y, M, C, K, all")
    return ch.lower()


def load_best_params(channel: str, output_dir: str | Path = "outputs/optuna") -> dict:
    """
    Load best Optuna parameters for a channel.

    Args:
        channel: Y / M / C / K / all
        output_dir: Optuna output directory

    Returns:
        Best parameter dictionary
    """
    suffix = normalize_channel(channel)
    path = Path(output_dir) / f"best_params_{suffix}.json"

    if not path.exists():
        raise FileNotFoundError(f"Best params file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_best_params(params: dict, channel: str, output_dir: str | Path = "outputs/optuna") -> Path:
    suffix = normalize_channel(channel)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"best_params_{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    return path


def save_trials_summary(trials: list, channel: str, output_dir: str | Path = "outputs/optuna") -> Path:
    suffix = normalize_channel(channel)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_summary = []
    for t in trials:
        trials_summary.append(
            {
                "number": t.number,
                "value": t.value,
                "state": str(t.state),
                "params": t.params,
            }
        )

    path = output_dir / f"trials_summary_{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trials_summary, f, indent=2, ensure_ascii=False)

    return path

def apply_phase2_params(cfg: dict, params: dict) -> dict:
    """
    Apply Optuna best params to phase2 config.

    Returns:
        cfg dict with updated phase2 hyperparameters.
    """
    required = {
        "learning_rate",
        "batch_size",
        "weight_decay",
        "epochs",
        "dropout",
        "hidden_dim",
    }

    missing = required - set(params.keys())
    if missing:
        raise KeyError(f"Missing required Optuna params: {sorted(missing)}")

    cfg["phase2"]["learning_rate"] = params["learning_rate"]
    cfg["phase2"]["batch_size"] = params["batch_size"]
    cfg["phase2"]["weight_decay"] = params["weight_decay"]
    cfg["phase2"]["epochs"] = params["epochs"]

    backbone_name = cfg["model"]["backbone"]

    cfg["phase2"]["heads"][backbone_name]["dropout"] = params["dropout"]
    cfg["phase2"]["heads"][backbone_name]["hidden_dim"] = params["hidden_dim"]

    if "mid_dim" in params:
        cfg["phase2"]["heads"][backbone_name]["mid_dim"] = params["mid_dim"]

    return cfg