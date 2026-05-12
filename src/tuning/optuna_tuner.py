"""
tuning/optuna_tuner.py

Optuna 기반 하이퍼파라미터 튜닝 모듈
Optuna-based hyperparameter tuning module

Baseline 학습 파이프라인(run_baseline)을 재사용하여
Phase 2 하이퍼파라미터를 자동 탐색한다.
Reuses the baseline training pipeline (run_baseline)
to automatically search Phase 2 hyperparameters.

지원 모드 / Supported modes:
    - 단일 채널 튜닝 / Single-channel tuning
      (Y / M / C / K)
    - 전체 채널 평균 튜닝 / All-channel average tuning
      (Y + M + C + K)

탐색 대상 / Search targets:
    - learning_rate
    - batch_size
    - weight_decay
    - epochs

목적 / Purpose:
    Baseline 대비 성능 향상을 위한 최적의 하이퍼파라미터 탐색
    Find optimal hyperparameters to improve performance over the baseline

    단일 채널 또는 전체 채널 기준으로 유연한 실험 수행
    Perform flexible experiments for a single channel or all channels

출력 / Outputs:
    outputs/optuna/
    ├── study_{channel}.db            ← Optuna 실험 데이터베이스 / Optuna study database
    ├── best_params_{channel}.json    ← 최적 하이퍼파라미터 / Best hyperparameters
    └── trials_summary_{channel}.json ← 전체 trial 결과 요약 / All trial results summary

실행 / Run:
    python -m src.scripts.run_optuna
    python -m src.scripts.run_optuna --channel C
    python -m src.scripts.run_optuna --channel all
    python -m src.scripts.run_optuna --trials 10 --channel M
"""

import json
from functools import partial
from pathlib import Path

import optuna
import torch

from src.tuning.search_space import get_phase2_search_space


def objective(trial: optuna.Trial, channel: str) -> float:
    """
    Optuna objective function
    Runs baseline training and returns validation accuracy

    Optuna 목적 함수
    Baseline 학습을 실행하고 validation accuracy를 반환
    """
    from src.utils import load_config
    from src.scripts.run_baseline import run_baseline

    cfg = load_config()

    # Sample hyperparameters
    # 하이퍼파라미터 샘플링
    params = get_phase2_search_space(trial, cfg)

    # Apply sampled parameters to config
    # 샘플링한 파라미터를 config에 반영
    cfg["phase2"]["learning_rate"] = params["learning_rate"]
    cfg["phase2"]["batch_size"] = params["batch_size"]
    cfg["phase2"]["weight_decay"] = params["weight_decay"]
    cfg["phase2"]["epochs"] = params["epochs"]

    # backbone별 head 파라미터를 phase2.heads에 반영
    # Apply backbone-specific head params into phase2.heads
    backbone_name = cfg["model"]["backbone"]
    if "heads" not in cfg["phase2"]:
        cfg["phase2"]["heads"] = {}
    if backbone_name not in cfg["phase2"]["heads"]:
        cfg["phase2"]["heads"][backbone_name] = {}

    cfg["phase2"]["heads"][backbone_name]["dropout"] = params["dropout"]
    cfg["phase2"]["heads"][backbone_name]["hidden_dim"] = params["hidden_dim"]
    if "mid_dim" in params:
        cfg["phase2"]["heads"][backbone_name]["mid_dim"] = params["mid_dim"]

    # Device setup
    # 디바이스 설정
    device = torch.device(cfg["system"]["device"])

    # Single-channel tuning
    # 단일 채널 튜닝
    if channel != "all":
        result = run_baseline(cfg, channel=channel.upper(), device=device)

        # Skip handling
        # 데이터가 없어서 skip된 경우 낮은 점수 반환
        if result.get("skipped", False):
            return 0.0

        return float(result["best_val_acc"])

    # All-channel tuning
    # 전체 채널 튜닝
    channels = ["Y", "M", "C", "K"]
    scores = []

    for ch in channels:
        result = run_baseline(cfg, channel=ch, device=device)

        # Skip channels with no training data
        # 학습 데이터가 없는 채널은 건너뜀
        if result.get("skipped", False):
            continue

        scores.append(result["best_val_acc"])

    # If all channels were skipped, return 0.0
    # 모든 채널이 skip되면 0.0 반환
    if len(scores) == 0:
        return 0.0

    # Return average validation accuracy across channels
    # 전체 채널 평균 validation accuracy 반환
    return float(sum(scores) / len(scores))


def run_optuna(n_trials: int | None = None, channel: str = "all") -> None:
    """
    Run Optuna hyperparameter optimization

    Optuna 하이퍼파라미터 튜닝 실행
    """
    from src.utils import load_config
    from src.scripts.run_baseline import run_baseline  # noqa: F401

    channel = channel.lower()

    # Load config once for global settings
    # 전역 설정 확인을 위해 config 1회 로드
    cfg = load_config()

    # Output directory
    # 결과 저장 폴더
    output_dir = Path("outputs/optuna")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Trial count
    # trial 수 결정: 인자로 주면 우선, 아니면 config 사용
    if n_trials is None:
        n_trials = int(cfg.get("optuna", {}).get("n_trials", 5))

    # Sampler / Pruner
    # 탐색기 및 조기 종료 설정
    seed = cfg["train"].get("seed", 42)
    sampler_name = cfg.get("optuna", {}).get("sampler", "tpe").lower()
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:  # tpe (default)
        sampler = optuna.samplers.TPESampler(seed=seed)
    pruner_cfg = cfg.get("optuna", {}).get("pruner", {})
    n_warmup_steps = int(pruner_cfg.get("n_warmup_steps", 10))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)

    # Direction
    # 최적화 방향 (config에서 읽기)
    direction = cfg.get("optuna", {}).get("direction", "maximize")

    # Study name / DB path
    # 채널별로 study 분리
    study_suffix = channel if channel != "all" else "all"
    study_name = f"phase2_tuning_{study_suffix}"
    storage_path = f"sqlite:///outputs/optuna/study_{study_suffix}.db"

    # Create or load study
    # study 생성 또는 불러오기
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # Bind channel to objective
    # objective에 channel 고정 전달
    objective_fn = partial(objective, channel=channel)

    # Run optimization
    # 최적화 실행
    study.optimize(objective_fn, n_trials=n_trials)

    # Print best result
    # 최적 결과 출력
    print("\nBest Trial")
    print("Target Channel:", channel.upper())
    print("Best Value (Val Acc):", study.best_value)
    print("Best Params:", study.best_trial.params)

    # Save best params
    # 최적 파라미터 저장
    best_params_path = output_dir / f"best_params_{study_suffix}.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, indent=2, ensure_ascii=False)

    # Save trial summary
    # trial 요약 저장
    trials_summary = []
    for t in study.trials:
        trials_summary.append(
            {
                "number": t.number,
                "value": t.value,
                "state": str(t.state),
                "params": t.params,
            }
        )

    trials_summary_path = output_dir / f"trials_summary_{study_suffix}.json"
    with open(trials_summary_path, "w", encoding="utf-8") as f:
        json.dump(trials_summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_optuna()
