import json
import sys
import types
import importlib
from pathlib import Path

import optuna
import torch

from src.tuning.search_space import get_phase2_search_space


# -------------------------------------------------------------------
# Compatibility shim for team imports
# 팀 코드(config_manager.py, grayspot_model.py 등)가
# `from utils import ...` 를 사용하므로,
# 현재 실행 환경에서 top-level `utils` 모듈을 직접 등록한다.
# -------------------------------------------------------------------
logger_mod = importlib.import_module("src.utils.logger")

utils_shim = types.ModuleType("utils")
utils_shim.LoggerMixin = logger_mod.LoggerMixin
utils_shim.get_logger = logger_mod.get_logger
utils_shim.setup_logging = logger_mod.setup_logging
utils_shim.log_training_config = logger_mod.log_training_config
utils_shim.log_epoch_summary = logger_mod.log_epoch_summary

sys.modules["utils"] = utils_shim


# 반드시 shim 등록 후 import
# Import only after shim registration
from src.scripts.run_baseline import load_config, run_baseline


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function
    Runs baseline training for each trial and returns best validation accuracy

    Optuna 목적 함수
    각 trial마다 baseline 학습을 실행하고 최고 validation accuracy를 반환
    """
    # Load configuration
    # 설정 불러오기
    config = load_config()
    cfg = config.config

    # Sample hyperparameters
    # 하이퍼파라미터 샘플링
    params = get_phase2_search_space(trial)

    # Apply sampled parameters to config
    # 샘플링한 파라미터를 config에 반영
    cfg["phase2"]["learning_rate"] = params["learning_rate"]
    cfg["phase2"]["batch_size"] = params["batch_size"]
    cfg["phase2"]["weight_decay"] = params["weight_decay"]
    cfg["phase2"]["epochs"] = params["epochs"]

    # Device setup
    # 디바이스 설정
    device = torch.device(config.get("system.device"))

    # Tune one channel first for speed
    # 속도를 위해 C 채널만 먼저 튜닝
    #result = run_baseline(cfg, channel="C", device=device)

    # Skip handling
    # 데이터가 없어서 skip된 경우 낮은 점수 반환
    #if result.get("skipped", False):
    #   return 0.0

    # Maximize best validation accuracy
    # 최고 validation accuracy 최대화
    #return float(result["best_val_acc"])
 
     # Tune all CMYK channels
    # CMYK 전체 채널 튜닝
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
    

def run_optuna(n_trials: int | None = None) -> None:
    """
    Run Optuna hyperparameter optimization

    Optuna 하이퍼파라미터 튜닝 실행
    """
    # Load config once for global settings
    # 전역 설정 확인을 위해 config 1회 로드
    config = load_config()
    cfg = config.config

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
    sampler = optuna.samplers.TPESampler(seed=cfg["train"].get("seed", 42))
    pruner = optuna.pruners.MedianPruner()

    # Create or load study
    # study 생성 또는 불러오기
    study = optuna.create_study(
        direction="maximize",
        study_name="phase2_tuning",
        storage="sqlite:///outputs/optuna/study.db",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # Run optimization
    # 최적화 실행
    study.optimize(objective, n_trials=n_trials)

    # Print best result
    # 최적 결과 출력
    print("\nBest Trial")
    print("Best Value (Val Acc):", study.best_value)
    print("Best Params:", study.best_trial.params)

    # Save best params
    # 최적 파라미터 저장
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, indent=2, ensure_ascii=False)

    # Save trial summary
    # trial 요약 저장
    trials_summary = []
    for t in study.trials:
        trials_summary.append({
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "params": t.params,
        })

    with open(output_dir / "trials_summary.json", "w", encoding="utf-8") as f:
        json.dump(trials_summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_optuna()