import json
from pathlib import Path

import optuna
import torch

from src.tuning.search_space import get_phase2_search_space
from src.scripts.run_baseline import load_config, run_baseline


def objective(trial):
    """
    Optuna objective function
    → Runs training for each trial and returns validation accuracy

    Optuna 목적 함수
    → 각 trial마다 학습을 수행하고 validation accuracy를 반환
    """

    # Load configuration
    # config 불러오기
    config = load_config()
    cfg = config.config

    # Sample hyperparameters from search space
    # search space에서 하이퍼파라미터 샘플링
    params = get_phase2_search_space(trial)

    # Apply sampled parameters to config
    # 샘플링된 파라미터를 config에 반영
    cfg["phase2"]["learning_rate"] = params["learning_rate"]
    cfg["phase2"]["batch_size"] = params["batch_size"]
    cfg["phase2"]["weight_decay"] = params["weight_decay"]
    cfg["phase2"]["epochs"] = params["epochs"]

    # Set device (CPU / GPU)
    # 디바이스 설정 (CPU 또는 GPU)
    device = torch.device(config.get("system.device"))

    # Run baseline training for one channel (for speed)
    # 속도를 위해 하나의 채널만 먼저 튜닝
    result = run_baseline(cfg, channel="C", device=device)

    # Return validation accuracy (objective to maximize)
    # validation accuracy 반환 (최대화 목표)
    return result["best_val_acc"]


def run_optuna(n_trials=5):
    """
    Run Optuna hyperparameter optimization

    Optuna 하이퍼파라미터 튜닝 실행 함수
    """

    # Create output directory
    # 결과 저장 폴더 생성
    output_dir = Path("outputs/optuna")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Optuna study (database included)
    # Optuna study 생성 (DB 저장 포함)
    study = optuna.create_study(
        direction="maximize",
        study_name="phase2_tuning",
        storage="sqlite:///outputs/optuna/study.db",
        load_if_exists=True,
    )

    # Run optimization
    # 최적화 실행
    study.optimize(objective, n_trials=n_trials)

    # Print best result
    # 최적 결과 출력
    print("\n🔥 Best Trial")
    print("Best Value (Val Acc):", study.best_value)
    print("Best Params:", study.best_trial.params)

    # Save best parameters as JSON
    # 최적 파라미터를 JSON 파일로 저장
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_optuna()