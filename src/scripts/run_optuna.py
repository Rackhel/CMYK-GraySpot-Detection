"""
scripts/run_optuna.py

Optuna Hyperparameter Tuning 실행 파일
Optuna Hyperparameter Tuning Execution Script

Baseline 모델을 기반으로 Phase 2 하이퍼파라미터를 자동 탐색한다.
Performs automated hyperparameter optimization for Phase 2 based on the baseline model.

사용자는 trial 수와 채널을 지정할 수 있다.
Users can specify the number of trials and target channel.

목적 / Purpose:
    Baseline 대비 성능 향상을 위한 최적의 하이퍼파라미터 탐색
    Find optimal hyperparameters to improve performance over the baseline

    단일 채널 또는 전체 채널 기준으로 유연하게 튜닝 수행
    Perform flexible tuning for a single channel or all channels

출력 / Outputs:
    outputs/optuna/
    ├── study_*.db              ← Optuna 실험 데이터베이스 / Optuna study database
    ├── best_params_*.json      ← 최적 하이퍼파라미터 / Best hyperparameters
    └── trials_summary_*.json   ← 전체 trial 결과 요약 / All trial results summary

실행 / Run:
    단일 채널(single-channel)
    python -m src.scripts.run_optuna --channel C
    전체 채널(All-channel)
    python -m src.scripts.run_optuna --channel all
    trial 지정(trial-select)
    python -m src.scripts.run_optuna --trials 10 --channel M
"""

import argparse

from src.tuning.optuna_tuner import run_optuna


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna Hyperparameter Tuning"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of trials (default: config value)"
    )

    parser.add_argument(
        "--channel",
        type=str,
        default="all",
        help="Target channel to tune: Y / M / C / K / all (default: all)"
    )

    args = parser.parse_args()
    channel = args.channel.upper()

    valid_channels = {"Y", "M", "C", "K", "ALL"}
    if channel not in valid_channels:
        raise ValueError(
            f"Unsupported channel: {args.channel}. "
            f"Available: Y, M, C, K, all"
        )

    print("\n======================================")
    print(" Starting Optuna Hyperparameter Tuning")
    print("======================================")
    print(f"Target Channel: {channel}")

    if args.trials is not None:
        print(f"Trials: {args.trials}")
    else:
        print("Trials: config-defined value")

    run_optuna(n_trials=args.trials, channel=channel.lower())

    print("\n======================================")
    print(" Optuna Tuning Finished")
    print("======================================")


if __name__ == "__main__":
    main()