"""
scripts/run_optuna.py

Optuna Hyperparameter Tuning 실행 파일
Optuna Hyperparameter Tuning Execution Script

Phase 0 SimCLR 또는 Phase 2 Supervised Classification 하이퍼파라미터를 자동 탐색한다.
Performs automated hyperparameter optimization for Phase 0 or Phase 2.

최적화 완료 후 best params로 최종 재학습하여 실제 모델 가중치를 갱신한다.
After optimization, retrains with best params to update the actual model weights.

출력 / Outputs:
    outputs/optuna/
    ├── study_phase{N}_{ch}.db              ← Optuna 실험 DB
    ├── best_params_phase{N}_{ch}.json      ← 최적 하이퍼파라미터
    └── trials_summary_phase{N}_{ch}.json   ← 전체 trial 요약

    최적 가중치 / Best weights (final retrain):
    Phase 0: data_set/models/phase0_backbone_{ch}_{tag}.pt
    Phase 2: data_set/models/best_{ch}.pt

실행 / Run:
    Phase 2 전체 채널
    python -m src.scripts.run_optuna --phase 2

    Phase 2 단일 채널
    python -m src.scripts.run_optuna --phase 2 --channel C

    Phase 0 전체 채널
    python -m src.scripts.run_optuna --phase 0

    Phase 0 단일 채널 + trial 수 지정
    python -m src.scripts.run_optuna --phase 0 --channel Y --trials 10
"""

import argparse

from src.tuning.optuna_tuner import run_optuna, run_phase0_optuna


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna Hyperparameter Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=[0, 2],
        default=2,
        help="Phase to tune: 0 (SimCLR) or 2 (Supervised). Default: 2",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of trials (default: config value)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="all",
        help="Target channel: Y / M / C / K / all (default: all)",
    )

    args = parser.parse_args()
    channel = args.channel.upper()

    valid_channels = {"Y", "M", "C", "K", "ALL"}
    if channel not in valid_channels:
        raise ValueError(
            f"Unsupported channel: {args.channel}. Available: Y, M, C, K, all"
        )

    print("\n" + "=" * 60)
    print(" Optuna Hyperparameter Tuning")
    print("=" * 60)
    print(f"  Phase  : {args.phase}")
    print(f"  Channel: {channel}")
    print(f"  Trials : {args.trials if args.trials else 'config-defined'}")
    print("=" * 60)

    if args.phase == 0:
        run_phase0_optuna(n_trials=args.trials, channel=channel.lower())
    else:
        run_optuna(n_trials=args.trials, channel=channel.lower())

    print("\n" + "=" * 60)
    print(f" Phase {args.phase} Optuna Tuning + Final Retrain Finished")
    print("=" * 60)


if __name__ == "__main__":
    main()
