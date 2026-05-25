"""
scripts/run_optuna.py

Optuna Hyperparameter Tuning 실행 파일
Optuna Hyperparameter Tuning Execution Script

Phase 0 SimCLR 또는 Phase 2 Supervised Classification 하이퍼파라미터를 자동 탐색한다.
Performs automated hyperparameter optimization for Phase 0 or Phase 2.

--channel all (기본값) 시 Y → M → C → K 채널을 순차적으로 독립 탐색한다.
채널마다 독립된 탐색 공간으로 최적 파라미터를 찾고, best params로 최종 재학습한다.
When --channel all (default), searches each channel independently: Y → M → C → K.
Each channel gets its own optimal parameters, then retrains with best params.

출력 / Outputs:
    outputs/optuna/
    ├── study_phase{N}_{ch}.db              ← Optuna 실험 DB (채널별 / per-channel)
    ├── best_params_phase{N}_{ch}.json      ← 최적 하이퍼파라미터 (채널별 / per-channel)
    └── trials_summary_phase{N}_{ch}.json   ← 전체 trial 요약 (채널별 / per-channel)

    최적 가중치 / Best weights (final retrain after each channel):
    Phase 0: data_set/models/phase0_backbone_{ch}_{tag}.pt
    Phase 2: data_set/models/best_{ch}.pt

실행 / Run:
    Phase 2 전체 채널 순차 실행 (Y → M → C → K)
    python -m src.scripts.run_optuna --phase 2

    Phase 2 단일 채널
    python -m src.scripts.run_optuna --phase 2 --channel C

    Phase 0 전체 채널 순차 실행
    python -m src.scripts.run_optuna --phase 0

    trial 수 지정
    python -m src.scripts.run_optuna --phase 2 --trials 30
"""

import argparse

from src.tuning.optuna_tuner import run_optuna, run_phase0_optuna

_ALL_CHANNELS = ["Y", "M", "C", "K"]


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna Hyperparameter Tuning (per-channel independent search)",
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
        help="Number of trials per channel (default: config value)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="all",
        help="Target channel: Y / M / C / K / all (default: all → Y→M→C→K 순차 실행)",
    )

    args = parser.parse_args()
    channel = args.channel.upper()

    valid_channels = {"Y", "M", "C", "K", "ALL"}
    if channel not in valid_channels:
        raise ValueError(
            f"Unsupported channel: {args.channel}. Available: Y, M, C, K, all"
        )

    # 실행 대상 채널 목록 결정 / Determine target channels
    target_channels = _ALL_CHANNELS if channel == "ALL" else [channel]

    print("\n" + "=" * 60)
    print(" Optuna Hyperparameter Tuning")
    print("=" * 60)
    print(f"  Phase    : {args.phase}")
    print(f"  Channels : {' → '.join(target_channels)}")
    print(f"  Trials   : {args.trials if args.trials else 'config-defined'} (채널별 / per-channel)")
    print("=" * 60)

    # 채널별 독립 순차 실행 / Run independently per channel in sequence
    for i, ch in enumerate(target_channels, 1):
        print(f"\n{'=' * 60}")
        print(f"  [{i}/{len(target_channels)}] Channel: {ch}")
        print(f"{'=' * 60}")

        if args.phase == 0:
            run_phase0_optuna(n_trials=args.trials, channel=ch.lower())
        else:
            run_optuna(n_trials=args.trials, channel=ch.lower())

    print("\n" + "=" * 60)
    print(f" Phase {args.phase} Optuna Finished — All channels: {' → '.join(target_channels)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
