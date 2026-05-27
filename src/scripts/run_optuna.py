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

    평가 리포트 / Evaluation reports (Phase 2 only, per-channel):
    outputs/reports/optuna_phase2_{ch}/report_{ch}.json
    outputs/reports/optuna_phase2_{ch}/metrics_summary_{ch}.json
    outputs/reports/optuna_phase2_{ch}/{ch}.html

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
import sys
from pathlib import Path

from src.tuning.optuna_tuner import run_optuna, run_phase0_optuna
from src.utils.utils_config import load_config

# evaluate.py / generate_optuna_report.py 는 src/ 경로가 sys.path에 있어야 하므로
# 함수 내부에서 lazy import 한다.
# evaluate.py / generate_optuna_report.py require src/ in sys.path — lazy-imported inside functions.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

_ALL_CHANNELS = ["Y", "M", "C", "K"]


def _evaluate_channel(ch: str) -> None:
    """
    채널의 best 가중치를 Evaluator로 검증하고 리포트를 생성한다.
    Verifies best channel weights with Evaluator and generates a report.
    """
    from src.scripts.evaluate import run_evaluate

    cfg = load_config()
    models_dir = Path(cfg["storage"]["models_dir"])
    checkpoint = models_dir / f"best_{ch}.pt"
    reports_dir = Path(cfg["storage"]["reports_dir"]) / f"optuna_phase2_{ch.lower()}"

    print(f"\n  [{ch}] Evaluator 검증 시작 / Starting Evaluator verification...")
    try:
        report_path = run_evaluate(
            channel=ch,
            cfg=cfg,
            checkpoint=checkpoint,
            output_dir=reports_dir,
        )
        print(f"  [{ch}] 검증 완료 → {report_path}")
    except Exception as exc:
        print(f"  [{ch}] [WARN] Evaluator 실패 (가중치는 보존됨): {exc}")


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
    print(
        f"  Trials   : {args.trials if args.trials else 'config-defined'} (채널별 / per-channel)"
    )
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
            _evaluate_channel(ch)

    print("\n" + "=" * 60)
    print(
        f" Phase {args.phase} Optuna Finished — All channels: {' → '.join(target_channels)}"
    )
    print("=" * 60)

    # Phase 2 완료 후 통합 HTML 리포트 자동 생성
    # Auto-generate unified HTML report after all Phase 2 channels complete
    if args.phase == 2:
        print("\n[Report] 통합 Optuna HTML 리포트 생성 중...")
        try:
            import sys as _sys

            from src.scripts.generate_optuna_report import main as gen_report_main

            _argv_backup = _sys.argv
            _sys.argv = ["generate_optuna_report"]
            gen_report_main()
            _sys.argv = _argv_backup
        except Exception as exc:
            print(f"[WARN] 리포트 생성 실패 (결과 파일은 보존됨): {exc}")


if __name__ == "__main__":
    main()
