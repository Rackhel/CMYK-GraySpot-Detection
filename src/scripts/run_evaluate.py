"""
scripts/run_evaluate.py

Grayspot 모델 평가 실행 스크립트 (독립 CLI 진입점).
Standalone CLI entry point for Grayspot model evaluation.

evaluate.py 의 run_evaluate() 를 채널별로 순차 실행하고
채널별 JSON + HTML 리포트를 생성한다.
Runs run_evaluate() per channel sequentially and generates
per-channel JSON + HTML reports.

출력 / Outputs:
    outputs/reports/{experiment}/
    ├── report_{ch}.json            ← 평가 지표 요약 / Metrics summary
    ├── metrics_summary_{ch}.json   ← 상세 지표 / Detailed metrics
    └── {ch}.html                   ← Plotly 대시보드 / Plotly dashboard

실행 / Run:
    # 전체 채널 (Y → M → C → K)
    python -m src.scripts.run_evaluate

    # 단일 채널
    python -m src.scripts.run_evaluate --channel Y

    # 체크포인트 직접 지정
    python -m src.scripts.run_evaluate --channel Y --checkpoint data_set/models/best_Y.pt

    # 출력 경로 지정
    python -m src.scripts.run_evaluate --output-dir outputs/reports/my_eval

    # 브라우저 자동 오픈
    python -m src.scripts.run_evaluate --channel Y --open-browser

Python 3.11.5 | macOS & Windows compatible
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# evaluate.py 는 src/ 경로에 의존하므로 미리 추가한다.
# evaluate.py depends on src/ being in sys.path.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from src.scripts.evaluate import run_evaluate
from src.utils.utils_config import load_config

_ALL_CHANNELS = ["Y", "M", "C", "K"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_evaluate",
        description="Grayspot 모델 평가 CLI / Grayspot model evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="all",
        metavar="{Y,M,C,K,all}",
        help="평가 채널 (기본: all → Y→M→C→K 순차 실행) / Channel to evaluate (default: all)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "모델 체크포인트 .pt 경로. 미지정 시 models_dir/best_{ch}.pt 자동 탐색. / "
            "Model checkpoint path. Omit to auto-resolve models_dir/best_{ch}.pt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help=(
            "리포트 저장 경로 (기본: config storage.reports_dir) / "
            "Report output directory (default: config storage.reports_dir)"
        ),
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="평가 완료 후 HTML 리포트를 브라우저로 자동 오픈 / Open HTML report in browser after evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    channel = args.channel.upper()
    valid = {"Y", "M", "C", "K", "ALL"}
    if channel not in valid:
        print(
            f"[ERROR] Unsupported channel: '{args.channel}'. Available: Y, M, C, K, all"
        )
        sys.exit(1)

    target_channels = _ALL_CHANNELS if channel == "ALL" else [channel]

    cfg = load_config()
    output_dir: Path | None = args.output_dir
    if output_dir is None:
        output_dir = Path(cfg.get("storage", {}).get("reports_dir", "outputs/reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --checkpoint 는 단일 채널일 때만 의미 있음 / --checkpoint only meaningful for single channel
    if args.checkpoint is not None and len(target_channels) > 1:
        print(
            "[WARN] --checkpoint 는 단일 채널 실행 시에만 적용됩니다 / "
            "--checkpoint is only applied when a single channel is specified."
        )

    print("\n" + "=" * 60)
    print(" Grayspot Evaluation")
    print("=" * 60)
    print(f"  Channels   : {' → '.join(target_channels)}")
    print(f"  Output dir : {output_dir}")
    if args.checkpoint:
        print(f"  Checkpoint : {args.checkpoint}")
    print("=" * 60)

    last_report: Path | None = None

    for i, ch in enumerate(target_channels, 1):
        print(f"\n[{i}/{len(target_channels)}] Channel: {ch}")

        # 단일 채널 실행 시 --checkpoint 사용, 다중 채널 시 None (자동 탐색)
        ckpt = args.checkpoint if len(target_channels) == 1 else None

        try:
            report_path = run_evaluate(
                channel=ch,
                cfg=cfg,
                checkpoint=ckpt,
                output_dir=output_dir,
            )
            print(f"  → {report_path}")
            last_report = report_path
        except FileNotFoundError as exc:
            print(f"  [ERROR] {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"  [WARN] 평가 실패 / Evaluation failed [{ch}]: {exc}")

    print("\n" + "=" * 60)
    print(f" Evaluation Finished — {' → '.join(target_channels)}")
    if last_report:
        print(f" Last report: {last_report}")
    print("=" * 60)

    if args.open_browser and last_report is not None:
        import webbrowser

        # HTML 리포트 경로 추론: JSON 리포트와 같은 디렉토리
        html_path = last_report.parent / f"{target_channels[-1]}.html"
        target = html_path if html_path.exists() else last_report
        webbrowser.open(target.resolve().as_uri())


if __name__ == "__main__":
    main()
