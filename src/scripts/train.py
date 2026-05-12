"""
scripts/train.py

Grayspot S2 통합 실행 스크립트 / Unified execution script for S2.

학습 → 평가 → 리포트 자동 생성 파이프라인.
Automated pipeline: training → evaluation → report generation.

S2 완료 기준 / S2 completion criteria:
    - Naive Baseline 학습 + Test셋 성능 기록
    - baseline_summary.json 생성
    - baseline.html 리포트 생성
    - (선택) Optuna 튜닝

실행 / Run:
    # 기본 — Baseline 학습 + 리포트
    python src/scripts/train.py

    # 채널 지정
    python src/scripts/train.py --channel C

    # Optuna 튜닝 포함
    python src/scripts/train.py --channel all --optuna --trials 10

    # 리포트만 재생성 (학습 skip)
    python src/scripts/train.py --report-only

    # 브라우저 자동 열기
    python src/scripts/train.py --open-browser

출력 / Outputs:
    data_set/baseline/
    ├── best_{channel}.pt
    ├── phase2_history_{channel}.csv
    └── baseline_summary.json

    outputs/reports/
    └── baseline.html

    outputs/optuna/              ← --optuna 옵션 시
    ├── best_params_{ch}.json
    └── trials_summary_{ch}.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from datetime import datetime
from pathlib import Path

# ── 경로 설정 / Path setup ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # CMYK_MAIN/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ── utils shim (팀 코드 호환) / utils shim for team code compatibility ──────
# evaluator.py 등이 `from utils import ...` 를 사용하므로
# top-level utils 모듈을 등록한다.
_logger_mod = importlib.import_module("src.utils.logger")
_utils_shim = types.ModuleType("utils")
_utils_shim.LoggerMixin = _logger_mod.LoggerMixin
_utils_shim.get_logger = _logger_mod.get_logger
_utils_shim.setup_logging = _logger_mod.setup_logging
_utils_shim.log_training_config = _logger_mod.log_training_config
_utils_shim.log_epoch_summary = _logger_mod.log_epoch_summary
sys.modules["utils"] = _utils_shim

# ── 이후 임포트 / Imports after shim ──────────────────────────────────────
from src.utils import (create_directories, get_logger, get_nested, load_config,
                       setup_logging, validate_config)

CHANNELS = ["Y", "M", "C", "K"]


# ---------------------------------------------------------------------------
# Step 1 — Naive Baseline 학습 / Naive Baseline training
# ---------------------------------------------------------------------------


def step_train(cfg, channels: list[str], device, logger) -> bool:
    """
    run_baseline 을 채널별로 실행한다.
    Runs run_baseline per channel.

    Returns:
        True if all channels succeeded, False otherwise.
    """
    import json
    from pathlib import Path

    import torch

    from src.scripts.run_baseline import run_baseline
    from src.utils import set_seed

    logger.info("=" * 60)
    logger.info("STEP 1 — Naive Baseline Training")
    logger.info(f"  Channels : {channels}")
    logger.info("=" * 60)

    set_seed(cfg["train"].get("seed") or 42, cfg)

    baseline_dir = Path(cfg["storage"]["data_root"]) / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ch in channels:
        result = run_baseline(cfg, ch, device)
        results.append(result)

    # 요약 출력 / Print summary
    logger.info("\n" + "=" * 60)
    logger.info("  Baseline 성능 요약 / Baseline Performance Summary")
    logger.info("=" * 60)
    logger.info(
        f"  {'Channel':<10} {'Test Acc':<12} {'MAE':<10} {'Val Acc':<10} Acc Pass"
    )
    logger.info(f"  {'─' * 50}")

    for r in results:
        if r.get("skipped"):
            logger.info(f"  {r['channel']:<10} SKIPPED")
            continue
        mark = "[PASS]" if r["pass_acc"] else "[FAIL]"
        logger.info(
            f"  {r['channel']:<10} {r['test_acc']:<12.4f} "
            f"{r['mae']:<10.4f} {r['best_val_acc']:<10.4f} {mark}"
        )

    # baseline_summary.json 저장 / Save summary
    summary = {
        "mode": "Naive Baseline (Supervised-only, no Phase 0)",
        "backbone": cfg["model"]["backbone"],
        "epochs": cfg["phase2"]["epochs"],
        "results": results,
    }
    summary_path = baseline_dir / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  Summary saved: {summary_path}")

    return True


# ---------------------------------------------------------------------------
# Step 2 — Optuna 튜닝 (선택) / Optuna tuning (optional)
# ---------------------------------------------------------------------------


def step_optuna(channels: list[str], n_trials: int, logger) -> bool:
    """
    run_optuna 를 채널별로 실행한다.
    Runs run_optuna per channel.
    """
    from src.tuning.optuna_tuner import run_optuna

    logger.info("=" * 60)
    logger.info("STEP 2 — Optuna Hyperparameter Tuning")
    logger.info(f"  Channels : {channels}")
    logger.info(f"  Trials   : {n_trials}")
    logger.info("=" * 60)

    for ch in channels:
        logger.info(f"  [{ch}] Starting Optuna tuning...")
        run_optuna(n_trials=n_trials, channel=ch.lower())
        logger.info(f"  [{ch}] Optuna tuning done")

    return True


# ---------------------------------------------------------------------------
# Step 3 — 리포트 생성 / Report generation
# ---------------------------------------------------------------------------


def step_report(cfg, channels: list[str], device, open_browser: bool, logger) -> bool:
    """
    generate_baseline_report 의 핵심 로직을 실행한다.
    Runs the core logic of generate_baseline_report.
    """
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn

    from evaluation.confusion import plot_confusion_matrix
    from evaluation.evaluator import Evaluator
    from evaluation.metrics import (NUM_LEVELS, TARGET_MAE,
                                    TARGET_PER_COLOR_ACC, compute_all_channels)
    from src.scripts.generate_baseline_report import (build_baseline_html,
                                                      load_baseline_summary,
                                                      summary_to_cards)
    from src.utils import build_model

    logger.info("=" * 60)
    logger.info("STEP 3 — Report Generation")
    logger.info("=" * 60)

    baseline_dir = Path(cfg["storage"]["data_root"]) / "baseline"
    labeled_dir = Path(cfg["storage"]["labeled_dir"])
    labels_csv = Path(cfg["storage"]["data_root"]) / "labels_v0.csv"
    output_dir = ROOT / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # baseline_summary.json 로드 (공식 Test셋 수치)
    summary = load_baseline_summary(baseline_dir)
    available = [
        r["channel"]
        for r in summary["results"]
        if not r.get("skipped") and r["channel"] in channels
    ]

    if not available:
        logger.error("No valid channels in baseline_summary.json.")
        return False

    cards = summary_to_cards(summary, available)

    # 전체 데이터 추론 (차트용)
    logger.info("Running full-dataset inference for charts...")
    all_results: dict = {}

    for ch in available:
        ckpt = baseline_dir / f"best_{ch}.pt"
        if not ckpt.exists():
            logger.warning(f"[{ch}] Checkpoint not found — skipping")
            continue

        model = build_model(cfg, ckpt, device)
        ev = Evaluator(
            model=model,
            labeled_dir=labeled_dir,
            labels_csv=labels_csv,
            output_dir=output_dir / "tmp",
            device=device,
            image_size=cfg["data"]["image_size"],
            batch_size=cfg["phase2"]["batch_size"],
            num_levels=cfg["data"]["num_levels"],
            cfg=cfg,
        )
        all_results.update(ev.run(channels=[ch]))

    metrics = compute_all_channels(
        all_results,
        list(all_results.keys()),
        num_classes=cfg["data"]["num_levels"],
    )

    # 차트 생성
    ev_report = Evaluator(
        model=build_model(cfg, baseline_dir / f"best_{available[0]}.pt", device),
        labeled_dir=labeled_dir,
        labels_csv=labels_csv,
        output_dir=output_dir,
        device=device,
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["phase2"]["batch_size"],
        num_levels=cfg["data"]["num_levels"],
        cfg=cfg,
    )

    df_miss = ev_report.get_misclassified(all_results, list(all_results.keys()))
    figs: dict = {}
    figs["per_class"] = ev_report._build_per_class_chart(metrics)
    figs["mae"] = ev_report._build_mae_heatmap(all_results, list(all_results.keys()))
    figs["mismatch"] = ev_report._build_mismatch_scatter(df_miss)
    figs["conf_dist"] = ev_report._build_confidence_dist(
        all_results, list(all_results.keys())
    )

    all_true = np.concatenate([all_results[c]["y_true"] for c in all_results])
    all_pred = np.concatenate([all_results[c]["y_pred"] for c in all_results])

    for ch in all_results:
        yt, yp = all_results[ch]["y_true"], all_results[ch]["y_pred"]
        figs[f"cm_{ch}"] = plot_confusion_matrix(
            yt,
            yp,
            title=f'[{ch}] Confusion Matrix  Full-dataset Acc={metrics[ch]["accuracy"]:.4f}',
            normalize=True,
        )
    figs["cm_overall"] = plot_confusion_matrix(
        all_true,
        all_pred,
        title=f'[Overall] Confusion Matrix  Acc={metrics["overall"]["accuracy"]:.4f}',
        normalize=True,
    )

    # Phase 3 판단 (baseline_summary.json 공식 수치 기반)
    lines = ["=== Phase 3 Feedback Decision (PRD 3.3.2) ==="]
    all_pass = all(cards[c]["pass_acc"] and cards[c]["pass_mae"] for c in available)

    if all_pass:
        lines.append("All targets met -- TERMINATE Swing")
    else:
        lines.append("Action required:")
        for ch in available:
            c = cards[ch]
            if not c["pass_acc"]:
                lines.append(
                    f'  [{ch}] Test Acc {c["test_acc"]:.3f} < {TARGET_PER_COLOR_ACC}'
                    " -> Phase 0 (retrain representation)"
                )
            if not c["pass_mae"]:
                lines.append(
                    f'  [{ch}] MAE {c["mae"]:.3f} > {TARGET_MAE}'
                    " -> Phase 0 (representation learning retry)"
                )

    overall = cards.get("overall", {})
    lines += [
        "",
        f'  Avg. Test Accuracy : {overall.get("test_acc", 0):.4f}  (target >= {TARGET_PER_COLOR_ACC})',
        f'  Avg. MAE           : {overall.get("mae", 0):.4f}  (target <= {TARGET_MAE})',
        f"  Source             : baseline_summary.json (Test-set 15%)",
    ]
    phase3_text = "\n".join(lines)

    # CSV / JSON 저장
    ev_report.save_csv(
        all_results, experiment_name="baseline", channels=list(all_results.keys())
    )
    ev_report.save_json(
        metrics,
        experiment_name="baseline",
        channels=list(all_results.keys()),
        checkpoint_path=", ".join(f"best_{ch}.pt" for ch in available),
    )
    miss_csv = output_dir / "misclassified_baseline.csv"
    df_miss.to_csv(miss_csv, index=False, encoding="utf-8-sig")

    # HTML 리포트
    meta = {
        "experiment": "baseline",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": summary.get("backbone", cfg["model"]["backbone"]),
        "epochs": summary.get("epochs", cfg["phase2"]["epochs"]),
        "image_size": cfg["data"]["image_size"],
        "channels": available,
    }

    baseline_html = output_dir / "baseline.html"
    build_baseline_html(
        report_data={
            "meta": meta,
            "cards": cards,
            "metrics": metrics,
            "figures": figs,
            "phase3_text": phase3_text,
            "misclassified": df_miss,
        },
        output_path=baseline_html,
    )

    logger.info(f"  Report saved: {baseline_html}")
    print("\n" + phase3_text)

    if open_browser:
        import webbrowser

        webbrowser.open(baseline_html.resolve().as_uri())

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grayspot S2 통합 실행 / Unified S2 pipeline"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="all",
        help="채널 / Channel (Y/M/C/K/all, default: all)",
    )
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Optuna 튜닝 실행 여부 / Run Optuna tuning",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Optuna trial 수 / Number of Optuna trials (default: 5)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="리포트만 재생성 (학습 skip) / Skip training, regenerate report only",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="완료 후 브라우저 열기 / Open browser after completion",
    )
    args = parser.parse_args()

    # 채널 결정 / Resolve channels
    target_channels = (
        CHANNELS if args.channel.lower() == "all" else [args.channel.upper()]
    )

    # config 로드 / Load config
    cfg = load_config()
    if not validate_config(cfg):
        print("[ERROR] Configuration validation failed.")
        sys.exit(1)

    create_directories(cfg)

    # 로깅 설정 / Setup logging
    setup_logging(
        log_dir=__import__("pathlib").Path(cfg["storage"]["logs_dir"]),
        level=get_nested(cfg, "logging.level") or "INFO",
        format_style=get_nested(cfg, "logging.format") or "detailed",
        console=get_nested(cfg, "logging.console_output"),
        file=get_nested(cfg, "logging.file_output"),
    )
    logger = get_logger(__name__)

    import torch

    device = torch.device(cfg["system"]["device"])

    # 시작 배너 / Start banner
    print()
    print("=" * 60)
    print("  Grayspot S2 — Unified Training Pipeline")
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print("=" * 60)
    print(f"  Channels     : {target_channels}")
    print(f"  Device       : {device}")
    print(f'  Backbone     : {cfg["model"]["backbone"]}')
    print(f"  Report only  : {args.report_only}")
    print(f"  Optuna       : {args.optuna} (trials={args.trials})")
    print("=" * 60)
    print()

    start_time = __import__("time").time()
    steps_done = []

    # STEP 1 — Baseline 학습 / Baseline training
    if not args.report_only:
        ok = step_train(cfg, target_channels, device, logger)
        steps_done.append(("Baseline Training", ok))
        if not ok:
            logger.error("Baseline training failed. Aborting.")
            sys.exit(1)
    else:
        logger.info("--report-only: Skipping training step.")
        steps_done.append(("Baseline Training", "SKIPPED"))

    # STEP 2 — Optuna 튜닝 (선택) / Optuna tuning (optional)
    if args.optuna and not args.report_only:
        ok = step_optuna(target_channels, args.trials, logger)
        steps_done.append(("Optuna Tuning", ok))
    else:
        steps_done.append(("Optuna Tuning", "SKIPPED"))

    # STEP 3 — 리포트 생성 / Report generation
    ok = step_report(cfg, target_channels, device, args.open_browser, logger)
    steps_done.append(("Report Generation", ok))

    # 완료 요약 / Completion summary
    elapsed = __import__("time").time() - start_time
    print()
    print("=" * 60)
    print("  Pipeline Complete")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 60)
    for step, result in steps_done:
        if result == "SKIPPED":
            mark = "  —  "
        elif result:
            mark = " [OK] "
        else:
            mark = "[FAIL]"
        print(f"  {mark}  {step}")
    print()
    print(f"  Report → outputs/reports/baseline.html")
    print(f"  Summary → data_set/baseline/baseline_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
