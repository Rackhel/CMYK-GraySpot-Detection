"""
Grayspot — 평가 메트릭 (Phase 3)
evaluation/metrics.py

Phase 3 평가 및 Swing 피드백 루프 판단을 담당한다.
Handles Phase 3 evaluation and Swing feedback loop decision-making.
"""

import json
import csv
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, mean_absolute_error,
)

CHANNELS = ["Y", "M", "C", "K"]


def evaluate_channel(
    y_true: list[int],
    y_pred: list[int],
    channel: str,
    num_levels: int = 6,
) -> dict:
    """
    채널 1개에 대한 전체 평가 메트릭을 산출한다.
    Computes all evaluation metrics for a single channel.

    Returns:
        {
          "channel":       "Y",
          "accuracy":      0.92,          # 정확도 / Accuracy
          "macro_f1":      0.88,          # 매크로 F1 / Macro F1
          "mae":           0.12,          # 평균 절대 오차 / Mean Absolute Error
          "per_class_f1":  [0.95, 0.88, ...],  # 레벨별 F1 / Per-level F1
          "confusion":     [[...], ...],  # 혼동 행렬 / Confusion matrix
        }
    """
    labels = list(range(num_levels))

    accuracy     = accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro",  labels=labels, zero_division=0)
    mae          = mean_absolute_error(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None,    labels=labels, zero_division=0).tolist()
    cm           = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "channel":      channel,
        "accuracy":     round(accuracy, 4),
        "macro_f1":     round(macro_f1, 4),
        "mae":          round(mae, 4),
        "per_class_f1": [round(f, 4) for f in per_class_f1],
        "confusion":    cm,
    }


def evaluate_all_channels(
    results: dict[str, dict],  # {"Y": {"y_true":[], "y_pred":[]}, ...}
    cfg: dict,
) -> dict:
    """
    CMYK 4채널 전체 평가 + Phase 3 피드백 루프 판단.
    Evaluates all 4 CMYK channels and determines Phase 3 Swing feedback actions.

    Returns:
        {
          "per_channel":      {"Y": {...}, ...},  # 채널별 메트릭 / Per-channel metrics
          "overall_accuracy": 0.91,               # 전체 정확도 / Overall accuracy
          "swing_decision":   {"Y": "pass", "M": "phase1", ...},  # Swing 판단 / Swing decisions
          "targets_met":      True | False,        # 목표 달성 여부 / Whether all targets are met
        }
    """
    tgt   = cfg["evaluation"]["targets"]
    fb    = cfg["evaluation"]["swing_feedback"]
    n_lvl = cfg["data"]["num_levels"]

    per_channel              = {}
    all_true, all_pred = [], []

    for ch in CHANNELS:
        y_true   = results[ch]["y_true"]
        y_pred   = results[ch]["y_pred"]
        metrics  = evaluate_channel(y_true, y_pred, ch, n_lvl)
        per_channel[ch] = metrics
        all_true.extend(y_true)
        all_pred.extend(y_pred)

    overall_accuracy = accuracy_score(all_true, all_pred)

    # ── Phase 3 피드백 루프 판단 / Phase 3 Feedback Loop Decision ──
    swing_decision = {}
    for ch, m in per_channel.items():

        # 특정 색상 Accuracy < 80% → Phase 0 복귀
        # Per-color accuracy below threshold → return to Phase 0
        if m["accuracy"] < fb["color_phase0_threshold"]:
            swing_decision[ch] = "phase0"

        # MAE > 0.8 → Phase 0 복귀 / MAE above threshold → return to Phase 0
        elif m["mae"] > fb["mae_phase0_threshold"]:
            swing_decision[ch] = "phase0"

        # 특정 Level F1 < 0.70 → Phase 1 복귀
        # Any per-level F1 below threshold → return to Phase 1
        elif any(f < fb["f1_phase1_threshold"] for f in m["per_class_f1"]):
            swing_decision[ch] = "phase1"

        else:
            swing_decision[ch] = "pass"  # 목표 달성 / Target met

    # 전체 목표 달성 여부 확인 / Check if all targets are met
    targets_met = (
        overall_accuracy >= tgt["overall_accuracy"] and
        all(m["macro_f1"] >= tgt["per_class_f1"]       for m in per_channel.values()) and
        all(m["accuracy"] >= tgt["per_color_accuracy"]  for m in per_channel.values()) and
        all(m["mae"]      <= tgt["mae"]                 for m in per_channel.values())
    )

    return {
        "per_channel":      per_channel,
        "overall_accuracy": round(overall_accuracy, 4),
        "swing_decision":   swing_decision,
        "targets_met":      targets_met,
    }


def print_evaluation_report(eval_result: dict) -> None:
    """평가 결과를 터미널에 출력한다. / Prints evaluation results to the terminal."""
    print("\n" + "="*60)
    print("  📊  Phase 3 — 평가 리포트 / Evaluation Report")
    print("="*60)
    print(f"\n  Overall Accuracy: {eval_result['overall_accuracy']:.4f}")
    print(f"  Targets Met:      {' 달성 / Met' if eval_result['targets_met'] else ' 미달 / Not met'}\n")

    for ch, m in eval_result["per_channel"].items():
        decision = eval_result["swing_decision"][ch]
        icon     = {"pass": "", "phase1": " →Phase1", "phase0": " →Phase0"}[decision]
        print(f"  [{ch}] Acc: {m['accuracy']:.3f} | F1: {m['macro_f1']:.3f} | "
              f"MAE: {m['mae']:.3f} | {icon}")

    # 레벨별 F1 출력 / Print per-level F1
    print("\n  Per-class F1 (Level 0~5):")
    for ch, m in eval_result["per_channel"].items():
        f1_str = " ".join(f"L{i}:{f:.2f}" for i, f in enumerate(m["per_class_f1"]))
        print(f"  [{ch}] {f1_str}")
    print()


def save_evaluation_results(eval_result: dict, cfg: dict) -> None:
    """평가 결과를 CSV와 JSON으로 저장한다. / Saves evaluation results to CSV and JSON."""
    reports_dir = Path(cfg["storage"]["reports_dir"])

    # JSON — 전체 결과 저장 / Save full results as JSON
    json_path = reports_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    # CSV — 채널별 요약 저장 / Save per-channel summary as CSV
    csv_path = reports_dir / cfg["reporting"]["csv_files"]["evaluation_results"]
    rows = []
    for ch, m in eval_result["per_channel"].items():
        rows.append({
            "channel":  ch,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "mae":      m["mae"],
            "swing":    eval_result["swing_decision"][ch],
        })
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"    평가 결과 저장 / Evaluation results saved: {csv_path}")