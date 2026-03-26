"""
Grayspot -- 그래프 생성 / Graph Generation
reporting/plots.py

평가 결과를 시각화하는 그래프를 생성하고 저장한다.
Generates and saves graphs visualizing evaluation results.

생성 그래프 목록 / Generated graphs:
    - Confusion Matrix 히트맵 / Confusion Matrix heatmap
    - Per-class F1 바 차트 / Per-class F1 bar chart
    - 학습 이력 곡선 / Training history curve (loss / accuracy)
    - 채널별 정확도 비교 / Per-channel accuracy comparison

사용법 / Usage:
    from reporting.plots import (
        plot_confusion_matrix,
        plot_f1_bars,
        plot_training_history,
        plot_channel_comparison,
    )
"""

import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 한글 폰트 설정 (없으면 기본 폰트 사용)
# Set Korean font (fallback to default if unavailable)
_KO_FONTS = ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]
for _font in _KO_FONTS:
    if any(_font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _font
        break
matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지 / Prevent minus sign corruption

CHANNELS   = ["Y", "M", "C", "K"]
LEVEL_NAMES = ["L0\nNormal", "L1\nVery\nSubtle", "L2\nSlight",
               "L3\nClear", "L4\nSevere", "L5\nCritical"]

# 채널별 색상 / Per-channel colors
CHANNEL_COLORS = {
    "Y": "#f9a825",
    "M": "#c2185b",
    "C": "#0277bd",
    "K": "#424242",
}


# ──────────────────────────────────────────────
# Confusion Matrix 히트맵 / Confusion Matrix Heatmap
# ──────────────────────────────────────────────
def plot_confusion_matrix(
    confusion: list[list[int]],
    channel: str,
    cfg: dict,
    save: bool = True,
) -> Path | None:
    """
    Confusion Matrix를 히트맵으로 시각화한다.
    Visualizes the Confusion Matrix as a heatmap.

    Args:
        confusion: 혼동 행렬 (num_levels x num_levels) / Confusion matrix
        channel:   "Y" | "M" | "C" | "K"
        cfg:       config.yaml 딕셔너리 / config.yaml dictionary
        save:      파일 저장 여부 / Whether to save to file

    Returns:
        저장된 파일 경로 또는 None / Saved file path or None

    Example:
        >>> from evaluation.metrics import evaluate_channel
        >>> m = evaluate_channel(y_true, y_pred, "Y")
        >>> plot_confusion_matrix(m["confusion"], "Y", cfg)
    """
    cm         = np.array(confusion)
    num_levels = cfg["data"]["num_levels"]
    labels     = [f"L{i}" for i in range(num_levels)]

    fig, ax = plt.subplots(figsize=(7, 6))

    # 정규화된 히트맵 / Normalized heatmap
    cm_norm = cm.astype(float)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sum, where=row_sum != 0)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 셀 텍스트 표시 / Display cell text
    for i in range(num_levels):
        for j in range(num_levels):
            count    = cm[i, j]
            norm_val = cm_norm[i, j]
            color    = "white" if norm_val > 0.5 else "black"
            ax.text(j, i, f"{count}\n({norm_val:.0%})",
                    ha="center", va="center", color=color, fontsize=8)

    ax.set_xticks(range(num_levels))
    ax.set_yticks(range(num_levels))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Level / 예측 레벨", fontsize=11)
    ax.set_ylabel("True Level / 실제 레벨", fontsize=11)
    ax.set_title(f"Confusion Matrix -- Channel [{channel}]", fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save:
        out_path = _get_plot_path(cfg, f"confusion_{channel}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    plt.show()
    plt.close(fig)
    return None


# ──────────────────────────────────────────────
# Per-class F1 바 차트 / Per-class F1 Bar Chart
# ──────────────────────────────────────────────
def plot_f1_bars(
    per_class_f1: list[float],
    channel: str,
    cfg: dict,
    save: bool = True,
) -> Path | None:
    """
    레벨별 F1 점수를 바 차트로 시각화한다.
    Visualizes per-level F1 scores as a bar chart.

    Args:
        per_class_f1: 레벨별 F1 점수 리스트 / Per-level F1 score list
        channel:      "Y" | "M" | "C" | "K"
        cfg:          config.yaml 딕셔너리 / config.yaml dictionary
        save:         파일 저장 여부 / Whether to save to file

    Returns:
        저장된 파일 경로 또는 None / Saved file path or None
    """
    target_f1  = cfg["evaluation"]["targets"]["per_class_f1"]
    num_levels = cfg["data"]["num_levels"]
    x          = np.arange(num_levels)
    color      = CHANNEL_COLORS.get(channel, "#607d8b")

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(x, per_class_f1, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)

    # 목표선 / Target line
    ax.axhline(y=target_f1, color="red", linestyle="--", linewidth=1.5,
               label=f"Target F1 >= {target_f1:.2f}")

    # 바 위에 수치 표시 / Show values above bars
    for bar, val in zip(bars, per_class_f1):
        color_text = "green" if val >= target_f1 else "red"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9, color=color_text, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_NAMES[:num_levels], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Level", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title(f"Per-class F1 Score -- Channel [{channel}]", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save:
        out_path = _get_plot_path(cfg, f"f1_bars_{channel}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    plt.show()
    plt.close(fig)
    return None


# ──────────────────────────────────────────────
# 학습 이력 곡선 / Training History Curve
# ──────────────────────────────────────────────
def plot_training_history(cfg: dict, channel: str, save: bool = True) -> Path | None:
    """
    training_history.csv를 읽어 Loss / Accuracy 학습 곡선을 시각화한다.
    Reads training_history.csv and visualizes Loss / Accuracy training curves.

    Args:
        cfg:     config.yaml 딕셔너리 / config.yaml dictionary
        channel: "Y" | "M" | "C" | "K"
        save:    파일 저장 여부 / Whether to save to file

    Returns:
        저장된 파일 경로 또는 None / Saved file path or None
    """
    reports_dir  = Path(cfg["storage"]["reports_dir"])
    history_path = reports_dir / cfg["reporting"]["csv_files"]["training_history"]

    if not history_path.exists():
        print(f"  training_history.csv 없음 / Not found: {history_path}")
        return None

    # CSV 로드 / Load CSV
    rows = []
    with open(history_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("channel") == channel and row.get("phase") == "2":
                rows.append(row)

    if not rows:
        print(f"  [{channel}] Phase 2 학습 이력 없음 / No Phase 2 training history")
        return None

    epochs     = [int(r["epoch"])        for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss   = [float(r["val_loss"])   for r in rows]
    train_acc  = [float(r["train_acc"])  for r in rows]
    val_acc    = [float(r["val_acc"])    for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    color = CHANNEL_COLORS.get(channel, "#607d8b")

    # Loss 곡선 / Loss curve
    ax1.plot(epochs, train_loss, label="Train Loss", color=color,     linewidth=2)
    ax1.plot(epochs, val_loss,   label="Val Loss",   color=color, linestyle="--", linewidth=2, alpha=0.7)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss",  fontsize=11)
    ax1.set_title(f"Training Loss -- Channel [{channel}]", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Accuracy 곡선 / Accuracy curve
    target_acc = cfg["evaluation"]["targets"]["per_color_accuracy"]
    ax2.plot(epochs, train_acc, label="Train Acc", color=color,     linewidth=2)
    ax2.plot(epochs, val_acc,   label="Val Acc",   color=color, linestyle="--", linewidth=2, alpha=0.7)
    ax2.axhline(y=target_acc, color="red", linestyle=":", linewidth=1.5,
                label=f"Target >= {target_acc:.0%}")
    ax2.set_xlabel("Epoch",    fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title(f"Training Accuracy -- Channel [{channel}]", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Stage 경계선 표시 / Show stage boundary
    s1_end = cfg["phase2"]["stage1_epochs"]
    for ax in (ax1, ax2):
        ax.axvline(x=s1_end, color="gray", linestyle=":", alpha=0.6,
                   label=f"Stage 2 start (ep {s1_end+1})")

    fig.suptitle(f"Training History -- Channel [{channel}]", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        out_path = _get_plot_path(cfg, f"training_history_{channel}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    plt.show()
    plt.close(fig)
    return None


# ──────────────────────────────────────────────
# 채널별 정확도 비교 / Per-channel Accuracy Comparison
# ──────────────────────────────────────────────
def plot_channel_comparison(
    eval_result: dict,
    cfg: dict,
    save: bool = True,
) -> Path | None:
    """
    CMYK 4채널의 Accuracy / F1 / MAE를 한 화면에 비교한다.
    Compares Accuracy / F1 / MAE across all 4 CMYK channels in one view.

    Args:
        eval_result: evaluate_all_channels() 반환값 / Return value of evaluate_all_channels()
        cfg:         config.yaml 딕셔너리 / config.yaml dictionary
        save:        파일 저장 여부 / Whether to save to file

    Returns:
        저장된 파일 경로 또는 None / Saved file path or None
    """
    per_channel = eval_result.get("per_channel", {})
    channels    = [ch for ch in CHANNELS if ch in per_channel]

    if not channels:
        print("  평가 결과 없음 / No evaluation results")
        return None

    tgt = cfg["evaluation"]["targets"]

    accuracy = [per_channel[ch]["accuracy"] for ch in channels]
    f1       = [per_channel[ch]["macro_f1"] for ch in channels]
    mae      = [per_channel[ch]["mae"]      for ch in channels]
    colors   = [CHANNEL_COLORS[ch]          for ch in channels]

    x     = np.arange(len(channels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    # Accuracy 비교 / Accuracy comparison
    bars = axes[0].bar(x, accuracy, width=0.5, color=colors, alpha=0.85, edgecolor="white")
    axes[0].axhline(y=tgt["per_color_accuracy"], color="red", linestyle="--",
                    linewidth=1.5, label=f"Target >= {tgt['per_color_accuracy']:.0%}")
    axes[0].set_ylim(0, 1.15)
    _annotate_bars(axes[0], bars, accuracy, fmt="{:.3f}")
    axes[0].set_title("Per-channel Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels, fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Macro F1 비교 / Macro F1 comparison
    bars = axes[1].bar(x, f1, width=0.5, color=colors, alpha=0.85, edgecolor="white")
    axes[1].axhline(y=tgt["per_class_f1"], color="red", linestyle="--",
                    linewidth=1.5, label=f"Target >= {tgt['per_class_f1']:.2f}")
    axes[1].set_ylim(0, 1.15)
    _annotate_bars(axes[1], bars, f1, fmt="{:.3f}")
    axes[1].set_title("Per-channel Macro F1", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels, fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    # MAE 비교 / MAE comparison
    bars = axes[2].bar(x, mae, width=0.5, color=colors, alpha=0.85, edgecolor="white")
    axes[2].axhline(y=tgt["mae"], color="red", linestyle="--",
                    linewidth=1.5, label=f"Target <= {tgt['mae']:.2f}")
    _annotate_bars(axes[2], bars, mae, fmt="{:.3f}")
    axes[2].set_title("Per-channel MAE", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(channels, fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(axis="y", alpha=0.3)

    overall = eval_result.get("overall_accuracy", 0)
    fig.suptitle(
        f"Channel Comparison  |  Overall Accuracy: {overall:.4f}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save:
        out_path = _get_plot_path(cfg, "channel_comparison.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    plt.show()
    plt.close(fig)
    return None


# ──────────────────────────────────────────────
# 전체 그래프 일괄 생성 / Generate All Plots
# ──────────────────────────────────────────────
def generate_all_plots(eval_result: dict, cfg: dict) -> list[Path]:
    """
    평가 결과 기반 모든 그래프를 일괄 생성한다.
    Generates all graphs based on evaluation results.

    Args:
        eval_result: evaluate_all_channels() 반환값 / Return value of evaluate_all_channels()
        cfg:         config.yaml 딕셔너리 / config.yaml dictionary

    Returns:
        생성된 파일 경로 목록 / List of generated file paths

    Example:
        >>> paths = generate_all_plots(eval_result, cfg)
        >>> for p in paths:
        ...     print(p)
    """
    paths       = []
    per_channel = eval_result.get("per_channel", {})

    for ch in CHANNELS:
        if ch not in per_channel:
            continue
        m = per_channel[ch]

        # Confusion Matrix
        p = plot_confusion_matrix(m["confusion"], ch, cfg)
        if p:
            paths.append(p)
            print(f"  Confusion Matrix [{ch}]: {p.name}")

        # F1 Bar Chart
        p = plot_f1_bars(m["per_class_f1"], ch, cfg)
        if p:
            paths.append(p)
            print(f"  F1 Bar Chart [{ch}]: {p.name}")

        # Training History
        p = plot_training_history(cfg, ch)
        if p:
            paths.append(p)
            print(f"  Training History [{ch}]: {p.name}")

    # 채널 비교 / Channel comparison
    p = plot_channel_comparison(eval_result, cfg)
    if p:
        paths.append(p)
        print(f"  Channel Comparison: {p.name}")

    print(f"\n  총 {len(paths)}개 그래프 생성 완료 / {len(paths)} graphs generated")
    return paths


# ──────────────────────────────────────────────
# 내부 헬퍼 / Internal Helpers
# ──────────────────────────────────────────────
def _get_plot_path(cfg: dict, filename: str) -> Path:
    """그래프 저장 경로를 반환한다. / Returns the save path for a graph."""
    out_dir = Path(cfg["storage"]["reports_dir"]) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def _annotate_bars(ax, bars, values: list[float], fmt: str = "{:.3f}") -> None:
    """바 차트 위에 수치를 표시한다. / Annotates bar values above each bar."""
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            fmt.format(val),
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )