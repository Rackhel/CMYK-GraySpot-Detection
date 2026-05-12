"""
evaluation/confusion.py

Confusion Matrix 생성 모듈.
Confusion Matrix generation module.

PRD Section 5.6.2, 8.2.3 에서 요구하는 6x6 혼동 행렬을 생성한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from evaluation.metrics import NUM_LEVELS

# ---------------------------------------------------------------------------
# Visualization constants used by html_report.py and evaluator.py
# ---------------------------------------------------------------------------

CMYK_COLORS: dict = {
    "Y": "#f5e642",
    "M": "#e91e8c",
    "C": "#00b4d8",
    "K": "#444444",
}

PLOTLY_TEMPLATE: str = "plotly_dark"
FONT_FAMILY: str = "Segoe UI, Tahoma, Geneva, Verdana, sans-serif"
FONT_SIZE: int = 12


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    num_classes: int = NUM_LEVELS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    6x6 혼동 행렬을 계산한다. / Computes the 6x6 confusion matrix.

    Args:
        y_true      : 정답 라벨 / True labels (N,)
        y_pred      : 예측 라벨 / Predicted labels (N,)
        normalize   : True 이면 행 정규화 (비율), False 이면 카운트
                      True = row-normalized (proportion), False = raw count
        num_classes : 클래스 수 / Number of classes (default: 6)

    Returns:
        cm_raw  : raw count matrix
        cm_norm : row-normalized (or same as cm_raw if normalize=False)
    """
    labels = list(range(num_classes))
    cm_raw = sk_confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm_raw / row_sums, 0.0)
    else:
        cm_norm = cm_raw.astype(float)

    return cm_raw, cm_norm


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    normalize: bool = True,
    output_path: Optional[str] = None,
    num_classes: int = NUM_LEVELS,
) -> go.Figure:
    """
    6x6 혼동 행렬 Plotly 히트맵을 생성한다.
    Generates a 6x6 confusion matrix as a Plotly heatmap.
    Args:
        y_true      : 정답 라벨 / True labels (N,)
        y_pred      : 예측 라벨 / Predicted labels (N,)
        title       : 차트 제목 / Chart title
        normalize   : 행 정규화 여부 / Whether to row-normalize
        output_path : HTML 파일 저장 경로 (None 이면 저장 안함)
                      HTML save path (None = do not save)
        num_classes : 클래스 수 / Number of classes (default: 6)

    Returns:
        go.Figure
    """
    labels = list(range(num_classes))
    level_names = [f"Level {i}" for i in labels]
    cm_raw, z = compute_confusion_matrix(y_true, y_pred, normalize, num_classes)

    if normalize:
        z_text = [[f"{v:.2f}" for v in row] for row in z]
        cbar_title = "Proportion / 비율"
        vmin, vmax = 0.0, 1.0
    else:
        z_text = [[str(int(v)) for v in row] for row in z]
        cbar_title = "Count / 개수"
        vmin, vmax = None, None

    # y축 반전: Level 0 을 상단에 배치 / Reverse y-axis: Level 0 at top
    z_flip = z[::-1]
    z_text_flip = z_text[::-1]
    y_labels = level_names[::-1]

    fig = go.Figure(
        go.Heatmap(
            z=z_flip,
            x=level_names,
            y=y_labels,
            text=z_text_flip,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title=cbar_title),
            hovertemplate=(
                "Predicted / 예측: %{x}<br>"
                "True / 실제: %{y}<br>"
                "Value: %{text}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Predicted Level / 예측 레벨",
        yaxis_title="True Level / 실제 레벨",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        template=PLOTLY_TEMPLATE,
        width=600,
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path, include_plotlyjs="cdn")
        print(f"Saved / 저장: {output_path}")

    return fig


def plot_all_channels(
    results: dict,
    output_dir: Path,
    channels: list = None,
    normalize: bool = True,
    open_browser: bool = False,
    num_classes: int = NUM_LEVELS,
) -> None:
    """
    색상별 + 전체 혼동 행렬 HTML 을 일괄 생성한다.
    Generates confusion matrix HTMLs for each channel and overall.

    Args:
        results      : {'Y': {'y_true': ..., 'y_pred': ...}, ...}
        output_dir   : HTML 저장 디렉토리 / Output directory for HTMLs
        channels     : 처리할 채널 / Channels to process (default: available in results)
        normalize    : 행 정규화 여부 / Row normalization
        open_browser : 생성 후 브라우저로 자동 열기 / Auto-open in browser after saving
        num_classes  : 클래스 수 / Number of classes
    """
    import webbrowser

    if channels is None:
        channels = [c for c in ["Y", "M", "C", "K"] if c in results]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from evaluation.metrics import compute_metrics

    all_true = np.concatenate([results[c]["y_true"] for c in channels])
    all_pred = np.concatenate([results[c]["y_pred"] for c in channels])
    overall_acc = compute_metrics(all_true, all_pred, num_classes)["accuracy"]

    for ch in channels:
        yt = results[ch]["y_true"]
        yp = results[ch]["y_pred"]
        acc = compute_metrics(yt, yp, num_classes)["accuracy"]

        path = str(output_dir / f"cm_{ch}.html")
        plot_confusion_matrix(
            yt,
            yp,
            title=f"[{ch}] Confusion Matrix  Acc={acc:.4f}",
            normalize=normalize,
            output_path=path,
            num_classes=num_classes,
        )
        if open_browser:
            webbrowser.open(Path(path).resolve().as_uri())

    overall_path = str(output_dir / "cm_overall.html")
    plot_confusion_matrix(
        all_true,
        all_pred,
        title=f"[Overall] Confusion Matrix  Acc={overall_acc:.4f}",
        normalize=normalize,
        output_path=overall_path,
        num_classes=num_classes,
    )
    if open_browser:
        webbrowser.open(Path(overall_path).resolve().as_uri())
