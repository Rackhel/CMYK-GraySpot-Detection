"""
evaluation/confusion.py
=======================
Grayspot Detection Pipeline — Confusion Matrix Module
Grayspot 탐지 파이프라인 — 혼동 행렬 모듈

Generates 6×6 interactive Plotly confusion matrices and MAE heatmaps.
6×6 인터랙티브 Plotly 혼동 행렬 및 MAE 히트맵을 생성합니다.

Source notebook : 04_evaluation.ipynb (Cell 7, Cell 10)
PRD reference   : Section 5.6.2, Section 8.2.3 (HTML Report)
Execution plan  : Stage 2 (W5~W6), Role R3

Design policy / 설계 방침:
  - fig.show() is NEVER called here. All figures are returned as go.Figure
    and written to HTML files so they open reliably on both macOS VS Code
    Jupyter and Windows.
    여기서 fig.show()는 절대 호출하지 않습니다. 모든 차트는 go.Figure로 반환하고
    HTML 파일로 저장하여 macOS VS Code Jupyter와 Windows 모두에서 안정적으로 열립니다.

Python 3.11.5 | macOS (MPS) & Windows (CUDA/CPU) compatible
"""

# ── Standard library / 표준 라이브러리 ────────────────────────────────────
from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Optional

# ── Third-party / 서드파티 ────────────────────────────────────────────────
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# ── Internal / 내부 ───────────────────────────────────────────────────────
from .metrics import NUM_LEVELS, CHANNELS, DEFAULT_TARGET_MAE


# ─────────────────────────────────────────────────────────────────────────────
# 0. Visualization style constants
#    시각화 스타일 상수
# ─────────────────────────────────────────────────────────────────────────────

# Dark-theme colors shared across all charts
# 모든 차트에서 공유하는 다크 테마 색상
PLOTLY_TEMPLATE: str = "plotly_dark"
FONT_FAMILY: str = "Segoe UI"
FONT_SIZE: int = 12

# Per-channel display colors (same as 04_evaluation.ipynb Cell 1)
# 채널별 표시 색상 (04_evaluation.ipynb Cell 1과 동일)
CMYK_COLORS: dict[str, str] = {
    "Y": "#f5e642",
    "M": "#e91e8c",
    "C": "#00b4d8",
    "K": "#444444",
}

# Level 0~5 marker colors
# 레벨 0~5 마커 색상
LEVEL_COLORS: list[str] = [
    "#2ecc71",
    "#f1c40f",
    "#e67e22",
    "#e74c3c",
    "#9b59b6",
    "#1a1a2e",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Utility helper
#    유틸리티 헬퍼
# ─────────────────────────────────────────────────────────────────────────────


def _open_in_browser(path: str | Path) -> None:
    """
    Open a saved HTML file in the system default browser.
    저장된 HTML 파일을 시스템 기본 브라우저로 엽니다.

    Uses Path.as_uri() (file:// prefix) for reliable cross-platform behavior.
    macOS와 Windows 모두에서 안정적인 동작을 위해 Path.as_uri()를 사용합니다.

    Identical implementation to 04_evaluation.ipynb Cell 0-B open_in_browser().
    04_evaluation.ipynb Cell 0-B의 open_in_browser()와 동일한 구현입니다.

    Args:
        path : Path to the HTML file to open / 열 HTML 파일 경로
    """
    uri = Path(path).resolve().as_uri()  # Absolute path → file:// URI / 절대경로 → file:// URI
    webbrowser.open(uri)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion Matrix
#    혼동 행렬
# ─────────────────────────────────────────────────────────────────────────────


def build_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    num_classes: int = NUM_LEVELS,
    normalize: bool = True,
) -> go.Figure:
    """
    Build an interactive Plotly 6×6 confusion matrix heatmap.
    인터랙티브 Plotly 6×6 혼동 행렬 히트맵을 생성합니다.

    Mirrors the plot_confusion_matrix() function from 04_evaluation.ipynb Cell 7.
    04_evaluation.ipynb Cell 7의 plot_confusion_matrix() 함수를 반영합니다.

    Design notes / 설계 참고:
      - Y-axis is reversed so Level 0 appears at top (standard convention).
        Level 0이 상단에 표시되도록 Y축을 반전합니다 (표준 관례).
      - fig.show() is NOT called. Caller decides how to display the figure.
        fig.show()는 호출하지 않습니다. 표시 방법은 호출자가 결정합니다.

    Args:
        y_true      : Ground-truth labels, shape (N,) / 정답 라벨
        y_pred      : Predicted labels, shape (N,) / 예측 라벨
        title       : Figure title string / 차트 제목 문자열
        num_classes : Number of severity levels / 심각도 레벨 수
        normalize   : If True, row-normalize the matrix (proportion)
                      True이면 행 정규화 적용 (비율)

    Returns:
        Plotly Figure object (NOT displayed) / Plotly Figure 객체 (표시하지 않음)
    """
    labels = list(range(num_classes))
    level_names = [f"Level {i}" for i in labels]

    # Build confusion matrix via sklearn
    # sklearn으로 혼동 행렬 구성
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        # Row-wise normalization: proportion of true label → predicted label
        # 행 단위 정규화: 실제 라벨 → 예측 라벨 비율
        row_sums = cm.sum(axis=1, keepdims=True)
        z = np.where(row_sums > 0, cm / row_sums, 0.0)
        z_text = [[f"{v:.2f}" for v in row] for row in z]
        cbar_title = "Proportion"
        vmin, vmax = 0.0, 1.0
    else:
        z = cm.astype(float)
        z_text = [[str(int(v)) for v in row] for row in cm]
        cbar_title = "Count"
        vmin, vmax = None, None

    # Reverse Y-axis so Level 0 is at top
    # Level 0이 상단에 오도록 Y축 반전
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
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Predicted Level / 예측 레벨",
        yaxis_title="True Level / 실제 레벨",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        template=PLOTLY_TEMPLATE,
        width=620,
        height=530,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    channel: str,
    accuracy: float,
    output_dir: str | Path,
    num_classes: int = NUM_LEVELS,
    normalize: bool = True,
    open_browser: bool = False,
    logger = None,
) -> Path:
    """
    Build a confusion matrix figure and save it as an HTML file.
    혼동 행렬 차트를 생성하고 HTML 파일로 저장합니다.

    Mirrors the per-channel loop in 04_evaluation.ipynb Cell 7.
    04_evaluation.ipynb Cell 7의 채널별 루프를 반영합니다.

    Args:
        y_true       : Ground-truth labels / 정답 라벨
        y_pred       : Predicted labels / 예측 라벨
        channel      : Channel name used in filename and title / 파일명 및 제목에 사용할 채널명
        accuracy     : Pre-computed accuracy for display in title / 제목 표시용 정확도
        output_dir   : Directory to write the HTML file / HTML 파일을 저장할 디렉토리
        num_classes  : Number of severity levels / 심각도 레벨 수
        normalize    : Row-normalize the matrix / 행 정규화 적용
        open_browser : If True, open the HTML in the system browser after saving
                       True이면 저장 후 시스템 브라우저에서 HTML 열기

    Returns:
        Resolved Path of the saved HTML file / 저장된 HTML 파일의 절대 경로
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_label = "Row-Normalized / 행 정규화" if normalize else "Count"
    title = (
        f"[{channel}] Confusion Matrix — Acc={accuracy:.4f} "
        f"({norm_label})"
    )

    fig = build_confusion_matrix_figure(
        y_true=y_true,
        y_pred=y_pred,
        title=title,
        num_classes=num_classes,
        normalize=normalize,
    )

    html_path = output_dir / f"cm_{channel}.html"
    # include_plotlyjs='cdn' keeps file size small
    # include_plotlyjs='cdn' 파일 크기를 줄임
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    log_func = logger.info if logger else print
    log_func(f"[저장 / Saved] {html_path}")

    if open_browser:
        _open_in_browser(html_path)

    return html_path.resolve()


def save_all_confusion_matrices(
    results: dict[str, dict],
    metrics: dict[str, dict],
    output_dir: str | Path,
    channels: list[str] = CHANNELS,
    num_classes: int = NUM_LEVELS,
    normalize: bool = True,
    open_browser: bool = False,
    logger = None,
) -> dict[str, Path]:
    """
    Generate and save confusion matrices for all channels plus overall.
    모든 채널 및 전체 데이터에 대한 혼동 행렬을 생성하고 저장합니다.

    Replicates the loop `for ch in CHANNELS + ['overall']` in Cell 7.
    Cell 7의 `for ch in CHANNELS + ['overall']` 루프를 재현합니다.

    Args:
        results      : Per-channel inference results dict:
                         {channel: {y_true, y_pred, confidences, filenames}}
                       채널별 추론 결과 딕셔너리
        metrics      : Pre-computed metrics dict (from evaluator)
                       사전 계산된 지표 딕셔너리 (evaluator에서 전달)
        output_dir   : Output directory / 출력 디렉토리
        channels     : CMYK channels to process / 처리할 CMYK 채널 목록
        num_classes  : Number of severity levels / 심각도 레벨 수
        normalize    : Row-normalize matrices / 행 정규화 적용
        open_browser : Open each HTML in browser after saving
                       저장 후 각 HTML을 브라우저에서 열기

    Returns:
        Dict mapping channel → resolved Path / 채널 → 절대 경로 매핑 딕셔너리
    """
    import numpy as np  # import here to avoid circular issues / 순환 임포트 방지

    saved: dict[str, Path] = {}

    # Build combined arrays for 'overall' / 'overall'을 위한 통합 배열 생성
    all_true = np.concatenate([results[c]["y_true"] for c in channels])
    all_pred = np.concatenate([results[c]["y_pred"] for c in channels])

    for ch in channels + ["overall"]:
        if ch == "overall":
            yt, yp = all_true, all_pred
        else:
            yt = results[ch]["y_true"]
            yp = results[ch]["y_pred"]

        # metrics dict may store ChannelMetrics or plain dict
        # metrics 딕셔너리는 ChannelMetrics 또는 일반 딕셔너리를 저장할 수 있음
        m = metrics.get(ch, {})
        acc = m.accuracy if hasattr(m, "accuracy") else m.get("accuracy", 0.0)

        saved[ch] = save_confusion_matrix(
            y_true=yt,
            y_pred=yp,
            channel=ch,
            accuracy=acc,
            output_dir=output_dir,
            num_classes=num_classes,
            normalize=normalize,
            open_browser=open_browser,
            logger=logger,
        )

    log_func = logger.info if logger else print
    log_func("\n✅  모든 혼동 행렬 저장 완료 / All confusion matrices saved")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAE Heatmap (Color × True Level)
#    MAE 히트맵 (색상 × 실제 레벨)
# ─────────────────────────────────────────────────────────────────────────────


def build_mae_heatmap_figure(
    results: dict[str, dict],
    channels: list[str] = CHANNELS,
    num_classes: int = NUM_LEVELS,
    target_mae: float = DEFAULT_TARGET_MAE,
) -> go.Figure:
    """
    Build a Plotly heatmap showing MAE per (color × true level).
    (색상 × 실제 레벨) 조합별 MAE Plotly 히트맵을 생성합니다.

    Mirrors plot_mae_heatmap() from 04_evaluation.ipynb Cell 10.
    04_evaluation.ipynb Cell 10의 plot_mae_heatmap()을 반영합니다.

    Args:
        results     : Per-channel inference results / 채널별 추론 결과
        channels    : CMYK channels / CMYK 채널 목록
        num_classes : Number of severity levels / 심각도 레벨 수
        target_mae  : MAE target shown in title / 제목에 표시할 MAE 목표

    Returns:
        Plotly Figure (NOT displayed) / Plotly Figure (표시하지 않음)
    """
    level_names = [f"Level {i}" for i in range(num_classes)]
    mae_matrix = np.full((len(channels), num_classes), np.nan)
    count_matrix = np.zeros((len(channels), num_classes), dtype=int)

    for ci, color in enumerate(channels):
        yt = results[color]["y_true"]
        yp = results[color]["y_pred"]
        for lv in range(num_classes):
            mask = yt == lv
            if mask.sum() > 0:
                mae_matrix[ci, lv] = float(
                    np.mean(np.abs(yt[mask].astype(float) - yp[mask].astype(float)))
                )
                count_matrix[ci, lv] = int(mask.sum())

    # Annotation: "mae_value\n(n=count)" or "N/A"
    # 주석: "mae_값\n(n=개수)" 또는 "N/A"
    annot_text = [
        [
            f"{mae_matrix[r, c]:.2f}<br>(n={count_matrix[r, c]})"
            if not np.isnan(mae_matrix[r, c])
            else "N/A"
            for c in range(num_classes)
        ]
        for r in range(len(channels))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=mae_matrix,
            x=level_names,
            y=channels,
            text=annot_text,
            texttemplate="%{text}",
            colorscale="YlOrRd",
            zmin=0,
            zmax=2.0,
            colorbar=dict(title="MAE"),
            hovertemplate=(
                "Color: %{y}<br>Level: %{x}<br>MAE: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"MAE per (Color × True Level) "
                f"— Target / 목표 ≤ {target_mae}"
            ),
            font=dict(size=16),
        ),
        xaxis=dict(title="True Level / 실제 레벨"),
        yaxis=dict(title="Color Channel / 색상 채널"),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        height=360,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def save_mae_heatmap(
    results: dict[str, dict],
    output_dir: str | Path,
    channels: list[str] = CHANNELS,
    num_classes: int = NUM_LEVELS,
    target_mae: float = DEFAULT_TARGET_MAE,
    open_browser: bool = False,
    logger = None,
) -> Path:
    """
    Build and save the MAE heatmap to HTML.
    MAE 히트맵을 생성하고 HTML로 저장합니다.

    Args:
        results      : Per-channel inference results / 채널별 추론 결과
        output_dir   : Output directory / 출력 디렉토리
        channels     : CMYK channels / CMYK 채널 목록
        num_classes  : Number of severity levels / 심각도 레벨 수
        target_mae   : MAE target / MAE 목표
        open_browser : Open HTML in system browser after saving
                       저장 후 시스템 브라우저에서 HTML 열기

    Returns:
        Resolved Path of saved HTML / 저장된 HTML의 절대 경로
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = build_mae_heatmap_figure(
        results=results,
        channels=channels,
        num_classes=num_classes,
        target_mae=target_mae,
    )

    html_path = output_dir / "mae_heatmap.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    log_func = logger.info if logger else print
    log_func(f"[저장 / Saved] {html_path}")

    if open_browser:
        _open_in_browser(html_path)

    return html_path.resolve()
