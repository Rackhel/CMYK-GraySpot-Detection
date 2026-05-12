"""
reporting/html_report.py
========================
Grayspot Detection Pipeline — Baseline HTML Report Generator
Grayspot 탐지 파이프라인 — Baseline HTML 리포트 생성기

Generates outputs/reports/baseline.html from an EvaluationSummary.
EvaluationSummary로부터 outputs/reports/baseline.html을 생성합니다.

Source notebook : 04_evaluation.ipynb (Cells 6~15)
PRD reference   : Section 8.2 (Reporting Module), Section 8.2.3 (HTML Report Layout)
Execution plan  : Stage 2 (W7~W8), Role R3

Design:
  - Pure HTML + CDN Plotly (no local assets, no Jinja2 dependency)
    순수 HTML + CDN Plotly (로컬 에셋 없음, Jinja2 의존성 없음)
  - Opens reliably on macOS (Safari/Chrome) and Windows (Edge/Chrome)
    macOS (Safari/Chrome) 및 Windows (Edge/Chrome) 에서 안정적으로 열림
  - Single self-contained .html file — copy-anywhere portable
    단일 독립 .html 파일 — 어디서나 복사 가능

Python 3.11.5 | macOS & Windows compatible
"""

# ── Standard library / 표준 라이브러리 ────────────────────────────────────
from __future__ import annotations

import json
# ── Internal / 내부 ───────────────────────────────────────────────────────
# Absolute import so this module works both standalone and as part of the package
# 패키지 일부 및 단독 실행 모두 동작하도록 절대 임포트 사용
#
# Folder layout / 폴더 구조:
#   CMYK_MAIN/
#     src/
#       reporting/html_report.py   ← this file / 이 파일
#       evaluation/                ← import target / 임포트 대상
#
# Path chain: html_report.py → reporting/ → src/
# 경로 체인: html_report.py → reporting/ → src/
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Third-party / 서드파티 ────────────────────────────────────────────────
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

_SRC_DIR = Path(__file__).parent.parent.resolve()  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Logging 설정 / Logger setup ─────────────────────────────────────────
try:
    from utils.logger import get_logger

    _logger = get_logger(__name__)
except ImportError:
    import logging

    _logger = logging.getLogger(__name__)

from evaluation.confusion import (CMYK_COLORS, FONT_FAMILY, FONT_SIZE,
                                  PLOTLY_TEMPLATE)
from evaluation.metrics import (CHANNELS, DEFAULT_TARGET_MAE,
                                DEFAULT_TARGET_OVERALL_ACC,
                                DEFAULT_TARGET_PER_CLASS_F1,
                                DEFAULT_TARGET_PER_COLOR_ACC, NUM_LEVELS,
                                ChannelMetrics, EvaluationSummary,
                                summary_to_dict)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Style constants — matches 04_evaluation.ipynb color palette
#    스타일 상수 — 04_evaluation.ipynb 색상 팔레트와 일치
# ─────────────────────────────────────────────────────────────────────────────

# CSS variables used in the report template
# 리포트 템플릿에서 사용하는 CSS 변수
_CSS = """
:root {
    --bg:        #0b1220;
    --fg:        #e6eef8;
    --card:      #111827;
    --border:    rgba(255,255,255,0.10);
    --h1:        #66d9ff;
    --h2:        #50e3c2;
    --h3:        #c792ea;
    --accent:    #ff7aa2;
    --pass:      #2ecc71;
    --fail:      #e74c3c;
    --warn:      #f39c12;
    --font-mono: 'Consolas', 'Menlo', monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
    background: var(--bg);
    color: var(--fg);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 14px;
    line-height: 1.7;
    padding: 0 0 4rem;
}

/* ── Header ── */
.report-header {
    background: linear-gradient(135deg, #0d1b2e 0%, #0f2a40 60%, #0b1220 100%);
    border-bottom: 2px solid rgba(102,217,255,0.20);
    padding: 2.5rem 3rem 2rem;
}
.report-header h1 {
    color: var(--h1);
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-bottom: 0.3rem;
}
.report-header .meta {
    color: rgba(230,238,248,0.55);
    font-size: 0.82rem;
    font-family: var(--font-mono);
    margin-top: 0.5rem;
}
.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-left: 0.5rem;
    vertical-align: middle;
}
.badge-stage  { background: rgba(80,227,194,0.15); color: #50e3c2; border: 1px solid rgba(80,227,194,0.3); }
.badge-r3     { background: rgba(199,146,234,0.15); color: #c792ea; border: 1px solid rgba(199,146,234,0.3); }

/* ── Navigation tabs ── */
.nav-tabs {
    display: flex;
    gap: 0;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 0 3rem;
    position: sticky;
    top: 0;
    z-index: 100;
}
.nav-tab {
    padding: 0.8rem 1.4rem;
    cursor: pointer;
    color: rgba(230,238,248,0.50);
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    border-bottom: 2px solid transparent;
    transition: color 0.2s, border-color 0.2s;
    user-select: none;
    text-transform: uppercase;
}
.nav-tab:hover { color: var(--fg); }
.nav-tab.active { color: var(--h1); border-bottom-color: var(--h1); }

/* ── Sections ── */
.section { display: none; padding: 2rem 3rem; }
.section.active { display: block; }

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
}
.card h2 {
    color: var(--h2);
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.card h3 {
    color: var(--h3);
    font-size: 0.9rem;
    font-weight: 600;
    margin: 0.8rem 0 0.5rem;
}

/* ── KPI grid ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.4rem;
}
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.pass::before { background: var(--pass); }
.kpi-card.fail::before { background: var(--fail); }
.kpi-card .kpi-label {
    font-size: 0.70rem;
    color: rgba(230,238,248,0.50);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.kpi-card .kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: var(--font-mono);
    line-height: 1.1;
}
.kpi-card.pass .kpi-value { color: var(--pass); }
.kpi-card.fail .kpi-value { color: var(--fail); }
.kpi-card .kpi-target {
    font-size: 0.68rem;
    color: rgba(230,238,248,0.40);
    margin-top: 0.3rem;
}
.kpi-card .kpi-flag {
    font-size: 1rem;
    margin-top: 0.2rem;
}

/* ── Tables ── */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    font-family: var(--font-mono);
}
.data-table th {
    background: rgba(255,255,255,0.05);
    color: rgba(230,238,248,0.65);
    font-weight: 600;
    padding: 0.5rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    letter-spacing: 0.04em;
    font-size: 0.70rem;
    text-transform: uppercase;
}
.data-table td {
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: var(--fg);
}
.data-table tr:last-child td { border-bottom: none; }
.data-table tr:hover td { background: rgba(255,255,255,0.025); }
.pass-text { color: var(--pass); }
.fail-text { color: var(--fail); }

/* ── Feedback decisions ── */
.decision-list { list-style: none; padding: 0; }
.decision-list li {
    padding: 0.5rem 0.8rem;
    border-left: 3px solid var(--fail);
    margin-bottom: 0.5rem;
    background: rgba(231,76,60,0.06);
    border-radius: 0 6px 6px 0;
    font-size: 0.83rem;
    font-family: var(--font-mono);
}
.decision-list li.phase0 { border-left-color: #66d9ff; background: rgba(102,217,255,0.06); }
.decision-list li.phase1 { border-left-color: #ffb347; background: rgba(255,179,71,0.06); }

.terminate-box {
    background: rgba(46,204,113,0.08);
    border: 1px solid rgba(46,204,113,0.30);
    border-radius: 8px;
    padding: 1rem 1.4rem;
    color: var(--pass);
    font-size: 0.9rem;
}
.no-critical-box {
    background: rgba(80,227,194,0.07);
    border: 1px solid rgba(80,227,194,0.25);
    border-radius: 8px;
    padding: 1rem 1.4rem;
    color: #50e3c2;
    font-size: 0.9rem;
}

/* ── Plotly containers ── */
.plotly-wrap {
    width: 100%;
    border-radius: 8px;
    overflow: hidden;
    margin: 0.6rem 0;
}

/* ── Footer ── */
.report-footer {
    text-align: center;
    padding: 2rem;
    color: rgba(230,238,248,0.25);
    font-size: 0.75rem;
    font-family: var(--font-mono);
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
"""

# Tab navigation labels / 탭 네비게이션 레이블
_TABS = [
    ("summary", "① Summary"),
    ("perclass", "② Per-Class"),
    ("confusion", "③ Confusion"),
    ("mae", "④ MAE"),
    ("confidence", "⑤ Confidence"),
    ("feedback", "⑥ Feedback"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Plotly figure builders — inline JSON for embedding
#    Plotly 차트 빌더 — 임베딩용 인라인 JSON
# ─────────────────────────────────────────────────────────────────────────────


def _fig_to_json(fig: go.Figure) -> str:
    """
    Serialize a Plotly figure to a compact JSON string for inline embedding.
    Plotly Figure를 인라인 임베딩용 컴팩트 JSON 문자열로 직렬화합니다.

    Uses pio.to_json() which is cross-platform and does not call fig.show().
    fig.show()를 호출하지 않는 크로스 플랫폼 pio.to_json()을 사용합니다.
    """
    return pio.to_json(fig, validate=False)


def _build_dashboard_fig(
    overall: ChannelMetrics,
    by_channel: dict[str, ChannelMetrics],
    channels: list[str],
    targets: dict[str, float],
) -> go.Figure:
    """
    Gauge + Bar dashboard (mirrors Cell 8 of 04_evaluation.ipynb).
    Gauge + Bar 대시보드 (04_evaluation.ipynb Cell 8 반영).
    """
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "bar", "colspan": 3}, None, None],
        ],
        subplot_titles=[
            "Overall Accuracy / 전체 정확도",
            "Overall Macro F1",
            "Overall MAE (낮을수록 좋음 / Lower is better)",
            "Per-Color Accuracy / 색상별 정확도",
        ],
        vertical_spacing=0.18,
    )

    # Gauge: Accuracy / 정확도
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=round(overall.accuracy * 100, 2),
            number={"suffix": "%", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#50e3c2"},
                "threshold": {
                    "line": {"color": "#ff7aa2", "width": 3},
                    "value": targets["overall_accuracy"] * 100,
                },
                "bgcolor": "#0b1220",
            },
            title={"text": f"Target ≥ {targets['overall_accuracy']:.0%}"},
        ),
        row=1,
        col=1,
    )

    # Gauge: Macro F1
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=round(overall.macro_f1, 4),
            number={"font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#66d9ff"},
                "threshold": {
                    "line": {"color": "#ff7aa2", "width": 3},
                    "value": targets["per_class_f1"],
                },
                "bgcolor": "#0b1220",
            },
            title={"text": f"Target ≥ {targets['per_class_f1']:.2f}"},
        ),
        row=1,
        col=2,
    )

    # Gauge: MAE
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=round(overall.mae, 4),
            number={"font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 3]},
                "bar": {"color": "#c792ea"},
                "threshold": {
                    "line": {"color": "#ffb347", "width": 3},
                    "value": targets["mae"],
                },
                "bgcolor": "#0b1220",
            },
            title={"text": f"Target ≤ {targets['mae']:.2f}"},
        ),
        row=1,
        col=3,
    )

    # Bar: per-color accuracy / 색상별 정확도
    acc_vals = [by_channel[c].accuracy * 100 for c in channels]
    fig.add_trace(
        go.Bar(
            x=channels,
            y=acc_vals,
            marker_color=[CMYK_COLORS.get(c, "#aaa") for c in channels],
            text=[f"{v:.2f}%" for v in acc_vals],
            textposition="outside",
            name="Accuracy",
        ),
        row=2,
        col=1,
    )

    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(channels) - 0.5,
        y0=targets["per_color_accuracy"] * 100,
        y1=targets["per_color_accuracy"] * 100,
        line=dict(color="#ff7aa2", dash="dash"),
        xref="x",
        yref="y",
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=channels[-1],
        y=targets["per_color_accuracy"] * 100,
        text=f"Target {targets['per_color_accuracy']:.0%}",
        showarrow=False,
        yshift=12,
        row=2,
        col=1,
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        height=720,
        showlegend=False,
        margin=dict(l=40, r=40, t=70, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_per_class_fig(
    overall: ChannelMetrics,
    target_f1: float,
) -> go.Figure:
    """
    Per-class Precision / Recall / F1 bar chart (mirrors Cell 9).
    클래스별 Precision / Recall / F1 막대 차트 (Cell 9 반영).
    """
    pc = overall.per_class
    levels = [f"Level {d.level}" for d in pc]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="F1", x=levels, y=[d.f1 for d in pc], marker_color="#1976D2")
    )
    fig.add_trace(
        go.Bar(
            name="Precision",
            x=levels,
            y=[d.precision for d in pc],
            marker_color="#388E3C",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Recall", x=levels, y=[d.recall for d in pc], marker_color="#F57C00"
        )
    )
    fig.add_hline(
        y=target_f1,
        line_dash="dash",
        line_color="#ff7aa2",
        annotation_text=f"F1 Target ≥ {target_f1:.2f}",
    )
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.15], title="Score"),
        xaxis=dict(title="Level"),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        legend=dict(orientation="h", y=1.1),
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_confusion_fig(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    channel: str,
    num_classes: int = NUM_LEVELS,
) -> go.Figure:
    """
    6×6 row-normalized confusion matrix heatmap (mirrors Cell 7).
    6×6 행 정규화 혼동 행렬 히트맵 (Cell 7 반영).
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    labels = list(range(num_classes))
    level_names = [f"L{i}" for i in labels]
    cm = sk_cm(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    z = np.where(row_sums > 0, cm / row_sums, 0.0)
    z_text = [[f"{v:.2f}" for v in row] for row in z]

    # Flip Y-axis: Level 0 at top / Y축 반전: Level 0이 상단
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
            zmin=0,
            zmax=1,
            colorbar=dict(title="Proportion", thickness=14),
            hovertemplate="Pred: %{x}<br>True: %{y}<br>Prop: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"[{channel}] Confusion Matrix (Row-Normalized)", font=dict(size=13)
        ),
        xaxis_title="Predicted Level / 예측 레벨",
        yaxis_title="True Level / 실제 레벨",
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        width=520,
        height=460,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_mae_heatmap_fig(
    results: dict[str, dict],
    channels: list[str],
    num_classes: int = NUM_LEVELS,
    target_mae: float = DEFAULT_TARGET_MAE,
) -> go.Figure:
    """
    MAE heatmap per (color × true level) (mirrors Cell 10).
    (색상 × 실제 레벨)별 MAE 히트맵 (Cell 10 반영).
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

    annot = [
        [
            (
                f"{mae_matrix[r, c]:.2f}<br>(n={count_matrix[r, c]})"
                if not np.isnan(mae_matrix[r, c])
                else "N/A"
            )
            for c in range(num_classes)
        ]
        for r in range(len(channels))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=mae_matrix,
            x=level_names,
            y=channels,
            text=annot,
            texttemplate="%{text}",
            colorscale="YlOrRd",
            zmin=0,
            zmax=2.0,
            colorbar=dict(title="MAE", thickness=14),
            hovertemplate="Color: %{y}<br>Level: %{x}<br>MAE: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"MAE per (Color × True Level) — Target ≤ {target_mae}",
            font=dict(size=13),
        ),
        xaxis=dict(title="True Level / 실제 레벨"),
        yaxis=dict(title="Color / 색상"),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_confidence_fig(
    results: dict[str, dict],
    channels: list[str],
    conf_thresh_auto: float = 0.8,
    conf_thresh_warn: float = 0.5,
    conf_thresh_manual: float = 0.3,
) -> go.Figure:
    """
    Per-color confidence histogram correct vs wrong (mirrors Cell 12).
    색상별 신뢰도 히스토그램 정답 vs 오답 (Cell 12 반영).
    """
    from plotly.subplots import make_subplots as ms

    fig = ms(
        rows=2,
        cols=2,
        subplot_titles=[f"[{c}]" for c in channels],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )
    bins = dict(start=0, end=1, size=0.04)

    for i, color in enumerate(channels):
        r = i // 2 + 1
        c = i % 2 + 1
        yt = results[color]["y_true"]
        yp = results[color]["y_pred"]
        cf = results[color]["confidences"]

        # Correct / 정답
        fig.add_trace(
            go.Histogram(
                x=cf[yt == yp],
                xbins=bins,
                name="Correct / 정답",
                marker_color="#4fc3f7",
                opacity=0.70,
                showlegend=(i == 0),
                legendgroup="correct",
            ),
            row=r,
            col=c,
        )

        # Wrong / 오답
        fig.add_trace(
            go.Histogram(
                x=cf[yt != yp],
                xbins=bins,
                name="Wrong / 오답",
                marker_color="#ef5350",
                opacity=0.70,
                showlegend=(i == 0),
                legendgroup="wrong",
            ),
            row=r,
            col=c,
        )

        # Threshold lines / 임계값 수직선
        for thresh, lc in [
            (conf_thresh_auto, "green"),
            (conf_thresh_warn, "orange"),
            (conf_thresh_manual, "red"),
        ]:
            fig.add_vline(
                x=thresh, line_dash="dash", line_color=lc, line_width=1.5, row=r, col=c
            )

    fig.update_layout(
        barmode="overlay",
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        height=560,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. HTML section builders
#    HTML 섹션 빌더
# ─────────────────────────────────────────────────────────────────────────────


def _kpi_card(label: str, value: str, target_str: str, passed: bool) -> str:
    """
    Render a single KPI card HTML block.
    단일 KPI 카드 HTML 블록을 렌더링합니다.
    """
    cls = "pass" if passed else "fail"
    flag = "✅" if passed else "❌"
    return f"""
    <div class="kpi-card {cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-target">{target_str}</div>
        <div class="kpi-flag">{flag}</div>
    </div>"""


def _build_summary_section(
    summary: EvaluationSummary,
    channels: list[str],
    dashboard_json: str,
) -> str:
    """
    Build HTML for the Summary tab.
    Summary 탭의 HTML을 생성합니다.

    Contains: KPI cards + model info table + per-color accuracy table.
    포함 내용: KPI 카드 + 모델 정보 테이블 + 색상별 정확도 테이블.
    """
    targets = summary.targets
    overall = summary.overall
    meta = summary.meta

    # ── KPI cards / KPI 카드 ─────────────────────────────────────────────
    kpi_html = "".join(
        [
            _kpi_card(
                "Overall Accuracy",
                f"{overall.accuracy * 100:.2f}%",
                f"Target ≥ {targets['overall_accuracy']:.0%}",
                overall.acc_pass,
            ),
            _kpi_card(
                "Macro F1",
                f"{overall.macro_f1:.4f}",
                f"Target ≥ {targets['per_class_f1']:.2f}",
                overall.f1_pass,
            ),
            _kpi_card(
                "MAE",
                f"{overall.mae:.4f}",
                f"Target ≤ {targets['mae']:.2f}",
                overall.mae_pass,
            ),
            _kpi_card(
                "Total Samples",
                f"{overall.n_samples:,}",
                "Y + M + C + K",
                True,
            ),
        ]
    )

    # ── Model info table / 모델 정보 테이블 ──────────────────────────────
    checkpoint_str = meta.get("checkpoint") or "None (random weights / 랜덤 가중치)"
    info_rows = "".join(
        [
            f"<tr><td>Backbone</td><td>{meta.get('backbone', 'N/A')}</td></tr>",
            f"<tr><td>Checkpoint</td><td>{checkpoint_str}</td></tr>",
            f"<tr><td>Total Samples</td><td>{overall.n_samples:,}</td></tr>",
            f"<tr><td>Stage</td><td>S2 · W7~W8 Baseline</td></tr>",
            f"<tr><td>Role</td><td>R3 (Evaluation &amp; Reporting)</td></tr>",
        ]
    )

    # ── Per-color accuracy table / 색상별 정확도 테이블 ──────────────────
    color_rows = ""
    for ch in channels:
        cm = summary.by_channel.get(ch)
        if cm is None:
            continue
        acc_cls = "pass-text" if cm.acc_pass else "fail-text"
        f1_cls = "pass-text" if cm.f1_pass else "fail-text"
        mae_cls = "pass-text" if cm.mae_pass else "fail-text"
        color_rows += f"""
        <tr>
            <td>{ch}</td>
            <td class="{acc_cls}">{cm.accuracy * 100:.2f}%</td>
            <td class="{f1_cls}">{cm.macro_f1:.4f}</td>
            <td class="{mae_cls}">{cm.mae:.4f}</td>
            <td>{cm.n_samples:,}</td>
        </tr>"""

    return f"""
    <div class="kpi-grid">{kpi_html}</div>

    <div class="card">
        <h2>📈 Dashboard — Accuracy / F1 / MAE</h2>
        <div class="plotly-wrap" id="dash-plot"></div>
    </div>

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.4rem;">
        <div class="card">
            <h2>🧠 Model Information / 모델 정보</h2>
            <table class="data-table">
                <tbody>{info_rows}</tbody>
            </table>
        </div>
        <div class="card">
            <h2>🎨 Per-Color Accuracy / 색상별 정확도</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Channel</th>
                        <th>Accuracy</th>
                        <th>Macro F1</th>
                        <th>MAE</th>
                        <th>Samples</th>
                    </tr>
                </thead>
                <tbody>{color_rows}</tbody>
            </table>
        </div>
    </div>

    <script>
        Plotly.newPlot('dash-plot', {dashboard_json});
    </script>"""


def _build_perclass_section(
    summary: EvaluationSummary,
    perclass_json: str,
) -> str:
    """
    Build HTML for the Per-Class tab.
    Per-Class 탭의 HTML을 생성합니다.
    """
    targets = summary.targets
    pc_rows = ""
    for pc in summary.overall.per_class:
        f1_cls = "pass-text" if pc.f1 >= targets["per_class_f1"] else "fail-text"
        pc_rows += f"""
        <tr>
            <td>Level {pc.level}</td>
            <td>{pc.precision:.4f}</td>
            <td>{pc.recall:.4f}</td>
            <td class="{f1_cls}">{pc.f1:.4f}</td>
            <td>{pc.support:,}</td>
            <td>{'✅' if pc.f1 >= targets['per_class_f1'] else '❌'}</td>
        </tr>"""

    return f"""
    <div class="card">
        <h2>📊 Per-Class Precision / Recall / F1 — Overall / 전체</h2>
        <div class="plotly-wrap" id="pc-plot"></div>
    </div>

    <div class="card">
        <h2>📋 Per-Class Table / 클래스별 테이블</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Level</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Support</th>
                    <th>Pass (F1 ≥ {targets['per_class_f1']:.2f})</th>
                </tr>
            </thead>
            <tbody>{pc_rows}</tbody>
        </table>
    </div>

    <script>
        Plotly.newPlot('pc-plot', {perclass_json});
    </script>"""


def _build_confusion_section(
    cm_jsons: dict[str, str],
    channels: list[str],
) -> str:
    """
    Build HTML for the Confusion Matrix tab (grid layout).
    Confusion Matrix 탭의 HTML을 생성합니다 (그리드 레이아웃).
    """
    # 2-column grid for per-channel, full-width for overall
    # 채널별은 2열 그리드, overall은 전체 너비
    grid_items = ""
    for ch in channels:
        grid_items += f"""
        <div class="card">
            <h2>🔲 [{ch}] Confusion Matrix</h2>
            <div class="plotly-wrap" id="cm-{ch}"></div>
            <script>Plotly.newPlot('cm-{ch}', {cm_jsons[ch]});</script>
        </div>"""

    overall_block = f"""
    <div class="card">
        <h2>🔲 [Overall] Confusion Matrix — All Channels / 전체</h2>
        <div class="plotly-wrap" id="cm-overall"></div>
        <script>Plotly.newPlot('cm-overall', {cm_jsons['overall']});</script>
    </div>"""

    return f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.4rem;">
        {grid_items}
    </div>
    {overall_block}"""


def _build_mae_section(mae_json: str) -> str:
    """
    Build HTML for the MAE Heatmap tab.
    MAE 히트맵 탭의 HTML을 생성합니다.
    """
    return f"""
    <div class="card">
        <h2>🌡️ MAE Heatmap — (Color × True Level) / (색상 × 실제 레벨)</h2>
        <div class="plotly-wrap" id="mae-plot"></div>
    </div>
    <script>Plotly.newPlot('mae-plot', {mae_json});</script>"""


def _build_confidence_section(conf_json: str) -> str:
    """
    Build HTML for the Confidence Distribution tab.
    신뢰도 분포 탭의 HTML을 생성합니다.
    """
    return f"""
    <div class="card">
        <h2>🔵 Confidence Distribution / 신뢰도 분포 — Correct vs Wrong</h2>
        <p style="color:rgba(230,238,248,0.45);font-size:0.78rem;margin-bottom:0.8rem;">
            Vertical lines: 🟢 Auto ≥ 0.8 &nbsp;|&nbsp; 🟠 Warn ≥ 0.5 &nbsp;|&nbsp; 🔴 Manual ≥ 0.3
            &nbsp; (PRD §14.2)
        </p>
        <div class="plotly-wrap" id="conf-plot"></div>
    </div>
    <script>Plotly.newPlot('conf-plot', {conf_json});</script>"""


def _build_feedback_section(
    summary: EvaluationSummary,
    channels: list[str],
) -> str:
    """
    Build HTML for the Phase 3 Feedback Decision tab.
    Phase 3 피드백 판단 탭의 HTML을 생성합니다.
    """
    from evaluation.metrics import determine_swing_feedback

    decision = determine_swing_feedback(summary, channels=channels)
    targets = summary.targets
    overall = summary.overall

    # Render status block / 상태 블록 렌더링
    if decision["terminate"]:
        status_html = f"""
        <div class="terminate-box">
            🎉 모든 목표 달성 → Swing 종료 / All targets met — TERMINATE Swing<br>
            <span style="opacity:0.7;font-size:0.82rem;">
                Accuracy {overall.accuracy:.4f} ≥ {targets['overall_accuracy']} &nbsp;|&nbsp;
                Macro F1 {overall.macro_f1:.4f} ≥ {targets['per_class_f1']} &nbsp;|&nbsp;
                MAE {overall.mae:.4f} ≤ {targets['mae']}
            </span>
        </div>"""
    elif not decision["decisions"]:
        status_html = """
        <div class="no-critical-box">
            ✅ 심각한 실패 없음 / No critical failures — continue training or proceed to next cycle
        </div>"""
    else:
        items = ""
        for d in decision["decisions"]:
            # Classify decision type for color coding
            # 색상 코딩을 위한 결정 유형 분류
            cls = "phase0" if "Phase 0" in d else "phase1"
            items += f"<li class='{cls}'>⮕ {d}</li>"
        status_html = f"""
        <div class="card" style="margin:0;">
            <h2>⚠️ Action Required / 조치 필요</h2>
            <ul class="decision-list">{items}</ul>
        </div>"""

    # Metrics summary table / 지표 요약 테이블
    def _row(label, val, target, passed):
        cls = "pass-text" if passed else "fail-text"
        return f"<tr><td>{label}</td><td class='{cls}'>{val}</td><td>{target}</td><td>{'✅' if passed else '❌'}</td></tr>"

    table_rows = "".join(
        [
            _row(
                "Overall Accuracy",
                f"{overall.accuracy:.4f}",
                f"≥ {targets['overall_accuracy']}",
                overall.acc_pass,
            ),
            _row(
                "Overall Macro F1",
                f"{overall.macro_f1:.4f}",
                f"≥ {targets['per_class_f1']}",
                overall.f1_pass,
            ),
            _row(
                "Overall MAE",
                f"{overall.mae:.4f}",
                f"≤ {targets['mae']}",
                overall.mae_pass,
            ),
        ]
    )

    return f"""
    <div class="card">
        <h2>🔄 Phase 3 Feedback Decision / Phase 3 피드백 복귀 판단 (PRD §3.3.2)</h2>
        {status_html}
    </div>

    <div class="card">
        <h2>📋 Metrics vs Targets / 지표 vs 목표</h2>
        <table class="data-table">
            <thead>
                <tr><th>Metric</th><th>Value</th><th>Target</th><th>Pass</th></tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main HTML assembler
#    메인 HTML 조립기
# ─────────────────────────────────────────────────────────────────────────────


def generate_baseline_report(
    summary: EvaluationSummary,
    results: dict[str, dict],
    output_path: str | Path = "outputs/reports/baseline.html",
    channels: list[str] = CHANNELS,
    open_browser: bool = False,
    logger=None,
) -> Path:
    """
    Generate outputs/reports/baseline.html from an EvaluationSummary.
    EvaluationSummary로부터 outputs/reports/baseline.html을 생성합니다.

    The report is a single self-contained HTML file with embedded Plotly CDN.
    리포트는 Plotly CDN이 내장된 단일 독립 HTML 파일입니다.

    macOS (Safari/Chrome) 및 Windows (Edge/Chrome) 모두에서 안정적으로 열립니다.

    Args:
        summary      : EvaluationSummary produced by GrayspotEvaluator.run()
                       GrayspotEvaluator.run()이 생성한 EvaluationSummary
        results      : Per-channel inference results dict
                       채널별 추론 결과 딕셔너리
        output_path  : Destination HTML path (default: outputs/reports/baseline.html)
                       저장 경로 (기본값: outputs/reports/baseline.html)
        channels     : CMYK channels to include / 포함할 CMYK 채널
        open_browser : If True, open the HTML in the system default browser
                       True이면 시스템 기본 브라우저에서 HTML 열기
        logger       : Optional logger instance for logging / 로깅용 선택사항 로거

    Returns:
        Resolved absolute Path of the generated HTML file
        생성된 HTML 파일의 절대 경로
    """
    # Use provided logger or fallback to module logger
    # 제공된 로거 사용 또는 모듈 로거로 폴백
    log = logger or _logger

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    targets = summary.targets
    overall = summary.overall
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.info(f"[Report] Starting HTML report generation...")
    log.debug(f"  Output path: {output_path}")
    log.debug(f"  Channels: {channels}")
    log.debug(
        f"  Overall metrics - Accuracy: {overall.accuracy:.4f}, F1: {overall.macro_f1:.4f}, MAE: {overall.mae:.4f}"
    )

    # ── Build all Plotly figures → JSON strings ───────────────────────────
    # 모든 Plotly 차트 → JSON 문자열 생성
    log.debug("[Report] Generating Plotly figures...")

    dashboard_json = _fig_to_json(
        _build_dashboard_fig(overall, summary.by_channel, channels, targets)
    )
    log.debug("  ✓ Dashboard figure generated")

    perclass_json = _fig_to_json(_build_per_class_fig(overall, targets["per_class_f1"]))
    log.debug("  ✓ Per-class figure generated")

    conf_json = _fig_to_json(_build_confidence_fig(results, channels))
    log.debug("  ✓ Confidence figure generated")

    mae_json = _fig_to_json(
        _build_mae_heatmap_fig(results, channels, target_mae=targets["mae"])
    )
    log.debug("  ✓ MAE heatmap generated")

    # Combined arrays for overall confusion matrix
    # 전체 혼동 행렬용 통합 배열
    log.debug("[Report] Generating confusion matrices...")
    all_true = np.concatenate([results[c]["y_true"] for c in channels])
    all_pred = np.concatenate([results[c]["y_pred"] for c in channels])

    cm_jsons: dict[str, str] = {}
    for ch in channels:
        cm_jsons[ch] = _fig_to_json(
            _build_confusion_fig(results[ch]["y_true"], results[ch]["y_pred"], ch)
        )
        log.debug(f"  ✓ Confusion matrix [{ch}] generated")

    cm_jsons["overall"] = _fig_to_json(
        _build_confusion_fig(all_true, all_pred, "Overall")
    )
    log.debug("  ✓ Overall confusion matrix generated")

    # ── Build each tab's HTML content ─────────────────────────────────────
    # 각 탭의 HTML 콘텐츠 생성
    log.debug("[Report] Building HTML sections...")
    sections = {
        "summary": _build_summary_section(summary, channels, dashboard_json),
        "perclass": _build_perclass_section(summary, perclass_json),
        "confusion": _build_confusion_section(cm_jsons, channels),
        "mae": _build_mae_section(mae_json),
        "confidence": _build_confidence_section(conf_json),
        "feedback": _build_feedback_section(summary, channels),
    }
    log.debug("  ✓ All sections built")

    # ── Render tab navigation HTML ────────────────────────────────────────
    # 탭 네비게이션 HTML 렌더링
    tabs_html = "".join(
        f'<div class="nav-tab{" active" if i == 0 else ""}" '
        f"onclick=\"switchTab('{tid}')\">{label}</div>"
        for i, (tid, label) in enumerate(_TABS)
    )

    # ── Render section HTML ────────────────────────────────────────────────
    # 섹션 HTML 렌더링
    sections_html = "".join(
        f'<section id="sec-{tid}" class="section{" active" if i == 0 else ""}">'
        f"{sections[tid]}</section>"
        for i, (tid, _) in enumerate(_TABS)
    )

    # ── Pass/Fail badge for header ─────────────────────────────────────────
    # 헤더용 합격/불합격 배지
    all_pass = overall.acc_pass and overall.f1_pass and overall.mae_pass
    status_badge = (
        '<span style="color:#2ecc71;font-size:0.9rem;">✅ Targets Met</span>'
        if all_pass
        else '<span style="color:#e74c3c;font-size:0.9rem;">❌ Below Target</span>'
    )

    log.debug(f"  Status badge: {'✅ Targets Met' if all_pass else '❌ Below Target'}")

    # ── Assemble final HTML ────────────────────────────────────────────────
    # 최종 HTML 조립
    log.debug("[Report] Assembling final HTML...")
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grayspot Baseline Report</title>
    <!-- Plotly loaded from CDN — no local assets required / CDN에서 로드 — 로컬 에셋 불필요 -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>{_CSS}</style>
</head>
<body>

<!-- ── Header / 헤더 ─────────────────────────────────────────────────── -->
<header class="report-header">
    <h1>
        Grayspot Baseline Evaluation Report
        <span class="badge badge-stage">Stage 2 · W7~W8</span>
        <span class="badge badge-r3">R3</span>
    </h1>
    <div style="margin-top:0.6rem;">{status_badge}</div>
    <div class="meta">
        Generated / 생성 : {now_str} &nbsp;|&nbsp;
        Backbone : {summary.meta.get('backbone', 'N/A')} &nbsp;|&nbsp;
        Samples  : {overall.n_samples:,} &nbsp;|&nbsp;
        PRD §1.4 targets · §3.3.2 feedback · §8.2 reporting
    </div>
</header>

<!-- ── Navigation / 탭 네비게이션 ───────────────────────────────────── -->
<nav class="nav-tabs">{tabs_html}</nav>

<!-- ── Content sections / 콘텐츠 섹션 ──────────────────────────────── -->
<main>{sections_html}</main>

<!-- ── Footer / 푸터 ─────────────────────────────────────────────────── -->
<footer class="report-footer">
    Grayspot Detection Pipeline &nbsp;·&nbsp; Stage 2 Baseline Report &nbsp;·&nbsp;
    Python 3.11.5 &nbsp;·&nbsp; {now_str}
</footer>

<!-- ── Tab switching logic / 탭 전환 로직 ──────────────────────────── -->
<script>
    /**
     * Switch active tab and section.
     * 활성 탭과 섹션을 전환합니다.
     * @param {{string}} tabId - Section identifier / 섹션 식별자
     */
    function switchTab(tabId) {{
        // Deactivate all tabs and sections / 모든 탭과 섹션 비활성화
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));

        // Activate selected / 선택된 항목 활성화
        const tabEl = document.querySelector(`.nav-tab[onclick="switchTab('${{tabId}}')"]`);
        const secEl = document.getElementById('sec-' + tabId);
        if (tabEl) tabEl.classList.add('active');
        if (secEl) secEl.classList.add('active');

        // Re-layout Plotly charts in newly visible section
        // 새로 보이는 섹션의 Plotly 차트를 다시 레이아웃합니다
        if (secEl) {{
            secEl.querySelectorAll('.plotly-wrap').forEach(el => {{
                if (el.id && el.layout) Plotly.relayout(el.id, {{}});
            }});
        }}
    }}
</script>

</body>
</html>"""

    # ── Write file (UTF-8) ────────────────────────────────────────────────
    # 파일 쓰기 (UTF-8)
    log.debug("[Report] Writing HTML to file...")
    output_path.write_text(html, encoding="utf-8")
    log.info(f"✓ Report saved to: {output_path.resolve()}")

    if open_browser:
        # Use file:// URI for reliable cross-platform browser opening
        # 크로스 플랫폼 브라우저 열기를 위해 file:// URI 사용
        uri = output_path.resolve().as_uri()
        log.debug(f"[Report] Opening browser: {uri}")
        webbrowser.open(uri)

    return output_path.resolve()
