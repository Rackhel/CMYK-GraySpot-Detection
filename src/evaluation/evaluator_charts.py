"""
evaluation/evaluator_charts.py

책임 / Responsibility: 시각화 차트 7종 생성 + Phase 3 판단 텍스트
Responsibility: Build 7 visualisation charts + Phase 3 decision text

SRP 준수: 이 모듈은 "Plotly 차트 생성과 Phase 3 피드백 텍스트"만 담당한다.
SRP compliant: this module handles only "Plotly chart building and Phase 3 feedback text".
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evaluation.metrics import (
    TARGET_OVERALL_ACC,
    TARGET_PER_CLASS_F1,
    TARGET_PER_COLOR_ACC,
    TARGET_MAE,
    check_targets,
)


class ChartsMixin:
    """
    시각화 차트 생성을 담당하는 Mixin.
    Mixin responsible for building visualisation charts.

    사용하는 self 속성 / Consumed self attributes (provided by Evaluator.__init__):
        self.num_levels, self.conf_thresh_auto, self.conf_thresh_warn,
        self.conf_thresh_manual
    """

    CMYK_COLORS: Dict[str, str] = {
        "Y": "#f5e642",
        "M": "#e91e8c",
        "C": "#00b4d8",
        "K": "#444444",
    }

    def _build_dashboard(
        self, metrics: Dict[str, dict], channels: List[str]
    ) -> go.Figure:
        """Gauge + Bar 평가 대시보드 / Gauge + Bar evaluation dashboard."""
        m_all = metrics["overall"]
        valid_chs = [c for c in channels if c in metrics]

        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar", "colspan": 3}, None, None],
            ],
            subplot_titles=[
                "Overall Accuracy",
                "Overall Macro F1",
                "Overall MAE (lower is better / 낮을수록 좋음)",
                "Per-Color Accuracy / 색상별 정확도",
            ],
            vertical_spacing=0.18,
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=round(m_all["accuracy"] * 100, 2),
                number={"suffix": "%", "font": {"size": 30}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#50e3c2"},
                    "threshold": {
                        "line": {"color": "#ff7aa2", "width": 3},
                        "value": TARGET_OVERALL_ACC * 100,
                    },
                    "bgcolor": "#0b1220",
                },
                title={"text": f"Target >= {TARGET_OVERALL_ACC:.0%}"},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=round(m_all["macro_f1"], 4),
                number={"font": {"size": 30}},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "#66d9ff"},
                    "threshold": {
                        "line": {"color": "#ff7aa2", "width": 3},
                        "value": TARGET_PER_CLASS_F1,
                    },
                    "bgcolor": "#0b1220",
                },
                title={"text": f"Target >= {TARGET_PER_CLASS_F1:.2f}"},
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=round(m_all["mae"], 4),
                number={"font": {"size": 30}},
                gauge={
                    "axis": {"range": [0, 3]},
                    "bar": {"color": "#c792ea"},
                    "threshold": {
                        "line": {"color": "#ffb347", "width": 3},
                        "value": TARGET_MAE,
                    },
                    "bgcolor": "#0b1220",
                },
                title={"text": f"Target <= {TARGET_MAE:.2f}"},
            ),
            row=1,
            col=3,
        )

        acc_vals = [metrics[c]["accuracy"] * 100 for c in valid_chs]
        fig.add_trace(
            go.Bar(
                x=valid_chs,
                y=acc_vals,
                marker_color=[self.CMYK_COLORS[c] for c in valid_chs],
                text=[f"{v:.2f}%" for v in acc_vals],
                textposition="outside",
                name="Accuracy",
            ),
            row=2,
            col=1,
        )

        if valid_chs:
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(valid_chs) - 0.5,
                y0=TARGET_PER_COLOR_ACC * 100,
                y1=TARGET_PER_COLOR_ACC * 100,
                line=dict(color="#ff7aa2", dash="dash"),
                xref="x",
                yref="y",
                row=2,
                col=1,
            )
            fig.add_annotation(
                x=valid_chs[-1],
                y=TARGET_PER_COLOR_ACC * 100,
                text=f"Target {TARGET_PER_COLOR_ACC:.0%}",
                showarrow=False,
                yshift=10,
                row=2,
                col=1,
            )

        fig.update_layout(
            title=dict(
                text="Grayspot Evaluation Dashboard / 평가 대시보드", font=dict(size=17)
            ),
            template="plotly_dark",
            font=dict(family="Segoe UI", size=12),
            height=750,
            showlegend=False,
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig

    def _build_per_class_chart(self, metrics: Dict[str, dict]) -> go.Figure:
        """클래스별 F1 막대 차트 / Per-class F1 bar chart."""
        pc = metrics["overall"]["per_class"]
        levels = [f"Level {d['level']}" for d in pc]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(name="F1", x=levels, y=[d["f1"] for d in pc], marker_color="#1976D2")
        )
        fig.add_trace(
            go.Bar(
                name="Precision",
                x=levels,
                y=[d["precision"] for d in pc],
                marker_color="#388E3C",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Recall",
                x=levels,
                y=[d["recall"] for d in pc],
                marker_color="#F57C00",
            )
        )
        fig.add_hline(
            y=TARGET_PER_CLASS_F1,
            line_dash="dash",
            line_color="#ff7aa2",
            annotation_text=f"F1 Target >= {TARGET_PER_CLASS_F1:.2f}",
        )
        fig.update_layout(
            title=dict(
                text="Per-Class Metrics (Overall) / 클래스별 지표", font=dict(size=15)
            ),
            barmode="group",
            yaxis=dict(range=[0, 1.15], title="Score"),
            xaxis=dict(title="Level"),
            template="plotly_dark",
            font=dict(family="Segoe UI", size=12),
            legend=dict(orientation="h", y=1.1),
            height=480,
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig

    def _build_mae_heatmap(
        self,
        results: Dict[str, dict],
        channels: List[str],
    ) -> go.Figure:
        """(color x level) MAE 히트맵 / MAE heatmap by (color x level)."""
        valid_chs = [c for c in channels if c in results]
        level_names = [f"Level {i}" for i in range(self.num_levels)]
        mae_matrix = np.full((len(valid_chs), self.num_levels), np.nan)
        count_matrix = np.zeros((len(valid_chs), self.num_levels), dtype=int)

        for ci, color in enumerate(valid_chs):
            yt = results[color]["y_true"]
            yp = results[color]["y_pred"]
            for lv in range(self.num_levels):
                mask = yt == lv
                if mask.sum() > 0:
                    mae_matrix[ci, lv] = float(np.mean(np.abs(yt[mask] - yp[mask])))
                    count_matrix[ci, lv] = int(mask.sum())

        annot = [
            [
                (
                    f"{mae_matrix[r, c]:.2f}\n(n={count_matrix[r, c]})"
                    if not np.isnan(mae_matrix[r, c])
                    else "N/A"
                )
                for c in range(self.num_levels)
            ]
            for r in range(len(valid_chs))
        ]

        fig = go.Figure(
            go.Heatmap(
                z=mae_matrix,
                x=level_names,
                y=valid_chs,
                text=annot,
                texttemplate="%{text}",
                colorscale="YlOrRd",
                zmin=0,
                zmax=2.0,
                colorbar=dict(title="MAE"),
                hovertemplate="Color: %{y}<br>Level: %{x}<br>MAE: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(
                text=f"MAE per (Color x True Level) — Target <= {TARGET_MAE}",
                font=dict(size=15),
            ),
            xaxis=dict(title="True Level / 실제 레벨"),
            yaxis=dict(title="Color Channel / 색상 채널"),
            template="plotly_dark",
            font=dict(family="Segoe UI", size=12),
            height=360,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    def _build_mismatch_scatter(self, df_miss: pd.DataFrame) -> go.Figure:
        """오분류 scatter / Misclassified samples scatter."""
        if len(df_miss) == 0:
            return go.Figure()

        fig = px.scatter(
            df_miss,
            x="true_level",
            y="pred_level",
            color="color",
            color_discrete_map=self.CMYK_COLORS,
            size="error_gap",
            size_max=20,
            hover_data=[
                "filename",
                "color",
                "true_level",
                "pred_level",
                "confidence",
                "error_gap",
            ],
            title="Misclassified Samples / 오분류 샘플 — True vs Predicted Level",
            labels={
                "true_level": "True Level / 실제 레벨",
                "pred_level": "Predicted Level / 예측 레벨",
                "color": "CMYK Channel",
            },
            template="plotly_dark",
            width=680,
            height=540,
        )
        fig.add_trace(
            go.Scatter(
                x=[0, self.num_levels - 1],
                y=[0, self.num_levels - 1],
                mode="lines",
                line=dict(color="gray", dash="dash", width=1),
                name="Correct boundary / 정답 경계",
                showlegend=True,
            )
        )
        fig.update_layout(
            font=dict(family="Segoe UI", size=12),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    def _build_confidence_dist(
        self,
        results: Dict[str, dict],
        channels: List[str],
    ) -> go.Figure:
        """신뢰도 분포 히스토그램 / Confidence distribution histogram."""
        valid_chs = [c for c in channels if c in results]
        n_rows = (len(valid_chs) + 1) // 2

        fig = make_subplots(
            rows=n_rows,
            cols=2,
            subplot_titles=[f"[{c}]" for c in valid_chs],
            horizontal_spacing=0.10,
            vertical_spacing=0.18,
        )
        bins = dict(start=0, end=1, size=0.04)

        for i, color in enumerate(valid_chs):
            r = i // 2 + 1
            c = i % 2 + 1
            yt = results[color]["y_true"]
            yp = results[color]["y_pred"]
            cf = results[color]["confidences"]

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

            for thresh, col_color in [
                (self.conf_thresh_auto, "green"),
                (self.conf_thresh_warn, "orange"),
                (self.conf_thresh_manual, "red"),
            ]:
                fig.add_vline(
                    x=thresh,
                    line_dash="dash",
                    line_color=col_color,
                    line_width=1.5,
                    row=r,
                    col=c,
                )

        fig.update_layout(
            title=dict(
                text="Confidence Distribution / 신뢰도 분포 — Correct vs Wrong",
                font=dict(size=15),
            ),
            barmode="overlay",
            template="plotly_dark",
            font=dict(family="Segoe UI", size=12),
            height=640,
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig

    def _build_phase3_decision(
        self,
        metrics: Dict[str, dict],
        channels: List[str],
    ) -> str:
        """
        PRD 3.3.2 피드백 복귀 판단 텍스트를 생성한다.
        Generates PRD 3.3.2 feedback-loop decision text.
        """
        targets = check_targets(metrics, channels)
        decisions = []
        overall_mae = metrics["overall"]["mae"]
        overall_acc = metrics["overall"]["accuracy"]
        overall_mf1 = metrics["overall"]["macro_f1"]

        for color in channels:
            if color not in metrics:
                continue
            acc = metrics[color]["accuracy"]
            if acc < 0.80:
                decisions.append(
                    f"[{color}] Accuracy {acc:.3f} < 0.80"
                    " -> Phase 0 (retrain representation / 표현 재학습)"
                )

        for pc in metrics["overall"]["per_class"]:
            if pc["f1"] < 0.70:
                decisions.append(
                    f"Level {pc['level']} F1={pc['f1']:.3f} < 0.70"
                    " -> Phase 1 (review level boundary / 레벨 경계 재검토)"
                )

        if overall_mae > 0.80:
            decisions.append(
                f"Overall MAE {overall_mae:.3f} > 0.80"
                " -> Phase 0 (representation learning retry / 표현 학습 재시도)"
            )

        lines = ["=== Phase 3 Feedback Decision / Phase 3 피드백 복귀 판단 ==="]

        all_color_ok = all(targets.get(c, {}).get("acc_pass", False) for c in channels)
        if targets["overall"]["all_pass"] and all_color_ok:
            lines.append(
                "All targets met -- TERMINATE Swing / 모든 목표 달성 -- Swing 종료"
            )
        elif not decisions:
            lines.append(
                "No critical failures -- continue / 심각한 실패 없음 -- 계속 진행"
            )
        else:
            lines.append("Action required / 조치 필요:")
            lines.extend(f"  {d}" for d in decisions)

        lines += [
            "",
            f"  Overall Accuracy : {overall_acc:.4f}  (target >= {TARGET_OVERALL_ACC})",
            f"  Overall Macro F1 : {overall_mf1:.4f}  (target >= {TARGET_PER_CLASS_F1})",
            f"  Overall MAE      : {overall_mae:.4f}  (target <= {TARGET_MAE})",
        ]
        return "\n".join(lines)
