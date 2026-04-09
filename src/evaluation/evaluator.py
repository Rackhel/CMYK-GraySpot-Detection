"""
evaluation/evaluator.py
=======================
Grayspot Detection Pipeline — Full Evaluation Pipeline Orchestrator
Grayspot 탐지 파이프라인 — 전체 평가 파이프라인 오케스트레이터

This module ties together metrics.py and confusion.py into a single
end-to-end evaluation pipeline that can be invoked from scripts or
the GUI Training Tab 3 (Evaluation).

이 모듈은 metrics.py와 confusion.py를 하나의 end-to-end 평가 파이프라인으로
묶어 scripts 또는 GUI Training Tab 3 (Evaluation)에서 호출할 수 있습니다.

Source notebook : 04_evaluation.ipynb (Cells 2~14)
PRD reference   : Section 5.6 (Evaluation Module), Section 8.2 (Reporting),
                  Section 3.3 (Phase 3 Feedback Loop)
Execution plan  : Stage 2 (W5~W6), Role R3

Python 3.11.5 | macOS (MPS) & Windows (CUDA/CPU) compatible
"""

# ── Standard library / 표준 라이브러리 ────────────────────────────────────
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Third-party / 서드파티 ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Internal modules / 내부 모듈 ──────────────────────────────────────────
from src.utils import get_logger, LoggerMixin
from .metrics import (
    NUM_LEVELS,
    CHANNELS,
    DEFAULT_TARGET_OVERALL_ACC,
    DEFAULT_TARGET_PER_CLASS_F1,
    DEFAULT_TARGET_PER_COLOR_ACC,
    DEFAULT_TARGET_MAE,
    ChannelMetrics,
    EvaluationSummary,
    compute_channel_metrics,
    determine_swing_feedback,
    print_summary,
    save_metrics_json,
)
from .confusion import (
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
    FONT_SIZE,
    CMYK_COLORS,
    LEVEL_COLORS,
    _open_in_browser,
    build_confusion_matrix_figure,
    save_all_confusion_matrices,
    save_mae_heatmap,
)

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 0. PRD §14.2 Confidence thresholds
#    PRD §14.2 신뢰도 임계값
# ─────────────────────────────────────────────────────────────────────────────

CONF_THRESH_AUTO: float = 0.8    # Auto judgment / 자동 판정
CONF_THRESH_WARN: float = 0.5    # Warn + auto / 경고 포함 자동
CONF_THRESH_MANUAL: float = 0.3  # Manual queue / 수동 검수 대기


# ─────────────────────────────────────────────────────────────────────────────
# 1. EvaluatorConfig — all tunable parameters in one place
#    EvaluatorConfig — 모든 조정 가능한 파라미터를 한 곳에
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluatorConfig:
    """
    Configuration for GrayspotEvaluator.
    GrayspotEvaluator 설정.

    All parameters have sensible defaults that match the PRD and notebook.
    모든 파라미터는 PRD 및 노트북과 일치하는 합리적인 기본값을 가집니다.

    Attributes:
        output_dir          : Directory for all output files / 모든 출력 파일 디렉토리
        channels            : CMYK channel identifiers / CMYK 채널 식별자
        num_levels          : Grayspot severity levels (0~5) / Grayspot 심각도 레벨 (0~5)
        backbone_name       : Model backbone identifier / 모델 백본 식별자
        checkpoint          : Path to checkpoint (for metadata) / 체크포인트 경로 (메타데이터용)
        target_overall_acc  : PRD §1.4 overall accuracy target / 전체 정확도 목표
        target_per_color_acc: PRD §1.4 per-color accuracy target / 색상별 정확도 목표
        target_per_class_f1 : PRD §1.4 per-class F1 target / 클래스별 F1 목표
        target_mae          : PRD §1.4 MAE target / MAE 목표
        normalize_cm        : Row-normalize confusion matrices / 혼동 행렬 행 정규화
        open_browser        : Auto-open HTML reports in browser / 브라우저에서 HTML 자동 열기
        conf_thresh_auto    : PRD §14.2 auto-judgment threshold / 자동 판정 임계값
        conf_thresh_warn    : PRD §14.2 warn+auto threshold / 경고 포함 자동 임계값
        conf_thresh_manual  : PRD §14.2 manual queue threshold / 수동 검수 대기 임계값
    """

    output_dir: str | Path = Path("outputs/evaluation")
    channels: list[str] = field(default_factory=lambda: list(CHANNELS))
    num_levels: int = NUM_LEVELS
    backbone_name: str = "efficientnet_b0"
    checkpoint: Optional[str | Path] = None

    # Performance targets / 성능 목표
    target_overall_acc: float = DEFAULT_TARGET_OVERALL_ACC
    target_per_color_acc: float = DEFAULT_TARGET_PER_COLOR_ACC
    target_per_class_f1: float = DEFAULT_TARGET_PER_CLASS_F1
    target_mae: float = DEFAULT_TARGET_MAE

    # Visualization / 시각화
    normalize_cm: bool = True
    open_browser: bool = False

    # Confidence thresholds / 신뢰도 임계값
    conf_thresh_auto: float = CONF_THRESH_AUTO
    conf_thresh_warn: float = CONF_THRESH_WARN
    conf_thresh_manual: float = CONF_THRESH_MANUAL

    def targets_dict(self) -> dict[str, float]:
        """Return targets as a plain dict for EvaluationSummary. / 타겟을 딕셔너리로 반환."""
        return {
            "overall_accuracy": self.target_overall_acc,
            "per_color_accuracy": self.target_per_color_acc,
            "per_class_f1": self.target_per_class_f1,
            "mae": self.target_mae,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. GrayspotEvaluator — main evaluation pipeline class
#    GrayspotEvaluator — 메인 평가 파이프라인 클래스
# ─────────────────────────────────────────────────────────────────────────────


class GrayspotEvaluator(LoggerMixin):
    """
    Full evaluation pipeline orchestrator.
    전체 평가 파이프라인 오케스트레이터.

    Orchestrates metrics computation, confusion matrix generation,
    Plotly chart creation, CSV/JSON export, and Phase 3 feedback logic.

    지표 계산, 혼동 행렬 생성, Plotly 차트 생성, CSV/JSON 내보내기,
    Phase 3 피드백 로직을 오케스트레이션합니다.

    Usage / 사용법:
        evaluator = GrayspotEvaluator(config)
        summary   = evaluator.run(results)  # results from inference pipeline

    The `results` dict must have the structure produced by run_inference():
    `results` 딕셔너리는 run_inference()가 생성하는 구조여야 합니다:
        {
            'Y': {
                'y_true'     : np.ndarray,  # shape (N,)
                'y_pred'     : np.ndarray,  # shape (N,)
                'confidences': np.ndarray,  # shape (N,), softmax max
                'filenames'  : list[str],
            },
            'M': { ... },
            'C': { ... },
            'K': { ... },
        }
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        """
        Initialize the evaluator.
        평가기를 초기화합니다.

        Args:
            config : EvaluatorConfig instance; defaults are used if None.
                     EvaluatorConfig 인스턴스; None이면 기본값 사용.
        """
        self.cfg = config or EvaluatorConfig()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Populated after run() / run() 이후 채워짐
        self.summary: Optional[EvaluationSummary] = None
        self.df_eval: Optional[pd.DataFrame] = None
        self.df_miss: Optional[pd.DataFrame] = None

    # ── Public entry point ────────────────────────────────────────────────

    def run(
        self,
        results: dict[str, dict],
        meta: Optional[dict] = None,
    ) -> EvaluationSummary:
        """
        Execute the full evaluation pipeline and save all outputs.
        전체 평가 파이프라인을 실행하고 모든 출력물을 저장합니다.

        Steps (mirrors 04_evaluation.ipynb Cells 6~14):
        단계 (04_evaluation.ipynb Cells 6~14 반영):
          1. Compute per-channel and overall metrics
             채널별 및 전체 지표 계산
          2. Build EvaluationSummary
             EvaluationSummary 생성
          3. Save confusion matrices (per-channel + overall)
             혼동 행렬 저장 (채널별 + 전체)
          4. Save MAE heatmap
             MAE 히트맵 저장
          5. Generate evaluation dashboard (Gauge + Bar)
             평가 대시보드 생성 (Gauge + Bar)
          6. Generate per-class metrics chart
             클래스별 지표 차트 생성
          7. Collect and save misclassified samples
             오분류 샘플 수집 및 저장
          8. Plot confidence distribution
             신뢰도 분포 차트 저장
          9. Export evaluation_results.csv and metrics_summary.json
             evaluation_results.csv 및 metrics_summary.json 내보내기
          10. Run Phase 3 feedback decision
              Phase 3 피드백 판단 실행

        Args:
            results : Dict of per-channel inference outputs / 채널별 추론 출력 딕셔너리
            meta    : Optional metadata dict (backbone, checkpoint, etc.)
                      선택적 메타데이터 딕셔너리

        Returns:
            Completed EvaluationSummary / 완성된 EvaluationSummary
        """
        cfg = self.cfg
        channels = cfg.channels

        # ── Step 1: Compute metrics / 지표 계산 ──────────────────────────
        logger.info("📊 Computing metrics / 지표 계산 중...")
        by_channel: dict[str, ChannelMetrics] = {}
        for ch in channels:
            yt = results[ch]["y_true"]
            yp = results[ch]["y_pred"]
            by_channel[ch] = compute_channel_metrics(
                y_true=yt,
                y_pred=yp,
                channel=ch,
                num_classes=cfg.num_levels,
                target_overall_acc=cfg.target_overall_acc,
                target_per_color_acc=cfg.target_per_color_acc,
                target_per_class_f1=cfg.target_per_class_f1,
                target_mae=cfg.target_mae,
            )

        # Combined arrays for 'overall' / 'overall'을 위한 통합 배열
        all_true = np.concatenate([results[c]["y_true"] for c in channels])
        all_pred = np.concatenate([results[c]["y_pred"] for c in channels])

        overall_cm = compute_channel_metrics(
            y_true=all_true,
            y_pred=all_pred,
            channel="overall",
            num_classes=cfg.num_levels,
            target_overall_acc=cfg.target_overall_acc,
            target_per_color_acc=cfg.target_per_color_acc,
            target_per_class_f1=cfg.target_per_class_f1,
            target_mae=cfg.target_mae,
        )

        # ── Step 2: Build EvaluationSummary / EvaluationSummary 생성 ──────
        _meta = meta or {}
        _meta.setdefault("backbone", cfg.backbone_name)
        _meta.setdefault("checkpoint", str(cfg.checkpoint) if cfg.checkpoint else None)
        _meta.setdefault(
            "n_samples",
            int(sum(len(results[c]["y_true"]) for c in channels))
        )

        self.summary = EvaluationSummary(
            overall=overall_cm,
            by_channel=by_channel,
            targets=cfg.targets_dict(),
            meta=_meta,
        )

        # ── Step 3: Print summary / 요약 출력 ────────────────────────────
        print_summary(self.summary, channels=channels, logger=self.logger)

        # ── Step 4: Confusion matrices / 혼동 행렬 ───────────────────────
        logger.info("\n📐 Saving confusion matrices / 혼동 행렬 저장 중...")
        save_all_confusion_matrices(
            results=results,
            metrics={**by_channel, "overall": overall_cm},
            output_dir=self.output_dir,
            channels=channels,
            num_classes=cfg.num_levels,
            normalize=cfg.normalize_cm,
            open_browser=cfg.open_browser,
            logger=self.logger,
        )

        # ── Step 5: MAE heatmap / MAE 히트맵 ─────────────────────────────
        logger.info("\n🌡️  Saving MAE heatmap / MAE 히트맵 저장 중...")
        save_mae_heatmap(
            results=results,
            output_dir=self.output_dir,
            channels=channels,
            num_classes=cfg.num_levels,
            target_mae=cfg.target_mae,
            open_browser=cfg.open_browser,
            logger=self.logger,
        )

        # ── Step 6: Evaluation dashboard / 평가 대시보드 ─────────────────
        logger.info("\n📈 Saving evaluation dashboard / 평가 대시보드 저장 중...")
        self._save_eval_dashboard(by_channel, overall_cm)

        # ── Step 7: Per-class metrics chart / 클래스별 지표 차트 ──────────
        logger.info("\n📊 Saving per-class metrics chart / 클래스별 지표 차트 저장 중...")
        self._save_per_class_chart(overall_cm)

        # ── Step 8: Misclassified samples / 오분류 샘플 ───────────────────
        logger.info("\n⚠️  Collecting misclassified samples / 오분류 샘플 수집 중...")
        self.df_miss = self._collect_misclassified(results, channels)
        self._save_misclassified(self.df_miss)

        # ── Step 9: Confidence distribution / 신뢰도 분포 ────────────────
        self.logger.info("\n🔵 Saving confidence distribution / 신뢰도 분포 저장 중...")
        self._save_confidence_distribution(results, channels)

        # ── Step 10: Export CSV + JSON / CSV + JSON 내보내기 ──────────────
        self.logger.info("\n💾 Exporting CSV & JSON / CSV & JSON 내보내기...")
        self.df_eval = self._build_eval_df(results, channels)
        self._export_csv(self.df_eval)
        save_metrics_json(self.summary, self.output_dir / "metrics_summary.json")
        self.logger.info(f"[저장 / Saved] {self.output_dir / 'metrics_summary.json'}")

        # ── Step 11: Phase 3 feedback / Phase 3 피드백 판단 ──────────────
        self.logger.info("\n🔄 Phase 3 Feedback Decision / Phase 3 피드백 복귀 판단...")
        self._print_feedback_decision(channels)

        self.logger.info(f"\n✅ 평가 완료 / Evaluation complete → {self.output_dir}")
        return self.summary

    # ── Private chart builders ────────────────────────────────────────────

    def _save_eval_dashboard(
        self,
        by_channel: dict[str, ChannelMetrics],
        overall_cm: ChannelMetrics,
    ) -> None:
        """
        Build and save the Gauge + Bar dashboard HTML.
        Gauge + Bar 대시보드 HTML을 생성하고 저장합니다.

        Mirrors plot_eval_dashboard() from 04_evaluation.ipynb Cell 8.
        04_evaluation.ipynb Cell 8의 plot_eval_dashboard()를 반영합니다.
        """
        cfg = self.cfg
        channels = cfg.channels

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

        # Gauge: Overall Accuracy
        # Gauge: 전체 정확도
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=round(overall_cm.accuracy * 100, 2),
                number={"suffix": "%", "font": {"size": 32}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#50e3c2"},
                    "threshold": {
                        "line": {"color": "#ff7aa2", "width": 3},
                        "value": cfg.target_overall_acc * 100,
                    },
                    "bgcolor": "#0b1220",
                },
                title={"text": f"Target / 목표 ≥ {cfg.target_overall_acc:.0%}"},
            ),
            row=1,
            col=1,
        )

        # Gauge: Macro F1
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=round(overall_cm.macro_f1, 4),
                number={"font": {"size": 32}},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "#66d9ff"},
                    "threshold": {
                        "line": {"color": "#ff7aa2", "width": 3},
                        "value": cfg.target_per_class_f1,
                    },
                    "bgcolor": "#0b1220",
                },
                title={"text": f"Target / 목표 ≥ {cfg.target_per_class_f1:.2f}"},
            ),
            row=1,
            col=2,
        )

        # Gauge: MAE (lower is better / 낮을수록 좋음)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=round(overall_cm.mae, 4),
                number={"font": {"size": 32}},
                gauge={
                    "axis": {"range": [0, 3]},
                    "bar": {"color": "#c792ea"},
                    "threshold": {
                        "line": {"color": "#ffb347", "width": 3},
                        "value": cfg.target_mae,
                    },
                    "bgcolor": "#0b1220",
                },
                title={"text": f"Target / 목표 ≤ {cfg.target_mae:.2f}"},
            ),
            row=1,
            col=3,
        )

        # Bar: Per-color Accuracy / 색상별 정확도
        bar_colors = [CMYK_COLORS.get(c, "#aaaaaa") for c in channels]
        acc_vals = [by_channel[c].accuracy * 100 for c in channels]

        fig.add_trace(
            go.Bar(
                x=channels,
                y=acc_vals,
                marker_color=bar_colors,
                text=[f"{v:.2f}%" for v in acc_vals],
                textposition="outside",
                name="Accuracy",
            ),
            row=2,
            col=1,
        )

        # Target dashed line / 목표 점선
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(channels) - 0.5,
            y0=cfg.target_per_color_acc * 100,
            y1=cfg.target_per_color_acc * 100,
            line=dict(color="#ff7aa2", dash="dash"),
            xref="x",
            yref="y",
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=channels[-1],
            y=cfg.target_per_color_acc * 100,
            text=f"Target {cfg.target_per_color_acc:.0%}",
            showarrow=False,
            yshift=10,
            row=2,
            col=1,
        )

        fig.update_layout(
            title=dict(
                text="Grayspot Evaluation Dashboard / 평가 대시보드",
                font=dict(size=18),
            ),
            template=PLOTLY_TEMPLATE,
            font=dict(family=FONT_FAMILY, size=FONT_SIZE),
            height=750,
            showlegend=False,
            margin=dict(l=40, r=40, t=80, b=40),
        )

        html_path = self.output_dir / "eval_dashboard.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        self.logger.info(f"[저장 / Saved] {html_path}")

        if self.cfg.open_browser:
            _open_in_browser(html_path)

    def _save_per_class_chart(self, overall_cm: ChannelMetrics) -> None:
        """
        Build and save the per-class Precision/Recall/F1 bar chart.
        클래스별 Precision/Recall/F1 막대 차트를 생성하고 저장합니다.

        Mirrors plot_per_class_metrics() from 04_evaluation.ipynb Cell 9.
        04_evaluation.ipynb Cell 9의 plot_per_class_metrics()를 반영합니다.
        """
        pc = overall_cm.per_class
        levels = [f"Level {d.level}" for d in pc]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="F1",
                x=levels,
                y=[d.f1 for d in pc],
                marker_color="#1976D2",
            )
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
                name="Recall",
                x=levels,
                y=[d.recall for d in pc],
                marker_color="#F57C00",
            )
        )

        fig.add_hline(
            y=self.cfg.target_per_class_f1,
            line_dash="dash",
            line_color="#ff7aa2",
            annotation_text=(
                f"F1 Target / 목표 ≥ {self.cfg.target_per_class_f1:.2f}"
            ),
        )

        fig.update_layout(
            title=dict(
                text="Per-Class Metrics (Overall) / 클래스별 지표 (전체)",
                font=dict(size=16),
            ),
            barmode="group",
            yaxis=dict(range=[0, 1.15], title="Score"),
            xaxis=dict(title="Level"),
            template=PLOTLY_TEMPLATE,
            font=dict(family=FONT_FAMILY, size=FONT_SIZE),
            legend=dict(orientation="h", y=1.1),
            height=480,
            margin=dict(l=40, r=40, t=80, b=40),
        )

        html_path = self.output_dir / "per_class_metrics.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        self.logger.info(f"[저장 / Saved] {html_path}")

        if self.cfg.open_browser:
            _open_in_browser(html_path)

    def _collect_misclassified(
        self,
        results: dict[str, dict],
        channels: list[str],
    ) -> pd.DataFrame:
        """
        Collect all misclassified samples into a DataFrame.
        모든 오분류 샘플을 DataFrame으로 수집합니다.

        Mirrors the misclassified_records loop in 04_evaluation.ipynb Cell 11.
        04_evaluation.ipynb Cell 11의 misclassified_records 루프를 반영합니다.

        Returns:
            DataFrame sorted by error_gap DESC, confidence ASC
            error_gap 내림차순, confidence 오름차순으로 정렬된 DataFrame
        """
        records: list[dict] = []
        for color in channels:
            yt = results[color]["y_true"]
            yp = results[color]["y_pred"]
            confs = results[color]["confidences"]
            fnames = results[color]["filenames"]

            for i in np.where(yt != yp)[0]:
                records.append(
                    {
                        "filename": fnames[i],
                        "color": color,
                        "true_level": int(yt[i]),
                        "pred_level": int(yp[i]),
                        "confidence": round(float(confs[i]), 4),
                        "correct": False,
                        "error_gap": int(abs(yt[i] - yp[i])),
                    }
                )

        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.sort_values(
                ["error_gap", "confidence"], ascending=[False, True]
            )
            self.logger.info(f"⚠️   오분류 샘플 수 / Misclassified: {len(df)}")
            self.logger.info(f"    최대 오류 간격 / Max error gap: {df['error_gap'].max()}")
            self.logger.info("\n[색상별 오분류 수 / Per-color mismatch]")
            self.logger.info(df.groupby("color").size().to_string())
        else:
            self.logger.info("✅  오분류 없음 / No misclassifications")
        return df

    def _save_misclassified(self, df_miss: pd.DataFrame) -> None:
        """
        Save misclassified samples CSV and scatter chart.
        오분류 샘플 CSV 및 scatter 차트를 저장합니다.

        CSV uses UTF-8-BOM to prevent Korean garbling in Windows Excel.
        CSV는 Windows Excel 한글 깨짐 방지를 위해 UTF-8-BOM을 사용합니다.
        """
        # Save CSV / CSV 저장
        csv_path = self.output_dir / "misclassified_samples.csv"
        df_miss.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
        self.logger.info(f"\n[저장 / Saved] {csv_path}")

        # Save scatter chart / scatter 차트 저장
        if len(df_miss) == 0:
            self.logger.info("ℹ️   오분류 없음 — scatter 건너뜀 / No misclassified — skipping scatter")
            return

        fig = px.scatter(
            df_miss,
            x="true_level",
            y="pred_level",
            color="color",
            color_discrete_map=CMYK_COLORS,
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
            title="Misclassified Samples — True vs Predicted Level / 오분류 샘플",
            labels={
                "true_level": "True Level / 실제 레벨",
                "pred_level": "Predicted Level / 예측 레벨",
                "color": "CMYK Channel",
            },
            template=PLOTLY_TEMPLATE,
            width=700,
            height=550,
        )

        # Diagonal reference line (correct boundary)
        # 대각선 기준선 (정답 경계)
        fig.add_trace(
            go.Scatter(
                x=[0, self.cfg.num_levels - 1],
                y=[0, self.cfg.num_levels - 1],
                mode="lines",
                line=dict(color="gray", dash="dash", width=1),
                name="Correct boundary / 정답 경계",
                showlegend=True,
            )
        )
        fig.update_layout(font=dict(family=FONT_FAMILY, size=FONT_SIZE))

        html_path = self.output_dir / "misclassified_scatter.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        self.logger.info(f"[저장 / Saved] {html_path}")

        if self.cfg.open_browser:
            _open_in_browser(html_path)

    def _save_confidence_distribution(
        self,
        results: dict[str, dict],
        channels: list[str],
    ) -> None:
        """
        Build and save the per-color confidence histogram.
        색상별 신뢰도 히스토그램을 생성하고 저장합니다.

        Shows PRD §14.2 confidence threshold lines.
        PRD §14.2 신뢰도 임계값 수직선을 표시합니다.

        Mirrors plot_confidence_distribution() from 04_evaluation.ipynb Cell 12.
        04_evaluation.ipynb Cell 12의 plot_confidence_distribution()를 반영합니다.
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
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

            # Correct predictions histogram / 정답 히스토그램
            fig.add_trace(
                go.Histogram(
                    x=cf[yt == yp],
                    xbins=bins,
                    name="정답 / Correct",
                    marker_color="#4fc3f7",
                    opacity=0.70,
                    showlegend=(i == 0),
                    legendgroup="correct",
                ),
                row=r,
                col=c,
            )

            # Wrong predictions histogram / 오답 히스토그램
            fig.add_trace(
                go.Histogram(
                    x=cf[yt != yp],
                    xbins=bins,
                    name="오답 / Wrong",
                    marker_color="#ef5350",
                    opacity=0.70,
                    showlegend=(i == 0),
                    legendgroup="wrong",
                ),
                row=r,
                col=c,
            )

            # PRD §14.2 threshold vertical lines
            # PRD §14.2 임계값 수직선
            for thresh, line_color in [
                (self.cfg.conf_thresh_auto, "green"),
                (self.cfg.conf_thresh_warn, "orange"),
                (self.cfg.conf_thresh_manual, "red"),
            ]:
                fig.add_vline(
                    x=thresh,
                    line_dash="dash",
                    line_color=line_color,
                    line_width=1.5,
                    row=r,
                    col=c,
                )

        fig.update_layout(
            title=dict(
                text="Confidence Distribution / 신뢰도 분포 — Correct vs Wrong",
                font=dict(size=16),
            ),
            barmode="overlay",
            template=PLOTLY_TEMPLATE,
            font=dict(family=FONT_FAMILY, size=FONT_SIZE),
            height=650,
            margin=dict(l=40, r=40, t=80, b=40),
        )

        html_path = self.output_dir / "confidence_distribution.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        self.logger.info(f"[저장 / Saved] {html_path}")

        if self.cfg.open_browser:
            _open_in_browser(html_path)

        # Coverage table / 커버리지 테이블
        all_confs = np.concatenate([results[c]["confidences"] for c in channels])
        total = len(all_confs)
        self.logger.info("\n=== Confidence-Driven Coverage (PRD §14.2) / 신뢰도 기반 커버리지 ===")
        self.logger.info(
            f"{'Policy':30s} {'Threshold':>12s} {'Samples':>10s} {'Coverage':>10s}"
        )
        self.logger.info("-" * 68)
        for name, thresh in [
            ("Auto judgment / 자동 판정", self.cfg.conf_thresh_auto),
            ("Warn + auto  / 경고 포함 자동", self.cfg.conf_thresh_warn),
            ("Manual queue / 수동 검수 대기", self.cfg.conf_thresh_manual),
        ]:
            n = int((all_confs >= thresh).sum())
            pct = n / total * 100 if total > 0 else 0.0
            self.logger.info(f"{name:30s} {'≥ ' + str(thresh):>12s} {n:>10d} {pct:>9.1f}%")
        self.logger.info(f"{'Total / 전체':30s} {'—':>12s} {total:>10d} {'100.0':>9s}%")

    def _build_eval_df(
        self,
        results: dict[str, dict],
        channels: list[str],
    ) -> pd.DataFrame:
        """
        Build the per-sample evaluation_results DataFrame.
        샘플별 evaluation_results DataFrame을 생성합니다.

        Matches PRD §8.2.2 evaluation_results.csv format.
        PRD §8.2.2 evaluation_results.csv 형식과 일치합니다.
        """
        rows: list[dict] = []
        for color in channels:
            yt = results[color]["y_true"]
            yp = results[color]["y_pred"]
            confs = results[color]["confidences"]
            fnames = results[color]["filenames"]
            for i in range(len(yt)):
                rows.append(
                    {
                        "filename": fnames[i],
                        "color": color,
                        "true_level": int(yt[i]),
                        "pred_level": int(yp[i]),
                        "confidence": round(float(confs[i]), 4),
                        "correct": bool(yt[i] == yp[i]),
                        "error_gap": int(abs(yt[i] - yp[i])),
                    }
                )
        return pd.DataFrame(rows)

    def _export_csv(self, df_eval: pd.DataFrame) -> None:
        """
        Save evaluation_results.csv with UTF-8-BOM encoding.
        UTF-8-BOM 인코딩으로 evaluation_results.csv를 저장합니다.

        UTF-8-BOM prevents Korean character garbling in Windows Excel.
        UTF-8-BOM은 Windows Excel에서 한글 깨짐을 방지합니다.
        """
        csv_path = self.output_dir / "evaluation_results.csv"
        df_eval.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
        self.logger.info(f"[저장 / Saved] {csv_path}  ({len(df_eval)} rows)")

    def _print_feedback_decision(self, channels: list[str]) -> None:
        """
        Print Phase 3 feedback loop decision (PRD §3.3.2).
        Phase 3 피드백 루프 판단을 출력합니다 (PRD §3.3.2).

        Mirrors Cell 14 of 04_evaluation.ipynb.
        04_evaluation.ipynb Cell 14를 반영합니다.
        """
        if self.summary is None:
            return

        decision = determine_swing_feedback(
            summary=self.summary,
            channels=channels,
        )

        sep = "=" * 65
        self.logger.info(sep)
        self.logger.info("  Phase 3 Feedback Decision / Phase 3 피드백 복귀 판단")
        self.logger.info(sep)

        if decision["terminate"]:
            self.logger.info("\n  🎉 모든 목표 달성 → Swing 종료 / All targets met — TERMINATE Swing")
            self.logger.info(f"     Overall Accuracy : {decision['overall_accuracy']:.4f}")
            self.logger.info(f"     Macro F1         : {decision['overall_macro_f1']:.4f}")
            self.logger.info(f"     Overall MAE      : {decision['overall_mae']:.4f}")
        elif decision["decisions"]:
            self.logger.info("\n  ⚠️  조치 필요 / Action required:")
            for d in decision["decisions"]:
                self.logger.info(f"  ⮕ {d}")
        else:
            self.logger.info("\n  ✅  심각한 실패 없음 / No critical failures — continue or next cycle")

        self.logger.info("")
        self.logger.info(
            f"  Overall Accuracy : {decision['overall_accuracy']:.4f}"
            f"  (target ≥ {self.summary.targets['overall_accuracy']})"
        )
        self.logger.info(
            f"  Overall Macro F1 : {decision['overall_macro_f1']:.4f}"
            f"  (target ≥ {self.summary.targets['per_class_f1']})"
        )
        self.logger.info(
            f"  Overall MAE      : {decision['overall_mae']:.4f}"
            f"  (target ≤ {self.summary.targets['mae']})"
        )
        self.logger.info(sep)

    # ── Convenience helpers for GUI / scripts ─────────────────────────────

    def print_output_checklist(self) -> None:
        """
        Print the list of generated output files with existence check.
        생성된 출력 파일 목록을 존재 여부와 함께 출력합니다.

        Mirrors Cell 15 of 04_evaluation.ipynb.
        04_evaluation.ipynb Cell 15를 반영합니다.
        """
        output_files = [
            # Plotly HTML charts / Plotly HTML 차트
            ("cm_Y.html",                   "Confusion Matrix [Y] — interactive / 혼동 행렬"),
            ("cm_M.html",                   "Confusion Matrix [M] — interactive / 혼동 행렬"),
            ("cm_C.html",                   "Confusion Matrix [C] — interactive / 혼동 행렬"),
            ("cm_K.html",                   "Confusion Matrix [K] — interactive / 혼동 행렬"),
            ("cm_overall.html",             "Confusion Matrix [Overall] / 전체"),
            ("eval_dashboard.html",         "Accuracy / F1 / MAE Gauge dashboard / 평가 대시보드"),
            ("per_class_metrics.html",      "Per-class Precision / Recall / F1 / 클래스별 지표"),
            ("mae_heatmap.html",            "MAE heatmap (color × level) / MAE 히트맵"),
            ("misclassified_scatter.html",  "Misclassified samples scatter / 오분류 분포"),
            ("confidence_distribution.html","Confidence distribution / 신뢰도 분포"),
            # Data files / 데이터 파일
            ("evaluation_results.csv",      "Per-sample predictions / 샘플별 예측 결과"),
            ("misclassified_samples.csv",   "Misclassified sample list / 오분류 샘플 목록"),
            ("metrics_summary.json",        "Aggregated metrics JSON / 성능 요약 JSON"),
        ]

        self.logger.info("=" * 65)
        self.logger.info("Generated Outputs / 생성된 산출물 목록")
        self.logger.info("=" * 65)
        for fname, desc in output_files:
            full_path = self.output_dir / fname
            exists = "[OK]" if full_path.exists() else "[  ]"
            self.logger.info(f"  {exists}  {fname:<45} {desc}")

    def get_summary_dict(self) -> Optional[dict]:
        """
        Return the EvaluationSummary as a JSON-serializable dict.
        EvaluationSummary를 JSON 직렬화 가능한 딕셔너리로 반환합니다.
        """
        if self.summary is None:
            return None
        from .metrics import summary_to_dict
        return summary_to_dict(self.summary)
